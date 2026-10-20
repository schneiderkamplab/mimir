"""Own a checkpointable single-node launcher and bound restore-to-disk recovery.

Evaluators still explicitly offload through controller.py. This owner restores
on restore.request or lease expiry; publication/merge success is irrelevant.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import psutil

from .controller import CudaBackend
from .protocol import atomic_write, exclusive, matches, read_json, process_identity, validate_process, steps


def overrides(command, values):
    keys = set(values)
    return [arg for arg in command if arg.split('=', 1)[0].lstrip('+') not in keys] + [
        f'{key}={value}' for key, value in values.items()]


def verified_checkpoint(ready):
    """Only restart a fully written regular checkpoint named by this handshake."""
    path = Path(ready['checkpoint']).resolve()
    tag = f"step_{int(ready['step'])}"
    if path.name != f'checkpoint_state_{tag}.json':
        raise RuntimeError('Suspension did not name its regular checkpoint')
    state = read_json(path)
    if state['step'] != ready['step'] or state.get('carry_policy') != 'none':
        raise RuntimeError('Checkpoint step mismatch or unsupported carry policy')
    # Use DCP storage offsets, not only the presence of its sidecar.
    from mimir.checkpoints.contract import validate
    if not validate(path.parent):
        raise RuntimeError(f'Incomplete sharded recovery checkpoint: {path}')
    return path.parent, tag


def restart_command(command, root, ready):
    directory, tag = verified_checkpoint(ready)
    queue = read_json(root / 'queue.json')
    if queue['job_id'] != ready['job_id']:
        raise RuntimeError('Queue identity mismatch')
    pending = list(queue['pending'])
    request = root / 'suspend_after_steps'
    if request.exists():
        pending += steps(request.read_text())
    remaining = [n for n in steps(pending) if n > ready['step']]
    return overrides(command, {
        'resume_checkpoint_path': str(directory), 'resume_checkpoint_tag': tag,
        'resume_step': 'null', 'resume_epoch': 'null', 'resume_batch_in_epoch': 'null',
        'reset_ema_on_resume': 'false', 'suspend_after_steps': str(remaining).replace(' ', ''),
    })


def remember_owned(proc):
    """torchrun gives workers separate sessions; record their actual ancestry."""
    owned = getattr(proc, 'owned_processes', {})
    try:
        validate_process(proc.launch_identity)
        for process in [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]:
            try:
                identity = process_identity(process.pid)
                owned[process.pid] = identity
            except (OSError, RuntimeError):
                pass
    except (OSError, RuntimeError, psutil.NoSuchProcess):
        pass
    proc.owned_processes = owned
    return owned


def stop_owned(proc, ready=None):
    """Signal only identity-checked descendants recorded for this launcher."""
    owned = remember_owned(proc)
    identities = list(owned.values())
    for rank in (ready or {}).get('ranks', []):
        try:
            validate_process(rank)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except RuntimeError:
            # PID reuse must never result in a signal to the replacement.
            continue
        if rank['pid'] not in owned or any(rank[k] != v for k, v in owned[rank['pid']].items()):
            raise RuntimeError('Refusing to signal a rank outside the recorded process tree')
    for sig, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 20)):
        for identity in identities:
            try:
                process = psutil.Process(identity['pid'])
                validate_process(identity)
                # psutil.send_signal rechecks PID reuse against its creation time.
                process.send_signal(sig)
            except (FileNotFoundError, ProcessLookupError, RuntimeError, psutil.NoSuchProcess):
                continue
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            proc.poll()  # Reap the direct child.
            alive = []
            for identity in identities:
                try:
                    validate_process(identity)
                    alive.append(identity)
                except (FileNotFoundError, ProcessLookupError, RuntimeError):
                    pass
            if not alive:
                return
            time.sleep(.2)
    raise RuntimeError('Owned workers did not exit; refusing a second training launch')


def gpu_uuids():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                     '--format=csv,noheader'], text=True)
    mapping = dict(line.strip().split(', ') for line in output.splitlines())
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(mapping)).split(',')
    return {mapping.get(value, value) for value in visible}


def wait_for_gpus(backend, selected, minimum, root, interval, peers=()):
    while True:
        if (root / 'stop.request').exists():
            raise RuntimeError('Supervisor stop requested while waiting for GPUs')
        free = backend.free()
        allowed = set()
        for peer in peers:
            try:
                validate_process(peer)
                allowed.add(peer['pid'])
            except (OSError, RuntimeError):
                pass
        if (not (root / 'hold').exists()
                and not any(gpu in selected and pid not in allowed for pid, gpu in backend.memory())
                and all(free.get(gpu, 0) >= minimum for gpu in selected)):
            return
        time.sleep(interval)


def restore_or_fail(root, executable, timeout, acknowledgement_timeout):
    result = subprocess.run([sys.executable, '-m', 'mimir.training.suspension.controller',
                             '--cuda-checkpoint', executable, 'restore', str(root)],
                            timeout=timeout)
    if result.returncode == 75:  # GPU occupancy or user hold, never a restart trigger.
        return False
    result.check_returncode()
    ready = read_json(root / 'suspend_ready.json')
    deadline = time.monotonic() + acknowledgement_timeout
    while time.monotonic() < deadline:
        path = root / 'resumed.json'
        if path.exists() and matches(read_json(path), ready):
            return True
        time.sleep(1)
    raise TimeoutError('Workers failed to acknowledge resume after CUDA restoration')


def supervise(args, command):
    root = args.control_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    backend = CudaBackend(args.cuda_checkpoint)
    selected = gpu_uuids()
    peers = [process_identity(pid) for pid in {pid for pid, gpu in backend.memory() if gpu in selected}] if args.allow_existing_peers else []
    with exclusive(root / 'supervisor.lock'):
        attempt = 0
        restarts = {}
        while True:
            wait_for_gpus(backend, selected, args.min_free_mib, root, args.interval, peers)
            attempt_root = root / f'attempt-{uuid.uuid4().hex}'
            attempt_root.mkdir()
            actual = overrides(command, {'suspend_control_dir': str(attempt_root)})
            proc = subprocess.Popen([
                sys.executable, '-m', 'mimir.training.suspension.controller',
                '--cuda-checkpoint', args.cuda_checkpoint, 'launch',
                '--control-root', str(attempt_root), '--', *actual], start_new_session=True)
            proc.launch_identity = process_identity(proc.pid)
            atomic_write(root / 'supervisor.json', {
                'owner': process_identity(), 'launcher': process_identity(proc.pid),
                'attempt': attempt, 'command': actual, 'attempt_root': str(attempt_root)})
            active = None
            ready = None
            try:
                while True:
                    remember_owned(proc)
                    if (root / 'stop.request').exists():
                        raise RuntimeError('Explicit supervisor stop; no automatic restart')
                    launches = list(attempt_root.glob('*/launch.json'))
                    if len(launches) > 1:
                        raise RuntimeError('Ambiguous worker launch')
                    if launches:
                        active = launches[0].parent
                        managed = active / 'managed.json'
                        if not managed.exists():
                            atomic_write(managed, {'supervisor_root': str(root)})
                        atomic_write(root / 'active.json', {'control_dir': str(active)})
                        path = active / 'suspend_ready.json'
                        ready = read_json(path) if path.exists() else None
                        resumed = active / 'resumed.json'
                        if ready and resumed.exists() and matches(read_json(resumed), ready):
                            ready = None
                    if ready:
                        generation = active / f"generation_{ready['generation']:06d}"
                        receipt = generation / 'offload.json'
                        requested = generation / 'restore.request'
                        fallback = generation / 'fallback.request'
                        due = requested.exists() or (receipt.exists() and
                              time.time() >= read_json(receipt)['lease_until'])
                        if fallback.exists() or (not (root / 'hold').exists() and not (active / 'hold').exists()):
                            if due or fallback.exists() or proc.poll() is not None:
                                try:
                                    if fallback.exists():
                                        raise RuntimeError('Controller requested disk fallback')
                                    if proc.poll() is not None:
                                        raise RuntimeError('Training launcher exited while suspended')
                                    restore_or_fail(active, args.cuda_checkpoint, args.restore_timeout,
                                                    args.ack_timeout)
                                except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                                    if restarts.get(ready['step'], 0) >= args.max_restarts:
                                        raise RuntimeError('Disk restart budget exhausted') from error
                                    # Serialize against offload/restore and request publication.
                                    with exclusive(active / 'controller.lock'), exclusive(active / 'requests.lock'):
                                        command = restart_command(command, active, ready)
                                        atomic_write(generation / 'disk_fallback.json', {
                                            'error': str(error), 'checkpoint': ready['checkpoint'],
                                            'command': command, 'time': time.time()})
                                        stop_owned(proc, ready)
                                    atomic_write(root / 'restart_pending.json', {
                                        'step': ready['step'], 'checkpoint': ready['checkpoint'],
                                        'command': command, 'time': time.time()})
                                    print(f'[suspend] disk fallback from step {ready["step"]}: {error}', flush=True)
                                    restarts[ready['step']] = restarts.get(ready['step'], 0) + 1
                                    attempt += 1
                                    break
                    if proc.poll() is not None:
                        if proc.returncode:
                            raise RuntimeError(f'Training exited with {proc.returncode}')
                        return
                    time.sleep(args.interval)
            finally:
                stop_owned(proc, ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--cuda-checkpoint', default='cuda-checkpoint')
    parser.add_argument('--max-restarts', type=int, default=3)
    parser.add_argument('--restore-timeout', type=float, default=900)
    parser.add_argument('--ack-timeout', type=float, default=120)
    parser.add_argument('--min-free-mib', type=int, default=178000)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--allow-existing-peers', action='store_true', help='Side-smoke only: preserve and allow identity-checked existing GPU users')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or min(args.restore_timeout, args.ack_timeout, args.interval) <= 0 or args.max_restarts < 0:
        parser.error('A command and positive timeouts are required; max-restarts must be nonnegative')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_, signum=sig: sys.exit(128 + signum))
    supervise(args, command)


if __name__ == '__main__':
    main()
