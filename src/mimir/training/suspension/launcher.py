"""Resource-owning launcher for the opt-in NCCL checkpoint shim."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .protocol import atomic_write, exclusive, read_json


def child_environment(settings, kvs):
    shim, nccl = Path(settings['shim_library']), Path(settings['nccl_library'])
    for library in (shim, nccl):
        if not library.is_file():
            raise FileNotFoundError(library)
    env = os.environ.copy()
    env.update(LD_PRELOAD=f'{shim}:{nccl}',
               LD_LIBRARY_PATH=':'.join([str(nccl.parent), *settings.get('library_dirs', [])]),
               NCCL_CHECKPOINT_KVS_PATH=str(kvs), TRAINING_SUSPEND_NCCL_SHIM='1')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Training command required')
    settings = read_json(args.settings)
    root = args.control_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_, signum=sig: sys.exit(128 + signum))
    with exclusive(root / 'launcher.lock'):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        kvs = root / 'kvs.txt'
        atomic_write(kvs, f'127.0.0.1:{port}\n')
        env = child_environment(settings, kvs)
        supervisor = None
        with (root / 'redis.log').open('a') as log:
            redis = subprocess.Popen([settings['redis_server'], '--bind', '127.0.0.1',
                                      '--port', str(port), '--save', '', '--appendonly', 'no',
                                      '--protected-mode', 'yes'], stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 20
                while True:
                    if redis.poll() is not None:
                        raise RuntimeError('Dedicated Redis failed to start; see redis.log')
                    try:
                        with socket.create_connection(('127.0.0.1', port), timeout=1) as conn:
                            conn.sendall(b'*1\r\n$4\r\nPING\r\n')
                            if conn.recv(1024).startswith(b'+PONG'):
                                break
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Redis startup deadline')
                    time.sleep(.1)
                supervisor = subprocess.Popen([
                    sys.executable, '-m', 'mimir.training.suspension.supervisor',
                    '--control-root', str(root), '--cuda-checkpoint', settings['cuda_checkpoint'],
                    '--max-restarts', str(settings.get('max_restarts', 3)),
                    '--min-free-mib', str(settings.get('min_free_mib', 178000)),
                    *(['--allow-existing-peers'] if settings.get('allow_existing_peers') else []),
                    '--', *command], env=env)
                status = supervisor.wait()
                if status:
                    raise SystemExit(status)
            finally:
                try:
                    if supervisor is not None and supervisor.poll() is None:
                        supervisor.terminate()
                        supervisor.wait(timeout=90)
                finally:
                    redis.terminate()
                    redis.wait(timeout=20)


if __name__ == '__main__':
    main()
