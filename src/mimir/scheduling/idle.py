"""Serial scheduler waiting for external GPU users without signalling them."""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import warnings

import psutil

from mimir.scheduling.store import Store
from mimir.scheduling.worker import recover_abandoned, run_one
from mimir.serving.server import ServerPool
from mimir.training.suspension.protocol import exclusive
from mimir.evaluation.metrics import write_record


def query(fields, section):
    output = subprocess.check_output(['nvidia-smi', f'--query-{section}={fields}',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10)
    return list(csv.reader(io.StringIO(output), skipinitialspace=True))


def available(gpus, owned=()):
    devices = {row[0]: row for row in query('index,uuid,memory.used,utilization.gpu', 'gpu')}
    if any(g not in devices for g in gpus):
        return False, 'Requested GPU missing'
    processes = query('gpu_uuid,pid', 'compute-apps')
    reasons = []
    for gpu in gpus:
        _, uuid, memory, utilization = devices[gpu]
        pids = [int(pid) for device,pid in processes if device == uuid]
        foreign = [pid for pid in pids if pid not in owned]
        if foreign:
            reasons.append(f'GPU {gpu}: external PIDs {foreign}')
        elif not pids and (float(memory) > 1024 or float(utilization) > 5):
            reasons.append(f'GPU {gpu}: memory/utilization not idle')
    return not reasons, '; '.join(reasons) or 'No external GPU users detected'


def refresh_backup(store):
    """Publish a verified replacement; backup failures must not strand training."""
    target=store.path.parent/'queue.backup.sqlite'
    staged=target.with_name(f'.{target.name}.{uuid.uuid4().hex}.partial')
    try:
        store.backup(staged)
        staged.replace(target)
        return True
    except Exception as exc:
        warnings.warn(f'Queue backup failed; previous snapshot retained: {exc}', RuntimeWarning)
        return False
    finally:
        staged.unlink(missing_ok=True)


def run(database, gpus, poll=30, quiet_polls=3):
    database = Path(database)
    store = Store(database)
    status = database.parent/'idle-status.json'
    with exclusive(database.with_suffix('.runner.lock')):
        pool = ServerPool(database.parent/'servers')
        stable = 0
        try:
            recover_abandoned(store, database.parent/'attempts')
            while True:
                rows = store.rows()
                if not any(r['state'] in ('pending','running') for r in rows):
                    write_record(status, {'state':'complete','pid':os.getpid(),'time':time.time()})
                    return
                owned = {p.pid for p in psutil.Process().children(recursive=True)}
                try:
                    free, reason = available(gpus, owned)
                except Exception as exc:
                    free, reason = False, f'GPU probe failed: {exc}'
                stable = stable + 1 if free else 0
                write_record(status, {'state':'waiting_for_gpus' if stable < quiet_polls else 'ready',
                    'pid':os.getpid(),'gpus':gpus,'reason':reason,'quiet_polls':stable,'time':time.time()})
                if stable >= quiet_polls:
                    if run_one(store, database.parent/'attempts', pool, predicate=lambda j: j['kind'] != 'eval'):
                        refresh_backup(store)
                        stable = 0
                        continue
                    # One isolated client/server process per GPU. Each drains its
                    # assigned ready tasks and reuses its TP=1 server. Joining all
                    # children ensures the next training run has no eval overlap.
                    children = [subprocess.Popen([sys.executable, '-m',
                        'mimir.scheduling.idle', str(database), '--gpus', gpu,
                        '--eval-worker'], env=os.environ | {'CUDA_VISIBLE_DEVICES':gpu}) for gpu in gpus]
                    codes = [child.wait() for child in children]
                    if any(codes):
                        raise RuntimeError('Evaluation worker exited unexpectedly; inspect runner log')
                    refresh_backup(store)
                    stable = 0
                    # Pending user holds remain authoritative; no automatic release.
                time.sleep(poll)
        finally:
            pool.close()


def evaluate_assigned(database, gpu):
    database = Path(database)
    store = Store(database)
    pool = ServerPool(database.parent/'servers'/f'gpu-{gpu}')
    try:
        while True:
            owned = {p.pid for p in psutil.Process().children(recursive=True)}
            try:
                free, _ = available([gpu], owned)
            except Exception:
                free = False
            if not free:
                time.sleep(30)
                continue
            if not run_one(store, database.parent/'attempts', pool,
                    predicate=lambda j: j['kind']=='eval' and j['config'].get('server',{}).get('gpus')==gpu):
                return
    finally:
        pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database',type=Path)
    parser.add_argument('--gpus',required=True)
    parser.add_argument('--poll',type=float,default=30)
    parser.add_argument('--eval-worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    gpus=args.gpus.split(',')
    if not gpus or any(not g.isdigit() for g in gpus) or len(set(gpus))!=len(gpus):
        parser.error('Provide distinct numeric GPU indices')
    if args.poll <= 0:parser.error('poll must be positive')
    if args.eval_worker:
        if len(gpus)!=1:parser.error('Evaluation worker requires exactly one GPU')
        evaluate_assigned(args.database,gpus[0])
    else:
        run(args.database,gpus,args.poll)


if __name__ == '__main__':
    main()
