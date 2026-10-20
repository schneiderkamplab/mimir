"""Real two-rank Gloo protocol test, with no CUDA initialization or offload."""
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import unittest

from mimir.training.suspension.protocol import atomic_write, read_json, submit


def worker(rank, root, rendezvous):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    import torch.distributed as dist
    from mimir.training.suspension.runtime import SuspendHook
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        hook = SuspendHook(root, [2, 2], torch.device('cpu'), poll_steps=1, wait_seconds=.02)
        totals = []
        for step in range(1, 7):
            value = torch.tensor(float(rank + step))
            dist.all_reduce(value)
            totals.append(value.item())
            hook.after_step(step, lambda: 'cpu-test-checkpoint')
        atomic_write(Path(root) / f'result.{rank}.json', {'totals': totals, 'queue': hook.queue,
                     'consumed': sorted(hook.consumed), 'generations': hook.generation})
    finally:
        dist.destroy_process_group()


class DistributedProtocolTests(unittest.TestCase):
    def test_two_cycles_late_requests_and_cpu_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = mp.get_context('spawn')
            processes = [ctx.Process(target=worker, args=(r, tmp, str(root / 'rendezvous'))) for r in range(2)]
            for p in processes:
                p.start()
            deadline = time.monotonic() + 90
            released = 0
            try:
                while any(p.is_alive() for p in processes):
                    self.assertLess(time.monotonic(), deadline, 'CPU ranks failed to resume')
                    for path in root.glob('*/suspend_ready.json'):
                        ready = read_json(path)
                        generation = ready['generation']
                        if generation <= released:
                            continue
                        self.assertEqual(len(ready['ranks']), 2)
                        self.assertEqual(ready['step'], 2 if generation == 1 else 3)
                        if generation == 1:
                            # One past request, one consumed duplicate: one new pause at 3.
                            submit(path.parent, [1, 2, 3])
                        directory = path.parent / f'generation_{generation:06d}'
                        for rank in ready['ranks']:
                            atomic_write(directory / f"resume.{rank['pid']}", ready | rank)
                        released = generation
                    time.sleep(.02)
                for p in processes:
                    p.join(5)
                    self.assertEqual(p.exitcode, 0)
                self.assertEqual(released, 2)
                for rank in range(2):
                    result = read_json(root / f'result.{rank}.json')
                    self.assertEqual(result['totals'], [3, 5, 7, 9, 11, 13])
                    self.assertEqual(result['queue'], [])
                    self.assertEqual(result['consumed'], [1, 2, 3])
            finally:
                for p in processes:
                    if p.is_alive():
                        p.terminate()  # Only this test's CPU child processes.
                    p.join(5)


if __name__ == '__main__':
    unittest.main()
