"""Opt-in training-loop hook. Once ready is published, only CPU polling occurs."""

import os
from pathlib import Path
import time
import uuid

from .protocol import atomic_write, claim, process_identity, steps, wait_json


class SuspendHook:
    def __init__(self, root, initial_steps, device, poll_steps=10, wait_seconds=2):
        import torch
        import torch.distributed as dist
        self.torch, self.dist, self.device = torch, dist, device
        self.nccl_shim = None
        if os.environ.get('TRAINING_SUSPEND_NCCL_SHIM', os.environ.get('SUSPEND_TEST_NCCL_SHIM')) == '1':
            if device.type != 'cuda':
                raise ValueError('NCCL checkpoint shim requires CUDA')
            from .nccl_shim import NcclCheckpointShim
            self.nccl_shim = NcclCheckpointShim()
        self.distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world = dist.get_world_size() if self.distributed else 1
        if int(os.environ.get('LOCAL_WORLD_SIZE', self.world)) != self.world:
            raise ValueError('Experimental suspension supports one node only')
        self.queue = steps(initial_steps)
        self.consumed = set()
        self.poll_steps, self.wait_seconds = poll_steps, wait_seconds
        if poll_steps < 1 or wait_seconds <= 0:
            raise ValueError('Suspension poll intervals must be positive')
        self.generation = 0
        self.job_id = self.broadcast(uuid.uuid4().hex if self.rank == 0 else None)
        self.root = Path(root).resolve() / self.job_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = process_identity() | {"rank": self.rank}
        if device.type == 'cuda':
            job_file = os.environ.get('CUDA_CHECKPOINT_JOB_FILE')
            if not job_file:
                raise RuntimeError('Use mimir.training.suspension.controller launch before CUDA initialization')
            stat = os.stat(job_file)
            self.identity['cuda_job'] = [stat.st_dev, stat.st_ino]
        ranks = [self.identity]
        if self.distributed:
            ranks = [None] * self.world
            dist.all_gather_object(ranks, self.identity)
        if len({r['host'] for r in ranks}) != 1:
            raise ValueError('Ranks must be on one host')
        if device.type == 'cuda' and len({tuple(r['cuda_job']) for r in ranks}) != 1:
            raise ValueError('Ranks did not inherit the same CUDA checkpoint job')
        self.ranks = ranks
        if self.rank == 0:
            atomic_write(self.root / 'launch.json', {"job_id": self.job_id, "ranks": ranks,
                         "device": device.type, "initial_steps": self.queue})
            self.save_queue()
            print(f'[suspend] control_dir={self.root}', flush=True)
        self.barrier()

    def broadcast(self, value):
        if self.distributed:
            data = [value]
            self.dist.broadcast_object_list(data, src=0)
            return data[0]
        return value

    def barrier(self):
        if self.distributed:
            self.dist.barrier()
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def save_queue(self):
        atomic_write(self.root / 'queue.json', {"pending": self.queue,
                     "consumed": sorted(self.consumed), "job_id": self.job_id})

    def after_step(self, step, save_checkpoint, stopping=False):
        if stopping:
            return False
        if step % self.poll_steps == 0:
            claimed = None
            values = []
            if self.rank == 0:
                try:
                    claimed, values = claim(self.root)
                except BlockingIOError:
                    pass  # A request writer is publishing; collect it next time.
                except OSError as error:
                    # Never leave peers waiting in a broadcast because of rank-0 I/O.
                    print(f'[suspend] request read failed, retrying next poll: {error}', flush=True)
            values = self.broadcast(values)
            self.queue = sorted((set(self.queue) | set(values)) - self.consumed)
            if self.rank == 0:
                self.save_queue()
                if claimed:
                    claimed.rename(claimed.with_suffix('.accepted'))
        if not self.queue or self.queue[0] > step:
            return False
        checkpoint = save_checkpoint()
        due = [s for s in self.queue if s <= step]
        self.consumed.update(due)
        self.queue = [s for s in self.queue if s > step]
        self.generation += 1
        token = {"job_id": self.job_id, "generation": self.generation, "step": step}
        generation = self.root / f'generation_{self.generation:06d}'
        generation.mkdir(exist_ok=True)
        # Finish collectives before publishing readiness. No GPU calls after this.
        self.barrier()
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()
            self.torch.cuda.synchronize(self.device)
        if self.nccl_shim:
            self.nccl_shim.prepare()
        atomic_write(generation / f'ready.{os.getpid()}.json', token | self.identity)
        if self.rank == 0:
            for rank in self.ranks:
                wait_json(generation / f"ready.{rank['pid']}.json", token, self.wait_seconds, 300)
            self.save_queue()
            atomic_write(self.root / 'suspend_ready.json', token | {"ranks": self.ranks,
                         "checkpoint": str(checkpoint), "requested_steps": due,
                         "ready_at": time.time()})
            print(f'[suspend] ready step={step} generation={self.generation}', flush=True)
        resume = generation / f'resume.{os.getpid()}'
        while True:
            receipt = wait_json(resume, token, self.wait_seconds)
            if receipt.get('start_ticks') == self.identity['start_ticks']:
                break
            time.sleep(self.wait_seconds)
        resume.rename(generation / f'resume.{os.getpid()}.consumed')
        if self.nccl_shim:
            self.nccl_shim.restore()
        self.barrier()
        if self.rank == 0:
            atomic_write(self.root / 'resumed.json', token)
            print(f'[suspend] resumed step={step} generation={self.generation}', flush=True)
        return True
