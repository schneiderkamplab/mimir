import argparse

import json

import os

from pathlib import Path

import signal

import subprocess

import sys

import time

from mimir.training.suspension.controller import Controller, CudaBackend

from mimir.training.suspension.protocol import atomic_write, read_json

def make_data(root, full_size=False):
    import numpy as np
    root.mkdir(parents=True)
    epoch = root / 'epoch_0'
    epoch.mkdir()
    rows, length, vocab = (1024, 4097, 262144) if full_size else (256, 65, 256)
    prefix = (length - 1) // 4
    rng = np.random.default_rng(42)
    np.save(root / 'tokens.npy', rng.integers(1, vocab, rows * length, dtype=np.uint32))
    for name, array in {
        'inst_start': np.arange(rows) * length, 'inst_len': np.full(rows, prefix),
        'resp_start': np.arange(rows) * length + prefix, 'resp_len': np.full(rows, length - prefix),
    }.items():
        np.save(epoch / f'{name}.npy', array.astype(np.int64))
    atomic_write(root / 'metadata.json', {'max_seq_len': length, 'total_length': rows * (length - 1),
                 'tokenizer_info': {'vocab_size': vocab}})

def wait_offloaded(proc, backend, controller, seconds, deadline, resume_request=None):
    """Check continued CPU liveness and released GPU memory during long pauses."""
    ready, generation = controller.ready()
    pids = {rank['pid'] for rank in ready['ranks']}
    end = time.monotonic() + seconds
    while True:
        if proc.poll() is not None:
            raise RuntimeError('Worker launcher exited during offloaded pause')
        if resume_request is not None and resume_request.exists():
            controller.journal(generation, {'phase': 'early_resume_requested',
                               'request': str(resume_request)})
            return
        controller.ready()  # Revalidate every worker's process identity.
        states = {pid: backend.state(pid) for pid in pids}
        memory = sum(mib for (pid, _), mib in backend.memory().items() if pid in pids)
        if set(states.values()) != {'checkpointed'} or memory > 64 * len(pids):
            raise RuntimeError(f'Offloaded state changed: {states}, {memory} MiB')
        remaining = max(0, end - time.monotonic())
        controller.journal(generation, {'phase': 'pause_health', 'states': states,
                           'owned_gpu_mib': memory, 'remaining_seconds': remaining})
        print(f'pause: {remaining:.0f}s remaining, {len(pids)} ranks checkpointed, {memory} MiB GPU memory', flush=True)
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Smoke deadline exceeded during offloaded pause')
        time.sleep(min(30, remaining, max(0, deadline - time.monotonic())))
