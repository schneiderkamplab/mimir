"""Portable single-owner checkpoint/stop/restart suspension."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from mimir.training.suspension.protocol import exclusive, atomic_write

class DiskHook:
    def __init__(self, root, boundaries, interval):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.boundaries=set(boundaries);self.interval=interval

    def after_step(self, step, save_checkpoint, stopping=False, **kwargs):
        if stopping:return
        request=self.root/'suspend.request'
        requested=step in self.boundaries or (step % self.interval == 0 and request.exists())
        if not requested:return
        checkpoint=save_checkpoint()
        atomic_write(self.root/'disk-ready.json',{'step':step,'checkpoint':checkpoint})
        request.unlink(missing_ok=True)
        raise SystemExit(75)


def supervise(overrides, config):
    if not config.checkpoint_path:
        raise ValueError('--suspendable requires checkpoint_path=<directory>')
    root=Path(config.suspend_control_dir or Path(config.checkpoint_path)/'control').resolve()
    root.mkdir(parents=True,exist_ok=True)
    args=[x for x in overrides if not x.startswith('suspend_control_dir=')]+[f'suspend_control_dir={root}']
    with exclusive(root/'recovery.lock'):
        while True:
            while (root/'hold').exists():time.sleep(1)
            result=subprocess.run([sys.executable,'-m','mimir','train',*args])
            if result.returncode==0:return
            if result.returncode!=75:raise RuntimeError(f'Training failed with status {result.returncode}')
            ready=json.loads((root/'disk-ready.json').read_text())
            directory=Path(ready['checkpoint']).parent
            from mimir.checkpoints.contract import validate
            manifest=validate(directory)
            if manifest['step']!=ready['step']:raise ValueError('Recovery step mismatch')
            replacements={'resume_checkpoint_path':str(directory),'resume_checkpoint_tag':manifest['tag']}
            args=[x for x in args if x.split('=',1)[0] not in replacements]+[f'{k}={v}' for k,v in replacements.items()]
