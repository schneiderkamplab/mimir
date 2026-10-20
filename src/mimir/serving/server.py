"""Direct custom-vLLM serving. No proxy and no prompt rewriting."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

@dataclass(frozen=True)
class ServerSpec:
    model: str
    gpus: str
    port: int = 8000
    eager: bool = False
    max_model_len: int | None = None
    template: str | None = None

    def key(self):
        model=Path(self.model)
        receipt=(model/'export_receipt.json').read_bytes()
        template=Path(self.template).read_bytes() if self.template else b''
        return hashlib.sha256(receipt+template+json.dumps(self.__dict__,sort_keys=True).encode()).hexdigest()


def command(spec):
    cfg=json.loads((Path(spec.model)/'config.json').read_text())
    if cfg.get('model_type')!='mimir2':raise ValueError('Only Mimir2 can be served')
    if cfg.get('recurrence_mode', 'hierarchical') != 'hierarchical':
        raise ValueError('Looped recurrence is not supported by the pinned inference forks')
    if cfg.get('mos_sparse') and not spec.eager:raise ValueError('Sparse MoS requires explicitly selected eager execution')
    if not spec.gpus or any(not s.isdigit() for s in spec.gpus.split(',')):raise ValueError('Explicit GPU indices required')
    args=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',spec.model,
          '--host','127.0.0.1','--port',str(spec.port),'--tensor-parallel-size',str(len(spec.gpus.split(',')))]
    if spec.eager:args.append('--enforce-eager')
    if spec.max_model_len:args.extend(['--max-model-len',str(spec.max_model_len)])
    if spec.template:args.extend(['--chat-template',spec.template])
    return args


def serve(model,port,gpus,*,eager=False):
    spec=ServerSpec(str(model),gpus,port,eager)
    subprocess.run(command(spec),env=os.environ|{'CUDA_VISIBLE_DEVICES':gpus},check=True)

class ServerPool:
    """Reuse exactly compatible servers for the life of a scheduler worker."""
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.entries={}
    def acquire(self,spec,timeout=300):
        key=spec.key()
        if key in self.entries:
            proc,handle=self.entries[key]
            if proc.poll() is None:return f'http://127.0.0.1:{spec.port}/v1'
            handle.close();del self.entries[key]
        handle=(self.root/f'{key}.log').open('ab')
        proc=subprocess.Popen(command(spec),env=os.environ|{'CUDA_VISIBLE_DEVICES':spec.gpus},stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        self.entries[key]=(proc,handle);deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError('Owned vLLM server exited; inspect its log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{spec.port}/health',timeout=1):return f'http://127.0.0.1:{spec.port}/v1'
            except OSError:time.sleep(1)
        self.close();raise TimeoutError('Server readiness timeout')
    def close(self):
        import signal
        for proc,handle in self.entries.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            handle.close()
        self.entries.clear()
