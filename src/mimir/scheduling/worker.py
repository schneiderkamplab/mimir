from pathlib import Path
import json
import os
import subprocess
import threading
import sys
import time
import psutil


def owned_alive(receipt):
    try:
        data=json.loads(Path(receipt).read_text()); p=psutil.Process(data['pid'])
        return p.create_time()==data['created'] and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except FileNotFoundError:
        # Missing ownership evidence is unsafe to reclaim automatically.
        return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess):return False


def execute(job, output, server_pool=None):
    cfg=job['config']; kind=job['kind']; output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if kind=='checkpoint_wait':
        from mimir.checkpoints.contract import validate
        deadline=time.monotonic()+cfg.get('timeout',3600)
        while time.monotonic()<deadline:
            try: return {'checkpoint': cfg['path'], 'manifest': validate(cfg['path'])}
            except (FileNotFoundError,ValueError):time.sleep(1)
        raise TimeoutError('Checkpoint did not become complete')
    if kind in ('merge','average'):
        from mimir.evaluation.metrics import complete_task_set, write_record
        result=complete_task_set(cfg['inputs'],cfg['expected'],classification_tasks=cfg.get('classification_tasks',()),pass_k=cfg.get('pass_k'));write_record(cfg['output'],result);return result
    if kind=='eval':
        from mimir.evaluation.clients import evaluate
        evaluation=dict(cfg['evaluation'])
        if 'server' in cfg:
            if server_pool is None:raise ValueError('A persistent server pool is required')
            from mimir.serving.server import ServerSpec
            spec=ServerSpec(**cfg['server'])
            if spec.key() not in server_pool.entries:
                server_pool.close()
            evaluation['base_url']=server_pool.acquire(spec)
        return evaluate(**evaluation, controls=cfg.get('controls',{}))
    command=cfg.get('command')
    if not isinstance(command,list) or not command or not all(isinstance(v,str) for v in command):
        raise ValueError('Execution requires an explicit argv list')
    env=os.environ.copy();env.update(cfg.get('env',{}))
    with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
        child=subprocess.Popen([sys.executable,'-m','mimir.scheduling.child',str((output/'process.json').resolve()),str(os.getpid()),str(psutil.Process().create_time()),*command],env=env,cwd=cfg.get('cwd'),stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=child.wait(timeout=cfg.get('timeout'))
        except subprocess.TimeoutExpired:
            # Only the process tree created by this attempt is signalled.
            parent=psutil.Process(child.pid); children=parent.children(recursive=True)
            for p in children+[parent]:
                try:p.terminate()
                except psutil.NoSuchProcess:pass
            _,alive=psutil.wait_procs(children+[parent],timeout=10)
            for p in alive:
                try:p.kill()
                except psutil.NoSuchProcess:pass
            child.wait();raise TimeoutError('Owned job timed out')
    return {'returncode':code,'stderr':str(output/'stderr.log')}


def run_one(store, output, server_pool=None, predicate=None):
    job=store.claim(predicate=predicate)
    if not job:return False
    if server_pool is not None and job['kind']=='train':server_pool.close()
    from mimir.evaluation.metrics import write_record
    directory=Path(output)/job['owner'];directory.mkdir(parents=True,exist_ok=True)
    write_record(directory/'owner.json',{'pid':os.getpid(),'created':psutil.Process().create_time()})
    stop=threading.Event(); errors=[]
    def renew():
        while not stop.wait(20):
            try:store.heartbeat(job['owner'])
            except Exception as exc:errors.append(exc);return
    thread=threading.Thread(target=renew,daemon=True);thread.start()
    try:
        result=execute(job,Path(output)/job['owner'],server_pool)
        success=result.get('returncode',0)==0
        stderr=Path(result['stderr']).read_text(errors='replace')[-65536:] if 'stderr' in result else ''
        oom=not success and any(marker in stderr for marker in ('torch.OutOfMemoryError','CUDA out of memory'))
        # Commands are not blindly retried on OOM: only clients with effective controls qualify.
        store.finish(job['owner'],success,result,oom=oom and job['kind']=='eval')
    except Exception as exc:
        from mimir.evaluation.clients import BackendOOM
        store.finish(job['owner'],False,{'error':str(exc),'type':type(exc).__name__},
                     transient=isinstance(exc,(ConnectionError,TimeoutError)),
                     oom=isinstance(exc,BackendOOM) and job['kind']=='eval')
    finally:stop.set();thread.join()
    if errors:raise errors[0]
    return True


def recover_abandoned(store,output):
    output=Path(output)
    def alive(job):
        owner=output/job['owner']/'owner.json'
        child=output/job['owner']/'process.json'
        if owner.exists() and owned_alive(owner):return True
        if child.exists():return owned_alive(child)
        # A claim is recorded before any launch. Missing owner evidence is reclaimed
        # only if the claiming process identity in SQLite has died.
        identity=json.loads(job.get('worker_identity') or '{}')
        if not identity:return True
        try:
            p=psutil.Process(identity['pid'])
            return p.create_time()==identity['created'] and p.is_running()
        except psutil.NoSuchProcess:return False
    return store.recover(alive)
