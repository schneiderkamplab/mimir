"""Administrative CLI; Hydra overrides are passed unchanged to training."""
from pathlib import Path
import json
import sys
import typer

app=typer.Typer(no_args_is_help=True)
checkpoint=typer.Typer(no_args_is_help=True)
app.add_typer(checkpoint,name='checkpoint')

@app.command(context_settings={'allow_extra_args':True,'ignore_unknown_options':True})
def train(ctx:typer.Context,suspendable:bool=False):
    """Train Mimir2 with Hydra overrides (for example size@arch=XXS)."""
    from mimir.training.entry import run
    run(ctx.args,suspendable=suspendable)

@checkpoint.command('import')
def import_checkpoint(source:Path,output:Path,mode:str=typer.Option(...),tag:str=typer.Option(...)):
    """Explicit one-way legacy checkpoint import: exact or weights."""
    from mimir.checkpoints.importer import import_legacy
    import_legacy(source,output,tag=tag,mode=mode)

@app.command('export')
def export_model(checkpoint:Path,output:Path,tokenizer:Path=typer.Option(...),ema:bool=True):
    from mimir.export.convert import export
    export(checkpoint,output,tokenizer,ema=ema)

@app.command()
def serve(model:Path,port:int=8000,gpus:str=typer.Option(...),eager:bool=False):
    from mimir.serving.server import serve as start
    start(model,port,gpus,eager=eager)

@app.command('eval')
def evaluation(suite:Path,model:str,output:Path,base_url:str='http://127.0.0.1:8000/v1'):
    from mimir.evaluation.clients import evaluate
    typer.echo(json.dumps(evaluate(str(suite),model,str(output),base_url)))

@app.command()
def schedule(database:Path,plan:Path|None=None,run:bool=False,
             hold_job:str|None=None,release_job:str|None=None,backup:Path|None=None,
             edit_job:str|None=None,config_file:Path|None=None):
    """Load a dependency plan and/or drain runnable jobs."""
    from mimir.scheduling.store import Store
    from mimir.scheduling.worker import run_one, recover_abandoned
    store=Store(database)
    if hold_job:store.hold(hold_job,True)
    if release_job:store.hold(release_job,False)
    if backup:store.backup(backup)
    if edit_job:
        if config_file is None:raise typer.BadParameter('--edit-job requires --config-file')
        store.edit(edit_job,json.loads(config_file.read_text()))
    if plan:
        for row in json.loads(plan.read_text())['jobs']:
            store.add(row['id'],row['kind'],row['config'],
                      [(d['id'],d.get('policy','success')) for d in row.get('depends',[])],row.get('max_attempts',3),
                      held=row.get('held',False))
    if run:
        from mimir.training.suspension.protocol import exclusive
        from mimir.serving.server import ServerPool
        with exclusive(database.with_suffix('.runner.lock')):
            pool=ServerPool(database.parent/'servers')
            try:
                recover_abandoned(store,database.parent/'attempts')
                while run_one(store,database.parent/'attempts',pool):pass
            finally:pool.close()
    typer.echo(json.dumps(store.rows(),indent=2))

@app.command()
def monitor(database:Path,watch:bool=False):
    from mimir.scheduling.store import Store
    from mimir.scheduling.monitor import monitor as show
    show(Store(database),watch)

@app.command()
def doctor():
    """Read-only dependency and CPU capability checks; never allocates a GPU."""
    from importlib.metadata import version,PackageNotFoundError
    result={}
    for name in ('mimir','torch','transformers','vllm','hydra-core','inspect-ai','euroeval'):
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]='not installed'
    typer.echo(json.dumps(result,indent=2))


def main():app()
