from pathlib import Path
import json
import time
from rich.table import Table


def render(store):
    rows=store.rows();counts={}
    table=Table(title='Mimir queue',caption='ETAs use recorded progress rates; absent evidence is shown as unknown.')
    for label in ('Job / task','State','Attempt','GPU / judge resources','Progress','ETA'):table.add_column(label)
    for row in rows:
        counts[row['state']]=counts.get(row['state'],0)+1
        cfg=json.loads(row['config']);progress='unknown';eta='unknown'
        path=cfg.get('progress_file')
        if path:
            try:
                data=json.loads(Path(path).read_text());done=data['completed'];total=data['total']
                progress=f'{done}/{total}'
                if 'passes' in data:progress+=f"; passes {data['passes']}"
                elapsed=data.get('elapsed_seconds',0)
                if done>0 and elapsed>0:eta=f'{max(0,total-done)*elapsed/done:.0f}s'
            except (OSError,ValueError,KeyError):pass
        table.add_row(row['id'],row['state']+(' (held)' if row['held'] else ''),str(row['attempt']),', '.join(cfg.get('resources',[])),progress,eta)
    table.title='Mimir queue — '+', '.join(f'{n} {s}' for s,n in sorted(counts.items()))
    try:
        status=json.loads((store.path.parent/'idle-status.json').read_text())
        age=max(0,time.time()-status['time'])
        table.caption=f"Scheduler: {status['state']} ({age:.0f}s ago). {status.get('reason','')}"
    except (OSError,ValueError,KeyError):pass
    return table


def monitor(store,watch=False):
    from rich.console import Console
    if not watch:Console().print(render(store));return
    from rich.live import Live
    with Live(render(store),refresh_per_second=1) as live:
        while True:time.sleep(1);live.update(render(store))
