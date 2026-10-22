"""Append the authorized Muon and balanced-routing runs to a prepared queue."""
import argparse
import copy
import json
from pathlib import Path
from omegaconf import OmegaConf
from mimir.scheduling.store import Store
from mimir.training.entry import configuration

ROOT=Path(__file__).resolve().parents[1]


def extend(output, *, release=False):
    output=Path(output).absolute()
    recipe=json.loads((ROOT/'campaigns/xxs-dfm15-20k/extensions.json').read_text())
    plan=json.loads((output/'scheduled-plan.json').read_text())
    receipt=json.loads((output/'preparation.json').read_text())
    store=Store(receipt['database'])
    existing={r['id'] for r in store.rows()}
    additions=[]
    previous=recipe['after']
    for name, specification in recipe['variants'].items():
        source=specification['source']
        templates=[j for j in plan['jobs'] if j['id'].startswith(source+'.')]
        if not templates:raise ValueError(f'Missing source: {source}')
        if sum(j['kind']=='train' for j in templates)!=1:
            raise ValueError('This extension requires final-only evaluation preparations')
        if any(j.startswith(name+'.') for j in existing):raise ValueError(f'Already queued: {name}')
        common=json.loads((output/'configs'/f'{source}.overrides.json').read_text())
        def replace(text):
            return text.replace(f'/runs/{source}/',f'/runs/{name}/').replace(
                f'run_name=xxs-dfm15-20k-{source}',f'run_name=xxs-dfm15-20k-{name}')
        overrides=[replace(s) for s in common]+specification['overrides']
        config=configuration(overrides)
        OmegaConf.save(config,output/'configs'/f'{name}.yaml')
        (output/'configs'/f'{name}.overrides.json').write_text(json.dumps(overrides,indent=2)+'\n')
        for template in templates:
            job=copy.deepcopy(template)
            job['id']=name+job['id'][len(source):]
            job['held']=True
            job['config']=json.loads(replace(json.dumps(job['config'])))
            for dependency in job['depends']:
                if dependency['id'].startswith(source+'.'):
                    dependency['id']=name+dependency['id'][len(source):]
            if job['kind']=='train':
                job['config']['command']+=specification['overrides']
                job['depends']=[{'id':j['id'],'policy':'terminal'} for j in plan['jobs']+additions
                                if j['id'].startswith(previous+'.') and j['kind']=='eval']
            additions.append(job)
        previous=name
    # Config validation precedes registration; every inserted job starts held.
    for job in additions:
        store.add(job['id'],job['kind'],job['config'],
                  [(d['id'],d['policy']) for d in job['depends']],job['max_attempts'],held=True)
    if release:
        with store.transaction() as db:
            db.executemany('UPDATE jobs SET held=0 WHERE id=? AND state=\'pending\'',[(j['id'],) for j in additions])
        for job in additions:job['held']=False
    plan['jobs']+=additions
    (output/'scheduled-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'extension-plan.json').write_text(json.dumps({'jobs':additions},indent=2)+'\n')
    store.backup(output/'queue.extended.backup.sqlite')
    receipt['training_order']+=list(recipe['variants'])
    receipt['jobs']=len(store.rows())
    receipt['held_jobs']=sum(r['held'] for r in store.rows())
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return len(additions)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--release',action='store_true',help='Release new jobs after preparation')
    args=parser.parse_args()
    print(f'Added {extend(args.output,release=args.release)} jobs')
