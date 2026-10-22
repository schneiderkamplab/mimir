import importlib.util
import json
from pathlib import Path
from mimir.scheduling.store import Store


def test_append_three_runs_preserves_existing_jobs(tmp_path):
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('extend_ablation',root/'tools/extend_xxs_ablation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    store=Store(tmp_path/'queue.sqlite')
    (tmp_path/'configs').mkdir()
    jobs=[]
    for variant in ['default','engram','moe','mos']:
        extra={'moe':['arch.moe_num_experts=4'],'mos':['arch.mos_num_experts=3']}.get(variant,[])
        overrides=['size@arch=XXS','data.path=fixture','training_total_steps=20000']+extra
        (tmp_path/'configs'/f'{variant}.overrides.json').write_text(json.dumps(overrides))
        for kind in ['train','eval']:
            job={'id':variant+'.'+kind,'kind':kind,'config':{'command':['python',*overrides]},'depends':[], 'max_attempts':1}
            store.add(job['id'],kind,job['config'],held=True);jobs.append(job)
    before=store.rows()
    (tmp_path/'scheduled-plan.json').write_text(json.dumps({'jobs':jobs}))
    (tmp_path/'preparation.json').write_text(json.dumps({'database':str(store.path),'training_order':['default','engram','moe','mos']}))
    assert module.extend(tmp_path,release=True)==6
    after={r['id']:r for r in store.rows()}
    assert all(after[r['id']]==r for r in before)
    additions=json.loads((tmp_path/'extension-plan.json').read_text())['jobs']
    trains=[j for j in additions if j['kind']=='train']
    assert [j['depends'] for j in trains]==[[{'id':v+'.eval','policy':'terminal'}] for v in ['mos','muon','moe_balanced']]
    assert 'optimizer=muon' in trains[0]['config']['command']
    assert 'arch.moe_aux_loss_free=true' in trains[1]['config']['command']
    assert 'arch.mos_aux_loss_free=true' in trains[2]['config']['command']
    assert all(not j['held'] for j in additions)
    assert all(r['attempt']==0 for r in store.rows())
