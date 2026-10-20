import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from mimir.scheduling.store import Store
from mimir.scheduling.worker import run_one
from mimir.evaluation.metrics import merge_samples


def test_failed_eval_and_merge_do_not_strand_training(tmp_path):
    store=Store(tmp_path/'queue.db')
    store.add('eval','inference',{'command':[sys.executable,'-c','raise RuntimeError("bad eval")'],'resources':['gpu:0']})
    store.add('merge','merge',{'inputs':['absent'],'output':str(tmp_path/'average')},[('eval','success')])
    marker=tmp_path/'resumed'
    store.add('resume','train',{'command':[sys.executable,'-c',f'open({str(marker)!r},"w").write("resumed")'],'resources':['gpu:0']},[('eval','terminal')])
    while run_one(store,tmp_path/'attempts'):pass
    assert marker.read_text()=='resumed'
    assert {r['id']:r['state'] for r in store.rows()}=={'eval':'failed','merge':'blocked','resume':'succeeded'}


def test_claims_are_exclusive_and_snapshot_immutable(tmp_path):
    store=Store(tmp_path/'queue.db');store.add('a','eval',{'controls':{'max_connections':8},'resources':['gpu:0']})
    store.add('b','eval',{'resources':['gpu:0']})
    with ThreadPoolExecutor(4) as pool:claims=list(pool.map(lambda _:store.claim(),range(4)))
    active=[x for x in claims if x];assert len(active)==1
    job=active[0]
    with pytest.raises(ValueError):store.edit(job['id'],{})
    store.finish(job['owner'],False,{'reason':'oom'},oom=True)
    retry=store.claim();assert retry['config']['controls']['max_connections']==4
    with store.connect() as db:
        original=json.loads(db.execute('SELECT config FROM attempts WHERE token=?',(job['owner'],)).fetchone()[0])
    assert original['controls']['max_connections']==8
    with pytest.raises(RuntimeError):store.finish(job['owner'],True,{})


def test_abandoned_lease_respects_alive_owners_and_holds(tmp_path):
    store=Store(tmp_path/'queue.db');store.add('a','train',{'resources':['gpu:0']})
    job=store.claim(ttl=.001);time.sleep(.01)
    assert store.recover(lambda _:True)==[]
    assert store.recover(lambda _:False)==['a']
    store.hold('a',True);assert store.claim() is None
    store.hold('a',False);assert store.claim()['attempt']==2
    store.backup(tmp_path/'backup.db');assert Store(tmp_path/'backup.db').rows()[0]['state']=='running'


def test_weighted_sample_merge_and_duplicates(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    a.write_text(json.dumps({'task':'t','sample_id':0,'pass':1,'scores':{'acc':0}})+'\n')
    b.write_text(''.join(json.dumps({'task':'t','sample_id':i,'pass':1,'scores':{'acc':1}})+'\n' for i in range(1,4)))
    assert merge_samples([a,b])['metrics']['t/acc']==.75
    with pytest.raises(ValueError,match='Duplicate'):merge_samples([a,a])


def test_partial_task_set_is_not_published(tmp_path):
    from mimir.evaluation.metrics import complete_task_set
    path=tmp_path/'samples'
    path.write_text(json.dumps({'task':'a','sample_id':1,'pass':1,'scores':{'acc':1}})+'\n')
    with pytest.raises(ValueError,match='Incomplete'):
        complete_task_set([path],{'a':{'samples':2,'passes':1}})
    assert complete_task_set([path],{'a':{'samples':1,'passes':1}})['metrics']['a/acc']==1


def test_invalid_training_handoff_is_rejected(tmp_path):
    store=Store(tmp_path/'queue.db');store.add('eval','eval',{})
    with pytest.raises(ValueError,match='terminal'):
        store.add('train','train',{},[('eval','success')])
    store.add('merge','merge',{},[('eval','success')])
    with pytest.raises(ValueError,match='administrative'):
        store.add('train','train',{},[('merge','terminal')])
    assert len(store.rows())==2


def test_confirmed_direct_backend_oom_retries_with_lower_concurrency(tmp_path,monkeypatch):
    from mimir.evaluation.clients import BackendOOM
    from mimir.scheduling import worker
    store=Store(tmp_path/'queue.db');store.add('eval','eval',{'controls':{'max_connections':8}})
    def oom(*args,**kwargs):raise BackendOOM('CUDA out of memory')
    monkeypatch.setattr(worker,'execute',oom)
    assert worker.run_one(store,tmp_path/'attempts')
    row=store.rows()[0]
    assert row['state']=='pending'
    assert json.loads(row['config'])['controls']['max_connections']==4
