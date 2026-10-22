from mimir.scheduling import idle


def test_external_process_blocks_even_when_utilization_zero(monkeypatch):
    def query(fields, section):
        return [['0','uuid','7500','0']] if section == 'gpu' else [['uuid','123']]
    monkeypatch.setattr(idle,'query',query)
    assert not idle.available(['0'])[0]
    assert idle.available(['0'],{123})[0]
    assert not idle.available(['1'])[0]


def test_idle_memory_and_utilization_guard(monkeypatch):
    row=['0','uuid','0','0']
    monkeypatch.setattr(idle,'query',lambda fields, section: [row] if section=='gpu' else [])
    assert idle.available(['0'])[0]
    row[2]='2000'
    assert not idle.available(['0'])[0]
    row[2]='0';row[3]='90'
    assert not idle.available(['0'])[0]


def test_gpu_workers_claim_only_their_tasks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from mimir.scheduling.store import Store
    store=Store(tmp_path/'queue.sqlite')
    for gpu in range(8):
        store.add(f'eval{gpu}','eval',{'server':{'gpus':str(gpu)},'resources':[f'gpu:{gpu}']})
    store.add('train','train',{})
    def claim(gpu):
        return store.claim(predicate=lambda j: j['kind']=='eval' and j['config']['server']['gpus']==str(gpu))
    with ThreadPoolExecutor(8) as executor:
        jobs=list(executor.map(claim,range(8)))
    assert {j['id'] for j in jobs}=={f'eval{g}' for g in range(8)}
    assert next(r for r in store.rows() if r['id']=='train')['attempt']==0


def test_monitor_reports_wait_reason(tmp_path):
    import json
    import time
    from mimir.scheduling.store import Store
    from mimir.scheduling.monitor import render
    store=Store(tmp_path/'queue.sqlite')
    (tmp_path/'idle-status.json').write_text(json.dumps({'state':'waiting_for_gpus','time':time.time(),'reason':'external PIDs'}))
    assert 'waiting_for_gpus' in render(store).caption


def test_repeated_backups_replace_only_after_success(tmp_path,monkeypatch):
    import pytest
    from mimir.scheduling.store import Store
    store=Store(tmp_path/'queue.sqlite')
    store.add('first','train',{},held=True)
    assert idle.refresh_backup(store)
    store.add('second','train',{},held=True)
    assert idle.refresh_backup(store)
    backup=tmp_path/'queue.backup.sqlite'
    assert len(Store(backup).rows())==2
    before=backup.read_bytes()
    def fail(path):
        path.write_text('partial')
        raise OSError('full disk')
    monkeypatch.setattr(store,'backup',fail)
    with pytest.warns(RuntimeWarning,match='previous snapshot retained'):
        assert not idle.refresh_backup(store)
    assert backup.read_bytes()==before
    assert not list(tmp_path.glob('*.partial'))
