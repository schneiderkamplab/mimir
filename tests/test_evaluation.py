import json
from pathlib import Path
from mimir.evaluation.clients import evaluate
from mimir.evaluation.metrics import classification_metrics,pass_at_k


def test_direct_inspect_records_raw_generations_and_passes(tmp_path):
    suite=tmp_path/'suite.json'
    suite.write_text(json.dumps({'schema':1,'tasks':[{'id':'smoke','backend':'inspect','provider':'mockllm','task':'mimir.evaluation.tasks.smoke:smoke','samples':1,'passes':2,'generation':{'temperature':0,'max_tokens':16}}]}))
    evaluate(suite,'model',tmp_path/'results')
    rows=[json.loads(s) for s in (tmp_path/'results/smoke.samples.jsonl').read_text().splitlines()]
    assert len(rows)==2 and {r['pass'] for r in rows}=={1,2}
    assert all(r['scores']['exact']==1 for r in rows)
    assert all(r['raw']['output']['completion']=='Default output from mockllm/model' for r in rows)


def test_nonlinear_metrics_and_pass_statistics():
    scores=classification_metrics([(0,0),(0,1),(1,1),(1,1)])
    assert scores['accuracy']==.75
    assert abs(scores['macro_f1']-(2/3+4/5)/2)<1e-8
    assert pass_at_k(1,4,2)==.5


def test_protocol_supersession_is_explicit():
    root=Path(__file__).resolve().parents[1]/'src/mimir/evaluation/suites'
    suite=json.loads((root/'comparison-inspect-v1.json').read_text())
    drop=next(t for t in suite['tasks'] if t['task']=='inspect_evals/drop')
    assert not drop['generation'].get('stop_seqs')
    assert drop['samples']==9535  # Producing log, not the older 9536 guide.
    assert all(t['passes']>0 and t['header_sha256'] for t in suite['tasks'])
    score=json.loads((root/'bertscore-v1.json').read_text())
    assert score['layers']==17 and not score['idf'] and not score['rescale_with_baseline']


def test_classification_is_aggregated_per_pass(tmp_path):
    from mimir.evaluation.metrics import complete_task_set
    path=tmp_path/'scores'
    rows=[]
    for epoch,pairs in enumerate([[(0,0),(1,0)],[(0,1),(1,1)]],1):
        for i,(target,predicted) in enumerate(pairs):
            rows.append({'task':'classes','sample_id':i,'pass':epoch,'target_class':target,'predicted_class':predicted,'scores':{}})
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    record=complete_task_set([path],{'classes':{'samples':2,'passes':2}},classification_tasks=['classes'])
    assert abs(record['metrics']['classes/macro_f1']-1/3)<1e-8
    assert len(record['pass_metrics']['classes'])==2
