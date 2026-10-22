import importlib.util
import json
from pathlib import Path

from typer.testing import CliRunner
from mimir.cli import app
from mimir.scheduling.store import Store


def test_held_plan_import_cannot_execute(tmp_path):
    marker = tmp_path/'executed'
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'jobs': [{'id': 'held', 'kind': 'train',
        'held': True, 'config': {'command': ['touch', str(marker)]}}]}))
    db = tmp_path/'queue.sqlite'
    result = CliRunner().invoke(app, ['schedule', str(db), '--plan', str(plan), '--run'])
    assert result.exit_code == 0, result.output
    assert not marker.exists()
    row = Store(db).rows()[0]
    assert row['held'] and row['state'] == 'pending' and row['attempt'] == 0


def test_prepared_campaign_capacity_and_handoffs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('prepare_ablation', Path(__file__).parents[1]/'tools/prepare_xxs_ablation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'inspect_data', lambda _: {'metadata': {
        'total_length': 4_000_000_000, 'tokenizer_info': {'vocab_size': 256}, 'max_seq_len': 17}})
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    tokenizer = tmp_path/'tokenizer'
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]')))
    tok.save_pretrained(tokenizer)
    template = tmp_path/'template.jinja'
    template.write_text("{% for message in messages %}{{ message['content'] }}{% endfor %}")
    output = module.prepare(tmp_path/'prepared', tmp_path/'data', tokenizer, template,
                            eval_steps=[5000, 20000], database=tmp_path/'queue.sqlite')
    jobs = json.loads((output/'plan.json').read_text())['jobs']
    assert len({j['id'] for j in jobs}) == len(jobs)
    assert all(j['held'] for j in jobs)
    assert Store(tmp_path/'queue.sqlite').claim() is None
    counts = json.loads((output/'parameters.json').read_text())
    assert {k: v['capacity'] for k,v in counts.items()} == {
        'default': 5505024, 'engram': 11010048, 'moe': 10816524, 'mos': 11010819}
    for job in jobs:
        if job['kind'] == 'train':
            assert all(d['policy'] == 'terminal' for d in job['depends'] if not d['id'].endswith('.train'))
            assert not any(d['id'].endswith('.merge') for d in job['depends'])
            assert job['config']['env']['CUDA_VISIBLE_DEVICES'] == ''
        if job['kind'] == 'eval':
            assert job['config']['server']['gpus'] == ''
    assert all(r['attempt'] == 0 for r in Store(tmp_path/'queue.sqlite').rows())
