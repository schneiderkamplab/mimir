import json
import torch
import yaml
import pytest
from mimir.checkpoints.importer import import_legacy,initialize_weights
from mimir.checkpoints.contract import validate,resolve


def legacy(root):
    root.mkdir()
    (root/'all_config.yaml').write_text(yaml.safe_dump({'arch':{'name':'baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel','head':'lm_head@LMHead'},'checkpoint_format':'unsharded','ema':.9}))
    (root/'train_metadata.yaml').write_text(yaml.safe_dump({'vocab_size':32,'max_seq_len':8}))
    (root/'checkpoint_state_step_1.json').write_text(json.dumps({'step':1,'epoch':1,'batch_in_epoch':1,'world_size':1,'carry_policy':'none'}))
    torch.save({'model':{'weight':torch.tensor([[3.]])},'optim':{'state':{0:{'step':torch.tensor(1.),'param_ema':torch.tensor([[2.]])}}}},root/'unsharded_step_1.pt')


def test_weights_initialization_is_explicit_and_exact_missing_rng_rejected(tmp_path):
    source=tmp_path/'legacy';legacy(source)
    with pytest.raises(ValueError,match='Exact continuation unavailable'):
        import_legacy(source,tmp_path/'exact',tag='step_1',mode='exact')
    assert not (tmp_path/'exact').exists()
    out=tmp_path/'weights';import_legacy(source,out,tag='step_1',mode='weights')
    assert validate(out)['discarded']==['optimizer','EMA','RNG','cursor','schedules']
    m=torch.nn.Linear(1,1,bias=False);initialize_weights(m,out)
    assert m.weight.item()==3.
    with pytest.raises(ValueError,match='Weights-only'):resolve(out,'step_1')


def test_import_rejects_other_architectures_and_never_overwrites(tmp_path):
    source=tmp_path/'legacy';legacy(source)
    (source/'all_config.yaml').write_text(yaml.safe_dump({'arch':{'name':'other'}}))
    with pytest.raises(ValueError,match='Only Mimir'):import_legacy(source,tmp_path/'out',tag='step_1',mode='weights')
    out=tmp_path/'out';out.mkdir()
    with pytest.raises(FileExistsError):import_legacy(source,out,tag='step_1',mode='weights')
