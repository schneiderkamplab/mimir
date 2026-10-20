"""Single-stream recurrence and orthogonal H MoE / L MoS coverage."""
import copy
from types import SimpleNamespace

import pytest
import torch

from mimir.model.mimir2 import Mimir2, Mimir2Config
from mimir.model.residual_branches import BranchedTransformer
from mimir.model.expert_mixtures import MLPMixture
from test_hrm_experts import config, cpu_attention


@pytest.mark.parametrize('mode', ['hierarchical', 'looped'])
@pytest.mark.parametrize('levels', ['H', 'L', 'both'])
@pytest.mark.parametrize('moe_sparse,mos_sparse', [(False, False), (True, False), (False, True), (True, True)])
def test_h_moe_l_mos_residual_combinations(mode, levels, moe_sparse, mos_sparse):
    model = Mimir2(config(recurrence_mode=mode, moe_levels='H', moe_num_experts=2,
                          moe_top_k=2, mos_num_experts=2, mos_top_k=2,
                          moe_sparse=moe_sparse, mos_sparse=mos_sparse,
                          residual_branches=3, residual_levels=levels))
    model.compute_train_extra_args(SimpleNamespace(step=1))
    assert isinstance(model.H_level.core, BranchedTransformer) == (levels in ('H', 'both'))
    assert isinstance(model.L_level.core, BranchedTransformer) == (levels in ('L', 'both'))
    assert any(isinstance(m, MLPMixture) for m in model.H_level.modules())
    assert not any(isinstance(m, MLPMixture) for m in model.L_level.modules())
    x = torch.randn(2, 4, 16, requires_grad=True)
    y = model(None, x, position_ids=torch.arange(4), bp_steps=8)[1]
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    for level in ('H', 'L'):
        grads = [p.grad for p in getattr(model, level + '_level').parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
        assert all(torch.isfinite(g).all() for g in grads if g is not None)
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(None, x, position_ids=torch.arange(4), bp_steps=8)[1], y)


@pytest.mark.parametrize('h,l', [(1, 1), (2, 1), (2, 3)])
def test_looped_matches_explicit_unrolling_outputs_and_gradients(h, l):
    model = Mimir2(config(recurrence_mode='looped', H_cycles=h, L_cycles=l))
    reference = copy.deepcopy(model)
    x = torch.randn(2, 4, 16, requires_grad=True)
    rx = x.detach().clone().requires_grad_(True)
    expected = rx
    for _ in range(h):
        for _ in range(l):
            expected = reference.L_level.core(expected, position_ids=torch.arange(4))
        expected = reference.H_level.core(expected, position_ids=torch.arange(4))
    actual = model(None, x, position_ids=torch.arange(4), bp_steps=h*(l+1))[1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
    for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(param.grad, ref.grad, msg=name)


@pytest.mark.parametrize('changes', [dict(gates_enabled=True), dict(h_memory_branches=2),
                                   dict(l_memory_branches=2), dict(mos_state_mode='separate')])
def test_looped_rejects_conflicting_state_models(changes):
    with pytest.raises(ValueError, match='Looped recurrence requires'):
        Mimir2Config(**config(recurrence_mode='looped', **changes))


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('h_layers', [0, 1])
def test_looped_cached_decode(h_layers, sparse):
    model = Mimir2(config(recurrence_mode='looped', H_override={'n_layers': h_layers},
                          moe_num_experts=2, moe_levels='H', moe_sparse=sparse,
                          mos_num_experts=2, mos_top_k=1, mos_sparse=sparse,
                          residual_branches=2)).eval()
    x = torch.randn(2, 4, 16)
    cache = model.create_cache(max_batch_size=2, max_seq_len=16, device='cpu', dtype=torch.float32)
    with torch.no_grad():
        full = model(None, x, position_ids=torch.arange(4))[1]
        parts = [model(None, x[:, i:i+1], position_ids=torch.tensor([i]), cache=cache,
                       cache_lengths=torch.full((2,), i))[1] for i in range(4)]
    torch.testing.assert_close(full, torch.cat(parts, 1), atol=1e-5, rtol=1e-4)
    if not h_layers:
        assert not list(model.H_level.parameters())
        assert all(not slot for slot in cache['H'])


@pytest.mark.parametrize('bp', [2, 6])
def test_no_h_unroll_gradients_and_auto_lr(bp):
    from mimir.model.module_learning_rates import configured_module_rates, module_lr_scales, module_lr_metrics
    from mimir.model.lm_head import LMHead
    cfg = config(recurrence_mode='looped', H_override={'n_layers': 0}, init_type='megatron')
    model = Mimir2(cfg)
    reference = copy.deepcopy(model)
    x = torch.randn(2, 4, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    for i in range(6):
        with torch.set_grad_enabled(i >= 6-bp):
            y = reference.L_level.core(y, position_ids=torch.arange(4))
    actual = model(None, x, position_ids=torch.arange(4), bp_steps=bp)[1]
    torch.testing.assert_close(actual, y, rtol=0, atol=0)
    actual.square().mean().backward()
    y.square().mean().backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    train = SimpleNamespace(arch=cfg | {'name': 'mimir2'}, lr_auto=True, lr=.001,
                            lr_h=None, lr_l=None, lr_head=None, lr_embeddings=None)
    rates = configured_module_rates(train, bp)
    assert rates['h'] is None and rates['l'] == pytest.approx(.001/bp)
    assert module_lr_scales(LMHead(model, cfg), train.lr, rates)
    assert 'train/lr_h' not in module_lr_metrics(train, train.lr, bp)
    assert 'train/lr_H' not in module_lr_metrics(train, train.lr, bp)


def test_zero_h_requires_looped():
    with pytest.raises(ValueError, match='Zero-layer H requires looped'):
        Mimir2Config(**config(H_override={'n_layers': 0}))


def test_looped_export_and_serve_fail_explicitly(tmp_path, monkeypatch):
    import importlib
    import json
    import yaml
    convert = importlib.import_module('mimir.export.convert')
    from mimir.serving.server import command, ServerSpec
    monkeypatch.setattr(convert, 'validate', lambda _: {})
    (tmp_path/'all_config.yaml').write_text(yaml.safe_dump({'arch': {'name': 'mimir2', 'recurrence_mode': 'looped'}}))
    (tmp_path/'train_metadata.yaml').write_text('{}')
    with pytest.raises(ValueError, match='Looped recurrence export'):
        convert.export(tmp_path, tmp_path/'output', tmp_path/'tokenizer')
    (tmp_path/'config.json').write_text(json.dumps({'model_type': 'mimir2', 'recurrence_mode': 'looped'}))
    with pytest.raises(ValueError, match='Looped recurrence'):
        command(ServerSpec(str(tmp_path), '0'))
