import copy
from types import SimpleNamespace

import pytest
import torch

from mimir.model.mimir2 import Mimir2, Mimir2Config
from mimir.model.residual_branches import BranchExpertMixer
from mimir.model.activation_checkpointing import apply_activation_checkpointing
from mimir.model.module_learning_rates import backward_call_counts
from test_hrm_experts import config, cpu_attention


def make(**changes):
    cfg = config(**(dict(residual_branches=2, residual_levels='H', h_residual_persistence=True,
                          mos_num_experts=2, mos_top_k=2) | changes))
    model = Mimir2(cfg)
    model.compute_train_extra_args(SimpleNamespace(step=2000))
    # Break expert symmetry so output-mixing tests are meaningful.
    with torch.no_grad():
        model.L_level.stack_experts[0].layers[0].mlp.down_proj.weight.add_(.03)
    return model


@pytest.mark.parametrize('persistence', [False, True])
@pytest.mark.parametrize('mixing', [False, True])
@pytest.mark.parametrize('bp', [2, 8])
def test_manual_unroll_outputs_and_gradients(persistence, mixing, bp):
    model = make(h_residual_persistence=persistence, h_branch_l_mixing=mixing)
    if mixing:
        model.H_level.l_branch_mixer.logits.data.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
    ref = copy.deepcopy(model)
    x = torch.randn(2, 4, 16, requires_grad=True)
    rx = x.detach().clone().requires_grad_(True)
    info = dict(position_ids=torch.arange(4))
    h, l, streams = rx, ref.zL_init, ref.H_level.core.initialize_streams(rx)
    h_bp, l_bp = backward_call_counts(2, 3, bp)
    for i in range(2):
        for k in range(i*3, (i+1)*3):
            with torch.set_grad_enabled(k >= 6-l_bp):
                w = ref.L_level.stack_router(l+h)
                ys = [core(l+h, **info) for core in [ref.L_level.core, *ref.L_level.stack_experts]]
                l = sum(w[..., e:e+1] * y for e, y in enumerate(ys))
        with torch.set_grad_enabled(i >= 2-h_bp):
            if not persistence:
                streams = ref.H_level.core.initialize_streams(h)
            if mixing:
                preference = ref.H_level.l_branch_mixer.logits.softmax(-1)
                weights = w.unsqueeze(-2) * preference
                weights = weights / weights.sum(-1, keepdim=True)
                injection = torch.einsum('...be,...ed->...bd', weights, torch.stack(ys, -2))
            else:
                injection = l.unsqueeze(-2)
            streams = ref.H_level.core.process_streams(streams+injection, **info)
            h = ref.H_level.core.readout(streams)
    actual = model(None, x, bp_steps=bp, **info)[1]
    torch.testing.assert_close(actual, h, atol=2e-6, rtol=1e-5)
    actual.square().sum().backward()
    h.square().sum().backward()
    torch.testing.assert_close(x.grad, rx.grad, atol=2e-5, rtol=1e-4)
    for (name, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_mixer_support_preferences_and_top1_surrogate():
    from mimir.model.expert_mixtures import TopKRouter
    mixer = BranchExpertMixer(2, 3, init_scale=0.0)
    values = [torch.randn(2, 4) for _ in range(3)]
    w = torch.tensor([[.3, .7, 0.], [.8, .2, 0.]])
    entries = [(y, w[:, i:i+1]) for i, y in enumerate(values)]
    expected = sum(y * w[:, i:i+1] for i, y in enumerate(values))
    torch.testing.assert_close(mixer(entries), expected[:, None].expand(-1, 2, -1))
    mixer.logits.data.copy_(torch.tensor([[3., -3., 100.], [-3., 3., 100.]]))
    actual = mixer(entries)
    assert not torch.allclose(actual[:, 0], actual[:, 1])
    altered = [(y if i != 2 else y+1000, weight) for i, (y, weight) in enumerate(entries)]
    torch.testing.assert_close(mixer(altered), actual)
    router = TopKRouter(4, 3, 1)
    weights = router(torch.randn(2, 4))
    out = mixer([(y, weights[:, i:i+1]) for i, y in enumerate(values)])
    torch.testing.assert_close(out[:, 0], values[0])
    torch.testing.assert_close(out[:, 1], values[0])
    out.square().sum().backward()
    assert router.biases[0].grad.abs().sum() > 0
    torch.testing.assert_close(mixer.logits.grad, torch.zeros_like(mixer.logits.grad), atol=1e-6, rtol=0)


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('state', ['shared', 'separate'])
@pytest.mark.parametrize('topk', [1, 2])
def test_cache_checkpointing_and_roundtrip(sparse, state, topk):
    model = make(h_branch_l_mixing=True, mos_sparse=sparse, mos_top_k=topk, mos_state_mode=state,
                 moe_num_experts=2, moe_levels='H', moe_top_k=2, moe_sparse=sparse,
                 gates_enabled=True, residual_input_dependent=True)
    model.H_level.l_branch_mixer.logits.data.copy_(torch.tensor([[.8, -.8], [-.8, .8]]))
    clone = copy.deepcopy(model)
    apply_activation_checkpointing(clone, 'full')
    x = torch.randn(2, 4, 16)
    info = dict(position_ids=torch.arange(4))
    a, b = model(None, x, bp_steps=8, **info)[1], clone(None, x, bp_steps=8, **info)[1]
    torch.testing.assert_close(a, b)
    a.square().sum().backward(); b.square().sum().backward()
    for p, q in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    model.eval()
    cache = model.create_cache(max_batch_size=2, max_seq_len=16, device='cpu', dtype=torch.float32)
    with torch.no_grad():
        full = model(None, x, **info)[1]
        parts = [model(None, x[:, i:i+1], position_ids=torch.tensor([i]), cache=cache,
                       cache_lengths=torch.full((2,), i))[1] for i in range(4)]
        torch.testing.assert_close(full, torch.cat(parts, 1), atol=1e-5, rtol=1e-4)
        clone.load_state_dict(model.state_dict())
        clone.eval()
        torch.testing.assert_close(clone(None, x, **info)[1], full)
        # State belongs to this forward, never to another request.
        model(None, x+1, **info)
        torch.testing.assert_close(model(None, x, **info)[1], full)


@pytest.mark.parametrize('changes', [dict(residual_branches=1), dict(residual_levels='L'),
    dict(recurrence_mode='looped'), dict(h_memory_branches=2),
    dict(h_branch_l_mixing=True, mos_num_experts=1),
    dict(h_branch_l_mixing=True, l_memory_branches=2)])
def test_invalid_combinations(changes):
    with pytest.raises(ValueError):
        Mimir2Config(**config(**(dict(h_residual_persistence=True, residual_branches=2) | changes)))


@pytest.mark.parametrize('mixing', [False, True])
def test_packed_sparse_matches_batched_outputs_and_gradients(monkeypatch, mixing):
    import numpy as np
    from mimir.model.flash_attention_prefixlm_v2 import compute_aux_seq_tensors_scalars
    from mimir.model.common import wrap_tensor
    def attention(q, k, v, causal, **kw):
        if q.ndim == 4:
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(-3, -2), k.transpose(-3, -2),
                      v.transpose(-3, -2), is_causal=causal).transpose(-3, -2)
        bounds = kw['cu_seqlens'][:int(kw['numseqs'])+1].tolist()
        result = torch.zeros_like(q)
        for a, b in zip(bounds, bounds[1:]):
            result[a:b] = torch.nn.functional.scaled_dot_product_attention(q[a:b].transpose(0, 1),
                      k[a:b].transpose(0, 1), v[a:b].transpose(0, 1), is_causal=causal).transpose(0, 1)
        return result
    monkeypatch.setattr('mimir.model.layers.flash_attn_varlen_prefixlm', attention)
    model = make(mos_sparse=True, mos_num_experts=3, h_branch_l_mixing=mixing)
    ref = copy.deepcopy(model)
    tensors, scalars = compute_aux_seq_tensors_scalars(np.array([1, 1], dtype=np.int32),
                                                       np.array([3, 3], dtype=np.int32), 8)
    info = {k: torch.from_numpy(v) for k, v in tensors.items()} | {k: wrap_tensor(torch.tensor(v)) for k, v in scalars.items()}
    x = torch.randn(2, 4, 16)
    a = model(None, x.reshape(8, 16), position_ids=torch.arange(4).repeat(2), bp_steps=8, **info)[1]
    b = ref(None, x, position_ids=torch.arange(4), bp_steps=8)[1].reshape(8, 16)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    a.square().sum().backward(); b.square().sum().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=1e-4, msg=name)
    assert all(p.grad is not None for p in model.L_level.stack_experts[1].parameters())


def test_optimizer_ema_resume_and_compile(tmp_path):
    from mimir.model.adam_atan2 import AdamATan2
    torch.manual_seed(42)
    model = make(h_branch_l_mixing=True)
    opt = AdamATan2(model.parameters(), lr=.001, ema=.9, ema_dtype=torch.float32)
    x = torch.randn(2, 4, 16)
    def step(m, optimizer):
        optimizer.zero_grad(set_to_none=True)
        out = m(None, x, bp_steps=8, position_ids=torch.arange(4))[1]
        out.square().sum().backward(); optimizer.step()
        return out
    step(model, opt)
    path = tmp_path/'state.pt'
    torch.save({'model': model.state_dict(), 'optim': opt.state_dict()}, path)
    clone = make(h_branch_l_mixing=True)
    resumed = AdamATan2(clone.parameters(), lr=.001, ema=.9, ema_dtype=torch.float32)
    saved = torch.load(path, weights_only=False)
    clone.load_state_dict(saved['model']); resumed.load_state_dict(saved['optim'])
    compiled = torch.compile(clone, backend='aot_eager', fullgraph=True, dynamic=False)
    torch.testing.assert_close(step(model, opt), step(compiled, resumed), atol=1e-5, rtol=1e-4)
    for p, q in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
        for key in opt.state[p]:
            torch.testing.assert_close(opt.state[p][key], resumed.state[q][key], atol=2e-5, rtol=1e-4)


def test_persistence_is_real_and_readout_does_not_reset_streams():
    model = make()
    snapshot = []
    def hook(_, args, kwargs, output):
        snapshot.append((args[0].detach().clone(), output[0].detach().clone()))
    handle = model.H_level.core.register_forward_hook(hook, with_kwargs=True)
    x = torch.randn(2, 4, 16)
    model(None, x, bp_steps=8, position_ids=torch.arange(4))
    handle.remove()
    # Shared L injection adds the same vector to both branches, so the branch
    # difference entering H call two is exactly the difference leaving call one.
    first_out, second_in = snapshot[0][1], snapshot[1][0]
    torch.testing.assert_close(second_in[..., 0, :]-second_in[..., 1, :],
                               first_out[..., 0, :]-first_out[..., 1, :], atol=1e-6, rtol=1e-4)
    assert (second_in[..., 0, :]-second_in[..., 1, :]).abs().max() > 1e-4


def test_old_checkpoint_defaults_and_changed_architecture():
    from mimir.training.checkpoint_io import same_architecture
    old = dict(name='mimir2', residual_branches=2)
    assert same_architecture(old, old | dict(h_residual_persistence=False, h_branch_l_mixing=False))
    assert not same_architecture(old, old | dict(h_residual_persistence=True))
    assert not same_architecture(old, old | dict(h_branch_l_mixing=True))


def test_dense_upcycling_and_diagnostics(tmp_path):
    from mimir.model.expert_mixtures import upcycle_dense_weights
    from mimir.model.recurrence import RecurrentBackbone
    from mimir.model.stability_diagnostics import StabilityDiagnostics
    baseline = RecurrentBackbone(config())
    model = make(h_branch_l_mixing=True)
    upcycle_dense_weights(model, baseline.state_dict())
    diag = StabilityDiagnostics(step=1, rank=0, output_path=str(tmp_path/'diagnostics.json'))
    model(None, torch.randn(2, 4, 16), position_ids=torch.arange(4), bp_steps=8,
          stability_diagnostics=diag)[1].sum().backward()


@pytest.mark.parametrize('flag', ['h_residual_persistence', 'h_branch_l_mixing'])
def test_inference_rejects_unsupported_flags(tmp_path, monkeypatch, flag):
    import importlib
    import json
    import yaml
    convert = importlib.import_module('mimir.export.convert')
    from mimir.serving.server import command, ServerSpec
    monkeypatch.setattr(convert, 'validate', lambda _: {})
    (tmp_path/'all_config.yaml').write_text(yaml.safe_dump({'arch': {'name': 'mimir2', flag: True}}))
    (tmp_path/'train_metadata.yaml').write_text('{}')
    with pytest.raises(ValueError, match='require inference fork support'):
        convert.export(tmp_path, tmp_path/'output', tmp_path/'tokenizer')
    (tmp_path/'config.json').write_text(json.dumps({'model_type': 'mimir2', flag: True}))
    with pytest.raises(ValueError, match='require inference fork support'):
        command(ServerSpec(str(tmp_path), '0'))


def _ddp_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    import mimir.model.layers as layers
    torch.set_num_threads(1)
    layers.flash_attn_varlen_prefixlm = lambda q, k, v, causal, **kw: torch.nn.functional.scaled_dot_product_attention(
        q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2), is_causal=causal).transpose(-3, -2)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        torch.manual_seed(81)
        model = make(mos_sparse=True, mos_top_k=1, h_branch_l_mixing=True)
        reference = copy.deepcopy(model)
        wrapped = DistributedDataParallel(model)
        torch.manual_seed(12)
        x = torch.randn(4, 4, 16)
        # Multiple backwards also exercises DDP's unused-parameter detection.
        for bp in (2, 8):
            model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
            wrapped(None, x[rank*2:(rank+1)*2], bp_steps=bp, position_ids=torch.arange(4))[1].square().mean().backward()
            reference(None, x, bp_steps=bp, position_ids=torch.arange(4))[1].square().mean().backward()
            for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=2e-4, msg=name)
    finally:
        dist.destroy_process_group()


def test_ddp_sparse_persistence_matches_global_batch(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path/'rendezvous'),), nprocs=2, join=True)


@pytest.mark.parametrize('levels', ['H', 'both'])
def test_persistence_without_mos(levels):
    model = Mimir2(config(h_residual_persistence=True, residual_branches=2, residual_levels=levels))
    model.compute_train_extra_args(SimpleNamespace(step=1))
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = model(None, x, bp_steps=8, position_ids=torch.arange(4))[1]
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_prefixlm_prompt_prefill_and_decode(monkeypatch):
    def attention(q, k, v, causal, **kw):
        positions = torch.arange(q.shape[-3])
        mask = (positions[:, None] >= positions[None, :]) | ((positions[:, None] < 2) & (positions[None, :] < 2))
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(-3, -2), k.transpose(-3, -2),
                  v.transpose(-3, -2), attn_mask=mask).transpose(-3, -2)
    monkeypatch.setattr('mimir.model.layers.flash_attn_varlen_prefixlm', attention)
    model = make(h_branch_l_mixing=True, mos_sparse=True, mos_sparse_routing='prefix_mean', attn_type='prefixlm').eval()
    model.H_level.l_branch_mixer.logits.data.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
    x = torch.randn(1, 5, 16)
    cache = model.create_cache(max_batch_size=1, max_seq_len=16, device='cpu', dtype=torch.float32)
    info = dict(prefix_lens=torch.tensor([2]))
    with torch.no_grad():
        full = model(None, x, position_ids=torch.arange(5), **info)[1]
        parts = [model(None, x[:, :2], position_ids=torch.arange(2), cache=cache,
                       cache_lengths=torch.tensor([0]), **info)[1]]
        parts += [model(None, x[:, i:i+1], position_ids=torch.tensor([i]), cache=cache,
                        cache_lengths=torch.tensor([i]), **info)[1] for i in range(2, 5)]
        torch.testing.assert_close(full, torch.cat(parts, 1), atol=1e-5, rtol=1e-4)


def test_copied_experts_can_break_symmetry():
    torch.manual_seed(3)
    model = Mimir2(config(residual_branches=2, residual_levels='H', h_residual_persistence=True,
                          h_branch_l_mixing=True, mos_num_experts=2, mos_top_k=2))
    model.compute_train_extra_args(SimpleNamespace(step=1))
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    x, target = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
    first = model.L_level.core.layers[0].mlp.down_proj.weight
    second = model.L_level.stack_experts[0].layers[0].mlp.down_proj.weight
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for _ in range(4):
        opt.zero_grad()
        (model(None, x, bp_steps=8, position_ids=torch.arange(4))[1]-target).square().mean().backward()
        opt.step()
    assert (first-second).abs().max() > 1e-5
