"""Opt-in multi-stream residuals for v2 pre-norm transformer stacks.

Read a normalized mixture, execute each sublayer once, and distribute its
update across streams. The caller can optionally retain streams across calls.
"""

import torch
from torch import nn

from mimir.model.transformer import Transformer, TransformerBlock


class ResidualConnection(nn.Module):
    def __init__(self, branches, hidden_size, preferred_branch, init_scale, input_dependent=False):
        super().__init__()
        self.read_logits = nn.Parameter(torch.zeros(branches, hidden_size))
        # Asymmetric writes break stream symmetry without changing their mean
        # update at initialization. No RNG is consumed by installing branches.
        write = torch.full((branches, hidden_size), -init_scale / branches)
        write[preferred_branch % branches] += init_scale
        self.write_logits = nn.Parameter(write)
        self.branches = branches
        if input_dependent:
            self.read_gain = nn.Parameter(torch.zeros(branches, hidden_size))
            self.write_gain = nn.Parameter(torch.zeros(branches, hidden_size))

    def weights(self, streams, alpha, kind):
        logits = getattr(self, f'{kind}_logits').float()
        if hasattr(self, f'{kind}_gain'):
            # Token-local, channel-wise modulation; no cross-sequence pooling.
            logits = logits + streams.float().tanh() * getattr(self, f'{kind}_gain').float()
        return (alpha.float() * logits).softmax(dim=-2)

    def read(self, streams, alpha):
        weights = self.weights(streams, alpha, 'read')
        return (streams.float() * weights).sum(dim=-2).to(streams.dtype)

    def write(self, streams, update, alpha):
        weights = self.branches * self.weights(streams, alpha, 'write')
        return (streams.float() + weights * update.float().unsqueeze(-2)).to(streams.dtype)


class BranchedTransformerBlock(TransformerBlock):
    def __init__(self, original, branches, hidden_size, layer_index, init_scale, input_dependent=False):
        # Preserve backbone names and module identities; do not reinitialize.
        nn.Module.__init__(self)
        self.attn = original.attn
        self.mlp = original.mlp
        self.norm = original.norm
        self.attn_residual = ResidualConnection(branches, hidden_size, 2 * layer_index, init_scale, input_dependent)
        self.mlp_residual = ResidualConnection(branches, hidden_size, 2 * layer_index + 1, init_scale, input_dependent)

    def forward(self, streams, residual_alpha, stability_diagnostics=None,
                stability_scope=None, **seq_info):
        routing_bias = seq_info.pop('routing_bias', None)
        if stability_diagnostics is not None and stability_scope is None:
            raise ValueError("stability_scope is required when diagnostics are enabled")
        attn_input = self.norm(self.attn_residual.read(streams, residual_alpha))
        update = self.attn(attn_input, **seq_info,
                           stability_diagnostics=stability_diagnostics,
                           stability_scope=f"{stability_scope}/attention" if stability_scope else None)
        streams = self.attn_residual.write(streams, update, residual_alpha)
        mlp_input = self.norm(self.mlp_residual.read(streams, residual_alpha))
        kwargs = {'routing_bias': routing_bias} if hasattr(self.mlp, 'router') else {}
        update = self.mlp(mlp_input, **kwargs, stability_diagnostics=stability_diagnostics,
                          stability_scope=f"{stability_scope}/mlp" if stability_scope else None)
        streams = self.mlp_residual.write(streams, update, residual_alpha)
        if stability_diagnostics is not None:
            stability_diagnostics.record_tensor(
                f"activation/{stability_scope}/residual_streams", streams, record_gradient=True)
        return streams


class BranchedTransformer(Transformer):
    def __init__(self, original, branches, init_scale=0.1, input_dependent=False):
        nn.Module.__init__(self)
        if branches < 2:
            raise ValueError("Use the original Transformer for a single residual branch")
        self.branches = branches
        self.head_hint = original.head_hint
        self.norm_f = original.norm_f
        self.create_cache = original.create_cache
        if hasattr(original, 'rotary_emb'):
            self.rotary_emb = original.rotary_emb
        hidden_size = original.head_hint['in']['dim']
        self.layers = nn.ModuleList(
            BranchedTransformerBlock(block, branches, hidden_size, index, init_scale, input_dependent)
            for index, block in enumerate(original.layers)
        )
        self.residual_readout = nn.Parameter(torch.zeros(branches, hidden_size))
        self.register_buffer('residual_alpha', torch.tensor(0.0))

    def initialize_streams(self, x):
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.branches, x.shape[-1])

    def process_streams(self, streams, cache=None, stability_diagnostics=None,
                        stability_scope=None, **seq_info):
        if stability_diagnostics is not None and stability_scope is None:
            raise ValueError("stability_scope is required when diagnostics are enabled")
        seq_info['cos_sin'] = self.rotary_emb(seq_info.pop('position_ids', None)) if hasattr(self, 'rotary_emb') else None
        for index, layer in enumerate(self.layers):
            streams = layer(streams, residual_alpha=self.residual_alpha, **seq_info,
                            cache=cache[index] if cache is not None else None,
                            stability_diagnostics=stability_diagnostics,
                            stability_scope=f"{stability_scope}/layer_{index:03d}" if stability_scope else None)
        return streams

    def readout(self, streams, stability_diagnostics=None, stability_scope=None):
        weights = (self.residual_alpha.float() * self.residual_readout.float()).softmax(dim=0)
        output = self.norm_f((streams.float() * weights).sum(dim=-2).to(streams.dtype))
        if stability_diagnostics is not None:
            stability_diagnostics.record_tensor(
                f"activation/{stability_scope}/final_norm", output, record_gradient=True)
        return output

    def forward(self, x, cache=None, stability_diagnostics=None, stability_scope=None,
                stream_input=False, return_streams=False, **seq_info):
        # Keep parameter access inside __call__ for recurrent-level FSDP hooks.
        streams = x if stream_input else self.initialize_streams(x)
        streams = self.process_streams(streams, cache=cache, stability_diagnostics=stability_diagnostics,
                                       stability_scope=stability_scope, **seq_info)
        output = self.readout(streams, stability_diagnostics, stability_scope)
        return (streams, output) if return_streams else output


class BranchExpertMixer(nn.Module):
    """Reweight the selected MoS experts independently for each H branch.

    Zero logits reproduce the router weights. Routing support is unchanged;
    with top-1 all branches necessarily receive the same expert output.
    """
    def __init__(self, branches, experts, init_scale=0.1):
        super().__init__()
        # Copied MoS experts start identical. A small deterministic preference
        # prevents the new mixer from locking them into symmetric gradients.
        logits = torch.full((branches, experts), -init_scale / experts)
        logits[torch.arange(branches), torch.arange(branches) % experts] += init_scale
        self.logits = nn.Parameter(logits)

    def forward(self, expert_outputs):
        # Entries are full-token expert candidates and scalar routing weights.
        values = torch.stack([value for value, _ in expert_outputs], dim=-2)
        weights = torch.cat([weight.expand(*values.shape[:-2], 1)
                             for _, weight in expert_outputs], dim=-1).float()
        # Stable masked softmax of log(router weight) + branch preference.
        active = weights > 0
        logits = self.logits.float().expand(*weights.shape[:-1], *self.logits.shape)
        shift = logits.masked_fill(~active.unsqueeze(-2), -torch.inf).amax(-1, keepdim=True)
        preference = (logits - shift).masked_fill(~active.unsqueeze(-2), -torch.inf).exp()
        mixture = weights.unsqueeze(-2) * preference
        mixture = mixture / mixture.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        # Preserve the router's selected-entry straight-through derivative for
        # top-1: forward routing mass is one, but its surrogate gradient is not.
        mixture = mixture * weights.sum(-1)[..., None, None]
        return torch.einsum('...be,...ed->...bd', mixture, values.float()).to(values.dtype)
