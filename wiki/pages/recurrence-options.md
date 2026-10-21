---
type: Technical Reference
title: Expert combinations and looped recurrence
description: Independent expert and residual selection, and optional single-stream recurrence.
status: stable
confidence: high
last_updated: 2026-10-20
---
# Expert combinations

H-only MoE and L MoS already compose in Mimir2: select `arch.moe_levels=H`,
`arch.moe_num_experts=4`, `arch.moe_top_k=2`, `arch.mos_num_experts=4`, and
`arch.mos_top_k=2`. MoS wraps complete L stacks, including their residual
branches. MoE replaces H feed-forward layers. Dense/sparse execution remains
independently selectable with `arch.moe_sparse` and `arch.mos_sparse`.

Use `mimir train +experiment=h_moe_l_mos` plus data/output overrides for the
packaged example. `arch.residual_branches=2 arch.residual_levels=both` enables
branches in both stacks; override `arch.residual_levels=H` for H only, `L` for
L only, or `arch.residual_branches=1` for neither. This reuses existing selection
fields rather than introducing competing controls. Branches require pre-norm.

# Recurrence

`arch.recurrence_mode=hierarchical` remains the default with unchanged v1
behavior. `arch.recurrence_mode=looped` carries a single stream initialized from
input embeddings through `[L repeated L_cycles times, then H]`, repeating this
block `H_cycles` times. Weights are shared across repetitions, with separate KV
cache slots per invocation. There are no cross-state additive injections.
Each nonempty stack retains its existing final normalization.

Set `+arch.H_override.n_layers=0` to remove H: no H parameters, normalization,
attention, or cache entries execute. Total L invocations are still
`H_cycles * L_cycles`. `half_layers=false` makes `n_layers` the actual L stack
depth; otherwise the existing halving convention still applies.
`mimir train +experiment=looped` selects this L-only layout with four repeats.
Override `arch.H_override.n_layers=2` with this preset to retain a two-layer H
stack. The expert preset and recurrence can also be combined using explicit
architecture overrides. H routing/residual settings have no effect when H is
absent; choose L MoE if experts are wanted inside an L-only loop.

Looped mode supports dense/sparse mixtures and in-stack residual branches. It
requires shared MoS states and disables invocation gates and recurrent memory
branches: those encode multiple recurrent states rather than this single-stream
pattern. Invalid combinations fail configuration validation. N-gram input
augmentation remains available. Zero-layer H is restricted to looped mode.

For a nonempty H, the existing H-prioritized truncated-backprop allocation and
automatic learning rates remain in effect. Without H, all `bp_steps` are
allocated to the final L invocations and automatic L learning rate is divided
by their count; no H rate is inferred. Set `bp_min_steps=bp_max_steps` to the
total invocation count for full backpropagation with no depth warmup.

Native CPU tests cover explicit-unroll output/gradient parity, truncated L-only
backpropagation, automatic LR, routing/branch combinations, state roundtrip,
and cached decode. GPU/distributed qualification of this new mode is pending.
Looped export and serving explicitly fail: the pinned Transformers/vLLM forks
implement hierarchical recurrence only. Native checkpoints retain the new
configuration; do not reinterpret looped weights as hierarchical exports.

For hierarchical recurrence, [persistent H streams](persistent-h-streams.md)
optionally preserve H residual branches and give them different L expert mixtures.
