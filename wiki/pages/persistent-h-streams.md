---
type: Technical Reference
title: Persistent H streams and branch-specific L mixing
description: Opt-in H stream lifetimes and learned per-branch mixtures of selected L experts.
status: stable
confidence: high
last_updated: 2026-10-21
---
# Configuration

Both new flags default to false, preserving existing state keys and numerical
behavior. The implementation is in the standalone Mimir package; the sibling
legacy HRM-Text checkout is unchanged.

```bash
mimir train +experiment=persistent_h data.path=<corpus> checkpoint_path=<output>
```

The preset chooses two H residual branches, single-stream L stacks, two L MoS
experts, top-k two, and enables both flags. Each flag can be disabled independently:

- `arch.h_residual_persistence=true` retains the H streams across depth iterations.
- `arch.h_branch_l_mixing=true` gives each H branch a different learned mixture
  of the selected L expert outputs. This also works without H persistence.

The existing `arch.moe_levels=H arch.moe_num_experts=4 arch.moe_top_k=2` can add
H MoE. Dense/sparse MoE and MoS remain separately configurable. Residual branches
on both H and L also work via `arch.residual_levels=both`. Invocation gates and
shared/separate MoS recurrent states remain supported.

Both flags require hierarchical recurrence and at least two H residual branches.
They reject `h_memory_branches>1`, because its additional recurrent states need a
separate integration design. Branch-specific mixing also requires MoS with at
least two experts and rejects `l_memory_branches>1`, whose readout would otherwise
be bypassed. Looped recurrence and absent H stacks are incompatible with these
options. Gate-only migration cannot enable them; use explicit weights
initialization/upcycling when changing architecture, not exact continuation.

# State flow

H begins with identical copies of input embeddings. The first L invocation
continues to see the original embedding as H context, preserving the existing
first-iteration convention. After each H update, its existing learned readout
and normalization supply a temporary single-vector context for L. That readout
does not replace the retained H streams.

Each L invocation still computes the ordinary router-weighted `z_L` for L's
own recurrence. Immediately before an H update, add either that shared `z_L`
to every H stream, or the branch-specific mixtures of the **last L invocation's**
expert outputs. H runs its existing attention/MLP layers on mixed reads and
writes updates into its streams. Final prediction uses the same learned H
readout and normalization. H attention/MLP invocation counts are unchanged.

Without persistence, H streams are initialized from the current `z_H` on every
H invocation, just as before. Branch-specific injection can still distinguish
their incoming L information. With persistence, streams initialize once per
forward and remain local to that forward; they never become cross-request or
cross-token residual-state caches. Native KV caches remain invocation-specific.

# Branch mixing

For branch b and expert e, let `w_e` be the existing MoS router weight and
`a_be` a learned logit. Each branch's L injection is:

```text
p_be = w_e * exp(a_be) / sum_j(w_j * exp(a_bj))
injection_b = sum_e(p_be * expert_output_e)
```

Only selected experts contribute. This changes neither top-k selection nor
expert dispatch counts. The implementation uses a numerically stable masked
normalization and preserves the selected-entry top-1 straight-through gradient
by retaining the router's forward-unit total mass in the backward path.
For top-k one, all branches receive the same selected expert: differing mixtures
require at least two active experts.

The mixer has `H_branches * L_experts` learned scalar logits, shared across
recurrent invocations. Router weights remain token-dependent for dense MoS and
sequence-dependent for sparse MoS. Initial logits mildly favor expert
`branch_index % expert_count` (a 0.1 logit advantage). This deterministic
asymmetry avoids locking copied experts into identical gradients. Zero mixer
logits recover the shared router mixture, useful for controlled comparisons.
This is a reweighting of MoS outputs, not Qwen's gated-residual equations.

The final L expert candidates must remain available until H injection. This
adds activation storage and small mixing work, particularly with many experts;
there are no extra H attention evaluations. Sparse dispatch retains dummy
zero-gradient connections for idle experts so DDP collective participation is
unchanged.

# Implementation and qualification

`BranchedTransformer` separates initialization, processing and readout, retaining
its ordinary `forward()` wrapper. The H stream path still enters the module
through `__call__`, keeping FSDP wrapper hooks around parameter access. New mixer
parameters belong to `H_level`, so module-specific LR assignment, Adam/Muon
auxiliary updates, EMA and checkpoint naming use the existing machinery.

Exact continuation compares both flags. Older checkpoints missing these flags
are interpreted as false; changing a flag is an architecture change, not an
exact resume. Persistent H activations are not checkpoint state: they are
recomputed inside each training forward, while parameters/optimizer/EMA and
configuration are saved normally.

CPU tests cover explicit unrolling with full/truncated gradients, independent
flags, packed sparse dispatch including idle experts, top-1 surrogate routing,
causal and PrefixLM cached decoding, activation checkpointing, dense static-shape
AOT compilation, optimizer/EMA roundtrip, request isolation, and two-rank Gloo
DDP. The regression suite uses `dynamic=False` compilation, matching training;
automatic symbolic integer specialization of recurrence counts is unsupported.

Native CPU execution is qualified. CUDA/FA4, distributed FSDP2 and compiled GPU
execution of these additions still need qualification. Export/serving explicitly
reject either new flag: the pinned Transformers/vLLM integrations do not yet
implement these semantics. The default architecture remains exportable.
