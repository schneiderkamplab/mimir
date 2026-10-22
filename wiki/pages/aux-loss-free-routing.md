---
type: Technical Reference
title: Auxiliary-loss-free routing balance
description: Optional step-level expert bias balancing for MoE and MoS.
status: stable
confidence: high
last_updated: 2026-10-22
---
# Optional balancing

Enable independently with `arch.moe_aux_loss_free=true` and/or
`arch.mos_aux_loss_free=true`; the corresponding mixture must have at least two
experts. `arch.routing_bias_update_rate=0.001` controls both, and zero freezes
updates while retaining the learned selection bias. All defaults remain off.

The rule follows [DeepSeek-V3 section 2.1.2](https://arxiv.org/html/2412.19437v2):
use sigmoid affinities, select top-k by affinity plus a per-expert correction,
and normalize the original selected affinities for output weights. At each
successful optimizer step, decrease the correction by the update rate for
above-average load and increase it for below-average load. Equal loads do not
change. No auxiliary objective or sequence-level balancing loss is added.
This is the balancing mechanism, not a complete DeepSeek-V3 architecture.
Mimir's optional top-1 straight-through surrogate remains available.

Metadata and existing logit biases retain their pre-affinity role. The new
selection bias is separate and never changes the output mixing weights for a
fixed expert selection. Disabled routers retain their original softmax path,
parameter names and state dict. Enabling balancing changes the affinity function
to sigmoid; compare against an unbalanced run with that distinction in mind.

# Counting and optimizer boundaries

Counts cover selected routes across all real forward invocations, including
no-grad recurrent warmup, and all gradient-accumulation microbatches. Activation
checkpoint backward replay does not add counts. Dense MoE/MoS count token-level
selections, even though dense execution evaluates every expert; sparse MoS counts
whole-sequence selections once per routing decision, not sequence token lengths.
Nested MoE routers balance independently within their respective stack experts.
No token dropping or added sequence/prompt lookahead is introduced.

Counts are ephemeral step-local int64 tensors outside model buffers, so DDP's
buffer broadcasts cannot overwrite a previous microbatch's counts. All ranks
sum counts once per router after the optimizer succeeds, including ranks with
zero local assignments. Skipped or disabled optimizer steps discard counts.
Forward-only inference and validation do not update biases. Collection uses an
eager boundary inside otherwise compiled execution; it is a correctness-first
implementation and its graph-break overhead has not been GPU-benchmarked.

Selection corrections are persistent FP32 buffers, excluded from optimizer and
EMA parameters. Checkpoints retain them exactly; EMA export uses the current
controller correction with EMA weights. No partially accumulated counts need
saving because checkpoints occur at optimizer-step boundaries. Dense upcycling
initializes the new correction to zero; exact continuation requires matching
architecture flags and update rate.

# Export and qualification

The captured Transformers and vLLM patches carry the flags and frozen bias
buffers through strict conversion and loading. vLLM uses the same Transformers
router implementation; its compiled inference has no training-state updates.
Only dependency checkouts inside Mimir were changed. Updated cumulative patches
and their hashes are in provenance/forks.json.

CPU tests cover selection-only bias, update direction, GAS, skipped steps,
checkpoint recomputation with compilation, idle distributed ranks, optimizer/EMA
and bias resume parity, FP32 buffer preservation, native/Transformers causal and
PrefixLM output parity, cached decode, and compiled inference router parity.
Full GPU training/FSDP and vLLM execution remain pending available resources.

The [XXS campaign](xxs-dfm15-ablation.md) adds one baseline Muon run and two
balanced runs using the existing capacity-matched MoE and MoS configurations.

Verification evidence: [routing-balance qualification](../../qualification/routing-balance.json).
The full CPU suite passed 309 tests and 28 subtests; installed-package targeted
checks passed 17 tests and 12 subtests. Older checkpoints lacking the new flags
are treated as balancing disabled with the default update rate; changing enabled
flags or update rate still rejects exact continuation.
