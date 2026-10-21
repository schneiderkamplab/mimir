---
type: Technical Reference
title: Package architecture
description: Component boundaries and retained model behavior.
status: draft
confidence: high
last_updated: 2026-10-20
---
# Architecture

`src/mimir/model` contains Mimir2 and its recurrence, attention, cache, routing,
branch and optimizer components. `Mimir2` and `Mimir2Config` are the public model
exports. Recurrence retains the baseline's state names and numerical path so
loaded weights, losses and gradients can be compared exactly. This internal
base class is not a second supported architecture.

Training is divided into typed configuration, state, setup, distributed wrapping,
steps/GAS, schedules, runtime, orchestration and checkpoint I/O. Hydra composes
one model group plus packaged sizes. Unknown architecture names and typed config
fields fail validation; no string-driven class loader remains. Typer handles
administration and passes training overrides to Hydra.

DDP, full/hybrid FSDP2, precision controls, FA4 dispatch, compilation, activation
checkpointing, gradient clipping/skipping, recurrence schedules, automatic and
module LRs, cooldown/rewarm, EMA, sparse/dense MoE/MoS, gates, residual and recurrent
memories, contextual n-grams, routing bias, specialists/upcycling and Muon were
ported from the working tree. These are correctness-preserving ports, not new
performance claims. Sparse MoS still requires eager vLLM. Native and serving
validation retain their unsupported-combination errors.

Corpus access reads the existing tokens.npy and epoch index vectors. It checks
integer shapes, aligned row counts, nonnegative lengths and in-bounds ranges.
No downloader, converter, auditor, generator or sampling pipeline is included.
Future MTC release identities are reserved; this implementation does not invent
an MTC release or cryptographic corpus registry.

`provenance/native-source.json` captures the source revision and file hashes,
including uncommitted architecture updates. `provenance/forks.json` plus patches
reproduce the Transformers/vLLM integration without editing sibling checkouts.
See [validation](validation.md) for the distinction between ported behavior and
qualified execution paths.

Opt-in single-stream recurrence and zero-layer H are documented in
[recurrence options](recurrence-options.md), with independent H MoE / L MoS
and residual-branch configuration examples. Default recurrence is unchanged.

[Persistent H streams](persistent-h-streams.md) add opt-in depth persistence and
branch-specific L expert mixtures. These extensions are native-only until the
inference forks gain matching implementations.
