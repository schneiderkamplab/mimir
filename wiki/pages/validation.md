---
type: Validation Record
title: Initial qualification and remaining acceptance gates
description: Verified CPU evidence and explicit limits of the initial implementation.
status: draft
confidence: high
last_updated: 2026-10-20
---
# Validation

Final installed-wheel CPU suite: **213 tests and 25 subtests passed**, with
28 dependency/checkpoint warnings, in 76.03 seconds. This includes vLLM CPU
contracts, direct Inspect, DDP/GAS, scheduler, importer and corpus/model tests.
The machine-readable record is `qualification/cpu.json`.

The isolated four-step CPU campaign passed train → versioned checkpoint → EMA
Transformers export → local evaluation → resume, plus opt-in disk suspension.
Resumed/suspended weights, optimizer, EMA, RNG and logged metrics matched
uninterrupted training exactly. A final audit initially found iterator creation
advanced restored Torch RNG and fresh Python/NumPy seeds differed. Per-rank
seeding and a dedicated DataLoader generator fixed the discrepancy; the campaign
now asserts all saved RNG states, not only losses and optimizer tensors. `tools/qualify_cpu.py` reproduces it in a fresh output directory.
The receipt is preserved in `qualification/campaign.json`.

The model test suite covers baseline logits/losses/gradients and optimizer/EMA
behavior, gates, dense/sparse mixtures, recurrence caches, residual/recurrent
memory, contextual n-grams, metadata routing, specialist/upcycling state,
compilation/checkpointing and two-rank CPU FSDP. Transformers tests cover fused
and separate projection conversion, cache/beam behavior and PrefixLM. vLLM CPU
contracts cover configuration/projection/routing/cache preparation and n-grams;
they do not instantiate a B200 engine.

Both dependency patches applied cleanly to independent copies of the recorded
base revisions. Tests use the copied/patched Transformers source inside this
repository; no native import depends on HRM-Text. The initial development venv shared
read-only binary dependencies with an existing environment; it did not modify
that environment. Superseded shared-environment-only limitation (2026-10-20): a
fully isolated `uv sync --frozen` environment installed the training, serving,
evaluation and test extras. Both fork patches were independently built there,
using the pinned vLLM precompiled wheel. `uv pip check` passed. The final suite
and campaign use an installed Mimir wheel in that isolated environment.

Dependency command finding: `uv pip install -e '.[evaluation]'` without training
constraints temporarily selected a newer Torch inside the disposable development
venv. This was undone there; no existing environment changed. Use the frozen
lock with all intended extras, as documented in README, rather than an unlocked
partial-extra installation. The final isolated environment uses Torch 2.11.0+cu130.

## Outstanding acceptance gates

- B200 FA4 numerical/performance checks, real compiled vLLM execution, CUDA graph
  replay, resource-pool reuse and multi-node training/evaluation.
- GPU qualification of the explicit CUDA-suspension CLI integration, whole-launcher distributed disk
  recovery, production-size failure injections and numerical qualification.
- Full latest-comparison replay, complete immutable dataset/sample identity,
  historical model-specific template reconciliation, actual EuroEval/BERTScore
  runs and independently provisioned judges.
- Real-result qualification of nonlinear metric reducers. The scheduler now
  requires declared complete sample/pass sets and supports explicit classification
  and pass@k reducers; raw Inspect label extraction still needs per-task mapping.
- Exhaustive legacy exact-import fixtures. Missing legacy RNG/schedule state is
  correctly rejected; no general exact-continuation claim is made for it.
- FA4 build/runtime qualification and dependency/license release audit.

No GPUs were allocated, no production migration/publication occurred, and no
sibling source/environment or ongoing job was modified. Remaining gates must
not be inferred passed from the CPU campaign.

## Lock tool qualification

The machine's pre-existing uv 0.8.4 repeatedly reported a stale universal lock
immediately after rewriting it, including a CUDA dependency platform-marker
change. No global tool was upgraded. An isolated uv 0.13.3 regenerated and
successfully checked the lock. The project pins that tool version and the
qualified Linux environment. Frozen installation and `uv pip check` passed;
non-Linux binary stacks are not claimed supported by this initial lock.

## Single-stream recurrence qualification (2026-10-20)

The rebuilt package passed 252 CPU tests and 25 subtests (28 warnings), including
39 added recurrence/combination cases. An isolated two-step L-only training run
with `+experiment=looped lr_auto=true` produced a complete step-2 checkpoint.
Hydra experiment presets composed and validated from the installed wheel.
A final logging-only correction omits H LR metrics when H has no layers;
the 73 recurrence and module-learning-rate tests were rerun for that correction.
See [recurrence options](recurrence-options.md) and the receipt under
`qualification/recurrence.json`. New GPU/distributed recurrence qualification
and looped Transformers/vLLM integration remain outstanding.


## Persistent H and branch-specific injection (2026-10-21)

The rebuilt wheel passed 288 CPU tests and 25 subtests (28 warnings), including
36 new tests. The new coverage includes two-rank Gloo DDP, dense static-shape AOT
compilation, activation checkpointing, packed sparse MoS, PrefixLM cache parity,
and both full and truncated backward paths. A four-step CPU campaign with H
MoE, L MoS, both new flags and automatic LR matched uninterrupted training
exactly after saving at step two and resuming: model, optimizer, EMA, RNG, losses
and LR metrics all matched. The installed Hydra preset composed successfully.

See `qualification/persistent-h.json` and
[configuration and semantics](persistent-h-streams.md). The initial full-suite
compile test exposed unsupported automatic symbolic specialization of recurrence
integers; the final test explicitly uses `dynamic=False`, as the training entry
point already does. GPU/FA4/FSDP2 qualification remains pending; export and
serving reject the two flags until the dependency forks implement them.
