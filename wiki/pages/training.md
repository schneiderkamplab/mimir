---
type: Runbook
title: Training, checkpointing and recovery
description: Exact continuation, explicit initialization and suspension boundaries.
status: draft
confidence: high
last_updated: 2026-10-20
---
# Training and checkpoints

Checkpoint schema 1 uses one immutable directory per tag. Each rank writes its
RNG state; model/optimizer state, EMA slots, configuration, corpus metadata,
recurrence buffers and sample cursor are included. The manifest records total
schedule steps, world size, file sizes and SHA-256 hashes. Publication fsyncs
files and atomically renames a sibling partial directory. Interrupted partials
are never accepted by checkpoint waits or exact resumes. Older ephemeral
directories are removed only after the replacement validates.

Exact continuation validates the contract and original optimizer/configuration,
corpus metadata, schedule length and rank layout. Source code hashes are only
provenance. Python, NumPy and Torch are seeded per rank; DataLoader owns a separate
generator so rebuilding an iterator cannot advance restored training RNG state. Changing world size is rejected for exact RNG continuation; existing
row-cursor helpers do not establish bitwise continuation across world-size
changes. Corpus metadata equality does not prove unchanged token contents;
production immutable corpus identity qualification remains outstanding.

`checkpoint import --mode weights` deliberately discards optimizer, EMA, RNG,
cursor and schedules and records that decision. Use `initialize_from` for this
artifact; optimizer/EMA then initialize from the loaded weights. Exact imports
require every necessary state component. Historical checkpoints missing RNG or
total schedule state are rejected, not relabeled exact. DCP and unsharded
weights are supported. Do not load checkpoints from untrusted pickle sources.

`train --suspendable` enables the portable disk controller. Specify a checkpoint
path and optional `suspend_after_steps`. One advisory lock owns recovery. A
runtime `suspend.request` in the control directory requests a boundary stop;
`hold` prevents restart until explicitly removed. The worker publishes a complete
checkpoint and exits with status 75; the controller validates and restarts from
that checkpoint. Unexpected training failures are surfaced, not endlessly retried.

The CUDA controller/launcher/shim protocol has CPU tests and is opt-in through
`suspension_backend=cuda suspension_settings=/path/to/settings.json` together with
`--suspendable`. Settings supply shim/NCCL libraries, Redis and cuda-checkpoint
executables; exactly one launcher/supervisor owns recovery. The default is the
qualified disk path. GPU execution of this new CLI integration and a distributed
disk-suspension controller remain unqualified.
Do not launch multiple recovery controllers or use the single-process disk
supervisor inside each torchrun rank. Use scheduler-managed whole-launcher
checkpoint/restart boundaries for distributed training until that integration
is qualified. This is an outstanding acceptance item, not a CUDA parity claim.

Export validates the complete checkpoint, explicitly selects EMA or live weights,
converts only Mimir2, preserves tokenizer-owned templates and checks serialization
parity. Missing named EMA state raises. Exports publish by atomic directory rename.
