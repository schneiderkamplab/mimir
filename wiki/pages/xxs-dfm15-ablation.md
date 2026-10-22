---
type: Experiment
title: XXS DFM15 capacity ablation preparation
description: Four held 20K-step runs using native Mimir scheduling and monitoring.
status: draft
confidence: high
last_updated: 2026-10-22
---
# Prepared campaign

See the [recipe and operational notes](../../campaigns/xxs-dfm15-20k/README.md)
and [parameter counts](../../campaigns/xxs-dfm15-20k/parameters.json).
The generator is [prepare_xxs_ablation.py](../../tools/prepare_xxs_ablation.py).

On 2026-10-22, prepared `.runtime/campaigns/xxs-dfm15-20k`: four XXS
20K-step runs, final EMA evaluation, 152 held jobs, zero attempts, no GPU
assignment. The live SQLite path is in its preparation.json; the queue is on
local `/tmp` (overlay), because the repository filesystem is Weka. The native
monitor renders the queue without executing jobs. A recoverable online backup
is stored with the prepared artifacts. No training or evaluation was launched.

The selected corpus's own status is `validated_staged_not_promoted`.
Metadata SHA256 is
`f041613e99a33dd5d4ed4567901e8347d0d6d788077571f0129894959d8a9b27`.
Tokenizer SHA256 is
`12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`.
Only data are read from sibling projects. Tokenizer files and the corpus-owned
chat template are copied into the prepared output; all executable components
come from Mimir and its installed dependencies.

Capacity excludes input token embeddings and LM head, but includes n-gram
memory tables. Engram is exactly 2x baseline; MoE is 1.76% below the doubling
target and MoS 0.007% above. Dense mixture execution retains defaults and does
not equalize FLOPs. New persistent H/mixing features remain disabled.

Native JSON plan entries now accept boolean `held`; Store.add inserts that hold
in the same transaction as the job. Tests cover held-plan import even when a
runner is requested, unique task identities across suites, capacity counts and
terminal evaluation handoffs. GPU memory, FA4, compiled inference and full
benchmark context lengths still require qualification before release. Bounded
corpus validation does not replace a full token/index audit.

Verification: rebuilt/reinstalled the Mimir wheel in the isolated environment;
full CPU suite passed (290 tests, 25 subtests, 28 warnings). OKF validation and
`git diff --check` passed. See [preparation evidence](../../qualification/xxs-ablation-preparation.json).

## Scheduled after user authorization, 2026-10-22

The preparation-only hold above is superseded by the user's request to start
serially when GPUs become free. All 152 jobs were bound and released with zero
attempts. Training uses GPUs 0–7 sequentially in default, Engram, MoE, MoS order.
Evaluation tasks are assigned round-robin across eight separate GPU workers;
each server uses TP=1 and ports 18100–18107. No tensor parallel evaluation.
The original held plan remains available; `scheduled-plan.json` records the
activated configuration and `queue.scheduled.backup.sqlite` its queue snapshot.

A detached `python -m mimir.scheduling.idle LOCAL_DB --gpus 0,1,2,3,4,5,6,7`
coordinator owns the native runner lock. It checks every 30 seconds and requires
three consecutive clear checks. Existing compute processes block launch even
at zero utilization; unexplained memory over 1 GiB or utilization over 5% also
blocks launch. Probe failures fail closed. This is cooperative availability
checking, not an exclusive reservation against unrelated launchers.

The coordinator runs non-evaluation jobs serially. Separate evaluation processes
claim only their assigned GPU's tasks transactionally, reuse compatible servers,
and join before training can resume. User holds are never automatically cleared
by the runner. In-flight evaluations can fail without stranding the next run;
merge success remains outside training dependencies. Unexpected worker crashes
stop the coordinator rather than risking overlapping GPU work; restart uses the
existing ownership-aware recovery path. The detached runner survives shell exit,
not host reboot. Live SQLite and updated backups remain on local disk.

The monitor displays the GPU wait reason and timestamp. Actual queue, process
identity and logs are in the campaign's preparation.json and runner.json.
Initial verification found the coordinator alive and waiting on eight unrelated
vLLM processes, with all jobs still pending and zero attempts. No processes were
signalled. The isolated Mimir environment now has pinned FA4 revision
`4178915405522cb161ece80005e882c7465693b7`; its CPU import succeeded.
Targeted CPU checks passed (13 tests including preparation; the final scheduler
subset passed 11 tests). GPU execution remains pending availability.

## Muon and balancing extension

The user subsequently requested one baseline Muon run (explicitly clarified),
plus capacity-matched MoE and MoS with DeepSeek-V3-style bias balancing. The final
order is default → Engram → MoE → MoS → Muon → balanced MoE → balanced MoS.
All seven use the same 20K-step DFM15 budget. The extension adds 114 jobs; the
live queue now has 266 released jobs, zero attempts at verification, and all
33 evaluation tasks per new run assigned across eight TP=1 workers.
Resolved configs, extension-plan.json, scheduled-plan.json, updated capacity
counts and queue.released.backup.sqlite are in the campaign directory.

The [balancing implementation](aux-loss-free-routing.md) and patched inference
forks passed 309 full CPU tests and 28 subtests, then 17 installed-package tests
and 12 subtests. GPU qualification remains pending. Queue refresh exposed an
existing overwrite limitation in Store.backup: the coordinator now stages and
atomically replaces each verified snapshot, retaining the old one and warning
on failure instead of stranding training. Twelve targeted scheduler tests passed,
including repeated backups and failure retention. Only the verified owned idle
coordinator was restarted to load this fix, with no active attempts or children;
all pre-existing holds were preserved. Its new identity is in runner.json.
The restarted coordinator was verified alive, polling, and waiting on unrelated
evaluation GPU users. No unrelated processes were signalled.
