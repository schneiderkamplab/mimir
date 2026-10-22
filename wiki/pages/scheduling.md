---
type: Runbook
title: SQLite execution and training continuity
description: Transactional claims, recovery, server ownership and terminal handoffs.
status: draft
confidence: high
last_updated: 2026-10-20
---
# Scheduling

Use SQLite on local disk. WAL plus immediate transactions serialize claims.
Each attempt has an immutable configuration snapshot, unique fencing token,
heartbeat lease and resource reservations. Edits affect pending jobs only.
Explicit holds block claims. SQLite backup uses its online backup API and checks
integrity. The CLI execution owner holds one advisory runner lock; it can be
restarted without stale lock files blocking it. Concurrent database claim tests
exercise independent connections, but multi-host SQLite execution is not qualified.

Administrative options include `--hold-job`, `--release-job`, `--edit-job` with
`--config-file`, and `--backup`. These use short transactions, not persistent
administrative locks.

Plans are JSON objects with a `jobs` list. Each job has `id`, `kind`, `config`,
optional `max_attempts`, boolean `held` (default false), and `depends` entries with `id` and `policy` (`success`
or `terminal`). Resource names such as `gpu:0` and `judge:0` are independent.
Training, checkpoint waits, export, inference, scoring, merge, publication and
average kinds are represented. Generic process jobs use explicit argv lists;
Inspect/EuroEval and numerical merges have direct executors.

**Training successors must depend on terminal inference attempts.** Merge and
publication success must never be their prerequisites. The store rejects such
training dependencies when plans are added. Failed success-dependent
children become blocked, while terminal handoffs remain claimable. Tests verify
failed evaluation plus blocked merge still start training successfully.

A heartbeat renews active leases. Recovery checks process identity, not just PID
or expiry. Recorded living children retain their GPU reservation even after an
owner disappears. Child execution records ownership before exec and checks the
claiming parent identity. Timeout cleanup signals only the owned process tree.
No unrelated GPU process is killed. User holds remain authoritative.

The CLI has a persistent direct-vLLM pool. Compatible model/export/server settings
reuse the same server; changing compatibility closes only owned servers. Before
training launches, the pool drains. Cross-checkpoint weight hot-swapping is not
implemented: different weight identities require a new server. No claim of
cross-node service sharing is made. Real GPU lifecycle qualification is pending.

Transient connection/time failures have bounded retries. Confirmed CUDA OOMs
reduce direct-client concurrency/batch controls. Arbitrary shell commands are
not rewritten or classified as transient from generic nonzero exits. Per-backend
OOM exception propagation, judge-service integration and production-scale leases
remain qualification work.

`monitor --watch` renders job/task/resource assignments, attempts, queue totals
and optional progress files. Training writes a progress.json containing session-relative completed/total steps
and elapsed time; set the job progress_file to that path. ETAs require observed
completion and elapsed time;
unknown progress is displayed as unknown. The initial runner drains currently
runnable work; it is not a continuously polling multi-node daemon.

Plan holds are inserted atomically with each job, including CLI imports. See the
[held XXS campaign](xxs-dfm15-ablation.md) for preparation without execution.

## Waiting for external GPU users and parallel task evaluation

`python -m mimir.scheduling.idle LOCAL_DB --gpus 0,1,2,3,4,5,6,7`
adds a continuously polling coordinator using the same runner lock, store,
attempt executor and monitor. The original `mimir schedule --run` still drains
only currently runnable jobs. The idle coordinator requires three clear GPU
checks (30-second intervals), fails closed on probe errors, and never signals
external GPU users. This cannot reserve devices against other launchers.

Non-evaluation jobs execute serially. GPU-specific evaluation subprocesses claim
only jobs whose server.gpus matches their single GPU, using an optional
transactional claim predicate. Server ports must be distinct. These workers reuse
compatible TP=1 servers and all exit before the coordinator starts more training.
The coordinator does not release user holds. Its local idle-status.json is shown
by the native monitor, with a timestamp so stale status is visible.

Periodic idle-coordinator backups use a fresh staging file and atomically replace
the prior snapshot only after SQLite integrity verification. A failed refresh
retains the previous backup and emits a warning; it does not strand training.
The administrative Store.backup API still refuses overwriting an existing path.
