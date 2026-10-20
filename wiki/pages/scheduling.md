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
optional `max_attempts`, and `depends` entries with `id` and `policy` (`success`
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
