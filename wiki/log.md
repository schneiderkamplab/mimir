# Update log

## 2026-10-20

Created independent Mimir package and OKF v0.2 bundle. Captured
  native working-tree provenance and dependency patches, ported model/training,
  implemented versioned checkpoints and SQLite scheduling, recovered comparison
  evidence, and qualified isolated CPU training/export/evaluation/resume.

Added opt-in single-stream looped recurrence, including absent H stacks,
cache/gradient checks and automatic LR handling. Qualified H MoE / L MoS
with independently selected residual levels and dense/sparse execution.
Inference export explicitly rejects looped recurrence pending fork support.

Prepared the initial implementation push to schneiderkamplab/Mimir on GitHub,
preserving the existing main history and initial MIT notice under licenses/.
Imported native code retains its Apache license and provenance. Runtime
environments, checkpoints, builds and logs remain excluded from Git.

## 2026-10-21

Added independent opt-ins for persistent H residual streams and branch-specific
L expert mixtures, with a preset and native CPU coverage. Branch mixtures retain
MoS top-k support and top-1 surrogate gradients. Small asymmetric initial mixer
preferences let copied experts specialize. Old checkpoints missing the flags
retain their disabled meaning. Inference exports reject the new semantics until
the pinned forks support them. See [the reference](pages/persistent-h-streams.md).

## 2026-10-22

Added an AGENTS.md verification checklist covering patched-dependency preservation,
source-versus-installed-wheel checks, CPU tests, isolated qualification campaigns,
and documentation validation. Verification reports must distinguish historical
results and unrun GPU checks from current evidence.

- 2026-10-22: Prepared four XXS DFM15 20K capacity ablations with native Mimir scheduling; 152 jobs held, zero attempts. Added atomic plan holds, capacity/queue tests and the linked campaign runbook. No training or evaluation started.

- 2026-10-22: User authorized activation of the XXS campaign after GPUs become free, then specified task-parallel evaluation without TP. Added native idle-GPU coordinator, filtered transactional claims and wait-state monitoring. Bound training to eight GPUs and evaluation to eight independent TP=1 workers; released 152 jobs and launched the detached coordinator. Verified waiting on existing GPU users with zero attempts. Installed pinned FA4 only in Mimir's isolated environment; CPU import and targeted tests passed.

- 2026-10-22: Implemented optional DeepSeek-V3-style MoE/MoS bias balancing, optimizer-step/GAS/distributed counting and persistent selection-only buffers. Updated reproducible Transformers/vLLM patches for inference. User requested one baseline Muon and two balanced-routing XXS runs; added the extension recipe and native queue-registration helper. Existing campaign settings remain unchanged.

- 2026-10-22: Completed and released the three additional XXS runs: one baseline Muon, balanced MoE and balanced MoS. Verified 266 jobs, no holds, zero attempts and a live GPU-waiting coordinator. Full CPU suite: 309 tests/28 subtests; installed feature checks: 17 tests/12 subtests. Fixed periodic queue backup replacement and verified its failure path (12 scheduler tests); restarted only the owned idle coordinator to load the fix.
