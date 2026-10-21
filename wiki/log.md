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
