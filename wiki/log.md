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
