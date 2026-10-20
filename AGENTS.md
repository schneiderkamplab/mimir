# Mimir operating notes

Mimir is an independent training/evaluation package with one public architecture,
Mimir2. Read [the wiki](wiki/index.md), [OKF rules](wiki/schema.md), and the relevant
runbook before substantial work. Record durable findings in this bundle in the
same turn; preserve superseded claims with date/context.

- Work only in this repository. Do not modify sibling repositories, environments,
  training jobs, dataset reconstruction jobs, or production services.
- Preserve default v1 numerical behavior and opt-in Mimir2 features. Do not add
  TRM variants, architecture import strings, historical campaign launchers,
  SimpleEngine, inference proxies, or a monolithic training entry point.
- Keep licenses, attribution, source hashes and reproducible dependency patches.
- CPU tests precede GPU qualification. Use GPUs only when explicitly available;
  never signal unrelated processes. No production migration or publishing is
  authorized by initial implementation work.
- Checkpoints must be complete before publication or ephemeral cleanup. Never
  silently downgrade exact continuation to weights initialization.
- Evaluation, merge and publication failures must not strand training. Training
  handoffs depend on terminal inference attempts, not successful publication.
  Honor user holds and resource safety; use exactly one recovery controller.
- No secrets, machine-specific execution paths, or native-source imports from a
  sibling repository. Configuration code hashes are provenance, not resume gates.

Validation and outstanding qualification: [validation](wiki/pages/validation.md).
