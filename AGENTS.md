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

## Verification checklist

- Follow [README installation instructions](README.md) and use the locked
  dependencies. After installing patched Transformers/vLLM, use the environment's
  Python/executables directly or `uv run --no-sync`. Ordinary `uv run` or another
  sync can replace patched installs with the base revisions.
- Confirm tests import the code being changed: use an editable Mimir installation
  or rebuild/reinstall its wheel before testing. An installed wheel does not
  automatically reflect edits under `src/`.
- Run relevant CPU tests first; run the full suite for changes affecting model,
  training or checkpoint behavior. With the prepared environment active:
  `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q`.
- For end-to-end checkpoint/export/resume changes, run
  `python tools/qualify_cpu.py .runtime/<new-campaign-directory>`.
  This qualifies the default architecture; feature-specific checks are still
  required for modes unsupported by the inference forks.
- Update linked wiki pages and qualification evidence as appropriate, then run
  `python scripts/validate_okf.py wiki` and `git diff --check`.
- Report what was actually tested and outstanding limits. Consult the
  [validation record](wiki/pages/validation.md) for dated results rather than
  treating historical test counts as current or CPU checks as GPU qualification.
