---
type: Playbook
title: OKF v0.2 maintenance
description: Rules for this repository knowledge bundle.
status: stable
confidence: high
last_updated: 2026-10-20
---
# Maintenance

Every concept starts with YAML frontmatter containing type, title, description,
status, confidence and last_updated. Use draft/stable/deprecated lifecycle and
high/medium/low confidence according to evidence. Reserved index.md and log.md
files need no concept frontmatter. wiki/index.md declares okf_version "0.2".

Keep concepts focused and below 50,000 bytes. Every directory index links its
immediate children. Use relative Markdown links and preserve incoming paths.
Record durable commands, design decisions, blockers and verification evidence
in the same turn. Mark contradicted claims superseded with date/context rather
than silently deleting them. Update the relevant index and log for structural
changes. Never label unrun GPU checks as passed because CPU tests passed.

Validate using `python scripts/validate_okf.py wiki`.
