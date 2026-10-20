#!/usr/bin/env python3
"""Validate the repository's Open Knowledge Format bundle."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
DATE_HEADING_RE = re.compile(r"^## \d{4}-\d{2}-\d{2}$", re.MULTILINE)
VALID_STATUS = {"draft", "stable", "deprecated"}
VALID_CONFIDENCE = {"high", "medium", "low"}
MAX_CONCEPT_BYTES = 50_000


def parse_frontmatter(path: Path, text: str) -> tuple[dict[str, Any] | None, str]:
    match = FRONTMATTER_RE.match(text)
    if match is None:
        return None, text
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return metadata, text[match.end() :]


def resolve_link(bundle: Path, source: Path, target: str) -> Path | None:
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target or "://" in target or target.startswith(("mailto:", "#")):
        return None
    if not (target.endswith(".md") or target.endswith("/")):
        return None
    if target.startswith("/"):
        candidate = bundle / target.removeprefix("/")
    else:
        candidate = source.parent / target
    if target.endswith("/"):
        candidate /= "index.md"
    return candidate.resolve()


def validate(bundle: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    bundle = bundle.resolve()
    markdown_files = sorted(bundle.rglob("*.md"))

    directories = {bundle, *(path.parent for path in markdown_files)}
    for directory in sorted(directories):
        index_path = directory / "index.md"
        relative_directory = directory.relative_to(bundle)
        if not index_path.is_file():
            errors.append(f"{relative_directory or Path('.')}: directory requires index.md")
            continue
        index_text = index_path.read_text(encoding="utf-8")
        immediate_concepts = [
            path.name
            for path in directory.glob("*.md")
            if path.name not in {"index.md", "log.md"}
        ]
        immediate_groups = [
            f"{path.name}/"
            for path in directory.iterdir()
            if path.is_dir() and any(path.rglob("*.md"))
        ]
        link_targets = {
            target.strip().split(maxsplit=1)[0].strip("<>").split("#", 1)[0]
            for target in LINK_RE.findall(index_text)
        }
        for child in sorted([*immediate_concepts, *immediate_groups]):
            if child not in link_targets:
                errors.append(f"{index_path.relative_to(bundle)}: missing immediate child link {child!r}")

    for path in markdown_files:
        relative = path.relative_to(bundle)
        text = path.read_text(encoding="utf-8")
        try:
            metadata, body = parse_frontmatter(path, text)
        except ValueError as exc:
            errors.append(f"{relative}: {exc}")
            continue

        if path.name == "index.md":
            if relative == Path("index.md"):
                if metadata is None or str(metadata.get("okf_version")) != "0.2":
                    errors.append(f'{relative}: root index must declare okf_version: "0.2"')
            elif metadata is not None:
                errors.append(f"{relative}: subdirectory indexes must not have frontmatter")
        elif path.name == "log.md":
            if metadata is not None:
                errors.append(f"{relative}: log files must not have frontmatter")
            if not DATE_HEADING_RE.search(body):
                errors.append(f"{relative}: log must contain ISO YYYY-MM-DD headings")
        else:
            if path.stat().st_size >= MAX_CONCEPT_BYTES:
                errors.append(
                    f"{relative}: concept is {path.stat().st_size} bytes; split concepts at {MAX_CONCEPT_BYTES} bytes"
                )
            if metadata is None:
                errors.append(f"{relative}: concept is missing YAML frontmatter")
                continue
            concept_type = metadata.get("type")
            if not isinstance(concept_type, str) or not concept_type.strip():
                errors.append(f"{relative}: concept requires a non-empty type")
            status = metadata.get("status")
            if status is not None and status not in VALID_STATUS:
                errors.append(f"{relative}: invalid status {status!r}")
            confidence = metadata.get("confidence")
            if confidence is not None and confidence not in VALID_CONFIDENCE:
                errors.append(f"{relative}: invalid confidence {confidence!r}")
            sources = metadata.get("sources", [])
            if sources and not isinstance(sources, list):
                errors.append(f"{relative}: sources must be a list")
            elif isinstance(sources, list):
                for index, source in enumerate(sources):
                    if not isinstance(source, dict) or not source.get("resource"):
                        errors.append(f"{relative}: sources[{index}] requires resource")

        if "[[" in body or "]]" in body:
            errors.append(f"{relative}: legacy wiki-link syntax remains")
        for raw_target in LINK_RE.findall(body):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            resolved = resolve_link(bundle, path, target)
            if resolved is not None and not resolved.exists():
                warnings.append(f"{relative}: unresolved local link {raw_target!r}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path, default=Path("wiki"))
    args = parser.parse_args()
    errors, warnings = validate(args.bundle)
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    print(
        f"OKF validation: {len(errors)} error(s), {len(warnings)} warning(s)",
        file=sys.stderr if errors else sys.stdout,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
