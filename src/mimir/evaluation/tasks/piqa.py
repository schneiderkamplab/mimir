from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Literal, cast

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import TaskState, generate
from typing_extensions import NotRequired, TypedDict

DEFAULT_DATASET_PATH = Path(__file__).parent / "piqa" / "piqa-dan.json"

PROMPT_TEMPLATE_DA = """Du får et spørgsmål om fysisk hverdagsfornuft og to svarmuligheder.
Vælg den bedste løsning.

Spørgsmål:
{prompt}

Mulighed A:
{solution0}

Mulighed B:
{solution1}

Svar kun med A eller B."""

_RE_FIRST_CHOICE = re.compile(r"^\s*([AaBb])\b")
_RE_ANY_CHOICE = re.compile(r"\b([AaBb])\b")


class PIQARecord(TypedDict):
    prompt: str
    solution0: str
    solution1: str
    label: Literal[0, 1]
    id: NotRequired[object]


@task(name="piqa")
def piqa(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
) -> Task:
    path = Path(dataset_path)
    records = _load_records(path)
    samples = [_record_to_sample(record) for record in records]

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]

    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name="PIQA-da",
            location=str(path),
        ),
        solver=[generate()],
        scorer=piqa_scorer(),
    )


def _load_records(path: Path) -> list[PIQARecord]:
    if not path.is_file():
        raise FileNotFoundError(f"PIQA dataset file not found: {path}")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(
            f"Expected PIQA dataset JSON list, got: {type(loaded).__name__}"
        )

    records: list[PIQARecord] = []
    malformed: list[str] = []
    for index, raw_record in enumerate(loaded):
        try:
            records.append(_normalize_record(raw_record))
        except ValueError as exc:
            if isinstance(raw_record, dict) and "id" in raw_record:
                malformed.append(f"idx={index}, id={raw_record['id']!r}: {exc}")
            else:
                malformed.append(f"idx={index}: {exc}")

    if not malformed:
        return records

    preview = malformed[:20]
    extra_count = len(malformed) - len(preview)
    extra_suffix = (
        f"\n... (+{extra_count} more malformed rows)" if extra_count > 0 else ""
    )
    message = f"Malformed PIQA rows in {path}:\n" + "\n".join(preview) + extra_suffix
    raise ValueError(message)


def _record_to_sample(record: PIQARecord) -> Sample:
    prompt = record["prompt"]
    solution0 = record["solution0"]
    solution1 = record["solution1"]
    label_raw = record["label"]

    target_choice = "A" if label_raw == 0 else "B"
    sample_id = str(record.get("id")) if record.get("id") is not None else None

    return Sample(
        id=sample_id,
        input=PROMPT_TEMPLATE_DA.format(
            prompt=prompt,
            solution0=solution0,
            solution1=solution1,
        ),
        target=target_choice,
        metadata={
            "solution0": solution0,
            "solution1": solution1,
        },
    )


def _normalize_record(raw_record: Any) -> PIQARecord:
    if not isinstance(raw_record, dict):
        raise ValueError(
            f"record must be an object, got {type(raw_record).__name__}"
        )

    label = raw_record.get("label")
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1 (got {label!r})")

    return {
        "id": raw_record.get("id"),
        "prompt": _normalize_text(raw_record.get("prompt"), field="prompt"),
        "solution0": _normalize_text(raw_record.get("solution0"), field="solution0"),
        "solution1": _normalize_text(raw_record.get("solution1"), field="solution1"),
        "label": cast(Literal[0, 1], label),
    }


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a non-empty string")

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must be a non-empty string")

    return cleaned


@scorer(metrics=[accuracy()])
def piqa_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        solution0 = state.metadata["solution0"]
        solution1 = state.metadata["solution1"]
        assert isinstance(solution0, str)
        assert isinstance(solution1, str)
        predicted = _extract_choice(
            text=state.output.completion,
            solution0=solution0,
            solution1=solution1,
        )
        expected = target.text.strip().upper()
        is_correct = predicted == expected

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=predicted if predicted is not None else "",
            explanation=f"predicted={predicted!r}, expected={expected!r}",
        )

    return score


def _extract_choice(text: str, solution0: str, solution1: str) -> str | None:
    choice = _extract_letter_choice(text)
    if choice is not None:
        return choice

    normalized = _normalize_for_match(text)
    sol0 = _normalize_for_match(solution0)
    sol1 = _normalize_for_match(solution1)

    has_0 = sol0 != "" and sol0 in normalized
    has_1 = sol1 != "" and sol1 in normalized
    if has_0 and not has_1:
        return "A"
    if has_1 and not has_0:
        return "B"

    return None


def _extract_letter_choice(text: str) -> str | None:
    if not text:
        return None

    for pattern in (_RE_FIRST_CHOICE, _RE_ANY_CHOICE):
        match = pattern.search(text)
        if match is not None:
            return match.group(1).upper()

    return None


def _normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())
