from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate

from mimir.evaluation.scorers.chrf import chrf3pp

DEFAULT_DATASET_ID = "synquid/wmt24pp"
DEFAULT_SUBSET = "en-da_DK"
DEFAULT_SOURCE_SPLIT = "train"
DEFAULT_TARGET_LANGUAGE = "Danish"
DEFAULT_PROMPT_TEMPLATE = """You are translating English text into {{target_language}}.

Return only the translation. Do not include an introduction, explanation, notes, alternatives, markdown formatting, or a quoted wrapper unless quotes are part of the translation.

English text:
{{source}}"""
DEFAULT_MAX_GEN_TOKS = 512
DEFAULT_TEMPERATURE = 0.0


@task(name="wmt24pp-en-da")
def wmt24pp_en_da(
    dataset_id: str = DEFAULT_DATASET_ID,
    subset: str = DEFAULT_SUBSET,
    source_split: str = DEFAULT_SOURCE_SPLIT,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    temperature: float = DEFAULT_TEMPERATURE,
    limit: int | None = None,
    preferred_metric: str | None = None,
) -> Task:
    # Exporters can read this from recorded task_args to override display defaults.
    _ = preferred_metric

    if max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1.")

    records = _load_records(
        dataset_id=dataset_id,
        subset=subset,
        source_split=source_split,
    )
    samples = [
        record_to_sample(
            record=record,
            prompt_template=prompt_template,
            target_language=target_language,
        )
        for record in records
    ]
    if limit is not None:
        samples = samples[:limit]

    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name="WMT24++ en-da",
            location=f"{dataset_id}:{subset}",
        ),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=chrf3pp(),
    )


def _load_records(
    *, dataset_id: str, subset: str, source_split: str
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if not source_split.strip():
        raise ValueError("`source_split` must be a non-empty string.")

    dataset = load_dataset(dataset_id, name=subset, split=source_split)
    return [
        dict(record)
        for record in dataset
        if _is_usable_record(record)
    ]


def _is_usable_record(record: Mapping[str, Any]) -> bool:
    if bool(record.get("is_bad_source")):
        return False

    source = record.get("source")
    target = record.get("target")
    return (
        isinstance(source, str)
        and bool(source.strip())
        and isinstance(target, str)
        and bool(target.strip())
    )


def record_to_sample(
    *,
    record: Mapping[str, Any],
    prompt_template: str,
    target_language: str,
) -> Sample:
    source = _require_string(record, "source")
    target = _require_string(record, "target")

    return Sample(
        id=_sample_id(record),
        input=(
            prompt_template.replace("{{source}}", source).replace(
                "{{target_language}}", target_language
            )
        ),
        target=[target],
        metadata={
            "source": source,
            "target": target,
            "original_target": _optional_string(record, "original_target"),
            "language_pair": _optional_string(record, "lp"),
            "domain": _optional_string(record, "domain"),
            "document_id": _optional_string(record, "document_id"),
            "segment_id": record.get("segment_id"),
        },
    )


def _sample_id(record: Mapping[str, Any]) -> str | None:
    document_id = _optional_string(record, "document_id")
    segment_id = record.get("segment_id")
    if document_id is not None and segment_id is not None:
        return f"{document_id}:{segment_id}"

    raw_id = record.get("id")
    if raw_id is None:
        return None

    sample_id = str(raw_id).strip()
    return sample_id or None


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record field '{field}' must be a non-empty string.")
    return value


def _optional_string(record: Mapping[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    text = str(value).strip()
    return text or None
