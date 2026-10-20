from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState, generate

DEFAULT_HUGGING_FACE_ID = "giannor/dala"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_GEN_TOKS = 8
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT_TEMPLATE = """Sætning: {{text}}

Bestem om sætningen er grammatisk korrekt eller ej. Svar kun med ja eller nej, og intet andet."""
DEFAULT_TEXT_FIELD = "text"
DEFAULT_LABEL_FIELD = "label"
VALID_LABELS = ("correct", "incorrect")

SplitName = Literal["train", "val", "test", "full_train"]

_RE_FIRST_LABEL = re.compile(
    r"\b(incorrect|correct|ukorrekt|korrekt|forkert|ja|nej)\b",
    re.IGNORECASE,
)
_LABEL_ALIASES = {
    "ja": "correct",
    "nej": "incorrect",
    "correct": "correct",
    "korrekt": "correct",
    "incorrect": "incorrect",
    "ukorrekt": "incorrect",
    "forkert": "incorrect",
}


@task(name="dala")
def dala(
    hugging_face_id: str = DEFAULT_HUGGING_FACE_ID,
    split: str = DEFAULT_SPLIT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    text_field: str = DEFAULT_TEXT_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    temperature: float = DEFAULT_TEMPERATURE,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    preferred_metric: str | None = None,
) -> Task:
    # Exporters can read this from recorded task_args to override display defaults.
    _ = preferred_metric

    if max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1.")

    normalized_split = _normalize_split_name(split)

    return Task(
        dataset=hf_dataset(
            path=hugging_face_id,
            split=normalized_split,
            sample_fields=lambda record: record_to_sample(
                record=record,
                prompt_template=prompt_template,
                text_field=text_field,
                label_field=label_field,
            ),
            auto_id=True,
            shuffle=shuffle,
            seed=seed,
            limit=limit,
        ),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=dala_scorer(),
    )


def _normalize_split_name(split: str) -> SplitName:
    normalized = split.strip().lower()
    match normalized:
        case "train" | "training":
            return "train"
        case "val" | "valid" | "validation" | "dev":
            return "val"
        case "test":
            return "test"
        case "full_train" | "full-train" | "fulltrain":
            return "full_train"
        case "":
            raise ValueError("`split` must be a non-empty string.")
        case _:
            raise ValueError(
                f"Unsupported split '{split}'. Supported values: "
                "['dev', 'full-train', 'full_train', 'fulltrain', "
                "'test', 'train', 'training', 'val', 'valid', 'validation']"
            )


def record_to_sample(
    *,
    record: Mapping[str, Any],
    prompt_template: str,
    text_field: str,
    label_field: str,
) -> Sample:
    text = _require_string(record, text_field)
    label = _normalize_label(_require_string(record, label_field))
    corruption_type = record.get("corruption_type")

    return Sample(
        input=prompt_template.replace("{{text}}", text),
        target=label,
        metadata={
            "text": text,
            "corruption_type": (
                str(corruption_type).strip() if corruption_type is not None else None
            ),
        },
    )


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record field '{field}' must be a non-empty string.")
    return value


def _normalize_label(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VALID_LABELS:
        raise ValueError(
            f"Unsupported label '{value}'. Supported labels: {list(VALID_LABELS)}"
        )
    return normalized


def _extract_label(text: str) -> str | None:
    if not text:
        return None

    match = _RE_FIRST_LABEL.search(text)
    if match is None:
        return None

    return _LABEL_ALIASES.get(match.group(1).lower())


@metric(name="macro_f1")
def dala_macro_f1() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        return _macro_f1_from_pairs(_label_pairs(scores))

    return metric


@metric(name="mcc")
def dala_mcc() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        return _mcc_from_pairs(_label_pairs(scores))

    return metric


@scorer(
    metrics=[dala_macro_f1(), dala_mcc()],
    name="linguistic-acceptability",
)
def dala_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        expected = _normalize_label(target.text)
        predicted = _extract_label(state.output.completion)

        return Score(
            value=predicted or "",
            answer=predicted or "",
            explanation=f"predicted={predicted!r}, expected={expected!r}",
            metadata={
                "prediction": predicted,
                "target": expected,
            },
        )

    return score


def _label_pairs(scores: list[SampleScore]) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    for sample_score in scores:
        metadata = sample_score.score.metadata or {}
        target = _normalize_label(str(metadata.get("target") or ""))
        prediction_raw = metadata.get("prediction")
        prediction = (
            _normalize_label(str(prediction_raw))
            if isinstance(prediction_raw, str) and prediction_raw.strip() in VALID_LABELS
            else None
        )
        pairs.append((target, prediction))
    return pairs


def _macro_f1_from_pairs(pairs: list[tuple[str, str | None]]) -> float:
    if not pairs:
        return 0.0

    f1_scores: list[float] = []
    for label in VALID_LABELS:
        tp = 0
        fp = 0
        fn = 0
        for target, prediction in pairs:
            if target == label and prediction == label:
                tp += 1
            elif target != label and prediction == label:
                fp += 1
            elif target == label and prediction != label:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)

    return sum(f1_scores) / len(f1_scores)


def _mcc_from_pairs(pairs: list[tuple[str, str | None]]) -> float:
    if not pairs:
        return 0.0

    invalid_label = "__invalid__"
    labels = [*VALID_LABELS]
    if any(prediction is None for _, prediction in pairs):
        labels.append(invalid_label)

    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for target, prediction in pairs:
        predicted_label = prediction if prediction is not None else invalid_label
        matrix[index[target]][index[predicted_label]] += 1

    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0

    correct = sum(matrix[i][i] for i in range(len(labels)))
    true_totals = [sum(row) for row in matrix]
    predicted_totals = [
        sum(matrix[row_idx][col_idx] for row_idx in range(len(labels)))
        for col_idx in range(len(labels))
    ]

    covariance = correct * total - sum(
        predicted * actual
        for predicted, actual in zip(predicted_totals, true_totals, strict=True)
    )
    pred_norm = total * total - sum(count * count for count in predicted_totals)
    true_norm = total * total - sum(count * count for count in true_totals)
    denominator = math.sqrt(pred_norm * true_norm)
    if denominator == 0.0:
        return 0.0

    return covariance / denominator
