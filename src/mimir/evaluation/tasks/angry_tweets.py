"""Angry Tweets — Danish sentiment classification (3-class).

Dataset: DDSC/angry-tweets (Hugging Face)
Labels: positiv, neutral, negativ
Test split: 1,047 samples

The model is asked to classify a Danish tweet as positive, neutral, or negative.
Scoring: macro-F1 + accuracy + MCC.
"""

from __future__ import annotations

import math
import re
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState, generate

DEFAULT_HUGGING_FACE_ID = "DDSC/angry-tweets"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_GEN_TOKS = 16
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT_TEMPLATE = (
    "Klassificer følgende danske tweet i en af tre kategorier: "
    "positiv, neutral eller negativ.\n\n"
    "Tweet: {{text}}\n\n"
    "Svar kun med et af følgende ord: positiv, neutral, negativ."
)

VALID_LABELS = ("positiv", "neutral", "negativ")
_LABEL_ALIASES = {
    "positiv": "positiv",
    "positive": "positiv",
    "pos": "positiv",
    "neutral": "neutral",
    "neutralt": "neutral",
    "negativ": "negativ",
    "negative": "negativ",
    "neg": "negativ",
}

_RE_LABEL = re.compile(
    r"\b(positiv|positive|pos|neutral|neutralt|negativ|negative|neg)\b",
    re.IGNORECASE,
)


@task(name="angry-tweets")
def angry_tweets(
    hugging_face_id: str = DEFAULT_HUGGING_FACE_ID,
    split: str = DEFAULT_SPLIT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    temperature: float = DEFAULT_TEMPERATURE,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
) -> Task:
    return Task(
        dataset=hf_dataset(
            path=hugging_face_id,
            split=split,
            sample_fields=lambda record: _record_to_sample(
                record=record,
                prompt_template=prompt_template,
            ),
            auto_id=True,
            shuffle=shuffle,
            seed=seed,
            limit=limit,
        ),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=angry_tweets_scorer(),
    )


def _record_to_sample(record: dict[str, Any], prompt_template: str) -> Sample:
    text = record["text"]
    label = record["label"].strip().lower()
    return Sample(
        input=prompt_template.replace("{{text}}", text),
        target=label,
        metadata={"text": text, "label": label},
    )


def _extract_label(text: str) -> str | None:
    if not text:
        return None
    match = _RE_LABEL.search(text)
    if match is None:
        return None
    return _LABEL_ALIASES.get(match.group(1).lower())


@metric(name="macro_f1")
def angry_tweets_macro_f1() -> Metric:
    def metric_fn(scores: list[SampleScore]) -> float:
        return _macro_f1(_label_pairs(scores))
    return metric_fn


@metric(name="accuracy")
def angry_tweets_accuracy() -> Metric:
    def metric_fn(scores: list[SampleScore]) -> float:
        pairs = _label_pairs(scores)
        if not pairs:
            return 0.0
        correct = sum(1 for target, pred in pairs if pred == target)
        return correct / len(pairs)
    return metric_fn


@metric(name="mcc")
def angry_tweets_mcc() -> Metric:
    def metric_fn(scores: list[SampleScore]) -> float:
        return _mcc(_label_pairs(scores))
    return metric_fn


@scorer(
    metrics=[angry_tweets_macro_f1(), angry_tweets_accuracy(), angry_tweets_mcc()],
    name="sentiment",
)
def angry_tweets_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        expected = target.text.strip().lower()
        predicted = _extract_label(state.output.completion)
        return Score(
            value=predicted or "",
            answer=predicted or "",
            explanation=f"predicted={predicted!r}, expected={expected!r}",
            metadata={"prediction": predicted, "target": expected},
        )
    return score


def _label_pairs(scores: list[SampleScore]) -> list[tuple[str, str | None]]:
    pairs = []
    for s in scores:
        meta = s.score.metadata or {}
        target = meta.get("target", "")
        pred = meta.get("prediction")
        if isinstance(pred, str) and pred in VALID_LABELS:
            pairs.append((target, pred))
        else:
            pairs.append((target, None))
    return pairs


def _macro_f1(pairs: list[tuple[str, str | None]]) -> float:
    if not pairs:
        return 0.0
    f1s = []
    for label in VALID_LABELS:
        tp = fp = fn = 0
        for target, pred in pairs:
            if target == label and pred == label:
                tp += 1
            elif target != label and pred == label:
                fp += 1
            elif target == label and pred != label:
                fn += 1
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        if p + r:
            f1s.append(2 * p * r / (p + r))
        else:
            f1s.append(0.0)
    return sum(f1s) / len(f1s)


def _mcc(pairs: list[tuple[str, str | None]]) -> float:
    if not pairs:
        return 0.0
    invalid = "__invalid__"
    labels = list(VALID_LABELS)
    if any(pred is None for _, pred in pairs):
        labels.append(invalid)
    index = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for target, pred in pairs:
        pl = pred if pred is not None else invalid
        matrix[index[target]][index[pl]] += 1
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0
    correct = sum(matrix[i][i] for i in range(len(labels)))
    true_totals = [sum(row) for row in matrix]
    pred_totals = [sum(matrix[r][c] for r in range(len(labels))) for c in range(len(labels))]
    cov = correct * total - sum(p * t for p, t in zip(pred_totals, true_totals, strict=True))
    pn = total * total - sum(c * c for c in pred_totals)
    tn = total * total - sum(c * c for c in true_totals)
    denom = math.sqrt(pn * tn)
    return cov / denom if denom else 0.0
