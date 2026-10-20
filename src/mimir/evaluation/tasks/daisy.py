from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

DEFAULT_HUGGING_FACE_ID = "schneiderkamplab/SDU-Daisy"
DEFAULT_SPLIT = "train"
DEFAULT_PROMPT_TEMPLATE = """Besvar spørgsmålet med kun det direkte svar, uden forklaring om hvorfor.
Regelsæt:
- Svar kun på dansk.
- Hvis svaret er i højde, svar i meter (m).
- Hvis svaret er i vægt, svar i kilogram (kg).
- Hvis svaret er om en størrelse, svar i centimeter (cm). Fx Hvor stort er maleriet Mona Lisa? Svar: 77 cm x 53 cm.
- Hvis svaret er en person angiv den måde de typisk bliver angivet på i danske tekster.

Spørgsmål: {question}
Svar:"""
DEFAULT_MAX_GEN_TOKS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_QUESTION_FIELD = "Question"
DEFAULT_TARGET_FIELD = "Answer"

_NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")


@task(name="daisy")
def daisy(
    hugging_face_id: str = DEFAULT_HUGGING_FACE_ID,
    split: str = DEFAULT_SPLIT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    question_field: str = DEFAULT_QUESTION_FIELD,
    target_field: str = DEFAULT_TARGET_FIELD,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    temperature: float = DEFAULT_TEMPERATURE,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    preferred_metric: str | None = None,
) -> Task:
    # Exporters can read this from recorded task_args to override display defaults.
    _ = preferred_metric

    if not split.strip():
        raise ValueError("`split` must be a non-empty string.")
    if max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1.")

    return Task(
        dataset=hf_dataset(
            path=hugging_face_id,
            split=split.strip(),
            sample_fields=lambda record: record_to_sample(
                record=record,
                prompt_template=prompt_template,
                question_field=question_field,
                target_field=target_field,
            ),
            auto_id=True,
            shuffle=shuffle,
            seed=seed,
            limit=limit,
        ),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=[daisy_scorer()],
    )


@scorer(
    metrics={
        "exact_match": [mean(), stderr()],
        "f1": [mean(), stderr()],
        "bleu": [mean(), stderr()],
    }
)
def daisy_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip().replace("\n", " ")
        references = [item.strip() for item in target if item.strip()]
        metrics = score_prediction(prediction, references)

        return Score(
            value=metrics,
            answer=prediction,
            explanation=(
                f"exact_match={metrics['exact_match']}, "
                f"f1={metrics['f1']}, bleu={metrics['bleu']}"
            ),
            metadata={
                "prediction": prediction,
                "targets": references,
            },
        )

    return score


def record_to_sample(
    *,
    record: Mapping[str, Any],
    prompt_template: str,
    question_field: str,
    target_field: str,
) -> Sample:
    question = _require_string(record, question_field)
    answer = _require_string(record, target_field)

    return Sample(
        id=_sample_id(record),
        input=prompt_template.format(question=question),
        target=[answer],
        metadata={
            "question": question,
            "answer": answer,
            "subject": record.get("Subject"),
        },
    )


def score_prediction(prediction: str, references: Iterable[str]) -> dict[str, float]:
    reference_list = [reference for reference in references if reference]
    if not reference_list:
        return {
            "exact_match": 0.0,
            "f1": 0.0,
            "bleu": 0.0,
        }

    return {
        "exact_match": max(exact_match_score(prediction, reference) for reference in reference_list),
        "f1": max(f1_score(prediction, reference) for reference in reference_list),
        "bleu": max(bleu_score(prediction, reference) for reference in reference_list),
    }


def normalize_text(text: str) -> str:
    """Mirror the upstream DAISY normalization, including ASCII-only filtering."""
    return _NON_ALPHANUMERIC_RE.sub(" ", text.lower()).strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())

    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def bleu_score(prediction: str, ground_truth: str) -> float:
    return compute_sentence_bleu(
        hypothesis=normalize_text(prediction).split(),
        references=[normalize_text(ground_truth).split()],
    )


def compute_sentence_bleu(
    *, hypothesis: list[str], references: list[list[str]], weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
) -> float:
    if not references:
        return 0.0

    max_order = len(weights)
    precisions: list[float] = []
    smoothing_index = 1

    for ngram_order in range(1, max_order + 1):
        numerator, denominator = _modified_precision(references, hypothesis, ngram_order)
        if ngram_order == 1 and numerator == 0:
            return 0.0

        if numerator == 0 and len(hypothesis) > 1:
            smoothed_numerator = 1 / (2**smoothing_index * 5 / math.log(len(hypothesis)))
            precisions.append(smoothed_numerator / denominator)
            smoothing_index += 1
            continue

        precisions.append(numerator / denominator)

    brevity_penalty = _brevity_penalty(
        closest_ref_len=_closest_ref_length(references, len(hypothesis)),
        hyp_len=len(hypothesis),
    )
    weighted_log_precision = math.fsum(
        weight * math.log(precision)
        for weight, precision in zip(weights, precisions, strict=True)
        if precision > 0
    )
    return brevity_penalty * math.exp(weighted_log_precision)


def _modified_precision(
    references: list[list[str]], hypothesis: list[str], ngram_order: int
) -> tuple[float, float]:
    hypothesis_counts = _count_ngrams(hypothesis, ngram_order)
    max_reference_counts: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        reference_counts = _count_ngrams(reference, ngram_order)
        for ngram, count in reference_counts.items():
            max_reference_counts[ngram] = max(max_reference_counts[ngram], count)

    clipped_total = sum(
        min(count, max_reference_counts[ngram])
        for ngram, count in hypothesis_counts.items()
    )
    total = max(1, sum(hypothesis_counts.values()))
    return float(clipped_total), float(total)


def _count_ngrams(tokens: list[str], ngram_order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < ngram_order:
        return Counter()
    return Counter(
        tuple(tokens[index : index + ngram_order])
        for index in range(len(tokens) - ngram_order + 1)
    )


def _closest_ref_length(references: list[list[str]], hyp_len: int) -> int:
    return min(
        (len(reference) for reference in references),
        key=lambda reference_length: (abs(reference_length - hyp_len), reference_length),
    )


def _brevity_penalty(*, closest_ref_len: int, hyp_len: int) -> float:
    if hyp_len > closest_ref_len:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1 - closest_ref_len / hyp_len)


def _sample_id(record: Mapping[str, Any]) -> str | None:
    raw_id = record.get("id")
    if raw_id is None:
        return None
    text_id = str(raw_id).strip()
    return text_id or None


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record field '{field}' must be a non-empty string.")
    return value