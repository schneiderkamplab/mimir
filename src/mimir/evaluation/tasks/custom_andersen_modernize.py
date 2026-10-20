"""Standalone Andersen Danish text modernization task for inspect_ai.

Inlined from dfm-evals/dfm_evals/tasks/andersen_modernize.py to avoid relative
import issues when loaded as a standalone file by inspect_ai.
"""

from __future__ import annotations

import json
import os
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate
from sacrebleu.metrics import CHRF, BLEU

DEFAULT_PAIRS_FILE = os.environ.get("MIMIR_ANDERSEN_VALIDATION", "andersen-validation.jsonl")
DEFAULT_SYSTEM_PROMPT = (
    "Du er en ekspert i at modernisere gammel dansk tekst til nutidig dansk "
    "stavemåde og sprogbrug. Bevar tekstens betydning, stil og tone, men "
    "opdater forældet stavemåde, bøjningsformer og gloser til moderne dansk."
)
DEFAULT_USER_TEMPLATE = "Moderniser følgende danske tekst til nutidig dansk:\n\n{text}"
DEFAULT_MAX_GEN_TOKS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_FEW_SHOT_FILE = os.environ.get("MIMIR_ANDERSEN_TRAIN", "andersen-train.jsonl")


@task(name="andersen-modernize")
def andersen_modernize(
    pairs_file: str = DEFAULT_PAIRS_FILE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    temperature: float = DEFAULT_TEMPERATURE,
    limit: int | None = None,
    n_shot: int = 0,
    few_shot_file: str = DEFAULT_FEW_SHOT_FILE,
    preferred_metric: str | None = None,
) -> Task:
    _ = preferred_metric

    samples = _load_samples(
        pairs_file=pairs_file,
        system_prompt=system_prompt,
        user_template=user_template,
        n_shot=n_shot,
        few_shot_file=few_shot_file,
        limit=limit,
    )

    solver = [generate(max_tokens=max_gen_toks, temperature=temperature)]

    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name="Andersen Modernization",
            location=pairs_file,
        ),
        solver=solver,
        scorer=[exact_match_scorer(), chrf3pp_scorer(), bleu_scorer()],
    )


def _load_samples(
    *,
    pairs_file: str,
    system_prompt: str,
    user_template: str,
    n_shot: int,
    few_shot_file: str,
    limit: int | None,
) -> list[Sample]:
    with open(pairs_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    few_shot_examples: list[dict[str, Any]] = []
    if n_shot > 0:
        with open(few_shot_file, encoding="utf-8") as f:
            train_records = [json.loads(line) for line in f]
        train_records.sort(
            key=lambda r: len(r["messages"][1]["content"])
        )
        few_shot_examples = train_records[:n_shot]

    samples: list[Sample] = []
    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break

        user_content = user_template.replace("{text}", rec["messages"][1]["content"])
        target = rec["messages"][2]["content"]

        prompt_parts: list[dict[str, str]] = []
        prompt_parts.append({"role": "system", "content": system_prompt})
        for ex in few_shot_examples:
            ex_user = user_template.replace("{text}", ex["messages"][1]["content"])
            ex_assistant = ex["messages"][2]["content"]
            prompt_parts.append({"role": "user", "content": ex_user})
            prompt_parts.append({"role": "assistant", "content": ex_assistant})

        prompt_parts.append({"role": "user", "content": user_content})

        sample = Sample(
            id=f"{rec.get('id', 'rec')}-{i}",
            input=prompt_parts,
            target=[target],
            metadata={
                "title": rec.get("title", ""),
                "year": rec.get("year", ""),
                "genre": rec.get("genre", ""),
                "n_shot": n_shot,
            },
        )
        samples.append(sample)

    return samples


@scorer(metrics=[mean(), stderr()])
def exact_match_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        references = [r.strip() for r in target.target if r]
        if not references:
            return Score(value=0.0, answer=prediction)
        match = any(prediction == ref for ref in references)
        return Score(value=1.0 if match else 0.0, answer=prediction)

    return score


@scorer(metrics=[mean(), stderr()])
def chrf3pp_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion
        references = [r for r in target.target if r]
        if not references:
            return Score(value=0.0, answer=prediction)
        metric = CHRF(char_order=6, word_order=2, beta=3)
        return Score(
            value=metric.sentence_score(
                hypothesis=prediction, references=references
            ).score
            / 100.0,
            answer=prediction,
        )

    return score


@scorer(metrics=[mean(), stderr()])
def bleu_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion
        references = [r for r in target.target if r]
        if not references:
            return Score(value=0.0, answer=prediction)
        bleu = BLEU()
        result = bleu.sentence_score(prediction, references)
        return Score(value=result.score / 100.0, answer=prediction)

    return score
