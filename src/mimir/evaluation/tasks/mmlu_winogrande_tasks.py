#!/usr/bin/env python3
"""Chunked MMLU 5-shot (per-subject dev) + Winogrande for Gemma 4.

MMLU: 4 chunks, 5-shot per-subject dev, inspect_evals multiple_choice format.
  - mmlu_5shot_chunk0..3: used for both no-think (max_tokens=64) and think (max_tokens=2048)
Winogrande: full validation, max_tokens=2048 for thinking.
"""

from typing import Any
from collections import defaultdict
from datasets import load_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, Solver
from inspect_ai.model import GenerateConfig
from inspect_ai.model import (
    ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessage
)
import random

SHUFFLE_SEED = 42
MMLU_NUM_CHUNKS = 4

# ─── MMLU 5-shot ─────────────────────────────────────────────────────────────

def _format_mmlu_choices(choices: list[str]) -> str:
    return "\n".join(
        f"({chr(ord('a') + i)}) {c}" for i, c in enumerate(choices)
    )

def _format_mmlu_question(question: str, choices: list[str]) -> str:
    letters = ",".join(chr(ord("a") + i) for i in range(len(choices)))
    return (
        f"Answer the following multiple choice question. "
        f"The entire content of your response should be of the following format: "
        f"'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n"
        f"{question}\n{_format_mmlu_choices(choices)}"
    )

def _build_mmlu_5shot_samples() -> list[Sample]:
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")
    dev_by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in dev_ds:
        dev_by_subject[r["subject"]].append(r)

    test_ds = load_dataset("cais/mmlu", "all", split="test")
    all_rows = list(test_ds)
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(all_rows)

    samples = []
    for row in all_rows:
        subject = row["subject"]
        target = "ABCD"[row["answer"]]
        shots = dev_by_subject[subject][:5]

        messages: list[ChatMessage] = []
        messages.append(ChatMessageSystem(
            content=f"The following are multiple choice questions about {subject}."
        ))
        for s in shots:
            s_target = "ABCD"[s["answer"]]
            messages.append(ChatMessageUser(
                content=_format_mmlu_question(s["question"], s["choices"])
            ))
            messages.append(ChatMessageAssistant(content=f"Answer: {s_target}"))
        messages.append(ChatMessageUser(content=row["question"]))

        samples.append(Sample(
            input=messages,
            target=target,
            choices=row["choices"],
            metadata={"subject": subject},
            id=len(samples),
        ))
    return samples

def _load_mmlu_chunk(chunk_idx: int) -> MemoryDataset:
    samples = _build_mmlu_5shot_samples()
    chunk_size = len(samples) // MMLU_NUM_CHUNKS
    start = chunk_idx * chunk_size
    end = start + chunk_size if chunk_idx < MMLU_NUM_CHUNKS - 1 else len(samples)
    chunk = samples[start:end]
    for i, s in enumerate(chunk):
        s.id = i
    return MemoryDataset(chunk)

def _mmlu_solver_nothink() -> Solver:
    return multiple_choice(cot=False)

def _mmlu_solver_think() -> Solver:
    return multiple_choice(cot=True)

@task
def mmlu_5shot_chunk0_nothink():
    return Task(
        dataset=_load_mmlu_chunk(0),
        solver=_mmlu_solver_nothink(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=64, temperature=0.0),
    )

@task
def mmlu_5shot_chunk1_nothink():
    return Task(
        dataset=_load_mmlu_chunk(1),
        solver=_mmlu_solver_nothink(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=64, temperature=0.0),
    )

@task
def mmlu_5shot_chunk2_nothink():
    return Task(
        dataset=_load_mmlu_chunk(2),
        solver=_mmlu_solver_nothink(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=64, temperature=0.0),
    )

@task
def mmlu_5shot_chunk3_nothink():
    return Task(
        dataset=_load_mmlu_chunk(3),
        solver=_mmlu_solver_nothink(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=64, temperature=0.0),
    )

@task
def mmlu_5shot_chunk0_think():
    return Task(
        dataset=_load_mmlu_chunk(0),
        solver=_mmlu_solver_think(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=2048, temperature=0.0),
    )

@task
def mmlu_5shot_chunk1_think():
    return Task(
        dataset=_load_mmlu_chunk(1),
        solver=_mmlu_solver_think(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=2048, temperature=0.0),
    )

@task
def mmlu_5shot_chunk2_think():
    return Task(
        dataset=_load_mmlu_chunk(2),
        solver=_mmlu_solver_think(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=2048, temperature=0.0),
    )

@task
def mmlu_5shot_chunk3_think():
    return Task(
        dataset=_load_mmlu_chunk(3),
        solver=_mmlu_solver_think(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=2048, temperature=0.0),
    )

# ─── Winogrande (thinking, max_tokens=2048) ──────────────────────────────────

def _winogrande_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=f"Sentence: {record['sentence'].replace('_', '[BLANK]')}",
        target={"1": "A", "2": "B"}[record["answer"]],
        choices=[record["option1"], record["option2"]],
    )

def _load_winogrande_full() -> MemoryDataset:
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    samples = [_winogrande_record_to_sample(r) for r in ds]
    seen = set()
    deduped = []
    for s in samples:
        key = str(s.input)
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    for i, s in enumerate(deduped):
        s.id = i
    return MemoryDataset(deduped)

WINO_USER = (
    "Answer the following multiple choice question by choosing the best logical option "
    "to replace the [BLANK]. The entire content of your response should be of the following "
    "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A, B.\n\n"
    "{question}\n{choices}"
)

def _winogrande_solver() -> Solver:
    return multiple_choice(template=WINO_USER, shuffle=False)

@task
def winogrande_think_full():
    return Task(
        dataset=_load_winogrande_full(),
        solver=_winogrande_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=2048, temperature=0.0),
    )
