#!/usr/bin/env python3
"""HellaSwag-da (Danish HellaSwag) — standalone inspect_ai task.

Loads from local parquet. Set HELLASWAG_DA_PATH env var to override the default location.
MCQ format: 4 choices (a/b/c/d), max_tokens=8, accuracy scorer with partial credit.
"""

from __future__ import annotations
import os
import re
import random
import pyarrow.parquet as pq
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, accuracy, scorer, Target
from inspect_ai.solver import generate, TaskState
from inspect_ai.model import GenerateConfig

_DEFAULT_DATASET_PATH = Path(__file__).parent / "hellaswag-da-mini" / "data" / "test-00000-of-00001.parquet"
DATASET_PATH = os.environ.get("HELLASWAG_DA_PATH", str(_DEFAULT_DATASET_PATH))
SHUFFLE_SEED = 4242


@scorer(metrics=[accuracy()])
def hellaswag_da_scorer():
    """MCQ scorer with partial credit 1/4 for invalid answers."""
    async def score(state: TaskState, target: Target):
        pred = state.output.completion.strip().lower()
        if len(pred) > 0:
            pred = pred[0]
        gold = target.text.lower()
        valid_set = state.metadata.get("valid_set", ["a", "b", "c", "d"])
        num_choices = state.metadata.get("num_choices", len(valid_set))
        if pred not in valid_set:
            return Score(value=1.0 / num_choices, answer=pred)
        elif pred == gold:
            return Score(value=1.0, answer=pred)
        else:
            return Score(value=0.0, answer=pred)
    return score


@task(name="hellaswag_da")
def hellaswag_da() -> Task:
    """HellaSwag-da: Danish HellaSwag (2048 test samples, 4-choice MCQ)."""
    table = pq.read_table(DATASET_PATH)
    records = table.to_pylist()

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(records)

    samples = []
    for i, rec in enumerate(records):
        text = rec["text"]
        label = rec["label"].lower()
        choices = ["a", "b", "c", "d"]
        samples.append(Sample(
            id=i,
            input=text + "\n\nSvar kun med ét bogstav: a, b, c eller d.\nSvar:",
            target=label,
            metadata={"valid_set": choices, "num_choices": len(choices)},
        ))

    return Task(
        dataset=MemoryDataset(samples, name="HellaSwag-da", location=DATASET_PATH),
        solver=generate(),
        scorer=hellaswag_da_scorer(),
        config=GenerateConfig(max_tokens=8, temperature=0.0, timeout=120),
    )
