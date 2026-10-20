#!/usr/bin/env python3
"""HellaSwag experiment: inspect_evals format (multiple_choice + system message) with 10-shot.

Tests whether Gemma 4's low HellaSwag score is caused by the number of shots
or the prompt format. Uses the same multiple_choice() solver and system message
as inspect_evals/hellaswag, but adds 10 few-shot examples.

Splits 10,042 samples across 8 chunks for 8 GPUs.

Tasks: hellaswag_ie_fmt_chunk0 .. hellaswag_ie_fmt_chunk7
"""

from typing import Any
from datasets import load_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message, generate, chain, Solver
from inspect_ai.model import GenerateConfig

SYSTEM_MESSAGE = """
Choose the most plausible continuation for the story.
"""

DATASET_PATH = "Rowan/hellaswag"
DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"

SHUFFLE_SEED = 4242
NUM_CHUNKS = 8


def _record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["ctx"],
        target=chr(ord("A") + int(record["label"])),
        choices=record["endings"],
        metadata=dict(source_id=record["source_id"]),
    )


def _load_chunk(chunk_idx: int) -> MemoryDataset:
    ds = load_dataset(DATASET_PATH, split="validation", revision=DATASET_REVISION)
    samples = [_record_to_sample(r) for r in ds]

    import random
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(samples)

    chunk_size = len(samples) // NUM_CHUNKS
    start = chunk_idx * chunk_size
    end = start + chunk_size if chunk_idx < NUM_CHUNKS - 1 else len(samples)
    chunk = samples[start:end]

    for i, s in enumerate(chunk):
        s.id = i

    return MemoryDataset(chunk)


def _build_fewshot_solver() -> Solver:
    train_ds = load_dataset(DATASET_PATH, split="train", revision=DATASET_REVISION)
    fewshot_samples = []
    for r in list(train_ds)[:10]:
        s = _record_to_sample(r)
        fewshot_samples.append(s)

    fewshot_text = "\n\n".join(
        f"{s.input}\n\n"
        + "\n".join(f"{chr(ord('A')+i)}) {c}" for i, c in enumerate(s.choices))
        + f"\n\nANSWER: {s.target}"
        for s in fewshot_samples
    )

    sys_msg = system_message(
        f"{SYSTEM_MESSAGE}\n\nHere are some examples:\n\n{fewshot_text}"
    )
    return chain([sys_msg, multiple_choice()])


@task
def hellaswag_ie_fmt_chunk0():
    return Task(
        dataset=_load_chunk(0),
        solver=_build_fewshot_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )


@task
def hellaswag_ie_fmt_chunk1():
    return Task(
        dataset=_load_chunk(1),
        solver=_build_fewshot_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )


@task
def hellaswag_ie_fmt_chunk2():
    return Task(
        dataset=_load_chunk(2),
        solver=_build_fewshot_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )


@task
def hellaswag_ie_fmt_chunk3():
    return Task(
        dataset=_load_chunk(3),
        solver=_build_fewshot_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )


@task
def hellaswag_ie_fmt_chunk4():
    return Task(
        dataset=_load_chunk(4),
        solver=_build_fewshot_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )


@task
def hellaswag_ie_fmt_chunk5():
    return Task(
        dataset=_load_chunk(5),
        solver=_build_fewshot_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )


@task
def hellaswag_ie_fmt_chunk6():
    return Task(
        dataset=_load_chunk(6),
        solver=_build_fewshot_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )


@task
def hellaswag_ie_fmt_chunk7():
    return Task(
        dataset=_load_chunk(7),
        solver=_build_fewshot_solver(),
        scorer=choice(),
        config=GenerateConfig(max_tokens=10, temperature=0.0),
    )
