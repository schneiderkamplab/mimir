#!/usr/bin/env python3
"""Chunked MATH 0-shot (IE prompt) + HumanEval 0-shot for 8 models.

MATH: 4 chunks (5,000 samples / 4 = 1,250 each), IE-style "ANSWER: $ANSWER" prompt, math_verify scorer, temp=0.
HumanEval: 164 samples, too small to chunk — run 1 per GPU but assign 2 models per GPU sequentially if needed.

Strategy: 8 models x 2 tasks = 16 evals. MATH chunks distributed to fill GPUs.
  Round 1 (8 GPUs): MATH chunk for 8 models (one each)
  Round 2 (8 GPUs): MATH chunk for models that need it + HumanEval
Actually simpler: 
  - MATH has 5000 samples. Split into 2 chunks per model = 2500 each.
  - 8 models x 2 MATH chunks = 16 MATH evals + 8 HumanEval = 24 total
  - Run in waves: 8 at a time, 3 waves = 24 evals

Better: 4 MATH chunks per model, run 8 models' chunk-0 first (8 GPUs),
then chunk-1, etc. HumanEval runs alongside (only 164 samples, fast).

Simplest approach: 
  GPU 0-7: each runs one model's MATH (full 5000, no chunking) + HumanEval (164)
  But MATH takes long. Let's chunk MATH into 2 and run HumanEval in parallel with chunk 0.

Final plan:
  Wave 1 (8 GPUs): MATH chunk0 for models 0-7
  Wave 2 (8 GPUs): MATH chunk1 for models 0-7 + HumanEval for models 0-7
  Actually HumanEval is only 164 samples and fast, run in wave 2 alongside MATH chunk1.

Even simpler: just run MATH full (no chunk) + HumanEval per model, one model per GPU.
  MATH at max_tokens=2048 for 5000 samples takes ~10 min for non-think, ~20 min for think.
  HumanEval at max_tokens=1024 for 164 samples takes ~2 min.
  Total: ~22 min, 1 wave of 8 GPUs.

Let's do that. 8 GPUs, 8 models, each runs MATH + HumanEval sequentially.
"""

import os
import re
import time
import signal
import shutil
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any

from datasets import load_dataset, get_dataset_config_names
from math_verify import parse as mv_parse, verify as mv_verify

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, accuracy, scorer, Target, mean
from inspect_ai.solver import generate, TaskState, prompt_template
from inspect_ai.model import GenerateConfig

import random

WORK_DIR = Path(__file__).parent

# ─── MATH helpers ────────────────────────────────────────────────────────────

def last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    for i in range(idx + 7, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[idx + 7 : i]
            depth -= 1
    return None


def extract_answer_tag(text: str) -> str | None:
    for pattern in [r"ANSWER:\s*(.+?)(?:\n|$)", r"Answer:\s*(.+?)(?:\n|$)"]:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()
    return None


MATH_IE_PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command."""


@scorer(metrics=[accuracy()])
def math_ie_scorer():
    """MATH scorer: extract 'ANSWER: X' first, then \\boxed, then math_verify."""
    async def score(state: TaskState, target: Target):
        text = state.output.completion
        ans = extract_answer_tag(text)
        if ans is not None:
            extracted = ans
        else:
            boxed = last_boxed_only_string(text)
            if boxed:
                extracted = boxed
            else:
                extracted = text.strip()
        gt = target.text
        try:
            result = mv_verify(mv_parse(gt), mv_parse(extracted))
            return Score(value=1.0 if result else 0.0, answer=extracted)
        except Exception:
            return Score(value=0.0, answer=extracted)
    return score


MATH_NUM_CHUNKS = 2
MATH_SHUFFLE_SEED = 4242


def _build_math_samples() -> list[Sample]:
    samples = []
    for subset in get_dataset_config_names("EleutherAI/hendrycks_math"):
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
        for row in ds:
            label = last_boxed_only_string(row["solution"])
            if label is None:
                continue
            prompt = MATH_IE_PROMPT.format(problem=row["problem"])
            samples.append(Sample(
                input=prompt,
                target=label,
                id=len(samples),
            ))
    rng = random.Random(MATH_SHUFFLE_SEED)
    rng.shuffle(samples)
    for i, s in enumerate(samples):
        s.id = i
    return samples


def _load_math_chunk(chunk_idx: int) -> MemoryDataset:
    samples = _build_math_samples()
    chunk_size = len(samples) // MATH_NUM_CHUNKS
    start = chunk_idx * chunk_size
    end = start + chunk_size if chunk_idx < MATH_NUM_CHUNKS - 1 else len(samples)
    chunk = samples[start:end]
    for i, s in enumerate(chunk):
        s.id = i
    return MemoryDataset(chunk)


@task
def math_ie_chunk0():
    return Task(
        dataset=_load_math_chunk(0),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=4096, temperature=0.0, timeout=600),
    )


@task
def math_ie_chunk1():
    return Task(
        dataset=_load_math_chunk(1),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=4096, temperature=0.0, timeout=600),
    )


@task
def math_ie_chunk2():
    return Task(
        dataset=_load_math_chunk(2),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=4096, temperature=0.0, timeout=600),
    )


@task
def math_ie_chunk3():
    return Task(
        dataset=_load_math_chunk(3),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=4096, temperature=0.0, timeout=600),
    )


@task
def math_ie_chunk4():
    return Task(
        dataset=_load_math_chunk(4),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=4096, temperature=0.0, timeout=600),
    )


@task
def math_ie_chunk5():
    return Task(
        dataset=_load_math_chunk(5),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=4096, temperature=0.0, timeout=600),
    )


@task
def math_ie_chunk6():
    return Task(
        dataset=_load_math_chunk(6),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=4096, temperature=0.0, timeout=600),
    )


@task
def math_ie_chunk7():
    return Task(
        dataset=_load_math_chunk(7),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=4096, temperature=0.0, timeout=600),
    )


# ─── HumanEval ───────────────────────────────────────────────────────────────

HUMANEVAL_INSTRUCTION = """Read the following function signature and docstring, and fully implement
the function described. Your response should only contain the code for
this function.\n
"""


def extract_code(completion: str) -> str:
    pattern_1 = re.compile(r"```python\n(.*?)```", re.DOTALL)
    pattern_2 = re.compile(r"```\n(.*?)```", re.DOTALL)
    matches = pattern_1.findall(completion) + pattern_2.findall(completion)
    if matches:
        code = matches[-1]
        if "def " in code:
            idx = code.find(":\n    ")
            if idx != -1:
                return code[idx + 2:]
        return code
    return completion.strip()


@scorer(metrics=[accuracy()])
def humaneval_scorer():
    """HumanEval scorer: extract code and verify via human_eval.check_correctness."""
    from human_eval.execution import check_correctness
    import asyncio

    async def score(state: TaskState, target: Target):
        completion = extract_code(state.output.completion)
        problem = {
            "prompt": state.metadata["prompt"],
            "test": state.metadata["test"],
            "entry_point": state.metadata["entry_point"],
            "task_id": state.metadata["task_id"],
        }
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, check_correctness, problem, completion, 30.0)
        passed = result.get("passed", False) and "passed" in result.get("result", "")
        if not passed:
            passed = result.get("result", "") == "passed"
        return Score(
            value=1.0 if passed else 0.0,
            answer=completion,
            explanation=result.get("result", ""),
        )

    return score


@task
def humaneval_ie():
    """HumanEval 0-shot with local code execution scoring."""
    from human_eval.data import read_problems

    problems = read_problems()
    samples = []
    for task_id, problem in problems.items():
        samples.append(Sample(
            id=task_id,
            input=HUMANEVAL_INSTRUCTION + problem["prompt"],
            target=problem["canonical_solution"],
            metadata={
                "prompt": problem["prompt"],
                "test": problem["test"],
                "entry_point": problem["entry_point"],
                "task_id": task_id,
            },
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=humaneval_scorer(),
        config=GenerateConfig(max_tokens=1024, temperature=0.0, timeout=300),
    )
