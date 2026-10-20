"""Nordjylland News Summarization — Danish news summarization benchmark.

Dataset: alexandrainst/nordjylland-news-summarization (test split, 4178 samples)
Input: news article (text)
Target: summary (Danish)
Scoring: ROUGE-1, ROUGE-2, ROUGE-L (F1), chrF

Loaded directly from parquet to avoid HF datasets schema mismatch.
Set NORDJYLLAND_DATASET_PATH env var to override the default location.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import generate
import pyarrow.parquet as pq

DEFAULT_MAX_GEN_TOKS = 256
DEFAULT_LIMIT = 500
DEFAULT_SHUFFLE_SEED = 4242
_DEFAULT_DATASET_PATH = Path(
    os.environ.get(
        "NORDJYLLAND_DATASET_PATH",
        str(Path.home() / ".cache/huggingface/hub/"
            "datasets--alexandrainst--nordjylland-news-summarization/"
            "snapshots/a7b9fe912984f4f6e2b74a5387f3e9afc097500d/data"),
    )
)

PROMPT_TEMPLATE = (
    "Skriv et kort resumé af følgende nyhedsartikel på dansk. "
    "Resuméet skal være koncis og dække hovedpointen i artiklen.\n\n"
    "Artikel:\n{text}\n\nResumé:"
)


@task(name="nordjylland-news-summarization")
def nordjylland_news_summarization(
    dataset_path: str | Path = _DEFAULT_DATASET_PATH,
    split: str = "test",
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    limit: int | None = DEFAULT_LIMIT,
    shuffle: bool = True,
    seed: int = DEFAULT_SHUFFLE_SEED,
) -> Task:
    samples = _load_samples(dataset_path, split, limit, shuffle, seed)
    return Task(
        dataset=MemoryDataset(samples=samples, name="nordjylland-news-summarization"),
        solver=[generate(max_tokens=max_gen_toks, temperature=0.0)],
        scorer=[rouge_scorer()],
    )


def _load_samples(
    path: str | Path, split: str, limit: int | None, shuffle: bool, seed: int
) -> list[Sample]:
    path = Path(path)
    pattern = str(path / f"{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' at {pattern}")

    all_records: list[dict[str, Any]] = []
    for f in files:
        table = pq.read_table(f)
        df = table.to_pandas()
        for _, row in df.iterrows():
            all_records.append({"text": row["text"], "summary": row["summary"]})

    if shuffle:
        import random
        random.Random(seed).shuffle(all_records)
    if limit is not None:
        all_records = all_records[:limit]

    return [
        Sample(
            id=str(i),
            input=PROMPT_TEMPLATE.format(text=rec["text"]),
            target=[rec["summary"]],
            metadata={"text": rec["text"]},
        )
        for i, rec in enumerate(all_records)
    ]


def _normalize(text: str) -> list[str]:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _rouge_n(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    from collections import Counter
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    pred_ngrams = Counter(ngrams(pred_tokens, n))
    ref_ngrams = Counter(ngrams(ref_tokens, n))

    overlap = sum((pred_ngrams & ref_ngrams).values())
    pred_count = sum(pred_ngrams.values())
    ref_count = sum(ref_ngrams.values())

    if pred_count == 0 or ref_count == 0:
        return 0.0

    precision = overlap / pred_count
    recall = overlap / ref_count

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_l(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def _chrf(pred: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF
    metric = CHRF(char_order=6, word_order=2, beta=3)
    return metric.sentence_score(hypothesis=pred, references=[ref]).score / 100.0


@scorer(
    metrics={
        "rouge1": [mean(), stderr()],
        "rouge2": [mean(), stderr()],
        "rougeL": [mean(), stderr()],
        "chrf": [mean(), stderr()],
    }
)
def rouge_scorer() -> Scorer:
    async def score(state, target: Target) -> Score:
        pred = state.output.completion.strip()
        references = [item.strip() for item in target if item.strip()]
        if not references:
            return Score(
                value={"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "chrf": 0.0},
                answer=pred,
            )

        ref = references[0]
        pred_tokens = _normalize(pred)
        ref_tokens = _normalize(ref)

        return Score(
            value={
                "rouge1": _rouge_n(pred_tokens, ref_tokens, 1),
                "rouge2": _rouge_n(pred_tokens, ref_tokens, 2),
                "rougeL": _rouge_l(pred_tokens, ref_tokens),
                "chrf": _chrf(pred, ref),
            },
            answer=pred,
            metadata={"prediction": pred, "reference": ref},
        )

    return score
