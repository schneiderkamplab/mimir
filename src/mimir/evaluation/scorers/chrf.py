from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from sacrebleu.metrics import CHRF


@scorer(metrics=[mean(), stderr()])
def chrf3pp() -> Scorer:
    """Score translations with sentence-level chrF3++.

    This matches EuroEval's translation metric configuration:
    ``CHRF(char_order=6, word_order=2, beta=3)``. SacreBLEU reports chrF scores
    as percentages on a 0-100 scale; this scorer normalizes to 0-1 to match the
    rest of the eval suite. Higher is better.
    """
    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion
        references = [reference for reference in target.target if reference]
        return Score(
            value=compute_chrf3pp(prediction, references),
            answer=prediction,
        )

    return score


def compute_chrf3pp(prediction: str, references: list[str]) -> float:
    """Compute sentence-level chrF3++ for one prediction on a 0-1 scale."""
    if not references:
        return 0.0

    metric = CHRF(char_order=6, word_order=2, beta=3)
    return metric.sentence_score(
        hypothesis=prediction,
        references=references,
    ).score / 100.0
