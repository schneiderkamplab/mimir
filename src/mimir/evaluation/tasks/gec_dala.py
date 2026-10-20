from __future__ import annotations

from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

DEFAULT_HUGGING_FACE_ID = "giannor/dala_gen_v3"
DEFAULT_DATASET_NAME = "gec_dala"
DEFAULT_PROMPT_TEMPLATE = """Givet følgende sætning på dansk, afgør om den er grammatisk korrekt eller ukorrekt.
- Hvis den er ukorrekt, skal du outputte den rettede version.
- Hvis den er korrekt, skal du outputte den originale sætning.
- Output kun den rettede eller originale sætning — intet andet.

Regler:
- Svar kun på dansk.
- Bevar samme formatering og store/små bogstaver som i den originale sætning.
- Output ingen forklaring.
- Sig ikke, om sætningen er korrekt eller ukorrekt.

Sætning: {{corrupted}}
Svar:"""
DEFAULT_TRAINING_SPLIT = "train"
DEFAULT_VALIDATION_SPLIT = "val"
DEFAULT_TEST_SPLIT = "test"
DEFAULT_SPLIT = DEFAULT_TEST_SPLIT
DEFAULT_MAX_GEN_TOKS = 128
DEFAULT_TEMPERATURE = 0.1
DEFAULT_INPUT_FIELD = "corrupted"
DEFAULT_TARGET_FIELD = "original"

SplitName = Literal["train", "val", "test"]


@task(name="gec_dala")
def gec_dala(
    hugging_face_id: str = DEFAULT_HUGGING_FACE_ID,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    training_split: str = DEFAULT_TRAINING_SPLIT,
    validation_split: str = DEFAULT_VALIDATION_SPLIT,
    test_split: str = DEFAULT_TEST_SPLIT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    input_field: str = DEFAULT_INPUT_FIELD,
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

    if max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1.")

    normalized_split = _normalize_split_name(split)
    resolved_split = _resolve_split(
        split=normalized_split,
        training_split=training_split,
        validation_split=validation_split,
        test_split=test_split,
    )

    dataset_kwargs: dict[str, Any] = {
        "path": hugging_face_id,
        "split": resolved_split,
        "sample_fields": lambda record: record_to_sample(
            record=record,
            prompt_template=prompt_template,
            input_field=input_field,
            target_field=target_field,
        ),
        "auto_id": True,
        "shuffle": shuffle,
        "seed": seed,
        "limit": limit,
    }
    dataset_name = dataset_name.strip()
    if dataset_name:
        dataset_kwargs["name"] = dataset_name

    return Task(
        dataset=_load_hf_dataset(dataset_kwargs),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=[gec_dala_scorer()],
    )


@scorer(metrics={"exact_match": [mean(), stderr()]})
def gec_dala_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        references = [item.strip() for item in target if item.strip()]
        exact_match = float(prediction in references) if references else 0.0

        return Score(
            value={"exact_match": exact_match},
            answer=prediction,
            explanation=f"exact_match={exact_match}",
            metadata={
                "prediction": prediction,
                "targets": references,
            },
        )

    return score


def _load_hf_dataset(dataset_kwargs: dict[str, Any]) -> Any:
    try:
        return hf_dataset(**dataset_kwargs)
    except ValueError as exc:
        if "BuilderConfig" not in str(exc) or "name" not in dataset_kwargs:
            raise

        # Some datasets expose only the default config; retry without a named config.
        fallback_kwargs = dict(dataset_kwargs)
        fallback_kwargs.pop("name", None)
        return hf_dataset(**fallback_kwargs)


def _normalize_split_name(split: str) -> SplitName:
    normalized = split.strip().lower()
    match normalized:
        case "train" | "training":
            return "train"
        case "val" | "valid" | "validation" | "dev":
            return "val"
        case "test":
            return "test"
        case "":
            raise ValueError("`split` must be a non-empty string.")
        case _:
            raise ValueError(
                f"Unsupported split '{split}'. Supported values: "
                "['dev', 'test', 'train', 'training', 'val', 'valid', 'validation']"
            )


def _resolve_split(
    *, split: SplitName, training_split: str, validation_split: str, test_split: str
) -> str:
    split_map = {
        "train": training_split,
        "val": validation_split,
        "test": test_split,
    }
    resolved = split_map[split].strip()
    if not resolved:
        raise ValueError(f"Resolved split for '{split}' must be a non-empty string.")
    return resolved


def record_to_sample(
    *,
    record: dict[str, Any],
    prompt_template: str,
    input_field: str,
    target_field: str,
) -> Sample:
    sentence = _require_string(record, input_field)
    target = _require_string(record, target_field)

    return Sample(
        id=_sample_id(record),
        input=prompt_template.replace("{{corrupted}}", sentence),
        target=[target],
        metadata={
            "corrupted": sentence,
            "target": target,
            "corruption_type": record.get("corruption_type"),
            "affected_token_1": record.get("affected_token_1"),
            "affected_token_2": record.get("affected_token_2"),
        },
    )


def _sample_id(record: dict[str, Any]) -> str | None:
    raw_id = record.get("id")
    if raw_id is None:
        return None
    text_id = str(raw_id).strip()
    return text_id or None


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record field '{field}' must be a non-empty string.")
    return value
