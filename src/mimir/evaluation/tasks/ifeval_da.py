from __future__ import annotations

from importlib.util import find_spec
from typing import Any, cast

import numpy as np
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState, generate

DATASET_PATH = "danish-foundation-models/ifeval-da"
INSTALL_URL = "https://github.com/josejg/instruction_following_eval"
_CONSTRAINED_RESPONSE_FALLBACK_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)


@task(name="ifeval-da")
def ifeval_da(
    dataset_path: str = DATASET_PATH,
    split: str = "train",
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
) -> Task:
    _require_optional_dependency(
        package="instruction_following_eval",
        task_name="ifeval-da",
        install_url=INSTALL_URL,
    )

    return Task(
        dataset=hf_dataset(
            path=dataset_path,
            split=split,
            sample_fields=record_to_sample,
            auto_id=True,
            shuffle=shuffle,
            seed=seed,
            limit=limit,
        ),
        solver=[generate()],
        scorer=instruction_following(),
    )


@metric
def if_metric() -> Metric:
    def _final_accuracy_stderr(
        scores: list[SampleScore], mean_final_accuracy: float
    ) -> float:
        total_num_instructions = int(
            sum(
                cast(dict[str, Any], score.score.value)["num_instructions"]
                for score in scores
            )
        )
        mean_num_instructions = total_num_instructions / len(scores)
        variance = 0.0
        cluster_count = len(scores)
        for score in scores:
            value = cast(dict[str, Any], score.score.value)
            inst_level_strict = int(value["inst_level_strict"])
            inst_level_loose = int(value["inst_level_loose"])
            prompt_level_strict = int(value["prompt_level_strict"])
            prompt_level_loose = int(value["prompt_level_loose"])
            num_instructions = int(value["num_instructions"])

            loose_only = inst_level_loose - inst_level_strict
            num_incorrect = int(num_instructions - inst_level_loose)
            prompt_adjustment = (
                0.25
                * (prompt_level_strict + prompt_level_loose)
                * mean_num_instructions
                / num_instructions
            )
            vector = [
                (0.5 + prompt_adjustment - mean_final_accuracy) * inst_level_strict,
                (0.25 + prompt_adjustment - mean_final_accuracy) * loose_only,
                (0.0 + prompt_adjustment - mean_final_accuracy) * num_incorrect,
            ]
            variance += np.outer(vector, vector).sum()

        stderr = (
            np.sqrt(variance * cluster_count / (cluster_count - 1))
            / total_num_instructions
            if cluster_count > 1
            else 0.0
        )

        return float(stderr)

    def metric_fn(scores: list[SampleScore]) -> Value:
        statistics: list[float] = []
        prompt_keys = ["prompt_level_strict", "prompt_level_loose"]
        instruct_keys = ["inst_level_strict", "inst_level_loose"]
        final_keys = [
            "prompt_strict_acc",
            "prompt_strict_stderr",
            "prompt_loose_acc",
            "prompt_loose_stderr",
            "inst_strict_acc",
            "inst_strict_stderr",
            "inst_loose_acc",
            "inst_loose_stderr",
            "final_acc",
            "final_stderr",
        ]

        for key in prompt_keys:
            score_list = [cast(dict[str, Any], score.score.value)[key] for score in scores]
            statistics.append(np.mean(score_list).item())
            stderr = (
                np.std(score_list, ddof=1).item() / np.sqrt(len(score_list))
                if len(score_list) > 1
                else 0.0
            )
            statistics.append(stderr)

        for key in instruct_keys:
            flattened: list[bool] = []
            for score in scores:
                value = cast(dict[str, Any], score.score.value)
                num_correct = int(value[key])
                num_incorrect = int(value["num_instructions"] - value[key])
                flattened.extend([True] * num_correct + [False] * num_incorrect)

            mean = np.mean(flattened).item()
            statistics.append(mean)

            variance = 0.0
            cluster_count = len(scores)
            for score in scores:
                value = cast(dict[str, Any], score.score.value)
                num_correct = int(value[key])
                num_incorrect = int(value["num_instructions"] - value[key])
                vector = [num_correct * (1 - mean), num_incorrect * (0 - mean)]
                variance += np.outer(vector, vector).sum()

            stderr = (
                np.sqrt(variance * cluster_count / (cluster_count - 1)) / len(flattened)
                if cluster_count > 1
                else 0.0
            )
            statistics.append(stderr)

        statistics.append(
            np.mean([statistics[i] for i in range(0, len(statistics), 2)]).item()
        )
        statistics.append(_final_accuracy_stderr(scores, statistics[-1]))

        return {k: v for k, v in zip(final_keys, statistics, strict=True)}

    return metric_fn


@scorer(metrics=[if_metric()])
def instruction_following() -> Scorer:
    from instruction_following_eval.evaluation import (  # type: ignore
        InputExample,
        ensure_nltk_resource,
        test_instruction_following,
    )

    _patch_instruction_registry()
    ensure_nltk_resource()

    async def score(state: TaskState, target: Target) -> Score:
        eval_input = InputExample(
            key=state.sample_id,
            instruction_id_list=state.metadata["instruction_id_list"],
            prompt=state.metadata["prompt"],
            kwargs=state.metadata["kwargs"],
        )

        out_strict = test_instruction_following(
            eval_input, state.output.completion, strict=True
        )
        out_loose = test_instruction_following(
            eval_input, state.output.completion, strict=False
        )
        value = {
            "prompt_level_strict": out_strict.follow_all_instructions,
            "inst_level_strict": sum(out_strict.follow_instruction_list),
            "prompt_level_loose": out_loose.follow_all_instructions,
            "inst_level_loose": sum(out_loose.follow_instruction_list),
            "num_instructions": len(out_loose.follow_instruction_list),
        }

        return Score(
            value=value,
            answer=state.output.completion,
            explanation=" ".join(state.metadata["instruction_id_list"]),
        )

    return score


def record_to_sample(record: dict[str, Any]) -> Sample:
    prompt = record["prompt"]
    key = record["key"]
    instruction_id_list = record["instruction_id_list"]
    kwargs_list = record["kwargs"]
    assert isinstance(prompt, str)
    assert isinstance(key, (str, int))
    assert isinstance(instruction_id_list, list)
    assert isinstance(kwargs_list, list)
    sample_id = str(key)
    assert len(kwargs_list) <= len(instruction_id_list)

    # Some rows omit kwargs entries for instructions that do not require
    # arguments, so pad them with empty dicts to preserve per-instruction
    # alignment for the evaluation library.
    normalized_kwargs: list[Any] = list(kwargs_list)
    if len(normalized_kwargs) < len(instruction_id_list):
        normalized_kwargs.extend(
            {} for _ in range(len(instruction_id_list) - len(normalized_kwargs))
        )

    cleaned_kwargs: list[dict[str, Any]] = []
    for instruction_id, kwargs in zip(
        instruction_id_list, normalized_kwargs, strict=True
    ):
        assert isinstance(instruction_id, str)
        assert isinstance(kwargs, dict)
        cleaned_kwargs.append({k: v for k, v in kwargs.items() if v is not None})

    return Sample(
        id=sample_id,
        input=prompt,
        metadata={
            "prompt": prompt,
            "instruction_id_list": instruction_id_list,
            "kwargs": cleaned_kwargs,
        },
    )


def _require_optional_dependency(
    package: str, task_name: str, install_url: str | None = None
) -> None:
    if find_spec(package) is not None:
        return

    install_hint = (
        f" Install it from {install_url}."
        if install_url is not None
        else ""
    )
    raise RuntimeError(
        f"Task '{task_name}' requires optional dependency '{package}'.{install_hint}"
    )


def _patch_instruction_registry() -> None:
    from instruction_following_eval import (
        instructions as ifeval_instructions,
    )
    from instruction_following_eval import (
        instructions_registry as ifeval_instructions_registry,
    )

    class _LowercaseLettersChecker(ifeval_instructions.Instruction):
        def build_description(self, **_: Any) -> str:
            self._description_pattern = (
                "Your entire response should be in lowercase letters."
            )
            return self._description_pattern

        def get_instruction_args(self) -> dict[str, Any]:
            return {}

        def get_instruction_args_keys(self) -> list[str]:
            return []

        def check_following(self, value: str) -> bool:
            return value.islower()

    class _CapitalLettersChecker(ifeval_instructions.Instruction):
        def build_description(self, **_: Any) -> str:
            self._description_pattern = (
                "Your entire response should be in capital letters."
            )
            return self._description_pattern

        def get_instruction_args(self) -> dict[str, Any]:
            return {}

        def get_instruction_args_keys(self) -> list[str]:
            return []

        def check_following(self, value: str) -> bool:
            return value.isupper()

    class _ConstrainedResponseWithArgumentChecker(ifeval_instructions.Instruction):
        def build_description(
            self, *, options: list[str] | tuple[str, ...] | None = None, **_: Any
        ) -> str:
            parsed_options = (
                [str(item) for item in options if str(item).strip()]
                if options is not None
                else list(_CONSTRAINED_RESPONSE_FALLBACK_OPTIONS)
            )
            self._options = parsed_options
            self._description_pattern = (
                "Your response should include one of the allowed options."
            )
            return self._description_pattern

        def get_instruction_args(self) -> dict[str, Any]:
            return {"options": self._options}

        def get_instruction_args_keys(self) -> list[str]:
            return ["options"]

        def check_following(self, value: str) -> bool:
            stripped = value.strip()
            return any(option in stripped for option in self._options)

    class _ResponseLanguageChecker(ifeval_instructions.Instruction):
        def build_description(self, *, language: str | None = None, **_: Any) -> str:
            self._language = (language or "en").strip()
            self._description_pattern = (
                "Your ENTIRE response should be in the requested language."
            )
            return self._description_pattern

        def get_instruction_args(self) -> dict[str, Any]:
            return {"language": self._language}

        def get_instruction_args_keys(self) -> list[str]:
            return ["language"]

        def check_following(self, value: str) -> bool:
            import langdetect

            try:
                return langdetect.detect(value) == self._language
            except langdetect.LangDetectException:
                return False

    custom_registry: dict[str, type[Any]] = {
        "change_case:lowercase_letters": _LowercaseLettersChecker,
        "change_case:capital_letters": _CapitalLettersChecker,
        "detectable_format:constrained_response_with_argument": (
            _ConstrainedResponseWithArgumentChecker
        ),
    }

    for instruction_id, checker_cls in custom_registry.items():
        ifeval_instructions_registry.INSTRUCTION_DICT.setdefault(
            instruction_id, checker_cls
        )
    ifeval_instructions_registry.INSTRUCTION_DICT["language:response_language"] = (
        _ResponseLanguageChecker
    )
