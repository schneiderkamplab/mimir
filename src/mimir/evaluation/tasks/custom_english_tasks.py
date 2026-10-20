#!/usr/bin/env python3
"""Custom English benchmark tasks matching HRM-Text original evaluation methodology.

Few-shot counts match the original HRM-Text evaluation code:
  - MMLU:       5-shot (per-subject dev)
  - ARC-C:      25-shot (validation)
  - HellaSwag:  10-shot (train)
  - Winogrande: 5-shot (train, winogrande_debiased)
  - BoolQ:      5-shot (train)
  - GSM8K:      0-shot
  - MATH:       0-shot
  - DROP:       3-shot (hardcoded examples)

Prompt format: "Question\nA. choice1\nB. choice2\n...\nAnswer:"
MCQ scoring: max_tokens=1, partial credit 1/N for invalid, 1 for correct, 0 for wrong
GSM8K scoring: extract boxed or last number, compare as int
MATH scoring: math_verify parse + verify
DROP scoring: EM + F1 with SQuAD normalization, answer extracted via regex
"""

import re
import random
import string
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Any

from datasets import load_dataset, get_dataset_config_names
from math_verify import parse as mv_parse, verify as mv_verify

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, accuracy, scorer, Target, mean
from inspect_ai.solver import generate, TaskState
from inspect_ai.model import GenerateConfig

logger = logging.getLogger(__name__)


# === Helper functions (matching original HRM-Text benchmarks.py) ===

def last_boxed_only_string(text: str) -> str | None:
    """Extract the content of the last \\boxed{} in text."""
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


@dataclass
class MCQDoc:
    query: str
    choices: list[str]
    gold_index: int


def format_mcq(doc: MCQDoc, include_gold: bool) -> tuple[str, list[str]]:
    text = doc.query.strip() + "\n"
    choices = []
    for idx, choice in enumerate(doc.choices):
        letter = chr(ord("A") + idx)
        text += f"{letter}. {choice.strip()}\n"
        choices.append(letter)
    text += f"Answer: {chr(ord('A') + doc.gold_index)}" if include_gold else "Answer:"
    return text, choices


def construct_mcq_prompt(target: MCQDoc, shots: list[MCQDoc]) -> tuple[str, list[str]]:
    prompt = ""
    for shot in shots:
        shot_text, _ = format_mcq(shot, include_gold=True)
        prompt += shot_text + "\n\n"
    target_text, choices = format_mcq(target, include_gold=False)
    return prompt + target_text, choices


# === Custom scorers ===

@scorer(metrics=[accuracy()])
def mcq_scorer():
    """MCQ scorer with partial credit 1/N for invalid answers (matching original)."""
    async def score(state: TaskState, target: Target):
        pred = state.output.completion.strip().upper()
        if len(pred) > 0:
            pred = pred[0]
        gold = target.text.upper()
        valid_set = set(state.metadata.get("valid_set", [gold]))
        num_choices = state.metadata.get("num_choices", len(valid_set))
        if pred not in valid_set:
            return Score(value=1.0 / num_choices, answer=pred)
        elif pred == gold:
            return Score(value=1.0, answer=pred)
        else:
            return Score(value=0.0, answer=pred)
    return score


@scorer(metrics=[accuracy()])
def gsm8k_scorer():
    """GSM8K scorer: extract boxed or last number, compare as int."""
    async def score(state: TaskState, target: Target):
        text = state.output.completion
        boxed = last_boxed_only_string(text)
        if boxed:
            text = boxed
        else:
            numbers = re.findall(r"-?\d[\d,]*", text)
            if numbers:
                text = numbers[-1]
        text = text.replace(",", "").replace("$", "").strip()
        try:
            ans = int(float(text))
        except (ValueError, OverflowError):
            return Score(value=0.0, answer="invalid")
        gt = int(target.text)
        return Score(value=1.0 if ans == gt else 0.0, answer=str(ans))
    return score


@scorer(metrics=[accuracy()])
def math_scorer():
    """MATH scorer using math_verify parse + verify."""
    async def score(state: TaskState, target: Target):
        text = state.output.completion
        ans = last_boxed_only_string(text)
        if ans is None:
            ans = text.strip()
        gt = target.text
        try:
            result = mv_verify(mv_parse(gt), mv_parse(ans))
            return Score(value=1.0 if result else 0.0, answer=ans)
        except Exception:
            return Score(value=0.0, answer=ans)
    return score


# === Task definitions ===

@task
def mmlu_5shot() -> Task:
    """MMLU 5-shot (per-subject dev shots). Full test split, shuffled."""
    num_shots = 5
    shuffle_seed = 4242

    shot_ds = load_dataset("cais/mmlu", "all", split="dev")
    shot_docs: dict[str, list[MCQDoc]] = defaultdict(list)
    for row in shot_ds:
        shot_docs[row["subject"]].append(MCQDoc(row["question"], row["choices"], row["answer"]))

    ds = load_dataset("cais/mmlu", "all", split="test")
    all_rows = list(ds)
    rng = random.Random(shuffle_seed)
    rng.shuffle(all_rows)

    samples = []
    for row in all_rows:
        subject = row["subject"]
        doc = MCQDoc(row["question"], row["choices"], row["answer"])
        prompt, choices = construct_mcq_prompt(doc, shot_docs[subject][:num_shots])
        gold = chr(ord("A") + doc.gold_index)
        samples.append(Sample(
            input=prompt,
            target=gold,
            metadata={"valid_set": choices, "num_choices": len(choices), "subject": subject},
            id=len(samples),
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=mcq_scorer(),
        config=GenerateConfig(max_tokens=1, temperature=0.0, timeout=120),
    )


@task
def arc_25shot() -> Task:
    """ARC-Challenge 25-shot (validation)."""
    num_shots = 25

    shot_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    shot_docs = [
        MCQDoc(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))
        for r in shot_ds
    ]

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    samples = []
    for row in ds:
        doc = MCQDoc(row["question"], row["choices"]["text"], row["choices"]["label"].index(row["answerKey"]))
        prompt, choices = construct_mcq_prompt(doc, shot_docs[:num_shots])
        gold = chr(ord("A") + doc.gold_index)
        samples.append(Sample(
            input=prompt,
            target=gold,
            metadata={"valid_set": choices, "num_choices": len(choices)},
            id=len(samples),
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=mcq_scorer(),
        config=GenerateConfig(max_tokens=1, temperature=0.0, timeout=120),
    )


@task
def hellaswag_10shot() -> Task:
    """HellaSwag 10-shot (train)."""
    num_shots = 10

    shot_ds = load_dataset("Rowan/hellaswag", split="train")
    shot_docs = [
        MCQDoc(
            f"{r['ctx_a']} {r['ctx_b'].capitalize()}\nQuestion: Which is the most logical continuation?",
            r["endings"],
            int(r["label"]),
        )
        for r in shot_ds.select(range(num_shots))
    ]

    ds = load_dataset("Rowan/hellaswag", split="validation")
    samples = []
    for row in ds:
        doc = MCQDoc(
            f"{row['ctx_a']} {row['ctx_b'].capitalize()}\nQuestion: Which is the most logical continuation?",
            row["endings"],
            int(row["label"]),
        )
        prompt, choices = construct_mcq_prompt(doc, shot_docs)
        gold = chr(ord("A") + doc.gold_index)
        samples.append(Sample(
            input=prompt,
            target=gold,
            metadata={"valid_set": choices, "num_choices": len(choices)},
            id=len(samples),
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=mcq_scorer(),
        config=GenerateConfig(max_tokens=1, temperature=0.0, timeout=120),
    )


@task
def winogrande_5shot() -> Task:
    """Winogrande 5-shot (train, winogrande_debiased)."""
    num_shots = 5

    shot_ds = load_dataset("allenai/winogrande", "winogrande_debiased", split=f"train[:{num_shots}]")
    shot_docs = [
        MCQDoc(r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1)
        for r in shot_ds
    ]

    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
    samples = []
    for row in ds:
        doc = MCQDoc(row["sentence"], [row["option1"], row["option2"]], int(row["answer"]) - 1)
        prompt, choices = construct_mcq_prompt(doc, shot_docs)
        gold = chr(ord("A") + doc.gold_index)
        samples.append(Sample(
            input=prompt,
            target=gold,
            metadata={"valid_set": choices, "num_choices": len(choices)},
            id=len(samples),
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=mcq_scorer(),
        config=GenerateConfig(max_tokens=1, temperature=0.0, timeout=120),
    )


@task
def boolq_5shot() -> Task:
    """BoolQ 5-shot (train)."""
    num_shots = 5

    shot_ds = load_dataset("google/boolq", split=f"train[:{num_shots}]")
    shot_docs = [
        MCQDoc(
            f"{r['passage']}\nQuestion: {r['question'].capitalize()}?",
            ["Yes", "No"],
            1 - int(bool(r["answer"])),
        )
        for r in shot_ds
    ]

    ds = load_dataset("google/boolq", split="validation")
    samples = []
    for row in ds:
        doc = MCQDoc(
            f"{row['passage']}\nQuestion: {row['question'].capitalize()}?",
            ["Yes", "No"],
            1 - int(bool(row["answer"])),
        )
        prompt, choices = construct_mcq_prompt(doc, shot_docs)
        gold = chr(ord("A") + doc.gold_index)
        samples.append(Sample(
            input=prompt,
            target=gold,
            metadata={"valid_set": choices, "num_choices": len(choices)},
            id=len(samples),
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=mcq_scorer(),
        config=GenerateConfig(max_tokens=1, temperature=0.0, timeout=120),
    )


@task
def gsm8k_0shot() -> Task:
    """GSM8K 0-shot."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = []
    for row in ds:
        gt = int(row["answer"].split("####")[-1].strip().replace(",", ""))
        samples.append(Sample(
            input=row["question"],
            target=str(gt),
            id=len(samples),
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=gsm8k_scorer(),
        config=GenerateConfig(max_tokens=1024, temperature=0.0, timeout=600),
    )


@task
def math_0shot() -> Task:
    """MATH 0-shot (EleutherAI/hendrycks_math, math_verify scoring). Full test split, shuffled."""
    shuffle_seed = 4242

    samples = []
    for subset in get_dataset_config_names("EleutherAI/hendrycks_math"):
        ds = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
        for row in ds:
            label = last_boxed_only_string(row["solution"])
            if label is None:
                continue
            samples.append(Sample(
                input=row["problem"],
                target=label,
                id=len(samples),
            ))

    rng = random.Random(shuffle_seed)
    rng.shuffle(samples)
    for i, s in enumerate(samples):
        s.id = i

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=math_scorer(),
        config=GenerateConfig(max_tokens=2048, temperature=0.0, timeout=600),
    )


# === DROP ===

DROP_EXAMPLES = [
    "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007\nQ: How many points did the buccaneers need to tie in the first?\nA: 3",
    "Trying to snap a two-game skid, the Bills flew to Gillette Stadium for a Week 3 divisional fight with the New England Patriots. In the first quarter, QB J. P. Losman was immediately injured on the first offensive play of the game. He would finish the series, but ended up on the bench for the rest of the game. After New England took the lead with kicker Stephen Gostkowski's 24-yard field goal, rookie QB Trent Edwards played the rest of the game for Buffalo. The Bills would get their only score of the game as RB Marshawn Lynch got an 8-yard TD run, and a Rian Lindell extra point put the Bills ahead surprisingly 7-3. However, in the second quarter, the Patriots were able to open up their running game when Bills rookie standout Paul Posluszny was lost due to a broken arm. This left passing lanes open, and for the rest of the game, the Patriots dominated. QB Tom Brady's 8-yard TD pass to TE Benjamin Watson and a 3-yard TD pass to WR Randy Moss made it 17-7 at the half. In the third quarter, New England continued its conquest with Brady's 4-yard TD pass to WR Jabar Gaffney and RB Sammy Morris' 4-yard TD run. In the fourth quarter, the Patriots ended the day with Brady and Moss hooking up with each other again on a 45-yard TD pass.\nQ: How many games had the Bills won before this game?\nA: 0",
    "The French king, John II, had been held captive in England. The Treaty of Brétigny set his ransom at 3 million crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.\nQ: How many years before Navarrase used the captivity of John II?\nA: 9",
]


def _normalize_answer(s: str) -> str:
    """SQuAD-style answer normalization (from lm_eval DROP utils)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _em_score(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def _get_answers_drop(record: dict) -> list[str]:
    """Extract valid answers from DROP record (matching lm_eval format)."""
    def _flatten_validated_answers(validated_answers: dict) -> list[dict]:
        valid_answers = []
        for i in range(len(validated_answers["number"])):
            valid_answers.append({
                "number": validated_answers["number"][i],
                "date": validated_answers["date"][i],
                "spans": validated_answers["spans"][i],
            })
        return valid_answers

    def _parse_answer(answer: dict) -> str:
        if answer["number"] != "":
            return str(answer["number"])
        if answer["spans"] != []:
            return ",".join(answer["spans"])
        return " ".join([
            answer["date"]["day"],
            answer["date"]["month"],
            answer["date"]["year"],
        ]).strip()

    answers: list[str] = []
    answers_set: set[str] = set()
    candidates = [record["answer"]] + _flatten_validated_answers(record["validated_answers"])
    for candidate in candidates:
        ans = _parse_answer(candidate)
        if ans in answers_set:
            continue
        answers_set.add(ans)
        answers.append(ans)
    return answers


@scorer(metrics=[mean()])
def drop_f1_scorer():
    """DROP F1 scorer: extract 'Answer: X' from generation, compute max F1 over gold answers."""
    async def score(state: TaskState, target: Target):
        text = state.output.completion.strip()
        match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", text)
        pred = match.group(1).strip() if match else text
        gold_answers = target.target if isinstance(target.target, list) else [target.text]
        best = max((_f1_score(pred, g) for g in gold_answers), default=0.0)
        return Score(value=best, answer=pred)
    return score


@scorer(metrics=[mean()])
def drop_em_scorer():
    """DROP EM scorer: extract 'Answer: X' from generation, compute max EM over gold answers."""
    async def score(state: TaskState, target: Target):
        text = state.output.completion.strip()
        match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", text)
        pred = match.group(1).strip() if match else text
        gold_answers = target.target if isinstance(target.target, list) else [target.text]
        best = max((_em_score(pred, g) for g in gold_answers), default=0.0)
        return Score(value=best, answer=pred)
    return score


# === HumanEval ===

HUMANEVAL_INSTRUCTION = """Read the following function signature and docstring, and fully implement
the function described. Your response should only contain the code for
this function.\n
"""


def extract_code(completion: str) -> str:
    """Extract code from model completion, handling markdown blocks and function signature repetition."""
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
        passed = result.get("passed", False)
        return Score(
            value=1.0 if passed else 0.0,
            answer=completion,
            explanation=result.get("result", ""),
        )

    return score


@task
def humaneval() -> Task:
    """HumanEval: Hand-Written Evaluation Set (164 problems, local code execution scoring)."""
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


@task
def humaneval_passthrough() -> Task:
    """HumanEval for passthrough models — raw code completion, no instruction prefix."""
    from human_eval.data import read_problems

    problems = read_problems()
    samples = []
    for task_id, problem in problems.items():
        samples.append(Sample(
            id=task_id,
            input=problem["prompt"],
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


@task
def drop_3shot() -> Task:
    """DROP 3-shot (hardcoded examples matching benchmarks.py). Full validation split, shuffled."""
    shuffle_seed = 4242

    prefix = "".join(f"{shot}\n\n" for shot in DROP_EXAMPLES[:3])

    ds = load_dataset("EleutherAI/drop", split="validation")
    all_records = list(ds)
    rng = random.Random(shuffle_seed)
    rng.shuffle(all_records)

    samples = []
    for record in all_records:
        answers = _get_answers_drop(record)
        prompt = f"{prefix}{record['passage']}\nQ: {record['question']}\nA:"
        samples.append(Sample(
            input=prompt,
            target=answers,
            id=len(samples),
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=[drop_f1_scorer(), drop_em_scorer()],
        config=GenerateConfig(max_tokens=256, temperature=0.0, timeout=600, stop_seqs=["\n\n"]),
    )


# === GovReport Summarization ===

GOVREPORT_PROMPT = "Summarize the following government report. Provide a concise summary covering the main points.\n\nReport:\n{report}\n\nSummary:"


def _normalize_summ(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _rouge_n_summ(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
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


def _rouge_l_summ(pred_tokens: list[str], ref_tokens: list[str]) -> float:
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


def _chrf_summ(pred: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF
    metric = CHRF(char_order=6, word_order=2, beta=3)
    return metric.sentence_score(hypothesis=pred, references=[ref]).score / 100.0


@scorer(metrics={"rouge1": [mean()], "rouge2": [mean()], "rougeL": [mean()], "chrf": [mean()]})
def govreport_scorer():
    async def score(state: TaskState, target: Target):
        pred = state.output.completion.strip()
        ref = target.text.strip()
        pred_tokens = _normalize_summ(pred)
        ref_tokens = _normalize_summ(ref)
        return Score(
            value={
                "rouge1": _rouge_n_summ(pred_tokens, ref_tokens, 1),
                "rouge2": _rouge_n_summ(pred_tokens, ref_tokens, 2),
                "rougeL": _rouge_l_summ(pred_tokens, ref_tokens),
                "chrf": _chrf_summ(pred, ref),
            },
            answer=pred,
        )
    return score


@task
def govreport_summarization() -> Task:
    """GovReport: Government report summarization (ccdv/govreport-summarization, test split, 500 sampled)."""
    from datasets import load_dataset

    ds = load_dataset("ccdv/govreport-summarization", split="test")
    all_rows = list(ds)
    rng = random.Random(4242)
    rng.shuffle(all_rows)

    samples = []
    for i, row in enumerate(all_rows):
        report = row["report"]
        report = report[:7000]
        prompt = GOVREPORT_PROMPT.format(report=report)
        samples.append(Sample(
            id=i,
            input=prompt,
            target=row["summary"],
        ))

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=govreport_scorer(),
        config=GenerateConfig(max_tokens=512, temperature=0.0, timeout=600),
    )
