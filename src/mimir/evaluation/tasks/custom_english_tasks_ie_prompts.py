#!/usr/bin/env python3
"""Custom English tasks with inspect_evals-style prompts for GSM8K, MATH, DROP.

Separate file so original tasks remain untouched. Uses IE prompt format:
- "Solve step by step... ANSWER: $ANSWER" for GSM8K and MATH
- "Think step by step, then write Answer: $ANSWER" for DROP

Scorers updated to extract "ANSWER: X" / "Answer: X" first, then fall back
to original extraction (\boxed, last number).
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
from inspect_ai.scorer import Score, accuracy, mean, scorer, Target
from inspect_ai.solver import generate, TaskState
from inspect_ai.model import GenerateConfig

logger = logging.getLogger(__name__)


# === Shared helpers ===

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
    """Extract 'ANSWER: X' or 'Answer: X' from model output."""
    for pattern in [r"ANSWER:\s*(.+?)(?:\n|$)", r"Answer:\s*(.+?)(?:\n|$)"]:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()
    return None


# === SQuAD normalization (for DROP) ===

def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


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


# === DROP answer extraction ===

def _get_answers_drop(record: dict) -> list[str]:
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


# === Scorers ===

@scorer(metrics=[accuracy()])
def gsm8k_ie_scorer():
    """GSM8K scorer: extract 'ANSWER: X' first, then \boxed, then last number."""
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
                numbers = re.findall(r"-?\d[\d,]*", text)
                if numbers:
                    extracted = numbers[-1]
                else:
                    return Score(value=0.0, answer="invalid")
        extracted = extracted.replace(",", "").replace("$", "").strip()
        try:
            ans_int = int(float(extracted))
        except (ValueError, OverflowError):
            return Score(value=0.0, answer="invalid")
        gt = int(target.text)
        return Score(value=1.0 if ans_int == gt else 0.0, answer=str(ans_int))
    return score


@scorer(metrics=[accuracy()])
def math_ie_scorer():
    """MATH scorer: extract 'ANSWER: X' first, then \boxed, then math_verify."""
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


@scorer(metrics=[mean()])
def drop_f1_ie_scorer():
    """DROP F1: extract 'Answer: X' from generation, max F1 over gold answers."""
    async def score(state: TaskState, target: Target):
        text = state.output.completion.strip()
        pred = extract_answer_tag(text) or text
        gold_answers = target.target if isinstance(target.target, list) else [target.text]
        best = max((_f1_score(pred, g) for g in gold_answers), default=0.0)
        return Score(value=best, answer=pred)
    return score


@scorer(metrics=[mean()])
def drop_em_ie_scorer():
    """DROP EM: extract 'Answer: X' from generation, max EM over gold answers."""
    async def score(state: TaskState, target: Target):
        text = state.output.completion.strip()
        pred = extract_answer_tag(text) or text
        gold_answers = target.target if isinstance(target.target, list) else [target.text]
        best = max((_em_score(pred, g) for g in gold_answers), default=0.0)
        return Score(value=best, answer=pred)
    return score


# === IE-style prompt templates ===

GSM8K_IE_PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

MATH_IE_PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command."""

DROP_IE_SYSTEM = """You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.

Examples
{examples}"""

DROP_IE_USER = """Your Task
---
{prompt}

Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response."""


# === DROP few-shot examples (same as original) ===

DROP_EXAMPLES = [
    "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007\nQ: How many points did the buccaneers need to tie in the first?\nA: 3",
    "Trying to snap a two-game skid, the Bills flew to Gillette Stadium for a Week 3 divisional fight with the New England Patriots. In the first quarter, QB J. P. Losman was immediately injured on the first offensive play of the game. He would finish the series, but ended up on the bench for the rest of the game. After New England took the lead with kicker Stephen Gostkowski's 24-yard field goal, rookie QB Trent Edwards played the rest of the game for Buffalo. The Bills would get their only score of the game as RB Marshawn Lynch got an 8-yard TD run, and a Rian Lindell extra point put the Bills ahead surprisingly 7-3. However, in the second quarter, the Patriots were able to open up their running game when Bills rookie standout Paul Posluszny was lost due to a broken arm. This left passing lanes open, and for the rest of the game, the Patriots dominated. QB Tom Brady's 8-yard TD pass to TE Benjamin Watson and a 3-yard TD pass to WR Randy Moss made it 17-7 at the half. In the third quarter, New England continued its conquest with Brady's 4-yard TD pass to WR Jabar Gaffney and RB Sammy Morris' 4-yard TD run. In the fourth quarter, the Patriots ended the day with Brady and Moss hooking up with each other again on a 45-yard TD pass.\nQ: How many games had the Bills won before this game?\nA: 0",
    "The French king, John II, had been held captive in England. The Treaty of Brétigny set his ransom at 3 million crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.\nQ: How many years before Navarrase used the captivity of John II?\nA: 9",
]


# === Task definitions ===

@task
def gsm8k_0shot_ie() -> Task:
    """GSM8K 0-shot with IE-style prompt."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = []
    for row in ds:
        gt = int(row["answer"].split("####")[-1].strip().replace(",", ""))
        prompt = GSM8K_IE_PROMPT.format(question=row["question"])
        samples.append(Sample(
            input=prompt,
            target=str(gt),
            id=len(samples),
        ))
    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=gsm8k_ie_scorer(),
        config=GenerateConfig(max_tokens=1024, temperature=0.0, timeout=600),
    )


@task
def math_0shot_ie() -> Task:
    """MATH 0-shot with IE-style prompt (EleutherAI/hendrycks_math, math_verify scoring)."""
    shuffle_seed = 4242
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
    rng = random.Random(shuffle_seed)
    rng.shuffle(samples)
    for i, s in enumerate(samples):
        s.id = i
    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=math_ie_scorer(),
        config=GenerateConfig(max_tokens=2048, temperature=0.0, timeout=600),
    )


@task
def drop_3shot_ie() -> Task:
    """DROP 3-shot with IE-style prompt. Full validation split, shuffled."""
    shuffle_seed = 4242

    examples_str = "\n\n".join(
        f"Passage: {ex.split(chr(10) + 'Q: ')[0]}\nQuestion: {ex.split(chr(10) + 'Q: ')[1].split(chr(10) + 'A: ')[0]}\nAnswer: {ex.split(chr(10) + 'A: ')[1]}"
        for ex in DROP_EXAMPLES[:3]
    )
    system_msg = DROP_IE_SYSTEM.format(examples=examples_str)

    ds = load_dataset("EleutherAI/drop", split="validation")
    all_records = list(ds)
    rng = random.Random(shuffle_seed)
    rng.shuffle(all_records)

    samples = []
    for record in all_records:
        answers = _get_answers_drop(record)
        passage_q = f"Passage: {record['passage']}\nQuestion: {record['question']}\nAnswer:"
        user_msg = DROP_IE_USER.format(prompt=passage_q)
        full_input = f"{system_msg}\n\n{user_msg}"
        samples.append(Sample(
            input=full_input,
            target=answers,
            id=len(samples),
        ))
    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=[drop_f1_ie_scorer(), drop_em_ie_scorer()],
        config=GenerateConfig(max_tokens=256, temperature=0.0, timeout=600, stop_seqs=["\n\n"]),
    )
