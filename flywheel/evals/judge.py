"""Blind pairwise judging by a model from a different family than the candidates.

Each pair runs in both orders to cancel position bias, and the judge never sees
model names. The prompt carries no reference, so this measures something chrF++
does not, and it still works on real inputs that have no reference.

Comparisons run on the Batches API. It costs half as much, and this work does not
need an answer quickly.
"""

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from flywheel.evals.translate import LANGUAGE_NAMES
from flywheel.schema import EvalSegment, JudgeVerdict
from flywheel.storage import read_jsonl, write_json

Letter = Literal["A", "B"]

MAX_TOKENS = 2048  # thinking is always on and counts against this
POLL_SECONDS = 60
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"better": {"type": "string", "enum": ["A", "B"]}},
    "required": ["better"],
    "additionalProperties": False,
}


class Comparison(NamedTuple):
    segment: EvalSegment
    first: str
    second: str
    first_output: str
    second_output: str

    @property
    def request_id(self) -> str:
        key = f"{self.segment.id}|{self.first}|{self.second}"
        return hashlib.sha256(key.encode()).hexdigest()[:40]


class Answer(NamedTuple):
    letter: Letter | None
    input_tokens: int = 0
    output_tokens: int = 0


class Judge(Protocol):
    def decide(self, comparisons: list[Comparison]) -> dict[str, Answer]: ...


def choose_segments(segments: Iterable[EvalSegment], count: int, seed: str) -> list[EvalSegment]:
    """A fixed sample: hashing by id keeps it stable if the set is reordered."""
    return sorted(segments, key=lambda s: hashlib.sha256(f"{seed}|{s.id}".encode()).digest())[
        :count
    ]


def build_prompt(comparison: Comparison) -> str:
    source = LANGUAGE_NAMES[comparison.segment.src_lang]
    target = LANGUAGE_NAMES[comparison.segment.tgt_lang]
    return (
        f"Two translations from {source} to {target} follow. Decide which one a fluent "
        f"{target} speaker would prefer. Accuracy of meaning comes first, then natural "
        "phrasing.\n\n"
        f"<source>{comparison.segment.source}</source>\n"
        f"<translation_a>{comparison.first_output}</translation_a>\n"
        f"<translation_b>{comparison.second_output}</translation_b>"
    )


def judge_key(verdict: JudgeVerdict) -> tuple[str, str, str]:
    return (verdict.segment_id, verdict.first, verdict.second)


def read_verdicts(path: Path) -> dict[tuple[str, str, str], JudgeVerdict]:
    if not path.exists():
        return {}
    return {judge_key(v): v for v in read_jsonl(path, JudgeVerdict)}


def plan_comparisons(
    segments: list[EvalSegment],
    pairs: Iterable[tuple[str, str]],
    outputs: dict[tuple[str, str], str],
) -> list[Comparison]:
    """Both orders of every pair that needs a judge.

    Identical answers tie, and an answer beats a failure, so neither costs a call.
    """
    comparisons = []
    for left, right in pairs:
        for segment in segments:
            left_output = outputs.get((left, segment.id))
            right_output = outputs.get((right, segment.id))
            if left_output is None or right_output is None or left_output == right_output:
                continue
            comparisons.append(Comparison(segment, left, right, left_output, right_output))
            comparisons.append(Comparison(segment, right, left, right_output, left_output))
    return comparisons


def to_verdict(comparison: Comparison, answer: Answer) -> JudgeVerdict:
    winners = {"A": comparison.first, "B": comparison.second}
    return JudgeVerdict(
        segment_id=comparison.segment.id,
        first=comparison.first,
        second=comparison.second,
        winner=winners[answer.letter] if answer.letter else None,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )


class PairScore(NamedTuple):
    """The first model's score per sentence: 1 for a win, 0.5 for a tie, 0 for a loss."""

    scores: list[float]
    segment_ids: list[str]
    split_orders: int
    unusable: int


def score_pair(
    segments: list[EvalSegment],
    left: str,
    right: str,
    outputs: dict[tuple[str, str], str],
    verdicts: dict[tuple[str, str, str], JudgeVerdict],
) -> PairScore:
    scores: list[float] = []
    segment_ids: list[str] = []
    split_orders = unusable = 0
    for segment in segments:
        left_output = outputs.get((left, segment.id))
        right_output = outputs.get((right, segment.id))
        if left_output is None and right_output is None:
            unusable += 1
            continue
        if left_output is None or right_output is None or left_output == right_output:
            score = 0.5 if left_output == right_output else float(right_output is None)
            scores.append(score)
            segment_ids.append(segment.id)
            continue
        one = verdicts.get((segment.id, left, right))
        other = verdicts.get((segment.id, right, left))
        if one is None or other is None or one.winner is None or other.winner is None:
            unusable += 1
            continue
        split_orders += one.winner != other.winner
        scores.append(((one.winner == left) + (other.winner == left)) / 2)
        segment_ids.append(segment.id)
    return PairScore(scores, segment_ids, split_orders, unusable)


class BatchJudge:
    """Submits one batch and remembers its id, so a rerun resumes instead of paying twice.

    The caller deletes the batch file once the verdicts are stored.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        effort: Literal["low", "medium", "high"],
        batch_file: Path,
        wait: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._batch_file = batch_file
        self._wait = wait

    def decide(self, comparisons: list[Comparison]) -> dict[str, Answer]:
        batch_id = self._resume_or_submit(comparisons)
        while self._client.messages.batches.retrieve(batch_id).processing_status != "ended":
            self._wait(POLL_SECONDS)
        return {
            result.custom_id: _answer_from(result)
            for result in self._client.messages.batches.results(batch_id)
        }

    def _resume_or_submit(self, comparisons: list[Comparison]) -> str:
        if self._batch_file.exists():
            saved: str = json.loads(self._batch_file.read_text(encoding="utf-8"))["id"]
            return saved
        batch = self._client.messages.batches.create(
            requests=[
                Request(
                    custom_id=comparison.request_id,
                    params=MessageCreateParamsNonStreaming(
                        model=self._model,
                        max_tokens=MAX_TOKENS,
                        output_config={
                            "effort": self._effort,
                            "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
                        },
                        messages=[{"role": "user", "content": build_prompt(comparison)}],
                    ),
                )
                for comparison in comparisons
            ]
        )
        write_json(self._batch_file, {"id": batch.id})
        return batch.id


def _answer_from(entry: anthropic.types.messages.MessageBatchIndividualResponse) -> Answer:
    if entry.result.type != "succeeded":
        return Answer(None)
    message = entry.result.message
    usage = message.usage
    letter: Letter | None = None
    if message.stop_reason == "end_turn":
        text = next((block.text for block in message.content if block.type == "text"), "{}")
        better = json.loads(text).get("better")
        letter = better if better in ("A", "B") else None
    return Answer(letter, usage.input_tokens, usage.output_tokens)
