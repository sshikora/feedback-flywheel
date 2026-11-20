"""Translate public English text with a teacher model, into synthetic training events.

The teacher is DeepSeek V4.1 Flash on DeepSeek's own API, whose terms allow training
other models on its output (docs/decisions/0017). Requests go straight to DeepSeek, not
through OpenRouter, so no other host's terms apply. The text is public, so the app's
routing to providers that keep no data is not needed here.

The prompt is the app's, so the model learns from the text the app sends. Answers are
stored as they arrive, like eval translations, so a rerun asks only for what is missing.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

from flywheel.evals import translate as translating
from flywheel.ingest import sources
from flywheel.prompts import translation_prompt
from flywheel.schema import (
    Candidate,
    Consent,
    EventContext,
    FeedbackEvent,
    SourceText,
    TeacherSignal,
    TranslationInput,
    TranslationRecord,
)

TEACHER = "deepseek/deepseek-v4.1-flash-1017"
TEACHER_NAME = "DeepSeek V4.1 Flash"
SRC, TGT = "en", "ceb"

DEEPSEEK_URL = "https://api.deepseek.com"
API_MODEL = "deepseek-flash"  # DeepSeek's name for V4.1 Flash 1017 on 2026-11-20

# DeepSeek's peak rates on 2026-11-20, in dollars per token. Off-peak hours cost half,
# so an estimate at the peak rate is an upper bound.
USD_PER_INPUT_TOKEN = 0.30e-6
USD_PER_CACHED_INPUT_TOKEN = 0.006e-6
USD_PER_OUTPUT_TOKEN = 1.20e-6
# Weekdays, in UTC. DeepSeek also bills Chinese public holidays off-peak, which this
# ignores, so a holiday is charged here at the higher rate.
PEAK_HOURS = (range(1, 4), range(6, 10))
CHARS_PER_TOKEN = 4  # rough, for English and Cebuano alike
OUTPUT_CHARS_PER_INPUT_CHAR = 1.5  # Cebuano runs longer than English, with room to spare


def prompt(text: SourceText) -> str:
    return translation_prompt(text.text, TGT, SRC)


def source_name(corpus: str) -> str:
    return f"teacher-{corpus}"


def estimate_usd(texts: Iterable[SourceText]) -> float:
    """An upper bound on what translating these texts costs."""
    total = 0.0
    for text in texts:
        input_tokens = len(prompt(text)) / CHARS_PER_TOKEN
        output_tokens = len(text.text) * OUTPUT_CHARS_PER_INPUT_CHAR / CHARS_PER_TOKEN
        total += input_tokens * USD_PER_INPUT_TOKEN + output_tokens * USD_PER_OUTPUT_TOKEN
    return total


def is_peak(when: datetime) -> bool:
    utc = when.astimezone(UTC)
    return utc.weekday() < 5 and any(utc.hour in hours for hours in PEAK_HOURS)


def cost_usd(usage: Mapping[str, Any], peak: bool) -> float:
    """What DeepSeek charges for one answer, from the token counts it reports."""
    cached = int(usage.get("prompt_cache_hit_tokens", 0))
    uncached = int(usage["prompt_tokens"]) - cached
    full = (
        cached * USD_PER_CACHED_INPUT_TOKEN
        + uncached * USD_PER_INPUT_TOKEN
        + int(usage["completion_tokens"]) * USD_PER_OUTPUT_TOKEN
    )
    return full if peak else full / 2


class DeepSeek:
    """DeepSeek's own chat API. Its answers carry token counts, not a cost, so the cost
    is worked out here. A request that starts or ends in peak hours is charged at peak."""

    def __init__(
        self,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        wait: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = httpx.Client(
            base_url=DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
            transport=transport,
        )
        self._wait = wait
        self._now = now

    def complete(self, model: str, prompt: str) -> translating.Completion:
        body = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": translating.MAX_TOKENS,
            "thinking": {"type": "disabled"},
        }
        started = self._now()
        payload = translating.post_chat(self._client, model, body, self._wait)
        peak = is_peak(started) or is_peak(self._now())
        # A missing token count fails the run, because the budget check depends on it.
        cost = cost_usd(payload["usage"], peak)
        return translating.Completion(translating.answer_text(payload), cost)


def to_events(
    texts: Iterable[SourceText], records: dict[tuple[str, str], TranslationRecord]
) -> list[FeedbackEvent]:
    """One event per answered text. Each is dated by its source, so rebuilding changes nothing."""
    events = []
    for text in texts:
        record = records.get((TEACHER, text.id))
        if record is None or record.output is None:
            continue
        events.append(
            FeedbackEvent(
                id=f"teacher:{TEACHER}:{text.id}",
                source=source_name(text.corpus),
                origin="synthetic",
                created_at=sources.source(text.corpus).released,
                input=TranslationInput(text=text.text, src_lang=SRC, tgt_lang=TGT),
                candidates=[Candidate(label="teacher", model_id=TEACHER, output=record.output)],
                signal=TeacherSignal(chosen=[TEACHER]),
                context=EventContext(blind=True),
                consent=Consent(training=True, publish=True),
            )
        )
    return events


def licenses(event_sources: dict[str, int]) -> dict[str, dict[str, str]]:
    """License and attribution for the source text behind every teacher source present."""
    found = {}
    for name, spec in sorted(sources.SOURCES.items()):
        if source_name(name) in event_sources:
            found[source_name(name)] = {
                "license": spec.license,
                "attribution": f"{spec.attribution} Translated by {TEACHER_NAME}, whose terms "
                "allow training on its output.",
                "url": spec.url,
            }
    return found
