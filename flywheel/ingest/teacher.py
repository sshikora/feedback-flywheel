"""Translate public English text with a teacher model, into synthetic training events.

The teacher is DeepSeek V4.1 Flash on DeepSeek's own API, whose terms allow training
other models on its output (docs/decisions/0017). Requests are pinned to that provider
with no fallback, so another host's terms never apply. The text is public, so the
app's routing to providers that keep no data is not needed here.

The prompt is the app's, so the model learns from the text the app sends. Answers are
stored as they arrive, like eval translations, so a rerun asks only for what is missing.
"""

from collections.abc import Iterable
from typing import Any

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
ROUTING: dict[str, Any] = {"provider": {"order": ["deepseek"], "allow_fallbacks": False}}
SRC, TGT = "en", "ceb"

# DeepSeek's peak rate through OpenRouter on 2026-11-19, in dollars per token. Off-peak
# hours cost half, so an estimate at this rate is an upper bound.
USD_PER_INPUT_TOKEN = 0.30e-6
USD_PER_OUTPUT_TOKEN = 1.20e-6
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
