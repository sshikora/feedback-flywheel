"""Build valid records, overriding only what a test cares about."""

from datetime import UTC, datetime, timedelta
from typing import Any

from flywheel.schema import (
    Candidate,
    Consent,
    EvalSegment,
    EventContext,
    FeedbackEvent,
    PreferenceSignal,
    TranslationInput,
)

START = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


def make_event(
    event_id: str = "e1",
    text: str = "Where is the market?",
    outputs: dict[str, str] | None = None,
    chosen: list[str] | None = None,
    minutes: int = 0,
    **overrides: Any,
) -> FeedbackEvent:
    outputs = (
        outputs if outputs is not None else {"A": "Asa ang merkado?", "B": "Hain ang tiyanggihan?"}
    )
    fields: dict[str, Any] = {
        "id": event_id,
        "source": "test",
        "origin": "human",
        "created_at": START + timedelta(minutes=minutes),
        "input": TranslationInput(text=text, src_lang="en", tgt_lang="ceb"),
        "candidates": [
            Candidate(label=str(i), model_id=model, output=output)
            for i, (model, output) in enumerate(outputs.items())
        ],
        "signal": PreferenceSignal(chosen=chosen or [next(iter(outputs))]),
        "context": EventContext(blind=False),
        "consent": Consent(training=True, publish=False),
    }
    return FeedbackEvent(**(fields | overrides))


def make_segment(
    index: int = 0, source: str = "Hello there.", reference: str = "Kumusta diha."
) -> EvalSegment:
    return EvalSegment(
        id=f"flores200:devtest:en-ceb:{index}",
        corpus="flores200",
        split="devtest",
        src_lang="en",
        tgt_lang="ceb",
        source=source,
        reference=reference,
    )
