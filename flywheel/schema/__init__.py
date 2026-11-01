"""Canonical records. Other languages get their types from the exported JSON Schema."""

from flywheel.schema.evals import HumanRating, JudgeVerdict, TranslationRecord
from flywheel.schema.events import (
    Candidate,
    Consent,
    CuratedEvent,
    EvalSegment,
    EventContext,
    FeedbackEvent,
    PreferenceSignal,
    TranslationInput,
)

__all__ = [
    "Candidate",
    "Consent",
    "CuratedEvent",
    "EvalSegment",
    "EventContext",
    "FeedbackEvent",
    "HumanRating",
    "JudgeVerdict",
    "PreferenceSignal",
    "TranslationInput",
    "TranslationRecord",
]
