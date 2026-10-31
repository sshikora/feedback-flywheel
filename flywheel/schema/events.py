"""The feedback event: one record type for all sources.

`signal` has a single kind in v1. Correction and implicit signals join as a
discriminated union once a source actually produces them.
"""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    """Immutable, and rejects unknown fields so schema drift fails loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TranslationInput(Record):
    text: str = Field(min_length=1)
    src_lang: str | None = Field(default=None, description="As requested; None for auto-detect.")
    detected_src_lang: str | None = None
    tgt_lang: str = Field(min_length=1)
    tone: str | None = None

    @property
    def effective_src_lang(self) -> str:
        return self.src_lang or self.detected_src_lang or "und"


class Candidate(Record):
    """One model's answer. Failed candidates are kept because they explain missing pairs."""

    label: str
    model_id: str
    output: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _output_xor_error(self) -> Self:
        if (self.output is None) == (self.error is None):
            raise ValueError("candidate needs exactly one of output or error")
        return self


class PreferenceSignal(Record):
    kind: Literal["preference"] = "preference"
    chosen: list[str] = Field(
        min_length=1,
        description="Model ids. An agreement group counts for all of its models.",
    )


class EventContext(Record):
    blind: bool = Field(description="False when the rater could see model names.")
    your_model: str | None = Field(default=None, description="The rater's usual model, if any.")
    model_set: str | None = Field(default=None, description="Which candidate set was offered.")
    ui_version: str | None = None


class Consent(Record):
    training: bool
    publish: bool


class FeedbackEvent(Record):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    origin: Literal["human", "synthetic"]
    created_at: datetime
    task: Literal["translation"] = "translation"
    input: TranslationInput
    candidates: list[Candidate] = Field(min_length=1)
    signal: PreferenceSignal
    context: EventContext
    consent: Consent

    @model_validator(mode="after")
    def _chosen_models_answered(self) -> Self:
        answered = {c.model_id for c in self.candidates if c.output is not None}
        unknown = set(self.signal.chosen) - answered
        if unknown:
            raise ValueError(f"chosen models have no output: {sorted(unknown)}")
        return self

    @property
    def chosen_output(self) -> str:
        chosen = set(self.signal.chosen)
        return next(c.output for c in self.candidates if c.model_id in chosen and c.output)

    @property
    def answered_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.output is not None]


class CuratedEvent(Record):
    """A feedback event after dedupe, with the evidence behind it."""

    event: FeedbackEvent
    support: int = Field(ge=1, description="Independent votes for the chosen answer.")
    group_size: int = Field(ge=1, description="Independent votes on this input for any answer.")
    contested: bool = Field(description="True when voters on this input chose different answers.")
    quality: float = Field(ge=0, le=1)


class EvalSegment(Record):
    """A reference pair from a fixed benchmark. Never enters training."""

    id: str
    corpus: str
    split: str
    src_lang: str
    tgt_lang: str
    source: str
    reference: str
