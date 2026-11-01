"""Evaluation records. None of these are feedback events, and none enter training."""

from typing import Literal, Self

from pydantic import Field, model_validator

from flywheel.schema.events import Record


class TranslationRecord(Record):
    """One model's answer on one eval segment."""

    segment_id: str
    model_id: str
    output: str | None = None
    error: str | None = None
    cost_usd: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _output_xor_error(self) -> Self:
        if (self.output is None) == (self.error is None):
            raise ValueError("translation needs exactly one of output or error")
        return self


class JudgeVerdict(Record):
    """One blind comparison. `winner` is None when the judge gave no usable answer."""

    segment_id: str
    first: str = Field(description="The model shown as translation A.")
    second: str = Field(description="The model shown as translation B.")
    winner: str | None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class HumanRating(Record):
    """One rater's blind score. The spreadsheet and the rater app both produce these."""

    segment_id: str
    model_id: str
    rater_id: str = Field(min_length=1, description="Pseudonymous. Never a name or email.")
    round: str = Field(min_length=1)
    adequacy: int = Field(ge=1, le=5, description="How much of the meaning survived.")
    fluency: int = Field(ge=1, le=5, description="How natural it sounds.")
    check: Literal["good", "wrong"] | None = Field(
        default=None, description="Set on hidden quality checks, which are never model scores."
    )
