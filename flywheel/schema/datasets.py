"""Training records built from curated events. Each one names the event it came from."""

from typing import Literal

from pydantic import Field

from flywheel.schema.events import Record


class SftExample(Record):
    id: str
    event_id: str
    origin: Literal["human", "synthetic"]
    src_lang: str
    tgt_lang: str
    tone: str | None = None
    auto_detect: bool = Field(
        description="The user left the source language on auto, so the app asked for JSON."
    )
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    target_models: list[str] = Field(min_length=1, description="Models that wrote the target.")
    support: int = Field(ge=1, description="Independent votes for the target.")


class DpoPair(Record):
    id: str
    event_id: str
    origin: Literal["human", "synthetic"]
    src_lang: str
    tgt_lang: str
    tone: str | None = None
    auto_detect: bool = Field(
        description="The user left the source language on auto, so the app asked for JSON."
    )
    source: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    chosen_models: list[str] = Field(min_length=1)
    rejected: str = Field(min_length=1)
    rejected_model: str
    support: int = Field(ge=1, description="Independent votes for the chosen answer.")
    close: bool = Field(
        description="True when chosen and rejected are near-identical, which is a weak signal."
    )
