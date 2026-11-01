"""Evaluation settings. Each run is driven by one checked-in TOML file."""

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(Settings):
    id: str = Field(pattern=r"^[^/]+/.+$", description="An OpenRouter model id.")
    role: Literal["compare", "base"]
    reasoning_effort: Literal["low"] | None = Field(
        default=None, description="For models that cannot run with reasoning off."
    )


class JudgeSettings(Settings):
    model: str = "claude-opus-5-5"
    family: str = Field(default="anthropic", description="The OpenRouter vendor of the judge.")
    effort: Literal["low", "medium", "high"] = "low"
    sentences: int = Field(default=200, ge=1)


class RatingSettings(Settings):
    sentences: int = Field(default=100, ge=1)
    checks: int = Field(default=5, ge=0, description="Hidden good checks, and as many wrong ones.")
    min_check_pass: float = Field(default=0.8, ge=0, le=1)
    min_agreement: float = Field(default=0.667, ge=-1, le=1)


class EvalConfig(Settings):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    eval_set: str = Field(description="A file in data/eval, without its extension.")
    models: list[ModelSpec] = Field(min_length=2)
    judge: JudgeSettings = JudgeSettings()
    rating: RatingSettings = RatingSettings()
    resamples: int = Field(default=2000, ge=100)
    seed: str = "flywheel-v1"
    max_cost_usd: float = Field(default=5.0, gt=0, description="Stops translation at this spend.")

    @model_validator(mode="after")
    def _judge_is_a_different_family(self) -> Self:
        shared = [m.id for m in self.models if m.id.split("/")[0] == self.judge.family]
        if shared:
            raise ValueError(f"judge family {self.judge.family!r} also wrote {shared}")
        return self

    @model_validator(mode="after")
    def _both_roles_present(self) -> Self:
        if {m.role for m in self.models} != {"compare", "base"}:
            raise ValueError("a run needs at least one compare model and one base model")
        return self

    def model_ids(self, role: Literal["compare", "base"]) -> list[str]:
        return [m.id for m in self.models if m.role == role]

    @classmethod
    def from_toml(cls, path: Path) -> "EvalConfig":
        with path.open("rb") as file:
            return cls.model_validate(tomllib.load(file))
