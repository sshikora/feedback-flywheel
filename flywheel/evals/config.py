"""Evaluation settings. Each run is driven by one checked-in TOML file."""

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(Settings):
    id: str = Field(
        pattern=r"^[^/]+/.+$",
        description="An OpenRouter model id, or a name for a self-hosted model.",
    )
    role: Literal["compare", "base"]
    reasoning_effort: Literal["low"] | None = Field(
        default=None, description="For models that cannot run with reasoning off."
    )
    host: Literal["openrouter", "modal"] = "openrouter"
    weights: str | None = Field(
        default=None,
        pattern=r"^(hf|run):.+$",
        description="For Modal: hf:<repo> for a base model, run:<name> for a trained one.",
    )

    @model_validator(mode="after")
    def _weights_only_when_self_hosted(self) -> Self:
        if (self.host == "modal") != (self.weights is not None):
            raise ValueError(f"{self.id}: a Modal model needs weights, and only it takes them")
        if self.host == "modal" and self.reasoning_effort is not None:
            raise ValueError(f"{self.id}: reasoning_effort applies to OpenRouter models only")
        return self

    @property
    def weights_key(self) -> str:
        """The model's folder in the Modal volume."""
        kind, _, name = (self.weights or "").partition(":")
        return name if kind == "run" else f"hf--{name.replace('/', '--')}"


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
    purpose: Literal["baseline", "candidate"] = Field(
        default="baseline",
        description=(
            "baseline ranks base models by headroom. candidate scores one trained model, "
            "role base, against the one model it must beat, role compare."
        ),
    )
    eval_set: str = Field(description="A file in data/eval, without its extension.")
    models: list[ModelSpec] = Field(min_length=2)
    judge: JudgeSettings = JudgeSettings()
    rating: RatingSettings = RatingSettings()
    resamples: int = Field(default=2000, ge=100)
    seed: str = "flywheel-v1"
    max_cost_usd: float = Field(default=5.0, gt=0, description="Stops translation at this spend.")
    max_output_ratio: float = Field(
        default=5.0, gt=1, description="Longer than this many times the source is a failed answer."
    )

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

    @model_validator(mode="after")
    def _a_candidate_meets_one_model(self) -> Self:
        if self.purpose == "candidate" and len(self.models) != 2:
            raise ValueError("a candidate run has one base model and one compare model")
        return self

    def model_ids(self, role: Literal["compare", "base"]) -> list[str]:
        return [m.id for m in self.models if m.role == role]

    @classmethod
    def from_toml(cls, path: Path) -> "EvalConfig":
        with path.open("rb") as file:
            return cls.model_validate(tomllib.load(file))
