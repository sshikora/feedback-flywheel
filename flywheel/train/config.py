"""Run configs. Every training run is driven by one checked-in TOML file.

There are no defaults. A run config states every setting, including the gate
thresholds, before training starts, so a result cannot move the bar it is held to.
"""

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Hyperparameters(Strict):
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(ge=1)
    lora_rank: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    dpo_beta: float | None = Field(default=None, gt=0, description="Required for DPO runs.")


class GateConfig(Strict):
    """Pre-registered thresholds. See PLAN.md section 8."""

    min_judge_win_rate: float = Field(gt=0.5, le=1)
    guard_set: str = Field(min_length=1)
    human_check: bool


class RunConfig(Strict):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    description: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    start_from: str | None = Field(
        default=None, description="Registry id of a model to continue from, such as SFT before DPO."
    )
    method: Literal["sft", "dpo"]
    dataset: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Snapshot hash from `flywheel build`."
    )
    seed: int
    max_cost_usd: float = Field(gt=0, description="The job is refused above this estimate.")
    hyperparameters: Hyperparameters
    gate: GateConfig

    @classmethod
    def from_toml(cls, path: Path) -> "RunConfig":
        with path.open("rb") as file:
            config = cls.model_validate(tomllib.load(file))
        if config.method == "dpo" and config.hyperparameters.dpo_beta is None:
            raise ValueError(f"{path}: a DPO run needs hyperparameters.dpo_beta")
        return config


def config_hash(path: Path) -> str:
    """The hash of the file as checked in, so an edit after the run is visible."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
