"""Curation settings. Defaults live in configs/curation.default.toml as well."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class CurationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    heldout_pct: float = Field(default=10.0, ge=0, lt=100)
    near_dup_threshold: float = Field(default=0.6, gt=0, le=1)
    burst_window_seconds: int = Field(default=600, ge=0)
    max_length_ratio: float = Field(default=5.0, gt=1)
    seed: str = "flywheel-v1"

    @classmethod
    def from_toml(cls, path: Path) -> "CurationConfig":
        with path.open("rb") as file:
            return cls.model_validate(tomllib.load(file))
