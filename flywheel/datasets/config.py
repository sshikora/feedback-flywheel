"""Dataset build settings. Defaults live in configs/dataset.default.toml as well."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_quality: float = Field(default=0.0, ge=0, le=1)
    sft_contested: bool = True
    dpo_contested: bool = False
    exclude_models: list[str] = Field(default_factory=list)
    close_pair_chrf: float = Field(default=70.0, gt=0, le=100)

    @classmethod
    def from_toml(cls, path: Path) -> "DatasetConfig":
        with path.open("rb") as file:
            return cls.model_validate(tomllib.load(file))
