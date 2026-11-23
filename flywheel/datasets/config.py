"""Dataset build settings. Defaults live in configs/dataset.default.toml as well."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Gemini's terms bar using its output to train competing models, so its answers never
# train by default (docs/decisions/0017). Add any Gemini model the app offers.
NEVER_TRAIN_ON = ["google/gemini-3.8-flash"]


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_languages: list[str] = Field(default_factory=list, description="Empty keeps all.")
    min_quality: float = Field(default=0.0, ge=0, le=1)
    sft_contested: bool = True
    dpo_contested: bool = False
    exclude_models: list[str] = Field(default_factory=lambda: list(NEVER_TRAIN_ON))
    exclude_sources: list[str] = Field(default_factory=list)
    close_pair_chrf: float = Field(default=70.0, gt=0, le=100)

    @classmethod
    def from_toml(cls, path: Path) -> "DatasetConfig":
        with path.open("rb") as file:
            return cls.model_validate(tomllib.load(file))
