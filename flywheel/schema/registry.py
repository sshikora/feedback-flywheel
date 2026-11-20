"""Registry records: one per trained model, with every status change and its evidence."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from flywheel.schema.events import Record

Status = Literal["candidate", "challenger", "champion", "retired"]
LIVE: frozenset[Status] = frozenset({"challenger", "champion"})


class ModelRecord(Record):
    id: str
    base_model: str
    start_from: str | None = None
    method: Literal["sft", "dpo"]
    dataset_hash: str
    run_config: str = Field(description="Path of the run config, as checked in.")
    run_config_hash: str
    provider: str
    provider_job_id: str | None = None
    weights: str | None = Field(default=None, description="Where the merged weights are kept.")
    eval_report: str | None = None
    cost_usd: float = Field(ge=0)
    status: Status
    created_at: datetime


class StatusChange(Record):
    """A promotion, a hold or a retirement. A hold keeps the status and records why."""

    model_id: str
    at: datetime
    from_status: Status | None
    to_status: Status
    reason: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
