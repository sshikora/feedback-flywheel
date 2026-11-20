"""One training run, from a run config to a registered candidate.

State is saved after every step, so a rerun picks up where the last one stopped.
Once a job id is saved, no command creates another job for that run.
"""

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from flywheel.schema import DpoPair, ModelRecord, SftExample
from flywheel.storage import read_jsonl, write_json
from flywheel.train.config import RunConfig, config_hash
from flywheel.train.formats import write_training_file
from flywheel.train.together import TERMINAL, Estimate, Together, TogetherError, job_request

PROVIDER = "together"
FILE_POLLS = 60
FILE_POLL_SECONDS = 5.0


class TrainingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    run_config: str
    run_config_hash: str
    dataset: str
    training_file_sha256: str
    lines: int
    skipped: dict[str, int]
    file_id: str | None = None
    estimate_usd: float | None = None
    estimate_tokens: int | None = None
    job_id: str | None = None
    status: str = "prepared"
    submitted_at: datetime | None = None
    job: dict[str, Any] | None = None
    weights: str | None = None

    @classmethod
    def load(cls, path: Path) -> "TrainingState":
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        write_json(path, self.model_dump(mode="json"))


class RunRefused(Exception):
    """The run cannot go ahead as configured. Nothing was billed."""


def run_dir(data_dir: Path, name: str) -> Path:
    return data_dir / "training" / name


def prepare(
    config: RunConfig,
    config_path: Path,
    data_dir: Path,
    client: Together,
    from_checkpoint: str | None,
    wait: Callable[[float], None],
) -> TrainingState:
    """Write and upload the training file, wait for validation and get a price. Free."""
    directory = run_dir(data_dir, config.name)
    state_path = directory / "state.json"
    if state_path.exists():
        state = TrainingState.load(state_path)
        if state.job_id is not None:
            return state
        if state.run_config_hash != config_hash(config_path):
            raise RunRefused(f"{config_path} changed since this run was prepared")
    examples = _snapshot_examples(data_dir, config)
    training = write_training_file(examples, directory / f"{config.method}.jsonl")
    if training.lines == 0:
        raise RunRefused("no example could be written in the app's prompt format")
    state = TrainingState(
        name=config.name,
        run_config=config_path.as_posix(),
        run_config_hash=config_hash(config_path),
        dataset=config.dataset,
        training_file_sha256=training.sha256,
        lines=training.lines,
        skipped=dict(training.skipped),
    )
    state.file_id = client.upload(training.path)
    state.save(state_path)
    _wait_for_validation(client, state.file_id, wait)
    estimate = client.estimate(job_request(config, state.file_id, from_checkpoint))
    _check_estimate(config, estimate)
    state.estimate_usd = estimate.total_usd
    state.estimate_tokens = estimate.train_tokens
    state.save(state_path)
    return state


def submit(
    config: RunConfig,
    config_path: Path,
    data_dir: Path,
    client: Together,
    from_checkpoint: str | None,
) -> TrainingState:
    """Start the billed job, once. The job id is saved before anything else happens."""
    state_path = run_dir(data_dir, config.name) / "state.json"
    state = TrainingState.load(state_path)
    if state.job_id is not None:
        return state
    if state.file_id is None or state.estimate_usd is None:
        raise RunRefused("the run is not prepared")
    if state.run_config_hash != config_hash(config_path):
        raise RunRefused(f"{config_path} changed since this run was prepared")
    job_id = client.create(job_request(config, state.file_id, from_checkpoint))
    state.job_id = job_id
    state.status = "pending"
    state.submitted_at = datetime.now(UTC)
    state.save(state_path)
    return state


def refresh(data_dir: Path, name: str, client: Together) -> TrainingState:
    state_path = run_dir(data_dir, name) / "state.json"
    state = TrainingState.load(state_path)
    if state.job_id is None or state.status in TERMINAL:
        return state
    job = client.job(state.job_id)
    state.job = job
    state.status = str(job.get("status", state.status))
    state.save(state_path)
    return state


def candidate_record(state: TrainingState, config: RunConfig) -> ModelRecord:
    """The registry record for a completed job. Cost is Together's estimate, in dollars."""
    if state.status != "completed" or state.job_id is None:
        raise RunRefused(f"{state.name} is {state.status}, not completed")
    return ModelRecord(
        id=state.name,
        base_model=config.base_model,
        start_from=config.start_from,
        method=config.method,
        dataset_hash=config.dataset,
        run_config=state.run_config,
        run_config_hash=state.run_config_hash,
        provider=PROVIDER,
        provider_job_id=state.job_id,
        weights=state.weights,
        cost_usd=state.estimate_usd or 0.0,
        status="candidate",
        created_at=datetime.now(UTC),
    )


def _snapshot_examples(data_dir: Path, config: RunConfig) -> list[SftExample] | list[DpoPair]:
    directory = data_dir / "datasets" / f"{config.method}-{config.dataset[:12]}"
    path = directory / "examples.jsonl"
    if not path.exists():
        raise RunRefused(f"no {config.method} snapshot has hash {config.dataset}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != config.dataset:
        raise RunRefused(f"{path} no longer matches its hash, so the snapshot was edited")
    if config.method == "sft":
        return read_jsonl(path, SftExample)
    return read_jsonl(path, DpoPair)


def _wait_for_validation(client: Together, file_id: str, wait: Callable[[float], None]) -> None:
    for _ in range(FILE_POLLS):
        status, report = client.file_status(file_id)
        if status == "COMPLETED":
            return
        if status in ("INVALID_FORMAT", "FAILED"):
            raise RunRefused(f"Together rejected the training file: {report.get('error', status)}")
        wait(FILE_POLL_SECONDS)
    raise TogetherError("the training file is still being validated. Run the command again.")


def _check_estimate(config: RunConfig, estimate: Estimate) -> None:
    if not estimate.allowed:
        raise RunRefused(
            f"Together will not run this job at ${estimate.total_usd:.2f}. "
            f"Your credit limit is ${estimate.limit_usd}."
        )
    if estimate.total_usd > config.max_cost_usd:
        raise RunRefused(
            f"the estimate of ${estimate.total_usd:.2f} is above max_cost_usd "
            f"of ${config.max_cost_usd:.2f}"
        )
