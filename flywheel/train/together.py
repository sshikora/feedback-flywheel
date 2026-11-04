"""Together AI's fine-tuning API, over plain HTTP.

Upload, validation, price estimate, job creation, status and download. Creating
a job is billed and never retried here, so a network error cannot start a
second job. The caller saves the job id before doing anything else.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from flywheel.train.config import RunConfig

TOGETHER_URL = "https://api.together.ai"
TERMINAL = frozenset({"completed", "error", "cancelled"})
ATTEMPTS = 5


class TogetherError(Exception):
    """Together refused a request, or it kept failing."""


@dataclass(frozen=True)
class Estimate:
    total_usd: float
    allowed: bool
    limit_usd: float | None
    train_tokens: int | None


class Together:
    def __init__(
        self,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        wait: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            base_url=TOGETHER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60, read=600),
            transport=transport,
        )
        self._wait = wait

    def upload(self, path: Path) -> str:
        """Upload a training file. Together returns the existing id for identical content."""
        response = self._request(
            "POST",
            "/v1/files/upload",
            data={"purpose": "fine-tune", "file_name": path.name},
            files={"file": (path.name, path.read_bytes(), "application/octet-stream")},
        )
        return str(response.json()["id"])

    def file_status(self, file_id: str) -> tuple[str, dict[str, Any]]:
        """The validation state, such as COMPLETED or INVALID_FORMAT, and its report."""
        body = self._request("GET", f"/v1/files/{file_id}").json()
        return str(body.get("processing_status", "PENDING")), body.get("validation_report") or {}

    def estimate(self, job: dict[str, Any]) -> Estimate:
        body = self._request("POST", "/v1/fine-tunes/estimate-price", json=job).json()
        if not body.get("estimation_available"):
            raise TogetherError(f"no price estimate: {body.get('unavailable_reason', 'unknown')}")
        tokens = body.get("estimated_train_token_count")
        return Estimate(
            total_usd=float(body["estimated_total_price"]),
            allowed=bool(body.get("allowed_to_proceed")),
            limit_usd=body.get("user_limit"),
            train_tokens=int(tokens) if tokens is not None else None,
        )

    def create(self, job: dict[str, Any]) -> str:
        """Start a billed job. Not retried, so an error here never starts two."""
        response = self._client.post("/v1/fine-tunes", json=job)
        if response.is_error:
            raise TogetherError(f"job not created: {_error_message(response)}")
        return str(response.json()["id"])

    def job(self, job_id: str) -> dict[str, Any]:
        body: dict[str, Any] = self._request("GET", f"/v1/fine-tunes/{job_id}").json()
        return body

    def download(self, job_id: str, destination: Path, checkpoint: str = "merged") -> int:
        """Stream the trained weights to a file and return its size in bytes."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        params = {"ft_id": job_id, "checkpoint": checkpoint}
        with self._client.stream("GET", "/v1/finetune/download", params=params) as response:
            if response.is_error:
                response.read()
                raise TogetherError(f"download refused: {_error_message(response)}")
            with partial.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        partial.replace(destination)
        return destination.stat().st_size

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        failure = "no attempts made"
        for attempt in range(ATTEMPTS):
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as err:
                failure = repr(err)
            else:
                if response.status_code != 429 and response.status_code < 500:
                    if response.is_error:
                        raise TogetherError(f"{method} {url}: {_error_message(response)}")
                    return response
                failure = f"HTTP {response.status_code}"
            self._wait(2**attempt)
        raise TogetherError(f"{method} {url} kept failing: {failure}")


def job_request(
    config: RunConfig, training_file: str, from_checkpoint: str | None = None
) -> dict[str, Any]:
    """The create body for a run config. The same body asks for the price estimate."""
    hp = config.hyperparameters
    method: dict[str, Any] = (
        {"method": "sft", "train_on_inputs": "auto"}
        if config.method == "sft"
        else {"method": "dpo", "dpo_beta": hp.dpo_beta}
    )
    body: dict[str, Any] = {
        "training_file": training_file,
        "model": config.base_model,
        "n_epochs": hp.epochs,
        "learning_rate": hp.learning_rate,
        "batch_size": hp.batch_size,
        "random_seed": config.seed,
        "n_checkpoints": 1,
        "suffix": config.name[:64],
        "training_method": method,
        "training_type": {"type": "Lora", "lora_r": hp.lora_rank, "lora_alpha": hp.lora_alpha},
    }
    if from_checkpoint is not None:
        body["from_checkpoint"] = from_checkpoint
    return body


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        error = body.get("error") or body.get("message") or body
        return str(error.get("message", error) if isinstance(error, dict) else error)
    except (ValueError, AttributeError):
        return f"HTTP {response.status_code}"
