"""Run models over an eval set through OpenRouter.

The prompt and request settings match the translation app, so these scores
describe what its users get. Outputs are stored as they arrive, so a rerun
retries only what is missing and a crash never pays twice.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, NamedTuple, Protocol

import httpx

from flywheel.schema import EvalSegment, TranslationRecord
from flywheel.storage import append_jsonl, read_jsonl

OPENROUTER_URL = "https://openrouter.ai/api/v1"
ATTEMPTS = 5
WORKERS = 8

# The app sends these on every OpenRouter request. See providers/openrouter.ts.
ROUTING = {"provider": {"data_collection": "deny", "zdr": True}}

# App language codes to the names its prompt uses.
LANGUAGE_NAMES = {"en": "English", "ceb": "Cebuano"}


class Completion(NamedTuple):
    text: str
    cost_usd: float


class CompletionError(Exception):
    """This one request failed. Other requests may still succeed."""


class RequestRejected(Exception):
    """The API refused this request, and retrying would not help. Stops the run."""


class BudgetExceeded(Exception):
    pass


class Completer(Protocol):
    def complete(self, model: str, prompt: str) -> Completion: ...


class OpenRouter:
    def __init__(
        self,
        api_key: str,
        reasoning_effort: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        wait: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            base_url=OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
            transport=transport,
        )
        self._reasoning_effort = dict(reasoning_effort or {})
        self._wait = wait

    def complete(self, model: str, prompt: str) -> Completion:
        effort = self._reasoning_effort.get(model)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True},
            **ROUTING,
            "reasoning": {"effort": effort} if effort else {"enabled": False},
        }
        failure = "no attempts made"
        for attempt in range(ATTEMPTS):
            try:
                response = self._client.post("/chat/completions", json=body)
            except httpx.TransportError as err:
                failure = repr(err)
            else:
                if response.status_code != 429 and response.status_code < 500:
                    # Other 4xx errors (bad key, no credit, unknown model) hit every
                    # request alike, so they stop the run and are not recorded.
                    if response.is_error:
                        raise RequestRejected(f"{model}: {_error_message(response)}")
                    return _parse(response.json())
                failure = f"HTTP {response.status_code}"
            self._wait(2**attempt)
        raise CompletionError(failure)


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return f"HTTP {response.status_code}"


def _parse(payload: dict[str, Any]) -> Completion:
    # A missing cost fails the run, because the budget check depends on it.
    cost = float(payload["usage"]["cost"])
    choices = payload.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    if not text:
        raise CompletionError("empty answer")
    return Completion(text, cost)


def build_prompt(segment: EvalSegment) -> str:
    source = LANGUAGE_NAMES[segment.src_lang]
    target = LANGUAGE_NAMES[segment.tgt_lang]
    return (
        f"Translate the following text from {source} to {target}. "
        "Return only the translated text with no explanation or commentary."
        f"\n\n{segment.source}"
    )


def read_translations(path: Path) -> dict[tuple[str, str], TranslationRecord]:
    """The latest record per model and segment, so a retry replaces a failure."""
    if not path.exists():
        return {}
    return {(r.model_id, r.segment_id): r for r in read_jsonl(path, TranslationRecord)}


def missing_work(
    segments: Iterable[EvalSegment],
    models: Iterable[str],
    done: dict[tuple[str, str], TranslationRecord],
) -> list[tuple[str, EvalSegment]]:
    return [
        (model, segment)
        for model in models
        for segment in segments
        if (found := done.get((model, segment.id))) is None or found.output is None
    ]


class Budget:
    """Shared by the workers, so spending stops within the requests already in flight."""

    def __init__(self, limit_usd: float, spent_usd: float) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        self._lock = Lock()

    def charge(self, cost_usd: float) -> None:
        with self._lock:
            self.spent_usd += cost_usd

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent_usd > self.limit_usd


def _translate(
    completer: Completer, model: str, segment: EvalSegment, budget: Budget
) -> TranslationRecord | None:
    if budget.exhausted:
        return None
    try:
        answer = completer.complete(model, build_prompt(segment))
    except CompletionError as err:
        return TranslationRecord(segment_id=segment.id, model_id=model, error=str(err))
    budget.charge(answer.cost_usd)
    return TranslationRecord(
        segment_id=segment.id, model_id=model, output=answer.text, cost_usd=answer.cost_usd
    )


def translate(
    work: list[tuple[str, EvalSegment]],
    completer: Completer,
    path: Path,
    max_cost_usd: float,
    spent_usd: float = 0.0,
) -> float:
    """Run the work and return total spend. Stops once the budget is passed."""
    budget = Budget(max_cost_usd, spent_usd)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures: list[Future[TranslationRecord | None]] = [
            pool.submit(_translate, completer, model, segment, budget) for model, segment in work
        ]
        try:
            for future in as_completed(futures):
                record = future.result()
                if record is not None:
                    append_jsonl(path, [record])
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if budget.exhausted:
        raise BudgetExceeded(f"spent ${budget.spent_usd:.2f} of ${max_cost_usd:.2f}")
    return budget.spent_usd
