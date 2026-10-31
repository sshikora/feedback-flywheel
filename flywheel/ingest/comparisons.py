"""Map an export of the translation app's comparisons table to feedback events.

Accepts `aws dynamodb scan` output (`{"Items": [...]}`), a DynamoDB export to S3
(JSON lines of `{"Item": ...}`, optionally gzipped), or plain JSON objects.

Only voted comparisons become events, because a vote is the only signal in v1.
Votes were cast with model names visible, so every event records `blind=False`.
The app's vote line covers training. Text is publishable only if the user shared
the comparison, since only shared comparisons have a public page.
"""

import gzip
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from flywheel.schema import (
    Candidate,
    Consent,
    EventContext,
    FeedbackEvent,
    PreferenceSignal,
    TranslationInput,
)

SOURCE = "ai-lang-translation"
_ATTRIBUTE_TYPES = {"S", "N", "BOOL", "NULL", "L", "M"}


@dataclass(frozen=True)
class ComparisonsImport:
    events: list[FeedbackEvent]
    skipped: Counter[str]


def load_comparisons(path: Path) -> ComparisonsImport:
    events: list[FeedbackEvent] = []
    skipped: Counter[str] = Counter()
    for item in _read_items(path):
        comparison = {key: _unwrap(value) for key, value in item.items()}
        if not comparison.get("vote"):
            skipped["no_vote"] += 1
            continue
        try:
            events.append(_to_event(comparison))
        except (ValidationError, KeyError, TypeError):
            skipped["invalid"] += 1
    return ComparisonsImport(events, skipped)


def _read_items(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        text = file.read()
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        rows = (json.loads(line) for line in text.splitlines() if line.strip())
        yield from (row.get("Item", row) for row in rows)
        return
    if isinstance(document, dict) and "Items" in document:
        yield from document["Items"]
    elif isinstance(document, list):
        yield from document
    else:
        yield document.get("Item", document)


def _unwrap(value: Any) -> Any:
    """Turn a DynamoDB attribute value into plain JSON. Plain values pass through."""
    if isinstance(value, dict) and len(value) == 1:
        kind, inner = next(iter(value.items()))
        if kind in _ATTRIBUTE_TYPES:
            if kind == "NULL":
                return None
            if kind == "N":
                return float(inner)
            if kind == "L":
                return [_unwrap(entry) for entry in inner]
            if kind == "M":
                return {key: _unwrap(entry) for key, entry in inner.items()}
            return inner
    if isinstance(value, dict):
        return {key: _unwrap(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_unwrap(entry) for entry in value]
    return value


def _to_event(comparison: dict[str, Any]) -> FeedbackEvent:
    vote = comparison["vote"]
    source_language = comparison.get("sourceLanguage")
    return FeedbackEvent(
        id=f"cmp_{comparison['id']}",
        source=SOURCE,
        origin="human",
        created_at=vote.get("votedAt") or comparison["createdAt"],
        input=TranslationInput(
            text=comparison["sourceText"],
            src_lang=None if source_language == "auto" else source_language,
            detected_src_lang=comparison.get("detectedSourceLanguage"),
            tgt_lang=comparison["targetLanguage"],
            tone=comparison.get("tone"),
        ),
        candidates=[_to_candidate(c) for c in comparison["candidates"]],
        signal=PreferenceSignal(chosen=vote.get("models") or [vote["model"]]),
        context=EventContext(
            blind=False,
            your_model=comparison.get("yourModel"),
            model_set="pro" if comparison.get("isPro") else "free",
        ),
        consent=Consent(training=True, publish=bool(comparison.get("shared"))),
    )


def _to_candidate(candidate: dict[str, Any]) -> Candidate:
    output = candidate.get("translatedText")
    return Candidate(
        label=candidate["label"],
        model_id=candidate["model"],
        output=output,
        error=None if output else (candidate.get("error") or "no output"),
    )
