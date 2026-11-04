"""Immutable dataset snapshots.

A snapshot's folder is named by the hash of its examples, so a training run that
names a hash names exactly what the model learned from. Building the same
content again changes nothing, and a snapshot whose examples were edited is refused.
"""

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from flywheel.curate.report import language_pairs, origin_shares
from flywheel.datasets.builders import Build
from flywheel.datasets.card import render_card
from flywheel.datasets.config import DatasetConfig
from flywheel.schema import DpoPair, SftExample
from flywheel.storage import write_json

Kind = Literal["sft", "dpo"]


@dataclass(frozen=True)
class Snapshot:
    directory: Path
    digest: str
    created: bool


def serialize(examples: Sequence[BaseModel]) -> str:
    return "".join(example.model_dump_json() + "\n" for example in examples)


def content_hash(examples: Sequence[BaseModel]) -> str:
    return hashlib.sha256(serialize(examples).encode()).hexdigest()


def manifest(
    kind: Kind,
    build: Build[SftExample] | Build[DpoPair],
    config: DatasetConfig,
    source: dict[str, Any],
) -> dict[str, Any]:
    events = [item.event for item in build.used]
    counts: dict[str, Any] = {
        "examples": len(build.examples),
        "events": len(events),
        "dropped": dict(sorted(build.dropped.items())),
        "origin": origin_shares(events),
        "sources": dict(sorted(Counter(e.source for e in events).items())),
        "language_pairs": language_pairs(events),
        "contested_events": sum(item.contested for item in build.used),
        "blind_share": round(sum(e.context.blind for e in events) / len(events), 4)
        if events
        else 0.0,
    }
    if kind == "sft":
        counts["target_models"] = _model_counts(
            m for e in build.examples if isinstance(e, SftExample) for m in e.target_models
        )
    else:
        pairs = [p for p in build.examples if isinstance(p, DpoPair)]
        counts["close_pairs"] = sum(p.close for p in pairs)
        counts["close_share"] = round(counts["close_pairs"] / len(pairs), 4) if pairs else 0.0
        counts["chosen_models"] = _model_counts(m for p in pairs for m in p.chosen_models)
        counts["rejected_models"] = _model_counts(p.rejected_model for p in pairs)
    return {
        "kind": kind,
        "hash": content_hash(build.examples),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config.model_dump(),
        "source": source,
        "counts": counts,
    }


def write_snapshot(
    root: Path, kind: Kind, examples: Sequence[BaseModel], meta: dict[str, Any]
) -> Snapshot:
    digest = content_hash(examples)
    if meta["hash"] != digest:
        raise ValueError("the manifest does not describe these examples")
    directory = root / f"{kind}-{digest[:12]}"
    stored = directory / "examples.jsonl"
    if stored.exists():
        if hashlib.sha256(stored.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{stored} no longer matches its hash, so the snapshot was edited")
        return Snapshot(directory, digest, created=False)
    directory.mkdir(parents=True)
    stored.write_text(serialize(examples), encoding="utf-8")
    write_json(directory / "manifest.json", meta)
    (directory / "DATA_CARD.md").write_text(render_card(meta), encoding="utf-8")
    return Snapshot(directory, digest, created=True)


def _model_counts(models: Any) -> dict[str, int]:
    return dict(sorted(Counter(models).items()))
