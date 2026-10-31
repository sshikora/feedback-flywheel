"""Write JSON Schema for the canonical records.

The generated TypeScript client is built from these files, so the TypeScript
side cannot drift from the Python models. CI fails if they are out of date.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from flywheel.schema import CuratedEvent, EvalSegment, FeedbackEvent

EXPORTED: dict[str, type[BaseModel]] = {
    "feedback_event": FeedbackEvent,
    "curated_event": CuratedEvent,
    "eval_segment": EvalSegment,
}


def render_schemas() -> dict[str, str]:
    return {
        f"{name}.schema.json": json.dumps(model.model_json_schema(), indent=2) + "\n"
        for name, model in EXPORTED.items()
    }


def write_schemas(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in render_schemas().items():
        (directory / filename).write_text(content, encoding="utf-8")


def stale_schemas(directory: Path) -> list[str]:
    """Return the schema files that are missing or differ from the models."""
    return [
        filename
        for filename, content in render_schemas().items()
        if not (directory / filename).exists()
        or (directory / filename).read_text(encoding="utf-8") != content
    ]
