"""Generic import: a file of FeedbackEvent records, one per line."""

from pathlib import Path

from flywheel.schema import FeedbackEvent
from flywheel.storage import read_jsonl


def load_events(path: Path) -> list[FeedbackEvent]:
    return read_jsonl(path, FeedbackEvent)
