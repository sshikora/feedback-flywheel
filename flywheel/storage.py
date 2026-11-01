"""JSONL reading and writing. Writes are atomic, so a crash never leaves half a file."""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    records = []
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as err:
                raise ValueError(f"{path}:{number}: {err}") from err
    return records


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    _write_atomic(path, "".join(record.model_dump_json() + "\n" for record in records))


def write_json(path: Path, value: Any) -> None:
    _write_atomic(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    """Add records as they finish, so a crash keeps the work already paid for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")
        file.flush()
