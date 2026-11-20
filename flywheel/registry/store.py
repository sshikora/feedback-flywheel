"""Model records and their history in SQLite.

History is append-only. Status moves only along allowed transitions, and every
move, including a hold that leaves the status unchanged, is written with its
evidence. At most one model is live. When a new version goes live as challenger,
the version it replaces is retired in the same transaction.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flywheel.schema import ModelRecord, Status, StatusChange
from flywheel.schema.registry import LIVE

ALLOWED: dict[Status, frozenset[Status]] = {
    "candidate": frozenset({"challenger", "retired"}),
    "challenger": frozenset({"champion", "retired"}),
    "champion": frozenset({"retired"}),
    "retired": frozenset(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    record TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL REFERENCES models(id),
    change TEXT NOT NULL
);
"""


class Registry:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def register(self, record: ModelRecord) -> ModelRecord:
        if record.status != "candidate":
            raise ValueError("a new model starts as a candidate")
        with self._transaction():
            if self._row(record.id) is not None:
                raise ValueError(f"{record.id} is already registered")
            self._db.execute(
                "INSERT INTO models (id, record, status) VALUES (?, ?, ?)",
                (record.id, record.model_dump_json(), record.status),
            )
            self._log(record.id, None, "candidate", "registered", {})
        return record

    def get(self, model_id: str) -> ModelRecord:
        row = self._row(model_id)
        if row is None:
            raise KeyError(model_id)
        return ModelRecord.model_validate_json(row)

    def models(self, status: Status | None = None) -> list[ModelRecord]:
        query = "SELECT record FROM models"
        rows = (
            self._db.execute(query + " WHERE status = ? ORDER BY id", (status,))
            if status
            else self._db.execute(query + " ORDER BY id")
        )
        return [ModelRecord.model_validate_json(record) for (record,) in rows]

    def live(self) -> ModelRecord | None:
        """The version a new candidate must beat. None until one passes the gate."""
        live = [m for m in self.models() if m.status in LIVE]
        return live[0] if live else None

    def update(self, model_id: str, **fields: Any) -> ModelRecord:
        """Fill in what is known after registration, such as the eval report or the cost."""
        if "status" in fields or "id" in fields:
            raise ValueError("status changes go through transition or hold")
        record = self.get(model_id).model_copy(update=fields)
        ModelRecord.model_validate(record.model_dump())
        with self._transaction():
            self._db.execute(
                "UPDATE models SET record = ? WHERE id = ?", (record.model_dump_json(), model_id)
            )
        return record

    def transition(
        self, model_id: str, to: Status, reason: str, evidence: dict[str, Any]
    ) -> ModelRecord:
        record = self.get(model_id)
        if to not in ALLOWED[record.status]:
            raise ValueError(f"{model_id} cannot move from {record.status} to {to}")
        replaced = self.live() if to == "challenger" else None
        with self._transaction():
            if replaced is not None:
                self._set_status(replaced, "retired")
                self._log(replaced.id, replaced.status, "retired", f"replaced by {model_id}", {})
            self._set_status(record, to)
            self._log(model_id, record.status, to, reason, evidence)
        return self.get(model_id)

    def hold(self, model_id: str, reason: str, evidence: dict[str, Any]) -> None:
        """Record a decision that leaves the status as it is, such as a failed gate."""
        record = self.get(model_id)
        with self._transaction():
            self._log(model_id, record.status, record.status, reason, evidence)

    def history(self, model_id: str | None = None) -> list[StatusChange]:
        query = "SELECT change FROM history"
        rows = (
            self._db.execute(query + " WHERE model_id = ? ORDER BY seq", (model_id,))
            if model_id
            else self._db.execute(query + " ORDER BY seq")
        )
        return [StatusChange.model_validate_json(change) for (change,) in rows]

    def _row(self, model_id: str) -> str | None:
        row = self._db.execute("SELECT record FROM models WHERE id = ?", (model_id,)).fetchone()
        return row[0] if row else None

    def _set_status(self, record: ModelRecord, status: Status) -> None:
        updated = record.model_copy(update={"status": status})
        self._db.execute(
            "UPDATE models SET record = ?, status = ? WHERE id = ?",
            (updated.model_dump_json(), status, record.id),
        )

    def _log(
        self,
        model_id: str,
        from_status: Status | None,
        to_status: Status,
        reason: str,
        evidence: dict[str, Any],
    ) -> None:
        change = StatusChange(
            model_id=model_id,
            at=datetime.now(UTC),
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            evidence=json.loads(json.dumps(evidence)),
        )
        self._db.execute(
            "INSERT INTO history (model_id, change) VALUES (?, ?)",
            (model_id, change.model_dump_json()),
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            yield
        except BaseException:
            self._db.rollback()
            raise
        self._db.commit()
