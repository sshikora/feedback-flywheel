"""Blind human rating by spreadsheet.

`export_sheet` writes rows with no model names, in an order that changes per round.
The key that says who wrote each row stays with the operator. `import_sheet` joins
the scores back to the models. The M4 rater app produces the same HumanRating records.

Hidden checks catch careless scoring: a reference should score well, and a reference
for a different sentence should score badly.
"""

import csv
import hashlib
import io
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, NamedTuple

from flywheel.evals.judge import choose_segments
from flywheel.schema import EvalSegment, HumanRating

HEADER = ["row", "source", "translation", "adequacy", "fluency"]
REFERENCE_CHECK = "check:reference"
MISMATCH_CHECK = "check:mismatch"
GOOD_CHECK_MIN = 3  # a real translation should reach at least this adequacy
WRONG_CHECK_MAX = 2  # a translation of another sentence should reach at most this
FORMULA_STARTS = ("=", "+", "-", "@")


class SheetRow(NamedTuple):
    row: str
    source: str
    translation: str
    segment_id: str
    model_id: str
    check: Literal["good", "wrong"] | None


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def build_rows(
    segments: list[EvalSegment],
    outputs: Mapping[tuple[str, str], str],
    models: list[str],
    settings_seed: str,
    round_name: str,
    sentences: int,
    checks: int,
) -> list[SheetRow]:
    """The rows for one round, shuffled. The sentences stay the same across rounds."""
    chosen = choose_segments(segments, sentences, settings_seed)

    def make(
        segment: EvalSegment,
        translation: str,
        model: str,
        check: Literal["good", "wrong"] | None,
    ) -> SheetRow:
        row = _digest(settings_seed, round_name, segment.id, model)[:10]
        return SheetRow(row, segment.source, translation, segment.id, model, check)

    rows = [
        make(segment, output, model, None)
        for segment in chosen
        for model in models
        if (output := outputs.get((model, segment.id))) is not None
    ]
    for index, segment in enumerate(chosen[:checks]):
        rows.append(make(segment, segment.reference, REFERENCE_CHECK, "good"))
        other = chosen[(index + 1) % len(chosen)]
        rows.append(make(segment, other.reference, MISMATCH_CHECK, "wrong"))
    return sorted(rows, key=lambda r: _digest(settings_seed, round_name, "order", r.row))


def _safe_cell(text: str) -> str:
    """Stop a spreadsheet from running a translation as a formula."""
    return f"'{text}" if text.startswith(FORMULA_STARTS) else text


def render_sheet(rows: Iterable[SheetRow]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    for row in rows:
        writer.writerow([row.row, _safe_cell(row.source), _safe_cell(row.translation), "", ""])
    return buffer.getvalue()


def key_entries(rows: Iterable[SheetRow]) -> dict[str, dict[str, str | None]]:
    return {
        row.row: {"segment_id": row.segment_id, "model_id": row.model_id, "check": row.check}
        for row in rows
    }


def _score(raw: str, column: str, row: str) -> int:
    if raw not in ("1", "2", "3", "4", "5"):
        raise ValueError(f"row {row}: {column} must be a whole number from 1 to 5, got {raw!r}")
    return int(raw)


class Imported(NamedTuple):
    ratings: list[HumanRating]
    blank: int


def import_sheet(
    path: Path, key: Mapping[str, Mapping[str, str | None]], rater_id: str, round_name: str
) -> Imported:
    """Read a scored sheet. Blank rows are counted, but half-scored rows are errors."""
    ratings: list[HumanRating] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not set(HEADER) <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected the columns {HEADER}")
        for line in reader:
            row = line["row"].strip()
            if row not in key:
                raise ValueError(f"{path}: row {row!r} is not in the key for this round")
            if row in seen:
                raise ValueError(f"{path}: row {row!r} appears twice")
            seen.add(row)
            adequacy, fluency = line["adequacy"].strip(), line["fluency"].strip()
            if not adequacy and not fluency:
                continue
            entry = key[row]
            check = entry["check"]
            ratings.append(
                HumanRating(
                    segment_id=str(entry["segment_id"]),
                    model_id=str(entry["model_id"]),
                    rater_id=rater_id,
                    round=round_name,
                    adequacy=_score(adequacy, "adequacy", row),
                    fluency=_score(fluency, "fluency", row),
                    check="good" if check == "good" else "wrong" if check == "wrong" else None,
                )
            )
    unseen = set(key) - seen
    if unseen:
        raise ValueError(f"{path}: {len(unseen)} rows from the key are missing")
    return Imported(ratings, len(key) - len(ratings))


def check_pass_rate(ratings: Iterable[HumanRating]) -> float | None:
    checks = [r for r in ratings if r.check is not None]
    if not checks:
        return None
    passed = sum(
        r.adequacy >= GOOD_CHECK_MIN if r.check == "good" else r.adequacy <= WRONG_CHECK_MAX
        for r in checks
    )
    return passed / len(checks)


def trusted_ratings(ratings: Iterable[HumanRating], min_pass: float) -> list[HumanRating]:
    """Drop whole rounds where a rater missed the hidden checks."""
    groups: dict[tuple[str, str], list[HumanRating]] = defaultdict(list)
    for rating in ratings:
        groups[(rating.rater_id, rating.round)].append(rating)
    return [
        rating
        for group in groups.values()
        if (rate := check_pass_rate(group)) is None or rate >= min_pass
        for rating in group
    ]
