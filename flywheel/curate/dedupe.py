"""Collapse repeated inputs into one event, counting the votes instead of losing them.

Votes carry no user id, by design. A burst of votes on one input in a short window
is treated as one voter, which is the only defence against stuffing that the data
allows. Tone is part of the key because casual and formal answers legitimately differ.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from flywheel.schema import FeedbackEvent
from flywheel.text import normalize_text


@dataclass(frozen=True)
class VoteGroup:
    """Everything voters said about one input."""

    event: FeedbackEvent
    support: int
    group_size: int
    contested: bool


def input_key(event: FeedbackEvent) -> str:
    parts = (
        normalize_text(event.input.text),
        event.input.effective_src_lang,
        event.input.tgt_lang,
        event.input.tone or "",
    )
    return "\x1f".join(parts)


def independent_votes(times: Iterable[datetime], burst_window_seconds: int) -> int:
    """Count votes, treating any vote inside the window of the last counted one as the same."""
    count = 0
    last: datetime | None = None
    for moment in sorted(times):
        if last is None or (moment - last).total_seconds() >= burst_window_seconds:
            count += 1
            last = moment
    return count


def dedupe(events: Iterable[FeedbackEvent], burst_window_seconds: int) -> list[VoteGroup]:
    """Return one group per input, in an order that never depends on input order."""
    unique = {event.id: event for event in events}
    by_input: dict[str, list[FeedbackEvent]] = defaultdict(list)
    for event in unique.values():
        by_input[input_key(event)].append(event)

    groups = []
    for voters in by_input.values():
        by_answer: dict[str, list[FeedbackEvent]] = defaultdict(list)
        for event in voters:
            by_answer[normalize_text(event.chosen_output)].append(event)

        def votes(answer: list[FeedbackEvent]) -> int:
            return independent_votes((e.created_at for e in answer), burst_window_seconds)

        # Most votes wins. Ties go to the earliest answer, then the lowest id,
        # so the result never depends on input order.
        winner = min(
            by_answer.values(),
            key=lambda answer: (-votes(answer), min((e.created_at, e.id) for e in answer)),
        )
        groups.append(
            VoteGroup(
                event=min(winner, key=lambda e: (e.created_at, e.id)),
                support=votes(winner),
                group_size=sum(votes(answer) for answer in by_answer.values()),
                contested=len(by_answer) > 1,
            )
        )
    return sorted(groups, key=lambda group: (group.event.created_at, group.event.id))
