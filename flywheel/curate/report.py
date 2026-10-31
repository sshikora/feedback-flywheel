"""Aggregate statistics for a curation run. Holds counts only, never text.

Model names were visible when people voted, so these numbers describe preference and
brand bias together. They are never evaluation results. Win rate divides by appearances
because the free and paid candidate sets differ, which makes raw vote share misleading.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from flywheel.schema import FeedbackEvent


def vote_statistics(events: Sequence[FeedbackEvent]) -> dict[str, Any]:
    wins: Counter[str] = Counter()
    appearances: Counter[str] = Counter()
    own_model_votes = own_model_picks = 0

    for event in events:
        chosen = set(event.signal.chosen)
        wins.update(chosen)
        appearances.update({c.model_id for c in event.answered_candidates})
        your_model = event.context.your_model
        if your_model is not None and any(
            c.model_id == your_model for c in event.answered_candidates
        ):
            own_model_votes += 1
            own_model_picks += your_model in chosen

    by_model = {
        model: {
            "wins": wins[model],
            "appearances": seen,
            "win_rate": round(wins[model] / seen, 4),
        }
        for model, seen in sorted(appearances.items())
    }
    return {
        "votes": len(events),
        "blind_share": _share(sum(e.context.blind for e in events), len(events)),
        "by_model": by_model,
        "own_model_pick_rate": _share(own_model_picks, own_model_votes),
        "model_sets": dict(
            sorted(Counter(e.context.model_set or "unknown" for e in events).items())
        ),
    }


def origin_shares(events: Sequence[FeedbackEvent]) -> dict[str, Any]:
    origins = Counter(event.origin for event in events)
    return {
        "human": origins["human"],
        "synthetic": origins["synthetic"],
        "human_share": _share(origins["human"], len(events)),
    }


def language_pairs(events: Sequence[FeedbackEvent]) -> dict[str, int]:
    pairs = Counter(f"{e.input.effective_src_lang}-{e.input.tgt_lang}" for e in events)
    return dict(sorted(pairs.items()))


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None
