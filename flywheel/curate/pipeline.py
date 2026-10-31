"""The curation run: consent, spam, dedupe, scoring, split, leakage.

Spam is removed before dedupe so that rejected votes cannot add support to anything.
"""

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from flywheel.curate.config import CurationConfig
from flywheel.curate.dedupe import VoteGroup, dedupe, input_key
from flywheel.curate.leakage import LeakageIndex, event_texts, leaks, segment_texts
from flywheel.curate.quality import quality_score, spam_reason
from flywheel.curate.report import language_pairs, origin_shares, vote_statistics
from flywheel.schema import CuratedEvent, EvalSegment, FeedbackEvent


@dataclass(frozen=True)
class CurationResult:
    train: list[CuratedEvent]
    heldout: list[CuratedEvent]
    report: dict[str, Any]


def is_heldout(key: str, config: CurationConfig) -> bool:
    """Assign by hash, so a given input lands on the same side on every run."""
    digest = hashlib.sha256(f"{config.seed}\x1f{key}".encode()).digest()
    return int.from_bytes(digest[:8]) % 10_000 < config.heldout_pct * 100


def curate(
    events: Sequence[FeedbackEvent],
    eval_segments: Sequence[EvalSegment],
    config: CurationConfig,
) -> CurationResult:
    dropped: Counter[str] = Counter()

    consented = [event for event in events if event.consent.training]
    dropped["no_training_consent"] = len(events) - len(consented)

    usable: list[FeedbackEvent] = []
    for event in consented:
        reason = spam_reason(event, config.max_length_ratio)
        if reason is None:
            usable.append(event)
        else:
            dropped[f"spam_{reason}"] += 1

    groups = dedupe(usable, config.burst_window_seconds)
    dropped["duplicate"] = len(usable) - len(groups)

    train_groups: list[VoteGroup] = []
    heldout_groups: list[VoteGroup] = []
    for group in groups:
        side = heldout_groups if is_heldout(input_key(group.event), config) else train_groups
        side.append(group)

    fixed_eval = LeakageIndex(segment_texts(eval_segments), config.near_dup_threshold)
    heldout_index = LeakageIndex(
        (text for group in heldout_groups for text in event_texts(group.event)),
        config.near_dup_threshold,
    )

    train: list[CuratedEvent] = []
    for group in train_groups:
        if leaks(group.event, fixed_eval):
            dropped["leak_fixed_eval"] += 1
        elif leaks(group.event, heldout_index):
            dropped["leak_heldout"] += 1
        else:
            train.append(_to_curated(group))

    heldout = [_to_curated(group) for group in heldout_groups]
    train_events = [item.event for item in train]
    report = {
        "config": config.model_dump(),
        "input": {
            "events": len(events),
            "eval_segments": len(eval_segments),
            "eval_corpora": dict(
                sorted(Counter(s.id.rsplit(":", 1)[0] for s in eval_segments).items())
            ),
        },
        "output": {"train": len(train), "heldout": len(heldout)},
        "dropped": {reason: count for reason, count in sorted(dropped.items()) if count},
        "train_origin": origin_shares(train_events),
        "train_language_pairs": language_pairs(train_events),
        "train_contested": sum(item.contested for item in train),
        "vote_statistics": vote_statistics(usable),
    }
    return CurationResult(train, heldout, report)


def _to_curated(group: VoteGroup) -> CuratedEvent:
    return CuratedEvent(
        event=group.event,
        support=group.support,
        group_size=group.group_size,
        contested=group.contested,
        quality=quality_score(group.event, group.support, group.group_size),
    )
