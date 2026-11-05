"""Turn curated events into SFT examples and DPO pairs.

Only curated training events come in, so held-out and eval text never does.
Rejected answers come from the same comparison as the chosen one, so pairs are
on-policy for the compare set. Many rejected answers are also correct, with
different phrasing, so near-identical pairs are flagged and counted.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from flywheel.datasets.config import DatasetConfig
from flywheel.evals import metrics
from flywheel.schema import CuratedEvent, DpoPair, FeedbackEvent, SftExample
from flywheel.text import normalize_text


@dataclass(frozen=True)
class Build[T]:
    examples: list[T]
    used: list[CuratedEvent]
    dropped: Counter[str]


def build_sft(items: Sequence[CuratedEvent], config: DatasetConfig) -> Build[SftExample]:
    examples: list[SftExample] = []
    used: list[CuratedEvent] = []
    dropped: Counter[str] = Counter()
    for item in items:
        reason = _event_reason(item, config, contested_allowed=config.sft_contested)
        target = _chosen(item.event, config.exclude_models)
        if reason is not None or target is None:
            dropped[reason or "excluded_model"] += 1
            continue
        text, models = target
        event = item.event
        examples.append(
            SftExample(
                id=event.id,
                event_id=event.id,
                origin=event.origin,
                src_lang=event.input.effective_src_lang,
                tgt_lang=event.input.tgt_lang,
                tone=event.input.tone,
                auto_detect=event.input.src_lang is None,
                source=event.input.text,
                target=text,
                target_models=models,
                support=item.support,
            )
        )
        used.append(item)
    return Build(sorted(examples, key=lambda e: e.id), used, dropped)


def build_dpo(items: Sequence[CuratedEvent], config: DatasetConfig) -> Build[DpoPair]:
    pairs: list[DpoPair] = []
    used: list[CuratedEvent] = []
    dropped: Counter[str] = Counter()
    for item in items:
        reason = _event_reason(item, config, contested_allowed=config.dpo_contested)
        chosen = _chosen(item.event, config.exclude_models)
        if reason is not None or chosen is None:
            dropped[reason or "excluded_model"] += 1
            continue
        found = _pairs(item, chosen, config, dropped)
        if not found:
            dropped["no_rejected"] += 1
            continue
        pairs.extend(found)
        used.append(item)
    return Build(sorted(pairs, key=lambda p: p.id), used, dropped)


def _event_reason(item: CuratedEvent, config: DatasetConfig, contested_allowed: bool) -> str | None:
    if config.target_languages and item.event.input.tgt_lang not in config.target_languages:
        return "other_language"
    if item.quality < config.min_quality:
        return "low_quality"
    if item.contested and not contested_allowed:
        return "contested"
    return None


def _chosen(event: FeedbackEvent, excluded: list[str]) -> tuple[str, list[str]] | None:
    """The chosen text, taken from a chosen model that is not excluded."""
    chosen = [m for m in event.signal.chosen if m not in excluded]
    for candidate in event.answered_candidates:
        if candidate.model_id in chosen and candidate.output:
            return candidate.output, chosen
    return None


def _pairs(
    item: CuratedEvent,
    chosen: tuple[str, list[str]],
    config: DatasetConfig,
    dropped: Counter[str],
) -> list[DpoPair]:
    event = item.event
    text, models = chosen
    pairs = []
    for candidate in event.answered_candidates:
        if candidate.model_id in event.signal.chosen or not candidate.output:
            continue
        if candidate.model_id in config.exclude_models:
            dropped["rejected_excluded_model"] += 1
            continue
        if normalize_text(candidate.output) == normalize_text(text):
            dropped["rejected_same_as_chosen"] += 1
            continue
        similarity = metrics.chrf(metrics.chrf_stats([candidate.output], [text]))
        pairs.append(
            DpoPair(
                id=f"{event.id}:{candidate.model_id}",
                event_id=event.id,
                origin=event.origin,
                src_lang=event.input.effective_src_lang,
                tgt_lang=event.input.tgt_lang,
                tone=event.input.tone,
                auto_detect=event.input.src_lang is None,
                source=event.input.text,
                chosen=text,
                chosen_models=models,
                rejected=candidate.output,
                rejected_model=candidate.model_id,
                support=item.support,
                close=similarity >= config.close_pair_chrf,
            )
        )
    return pairs
