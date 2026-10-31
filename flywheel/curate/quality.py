"""Spam rejection and quality scoring.

The default for any vote is low confidence, and it earns its way up through
independent corroboration. The score weights are hand-set and not fitted to
anything. They rank events for dataset thresholds, nothing more.
"""

from flywheel.schema import FeedbackEvent
from flywheel.text import normalize_text

_MIN_LENGTH_FOR_RATIO = 10
_MIN_LENGTH_FOR_REPEATS = 6

_BASE = 0.2
_AGREEMENT_WEIGHT = 0.3
_SUPPORT_WEIGHT = 0.3
_UNANIMITY_WEIGHT = 0.2
_FULL_SUPPORT_AT = 4


def spam_reason(event: FeedbackEvent, max_length_ratio: float) -> str | None:
    """Return why an event should be rejected, or None if it looks usable."""
    text = normalize_text(event.input.text)
    chosen = normalize_text(event.chosen_output)

    if not any(char.isalpha() for char in text):
        return "no_letters"
    compact = text.replace(" ", "")
    if len(compact) >= _MIN_LENGTH_FOR_REPEATS and len(set(compact)) <= 2:
        return "repeated_characters"
    if chosen == text and event.input.effective_src_lang != event.input.tgt_lang:
        return "untranslated"
    if len(text) >= _MIN_LENGTH_FOR_RATIO:
        ratio = len(chosen) / len(text)
        if ratio > max_length_ratio or ratio < 1 / max_length_ratio:
            return "length_outlier"
    return None


def quality_score(event: FeedbackEvent, support: int, group_size: int) -> float:
    """Score from 0 to 1.

    Agreement: models that independently gave the winning answer.
    Support: independent votes for it, saturating at `_FULL_SUPPORT_AT`.
    Unanimity: share of votes on this input that went to it. It only counts once
    a second vote exists, because one vote agreeing with itself proves nothing.
    """
    agreement = len(set(event.signal.chosen)) / len(event.answered_candidates)
    corroboration = min(support - 1, _FULL_SUPPORT_AT - 1) / (_FULL_SUPPORT_AT - 1)
    unanimity = support / group_size if group_size > 1 else 0.0
    score = (
        _BASE
        + _AGREEMENT_WEIGHT * agreement
        + _SUPPORT_WEIGHT * corroboration
        + _UNANIMITY_WEIGHT * unanimity
    )
    return round(min(score, 1.0), 4)
