"""Scores with uncertainty. A difference whose interval includes zero is a tie."""

import hashlib
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from sacrebleu.metrics.chrf import CHRF

# word_order=2 is what makes this chrF++ and not plain chrF.
_CHRF_PLUS_PLUS = CHRF(word_order=2)

Stats = NDArray[np.int64]
Weights = NDArray[np.int64]
Values = NDArray[np.float64]


class Interval(NamedTuple):
    value: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def seed_from(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def resample_weights(size: int, resamples: int, seed: str) -> Weights:
    """How often each sentence appears in each resample.

    Sharing these across models pairs the comparison, so a hard sentence counts
    against both sides of a difference.
    """
    generator = np.random.default_rng(seed_from(seed))
    return generator.multinomial(size, np.full(size, 1 / size), size=resamples)


def chrf_stats(outputs: list[str], references: list[str]) -> Stats:
    """Per-sentence counts. An empty output scores as a miss, not as a gap."""
    return np.array(_CHRF_PLUS_PLUS._extract_corpus_statistics(outputs, [references]))


def chrf(stats: Stats) -> float:
    """Corpus chrF++ from summed counts, or from one sentence's counts."""
    total = stats if stats.ndim == 1 else stats.sum(axis=0)
    return float(_CHRF_PLUS_PLUS._compute_f_score(total.tolist()))


def _interval(value: float, resampled: Values) -> Interval:
    low, high = np.percentile(resampled, [2.5, 97.5])
    return Interval(value, float(low), float(high))


def _chrf_resamples(stats: Stats, weights: Weights) -> Values:
    return np.array([chrf(total) for total in weights @ stats])


def chrf_interval(stats: Stats, weights: Weights) -> Interval:
    return _interval(chrf(stats), _chrf_resamples(stats, weights))


def chrf_difference(first: Stats, second: Stats, weights: Weights) -> Interval:
    """First minus second, resampled together."""
    resampled = _chrf_resamples(first, weights) - _chrf_resamples(second, weights)
    return _interval(chrf(first) - chrf(second), resampled)


def mean_interval(values: Values, weights: Weights) -> Interval:
    return _interval(float(values.mean()), (weights @ values) / len(values))


def difference_interval(first: Values, second: Values, weights: Weights) -> Interval:
    return mean_interval(first - second, weights)


def _squared_gaps(values: Values) -> float:
    return float(((values[:, None] - values[None, :]) ** 2).sum())


def agreement(units: list[list[float]]) -> float | None:
    """Krippendorff's alpha for interval data. 1 is perfect, 0 is chance.

    Each unit holds the scores given to one thing. Pass one score per rater for
    agreement between raters, or one per round for one rater's consistency.
    """
    scored = [np.array(unit, dtype=float) for unit in units if len(unit) >= 2]
    if not scored:
        return None
    every_score = np.concatenate(scored)
    total = len(every_score)
    expected = _squared_gaps(every_score) / (total * (total - 1))
    if expected == 0:
        return 1.0
    observed = sum(_squared_gaps(unit) / (len(unit) - 1) for unit in scored) / total
    return 1 - observed / expected
