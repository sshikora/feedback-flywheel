"""Stop evaluation text from reaching training.

An event leaks if any of its text, input or candidate output, matches any eval text
exactly or nearly. Checking both sides in both directions catches a Cebuano input that
matches a FLORES reference, and a model output that reproduces one. Near-duplicates are
found by Jaccard similarity over character trigrams.

Comparing every text that shares a trigram does not scale, because common trigrams
are shared by nearly everything. Three exact filters from the set similarity join
literature (AllPairs and PPJoin) cut the comparisons without missing a match:

- Size: Jaccard of at least t needs the two trigram counts within a factor of t.
- Prefix: sort each text's trigrams rarest first, in one global order. A pair at t
  shares at least ceil(t / (1 + t) * (n + m)) trigrams, so it must share one among
  the first n - ceil(t * n) + 1 of each. Only those prefixes are indexed and probed.
- Position: at each shared prefix trigram, the overlap found so far plus the trigrams
  left after it in the shorter remainder must still reach that minimum.

Candidates that survive are checked in full.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from flywheel.schema import EvalSegment, FeedbackEvent
from flywheel.text import normalize_text


def trigrams(normalized: str) -> set[str]:
    if len(normalized) < 3:
        return {normalized}
    return {normalized[i : i + 3] for i in range(len(normalized) - 2)}


class LeakageIndex:
    def __init__(self, texts: Iterable[str], threshold: float) -> None:
        self._threshold = threshold
        self._exact = {normalize_text(text) for text in texts}
        self._grams = [trigrams(normalized) for normalized in sorted(self._exact)]
        self._frequency = Counter(gram for grams in self._grams for gram in grams)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, grams in enumerate(self._grams):
            for position, gram in enumerate(self._prefix(self._ordered(grams))):
                self._postings[gram].append((index, position))

    def _ordered(self, grams: set[str]) -> list[str]:
        return sorted(grams, key=lambda gram: (self._frequency[gram], gram))

    def _prefix(self, ordered: list[str]) -> list[str]:
        return ordered[: len(ordered) - self._at_least(self._threshold * len(ordered)) + 1]

    @staticmethod
    def _at_least(value: float) -> int:
        """The smallest whole count at or above value. The margin absorbs float error,
        as in 0.6 * 5, which would otherwise round up past an exact 3."""
        return math.ceil(value - 1e-9)

    def matches(self, text: str) -> bool:
        normalized = normalize_text(text)
        if normalized in self._exact:
            return True
        ordered = self._ordered(trigrams(normalized))
        size = len(ordered)
        low, high = self._threshold * size, size / self._threshold
        found: dict[int, int] = {}
        for i, gram in enumerate(self._prefix(ordered)):
            for candidate, j in self._postings.get(gram, ()):
                overlap = found.get(candidate, 0)
                if overlap < 0:
                    continue
                other = len(self._grams[candidate])
                need = self._at_least(self._threshold / (1 + self._threshold) * (size + other))
                if not low <= other <= high or overlap + min(size - i, other - j) < need:
                    found[candidate] = -1
                else:
                    found[candidate] = overlap + 1
        grams = set(ordered)
        return any(
            self._jaccard(grams, self._grams[candidate]) >= self._threshold
            for candidate, overlap in found.items()
            if overlap > 0
        )

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        overlap = len(left & right)
        return overlap / (len(left) + len(right) - overlap)


def event_texts(event: FeedbackEvent) -> list[str]:
    return [event.input.text, *(c.output for c in event.answered_candidates if c.output)]


def segment_texts(segments: Iterable[EvalSegment]) -> Iterable[str]:
    for segment in segments:
        yield segment.source
        yield segment.reference


def leaks(event: FeedbackEvent, index: LeakageIndex) -> bool:
    return any(index.matches(text) for text in event_texts(event))
