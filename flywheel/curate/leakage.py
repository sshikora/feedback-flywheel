"""Stop evaluation text from reaching training.

An event leaks if any of its text, input or candidate output, matches any eval text
exactly or nearly. Checking both sides in both directions catches a Cebuano input that
matches a FLORES reference, and a model output that reproduces one. Near-duplicates are
found by Jaccard similarity over character trigrams, with an inverted index so only
texts sharing a trigram are compared.
"""

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
        self._exact: set[str] = set()
        self._grams: list[set[str]] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        for text in texts:
            self._add(normalize_text(text))

    def _add(self, normalized: str) -> None:
        if normalized in self._exact:
            return
        self._exact.add(normalized)
        grams = trigrams(normalized)
        index = len(self._grams)
        self._grams.append(grams)
        for gram in grams:
            self._postings[gram].append(index)

    def matches(self, text: str) -> bool:
        normalized = normalize_text(text)
        if normalized in self._exact:
            return True
        grams = trigrams(normalized)
        shared = Counter(i for gram in grams for i in self._postings.get(gram, ()))
        return any(
            overlap / (len(grams) + len(self._grams[i]) - overlap) >= self._threshold
            for i, overlap in shared.items()
        )


def event_texts(event: FeedbackEvent) -> list[str]:
    return [event.input.text, *(c.output for c in event.answered_candidates if c.output)]


def segment_texts(segments: Iterable[EvalSegment]) -> Iterable[str]:
    for segment in segments:
        yield segment.source
        yield segment.reference


def leaks(event: FeedbackEvent, index: LeakageIndex) -> bool:
    return any(index.matches(text) for text in event_texts(event))
