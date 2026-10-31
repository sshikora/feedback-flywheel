"""Text normalization shared by dedupe, agreement grouping and leakage checks."""

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCTUATION = re.compile(r"[\s.!?…。]+$")


def normalize_text(text: str) -> str:
    """Fold differences that do not change an answer.

    Matches `normalizeAnswer` in the translation app, so two answers the app
    grouped as an agreement stay grouped here. NFKC folds full-width punctuation
    into its ASCII form.
    """
    folded = unicodedata.normalize("NFKC", text).strip()
    folded = _WHITESPACE.sub(" ", folded).lower()
    return _TRAILING_PUNCTUATION.sub("", folded)
