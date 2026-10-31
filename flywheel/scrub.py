"""Remove personal data and secrets before anything is stored.

Secrets drop the whole event, since a translation pair is worthless next to the
risk of keeping a live credential. Personal data is redacted to a placeholder in
the input and in every candidate, because models echo it back.

These are patterns, so they cannot catch names or street addresses. The primary
protection is that the source app stores no account link.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from flywheel.schema import Candidate, FeedbackEvent


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _digits(match: str) -> str:
    return re.sub(r"\D", "", match)


def _is_card(match: str) -> bool:
    digits = _digits(match)
    return 13 <= len(digits) <= 19 and _luhn_valid(digits)


def _is_phone(match: str) -> bool:
    return 9 <= len(_digits(match)) <= 15


def _is_ipv4(match: str) -> bool:
    return all(int(octet) <= 255 for octet in match.split("."))


SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{22,})\b"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "jwt": re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}"),
    "bearer_token": re.compile(r"(?i)\bbearer\s+[\w.~+/=-]{20,}"),
    "credential_assignment": re.compile(
        r"(?i)\b(?:password|passwd|api[_-]?key|secret)\s*[:=]\s*\S{6,}"
    ),
    "url_credentials": re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
}

# Order matters: a card number would otherwise also match the phone pattern.
PII_PATTERNS: list[tuple[str, re.Pattern[str], Callable[[str], bool] | None]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None),
    ("CARD", re.compile(r"(?<![\d.])\d(?:[ -]?\d){12,18}(?![\d.])"), _is_card),
    ("IP", re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])"), _is_ipv4),
    (
        "PHONE",
        re.compile(
            r"(?<![\w.,])(?<!\d[ .-])(?:\+\d{1,3}[ .-]?)?"
            r"(?:\(\d{1,4}\)[ .-]?\d{3,4}|\d{2,4}[ .-]\d{3,4})"
            r"[ .-]?\d{3,4}(?![\w,]|\.\d|[ .-]\d)"
        ),
        _is_phone,
    ),
]


@dataclass(frozen=True)
class ScrubResult:
    text: str
    secrets: tuple[str, ...]
    pii: tuple[str, ...]


def scrub_text(text: str) -> ScrubResult:
    secrets = tuple(kind for kind, pattern in SECRET_PATTERNS.items() if pattern.search(text))
    found: list[str] = []

    def redact(placeholder: str, is_valid: Callable[[str], bool] | None) -> Callable[..., str]:
        def replace(match: re.Match[str]) -> str:
            if is_valid is not None and not is_valid(match.group()):
                return match.group()
            found.append(placeholder)
            return f"[{placeholder}]"

        return replace

    for placeholder, pattern, is_valid in PII_PATTERNS:
        text = pattern.sub(redact(placeholder, is_valid), text)
    return ScrubResult(text=text, secrets=secrets, pii=tuple(found))


@dataclass(frozen=True)
class ScrubbedEvents:
    events: list[FeedbackEvent]
    redacted: int
    dropped_for_secrets: int


def scrub_events(events: Iterable[FeedbackEvent]) -> ScrubbedEvents:
    """Return only what is safe to store. Dropped events leave no text behind."""
    kept: list[FeedbackEvent] = []
    redacted = dropped = 0
    for event in events:
        outcome = _scrub_event(event)
        if outcome is None:
            dropped += 1
            continue
        kept.append(outcome)
        redacted += outcome != event
    return ScrubbedEvents(kept, redacted, dropped)


def _scrub_event(event: FeedbackEvent) -> FeedbackEvent | None:
    source = scrub_text(event.input.text)
    if source.secrets:
        return None
    candidates: list[Candidate] = []
    for candidate in event.candidates:
        output = scrub_text(candidate.output) if candidate.output is not None else None
        error = scrub_text(candidate.error) if candidate.error is not None else None
        if (output and output.secrets) or (error and error.secrets):
            return None
        candidates.append(
            candidate.model_copy(
                update={
                    "output": output.text if output else None,
                    "error": error.text if error else None,
                }
            )
        )
    return event.model_copy(
        update={
            "input": event.input.model_copy(update={"text": source.text}),
            "candidates": candidates,
        }
    )
