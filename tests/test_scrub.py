import pytest

from flywheel.schema import Candidate
from flywheel.scrub import scrub_events, scrub_text
from tests.factories import make_event

# Assembled at runtime so that secret scanners do not flag obvious fakes.
FAKE_SECRETS = {
    "aws": "AKIA" + "ABCDEFGHIJKLMNOP",
    "github": "ghp_" + "a1" * 18,
    "api_key": "sk-" + "abc123" * 5,
    "slack": "xoxb-" + "123456789012-abcdef",
    "jwt": "eyJ" + "hbGciOiJIUzI1." + "eyJzdWIiOiIxMjM0." + "SflKxwRJSMeKKF2QT4",
    "bearer": "Bearer " + "abcdefghijklmnopqrstuvwxyz",
    "assignment": "password=" + "hunter2hunter2",
    "url": "postgres://admin:" + "s3cret@db.example.com/prod",
    "key_block": "-----BEGIN RSA " + "PRIVATE KEY-----",
}


@pytest.mark.parametrize("secret", FAKE_SECRETS.values(), ids=FAKE_SECRETS.keys())
def test_secrets_are_detected(secret):
    assert scrub_text(f"Please translate {secret} now").secrets


@pytest.mark.parametrize(
    ("text", "placeholder"),
    [
        ("Email maria.santos+x@example.com please", "[EMAIL]"),
        ("Tawag sa +63 917 123 4567 ugma", "[PHONE]"),
        ("Call 0917 123 4567 tomorrow", "[PHONE]"),
        ("Call (415) 555-2671 now", "[PHONE]"),
        ("Call 415-555-2671 now", "[PHONE]"),
        ("Card 4111 1111 1111 1111 expires soon", "[CARD]"),
        ("SSN is 078-05-1120", "[SSN]"),
        ("Server at 192.168.1.20 is down", "[IP]"),
    ],
)
def test_personal_data_is_redacted(text, placeholder):
    result = scrub_text(text)
    assert placeholder in result.text
    assert result.pii


@pytest.mark.parametrize(
    "text",
    [
        "It costs 1,000,000 pesos.",
        "The meeting is on 2026-10-31.",
        "Pi is about 3.14159265.",
        "Population 113,900,000 in 2023",
        "Room 12 on floor 3",
        "Order 4111 1111 1111 1112 failed the checksum",
        "Version 300.400.500.600 is not an address",
        "He ran 100 metres in 9.58 seconds.",
        "@example is a handle, not an address.",
    ],
)
def test_ordinary_numbers_are_left_alone(text):
    result = scrub_text(text)
    assert result.text == text
    assert result.pii == ()


def test_redaction_reaches_the_input_and_every_candidate():
    event = make_event(
        text="Write to ana@example.com today",
        outputs={"A": "Sulat sa ana@example.com karon", "B": "Sulati si ana@example.com"},
    )
    result = scrub_events([event])
    stored = result.events[0]
    assert result.redacted == 1
    assert "ana@example.com" not in stored.model_dump_json()
    assert stored.input.text == "Write to [EMAIL] today"


def test_a_secret_in_a_candidate_drops_the_whole_event():
    secret = FAKE_SECRETS["api_key"]
    event = make_event(outputs={"A": f"Ang yawe mao ang {secret}", "B": "Wala"}, chosen=["B"])
    result = scrub_events([event])
    assert result.events == []
    assert result.dropped_for_secrets == 1


def test_a_secret_in_a_provider_error_drops_the_whole_event():
    candidates = [
        Candidate(label="A", model_id="A", output="Maayo"),
        Candidate(label="B", model_id="B", error=f"401 for key {FAKE_SECRETS['api_key']}"),
    ]
    result = scrub_events([make_event(candidates=candidates, chosen=["A"])])
    assert result.dropped_for_secrets == 1


def test_clean_events_pass_unchanged():
    event = make_event()
    result = scrub_events([event])
    assert result.events == [event]
    assert result.redacted == 0
