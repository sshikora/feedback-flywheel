import json
from datetime import UTC, datetime

import httpx
import pytest

from flywheel.evals import translate as translating
from flywheel.ingest import sources, teacher
from flywheel.schema import FeedbackEvent, SourceText, TranslationRecord

TEXT = SourceText(id="tatoeba:v:en:7", corpus="tatoeba", document=None, text="I woke up at five.")


def answered(output: str | None = "Alas singko ko mibangon.") -> dict:
    record = (
        TranslationRecord(segment_id=TEXT.id, model_id=teacher.TEACHER, output=output)
        if output
        else TranslationRecord(segment_id=TEXT.id, model_id=teacher.TEACHER, error="broken")
    )
    return {(teacher.TEACHER, TEXT.id): record}


def test_a_teacher_answer_becomes_a_synthetic_event():
    (event,) = teacher.to_events([TEXT], answered())
    assert event.id == f"teacher:{teacher.TEACHER}:{TEXT.id}"
    assert (event.source, event.origin) == ("teacher-tatoeba", "synthetic")
    assert (event.input.text, event.chosen_output) == (TEXT.text, "Alas singko ko mibangon.")
    assert (event.input.src_lang, event.input.tgt_lang) == ("en", "ceb")
    assert event.signal.kind == "teacher" and not event.is_vote
    assert event.created_at == sources.SOURCES["tatoeba"].released
    assert event.consent.training
    assert FeedbackEvent.model_validate_json(event.model_dump_json()) == event


def test_failed_or_missing_answers_make_no_event():
    assert teacher.to_events([TEXT], answered(None)) == []
    assert teacher.to_events([TEXT], {}) == []


def test_the_prompt_is_the_apps_with_english_named():
    assert teacher.prompt(TEXT) == (
        "Translate the following text from English to Cebuano. Return only the translated "
        "text with no explanation or commentary.\n\nI woke up at five."
    )


def test_the_estimate_grows_with_the_text():
    longer = TEXT.model_copy(update={"text": TEXT.text * 10})
    assert 0 < teacher.estimate_usd([TEXT]) < teacher.estimate_usd([longer])
    assert teacher.estimate_usd([TEXT, TEXT]) == 2 * teacher.estimate_usd([TEXT])


def test_licenses_credit_the_source_text_and_the_teacher():
    found = teacher.licenses({"teacher-wikinews": 4, "opus-tatoeba": 2})
    assert list(found) == ["teacher-wikinews"]
    assert found["teacher-wikinews"]["license"].startswith("CC BY 2.5")
    assert "DeepSeek V4.1 Flash" in found["teacher-wikinews"]["attribution"]


# 2026-11-20 is a Friday and 2026-11-21 a Saturday.
PEAK = datetime(2026, 11, 20, 2, 30, tzinfo=UTC)
OFF_PEAK = datetime(2026, 11, 20, 5, 0, tzinfo=UTC)
WEEKEND = datetime(2026, 11, 21, 2, 30, tzinfo=UTC)
USAGE = {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 400, "completion_tokens": 500}


def deepseek(handler, *times: datetime) -> teacher.DeepSeek:
    clock = iter(times or (OFF_PEAK, OFF_PEAK))
    return teacher.DeepSeek(
        "key", httpx.MockTransport(handler), wait=lambda _: None, now=lambda: next(clock)
    )


def answer(text: str = " Kumusta \n", usage: dict | None = None) -> httpx.Response:
    body = {"choices": [{"message": {"content": text}}], "usage": usage or USAGE}
    return httpx.Response(200, json=body)


def test_peak_hours_are_weekday_windows_in_utc():
    assert teacher.is_peak(PEAK)
    assert not teacher.is_peak(OFF_PEAK)
    assert not teacher.is_peak(WEEKEND)
    assert teacher.is_peak(datetime(2026, 11, 20, 9, 59, tzinfo=UTC))
    assert not teacher.is_peak(datetime(2026, 11, 20, 10, 0, tzinfo=UTC))


def test_cost_counts_cached_input_and_halves_off_peak():
    peak = 400 * 0.006e-6 + 600 * 0.30e-6 + 500 * 1.20e-6
    assert teacher.cost_usd(USAGE, peak=True) == pytest.approx(peak)
    assert teacher.cost_usd(USAGE, peak=False) == pytest.approx(peak / 2)


def test_asks_deepseek_directly_with_thinking_off():
    seen = {}

    def handler(request):
        seen.update(url=str(request.url), auth=request.headers["authorization"])
        seen.update(json.loads(request.content))
        return answer()

    result = deepseek(handler).complete(teacher.TEACHER, "prompt")
    assert result.text == "Kumusta"
    assert result.cost_usd == pytest.approx(teacher.cost_usd(USAGE, peak=False))
    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["auth"] == "Bearer key"
    assert seen["model"] == teacher.API_MODEL
    assert seen["thinking"] == {"type": "disabled"}
    assert "provider" not in seen


def test_a_request_that_touches_peak_hours_is_charged_at_peak():
    result = deepseek(lambda _: answer(), OFF_PEAK, PEAK).complete(teacher.TEACHER, "p")
    assert result.cost_usd == pytest.approx(teacher.cost_usd(USAGE, peak=True))


def test_retries_and_rejections_match_openrouter():
    responses = [httpx.Response(429), httpx.Response(503), answer()]
    assert deepseek(lambda _: responses.pop(0)).complete(teacher.TEACHER, "p").text == "Kumusta"
    no_credit = httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})
    with pytest.raises(translating.RequestRejected, match="Insufficient Balance"):
        deepseek(lambda _: no_credit).complete(teacher.TEACHER, "p")
    with pytest.raises(translating.CompletionError, match="empty answer"):
        deepseek(lambda _: answer("  ")).complete(teacher.TEACHER, "p")


def test_missing_token_counts_stop_the_run():
    with pytest.raises(KeyError):
        deepseek(lambda _: answer(usage={"total_tokens": 9})).complete(teacher.TEACHER, "p")
