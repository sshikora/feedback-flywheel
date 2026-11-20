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
