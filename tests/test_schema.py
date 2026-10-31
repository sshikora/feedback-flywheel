import pytest
from pydantic import ValidationError

from flywheel.schema import Candidate, FeedbackEvent, TranslationInput
from flywheel.schema.export import render_schemas, stale_schemas, write_schemas
from tests.factories import make_event


def test_candidate_needs_exactly_one_of_output_or_error():
    with pytest.raises(ValidationError):
        Candidate(label="A", model_id="m")
    with pytest.raises(ValidationError):
        Candidate(label="A", model_id="m", output="x", error="y")


def test_chosen_model_must_have_answered():
    with pytest.raises(ValidationError, match="no output"):
        make_event(outputs={"A": "x"}, chosen=["MISSING"])


def test_unknown_fields_are_rejected():
    event = make_event().model_dump(mode="json")
    event["surprise"] = True
    with pytest.raises(ValidationError):
        FeedbackEvent.model_validate(event)


def test_effective_source_language_falls_back_to_detected_then_undetermined():
    assert TranslationInput(text="x", src_lang="en", tgt_lang="ceb").effective_src_lang == "en"
    detected = TranslationInput(text="x", detected_src_lang="fr", tgt_lang="ceb")
    assert detected.effective_src_lang == "fr"
    assert TranslationInput(text="x", tgt_lang="ceb").effective_src_lang == "und"


def test_events_survive_a_json_round_trip():
    event = make_event()
    assert FeedbackEvent.model_validate_json(event.model_dump_json()) == event


def test_chosen_output_is_the_first_answer_from_a_chosen_model():
    event = make_event(outputs={"A": "one", "B": "two"}, chosen=["B"])
    assert event.chosen_output == "two"


def test_schema_export_detects_stale_files(tmp_path):
    assert set(stale_schemas(tmp_path)) == set(render_schemas())
    write_schemas(tmp_path)
    assert stale_schemas(tmp_path) == []
    (tmp_path / "feedback_event.schema.json").write_text("{}")
    assert stale_schemas(tmp_path) == ["feedback_event.schema.json"]
