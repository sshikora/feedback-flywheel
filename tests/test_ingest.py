import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from flywheel.ingest import comparisons, flores, jsonl
from flywheel.ingest.comparisons import load_comparisons
from flywheel.storage import write_jsonl
from tests.factories import make_event

FIXTURE = Path(__file__).parent / "fixtures" / "comparisons_scan.json"


def test_only_voted_comparisons_become_events():
    result = load_comparisons(FIXTURE)
    assert [e.id for e in result.events] == ["cmp_voted-agreement", "cmp_voted-shared-auto"]
    assert result.skipped == {"no_vote": 2, "invalid": 1}


def test_a_vote_on_an_agreement_counts_for_every_model_in_it():
    event = load_comparisons(FIXTURE).events[0]
    assert event.signal.chosen == ["GEMINI_FLASH", "DEEPSEEK_FLASH"]
    assert event.chosen_output == "Asa ang merkado?"


def test_failed_candidates_are_kept_with_their_error():
    gemma = load_comparisons(FIXTURE).events[0].candidates[2]
    assert gemma.output is None
    assert gemma.error == "timed out"


def test_votes_are_recorded_as_not_blind():
    assert all(e.context.blind is False for e in load_comparisons(FIXTURE).events)


def test_only_shared_comparisons_may_be_published():
    private, shared = load_comparisons(FIXTURE).events
    assert (private.consent.training, private.consent.publish) == (True, False)
    assert (shared.consent.training, shared.consent.publish) == (True, True)


def test_auto_source_language_means_unspecified():
    event = load_comparisons(FIXTURE).events[1]
    assert event.input.src_lang is None
    assert event.input.effective_src_lang == "en"


def test_context_keeps_what_the_bias_analysis_needs():
    private, shared = load_comparisons(FIXTURE).events
    assert (private.context.your_model, private.context.model_set) == ("GEMINI_FLASH", "free")
    assert (shared.context.your_model, shared.context.model_set) == ("GPT4O", "pro")
    assert shared.input.tone == "casual"


def test_s3_export_lines_and_gzip_are_accepted(tmp_path):
    items = json.loads(FIXTURE.read_text())["Items"]
    path = tmp_path / "export.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.writelines(json.dumps({"Item": item}) + "\n" for item in items)
    assert len(load_comparisons(path).events) == 2


def test_plain_document_client_objects_are_accepted(tmp_path):
    item = {
        "id": "plain",
        "sourceText": "Hello",
        "sourceLanguage": "en",
        "targetLanguage": "ceb",
        "candidates": [{"label": "A", "model": "GEMMA", "translatedText": "Kumusta"}],
        "isPro": False,
        "createdAt": "2026-10-27T10:00:00.000Z",
        "vote": {
            "label": "A",
            "model": "GEMMA",
            "models": ["GEMMA"],
            "votedAt": "2026-10-27T10:01:00Z",
        },
    }
    path = tmp_path / "plain.json"
    path.write_text(json.dumps([item]))
    assert comparisons.load_comparisons(path).events[0].id == "cmp_plain"


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, [make_event("a"), make_event("b")])
    assert [e.id for e in jsonl.load_events(path)] == ["a", "b"]


def test_jsonl_errors_name_the_file_and_line(tmp_path):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, [make_event("a")])
    with path.open("a") as file:
        file.write('{"id": "broken"}\n')
    with pytest.raises(ValueError, match=r"events\.jsonl:2"):
        jsonl.load_events(path)


def _flores_archive(path: Path, english: str, cebuano: str) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, body in (("eng_Latn", english), ("ceb_Latn", cebuano)):
            data = body.encode()
            info = tarfile.TarInfo(f"./flores200_dataset/devtest/{name}.devtest")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_flores_pairs_line_up_by_position(tmp_path):
    archive = tmp_path / "flores.tar.gz"
    _flores_archive(archive, "One.\nTwo.\n", "Usa.\nDuha.\n")
    segments = flores.load_segments(archive, "en", "ceb", "devtest")
    assert [(s.source, s.reference) for s in segments] == [("One.", "Usa."), ("Two.", "Duha.")]
    assert segments[1].id == "flores200:devtest:en-ceb:1"


def test_flores_rejects_misaligned_files(tmp_path):
    archive = tmp_path / "flores.tar.gz"
    _flores_archive(archive, "One.\nTwo.\n", "Usa.\n")
    with pytest.raises(ValueError, match="misaligned"):
        flores.load_segments(archive, "en", "ceb", "devtest")


def test_flores_rejects_unknown_languages_and_splits(tmp_path):
    with pytest.raises(ValueError, match="no FLORES code"):
        flores.load_segments(tmp_path / "unused", "en", "xx", "devtest")
    with pytest.raises(ValueError, match="unknown split"):
        flores.load_segments(tmp_path / "unused", "en", "ceb", "test")


def test_flores_refuses_an_archive_with_the_wrong_hash(tmp_path):
    (tmp_path / flores.ARCHIVE_NAME).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        flores.fetch_archive(tmp_path)


def test_flores_accepts_an_archive_with_the_pinned_hash(tmp_path, monkeypatch):
    content = b"pretend archive"
    monkeypatch.setattr(flores, "FLORES_SHA256", hashlib.sha256(content).hexdigest())
    (tmp_path / flores.ARCHIVE_NAME).write_bytes(content)
    assert flores.fetch_archive(tmp_path).name == flores.ARCHIVE_NAME
