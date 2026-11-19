import bz2
import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from flywheel.curate.leakage import LeakageIndex
from flywheel.ingest import comparisons, flores, jsonl, opus, sources
from flywheel.ingest.comparisons import load_comparisons
from flywheel.schema import SourceText
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
    assert (private.consent.publish, shared.consent.publish) == (False, True)


def test_no_vote_may_train_until_the_privacy_page_names_the_provider():
    assert not any(e.consent.training for e in load_comparisons(FIXTURE).events)


def test_only_votes_cast_after_the_privacy_change_may_train():
    changed = datetime(2026, 10, 27, 11, 0, tzinfo=UTC)
    before, after = load_comparisons(FIXTURE, training_from=changed).events
    assert (before.consent.training, after.consent.training) == (False, True)


def test_a_change_date_without_a_time_zone_is_refused():
    with pytest.raises(ValueError, match="time zone"):
        load_comparisons(FIXTURE, training_from=datetime(2026, 10, 27))


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


def _opus_archive(path: Path, name: str, english: list[str], cebuano: list[str]) -> Path:
    stem = opus.CORPORA[name].opus_name
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr(f"{stem}.ceb-en.en", "".join(line + "\n" for line in english))
        zipped.writestr(f"{stem}.ceb-en.ceb", "".join(line + "\n" for line in cebuano))
    return path


def test_a_corpus_pair_becomes_a_reference_event(tmp_path):
    archive = _opus_archive(tmp_path / "t.zip", "tatoeba", ["Please have a drink."], ["Tagay."])
    (event,) = opus.load_events(archive, "tatoeba").events
    assert event.id == "opus:tatoeba:v2026-07-08:en-ceb:0"
    assert (event.source, event.origin) == ("opus-tatoeba", "human")
    assert (event.input.text, event.chosen_output) == ("Please have a drink.", "Tagay.")
    assert (event.input.src_lang, event.input.tgt_lang) == ("en", "ceb")
    assert event.signal.kind == "reference" and not event.is_vote
    assert event.consent.training and event.consent.publish


def test_the_other_direction_swaps_the_sides(tmp_path):
    archive = _opus_archive(tmp_path / "t.zip", "tatoeba", ["Wait."], ["Hulat."])
    (event,) = opus.load_events(archive, "tatoeba", src="ceb", tgt="en").events
    assert (event.input.text, event.chosen_output) == ("Hulat.", "Wait.")


@pytest.mark.parametrize(
    ("english", "cebuano", "reason"),
    [
        ("Edit $1", "Usba ang $1", "markup"),
        ("{{PLURAL:$1|page|pages}}", "panid", "markup"),
        ("Line<br>break", "Linya", "markup"),
        ("&lt; &lt; A Psalm by David. &gt; &gt;", "Salmo ni David.", "markup"),
        ("Tools", "  ", "empty"),
    ],
)
def test_markup_and_empty_pairs_are_skipped(tmp_path, english, cebuano, reason):
    archive = _opus_archive(tmp_path / "w.zip", "translatewiki", [english], [cebuano])
    result = opus.load_events(archive, "translatewiki")
    assert result.events == []
    assert result.skipped == {reason: 1}


def test_a_corpus_is_refused_for_another_pair_or_when_unknown(tmp_path):
    with pytest.raises(ValueError, match="pinned for ceb-en"):
        opus.load_events(tmp_path / "unused", "tatoeba", src="en", tgt="tl")
    with pytest.raises(ValueError, match="unknown corpus"):
        opus.load_events(tmp_path / "unused", "wikimedia")


def test_a_misaligned_corpus_is_refused(tmp_path):
    archive = _opus_archive(tmp_path / "t.zip", "tatoeba", ["One.", "Two."], ["Usa."])
    with pytest.raises(ValueError, match="misaligned"):
        opus.load_events(archive, "tatoeba")


def test_a_corpus_download_with_the_wrong_hash_is_refused(tmp_path):
    archive = tmp_path / "opus" / opus.CORPORA["tatoeba"].archive_name
    archive.parent.mkdir()
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        opus.fetch_archive("tatoeba", tmp_path)


def test_a_corpus_download_with_the_pinned_hash_is_used(tmp_path, monkeypatch):
    content = b"pretend archive"
    pinned = replace(opus.CORPORA["tatoeba"], sha256=hashlib.sha256(content).hexdigest())
    monkeypatch.setitem(opus.CORPORA, "tatoeba", pinned)
    archive = tmp_path / "opus" / pinned.archive_name
    archive.parent.mkdir()
    archive.write_bytes(content)
    assert opus.fetch_archive("tatoeba", tmp_path) == archive


def test_licenses_cover_only_the_corpora_present():
    found = opus.licenses({"opus-tatoeba": 3, "ai-lang-translation": 9})
    assert list(found) == ["opus-tatoeba"]
    assert found["opus-tatoeba"]["license"] == "CC BY 2.0 FR"
    assert "tatoeba.org" in found["opus-tatoeba"]["attribution"]


def _wikinews_dump(path: Path, pages: list[tuple[str, str, str]]) -> Path:
    """A dump of (namespace, title, wikitext) pages. A title ending in '->' is a redirect."""
    body = "".join(
        f"<page><title>{escape(title.removesuffix('->'))}</title><ns>{namespace}</ns>"
        f"<id>{index}</id>{'<redirect title="x" />' if title.endswith('->') else ''}"
        f"<revision><id>9{index}</id><contributor><id>7</id></contributor>"
        f"<text>{escape(text)}</text></revision></page>"
        for index, (namespace, title, text) in enumerate(pages, start=1)
    )
    xml = f'<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">{body}</mediawiki>'
    path.write_bytes(bz2.compress(xml.encode()))
    return path


def _tatoeba_text(path: Path, lines: list[str]) -> Path:
    path.write_bytes(gzip.compress("".join(line + "\n" for line in lines).encode()))
    return path


_ARTICLE = """{{date|May 1, 2026}}
[[File:Ferry.jpg|thumb|The [[w:Ferry|ferry]] at port]]
The {{w|Philippine Coast Guard|coast guard}} said the ferry left [[w:Cebu City|Cebu]] at noon. \
It reached '''Bohol''' two hours later.<ref>Port log</ref>
{{QuoteLeft|Nobody was hurt.}}
* A list item that is not prose.
== Sources ==
* {{source|url=http://example.com|title=Ferry arrives safely}}
{{publish}}
[[Category:Philippines]]
"""

NO_LEAKS = LeakageIndex([], threshold=0.6)


def test_an_article_keeps_its_prose_and_loses_its_markup():
    assert sources.article_text(_ARTICLE) == (
        "The coast guard said the ferry left Cebu at noon. It reached Bohol two hours later."
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It rained. Then it stopped.", ["It rained.", "Then it stopped."]),
        ('He said "go home." She stayed.', ['He said "go home."', "She stayed."]),
        ("Mr. Smith met Dr. Cruz at 3 p.m. Monday.", ["Mr. Smith met Dr. Cruz at 3 p.m. Monday."]),
        ("George W. Bush spoke in the U.S. Senate.", ["George W. Bush spoke in the U.S. Senate."]),
        ("Is it over? Yes! 2026 was hard.", ["Is it over?", "Yes!", "2026 was hard."]),
        ("One.\nTwo here.", ["One.", "Two here."]),
    ],
)
def test_sentences_split_at_their_ends_only(text, expected):
    assert sources.split_sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Hi.", "too short"),
        ("A" * 300 + ".", "too long"),
        ("See the page at https://example.com today.", "markup"),
        ("the start is missing here.", "not a whole sentence"),
        ("The end is missing here", "not a whole sentence"),
        ('He said "come back tomorrow.', "unbalanced quotes"),
        ("The ferry reached Bohol.", None),
    ],
)
def test_a_sentence_unfit_to_translate_says_why(text, reason):
    assert sources.sentence_problem(text, min_chars=10) == reason


def test_only_published_main_articles_outside_flores_are_read(tmp_path):
    dump = _wikinews_dump(
        tmp_path / "news.xml.bz2",
        [
            ("0", "Ferry arrives safely", _ARTICLE),
            ("0", "Draft story", "A draft that never passed review."),
            ("0", "Old title->", "#REDIRECT [[Ferry arrives safely]]"),
            ("4", "Wikinews:Style guide", "Write in the past tense. {{publish}}"),
            ("0", "Benchmark story", "This story became part of a benchmark. {{Publish}}"),
        ],
    )
    result = sources.load("wikinews", dump, 10, "s", NO_LEAKS, ["Benchmark story"])
    assert {(t.id, t.document, t.text) for t in result.texts} == {
        (
            "wikinews:20261101:1:0",
            "Ferry arrives safely",
            "The coast guard said the ferry left Cebu at noon.",
        ),
        ("wikinews:20261101:1:1", "Ferry arrives safely", "It reached Bohol two hours later."),
    }
    assert result.skipped == {"unpublished articles": 1, "FLORES articles": 1}


def test_a_draw_ignores_file_order_and_grows_without_changing(tmp_path):
    lines = [f"Sentence number {n} is here." for n in range(50)]
    forward = _tatoeba_text(tmp_path / "a.gz", lines)
    backward = _tatoeba_text(tmp_path / "b.gz", lines[::-1])

    def draw(path: Path, size: int, seed: str = "s") -> list[str]:
        return [t.text for t in sources.load("tatoeba", path, size, seed, NO_LEAKS).texts]

    assert draw(forward, 10) == draw(backward, 10)
    assert draw(forward, 20)[:10] == draw(forward, 10)
    assert draw(forward, 10, seed="other") != draw(forward, 10)


def test_a_draw_skips_leaks_and_duplicates_and_caps_each_article(tmp_path):
    flores_text = "The river is high after the rain."
    lines = [flores_text, "Thank you for the dinner.", "Thank you for the dinner!"]
    path = _tatoeba_text(tmp_path / "t.gz", lines)
    result = sources.load("tatoeba", path, 10, "s", LeakageIndex([flores_text], 0.6))
    assert [t.text for t in result.texts] == ["Thank you for the dinner."]
    assert result.skipped == {"duplicate": 1, "leaks into FLORES": 1}
    assert result.available == 2

    article = "{{publish}}\n" + " ".join(f"Reporters confirmed fact number {n}." for n in range(5))
    dump = _wikinews_dump(tmp_path / "n.xml.bz2", [("0", "Many facts", article)])
    drawn = sources.load("wikinews", dump, 10, "s", NO_LEAKS).texts
    assert len(drawn) == sources.MAX_PER_ARTICLE


def test_a_tatoeba_line_becomes_a_source_text(tmp_path):
    path = _tatoeba_text(tmp_path / "t.gz", ["I woke up at five."])
    (text,) = sources.load("tatoeba", path, 1, "s", NO_LEAKS).texts
    assert text == SourceText(
        id="tatoeba:v2026-07-08:en:0", corpus="tatoeba", document=None, text="I woke up at five."
    )


def test_flores_links_name_wikinews_titles():
    assert sources.wikinews_title(
        "https://en.wikinews.org/wiki/Fighter_jet_crashes_during_Children%27s_Day_airshow"
    ) == ("Fighter jet crashes during Children's Day airshow")
    assert sources.wikinews_title("https://en.wikivoyage.org/wiki/Cebu") is None


def test_flores_source_urls_come_from_both_metadata_files(tmp_path):
    archive = tmp_path / "flores.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for split, url in (("dev", "https://a.example/1"), ("devtest", "https://b.example/2")):
            data = f"URL\tdomain\n{url}\twikinews\n{url}\twikinews\n".encode()
            info = tarfile.TarInfo(f"./flores200_dataset/metadata_{split}.tsv")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    assert flores.source_urls(archive) == {"https://a.example/1", "https://b.example/2"}


def test_a_source_download_with_the_wrong_hash_is_refused(tmp_path):
    archive = tmp_path / "sources" / sources.SOURCES["tatoeba"].archive_name
    archive.parent.mkdir()
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        sources.fetch_archive("tatoeba", tmp_path)
    with pytest.raises(ValueError, match="unknown source"):
        sources.fetch_archive("wikipedia", tmp_path)
