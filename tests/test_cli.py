"""End to end through the command line, with fixture data only."""

import csv
import gzip
import json
import re
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flywheel.cli import app
from flywheel.evals import translate as translating
from flywheel.ingest import flores, opus, sources, teacher
from flywheel.schema import CuratedEvent, EvalSegment, HumanRating, SourceText, TranslationRecord
from flywheel.storage import read_jsonl, write_jsonl
from tests.factories import make_event, make_reference

FIXTURE = Path(__file__).parent / "fixtures" / "comparisons_scan.json"
runner = CliRunner()


def run(*args: object) -> str:
    result = runner.invoke(app, [str(arg) for arg in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def fail(*args: object) -> str:
    """The error text of a refused command, without the colors, box and wrapping around it."""
    result = runner.invoke(app, [str(arg) for arg in args])
    assert result.exit_code != 0, result.output
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", text).split())


@pytest.fixture
def flores_cache(tmp_path, monkeypatch):
    """A tiny archive that repeats one fixture sentence, as if it were benchmark text."""
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / flores.ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as tar:
        for name, body in (
            ("eng_Latn", b"Thank you very much for dinner.\nThe river is high.\n"),
            ("ceb_Latn", b"Salamat kaayo sa panihapon.\nTaas ang suba.\n"),
        ):
            info = tarfile.TarInfo(f"./flores200_dataset/devtest/{name}.devtest")
            info.size = len(body)
            tar.addfile(info, BytesIO(body))
    monkeypatch.setattr(flores, "_verify", lambda _archive: None)
    return cache


def test_ingest_then_curate(tmp_path, flores_cache):
    data = tmp_path / "data"
    output = run(
        "ingest", "comparisons", FIXTURE, "--data-dir", data, "--training-from", "2026-10-27"
    )
    assert "skipped 2 (no_vote)" in output
    assert "2 of 2 votes cleared for training" in output
    assert "stored 2 events" in output

    output = run("ingest", "flores", "--data-dir", data, "--cache-dir", flores_cache)
    assert "stored 2 devtest segments" in output
    assert len(read_jsonl(data / "eval" / "flores200-devtest-en-ceb.jsonl", EvalSegment)) == 2

    config = tmp_path / "curation.toml"
    config.write_text("heldout_pct = 0\n")
    output = run("curate", "--data-dir", data, "--config", config)

    train = read_jsonl(data / "curated" / "train.jsonl", CuratedEvent)
    assert [item.event.id for item in train] == ["cmp_voted-agreement"]

    report = json.loads((data / "curated" / "report.json").read_text())
    assert report["dropped"] == {"leak_fixed_eval": 1}
    assert report["train_origin"]["human_share"] == 1.0
    assert report["vote_statistics"]["blind_share"] == 0.0

    output = run("build", "sft", "--data-dir", data)
    assert "built" in output and "(1 examples, 1 events)" in output
    (snapshot,) = (data / "datasets").iterdir()
    assert snapshot.name.startswith("sft-")
    assert "already built, unchanged" in run("build", "sft", "--data-dir", data)


def test_ingest_never_writes_a_secret_or_personal_data(tmp_path):
    secret = "sk-" + "abc123" * 5
    events = [
        make_event("secret", text=f"My key is {secret} please translate"),
        make_event("personal", text="Write to ana@example.com today", minutes=1),
    ]
    source = tmp_path / "input.jsonl"
    write_jsonl(source, events)
    data = tmp_path / "data"

    output = run("ingest", "jsonl", source, "--data-dir", data)

    stored = (data / "events" / "jsonl.jsonl").read_text()
    assert "1 redacted, 1 dropped for secrets" in output
    assert secret not in stored
    assert "ana@example.com" not in stored
    assert "[EMAIL]" in stored


def test_default_config_file_matches_the_code_defaults():
    from flywheel.curate.config import CurationConfig

    path = Path(__file__).parent.parent / "configs" / "curation.default.toml"
    assert CurationConfig.from_toml(path) == CurationConfig()


def test_schema_check_passes_after_export_and_fails_before(tmp_path):
    out = tmp_path / "schemas"
    assert runner.invoke(app, ["schema", "--out", str(out), "--check"]).exit_code == 1
    run("schema", "--out", out)
    assert "current" in run("schema", "--out", out, "--check")


def test_build_register_and_card(tmp_path):
    data = tmp_path / "data"
    run("ingest", "comparisons", FIXTURE, "--data-dir", data, "--training-from", "2026-10-27")
    config = tmp_path / "curation.toml"
    config.write_text("heldout_pct = 0\n")
    run("curate", "--data-dir", data, "--config", config)
    run("build", "sft", "--data-dir", data)
    (snapshot,) = (data / "datasets").iterdir()
    digest = json.loads((snapshot / "manifest.json").read_text())["hash"]

    example = Path(__file__).parent.parent / "configs" / "run.example.toml"
    run_config = tmp_path / "v1.toml"
    run_config.write_text(
        example.read_text().replace("0" * 64, digest).replace("example-qwen3.5-9b-sft", "v1")
    )
    assert "registered v1 as a candidate" in run(
        "registry", "add", run_config, "--provider", "together", "--cost", "3.5", "--data-dir", data
    )
    assert "v1  candidate  sft  Qwen/Qwen3.5-9B" in run("registry", "list", "--data-dir", data)
    run("registry", "card", "v1", "--data-dir", data)
    card = (data / "registry" / "cards" / "v1.md").read_text()
    assert "**Size:** 2 records from 2 curated events" in card
    assert "**Cost:** $3.50" in card

    run_config.write_text(run_config.read_text().replace("epochs = 3", "epochs = 4"))
    assert "has changed since v1 was trained" in fail("registry", "card", "v1", "--data-dir", data)


def test_a_run_config_must_name_a_built_snapshot(tmp_path):
    example = Path(__file__).parent.parent / "configs" / "run.example.toml"
    output = fail("registry", "add", example, "--provider", "together", "--data-dir", tmp_path)
    assert f"has hash {'0' * 64}" in output


def test_ingest_a_corpus_and_build_from_it(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    archive = cache / "opus" / opus.CORPORA["tatoeba"].archive_name
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Tatoeba.ceb-en.en", "I woke up at five.\nEdit $1\n")
        zipped.writestr("Tatoeba.ceb-en.ceb", "Alas singko ko mibangon.\nUsba ang $1\n")
    monkeypatch.setattr(opus, "_verify", lambda _archive, _spec: None)
    data = tmp_path / "data"

    output = run("ingest", "opus", "tatoeba", "--data-dir", data, "--cache-dir", cache)
    assert "skipped 1 (markup)" in output
    assert "CC BY 2.0 FR" in output
    assert "stored 1 events" in output

    config = tmp_path / "curation.toml"
    config.write_text("heldout_pct = 0\n")
    run("curate", "--data-dir", data, "--config", config)
    run("build", "sft", "--data-dir", data)
    (snapshot,) = (data / "datasets").iterdir()
    card = (snapshot / "DATA_CARD.md").read_text()
    assert "| human:opus-tatoeba | 1 |" in card
    assert "**opus-tatoeba:** CC BY 2.0 FR." in card


def test_an_unknown_corpus_is_refused(tmp_path):
    assert "unknown corpus 'wikimedia'" in fail(
        "ingest", "opus", "wikimedia", "--data-dir", tmp_path
    )


def test_ingest_english_source_text(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "sources").mkdir(parents=True)
    with tarfile.open(cache / flores.ARCHIVE_NAME, "w:gz") as tar:
        for path, body in (
            ("dev/eng_Latn.dev", "The river is high after the rain.\n"),
            ("dev/ceb_Latn.dev", "Taas ang suba human sa ulan.\n"),
            ("devtest/eng_Latn.devtest", "Thank you very much for dinner.\n"),
            ("devtest/ceb_Latn.devtest", "Salamat kaayo sa panihapon.\n"),
            ("metadata_dev.tsv", "URL\tdomain\nhttps://en.wikinews.org/wiki/Rain\twikinews\n"),
            ("metadata_devtest.tsv", "URL\tdomain\nhttps://example.org/dinner\twikibooks\n"),
        ):
            info = tarfile.TarInfo(f"./flores200_dataset/{path}")
            info.size = len(body.encode())
            tar.addfile(info, BytesIO(body.encode()))
    lines = ["Thank you very much for dinner!", "I woke up at five.", "Ok."]
    archive = cache / "sources" / sources.SOURCES["tatoeba"].archive_name
    archive.write_bytes(gzip.compress("".join(line + "\n" for line in lines).encode()))
    monkeypatch.setattr(flores, "_verify", lambda _archive: None)
    monkeypatch.setattr(sources, "_verify", lambda _archive, _spec: None)
    data = tmp_path / "data"

    output = run("ingest", "source", "tatoeba", "--data-dir", data, "--cache-dir", cache)
    assert "skipped 1 (leaks into FLORES)" in output
    assert "skipped 1 (too short)" in output
    assert "CC BY 2.0 FR" in output
    assert "stored 1 of 2 sentences" in output
    (text,) = read_jsonl(data / "sources" / "tatoeba.jsonl", SourceText)
    assert text.text == "I woke up at five."
    assert "unknown source 'wikipedia'" in fail("ingest", "source", "wikipedia")


class FlakyTeacher:
    """Answers every prompt, except that the first one naming `fail` fails once."""

    def __init__(self, fail: str) -> None:
        self.prompts: list[str] = []
        self.fail = fail

    def __call__(self, *_args: object, **_kwargs: object) -> "FlakyTeacher":
        return self

    def complete(self, model: str, prompt: str) -> translating.Completion:
        self.prompts.append(prompt)
        if self.fail in prompt:
            self.fail = "\x00never"
            raise translating.CompletionError("broken")
        return translating.Completion(f"Cebuano for: {prompt.rsplit(chr(10), 1)[-1]}", 0.001)


def test_teacher_translates_stored_text_into_synthetic_events(tmp_path, monkeypatch):
    data = tmp_path / "data"
    texts = [
        SourceText(id=f"tatoeba:v:en:{n}", corpus="tatoeba", document=None, text=text)
        for n, text in enumerate(
            ["I woke up at five.", "The market opens early.", "Where is the church?"]
        )
    ]
    write_jsonl(data / "sources" / "tatoeba.jsonl", texts)
    human = make_event("h1", text="Where is the church?", outputs={"human:x": "Asa ang simbahan?"})
    write_jsonl(data / "events" / "opus-tatoeba.jsonl", [human])
    fake = FlakyTeacher(fail="market")
    monkeypatch.setattr(teacher, "DeepSeek", fake)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")

    output = run("ingest", "teacher", "tatoeba", "--data-dir", data)
    assert "skipped 1 (already has a human translation)" in output
    assert "2 sentences for DeepSeek V4.1 Flash" in output
    assert "Nothing was sent" in output
    assert fake.prompts == []
    assert not (data / "events" / "teacher-tatoeba.jsonl").exists()

    output = run("ingest", "teacher", "tatoeba", "--yes", "--data-dir", data)
    assert "1 failed" in output
    assert "stored 1 events" in output
    output = run("ingest", "teacher", "tatoeba", "--yes", "--data-dir", data)
    assert "1 sentences for DeepSeek" in output
    assert "0 failed" in output and "stored 2 events" in output
    assert len(fake.prompts) == 3

    config = tmp_path / "curation.toml"
    config.write_text("heldout_pct = 0\n")
    run("curate", "--data-dir", data, "--config", config)
    run("build", "sft", "--data-dir", data)
    (snapshot,) = (data / "datasets").iterdir()
    card = (snapshot / "DATA_CARD.md").read_text()
    assert "**teacher-tatoeba:** CC BY 2.0 FR." in card
    assert "Translated by DeepSeek V4.1 Flash" in card
    assert "1 human and 2 synthetic events" in card


def test_teacher_refuses_without_source_text_or_over_its_limit(tmp_path, monkeypatch):
    assert "Run `flywheel ingest source wikinews`" in fail(
        "ingest", "teacher", "wikinews", "--data-dir", tmp_path
    )
    text = SourceText(id="t:0", corpus="tatoeba", document=None, text="I woke up at five.")
    write_jsonl(tmp_path / "sources" / "tatoeba.jsonl", [text])
    fake = FlakyTeacher(fail="\x00never")
    monkeypatch.setattr(teacher, "DeepSeek", fake)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    output = fail(
        "ingest", "teacher", "tatoeba", "--yes", "--max-cost-usd", "0", "--data-dir", tmp_path
    )
    assert "over the limit" in output
    assert fake.prompts == []


def test_a_speaker_checks_a_sample_of_teacher_pairs(tmp_path):
    data = tmp_path / "data"
    texts = [
        SourceText(id=f"tatoeba:v:en:{n}", corpus="tatoeba", document=None, text=f"Line {n}.")
        for n in range(8)
    ]
    answers = {
        (teacher.TEACHER, text.id): TranslationRecord(
            segment_id=text.id, model_id=teacher.TEACHER, output=f"Linya {n}."
        )
        for n, text in enumerate(texts)
    }
    write_jsonl(data / "events" / "teacher-tatoeba.jsonl", teacher.to_events(texts, answers))
    human = [make_reference(f"h{n}", text=f"Human line {n}.") for n in range(3)]
    write_jsonl(data / "events" / "opus-tatoeba.jsonl", human)
    config = tmp_path / "curation.toml"
    config.write_text("heldout_pct = 0\n")
    run("curate", "--data-dir", data, "--config", config)

    output = run(
        "rate", "sample", "--round", "t1", "--size", "4", "--checks", "1", "--data-dir", data
    )
    assert "wrote 6 rows" in output
    sheet = data / "ratings" / "t1" / "sheet.csv"
    with sheet.open(encoding="utf-8", newline="") as file:
        rows = [{**row, "adequacy": "5", "fluency": "5"} for row in csv.DictReader(file)]
    scored = tmp_path / "scored.csv"
    with scored.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = run(
        "rate", "import", scored, "--round", "t1", "--rater", "speaker-1", "--data-dir", data
    )
    assert "stored 6 ratings" in output
    ratings = read_jsonl(data / "ratings" / "t1" / "ratings-speaker-1.jsonl", HumanRating)
    assert sum(r.model_id == teacher.TEACHER for r in ratings) == 4
