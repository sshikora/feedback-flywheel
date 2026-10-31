"""End to end through the command line, with fixture data only."""

import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flywheel.cli import app
from flywheel.ingest import flores
from flywheel.schema import CuratedEvent, EvalSegment
from flywheel.storage import read_jsonl, write_jsonl
from tests.factories import make_event

FIXTURE = Path(__file__).parent / "fixtures" / "comparisons_scan.json"
runner = CliRunner()


def run(*args: object) -> str:
    result = runner.invoke(app, [str(arg) for arg in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


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
    output = run("ingest", "comparisons", FIXTURE, "--data-dir", data)
    assert "skipped 2 (no_vote)" in output
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
