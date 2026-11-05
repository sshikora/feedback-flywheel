import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from flywheel import cli
from flywheel.datasets.builders import build_dpo, build_sft
from flywheel.datasets.config import DatasetConfig
from flywheel.datasets.snapshot import manifest, write_snapshot
from flywheel.prompts import detection_answer, translation_prompt
from flywheel.schema import CuratedEvent, SftExample, TranslationInput
from flywheel.train import run as training
from flywheel.train.config import RunConfig
from flywheel.train.formats import dpo_line, sft_line, write_training_file
from flywheel.train.together import Together, TogetherError, job_request
from tests.factories import make_event

EXAMPLE = Path(__file__).parent.parent / "configs" / "run.example.toml"
SOURCE = {"path": "curated/train.jsonl", "sha256": "0" * 64}
runner = CliRunner()


def example(**overrides) -> SftExample:
    fields = {
        "id": "e1",
        "event_id": "e1",
        "origin": "human",
        "src_lang": "en",
        "tgt_lang": "ceb",
        "auto_detect": False,
        "source": "Where is the market?",
        "target": "Asa ang merkado?",
        "target_models": ["A"],
        "support": 1,
    }
    return SftExample(**(fields | overrides))


def curated(event) -> CuratedEvent:
    return CuratedEvent(event=event, support=1, group_size=1, contested=False, quality=0.5)


# --- prompts ---


def test_the_prompt_matches_the_app_including_tone():
    assert translation_prompt("Hi", "ceb", "en", "formal") == (
        "Translate the following text from English to Cebuano. Use a formal, professional tone. "
        "Return only the translated text with no explanation or commentary.\n\nHi"
    )


def test_an_auto_source_asks_for_json_with_the_detected_language():
    prompt = translation_prompt("Hi", "ceb", None)
    assert prompt.startswith("Translate the following text to Cebuano and identify the input's")
    assert prompt.endswith("Text to translate:\nHi")
    assert json.loads(detection_answer("Kumusta", "en")) == {
        "translatedText": "Kumusta",
        "detectedSourceLanguage": "en",
    }


# --- datasets feed the format ---


def test_examples_record_whether_the_source_was_auto_detected():
    auto = make_event(
        input=TranslationInput(text="Hi", src_lang=None, detected_src_lang="en", tgt_lang="ceb")
    )
    (built,) = build_sft([curated(auto)], DatasetConfig()).examples
    assert (built.auto_detect, built.src_lang) == (True, "en")


def test_a_dataset_can_keep_only_one_target_language():
    french = make_event(
        "fr", input=TranslationInput(text="Hello", src_lang="en", tgt_lang="fr"), minutes=1
    )
    build = build_sft(
        [curated(make_event()), curated(french)], DatasetConfig(target_languages=["ceb"])
    )
    assert [e.tgt_lang for e in build.examples] == ["ceb"]
    assert build.dropped == {"other_language": 1}


# --- training files ---


def test_an_sft_line_is_the_app_prompt_and_the_chosen_answer():
    line = sft_line(example())
    assert [m["role"] for m in line["messages"]] == ["user", "assistant"]
    assert line["messages"][0]["content"] == translation_prompt("Where is the market?", "ceb", "en")
    assert line["messages"][1]["content"] == "Asa ang merkado?"


def test_an_auto_detected_example_teaches_the_json_answer():
    line = sft_line(example(auto_detect=True))
    assert "identify the input's language" in line["messages"][0]["content"]
    assert json.loads(line["messages"][1]["content"])["detectedSourceLanguage"] == "en"


def test_a_dpo_line_has_one_assistant_message_per_side():
    event = make_event(outputs={"A": "Asa ang merkado?", "B": "Hain ang tiyanggihan?"})
    (pair,) = build_dpo([curated(event)], DatasetConfig()).examples
    line = dpo_line(pair)
    assert line["preferred_output"] == [{"role": "assistant", "content": "Asa ang merkado?"}]
    assert line["non_preferred_output"] == [
        {"role": "assistant", "content": "Hain ang tiyanggihan?"}
    ]
    assert [m["role"] for m in line["input"]["messages"]] == ["user"]


def test_examples_the_app_prompt_cannot_express_are_skipped_and_counted(tmp_path):
    examples = [
        example(),
        example(id="fr", tgt_lang="fr"),
        example(id="und", auto_detect=True, src_lang="und"),
    ]
    written = write_training_file(examples, tmp_path / "sft.jsonl")
    assert written.lines == 1
    assert written.skipped == {"unsupported_language": 1, "unknown_source_language": 1}
    assert len((tmp_path / "sft.jsonl").read_text().splitlines()) == 1


# --- job requests ---


def test_a_run_config_maps_to_a_lora_job():
    config = RunConfig.from_toml(EXAMPLE)
    body = job_request(config, "file-1")
    assert body["training_type"] == {"type": "Lora", "lora_r": 64, "lora_alpha": 128}
    assert body["training_method"] == {"method": "sft", "train_on_inputs": "auto"}
    assert (body["model"], body["n_epochs"], body["random_seed"]) == ("Qwen/Qwen3.5-9B", 3, 1)
    assert "from_checkpoint" not in body


def test_a_dpo_run_continues_from_its_parent_job(tmp_path):
    path = tmp_path / "dpo.toml"
    path.write_text(
        EXAMPLE.read_text()
        .replace('method = "sft"', 'method = "dpo"')
        .replace("lora_alpha = 128\n", "lora_alpha = 128\ndpo_beta = 0.1\n")
    )
    body = job_request(RunConfig.from_toml(path), "file-1", from_checkpoint="ft-parent")
    assert body["training_method"] == {"method": "dpo", "dpo_beta": 0.1}
    assert body["from_checkpoint"] == "ft-parent"


# --- a fake Together ---


class FakeTogether:
    def __init__(self, estimate: float = 4.0, allowed: bool = True, file_status: str = "COMPLETED"):
        self.estimate = estimate
        self.allowed = allowed
        self.file_status = file_status
        self.created = 0
        self.job_status = "running"
        self.uploads: list[bytes] = []

    def client(self) -> Together:
        return Together("key", httpx.MockTransport(self.handle), wait=lambda _: None)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/files/upload":
            self.uploads.append(request.content)
            return httpx.Response(200, json={"id": "file-1"})
        if path == "/v1/files/file-1":
            report = (
                {"valid": False, "error": "Line 1: bad"} if self.file_status != "COMPLETED" else {}
            )
            return httpx.Response(
                200, json={"processing_status": self.file_status, "validation_report": report}
            )
        if path == "/v1/fine-tunes/estimate-price":
            return httpx.Response(
                200,
                json={
                    "estimation_available": True,
                    "estimated_total_price": self.estimate,
                    "allowed_to_proceed": self.allowed,
                    "user_limit": 100.0,
                    "estimated_train_token_count": 900000,
                },
            )
        if path == "/v1/fine-tunes" and request.method == "POST":
            self.created += 1
            return httpx.Response(200, json={"id": "ft-1", "status": "pending"})
        if path == "/v1/fine-tunes/ft-1":
            return httpx.Response(
                200, json={"id": "ft-1", "status": self.job_status, "token_count": 900000}
            )
        if path == "/v1/finetune/download":
            assert request.url.params["checkpoint"] == "merged"
            return httpx.Response(302, headers={"location": "https://storage.example/weights"})
        if request.url.host == "storage.example":
            assert "authorization" not in request.headers
            return httpx.Response(200, content=b"weights")
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


def test_retryable_errors_are_retried_but_job_creation_is_not():
    calls = {"files": 0, "create": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/files/file-1":
            calls["files"] += 1
            if calls["files"] < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"processing_status": "COMPLETED"})
        calls["create"] += 1
        return httpx.Response(500)

    client = Together("key", httpx.MockTransport(handle), wait=lambda _: None)
    assert client.file_status("file-1")[0] == "COMPLETED"
    with pytest.raises(TogetherError, match="job not created"):
        client.create({"model": "m"})
    assert calls == {"files": 3, "create": 1}


def test_an_unavailable_estimate_is_an_error():
    def handle(_: httpx.Request) -> httpx.Response:
        body = {"estimation_available": False, "unavailable_reason": "train_file_not_validated"}
        return httpx.Response(200, json=body)

    with pytest.raises(TogetherError, match="train_file_not_validated"):
        Together("key", httpx.MockTransport(handle)).estimate({})


def test_a_download_lands_whole_or_not_at_all(tmp_path):
    destination = tmp_path / "model" / "merged.tar.zst"
    assert FakeTogether().client().download("ft-1", destination) == len(b"weights")
    assert destination.read_bytes() == b"weights"
    assert not (destination.parent / "merged.tar.zst.part").exists()


def test_an_empty_download_is_refused(tmp_path):
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    destination = tmp_path / "merged.tar.zst"
    with pytest.raises(TogetherError, match="download was empty"):
        Together("key", httpx.MockTransport(handle)).download("ft-1", destination)
    assert not destination.exists()
    assert not (tmp_path / "merged.tar.zst.part").exists()


# --- a run ---


def built_run(tmp_path, max_cost: str = "10.0") -> tuple[RunConfig, Path, Path]:
    data = tmp_path / "data"
    items = [curated(make_event())]
    build = build_sft(items, DatasetConfig())
    meta = manifest("sft", build, DatasetConfig(), SOURCE)
    write_snapshot(data / "datasets", "sft", build.examples, meta)
    path = tmp_path / "v1.toml"
    path.write_text(
        EXAMPLE.read_text()
        .replace("0" * 64, meta["hash"])
        .replace("example-qwen3.5-9b-sft", "v1")
        .replace("max_cost_usd = 10.0", f"max_cost_usd = {max_cost}")
    )
    return RunConfig.from_toml(path), path, data


def test_preparing_uploads_and_prices_without_starting_a_job(tmp_path):
    config, path, data = built_run(tmp_path)
    fake = FakeTogether()
    state = training.prepare(config, path, data, fake.client(), None, lambda _: None)
    assert (state.file_id, state.estimate_usd, state.lines, state.job_id) == (
        "file-1",
        4.0,
        1,
        None,
    )
    assert fake.created == 0
    assert b"Asa ang merkado?" in fake.uploads[0]


def test_a_job_is_created_once_however_often_it_is_submitted(tmp_path):
    config, path, data = built_run(tmp_path)
    fake = FakeTogether()
    training.prepare(config, path, data, fake.client(), None, lambda _: None)
    first = training.submit(config, path, data, fake.client(), None)
    again = training.submit(config, path, data, fake.client(), None)
    prepared_again = training.prepare(config, path, data, fake.client(), None, lambda _: None)
    assert first.job_id == again.job_id == prepared_again.job_id == "ft-1"
    assert fake.created == 1


def test_an_estimate_above_the_limit_is_refused(tmp_path):
    config, path, data = built_run(tmp_path, max_cost="3.0")
    with pytest.raises(training.RunRefused, match="above max_cost_usd"):
        training.prepare(
            config, path, data, FakeTogether(estimate=4.0).client(), None, lambda _: None
        )


def test_a_job_together_will_not_run_is_refused(tmp_path):
    config, path, data = built_run(tmp_path)
    with pytest.raises(training.RunRefused, match="will not run"):
        training.prepare(
            config, path, data, FakeTogether(allowed=False).client(), None, lambda _: None
        )


def test_a_rejected_training_file_stops_the_run(tmp_path):
    config, path, data = built_run(tmp_path)
    fake = FakeTogether(file_status="INVALID_FORMAT")
    with pytest.raises(training.RunRefused, match="Line 1: bad"):
        training.prepare(config, path, data, fake.client(), None, lambda _: None)


def test_a_config_edited_after_preparing_cannot_be_submitted(tmp_path):
    config, path, data = built_run(tmp_path)
    fake = FakeTogether()
    training.prepare(config, path, data, fake.client(), None, lambda _: None)
    path.write_text(path.read_text().replace("epochs = 3", "epochs = 4"))
    with pytest.raises(training.RunRefused, match="changed since"):
        training.submit(config, path, data, fake.client(), None)
    assert fake.created == 0


def test_an_edited_snapshot_is_never_uploaded(tmp_path):
    config, path, data = built_run(tmp_path)
    (stored,) = (data / "datasets").glob("*/examples.jsonl")
    stored.write_text(stored.read_text().replace("merkado", "tiyanggihan"))
    fake = FakeTogether()
    with pytest.raises(training.RunRefused, match="edited"):
        training.prepare(config, path, data, fake.client(), None, lambda _: None)
    assert fake.uploads == []


# --- the commands ---


def test_start_status_and_download_register_a_candidate(tmp_path, monkeypatch):
    _, path, data = built_run(tmp_path)
    fake = FakeTogether()
    monkeypatch.setattr(cli, "_together", fake.client)
    monkeypatch.setattr(training, "FILE_POLL_SECONDS", 0)

    def run(*args: object) -> str:
        result = runner.invoke(cli.app, [str(a) for a in args], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        return result.output

    planned = run("train", "start", path, "--data-dir", data)
    assert "estimated $4.00 (limit $10.00)" in planned
    assert "No job started" in planned
    assert "uploaded to Together" in planned
    assert "answers to learn were written by: A (1)" in planned
    assert fake.created == 0

    assert "started job ft-1" in run("train", "start", path, "--yes", "--data-dir", data)
    assert "already started as job ft-1" in run("train", "start", path, "--yes", "--data-dir", data)
    assert fake.created == 1

    assert "v1: running" in run("train", "status", "v1", "--data-dir", data)
    fake.job_status = "completed"
    assert "registered v1 as a candidate" in run("train", "status", "v1", "--data-dir", data)
    assert "v1  candidate  sft" in run("registry", "list", "--data-dir", data)

    assert "merged.tar.zst" in run("train", "download", "v1", "--data-dir", data)
    state = json.loads((data / "training" / "v1" / "state.json").read_text())
    assert state["weights"].endswith("merged.tar.zst")
    assert state["estimate_usd"] == 4.0


def test_the_download_link_is_the_redirect_target():
    assert FakeTogether().client().download_url("ft-1") == "https://storage.example/weights"


def test_a_download_link_needs_a_redirect():
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "no such job"}})

    with pytest.raises(TogetherError, match="no such job"):
        Together("key", httpx.MockTransport(handle)).download_url("ft-9")
