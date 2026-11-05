from datetime import UTC, datetime
from pathlib import Path

import pytest

from flywheel.registry.card import render_card
from flywheel.registry.store import Registry
from flywheel.schema import ModelRecord
from flywheel.train.config import RunConfig, config_hash

EXAMPLE = Path(__file__).parent.parent / "configs" / "run.example.toml"
DIGEST = "ab" * 32


def record(model_id: str = "v1", **overrides) -> ModelRecord:
    fields = {
        "id": model_id,
        "base_model": "Qwen/Qwen3.5-9B",
        "method": "sft",
        "dataset_hash": DIGEST,
        "run_config": "configs/runs/v1.toml",
        "run_config_hash": "cd" * 32,
        "provider": "together",
        "cost_usd": 4.2,
        "status": "candidate",
        "created_at": datetime(2026, 11, 4, tzinfo=UTC),
    }
    return ModelRecord(**(fields | overrides))


@pytest.fixture
def registry(tmp_path):
    store = Registry(tmp_path / "registry.sqlite")
    yield store
    store.close()


# --- run configs ---


def test_the_example_run_config_loads():
    config = RunConfig.from_toml(EXAMPLE)
    assert (config.method, config.gate.min_judge_win_rate) == ("sft", 0.55)


def test_a_run_config_has_no_defaults(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(EXAMPLE.read_text().replace("seed = 1\n", ""))
    with pytest.raises(ValueError, match="seed"):
        RunConfig.from_toml(path)


def test_a_dpo_run_needs_its_beta(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(EXAMPLE.read_text().replace('method = "sft"', 'method = "dpo"'))
    with pytest.raises(ValueError, match="dpo_beta"):
        RunConfig.from_toml(path)


def test_the_gate_must_require_more_than_a_coin_flip(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        EXAMPLE.read_text().replace("min_judge_win_rate = 0.55", "min_judge_win_rate = 0.5")
    )
    with pytest.raises(ValueError, match="min_judge_win_rate"):
        RunConfig.from_toml(path)


def test_the_config_hash_changes_with_any_edit(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(EXAMPLE.read_text())
    before = config_hash(path)
    path.write_text(EXAMPLE.read_text().replace("epochs = 3", "epochs = 4"))
    assert config_hash(path) != before


# --- registry ---


def test_a_registered_model_starts_as_a_candidate(registry):
    registry.register(record())
    assert registry.get("v1").status == "candidate"
    assert [(c.from_status, c.to_status, c.reason) for c in registry.history("v1")] == [
        (None, "candidate", "registered")
    ]


def test_a_model_cannot_be_registered_twice_or_skip_the_gate(registry):
    registry.register(record())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(record())
    with pytest.raises(ValueError, match="starts as a candidate"):
        registry.register(record("v2", status="champion"))


def test_status_moves_only_along_allowed_steps(registry):
    registry.register(record())
    with pytest.raises(ValueError, match="cannot move from candidate to champion"):
        registry.transition("v1", "champion", "skipped the gate", {})
    registry.transition("v1", "retired", "abandoned", {})
    with pytest.raises(ValueError, match="cannot move from retired"):
        registry.transition("v1", "candidate", "revived", {})


def test_a_failed_gate_holds_the_model_and_keeps_the_evidence(registry):
    registry.register(record())
    registry.hold("v1", "failed the gate", {"chrf_gain": {"low": -0.4, "high": 0.9}})
    assert registry.get("v1").status == "candidate"
    held = registry.history("v1")[-1]
    assert (held.from_status, held.to_status) == ("candidate", "candidate")
    assert held.evidence == {"chrf_gain": {"low": -0.4, "high": 0.9}}


def test_one_model_is_live_and_a_new_challenger_retires_it(registry):
    assert registry.live() is None
    registry.register(record("v1"))
    registry.transition("v1", "challenger", "passed the gate", {"judge_win_rate": 0.61})
    registry.transition("v1", "champion", "met the vote target", {"votes": 200})
    registry.register(record("v2", start_from="v1"))
    registry.transition("v2", "challenger", "beat v1 at the gate", {})
    assert registry.live().id == "v2"
    assert registry.get("v1").status == "retired"
    assert registry.history("v1")[-1].reason == "replaced by v2"


def test_details_can_be_filled_in_but_status_cannot(registry):
    registry.register(record())
    updated = registry.update("v1", eval_report="data/runs/v1/report.json", cost_usd=5.1)
    assert (updated.eval_report, registry.get("v1").cost_usd) == ("data/runs/v1/report.json", 5.1)
    with pytest.raises(ValueError, match="transition or hold"):
        registry.update("v1", status="champion")


def test_the_registry_survives_reopening(tmp_path):
    path = tmp_path / "registry.sqlite"
    first = Registry(path)
    first.register(record())
    first.close()
    second = Registry(path)
    assert [m.id for m in second.models("candidate")] == ["v1"]
    second.close()


# --- model cards ---


def test_the_card_states_the_pre_registered_gate_and_makes_no_claim_untested(registry):
    registry.register(record())
    registry.hold("v1", "failed the gate", {})
    card = render_card(
        registry.get("v1"), RunConfig.from_toml(EXAMPLE), registry.history("v1"), None
    )
    assert "at least 55% of the time" in card
    assert "no regression on the guard set `flores200-dev-en-ceb`" in card.lower()
    assert "makes no quality claim" in card
    assert "| held as candidate | failed the gate |" in card
    assert DIGEST in card


def test_the_card_credits_each_corpus_the_model_learned_from(registry):
    registry.register(record())
    dataset = {
        "counts": {
            "examples": 2,
            "events": 2,
            "origin": {"human_share": 1.0},
            "sources": {"opus-tatoeba": 2},
        },
        "licenses": {"opus-tatoeba": {"license": "CC BY 2.0 FR", "attribution": "By tatoeba.org."}},
    }
    card = render_card(
        registry.get("v1"), RunConfig.from_toml(EXAMPLE), registry.history("v1"), dataset
    )
    assert "- **Credit for opus-tatoeba:** CC BY 2.0 FR. By tatoeba.org." in card
