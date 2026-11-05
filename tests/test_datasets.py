import json

import pytest

from flywheel.datasets.builders import build_dpo, build_sft
from flywheel.datasets.card import render_card
from flywheel.datasets.config import DatasetConfig
from flywheel.datasets.snapshot import content_hash, manifest, write_snapshot
from flywheel.schema import Candidate, CuratedEvent, EventContext, FeedbackEvent
from tests.factories import make_event, make_reference

DEFAULTS = DatasetConfig()
SOURCE = {"path": "curated/train.jsonl", "sha256": "0" * 64}


def curated(event: FeedbackEvent, **overrides) -> CuratedEvent:
    fields = {"support": 1, "group_size": 1, "contested": False, "quality": 0.5}
    return CuratedEvent(event=event, **(fields | overrides))


def agreement() -> FeedbackEvent:
    return make_event(
        "agree",
        outputs={"G": "Asa ang merkado?", "D": "Asa ang merkado", "Q": "Hain ang tiyanggihan?"},
        chosen=["G", "D"],
    )


# --- SFT ---


def test_an_sft_example_is_the_input_with_the_chosen_answer():
    (example,) = build_sft([curated(make_event(), support=3)], DEFAULTS).examples
    assert (example.source, example.target) == ("Where is the market?", "Asa ang merkado?")
    assert (example.event_id, example.target_models, example.support) == ("e1", ["A"], 3)
    assert (example.src_lang, example.tgt_lang, example.origin) == ("en", "ceb", "human")


def test_events_below_the_quality_bar_are_dropped():
    build = build_sft([curated(make_event(), quality=0.3)], DatasetConfig(min_quality=0.4))
    assert build.examples == []
    assert build.dropped == {"low_quality": 1}


def test_contested_events_train_sft_but_not_dpo_by_default():
    item = curated(make_event(), contested=True, group_size=2)
    assert len(build_sft([item], DEFAULTS).examples) == 1
    assert build_dpo([item], DEFAULTS).dropped == {"contested": 1}


def test_an_answer_written_only_by_an_excluded_model_is_dropped():
    build = build_sft([curated(make_event())], DatasetConfig(exclude_models=["A"]))
    assert build.examples == []
    assert build.dropped == {"excluded_model": 1}


def test_an_agreement_keeps_the_text_of_a_model_that_is_not_excluded():
    config = DatasetConfig(exclude_models=["G"])
    (example,) = build_sft([curated(agreement())], config).examples
    assert (example.target, example.target_models) == ("Asa ang merkado", ["D"])


# --- DPO ---


def test_each_answer_passed_over_becomes_a_pair():
    event = make_event(outputs={"A": "Asa ang merkado?", "B": "Hain ang tiyanggihan?", "C": "Asa?"})
    pairs = build_dpo([curated(event)], DEFAULTS).examples
    assert [(p.chosen, p.rejected, p.rejected_model) for p in pairs] == [
        ("Asa ang merkado?", "Hain ang tiyanggihan?", "B"),
        ("Asa ang merkado?", "Asa?", "C"),
    ]


def test_failed_excluded_and_identical_answers_make_no_pair():
    event = make_event(outputs={"A": "Asa ang merkado?", "B": "asa ang merkado", "C": "Hain?"})
    event = event.model_copy(
        update={
            "candidates": [*event.candidates, Candidate(label="3", model_id="X", error="timed out")]
        }
    )
    build = build_dpo([curated(event)], DatasetConfig(exclude_models=["C"]))
    assert build.examples == []
    assert build.dropped == {
        "no_rejected": 1,
        "rejected_excluded_model": 1,
        "rejected_same_as_chosen": 1,
    }


def test_near_identical_pairs_are_flagged_as_close():
    event = make_event(
        outputs={
            "A": "Tan-awa ang marka alang sa piho nga mga panudlo sa first aid alang sa hilo.",
            "B": "Susiha ang timaan sa piho nga mga panudlo sa first aid alang sa piho nga hilo.",
            "C": "Hain ang tiyanggihan?",
        }
    )
    pairs = {p.rejected_model: p.close for p in build_dpo([curated(event)], DEFAULTS).examples}
    assert pairs == {"B": True, "C": False}


def test_agreement_models_are_never_rejected_against_each_other():
    pairs = build_dpo([curated(agreement())], DEFAULTS).examples
    assert [p.rejected_model for p in pairs] == ["Q"]
    assert pairs[0].chosen_models == ["G", "D"]


# --- snapshots ---


def build_items() -> list[CuratedEvent]:
    return [
        curated(
            make_event("b", text="Thank you", outputs={"A": "Salamat", "B": "Daghang salamat"})
        ),
        curated(make_event("a", outputs={"A": "Asa ang merkado?", "B": "Hain?"}, minutes=1)),
    ]


def test_the_hash_does_not_depend_on_input_order():
    items = build_items()
    first = build_sft(items, DEFAULTS).examples
    second = build_sft(list(reversed(items)), DEFAULTS).examples
    assert content_hash(first) == content_hash(second)


def test_a_snapshot_is_named_by_its_hash_and_written_once(tmp_path):
    build = build_sft(build_items(), DEFAULTS)
    meta = manifest("sft", build, DEFAULTS, SOURCE)
    first = write_snapshot(tmp_path, "sft", build.examples, meta)
    assert first.created
    assert first.directory.name == f"sft-{meta['hash'][:12]}"
    assert sorted(p.name for p in first.directory.iterdir()) == [
        "DATA_CARD.md",
        "examples.jsonl",
        "manifest.json",
    ]
    again = write_snapshot(
        tmp_path, "sft", build.examples, manifest("sft", build, DEFAULTS, SOURCE)
    )
    assert (again.directory, again.created) == (first.directory, False)


def test_an_edited_snapshot_is_refused(tmp_path):
    build = build_sft(build_items(), DEFAULTS)
    meta = manifest("sft", build, DEFAULTS, SOURCE)
    stored = write_snapshot(tmp_path, "sft", build.examples, meta).directory / "examples.jsonl"
    stored.write_text(stored.read_text().replace("Salamat", "Salamat kaayo"))
    with pytest.raises(ValueError, match="edited"):
        write_snapshot(tmp_path, "sft", build.examples, meta)


def test_a_manifest_must_describe_the_examples(tmp_path):
    build = build_sft(build_items(), DEFAULTS)
    meta = manifest("sft", build, DEFAULTS, SOURCE) | {"hash": "f" * 64}
    with pytest.raises(ValueError, match="does not describe"):
        write_snapshot(tmp_path, "sft", build.examples, meta)


def test_the_manifest_counts_what_went_in_and_what_was_dropped():
    items = [*build_items(), curated(make_event("low", minutes=2), quality=0.1)]
    config = DatasetConfig(min_quality=0.2)
    counts = manifest("dpo", build_dpo(items, config), config, SOURCE)["counts"]
    assert (counts["examples"], counts["events"]) == (2, 2)
    assert counts["dropped"] == {"low_quality": 1}
    assert counts["rejected_models"] == {"B": 2}
    assert counts["blind_share"] == 0.0
    assert counts["origin"]["human_share"] == 1.0


def test_the_data_card_names_the_hash_and_the_brand_bias():
    build = build_dpo(build_items(), DEFAULTS)
    meta = manifest("dpo", build, DEFAULTS, SOURCE)
    card = render_card(meta)
    assert meta["hash"] in card
    assert "100% of votes (2) were cast with model names visible" in card
    assert "| B | 2 |" in card


def test_a_card_for_blind_events_records_no_brand_bias():
    blind = make_event(context=EventContext(blind=True))
    build = build_sft([curated(blind)], DEFAULTS)
    card = render_card(manifest("sft", build, DEFAULTS, SOURCE))
    assert "brand bias" not in card
    assert json.loads(json.dumps(build.examples[0].model_dump()))["target"] == "Asa ang merkado?"


def test_a_source_can_be_left_out_of_a_build():
    items = [curated(make_event()), curated(make_reference(source="opus-bible-uedin"))]
    config = DatasetConfig(exclude_sources=["opus-bible-uedin"])
    build = build_sft(items, config)
    assert [e.id for e in build.examples] == ["e1"]
    assert build.dropped == {"excluded_source": 1}


def test_a_corpus_reference_trains_sft_but_makes_no_dpo_pair():
    items = [curated(make_reference())]
    (example,) = build_sft(items, DEFAULTS).examples
    assert (example.target, example.target_models) == ("Asa ang merkado?", ["human:opus-tatoeba"])
    assert build_dpo(items, DEFAULTS).dropped == {"no_rejected": 1}


def test_brand_bias_is_measured_over_votes_and_licenses_are_listed():
    items = [curated(make_event()), curated(make_reference(text="Thank you."))]
    meta = manifest("sft", build_sft(items, DEFAULTS), DEFAULTS, SOURCE)
    assert (meta["counts"]["votes"], meta["counts"]["blind_share"]) == (1, 0.0)
    assert list(meta["licenses"]) == ["opus-tatoeba"]
    card = render_card(meta)
    assert "100% of votes (1) were cast with model names visible" in card
    assert "## Licenses" in card and "**opus-tatoeba:** CC BY 2.0 FR." in card


def test_a_card_without_corpus_text_has_no_license_section():
    card = render_card(manifest("sft", build_sft(build_items(), DEFAULTS), DEFAULTS, SOURCE))
    assert "## Licenses" not in card
