from datetime import UTC, datetime, timedelta

import pytest

from flywheel.curate.config import CurationConfig
from flywheel.curate.dedupe import dedupe, independent_votes
from flywheel.curate.leakage import LeakageIndex, leaks
from flywheel.curate.pipeline import curate, is_heldout
from flywheel.curate.quality import quality_score, spam_reason
from flywheel.curate.report import origin_shares, vote_statistics
from flywheel.schema import Consent, EventContext, TranslationInput
from tests.factories import make_event, make_segment

NO_HOLDOUT = CurationConfig(heldout_pct=0)
WINDOW = 600
EVAL_SENTENCE = "On Monday, scientists announced the invention of a new diagnostic tool."
EVAL_REFERENCE = "Kaniadtong Lunes, gipahibalo sa mga siyentista ang bag-ong himan sa pagdayagnos."


# --- dedupe ---


def test_repeat_inputs_collapse_and_their_votes_are_counted():
    events = [
        make_event("a", text="Hello there", outputs={"X": "Kumusta"}, minutes=0),
        make_event("b", text="hello   there.", outputs={"X": "Kumusta!"}, minutes=30),
    ]
    [group] = dedupe(events, WINDOW)
    assert (group.support, group.group_size, group.contested) == (2, 2, False)


def test_most_supported_answer_wins_and_the_conflict_is_recorded():
    events = [
        make_event("a", outputs={"X": "One"}, minutes=0),
        make_event("b", outputs={"X": "Two"}, minutes=30),
        make_event("c", outputs={"X": "Two"}, minutes=60),
    ]
    [group] = dedupe(events, WINDOW)
    assert group.event.chosen_output == "Two"
    assert (group.support, group.group_size, group.contested) == (2, 3, True)


def test_a_vote_burst_counts_once():
    events = [make_event(str(i), outputs={"X": "Stuffed"}, minutes=0) for i in range(50)]
    [group] = dedupe(events, WINDOW)
    assert group.support == 1


def test_a_burst_cannot_outvote_independent_voters():
    burst = [make_event(f"b{i}", outputs={"X": "Stuffed"}, minutes=i) for i in range(5)]
    honest = [make_event(f"h{i}", outputs={"X": "Honest"}, minutes=30 * (i + 1)) for i in range(2)]
    [group] = dedupe(burst + honest, WINDOW)
    assert group.event.chosen_output == "Honest"


def test_a_steady_trickle_is_not_mistaken_for_a_burst():
    start = datetime(2026, 10, 27, tzinfo=UTC)
    assert independent_votes((start + timedelta(minutes=11 * i) for i in range(4)), WINDOW) == 4


def test_ties_do_not_depend_on_input_order():
    events = [
        make_event("a", outputs={"X": "One"}, minutes=0),
        make_event("b", outputs={"X": "Two"}, minutes=30),
    ]
    assert dedupe(events, WINDOW) == dedupe(list(reversed(events)), WINDOW)
    assert dedupe(events, WINDOW)[0].event.id == "a"


def test_reingesting_the_same_event_does_not_add_support():
    [group] = dedupe([make_event("a"), make_event("a")], WINDOW)
    assert group.support == 1


@pytest.mark.parametrize(
    "other",
    [
        TranslationInput(text="Hello", src_lang="en", tgt_lang="fil"),
        TranslationInput(text="Hello", src_lang="es", tgt_lang="ceb"),
        TranslationInput(text="Hello", src_lang="en", tgt_lang="ceb", tone="formal"),
    ],
    ids=["target", "source", "tone"],
)
def test_different_language_pairs_and_tones_stay_separate(other):
    base = TranslationInput(text="Hello", src_lang="en", tgt_lang="ceb")
    events = [make_event("a", input=base), make_event("b", input=other)]
    assert len(dedupe(events, WINDOW)) == 2


# --- spam and quality ---


@pytest.mark.parametrize(
    ("text", "output", "reason"),
    [
        ("12345 !!!", "12345", "no_letters"),
        ("aaaaaaaa", "Tubag", "repeated_characters"),
        ("hahahaha", "Tubag", "repeated_characters"),
        ("Manila Bay", "Manila Bay.", "untranslated"),
        ("A sentence of ordinary length.", "x" * 200, "length_outlier"),
        ("A sentence of ordinary length.", "Oo", "length_outlier"),
    ],
)
def test_spam_is_rejected_with_a_reason(text, output, reason):
    assert spam_reason(make_event(text=text, outputs={"X": output}), 5.0) == reason


def test_short_answers_to_short_inputs_are_fine():
    assert spam_reason(make_event(text="Yes", outputs={"X": "Oo"}), 5.0) is None


def test_ordinary_events_pass():
    assert spam_reason(make_event(), 5.0) is None


def test_a_lone_vote_scores_lower_than_a_corroborated_one():
    event = make_event()
    assert quality_score(event, 1, 1) < quality_score(event, 2, 2) < quality_score(event, 4, 4)


def test_model_agreement_raises_the_score():
    alone = make_event(outputs={"X": "Same", "Y": "Other"}, chosen=["X"])
    agreed = make_event(outputs={"X": "Same", "Y": "Same"}, chosen=["X", "Y"])
    assert quality_score(agreed, 1, 1) > quality_score(alone, 1, 1)


def test_a_contested_answer_scores_lower_than_an_unopposed_one():
    event = make_event()
    assert quality_score(event, 2, 4) < quality_score(event, 2, 2)


def test_one_vote_does_not_count_as_unanimous():
    event = make_event(outputs={"X": "Only"})
    assert quality_score(event, 1, 1) == pytest.approx(0.5)


def test_the_best_possible_score_is_one():
    event = make_event(outputs={"X": "Only"})
    assert quality_score(event, 10, 10) == pytest.approx(1.0)


# --- leakage ---


def test_exact_and_near_matches_are_found():
    index = LeakageIndex([EVAL_SENTENCE], 0.8)
    assert index.matches(EVAL_SENTENCE.upper())
    assert index.matches(EVAL_SENTENCE.replace("new", "novel"))


def test_unrelated_text_is_not_a_match():
    index = LeakageIndex([EVAL_SENTENCE], 0.8)
    assert not index.matches("Where can I buy fresh mangoes near the pier?")


def test_empty_index_matches_nothing():
    assert not LeakageIndex([], 0.8).matches(EVAL_SENTENCE)


@pytest.mark.parametrize(
    ("text", "output"),
    [(EVAL_SENTENCE, "Tubag"), ("Something else entirely here", EVAL_REFERENCE)],
    ids=["input", "output"],
)
def test_leaks_check_the_input_and_the_candidates(text, output):
    index = LeakageIndex([EVAL_SENTENCE, EVAL_REFERENCE], 0.8)
    assert leaks(make_event(text=text, outputs={"X": output}), index)


def test_rejected_candidates_are_checked_too():
    index = LeakageIndex([EVAL_REFERENCE], 0.8)
    event = make_event(outputs={"X": "Tubag", "Y": EVAL_REFERENCE}, chosen=["X"])
    assert leaks(event, index)


# --- full run ---


def test_nothing_in_an_eval_set_reaches_training():
    segment = make_segment(source=EVAL_SENTENCE, reference=EVAL_REFERENCE)
    events = [
        make_event("source", text=EVAL_SENTENCE),
        make_event("near", text=EVAL_SENTENCE.replace("new", "novel"), minutes=1),
        make_event(
            "reference",
            text="An unrelated sentence of about the same length as the reference.",
            outputs={"X": EVAL_REFERENCE},
            minutes=2,
        ),
        make_event("clean", text="Where can I buy mangoes?", minutes=3),
    ]
    result = curate(events, [segment], NO_HOLDOUT)
    assert [item.event.id for item in result.train] == ["clean"]
    assert result.report["dropped"]["leak_fixed_eval"] == 3


def test_held_out_text_cannot_reach_training_through_a_near_duplicate():
    config = CurationConfig(heldout_pct=50, seed="pinned")
    sentence = "The ferry to Bohol leaves from the third pier every morning."
    base = next(
        f"{sentence} Trip {i}."
        for i in range(100)
        if is_heldout(f"{sentence} trip {i}\x1fen\x1fceb\x1f".lower(), config)
    )
    near = next(
        f"{base} Yes {i}"
        for i in range(100)
        if not is_heldout(f"{base} yes {i}\x1fen\x1fceb\x1f".lower(), config)
    )
    result = curate([make_event("h", text=base), make_event("t", text=near)], [], config)
    assert [item.event.id for item in result.heldout] == ["h"]
    assert result.train == []
    assert result.report["dropped"]["leak_heldout"] == 1


def test_the_split_is_stable_and_near_the_requested_share():
    config = CurationConfig(heldout_pct=10)
    keys = [f"sentence number {i}" for i in range(2000)]
    first = [is_heldout(key, config) for key in keys]
    assert first == [is_heldout(key, config) for key in keys]
    assert 150 < sum(first) < 250


def test_the_same_input_always_lands_on_one_side():
    config = CurationConfig(heldout_pct=50)
    events = [make_event(str(i), minutes=30 * i) for i in range(3)]
    result = curate(events, [], config)
    assert len(result.train) + len(result.heldout) == 1


def test_events_without_training_consent_are_dropped_first():
    refused = make_event(consent=Consent(training=False, publish=False))
    result = curate([refused], [], NO_HOLDOUT)
    assert result.train == []
    assert result.report["dropped"] == {"no_training_consent": 1}


def test_spam_cannot_add_support():
    events = [
        make_event("good", text="aaaaaaaa", outputs={"X": "Tubag"}),
        make_event("real", text="Where can I buy mangoes?", minutes=1),
    ]
    result = curate(events, [], NO_HOLDOUT)
    assert [item.event.id for item in result.train] == ["real"]
    assert result.report["dropped"] == {"spam_repeated_characters": 1}


def test_curation_is_deterministic():
    events = [
        make_event(str(i), text=f"Sentence number {i} goes here", minutes=i) for i in range(50)
    ]
    config = CurationConfig()
    assert curate(events, [], config) == curate(list(reversed(events)), [], config)


# --- report ---


def test_win_rate_divides_by_appearances():
    events = [
        make_event("a", outputs={"X": "1", "Y": "2"}, chosen=["X"]),
        make_event("b", outputs={"X": "1", "Y": "2"}, chosen=["Y"]),
        make_event("c", outputs={"X": "1", "Z": "2"}, chosen=["X"]),
    ]
    by_model = vote_statistics(events)["by_model"]
    assert by_model["X"] == {"wins": 2, "appearances": 3, "win_rate": 0.6667}
    assert by_model["Z"] == {"wins": 0, "appearances": 1, "win_rate": 0.0}


def test_an_agreement_vote_counts_for_each_model():
    event = make_event(outputs={"X": "1", "Y": "1", "Z": "2"}, chosen=["X", "Y"])
    by_model = vote_statistics([event])["by_model"]
    assert by_model["X"]["wins"] == by_model["Y"]["wins"] == 1


def test_own_model_pick_rate_only_counts_votes_where_it_answered():
    own = EventContext(blind=False, your_model="X")
    absent = EventContext(blind=False, your_model="GONE")
    events = [
        make_event("a", outputs={"X": "1", "Y": "2"}, chosen=["X"], context=own),
        make_event("b", outputs={"X": "1", "Y": "2"}, chosen=["Y"], context=own),
        make_event("c", outputs={"X": "1", "Y": "2"}, chosen=["Y"], context=absent),
        make_event("d", outputs={"X": "1", "Y": "2"}, chosen=["Y"]),
    ]
    assert vote_statistics(events)["own_model_pick_rate"] == 0.5


def test_statistics_handle_no_votes():
    stats = vote_statistics([])
    assert (stats["votes"], stats["own_model_pick_rate"], stats["blind_share"]) == (0, None, None)


def test_human_and_synthetic_shares_are_reported():
    events = [
        make_event("a"),
        make_event("b", origin="synthetic"),
        make_event("c", origin="synthetic"),
    ]
    assert origin_shares(events) == {"human": 1, "synthetic": 2, "human_share": 0.3333}


def test_the_report_never_contains_text():
    event = make_event(text="A very identifiable sentence about my neighbour.")
    report = curate([event], [], NO_HOLDOUT).report
    assert "identifiable" not in str(report)
