"""The evaluation harness, with fake model clients. No test here spends money."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from sacrebleu.metrics.chrf import CHRF
from typer.testing import CliRunner

from flywheel.cli import app
from flywheel.evals import judge, metrics, rating, report, translate
from flywheel.evals.config import EvalConfig
from flywheel.schema import HumanRating, JudgeVerdict, TranslationRecord
from flywheel.storage import read_jsonl, write_jsonl
from tests.factories import make_segment

BASELINE = Path(__file__).parent.parent / "configs" / "eval.baseline.toml"
TEACHER = "google/teacher"
BASE = "qwen/base"
runner = CliRunner()


def make_config(**overrides) -> EvalConfig:
    fields = {
        "name": "test-run",
        "eval_set": "test-set",
        "models": [{"id": TEACHER, "role": "compare"}, {"id": BASE, "role": "base"}],
        "resamples": 200,
    }
    return EvalConfig.model_validate(fields | overrides)


def make_segments(count: int = 12):
    return [
        make_segment(i, f"Source sentence number {i}.", f"Reperensya nga tudling {i} sa teksto.")
        for i in range(count)
    ]


class TestMetrics:
    def test_chrf_matches_sacrebleu(self):
        outputs = ["Asa ang merkado?", "Kumusta ka"]
        references = ["Asa ang tiyanggihan?", "Kumusta ka diha"]
        expected = CHRF(word_order=2).corpus_score(outputs, [references]).score
        assert metrics.chrf(metrics.chrf_stats(outputs, references)) == pytest.approx(expected)

    def test_empty_output_counts_as_a_miss(self):
        references = ["Asa ang tiyanggihan?", "Kumusta ka diha"]
        whole = metrics.chrf(metrics.chrf_stats(references, references))
        partial = metrics.chrf(metrics.chrf_stats([references[0], ""], references))
        assert whole == pytest.approx(100)
        assert partial < whole

    def test_interval_surrounds_the_score(self):
        segments = make_segments()
        stats = metrics.chrf_stats(
            [s.reference[:12] for s in segments], [s.reference for s in segments]
        )
        weights = metrics.resample_weights(len(segments), 200, "seed")
        interval = metrics.chrf_interval(stats, weights)
        assert interval.low <= interval.value <= interval.high
        assert interval.low < interval.high

    def test_resamples_are_repeatable_and_sum_to_the_set(self):
        first = metrics.resample_weights(10, 50, "seed")
        assert (first == metrics.resample_weights(10, 50, "seed")).all()
        assert (first != metrics.resample_weights(10, 50, "other")).any()
        assert (first.sum(axis=1) == 10).all()

    def test_a_model_against_itself_is_a_tie(self):
        segments = make_segments()
        references = [s.reference for s in segments]
        stats = metrics.chrf_stats([r[:10] for r in references], references)
        weights = metrics.resample_weights(len(segments), 200, "seed")
        interval = metrics.chrf_difference(stats, stats, weights)
        assert interval == (0, 0, 0)
        assert not interval.excludes_zero

    def test_clear_difference_excludes_zero(self):
        segments = make_segments(40)
        references = [s.reference for s in segments]
        better = metrics.chrf_stats(references, references)
        worse = metrics.chrf_stats([r[:8] for r in references], references)
        weights = metrics.resample_weights(len(segments), 200, "seed")
        assert metrics.chrf_difference(better, worse, weights).low > 0

    def test_mean_interval(self):
        values = np.array([1.0, 0.0, 1.0, 1.0])
        weights = metrics.resample_weights(4, 500, "seed")
        interval = metrics.mean_interval(values, weights)
        assert interval.value == 0.75
        assert 0 <= interval.low <= 0.75 <= interval.high <= 1

    def test_perfect_agreement(self):
        assert metrics.agreement([[1, 1], [3, 3], [5, 5]]) == 1

    def test_disagreement_scores_below_chance(self):
        assert metrics.agreement([[1, 5], [5, 1], [1, 5]]) < 0

    def test_known_alpha(self):
        # Worked by hand from the interval formula: observed 1, expected 66/30.
        assert metrics.agreement([[1, 2], [2, 3], [3, 4]]) == pytest.approx(6 / 11)

    def test_agreement_needs_two_scores(self):
        assert metrics.agreement([[3], [4]]) is None


class TestConfig:
    def test_baseline_file_is_valid(self):
        config = EvalConfig.from_toml(BASELINE)
        assert len(config.model_ids("compare")) == 3
        assert len(config.model_ids("base")) == 3

    def test_judge_cannot_share_a_family_with_a_candidate(self):
        with pytest.raises(ValueError, match="also wrote"):
            make_config(
                models=[
                    {"id": "anthropic/claude", "role": "compare"},
                    {"id": BASE, "role": "base"},
                ]
            )

    def test_both_roles_are_required(self):
        with pytest.raises(ValueError, match="at least one"):
            make_config(
                models=[{"id": TEACHER, "role": "compare"}, {"id": "x/y", "role": "compare"}]
            )

    def test_unknown_settings_fail(self):
        with pytest.raises(ValueError):
            make_config(surprise=1)


def openrouter(handler) -> translate.OpenRouter:
    return translate.OpenRouter(
        "key", {"x/thinks": "low"}, httpx.MockTransport(handler), wait=lambda _: None
    )


def success(text: str = "Hello", cost: float = 0.01) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}], "usage": {"cost": cost}},
    )


class TestOpenRouter:
    def test_sends_the_apps_routing(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return success(" Kumusta \n")

        answer = openrouter(handler).complete("x/y", "prompt")
        assert answer == translate.Completion("Kumusta", 0.01)
        assert seen["provider"] == {"data_collection": "deny", "zdr": True}
        assert seen["reasoning"] == {"enabled": False}

    def test_retries_rate_limits_and_server_errors(self):
        responses = [httpx.Response(429), httpx.Response(503), success()]
        assert openrouter(lambda _: responses.pop(0)).complete("x/y", "p").text == "Hello"

    def test_retries_network_errors(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ConnectError("down")
            return success()

        assert openrouter(handler).complete("x/y", "p").text == "Hello"

    def test_gives_up_after_repeated_failures(self):
        with pytest.raises(translate.CompletionError, match="503"):
            openrouter(lambda _: httpx.Response(503)).complete("x/y", "p")

    def test_auth_failure_stops_the_run(self):
        with pytest.raises(translate.RequestRejected, match="HTTP 401"):
            openrouter(lambda _: httpx.Response(401)).complete("x/y", "p")

    def test_rejection_names_the_model_and_the_reason(self):
        body = {"error": {"message": "Reasoning is mandatory", "code": 400}}
        with pytest.raises(translate.RequestRejected, match="x/y: Reasoning is mandatory"):
            openrouter(lambda _: httpx.Response(400, json=body)).complete("x/y", "p")

    def test_reasoning_is_off_unless_a_model_needs_it(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return success()

        client = openrouter(handler)
        client.complete("x/thinks", "p")
        client.complete("x/y", "p")
        assert seen[0]["reasoning"] == {"effort": "low"}
        assert seen[1]["reasoning"] == {"enabled": False}

    def test_missing_cost_stops_the_run(self):
        response = httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})
        with pytest.raises(KeyError):
            openrouter(lambda _: response).complete("x/y", "p")

    def test_empty_answer_is_a_failure(self):
        with pytest.raises(translate.CompletionError, match="empty"):
            openrouter(lambda _: success("  ")).complete("x/y", "p")


class FakeCompleter:
    def __init__(self, fail_on: str | None = None, cost: float = 0.01):
        self.calls: list[tuple[str, str]] = []
        self.fail_on = fail_on
        self.cost = cost

    def complete(self, model: str, prompt: str) -> translate.Completion:
        self.calls.append((model, prompt))
        if self.fail_on and self.fail_on in prompt:
            raise translate.CompletionError("broken")
        return translate.Completion(f"{model} says hi", self.cost)


class TestTranslate:
    def test_prompt_matches_the_app(self):
        segment = make_segment(0, "Hello.", "Kumusta.")
        assert translate.build_prompt(segment) == (
            "Translate the following text from English to Cebuano. Return only the translated "
            "text with no explanation or commentary.\n\nHello."
        )

    def test_records_failures_without_stopping(self, tmp_path):
        path = tmp_path / "out.jsonl"
        segments = make_segments(3)
        work = translate.missing_work(segments, [BASE], {})
        spent = translate.translate(work, FakeCompleter("number 1."), path, 5.0)
        records = translate.read_translations(path)
        assert spent == pytest.approx(0.02)
        assert records[(BASE, segments[1].id)].error == "broken"
        assert records[(BASE, segments[0].id)].output == f"{BASE} says hi"

    def test_rerun_fetches_only_what_is_missing_or_failed(self, tmp_path):
        path = tmp_path / "out.jsonl"
        segments = make_segments(3)
        translate.translate(
            translate.missing_work(segments, [BASE], {}), FakeCompleter("number 1."), path, 5.0
        )
        done = translate.read_translations(path)
        retry = translate.missing_work(segments, [BASE], done)
        assert [segment.id for _, segment in retry] == [segments[1].id]
        translate.translate(retry, FakeCompleter(), path, 5.0)
        assert translate.read_translations(path)[(BASE, segments[1].id)].output is not None

    def test_budget_stops_the_run_and_keeps_finished_work(self, tmp_path):
        path = tmp_path / "out.jsonl"
        work = translate.missing_work(make_segments(200), [BASE], {})
        completer = FakeCompleter(cost=1.0)
        with pytest.raises(translate.BudgetExceeded):
            translate.translate(work, completer, path, 2.5)
        assert len(completer.calls) <= 3 + translate.WORKERS
        assert len(translate.read_translations(path)) == len(completer.calls)

    def test_earlier_spend_counts_against_the_budget(self, tmp_path):
        work = translate.missing_work(make_segments(1), [BASE], {})
        with pytest.raises(translate.BudgetExceeded):
            translate.translate(work, FakeCompleter(cost=1.0), tmp_path / "o.jsonl", 2.5, 2.0)


def outputs_for(segments, left="left words", right="right words"):
    return {
        **{(BASE, s.id): f"{left} {i}" for i, s in enumerate(segments)},
        **{(TEACHER, s.id): f"{right} {i}" for i, s in enumerate(segments)},
    }


def verdict(segment, first, second, winner):
    return JudgeVerdict(segment_id=segment.id, first=first, second=second, winner=winner)


class TestJudge:
    def test_sample_is_stable_and_ignores_order(self):
        segments = make_segments(30)
        first = judge.choose_segments(segments, 5, "seed")
        assert first == judge.choose_segments(list(reversed(segments)), 5, "seed")
        assert first != judge.choose_segments(segments, 5, "other")

    def test_plans_both_orders(self):
        segments = make_segments(2)
        planned = judge.plan_comparisons(segments, [(BASE, TEACHER)], outputs_for(segments))
        assert len(planned) == 4
        assert {(c.first, c.second) for c in planned} == {(BASE, TEACHER), (TEACHER, BASE)}
        assert len({c.request_id for c in planned}) == 4

    def test_identical_and_failed_answers_cost_nothing(self):
        segments = make_segments(2)
        outputs = outputs_for(segments)
        outputs[(BASE, segments[0].id)] = outputs[(TEACHER, segments[0].id)]
        del outputs[(BASE, segments[1].id)]
        assert judge.plan_comparisons(segments, [(BASE, TEACHER)], outputs) == []

    def test_prompt_never_names_a_model(self):
        segments = make_segments(1)
        comparison = judge.plan_comparisons(segments, [(BASE, TEACHER)], outputs_for(segments))[0]
        prompt = judge.build_prompt(comparison)
        assert BASE not in prompt
        assert TEACHER not in prompt
        assert "left words 0" in prompt
        assert segments[0].reference not in prompt

    def test_letter_maps_to_the_model_shown_there(self):
        segments = make_segments(1)
        one, other = judge.plan_comparisons(segments, [(BASE, TEACHER)], outputs_for(segments))
        assert judge.to_verdict(one, judge.Answer("A")).winner == one.first
        assert judge.to_verdict(other, judge.Answer("A")).winner == other.first
        assert judge.to_verdict(one, judge.Answer(None)).winner is None

    def test_scores_wins_ties_and_splits(self):
        segments = make_segments(3)
        rows = [
            verdict(segments[0], BASE, TEACHER, BASE),
            verdict(segments[0], TEACHER, BASE, BASE),
            verdict(segments[1], BASE, TEACHER, BASE),
            verdict(segments[1], TEACHER, BASE, TEACHER),
            verdict(segments[2], BASE, TEACHER, TEACHER),
            verdict(segments[2], TEACHER, BASE, TEACHER),
        ]
        verdicts = {judge.judge_key(v): v for v in rows}
        scored = judge.score_pair(segments, BASE, TEACHER, outputs_for(segments), verdicts)
        assert scored.scores == [1, 0.5, 0]
        assert scored.split_orders == 1
        assert scored.unusable == 0

    def test_automatic_results_need_no_verdict(self):
        segments = make_segments(3)
        outputs = outputs_for(segments)
        outputs[(BASE, segments[0].id)] = outputs[(TEACHER, segments[0].id)]
        del outputs[(TEACHER, segments[1].id)]
        del outputs[(BASE, segments[2].id)]
        scored = judge.score_pair(segments, BASE, TEACHER, outputs, {})
        assert scored.scores == [0.5, 1, 0]

    def test_unanswered_sentences_are_counted_not_scored(self):
        segments = make_segments(1)
        rows = [
            verdict(segments[0], BASE, TEACHER, None),
            verdict(segments[0], TEACHER, BASE, BASE),
        ]
        verdicts = {judge.judge_key(v): v for v in rows}
        scored = judge.score_pair(segments, BASE, TEACHER, outputs_for(segments), verdicts)
        assert scored.scores == []
        assert scored.unusable == 1


def batch_result(request_id, text='{"better": "A"}', stop_reason="end_turn", kind="succeeded"):
    message = SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=7),
    )
    return SimpleNamespace(custom_id=request_id, result=SimpleNamespace(type=kind, message=message))


class FakeBatches:
    def __init__(self, results=None):
        self.created = []
        self.results_for = results or (lambda requests: [])
        self.polls = 0

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id="batch-1")

    def retrieve(self, batch_id):
        self.polls += 1
        return SimpleNamespace(processing_status="ended" if self.polls > 1 else "in_progress")

    def results(self, batch_id):
        return self.results_for(self.created[-1] if self.created else [])


def fake_client(batches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


class TestBatchJudge:
    def comparisons(self):
        segments = make_segments(1)
        return judge.plan_comparisons(segments, [(BASE, TEACHER)], outputs_for(segments))

    def test_submits_polls_and_reads_answers(self, tmp_path):
        batches = FakeBatches(lambda reqs: [batch_result(r["custom_id"]) for r in reqs])
        client = judge.BatchJudge(
            fake_client(batches), "claude-opus-5-5", "low", tmp_path / "b.json", wait=lambda _: None
        )
        comparisons = self.comparisons()
        answers = client.decide(comparisons)
        request = batches.created[0][0]
        assert request["params"]["model"] == "claude-opus-5-5"
        assert request["params"]["output_config"]["effort"] == "low"
        assert len(answers) == 2
        assert answers[comparisons[0].request_id] == judge.Answer("A", 100, 7)
        assert batches.polls == 2

    def test_rerun_resumes_instead_of_paying_again(self, tmp_path):
        comparisons = self.comparisons()
        saved = tmp_path / "b.json"
        saved.write_text('{"id": "batch-old"}')
        batches = FakeBatches(lambda _: [batch_result(c.request_id) for c in comparisons])
        client = judge.BatchJudge(fake_client(batches), "m", "low", saved, wait=lambda _: None)
        assert len(client.decide(comparisons)) == 2
        assert batches.created == []

    @pytest.mark.parametrize(
        "result",
        [
            batch_result("x", stop_reason="refusal"),
            batch_result("x", stop_reason="max_tokens"),
            batch_result("x", kind="errored"),
            batch_result("x", kind="expired"),
            batch_result("x", text='{"better": "C"}'),
        ],
    )
    def test_unusable_results_give_no_letter(self, result):
        assert judge._answer_from(result).letter is None


def sheet_rows(segments, models=(BASE, TEACHER), round_name="r1", sentences=4, checks=1):
    outputs = {(m, s.id): f"output {models.index(m)} {s.id}" for m in models for s in segments}
    return rating.build_rows(segments, outputs, list(models), "seed", round_name, sentences, checks)


def write_scores(path, rows, score=lambda row: ("4", "3")):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(rating.HEADER)
        for row in rows:
            writer.writerow([row.row, row.source, row.translation, *score(row)])


class TestRating:
    def test_sheet_hides_who_wrote_each_row(self):
        sheet = rating.render_sheet(sheet_rows(make_segments()))
        assert sheet.startswith(",".join(rating.HEADER))
        for hidden in (BASE, TEACHER, "check:"):
            assert hidden not in sheet

    def test_row_ids_do_not_reveal_the_model(self):
        rows = sheet_rows(make_segments())
        assert all(len(row.row) == 10 and "/" not in row.row for row in rows)
        assert len({row.row for row in rows}) == len(rows)

    def test_same_sentences_but_a_new_order_each_round(self):
        segments = make_segments()
        first = sheet_rows(segments, round_name="a")
        second = sheet_rows(segments, round_name="b")
        assert {r.segment_id for r in first} == {r.segment_id for r in second}
        assert [r.segment_id for r in first] != [r.segment_id for r in second]

    def test_models_are_shuffled_together(self):
        models = [row.model_id for row in sheet_rows(make_segments(), checks=0)]
        assert models != sorted(models)

    def test_checks_are_mixed_in(self):
        rows = sheet_rows(make_segments(), checks=2)
        good = [r for r in rows if r.check == "good"]
        wrong = [r for r in rows if r.check == "wrong"]
        assert len(good) == len(wrong) == 2
        segments = {s.id: s for s in make_segments()}
        assert all(r.translation == segments[r.segment_id].reference for r in good)
        assert all(r.translation != segments[r.segment_id].reference for r in wrong)

    def test_failed_translations_are_left_out(self):
        segments = make_segments(1)
        rows = rating.build_rows(segments, {}, [BASE], "seed", "r", 1, 0)
        assert rows == []

    @pytest.mark.parametrize("text", ["=1+1", "+63 917", "-dash", "@home"])
    def test_cells_cannot_run_as_formulas(self, text):
        row = rating.SheetRow("r", text, text, "s", BASE, None)
        cells = next(csv.reader(rating.render_sheet([row]).splitlines()[1:]))
        assert cells[1] == f"'{text}"
        assert cells[2] == f"'{text}"

    def test_import_joins_scores_to_models(self, tmp_path):
        rows = sheet_rows(make_segments())
        path = tmp_path / "scored.csv"
        write_scores(path, rows)
        imported = rating.import_sheet(path, rating.key_entries(rows), "rater-1", "r1")
        assert imported.blank == 0
        assert len(imported.ratings) == len(rows)
        first = next(r for r in imported.ratings if r.check is None)
        assert first.model_id in (BASE, TEACHER)
        assert (first.adequacy, first.fluency) == (4, 3)

    def test_blank_rows_are_counted(self, tmp_path):
        rows = sheet_rows(make_segments())
        path = tmp_path / "scored.csv"
        write_scores(path, rows, lambda row: ("", "") if row == rows[0] else ("4", "4"))
        imported = rating.import_sheet(path, rating.key_entries(rows), "rater-1", "r1")
        assert imported.blank == 1

    @pytest.mark.parametrize("bad", [("6", "3"), ("0", "3"), ("4", ""), ("x", "3"), ("3.5", "3")])
    def test_bad_scores_are_rejected(self, tmp_path, bad):
        rows = sheet_rows(make_segments())
        path = tmp_path / "scored.csv"
        write_scores(path, rows, lambda row: bad)
        with pytest.raises(ValueError, match="whole number"):
            rating.import_sheet(path, rating.key_entries(rows), "r", "r1")

    def test_foreign_rows_are_rejected(self, tmp_path):
        rows = sheet_rows(make_segments())
        path = tmp_path / "scored.csv"
        write_scores(path, rows)
        key = rating.key_entries(sheet_rows(make_segments(), round_name="other"))
        with pytest.raises(ValueError, match="not in the key"):
            rating.import_sheet(path, key, "r", "r1")

    def test_missing_rows_are_rejected(self, tmp_path):
        rows = sheet_rows(make_segments())
        path = tmp_path / "scored.csv"
        write_scores(path, rows[:-1])
        with pytest.raises(ValueError, match="missing"):
            rating.import_sheet(path, rating.key_entries(rows), "r", "r1")

    def test_excel_byte_order_mark_is_accepted(self, tmp_path):
        rows = sheet_rows(make_segments())
        path = tmp_path / "scored.csv"
        write_scores(path, rows)
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
        assert rating.import_sheet(path, rating.key_entries(rows), "r", "r1").ratings

    def test_careless_rounds_are_dropped(self):
        def score(check, adequacy, rater="careful"):
            return HumanRating(
                segment_id="s",
                model_id=BASE,
                rater_id=rater,
                round="r",
                adequacy=adequacy,
                fluency=3,
                check=check,
            )

        careful = [score("good", 5), score("wrong", 1), score(None, 4)]
        careless = [
            score("good", 1, "rushed"),
            score("wrong", 5, "rushed"),
            score(None, 4, "rushed"),
        ]
        assert rating.check_pass_rate(careful) == 1
        assert rating.check_pass_rate(careless) == 0
        assert rating.trusted_ratings(careful + careless, 0.8) == careful


def make_inputs(segments, base_output, teacher_output=None, **overrides) -> report.EvalInputs:
    config = overrides.pop("config", make_config())
    teacher_output = teacher_output or (lambda s: s.reference)
    records = {}
    for model, output in ((BASE, base_output), (TEACHER, teacher_output)):
        for segment in segments:
            text = output(segment)
            records[(model, segment.id)] = TranslationRecord(
                segment_id=segment.id,
                model_id=model,
                output=text,
                error=None if text else "broken",
                cost_usd=0.001 if text else 0,
            )
    return report.EvalInputs(config, segments, records, **overrides)


class TestReport:
    def test_finds_headroom(self):
        segments = make_segments(40)
        built = report.build_report(make_inputs(segments, lambda s: s.reference[:8]))
        assert built["best_compare"] == TEACHER
        assert built["headroom"][0]["has_headroom"] is True
        assert built["recommended_base"] == BASE
        assert built["models"][0]["model"] == TEACHER
        assert built["partial"] is False

    def test_no_headroom_when_the_base_matches_the_teachers(self):
        segments = make_segments(40)
        built = report.build_report(make_inputs(segments, lambda s: s.reference))
        assert built["recommended_base"] is None
        assert "No candidate base" in report.render_html(built)

    def test_failures_count_against_a_model(self):
        segments = make_segments(10)
        built = report.build_report(
            make_inputs(segments, lambda s: s.reference if "1" not in s.id[-1] else None)
        )
        row = next(m for m in built["models"] if m["model"] == BASE)
        assert row["failed"] == 1
        assert row["chrf"]["value"] < 100

    def test_flags_partial_runs(self):
        segments = make_segments(10)
        inputs = make_inputs(segments[:4], lambda s: s.reference)
        partial = report.EvalInputs(inputs.config, segments, inputs.translations)
        built = report.build_report(partial)
        assert built["partial"] is True
        assert "Do not cite" in report.render_html(built)

    def test_no_translations_is_an_error(self):
        with pytest.raises(ValueError, match="no translations"):
            report.build_report(report.EvalInputs(make_config(), make_segments(2), {}))

    def test_judge_win_rate_in_report(self):
        segments = make_segments(20)
        config = make_config(judge={"sentences": 20})
        verdicts = {}
        for segment in segments:
            for first, second in ((BASE, TEACHER), (TEACHER, BASE)):
                found = verdict(segment, first, second, BASE)
                verdicts[judge.judge_key(found)] = found
        built = report.build_report(
            make_inputs(segments, lambda s: "other", config=config, verdicts=verdicts)
        )
        pair = built["judge"]["pairs"][0]
        assert pair["base_win_rate"]["value"] == 1
        assert pair["sentences"] == 20

    def test_partial_run_still_shows_the_judge(self):
        segments = make_segments(30)
        covered = segments[:3]
        verdicts = {}
        for segment in covered:
            for first, second in ((BASE, TEACHER), (TEACHER, BASE)):
                found = verdict(segment, first, second, TEACHER)
                verdicts[judge.judge_key(found)] = found
        inputs = make_inputs(covered, lambda s: "other")
        partial = report.EvalInputs(inputs.config, segments, inputs.translations, verdicts)
        pairs = report.build_report(partial)["judge"]["pairs"]
        assert pairs[0]["sentences"] == 3
        assert pairs[0]["base_win_rate"]["value"] == 0

    def test_human_section(self):
        segments = make_segments(20)
        ratings = [
            HumanRating(
                segment_id=s.id,
                model_id=model,
                rater_id=rater,
                round="r",
                adequacy=score,
                fluency=score,
            )
            for s in segments
            for model, score in ((BASE, 2), (TEACHER, 5))
            for rater in ("one", "two")
        ]
        built = report.build_report(make_inputs(segments, lambda s: "x", ratings=ratings))
        models = {row["model"]: row for row in built["human"]["models"]}
        assert models[TEACHER]["adequacy"]["value"] == 5
        assert models[BASE]["fluency"]["value"] == 2
        assert built["human"]["raters"] == ["one", "two"]
        assert built["human"]["agreement"]["adequacy_between_raters"] == 1
        assert "Human ratings" in report.render_html(built)

    def test_html_escapes_model_output(self):
        segments = make_segments(12)
        built = report.build_report(make_inputs(segments, lambda s: "<script>alert(1)</script>"))
        page = report.render_html(built)
        assert "<script>" not in page
        assert "&lt;script&gt;" in page


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    data = tmp_path / "data"
    segments = make_segments(30)
    write_jsonl(data / "eval" / "test-set.jsonl", segments)
    config = tmp_path / "eval.toml"
    config.write_text(
        'name = "test-run"\neval_set = "test-set"\nresamples = 100\n'
        f'[[models]]\nid = "{TEACHER}"\nrole = "compare"\n'
        f'[[models]]\nid = "{BASE}"\nrole = "base"\n'
        "[judge]\nsentences = 5\n[rating]\nsentences = 4\nchecks = 1\n",
        encoding="utf-8",
    )
    completer = FakeCompleter()
    monkeypatch.setattr(translate, "OpenRouter", lambda *_args: completer)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    return SimpleNamespace(data=data, config=config, completer=completer, tmp=tmp_path)


def cli(*args: object) -> str:
    result = runner.invoke(app, [str(arg) for arg in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


class TestCommands:
    def common(self, workspace):
        return ("--config", workspace.config, "--data-dir", workspace.data)

    def test_translate_sends_nothing_without_yes(self, workspace):
        output = cli("eval", "translate", *self.common(workspace))
        assert "60 requests" in output
        assert "Nothing was sent" in output
        assert workspace.completer.calls == []

    def test_translate_needs_a_key(self, workspace, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY")
        result = runner.invoke(
            app,
            [
                "eval",
                "translate",
                "--yes",
                "--config",
                str(workspace.config),
                "--data-dir",
                str(workspace.data),
            ],
        )
        assert result.exit_code == 1
        assert "OPENROUTER_API_KEY" in result.output

    def test_limit_then_report_is_marked_partial(self, workspace):
        cli("eval", "translate", "--limit", 3, "--yes", *self.common(workspace))
        assert len(workspace.completer.calls) == 6
        output = cli("report", *self.common(workspace))
        assert "partial: 3 of 30" in output
        assert (workspace.data / "runs" / "test-run" / "report.html").exists()

    def test_budget_stop_exits_with_an_error(self, workspace):
        workspace.completer.cost = 100.0
        result = runner.invoke(
            app,
            [
                "eval",
                "translate",
                "--yes",
                "--config",
                str(workspace.config),
                "--data-dir",
                str(workspace.data),
            ],
        )
        assert result.exit_code == 1
        assert "Finished work is saved" in result.output

    def test_judge_shows_the_plan_without_yes(self, workspace):
        cli("eval", "translate", "--yes", *self.common(workspace))
        output = cli("eval", "judge", *self.common(workspace))
        assert "0 comparisons" in output or "comparisons on 5 sentences" in output
        assert "Nothing was sent" in output

    def test_rating_round_trip(self, workspace):
        cli("eval", "translate", "--yes", *self.common(workspace))
        cli("rate", "export", "--round", "m2-a", *self.common(workspace))
        directory = workspace.data / "ratings" / "m2-a"
        key = json.loads((directory / "key.json").read_text())
        rows = list(csv.DictReader((directory / "sheet.csv").open(encoding="utf-8")))
        assert len(rows) == len(key) == 10
        for row in rows:
            entry = key[row["row"]]
            bad = entry["check"] == "wrong"
            row["adequacy"], row["fluency"] = ("1", "1") if bad else ("5", "4")
        scored = workspace.tmp / "scored.csv"
        with scored.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rating.HEADER)
            writer.writeheader()
            writer.writerows(rows)
        output = cli(
            "rate",
            "import",
            scored,
            "--round",
            "m2-a",
            "--rater",
            "rater-1",
            *self.common(workspace),
        )
        assert "hidden checks passed: 100% (ok)" in output
        assert len(read_jsonl(directory / "ratings-rater-1.jsonl", HumanRating)) == 10
        cli("report", *self.common(workspace))
        built = json.loads((workspace.data / "runs" / "test-run" / "report.json").read_text())
        assert built["human"]["raters"] == ["rater-1"]
