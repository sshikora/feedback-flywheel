"""The `flywheel` command line. Each stage reads and writes files under --data-dir."""

import json
import os
from pathlib import Path
from typing import Annotated

import anthropic
import typer

from flywheel.curate.config import CurationConfig
from flywheel.curate.pipeline import curate as run_curation
from flywheel.evals import judge as judging
from flywheel.evals import rating, report
from flywheel.evals import translate as translating
from flywheel.evals.config import EvalConfig
from flywheel.ingest import comparisons, flores, jsonl
from flywheel.schema import EvalSegment, FeedbackEvent, HumanRating
from flywheel.schema.export import stale_schemas, write_schemas
from flywheel.scrub import scrub_events
from flywheel.storage import append_jsonl, read_jsonl, write_json, write_jsonl

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
ingest_app = typer.Typer(
    no_args_is_help=True, help="Load a source into scrubbed, canonical records."
)
app.add_typer(ingest_app, name="ingest")
eval_app = typer.Typer(no_args_is_help=True, help="Run models and the judge over an eval set.")
app.add_typer(eval_app, name="eval")
rate_app = typer.Typer(no_args_is_help=True, help="Blind human rating by spreadsheet.")
app.add_typer(rate_app, name="rate")

DataDir = Annotated[Path, typer.Option(help="Where pipeline data lives. Never committed.")]
Config = Annotated[Path, typer.Option(help="An eval TOML file.")]
Confirm = Annotated[
    bool, typer.Option("--yes", help="Spend money. Without it, only show the plan.")
]


def _store_events(events: list[FeedbackEvent], name: str, data_dir: Path) -> None:
    scrubbed = scrub_events(events)
    write_jsonl(data_dir / "events" / f"{name}.jsonl", scrubbed.events)
    typer.echo(
        f"stored {len(scrubbed.events)} events "
        f"({scrubbed.redacted} redacted, {scrubbed.dropped_for_secrets} dropped for secrets)"
    )


@ingest_app.command("jsonl")
def ingest_jsonl(path: Path, name: str = "jsonl", data_dir: DataDir = Path("data")) -> None:
    """Import a file of FeedbackEvent records."""
    _store_events(jsonl.load_events(path), name, data_dir)


@ingest_app.command("comparisons")
def ingest_comparisons(path: Path, data_dir: DataDir = Path("data")) -> None:
    """Import an export of the translation app's comparisons table."""
    result = comparisons.load_comparisons(path)
    for reason, count in sorted(result.skipped.items()):
        typer.echo(f"skipped {count} ({reason})")
    _store_events(result.events, "comparisons", data_dir)


@ingest_app.command("flores")
def ingest_flores(
    src: str = "en",
    tgt: str = "ceb",
    split: str = "devtest",
    data_dir: DataDir = Path("data"),
    cache_dir: Annotated[Path, typer.Option(help="Where the download is kept.")] = Path(".cache"),
) -> None:
    """Download FLORES-200 and store one split as an eval set."""
    segments = flores.load_segments(flores.fetch_archive(cache_dir), src, tgt, split)
    write_jsonl(data_dir / "eval" / f"flores200-{split}-{src}-{tgt}.jsonl", segments)
    typer.echo(f"stored {len(segments)} {split} segments for {src}-{tgt}")


@app.command()
def curate(
    data_dir: DataDir = Path("data"),
    config: Annotated[Path | None, typer.Option(help="A curation TOML file.")] = None,
) -> None:
    """Dedupe, score, split and leak-check stored events."""
    settings = CurationConfig.from_toml(config) if config else CurationConfig()
    events = [
        event
        for path in sorted((data_dir / "events").glob("*.jsonl"))
        for event in read_jsonl(path, FeedbackEvent)
    ]
    segments = [
        segment
        for path in sorted((data_dir / "eval").glob("*.jsonl"))
        for segment in read_jsonl(path, EvalSegment)
    ]
    result = run_curation(events, segments, settings)
    write_jsonl(data_dir / "curated" / "train.jsonl", result.train)
    write_jsonl(data_dir / "curated" / "heldout.jsonl", result.heldout)
    write_json(data_dir / "curated" / "report.json", result.report)
    typer.echo(f"train {len(result.train)}, held out {len(result.heldout)}")
    for reason, count in result.report["dropped"].items():
        typer.echo(f"dropped {count} ({reason})")


def _load_segments(config: EvalConfig, data_dir: Path) -> list[EvalSegment]:
    path = data_dir / "eval" / f"{config.eval_set}.jsonl"
    if not path.exists():
        raise typer.BadParameter(f"{path} is missing. Run `flywheel ingest flores` first.")
    return read_jsonl(path, EvalSegment)


def _run_dir(config: EvalConfig, data_dir: Path) -> Path:
    return data_dir / "runs" / config.name


def _load_ratings(data_dir: Path) -> list[HumanRating]:
    return [
        rating_record
        for path in sorted((data_dir / "ratings").glob("*/ratings-*.jsonl"))
        for rating_record in read_jsonl(path, HumanRating)
    ]


def _load_inputs(config: EvalConfig, data_dir: Path) -> report.EvalInputs:
    run = _run_dir(config, data_dir)
    return report.EvalInputs(
        config=config,
        segments=_load_segments(config, data_dir),
        translations=translating.read_translations(run / "translations.jsonl"),
        verdicts=judging.read_verdicts(run / "verdicts.jsonl"),
        ratings=_load_ratings(data_dir),
    )


@eval_app.command("translate")
def eval_translate(
    config: Config = Path("configs/eval.baseline.toml"),
    limit: Annotated[int | None, typer.Option(help="Use only the first N sentences.")] = None,
    yes: Confirm = False,
    data_dir: DataDir = Path("data"),
) -> None:
    """Run every model over the eval set. Reruns fetch only what is missing."""
    settings = EvalConfig.from_toml(config)
    segments = _load_segments(settings, data_dir)[:limit]
    path = _run_dir(settings, data_dir) / "translations.jsonl"
    done = translating.read_translations(path)
    models = [model.id for model in settings.models]
    work = translating.missing_work(segments, models, done)
    spent = sum(record.cost_usd for record in done.values())
    typer.echo(
        f"{len(work)} requests across {len(models)} models, "
        f"${spent:.2f} spent so far, limit ${settings.max_cost_usd:.2f}"
    )
    if not yes:
        typer.echo("Nothing was sent. Pass --yes to run this.")
        return
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("OPENROUTER_API_KEY is not set.", err=True)
        raise typer.Exit(code=1)
    efforts = {m.id: m.reasoning_effort for m in settings.models if m.reasoning_effort}
    completer = translating.OpenRouter(api_key, efforts)
    try:
        total = translating.translate(work, completer, path, settings.max_cost_usd, spent)
    except (translating.BudgetExceeded, translating.RequestRejected) as err:
        typer.echo(f"stopped: {err}. Finished work is saved.", err=True)
        raise typer.Exit(code=1) from err
    failed = sum(record.output is None for record in translating.read_translations(path).values())
    typer.echo(f"done, ${total:.2f} spent in total, {failed} failed")


@eval_app.command("judge")
def eval_judge(
    config: Config = Path("configs/eval.baseline.toml"),
    yes: Confirm = False,
    data_dir: DataDir = Path("data"),
) -> None:
    """Compare each base with each compare model, blind and in both orders."""
    settings = EvalConfig.from_toml(config)
    inputs = _load_inputs(settings, data_dir)
    covered = {segment_id for _, segment_id in inputs.translations}
    sample = judging.choose_segments(
        [s for s in inputs.segments if s.id in covered], settings.judge.sentences, settings.seed
    )
    pairs = [
        (base, teacher)
        for base in settings.model_ids("base")
        for teacher in settings.model_ids("compare")
    ]
    comparisons_needed = [
        comparison
        for comparison in judging.plan_comparisons(sample, pairs, report.stored_outputs(inputs))
        if (segment_key := (comparison.segment.id, comparison.first, comparison.second))
        not in inputs.verdicts
        or inputs.verdicts[segment_key].winner is None
    ]
    typer.echo(
        f"{len(comparisons_needed)} comparisons on {len(sample)} sentences "
        f"with {settings.judge.model}, batched at half price"
    )
    if not yes or not comparisons_needed:
        typer.echo("Nothing was sent." + (" Pass --yes to run this." if comparisons_needed else ""))
        return
    run = _run_dir(settings, data_dir)
    try:
        client = anthropic.Anthropic()
        judge = judging.BatchJudge(
            client, settings.judge.model, settings.judge.effort, run / "judge-batch.json"
        )
        answers = judge.decide(comparisons_needed)
    except anthropic.AnthropicError as err:
        typer.echo(f"judge failed: {err}", err=True)
        raise typer.Exit(code=1) from err
    verdicts = [
        judging.to_verdict(comparison, answers.get(comparison.request_id, judging.Answer(None)))
        for comparison in comparisons_needed
    ]
    append_jsonl(run / "verdicts.jsonl", verdicts)
    (run / "judge-batch.json").unlink(missing_ok=True)
    unusable = sum(verdict.winner is None for verdict in verdicts)
    typer.echo(f"stored {len(verdicts)} verdicts, {unusable} unusable")


@rate_app.command("export")
def rate_export(
    round_name: Annotated[str, typer.Option("--round", help="For example m2-a.")],
    config: Config = Path("configs/eval.baseline.toml"),
    models: Annotated[
        str | None, typer.Option(help="Comma-separated model ids. Defaults to all of them.")
    ] = None,
    data_dir: DataDir = Path("data"),
) -> None:
    """Write a blind sheet for raters, and a key that stays with you."""
    settings = EvalConfig.from_toml(config)
    inputs = _load_inputs(settings, data_dir)
    chosen = models.split(",") if models else [model.id for model in settings.models]
    unknown = set(chosen) - {model.id for model in settings.models}
    if unknown:
        raise typer.BadParameter(f"not in this run: {sorted(unknown)}")
    rows = rating.build_rows(
        inputs.segments,
        report.stored_outputs(inputs),
        chosen,
        settings.seed,
        round_name,
        settings.rating.sentences,
        settings.rating.checks,
    )
    directory = data_dir / "ratings" / round_name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sheet.csv").write_text(rating.render_sheet(rows), encoding="utf-8")
    write_json(directory / "key.json", rating.key_entries(rows))
    typer.echo(f"wrote {len(rows)} rows to {directory / 'sheet.csv'}")
    typer.echo(f"Keep {directory / 'key.json'} away from raters.")


@rate_app.command("import")
def rate_import(
    path: Path,
    round_name: Annotated[str, typer.Option("--round", help="The round this sheet belongs to.")],
    rater: Annotated[str, typer.Option(help="A pseudonymous id. Never a name or email.")],
    config: Config = Path("configs/eval.baseline.toml"),
    data_dir: DataDir = Path("data"),
) -> None:
    """Load a scored sheet and report how the hidden checks went."""
    settings = EvalConfig.from_toml(config)
    directory = data_dir / "ratings" / round_name
    key = json.loads((directory / "key.json").read_text(encoding="utf-8"))
    imported = rating.import_sheet(path, key, rater, round_name)
    write_jsonl(directory / f"ratings-{rater}.jsonl", imported.ratings)
    typer.echo(f"stored {len(imported.ratings)} ratings, {imported.blank} blank")
    rate = rating.check_pass_rate(imported.ratings)
    if rate is not None:
        status = "ok" if rate >= settings.rating.min_check_pass else "below the limit, left out"
        typer.echo(f"hidden checks passed: {rate:.0%} ({status})")


@app.command("report")
def write_report(
    config: Config = Path("configs/eval.baseline.toml"),
    data_dir: DataDir = Path("data"),
) -> None:
    """Build the report from stored results. Makes no API calls."""
    settings = EvalConfig.from_toml(config)
    results = report.build_report(_load_inputs(settings, data_dir))
    run = _run_dir(settings, data_dir)
    write_json(run / "report.json", results)
    (run / "report.html").write_text(report.render_html(results), encoding="utf-8")
    typer.echo(f"wrote {run / 'report.html'}")
    if results["partial"]:
        typer.echo(f"partial: {results['sentences']} of {results['sentences_in_set']} sentences")


@app.command()
def schema(
    out: Path = Path("schemas"),
    check: Annotated[bool, typer.Option(help="Fail if the files are out of date.")] = False,
) -> None:
    """Export JSON Schema for the canonical records."""
    if not check:
        write_schemas(out)
        typer.echo(f"wrote schemas to {out}")
        return
    stale = stale_schemas(out)
    if stale:
        typer.echo(f"out of date: {', '.join(stale)}. Run `flywheel schema`.", err=True)
        raise typer.Exit(code=1)
    typer.echo("schemas are current")


if __name__ == "__main__":
    app()
