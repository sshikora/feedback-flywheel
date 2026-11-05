"""The `flywheel` command line. Each stage reads and writes files under --data-dir."""

import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import anthropic
import typer

from flywheel.curate.config import CurationConfig
from flywheel.curate.pipeline import curate as run_curation
from flywheel.datasets import builders, snapshot
from flywheel.datasets.config import DatasetConfig
from flywheel.evals import hosted, rating, report
from flywheel.evals import judge as judging
from flywheel.evals import translate as translating
from flywheel.evals.config import EvalConfig, ModelSpec
from flywheel.ingest import comparisons, flores, jsonl, opus
from flywheel.registry.card import render_card as render_model_card
from flywheel.registry.store import Registry
from flywheel.schema import CuratedEvent, EvalSegment, FeedbackEvent, HumanRating, ModelRecord
from flywheel.schema.export import stale_schemas, write_schemas
from flywheel.scrub import scrub_events
from flywheel.storage import append_jsonl, read_jsonl, write_json, write_jsonl
from flywheel.train import run as training
from flywheel.train.config import RunConfig, config_hash
from flywheel.train.together import Together, TogetherError

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
ingest_app = typer.Typer(
    no_args_is_help=True, help="Load a source into scrubbed, canonical records."
)
app.add_typer(ingest_app, name="ingest")
eval_app = typer.Typer(no_args_is_help=True, help="Run models and the judge over an eval set.")
app.add_typer(eval_app, name="eval")
rate_app = typer.Typer(no_args_is_help=True, help="Blind human rating by spreadsheet.")
app.add_typer(rate_app, name="rate")
build_app = typer.Typer(no_args_is_help=True, help="Build training datasets from curated events.")
app.add_typer(build_app, name="build")
registry_app = typer.Typer(no_args_is_help=True, help="Trained models, their status and cards.")
app.add_typer(registry_app, name="registry")
train_app = typer.Typer(no_args_is_help=True, help="Fine-tune through Together AI.")
app.add_typer(train_app, name="train")

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
def ingest_comparisons(
    path: Path,
    data_dir: DataDir = Path("data"),
    training_from: Annotated[
        datetime | None,
        typer.Option(
            help="When the privacy page began naming the training provider, in UTC. "
            "Only votes from then on may train. Without it, none may."
        ),
    ] = None,
) -> None:
    """Import an export of the translation app's comparisons table."""
    if training_from is not None and training_from.tzinfo is None:
        training_from = training_from.replace(tzinfo=UTC)
    result = comparisons.load_comparisons(path, training_from)
    for reason, count in sorted(result.skipped.items()):
        typer.echo(f"skipped {count} ({reason})")
    cleared = sum(event.consent.training for event in result.events)
    typer.echo(f"{cleared} of {len(result.events)} votes cleared for training")
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


@ingest_app.command("opus")
def ingest_opus(
    corpus: Annotated[str, typer.Argument(help=f"One of: {', '.join(opus.CORPORA)}.")],
    src: str = "en",
    tgt: str = "ceb",
    data_dir: DataDir = Path("data"),
    cache_dir: Annotated[Path, typer.Option(help="Where the download is kept.")] = Path(".cache"),
) -> None:
    """Download a human-translated OPUS corpus and store its pairs as training events."""
    if corpus not in opus.CORPORA:
        raise typer.BadParameter(f"unknown corpus {corpus!r}, expected one of {list(opus.CORPORA)}")
    result = opus.load_events(opus.fetch_archive(corpus, cache_dir), corpus, src, tgt)
    for reason, count in sorted(result.skipped.items()):
        typer.echo(f"skipped {count} ({reason})")
    spec = opus.CORPORA[corpus]
    typer.echo(f"{spec.opus_name} {spec.version}, {spec.license}. {spec.attribution}")
    _store_events(result.events, opus.source_name(corpus), data_dir)


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


DatasetConfigOption = Annotated[Path | None, typer.Option("--config", help="A dataset TOML file.")]


@build_app.command("sft")
def build_sft(data_dir: DataDir = Path("data"), config: DatasetConfigOption = None) -> None:
    """Build an SFT snapshot: each input with the answer voters chose."""
    _build("sft", data_dir, config)


@build_app.command("dpo")
def build_dpo(data_dir: DataDir = Path("data"), config: DatasetConfigOption = None) -> None:
    """Build a DPO snapshot: the chosen answer against each answer voters passed over."""
    _build("dpo", data_dir, config)


def _build(kind: snapshot.Kind, data_dir: Path, config: Path | None) -> None:
    settings = DatasetConfig.from_toml(config) if config else DatasetConfig()
    path = data_dir / "curated" / "train.jsonl"
    if not path.exists():
        raise typer.BadParameter(f"{path} is missing. Run `flywheel curate` first.")
    items = read_jsonl(path, CuratedEvent)
    build = (builders.build_sft if kind == "sft" else builders.build_dpo)(items, settings)
    report_path = data_dir / "curated" / "report.json"
    source = {
        "path": "curated/train.jsonl",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "curated_events": len(items),
        "curation_config": json.loads(report_path.read_text())["config"]
        if report_path.exists()
        else None,
    }
    meta = snapshot.manifest(kind, build, settings, source)
    result = snapshot.write_snapshot(data_dir / "datasets", kind, build.examples, meta)
    state = "built" if result.created else "already built, unchanged:"
    typer.echo(
        f"{state} {result.directory} ({len(build.examples)} examples, {len(build.used)} events)"
    )
    for reason, count in sorted(build.dropped.items()):
        typer.echo(f"dropped {count} ({reason})")


def _registry(data_dir: Path) -> Registry:
    return Registry(data_dir / "registry" / "registry.sqlite")


def _snapshot_manifest(data_dir: Path, digest: str) -> dict[str, Any] | None:
    for path in sorted((data_dir / "datasets").glob(f"*-{digest[:12]}/manifest.json")):
        meta: dict[str, Any] = json.loads(path.read_text())
        if meta["hash"] == digest:
            return meta
    return None


@registry_app.command("add")
def registry_add(
    run_config: Path,
    provider: Annotated[str, typer.Option(help="Where it was trained, such as together.")],
    job_id: Annotated[str | None, typer.Option(help="The provider's job id.")] = None,
    weights: Annotated[str | None, typer.Option(help="Where the merged weights are kept.")] = None,
    cost: Annotated[float, typer.Option(help="Training cost in US dollars.")] = 0.0,
    data_dir: DataDir = Path("data"),
) -> None:
    """Register a trained model as a candidate, named after its run config."""
    config = RunConfig.from_toml(run_config)
    if _snapshot_manifest(data_dir, config.dataset) is None:
        raise typer.BadParameter(
            f"no snapshot in {data_dir / 'datasets'} has hash {config.dataset}"
        )
    registry = _registry(data_dir)
    try:
        if config.start_from is not None:
            registry.get(config.start_from)
        record = registry.register(
            ModelRecord(
                id=config.name,
                base_model=config.base_model,
                start_from=config.start_from,
                method=config.method,
                dataset_hash=config.dataset,
                run_config=run_config.as_posix(),
                run_config_hash=config_hash(run_config),
                provider=provider,
                provider_job_id=job_id,
                weights=weights,
                cost_usd=cost,
                status="candidate",
                created_at=datetime.now(UTC),
            )
        )
    except KeyError as missing:
        raise typer.BadParameter(f"start_from {missing} is not registered") from None
    except ValueError as err:
        raise typer.BadParameter(str(err)) from None
    finally:
        registry.close()
    typer.echo(f"registered {record.id} as a candidate")


@registry_app.command("list")
def registry_list(data_dir: DataDir = Path("data")) -> None:
    """Every registered model and its status."""
    registry = _registry(data_dir)
    try:
        for record in registry.models():
            typer.echo(f"{record.id}  {record.status}  {record.method}  {record.base_model}")
    finally:
        registry.close()


@registry_app.command("card")
def registry_card(model_id: str, data_dir: DataDir = Path("data")) -> None:
    """Write a model's card from the registry, its run config and its dataset."""
    registry = _registry(data_dir)
    try:
        record = registry.get(model_id)
        history = registry.history(model_id)
    except KeyError:
        raise typer.BadParameter(f"{model_id} is not registered") from None
    finally:
        registry.close()
    config = RunConfig.from_toml(Path(record.run_config))
    if config_hash(Path(record.run_config)) != record.run_config_hash:
        raise typer.BadParameter(f"{record.run_config} has changed since {model_id} was trained")
    card = render_model_card(
        record, config, history, _snapshot_manifest(data_dir, record.dataset_hash)
    )
    path = data_dir / "registry" / "cards" / f"{model_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")
    typer.echo(f"wrote {path}")


def _together() -> Together:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        typer.echo("TOGETHER_API_KEY is not set.", err=True)
        raise typer.Exit(code=1)
    return Together(api_key)


@contextmanager
def _training_errors() -> Iterator[None]:
    try:
        yield
    except (training.RunRefused, TogetherError) as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(code=1) from err


def _from_checkpoint(config: RunConfig, data_dir: Path) -> str | None:
    if config.start_from is None:
        return None
    registry = _registry(data_dir)
    try:
        parent = registry.get(config.start_from)
    except KeyError:
        raise typer.BadParameter(f"start_from {config.start_from} is not registered") from None
    finally:
        registry.close()
    if parent.provider != training.PROVIDER or parent.provider_job_id is None:
        raise typer.BadParameter(f"{parent.id} was not trained on Together, so it cannot continue")
    return parent.provider_job_id


@train_app.command("start")
def train_start(run_config: Path, yes: Confirm = False, data_dir: DataDir = Path("data")) -> None:
    """Upload the dataset and price the job, then start it with --yes."""
    config = RunConfig.from_toml(run_config)
    registry = _registry(data_dir)
    try:
        if any(m.id == config.name for m in registry.models()):
            raise typer.BadParameter(f"{config.name} is already registered. Use a new run name.")
    finally:
        registry.close()
    parent = _from_checkpoint(config, data_dir)
    client = _together()
    with _training_errors():
        state = training.prepare(config, run_config, data_dir, client, parent, time.sleep)
        if state.job_id is not None:
            typer.echo(f"{config.name} already started as job {state.job_id} ({state.status})")
            return
        typer.echo(
            f"{config.base_model}, LoRA {config.method.upper()}: {state.lines} examples, "
            f"about {state.estimate_tokens} training tokens, estimated ${state.estimate_usd:.2f} "
            f"(limit ${config.max_cost_usd:.2f})"
        )
        for reason, count in sorted(state.skipped.items()):
            typer.echo(f"skipped {count} ({reason})")
        typer.echo("The training file is uploaded to Together, which is free, to get this price.")
        meta = _snapshot_manifest(data_dir, config.dataset)
        writers = (
            (meta or {})
            .get("counts", {})
            .get("target_models" if config.method == "sft" else "chosen_models", {})
        )
        if writers:
            listed = ", ".join(f"{model} ({count})" for model, count in writers.items())
            typer.echo(f"answers to learn were written by: {listed}")
        if not yes:
            typer.echo("No job started. Add --yes to start it.")
            return
        state = training.submit(config, run_config, data_dir, client, parent)
    typer.echo(
        f"started job {state.job_id}. Check on it with `flywheel train status {config.name}`."
    )


@train_app.command("status")
def train_status(name: str, data_dir: DataDir = Path("data")) -> None:
    """Check a job, and register the model as a candidate once it completes."""
    if not (training.run_dir(data_dir, name) / "state.json").exists():
        raise typer.BadParameter(f"no training run named {name}")
    with _training_errors():
        state = training.refresh(data_dir, name, _together())
    typer.echo(f"{name}: {state.status}")
    if state.status in ("error", "cancelled"):
        typer.echo("Together refunds failed jobs. Cancelled jobs bill completed steps only.")
    if state.status != "completed":
        return
    registry = _registry(data_dir)
    try:
        if not any(m.id == name for m in registry.models()):
            config = RunConfig.from_toml(Path(state.run_config))
            registry.register(training.candidate_record(state, config))
            typer.echo(f"registered {name} as a candidate")
    finally:
        registry.close()


@train_app.command("download")
def train_download(name: str, data_dir: DataDir = Path("data")) -> None:
    """Download the merged weights of a completed job. A 9B model is about 19 GB."""
    state_path = training.run_dir(data_dir, name) / "state.json"
    if not state_path.exists():
        raise typer.BadParameter(f"no training run named {name}")
    state = training.TrainingState.load(state_path)
    if state.status != "completed" or state.job_id is None:
        raise typer.BadParameter(f"{name} is {state.status}, not completed")
    destination = training.run_dir(data_dir, name) / "merged.tar.zst"
    with _training_errors():
        size = _together().download(state.job_id, destination)
    state.weights = destination.as_posix()
    state.save(state_path)
    registry = _registry(data_dir)
    try:
        if any(m.id == name for m in registry.models()):
            registry.update(name, weights=state.weights)
    finally:
        registry.close()
    typer.echo(f"wrote {destination} ({size / 1e9:.1f} GB)")


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
    self_hosted = {m.id: m for m in settings.models if m.host == "modal"}
    if any(model in self_hosted for model, _ in work):
        typer.echo(f"each Modal model costs at most ${hosted.worst_case_usd():.2f} of GPU time")
    if not yes:
        typer.echo("Nothing was sent. Pass --yes to run this.")
        return
    routed = [(model, segment) for model, segment in work if model not in self_hosted]
    if routed:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            typer.echo("OPENROUTER_API_KEY is not set.", err=True)
            raise typer.Exit(code=1)
        efforts = {m.id: m.reasoning_effort for m in settings.models if m.reasoning_effort}
        completer = translating.OpenRouter(api_key, efforts)
        try:
            spent = translating.translate(routed, completer, path, settings.max_cost_usd, spent)
        except (translating.BudgetExceeded, translating.RequestRejected) as err:
            typer.echo(f"stopped: {err}. Finished work is saved.", err=True)
            raise typer.Exit(code=1) from err
    for spec in self_hosted.values():
        todo = [segment for model, segment in work if model == spec.id]
        if todo:
            spent += _translate_on_modal(spec, todo, path, settings.max_cost_usd - spent, data_dir)
    failed = sum(record.output is None for record in translating.read_translations(path).values())
    typer.echo(f"done, ${spent:.2f} spent in total, {failed} failed")


def _translate_on_modal(
    spec: ModelSpec, segments: list[EvalSegment], path: Path, left_usd: float, data_dir: Path
) -> float:
    """Run one self-hosted model and return what it cost."""
    worst = hosted.worst_case_usd()
    if worst > left_usd:
        typer.echo(
            f"stopped: {spec.id} could cost up to ${worst:.2f}, and ${left_usd:.2f} is left.",
            err=True,
        )
        raise typer.Exit(code=1)
    url = None
    kind, _, name = (spec.weights or "").partition(":")
    if kind == "run":
        state_path = training.run_dir(data_dir, name) / "state.json"
        if not state_path.exists():
            raise typer.BadParameter(f"no training run named {name}")
        state = training.TrainingState.load(state_path)
        if state.status != "completed" or state.job_id is None:
            raise typer.BadParameter(f"{name} is {state.status}, not completed")
        with _training_errors():
            url = _together().download_url(state.job_id)
    typer.echo(f"{spec.id}: {len(segments)} sentences on Modal")
    records = hosted.translate_on_modal(spec, segments, url)
    append_jsonl(path, records)
    return sum(record.cost_usd for record in records)


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
