"""The `flywheel` command line. Each stage reads and writes files under --data-dir."""

from pathlib import Path
from typing import Annotated

import typer

from flywheel.curate.config import CurationConfig
from flywheel.curate.pipeline import curate as run_curation
from flywheel.ingest import comparisons, flores, jsonl
from flywheel.schema import EvalSegment, FeedbackEvent
from flywheel.schema.export import stale_schemas, write_schemas
from flywheel.scrub import scrub_events
from flywheel.storage import read_jsonl, write_json, write_jsonl

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
ingest_app = typer.Typer(
    no_args_is_help=True, help="Load a source into scrubbed, canonical records."
)
app.add_typer(ingest_app, name="ingest")

DataDir = Annotated[Path, typer.Option(help="Where pipeline data lives. Never committed.")]


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
