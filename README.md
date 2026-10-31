# Feedback Flywheel

An open source pipeline that turns human feedback on model output into better models, and proves each new model is better before it ships.

```
feedback events -> curation -> datasets -> fine-tune -> evaluate -> promote -> serve -> more feedback
```

Translation is the first worked example, using English to Cebuano. [translate.moretosay.org](https://translate.moretosay.org) is the reference deployment: its compare-and-vote feature produces real human preference labels.

## Status

Early. There are no model results yet, and this section will lead with them once there are.

| Stage | State |
|---|---|
| Event schema, JSON Schema export | Done |
| Ingest: JSONL, comparisons export, FLORES-200 | Done |
| Curation: PII and secret scrubbing, dedupe, quality scoring, leakage protection | Done |
| Evaluation harness and baseline | Next |
| Training, registry, promotion gates | Planned |
| Live integration | Planned |

The full design is in [PLAN.md](PLAN.md), and the reasoning behind major choices is in [docs/decisions](docs/decisions).

## Try it

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run flywheel ingest flores --src en --tgt ceb --split devtest
uv run flywheel ingest comparisons tests/fixtures/comparisons_scan.json
uv run flywheel curate
```

Output lands in `data/`, which is never committed. `data/curated/report.json` has the counts behind each decision: what was dropped and why, the human versus synthetic share, and vote statistics by model.

## How curation protects the evaluation

- **Votes are not ground truth.** People voted with model names visible, so votes carry brand bias. They are training signal only, and evaluation uses blind methods.
- **Nothing in an eval set reaches training.** Any event whose input or candidate output matches eval text, exactly or as a near-duplicate, is removed from training.
- **Repeat votes are counted, not discarded.** Votes on the same input collapse into one event that records how many independent voters agreed, and whether they disagreed.
- **Nothing sensitive is stored.** Scrubbing runs at ingest. Secrets drop the event, and personal data is redacted everywhere it appears.

## Development

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run flywheel schema --check
```

## License

Code is [Apache 2.0](LICENSE). Datasets keep their own licenses: FLORES-200 is CC-BY-SA 4.0, so it is downloaded at runtime and never committed.
