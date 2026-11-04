# Feedback Flywheel

An open source pipeline that turns human feedback on model output into better models, and proves each new model is better before it ships.

```
feedback events -> curation -> datasets -> fine-tune -> evaluate -> promote -> serve -> more feedback
```

Translation is the first worked example, using English to Cebuano. [translate.moretosay.org](https://translate.moretosay.org) is the reference deployment: its compare-and-vote feature produces real human preference labels.

## Status

The baseline is measured, and nothing is trained yet. Six models translated all 1,012 English sentences of the FLORES-200 devtest set into Cebuano:

| Model | chrF++, 95% CI | Failed |
|---|---|---|
| Gemini 3.8 Flash | 61.70 [61.08, 62.33] | 1 |
| DeepSeek V4.1 Flash | 60.24 [59.59, 60.87] | 0 |
| Gemma 4 31B | 59.72 [59.07, 60.36] | 0 |
| Gemma 4 26B A4B | 59.54 [58.90, 60.20] | 0 |
| Qwen3.5 9B | 45.74 [45.18, 46.36] | 0 |
| Llama 3.1 8B | 37.15 [36.49, 37.84] | 18 |

The first three make up the translation app's free compare set. The other three are candidates to fine-tune, and the goal is to close the gap. A blind judge comparing each candidate with the app's models finds the same gaps. Qwen3.5 9B and Llama 3.1 8B lose nearly every comparison, while Gemma 4 26B A4B is close to even with all but Gemini. A native speaker is scoring a blind sample now, and those scores settle the base model.

| Stage | State |
|---|---|
| Event schema, JSON Schema export | Done |
| Ingest: JSONL, comparisons export, FLORES-200 | Done |
| Curation: PII and secret scrubbing, dedupe, quality scoring, leakage protection | Done |
| Evaluation harness: chrF++, blind judge, human ratings, HTML report | Done |
| Baseline: machine scores | Done |
| Baseline: first native-speaker round | In progress |
| Base model checks: export, license, GPU fit | Done, see [0011](docs/decisions/0011-base-model-checks.md) |
| Datasets: SFT and DPO builders, hashed snapshots, data cards | Done |
| Registry: run configs, status history, model cards | Done |
| Training, promotion gates | Planned |
| Live integration | Planned |

The full design is in [PLAN.md](PLAN.md), and the reasoning behind major choices is in [docs/decisions](docs/decisions).

## Try it

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run flywheel ingest flores --src en --tgt ceb --split devtest
uv run flywheel ingest comparisons tests/fixtures/comparisons_scan.json --training-from 2026-10-27
uv run flywheel curate
uv run flywheel build sft    # and `build dpo`
```

Votes may train only if they were cast after `--training-from`, the date the app's privacy page began naming the training provider. Without it, none may. Output lands in `data/`, which is never committed. `data/curated/report.json` has the counts behind each decision: what was dropped and why, the human versus synthetic share, and vote statistics by model. Each build writes a snapshot to `data/datasets/`, named by the hash of its examples, with a manifest and a data card.

### Evaluate models

These commands show their plan and spend nothing until you add `--yes`. Translation needs `OPENROUTER_API_KEY` and the judge needs `ANTHROPIC_API_KEY`.

```sh
uv run flywheel eval translate --limit 3      # a small test first, then without --limit
uv run flywheel eval judge
uv run flywheel report                        # free, and can be rerun at any time
```

The report is written to `data/runs/`. For human scoring, `flywheel rate export --round m2-a` writes a blind sheet and a separate key, and `flywheel rate import` loads the scores. Raters follow [docs/rating-guide.md](docs/rating-guide.md).

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
