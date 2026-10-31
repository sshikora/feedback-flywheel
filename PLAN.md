# Feedback Flywheel: Project Plan

**Status:** M0 and M1 built. Live serving is decided, subject to a cold-start test at M4 (see `docs/decisions/0005`).
**Last updated:** 2026-10-31
**Owner:** seth
**Related:** `~/source/ai-lang-translation/docs/plans/community-trained-model.md` (product strategy for the in-app side), `~/source/ai-lang-translation/docs/plans/compare-and-vote.md` (the vote data this project consumes)

---

## What this is

An open source pipeline that turns human feedback on model output into better models, and proves each new model is better before it ships:

```
feedback events -> curation -> datasets -> fine-tune -> evaluate -> promote -> serve -> more feedback
```

It is domain-agnostic, with translation as the first worked example. translate.moretosay.org is the reference deployment: its compare-and-vote feature already stores human preference labels, so the loop has a real source of data from day one.

## Why (the portfolio goal)

This is the public proof for an AI platform / applied AI engineering job search. It should demonstrate, in one readable repo:

1. Research-to-production engineering: data pipelines, model registry, async training jobs, promotion gates.
2. Evaluation discipline: fixed eval sets, automated metrics, LLM-as-judge, human ratings, confidence intervals, regression checks.
3. Human feedback loops: preference data feeding LoRA SFT and DPO.
4. Production integration: a fine-tuned model running as a challenger inside a live product.

Every design choice below favors "a reviewer understands it in two minutes, and can verify it in ten" over "maximum capability." Reviewers skim: the README leads with one real result table and a link to a live evaluation report, then the architecture diagram, then everything else.

## What the feedback signal can and cannot do

Stated up front because it shapes the base-model choice and the write-up:

- Free comparisons run Gemini Flash, DeepSeek V4.1 Flash and Gemma 4 31B on inputs under 300 characters. A vote **selects among their outputs**. It adds no knowledge those models lack. Training on voted outputs is distillation from the compare set, steered by human preference. Synthetic preference pairs ranked by metric or judge have the same ceiling.
- Therefore the fine-tuned base model must start **measurably below the compare-set models** on the chosen language pair, or there is no headroom and the promotion gate can never pass. The M2 baseline ranks candidate base models by headroom, not by raw quality.
- Votes are cast with model names visible, so they carry brand bias and "picked my usual model" bias. They are training signal only, never evaluation signal.
- Real vote volume will be small for months (votes started 2026-10-26). The pipeline has to be useful before the data is.

The write-up says all of this plainly. The thing being demonstrated is the machinery and the evaluation discipline, not a claim that a few hundred votes beat frontier models.

## Relationship to ai-lang-translation

- **This repo** is public (Apache 2.0) and owns the pipeline: schemas, curation, dataset builders, eval harness, training orchestration, registry, promotion logic.
- **ai-lang-translation** stays private. It exports feedback (from the comparisons table) and consumes promoted models as one more provider, the same way `providers/openrouter.ts` works.
- No user data ever enters this repo. Example data is public corpora or synthetic.
- This project is Layer 0 to Layer 2 of `community-trained-model.md`, built as reusable infrastructure. That plan's gate still applies to users: a fine-tuned model reaches the default path only after it wins. Before then it runs offline or as an opt-in compare card.
- **Consent is already covered.** The app's vote line reads "Your vote and the text you compared are saved anonymously to improve translations", and the privacy section on `/about#data-retention` says comparisons and votes are kept to improve translation quality and publish aggregate results. Training is within that scope. Publishing raw text is not, except for comparisons the user shared.

---

## Language choice

**Python for the pipeline.** Metrics (sacrebleu, COMET), training SDKs, dataset tooling and judges are all Python-native, and Python is what AI platform hiring expects to see. One language means one package manager, one CI setup, one test suite.

**TypeScript client later.** The pipeline is designed so a typed TypeScript client can be generated rather than hand-written, which makes it easy to integrate into other projects (ai-lang-translation first):

- Event and registry schemas are pydantic models. JSON Schema is exported from them in CI, and TypeScript types are generated from that JSON Schema. The TypeScript side can never drift from the Python side.
- The pipeline exposes a small local HTTP service (FastAPI) for the operations another app needs at runtime: submit feedback events, query the registry, read promotion status. Its OpenAPI spec drives a generated TypeScript client (`packages/client-ts`).
- Batch operations (ingest, curate, build, train, eval, promote) stay CLI-only. Other projects do not need them at runtime.

In v1 ai-lang-translation needs none of this: it exports comparisons as a file for ingest, and it calls the promoted model through an OpenAI-compatible endpoint. The TypeScript client is for the later step where the app pushes events directly and reads challenger status.

---

## Architecture

### 1. Canonical event schema

One record type for all feedback, independent of the source app, defined once as a pydantic model with JSON Schema exported:

```python
class FeedbackEvent:
    id: str
    source: str                      # "ai-lang-translation", "jsonl", ...
    origin: Literal["human", "synthetic"]
    created_at: datetime
    task: Literal["translation"]
    input: TranslationInput          # text, src_lang (as requested), detected_src_lang, tgt_lang, tone
    candidates: list[Candidate]      # label, model_id, output | error
    signal: PreferenceSignal         # kind="preference", chosen: list[model_id]  (an agreement group counts for all its models)
    context: EventContext            # blind: bool, your_model: model_id | None, model_set: "free" | "pro", ui_version
    consent: Consent                 # training: bool, publish: bool
```

- `signal` is a discriminated union with one member in v1. `correction` and `implicit` kinds are deliberately left out until the app produces them.
- `context.your_model` and `context.model_set` are carried because the compare spec promises a split into "picked their own model" and "picked another", and because the free and Pro model sets are different output distributions.
- `consent.publish` is derived from the comparison's `shared` attribute. Only shared comparisons have public text.
- Candidates with an error are kept in the record (they explain missing rejected pairs) but never enter a dataset.

### 2. Ingest adapters

- `jsonl`: generic file import.
- `comparisons-export`: reads an export of the ai-lang-translation comparisons table and maps votes to preference events. Only voted comparisons become events, because shared but unvoted ones carry no signal.
- `flores`: downloads FLORES-200 at runtime and stores reference pairs as eval segments, a separate record type from feedback events. The data is CC-BY-SA, so it is never committed to this repo.

### 3. Curation

- PII and secret scrubbing at ingest, before anything is stored. Secrets drop the event, and personal data is redacted in the input and every candidate.
- Dedupe by normalized input text, language pair and tone. Repeat votes are counted, not discarded: each curated event records its independent support and whether voters disagreed. Votes inside a short window count once, since votes carry no user id.
- Quality scoring: agreement weighting (several models giving the same answer that also wins the vote is stronger signal), outlier and spam heuristics.
- **Bias handling:** record `blind: false`, report vote share by model and the own-model pick rate, and never treat raw votes as ground truth for evaluation. Evaluation uses blind methods only (section 5).
- Train/eval split with leakage protection: nothing in an eval set, or a near-duplicate of it, can enter training. FLORES dev and devtest are both excluded from training, since they share a domain. Matching covers inputs and all candidate outputs against both eval sides, at a threshold measured on real data (`docs/decisions/0006`).

### 4. Datasets and versioning

- Builders: SFT (`input -> chosen output`) and DPO (`input, chosen, rejected`). Rejected candidates come from the same comparison, so pairs are on-policy for the compare set. Many rejected outputs are also correct translations with a different phrasing. That is a weak signal and the reports say what share of pairs differ only stylistically.
- Every built dataset is an immutable, content-hashed snapshot with a manifest (source events, filters, counts, human versus synthetic share, hash). Training runs reference a dataset hash, so any model traces back to exactly what it learned from.
- A **data card** per dataset snapshot, generated from the manifest.
- Storage: local filesystem in v1, S3 in the AWS deployment.

### 5. Evaluation harness

Cebuano is a low-resource language, and that breaks two of the three usual metrics. The harness is designed around that.

- **Eval sets:** FLORES-200 devtest for the chosen language pair (fixed benchmark, about 1,000 sentences), a held-out set of community-preferred pairs once there are enough, and a **human-rated set** of about 100 sentences scored by seth and one native speaker on a simple adequacy and fluency scale.
- **Primary metric: chrF++** via sacrebleu. It is language-agnostic and reference-based, so it is valid for Cebuano.
- **COMET is reported, not trusted.** COMET's backbone (XLM-R) does not cover Cebuano, so its scores are unvalidated for this pair. Reports label it as such. If MetricX or a Cebuano-validated metric appears, swap it in.
- **LLM-as-judge:** pairwise, blind to model names, run in both orders to cancel position bias, reported as win rate. The judge must be a model from a **different family than any candidate**, to avoid self-preference bias. The write-up notes that frontier models are only moderately competent in Cebuano, which is why the human-rated set exists as the tie-breaker.
- **Statistics:** bootstrap confidence intervals on every score. A difference inside the interval is reported as a tie.
- **Reports:** one static HTML page per run (scores, intervals, per-sentence diffs, worst regressions, human versus synthetic share of training data). These double as portfolio artifacts.

### 6. Training orchestration

- One provider interface: `submit(job) -> job_id`, `poll(job_id)`, `fetch_model(job_id)`.
- **v1 provider: Together AI** (hosted LoRA SFT and DPO). A second adapter (Tinker, or local MLX for zero-cost smoke tests) is a stretch item, not v1. The interface is there so adding one is a small change.
- Every run is driven by a **checked-in config file** (base model, method, dataset hash, hyperparameters, seed). The config is what the registry links to, so any run is reproducible from the repo.
- Jobs run asynchronously: submit, persist state, poll, record result. In the AWS deployment this becomes a Step Functions workflow with Lambda tasks.

### 7. Model registry

Each model gets a record: id, base model, method (SFT/DPO), dataset hash, run config, eval report link, cost, and status (`candidate`, `challenger`, `champion`, `retired`). A **model card** is generated per registered model. SQLite locally, DynamoDB on AWS.

### 8. Promotion

Two separate questions, answered separately:

- **Gate: better than its own base model.** This proves the pipeline works. Candidate becomes challenger when it beats its base on chrF++ beyond the confidence interval, has a judge win rate against the base of at least 55%, and shows no regression on a guard set. Thresholds are **pre-registered** in the run config before training starts.
- **Report: where it stands against the production models.** The challenger report also scores it against the app's compare-set models. This is positioning, not a gate, and the write-up presents it honestly, including when the challenger loses.
- **Online stage** (challenger to champion): the model appears as one card in the app's compare view behind a flag, on English to Cebuano comparisons only, for a fixed 30-day window. Votes on it are tracked separately. Promotion needs a minimum vote count and win rate, both set before the test starts. At current traffic the window may close without reaching that count. That is reported as inconclusive, and the plan does not depend on it.
- Every promotion or rejection is written to the registry with its evidence.

### 9. Serving

- **Together has discontinued serverless LoRA inference**, found during M0. Adapters there run only on dedicated endpoints, and an always-on endpoint would cost about $2,900 a month, so that is ruled out. Full analysis is in `docs/decisions/0005`.
- **Host:** a scale-to-zero GPU on Modal or RunPod, billed by the second. Training stays on Together. The adapter is exported and merged into the base model for self-hosting.
- **Offline evaluation** uses the same host. A short-lived Together dedicated endpoint is the fallback if export or self-hosting is blocked.
- **Live serving limits (M5):**
  - The challenger runs only on English to Cebuano comparisons.
  - The app enforces a daily cap, starting at 50 requests, then carries on with the other models. This sets a firm spending limit that does not depend on any host feature.
  - The app wakes the model when a user selects Cebuano as the target language. At low traffic nearly every request finds the model asleep, and the app gives each model 12 seconds, so without this the challenger would time out and collect few votes.
  - The live test runs for a fixed 30 days, then the results are published and the model is switched off.
- **Cold-start test (M4):** the live plan goes ahead if a pre-warmed model reliably answers within 12 seconds.
- **Fallback, at no ongoing cost:** skip live hosting. Run the challenger offline on real voted inputs, have the Cebuano rater score the results blind, and record the demo through Ollama. The "ran it live" claim comes off the resume line.
- ai-lang-translation adds an OpenAI-compatible provider pointed at whichever host is chosen, mirroring `openrouter.ts`.
- Local demo path: export the adapter and run it through Ollama for recordings, at zero cost.

---

## Data bootstrap

1. **Baseline first.** Evaluating the app's existing models and three candidate base models on FLORES needs no user data and is a publishable result on its own.
2. **SFT seed from public parallel data**, licenses checked per corpus. **Avoid anything derived from Cebuano Wikipedia**: it is overwhelmingly bot-generated and is a known trap for Cebuano corpora. Prefer OPUS sources with human origin (Tatoeba, bible-uedin, QED, GlobalVoices where available).
3. **Synthetic preference pairs:** run several models on source sentences, rank outputs with chrF++ against references and the blind judge, build DPO pairs from the ranking. Labeled synthetic everywhere.
4. **Real votes** join as they accumulate. Every dataset manifest and report shows the human versus synthetic share.

---

## Tech stack

- **Python 3.12, uv, pydantic, typer (CLI), FastAPI (local service), SQLite.**
- **Metrics and judge:** sacrebleu, COMET (reported only), judge via the Anthropic or OpenAI SDK depending on the candidate families in play.
- **Generated TypeScript client** (later): JSON Schema from pydantic, OpenAPI from FastAPI, generated types and client in `packages/client-ts`.
- **Storage:** filesystem + SQLite (v1), S3 + DynamoDB (AWS).
- **Infra (stretch):** OpenTofu for S3, DynamoDB, Lambda, Step Functions.
- **CI:** GitHub Actions running unit tests, schema export check, and a tiny end-to-end pipeline on fixture data.

## Repo layout

```
feedback-flywheel/
  flywheel/
    schema/      pydantic models, JSON Schema export
    ingest/      jsonl, comparisons-export, flores adapters
    curate/      dedupe, scrubbing, scoring, leakage protection
    datasets/    SFT and DPO builders, manifests, data cards
    evals/       chrF++, COMET, pairwise judge, human-rating import, bootstrap CIs, HTML reports
    train/       provider interface, Together adapter, run configs
    registry/    SQLite registry, model cards
    promote/     gate rules, challenger reports
    serve/       OpenAI-compatible client for the hosted adapter
    scrub.py     PII and secret removal, run at ingest
    api/         FastAPI service (events, registry, status)
    cli.py       flywheel ingest | curate | build | train | eval | promote | report | serve
  configs/       one file per training run and per eval set
  schemas/       generated JSON Schema, checked in CI
  scripts/       reproducible measurements behind decisions
  packages/
    client-ts/   generated TypeScript types and client (later)
  infra/aws/     OpenTofu (stretch)
  docs/
    decisions/   one short file per decision (ADR style)
    architecture.md, write-ups
  tests/
```

---

## Milestones

Part-time pace. Roughly 10 to 12 weeks for M0 to M6, with a **minimum showable repo at the end of M4** (about week 7). Reviewer value ranks: eval report, then training plus gate, then live integration, then AWS. Cuts happen in reverse order.

| # | Milestone | Done when |
|---|---|---|
| M0 | Decisions and skeleton | Open decisions below answered and logged in `docs/decisions/`; verification items checked (LoRA serving, FLORES license); uv project, CI, license, README stub pushed to a public GitHub repo. **Built.** Other corpus licenses are checked when each corpus is added at M3. |
| M1 | Schema, ingest, curation | `flywheel ingest` loads FLORES and a comparisons export into curated events; JSON Schema exported in CI; unit tests cover dedupe, PII scrubbing, leakage protection. **Built.** |
| M2 | Eval harness and baseline | Baseline report on the chosen pair's FLORES devtest for the app's compare-set models and three candidate base models, with chrF++, judge win rates and confidence intervals, plus the first human-rated set. Base model chosen by headroom, from those that fit the cheapest GPU tier, can be exported, and are licensed for self-hosting. Published as the first write-up. |
| M3 | Training and registry | A LoRA SFT model trained through Together from a versioned dataset and a checked-in run config, registered with dataset hash, cost and a model card. Then a DPO run on preference pairs. |
| M4 | Promotion gates | `flywheel promote` runs the pre-registered gate against the base model and produces a challenger report, pass or fail, with evidence, plus the positioning report against production models. Cold-start test run and serving host chosen. **Minimum showable repo.** README leads with the result table and report link. |
| M5 | Live integration | Challenger added to the app as a new `AIModel` (union, selector, dispatch, provider), in the free compare set for English to Cebuano only, behind a flag, with the daily cap and pre-warm in place. It runs for 30 days and its votes flow back through ingest. The loop is closed. If the M4 test fails, this milestone becomes the offline fallback in section 9. |
| M6 | Polish and launch | Architecture diagram, README walkthrough, short demo video, decision log complete, write-up with a "what I would do differently" section, LinkedIn post. Resume updated. |
| M7 | Stretch | AWS deployment with OpenTofu; second training provider; generated TypeScript client and the app pushing events directly. |

## Cost estimate

- Development: about $20 to $75 a month (hosted fine-tuning per token, eval API calls, per-token serving).
- One-time experimentation budget: $100 to $200.
- No always-on GPU. COMET runs on the MacBook.
- Offline evals run on the same small GPU as live serving, at about $0.59 an hour. A run may take about an hour, which has not been measured yet.
- Live test at M5: about $1.50 a month at 5 Cebuano comparisons a day, and about $15 at most at the 50-a-day cap. Free host credits may cover this. These figures assume each request keeps the GPU awake for about 60 seconds, which M4 will measure.
- Re-check provider prices at M0; GPU and fine-tuning prices have moved during 2026.

---

## Risks

| Risk | Mitigation |
|---|---|
| No headroom: base model already matches the compare set | M2 ranks base candidates by headroom before any training. Choose the strongest base that is still clearly below the teachers on Cebuano. |
| Metric validity on Cebuano | chrF++ is the gate metric; COMET is reported as unvalidated; human-rated set is the tie-breaker. |
| Judge is not much better than the candidates | Judge from a different family, run in both orders, and the human set decides close calls. |
| Too little real feedback data | Baseline and synthetic pairs carry M2 to M4; every manifest and report states human versus synthetic share. |
| Vote bias (names visible) | Votes used for training signal only, never for evaluation. Own-model pick rate reported. |
| Live serving cost | Serverless LoRA is gone, so the challenger runs on a scale-to-zero GPU. English to Cebuano only, a daily cap enforced in the app and a 30-day window keep the cost bounded. |
| Challenger times out on cold starts and collects few votes | The app pre-warms the model when Cebuano is selected. The M4 test confirms this works before anything goes live, and the offline fallback costs nothing. |
| Adapter cannot be exported or self-hosted | Confirm export and the base model's license before choosing the base at M2. Fall back to a short-lived Together endpoint for evals. |
| Corpus licenses and quality | Check each corpus. FLORES-200 is CC-BY-SA and is eval-only. NLLB-200 weights are CC-BY-NC, so no NLLB in anything commercial. No Cebuano Wikipedia derivatives. |
| Timeline | Minimum showable repo at M4. M5 to M7 are additive; each can be cut without hollowing out what is already published. |
| Fine-tuned model is worse | That is a valid result. The gate rejecting it, with evidence, is a feature to show, not a failure to hide. |

---

## Open decisions

1. **Repo name.** Working name `feedback-flywheel`.
2. **First language pair.** English to Cebuano. Tier C in the community model plan, in FLORES-200, and sanity-checkable in Cebu. The metric caveats above are accepted as part of the story.
3. **Base model.** Chosen at M2 by headroom from Together's LoRA fine-tuning list. Candidates include Qwen3.5 4B and 9B, Gemma 4 26B A4B and Llama 3.1 8B. Gemma 4 31B IT is on that list too and is also a teacher in the app's free compare set, but a teacher has no headroom over itself, so a smaller Gemma is the cleaner distillation story. A small base is preferred because it must fit the cheapest GPU tier. Confirm that the adapter can be exported and that the license allows self-hosting.
4. **License.** Decided: Apache 2.0. It covers the code only; datasets keep their own licenses.
5. **Native speaker for the human-rated set.** One person, about 100 sentences, scored twice a few weeks apart for consistency.
6. **Live serving host.** Modal or RunPod, chosen at M4 from the cold-start test. See `docs/decisions/0005`.

## Resume line when shipped

> Built and open-sourced a human-feedback model improvement pipeline in Python (preference curation, LoRA SFT and DPO, blind evaluation with bootstrap confidence intervals, pre-registered promotion gates) and ran it live behind translate.moretosay.org, where a fine-tuned challenger competes in the product's compare view. [N] human votes, [X]-point chrF++ gain over base on FLORES-200 Cebuano, 95% CI [a, b].

Fill in the bracketed numbers from the M4 report. If the gain is zero or negative, the line says so and leads with the gate instead. If the offline fallback is used, replace the live claim with the blind human evaluation.
