# Feedback Flywheel: Project Plan

**Status:** M0 and M1 built. The M2 harness is built, and the baseline run is next. Live serving is decided, subject to a cold-start test at M4 (see `docs/decisions/0005`). A native-speaker rater app is added at M4 (see `docs/decisions/0007`). The loop keeps running after launch (see `docs/decisions/0008`).
**Last updated:** 2026-11-01
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
3. Human feedback loops: preference data and native-speaker corrections feeding LoRA SFT and DPO.
4. Production integration: a fine-tuned model running as a challenger inside a live product.

Every design choice below favors "a reviewer understands it in two minutes, and can verify it in ten" over "maximum capability." Reviewers skim: the README leads with one real result table and a link to a live evaluation report, then the architecture diagram, then everything else.

## What the feedback signal can and cannot do

Stated up front because it shapes the base-model choice and the write-up:

- Free comparisons run Gemini Flash, DeepSeek V4.1 Flash and Gemma 4 31B on inputs under 300 characters. A vote **selects among their outputs**. It adds no knowledge those models lack. Training on voted outputs is distillation from the compare set, steered by human preference. Synthetic preference pairs ranked by metric or judge have the same ceiling.
- Therefore the fine-tuned base model must start **measurably below the compare-set models** on the chosen language pair, or there is no headroom and the promotion gate can never pass. The M2 baseline ranks candidate base models by headroom, not by raw quality.
- **Native-speaker corrections are the exception.** A rater who fixes a translation adds text no model wrote, so corrections are the one signal that can go beyond this ceiling. They are slow to collect, so they extend the plan and do not replace it.
- Votes are cast with model names visible, so they carry brand bias and "picked my usual model" bias. They are training signal only, never evaluation signal.
- Real vote volume will be small for months (votes started 2026-10-26). The pipeline has to be useful before the data is.

The write-up says all of this plainly. The thing being demonstrated is the machinery and the evaluation discipline, not a claim that a few hundred votes beat frontier models.

## Relationship to ai-lang-translation

- **This repo** is public (Apache 2.0) and owns the pipeline: schemas, curation, dataset builders, eval harness, training orchestration, registry, promotion logic.
- **ai-lang-translation** stays private. It exports feedback (from the comparisons table) and consumes promoted models as one more provider, the same way `providers/openrouter.ts` works.
- No user data ever enters this repo. The same goes for ratings, corrections and rater ids. Example data is public corpora or synthetic.
- This project is Layer 0 to Layer 2 of `community-trained-model.md`, built as reusable infrastructure. That plan's gate still applies to users: a fine-tuned model reaches the default path only after it wins. Before then it runs offline or as an opt-in compare card.
- **Consent is partly covered.** The app's vote line reads "Your vote and the text you compared are saved anonymously to improve translations", and the privacy section on `/about#data-retention` says comparisons and votes are kept to improve translation quality and publish aggregate results. Training is within that purpose, but the page names no training provider, so real votes reach one only after it does, and then only votes saved after that change. Raters never see user text. Publishing raw text is out of scope, except for comparisons the user shared. See `docs/decisions/0009`.

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
    context: EventContext            # blind: bool, your_model: model_id | None, model_set: "free" | "pro", ui_version, rater_id: str | None
    consent: Consent                 # training: bool, publish: bool
```

- `signal` is a discriminated union with one member in v1. The rater app adds a `correction` kind at M4, holding the rater's corrected text. `implicit` stays out until the app produces it.
- `context.rater_id` is a pseudonymous id, set only on rater app events. Votes still carry no user id.
- Evaluation scores are a separate `HumanRating` record (segment, model, rater, round, adequacy, fluency), like eval segments. They are not feedback events and never enter training.
- `context.your_model` and `context.model_set` are carried because the compare spec promises a split into "picked their own model" and "picked another", and because the free and Pro model sets are different output distributions.
- `consent.publish` is derived from the comparison's `shared` attribute. Only shared comparisons have public text.
- `consent.training` is false for votes saved before the privacy page names a training provider. The comparisons adapter currently sets it to true for every vote, which must change before the first upload at M3. Curation already drops events without it.
- Candidates with an error are kept in the record (they explain missing rejected pairs) but never enter a dataset.

### 2. Ingest adapters

- `jsonl`: generic file import.
- `comparisons-export`: reads an export of the ai-lang-translation comparisons table and maps votes to preference events. Only voted comparisons become events, because shared but unvoted ones carry no signal.
- `rater`: reads an export from the rater app. Picks become preference events and corrections become correction events, both with `blind: true`, and they go through curation like any other feedback.
- `flores`: downloads FLORES-200 at runtime and stores reference pairs as eval segments, a separate record type from feedback events. The data is CC-BY-SA, so it is never committed to this repo.

### 3. Curation

- PII and secret scrubbing at ingest, before anything is stored. Secrets drop the event, and personal data is redacted in the input and every candidate.
- Dedupe by normalized input text, language pair and tone. Repeat votes are counted, not discarded: each curated event records its independent support and whether voters disagreed. Votes inside a short window count once, since votes carry no user id.
- Quality scoring: agreement weighting (several models giving the same answer that also wins the vote is stronger signal), outlier and spam heuristics.
- **Bias handling:** record `blind: false`, report vote share by model and the own-model pick rate, and never treat raw votes as ground truth for evaluation. Evaluation uses blind methods only (section 5).
- Train/eval split with leakage protection: nothing in an eval set, or a near-duplicate of it, can enter training. FLORES dev and devtest are both excluded from training, since they share a domain. Matching covers inputs and all candidate outputs against both eval sides, at a threshold measured on real data (`docs/decisions/0006`).

### 4. Datasets and versioning

- Builders: SFT (`input -> chosen output`) and DPO (`input, chosen, rejected`). Rejected candidates come from the same comparison, so pairs are on-policy for the compare set. A correction gives an SFT example, and a DPO pair with the corrected text as chosen and the model's output as rejected. Many rejected outputs are also correct translations with a different phrasing. That is a weak signal and the reports say what share of pairs differ only stylistically.
- Every built dataset is an immutable, content-hashed snapshot with a manifest (source events, filters, counts, human versus synthetic share, hash). Training runs reference a dataset hash, so any model traces back to exactly what it learned from.
- A **data card** per dataset snapshot, generated from the manifest.
- Storage: local filesystem in v1, S3 in the AWS deployment.

### 5. Evaluation harness

Cebuano is a low-resource language, and that breaks two of the three usual metrics. The harness is designed around that.

- **Eval sets:** FLORES-200 devtest for the chosen language pair (fixed benchmark, about 1,000 sentences), a held-out set of community-preferred pairs once there are enough, and a **human-rated set** of about 100 FLORES devtest sentences, selected by a fixed seed and scored blind by native speakers for adequacy and fluency (section 10).
- **Primary metric: chrF++** via sacrebleu. It is language-agnostic and reference-based, so it is valid for Cebuano.
- **COMET is reported, not trusted.** COMET's backbone (XLM-R) does not cover Cebuano, so its scores are unvalidated for this pair. Reports will label it as such. It is deferred past M2, because the baseline does not depend on it. If MetricX or a Cebuano-validated metric appears, swap it in.
- **LLM-as-judge:** pairwise, blind to model names, run in both orders to cancel position bias, reported as win rate. It sees no reference, and each base meets each compare model on 200 sentences (`docs/decisions/0010`). The judge must be a model from a **different family than any candidate**, to avoid self-preference bias. The write-up notes that frontier models are only moderately competent in Cebuano, which is why the human-rated set can veto a model at the gate.
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

- **Gate: better than what it replaces.** The first version is compared with its own base model, which proves the pipeline works. Later versions are compared with the current live version, which proves the loop is still improving. A candidate becomes challenger when it beats that version on chrF++ beyond the confidence interval, has a judge win rate against it of at least 55%, shows no regression on a guard set, and passes a blind human check. Thresholds are **pre-registered** in the run config before training starts.
- **Human check.** Candidates that pass the automated checks are scored by raters on the human-rated set, mixed with the version they must beat and blind to which is which. Neither adequacy nor fluency may fall below that version beyond the confidence interval. With about 100 sentences this is a veto, not proof of improvement. A candidate that misses the check, or has no round because no rater is available, stays a candidate and does not reach the app.
- **Report: where it stands against the production models.** The challenger report also scores it against the app's compare-set models. This is positioning, not a gate, and the write-up presents it honestly, including when the challenger loses.
- **Online stage** (challenger to champion): the model appears as one card in the app's compare view behind a flag, named AI Translate Cebuano, on free English to Cebuano comparisons only. It stays live until a better version replaces it or it is switched off. Votes are tracked per version, and first results are published after 30 days. Champion status needs a minimum vote count and win rate, both set before the model goes live. At current traffic that count may take a long time to reach. Until then the result is reported as inconclusive, and replacing a version depends on the gate, not on votes.
- **A failed version is held back, not final.** It stays offline, users keep the current version, and the next attempt follows section 11.
- Every promotion or rejection is written to the registry with its evidence.

### 9. Serving

- **Together has discontinued serverless LoRA inference**, found during M0. Adapters there run only on dedicated endpoints, and an always-on endpoint would cost about $2,900 a month, so that is ruled out. Full analysis is in `docs/decisions/0005`.
- **Host:** a scale-to-zero GPU on Modal or RunPod, billed by the second. Training stays on Together. The adapter is exported and merged into the base model for self-hosting.
- **Offline evaluation** uses the same host. A short-lived Together dedicated endpoint is the fallback if export or self-hosting is blocked.
- **Live serving limits (M5):**
  - The challenger runs only on free English to Cebuano comparisons. It stays off the app's `ALL_MODELS` list, because Pro comparisons accept anything on that list and could then select it for any language.
  - The app enforces a daily cap, starting at 50 requests, then carries on with the other models. This sets a firm spending limit that does not depend on any host feature.
  - The app wakes the model when a user selects Cebuano as the target language. At low traffic nearly every request finds the model asleep. The app gives each model 12 seconds and returns all answers together, so without this a sleeping challenger would stall the whole comparison, then fail and collect no vote.
  - The model stays live under these limits, and the daily cap bounds serving costs for as long as it runs. First results are published after 30 days. The previous version's weights are kept, so the flag can switch back or off at any time.
  - The about page names the serving host before the first live request, and the host must meet the page's promise not to store content or use it for training.
- **Cold-start test (M4):** the live plan goes ahead if a pre-warmed model reliably answers within 12 seconds.
- **Fallback, at no ongoing cost:** skip live hosting. Run each passing version offline on public sentences, have raters score the results blind, and record the demo through Ollama. The loop continues offline. The "ran it live" claim comes off the resume line.
- ai-lang-translation adds an OpenAI-compatible provider pointed at whichever host is chosen, mirroring `openrouter.ts`.
- Local demo path: export the adapter and run it through Ollama for recordings, at zero cost.

### 10. Rater app

A small web app for native speakers to score and correct model output, in `flywheel/rater/`. Full reasoning is in `docs/decisions/0007`. The first version covers English to Cebuano, with the language pair set in config, and any number of raters.

- **Spreadsheet first.** The M2 round uses a blind sheet from `flywheel rate export`, scored in Google Sheets and read back by `flywheel rate import`. A separate key file, which raters never see, records which model wrote each row. The app and the sheet share one rating format, so the app can be cut without touching evaluation or the gate.
- **Rater flow.** A rater opens a private invite link, accepts terms covering training use and aggregate publication, reads the scale's written anchors, and works through one task at a time. Progress is saved after each task. Raters never see model names.
- **Evaluation pool: scored only.** The frozen human-rated set, with outputs shuffled across models, scored 1 to 5 for adequacy and 1 to 5 for fluency. There is no correction box here, because this text must never reach training.
- **Training pool: picks and corrections.** Sentences from the public corpora cleared at M3, never FLORES. A rater picks the better of two outputs, or corrects the translation. The existing leakage check still applies.
- **Quality checks.** The queue hides known-good and deliberately wrong translations. A rater who misses them is excluded from that round. At least one rater scores the evaluation set twice, a few weeks apart, and reports show agreement within and between raters. The exclusion rule and minimum agreement are set before each round starts.
- **Scoring.** A sentence's score is the mean across raters, and a model's score is the mean across sentences, with bootstrap intervals over sentences. Every rater scores the same outputs, so a harsh rater affects all models equally. Per-rater means are still reported.
- **Service boundary.** The rater app is a separate FastAPI app from `flywheel/api/`, with only task and submit routes. The event and registry routes are never exposed publicly.
- **Privacy.** Raters get a pseudonymous `rater_id` and a random invite token, stored hashed. The app stores no names or email addresses. Its database lives outside the repo. Raters see public text only (`docs/decisions/0009`).
- **Hosting.** The cheapest small host that can hold a SQLite file, running only during rating rounds. Results are exported to local files when a round closes.
- **Other languages.** A second pair needs config, a rater, a FLORES set and its own license and metric checks, but no new app code. It waits until Cebuano has worked end to end.

### 11. Continuous improvement

The loop keeps running after M5. Full reasoning is in `docs/decisions/0008`.

- **Trigger.** A training run starts when enough new human examples have cleared curation, at a count set in config, within a monthly training budget. A calendar would pay for runs with nothing new to learn from. Runs are started by hand until the app pushes events directly at M7.
- **Comparison.** Each version must beat the current live version through the same gate, with thresholds pre-registered for that run. Only candidates that pass the automated checks reach raters.
- **A failed version.** It stays offline, users keep the current version, and it is logged. The next attempt needs new data, or a config change recorded before the run. If no version has passed yet, the live stage waits.
- **Run history.** One public page, generated from the registry, shows each version's data, scores and whether it shipped.
- **Test sets.** Tuning uses FLORES dev, and FLORES devtest is used only at the gate. Part of the human-rated set is replaced with fresh devtest sentences each round. Reports state how many times each set has been used.
- **When gains flatten.** Votes stop helping once the model catches up with the models it learns from, and corrections can still help. The run history will show this, and it is the signal to collect more corrections rather than more votes.

---

## Data bootstrap

1. **Baseline first.** Evaluating the app's existing models and three candidate base models on FLORES needs no user data and is a publishable result on its own.
2. **SFT seed from public parallel data**, licenses checked per corpus. **Avoid anything derived from Cebuano Wikipedia**: it is overwhelmingly bot-generated and is a known trap for Cebuano corpora. Prefer OPUS sources with human origin (Tatoeba, bible-uedin, QED, GlobalVoices where available).
3. **Synthetic preference pairs:** run several models on source sentences, rank outputs with chrF++ against references and the blind judge, build DPO pairs from the ranking. Labeled synthetic everywhere.
4. **Real votes** join once the privacy page names the training provider, and only votes saved after that change. Every dataset manifest and report shows the human versus synthetic share.
5. **Rater picks and corrections** join once the app is live. They reach the next training run after they are collected, so the first challenger does not benefit from them. Manifests count them separately from votes.

---

## Tech stack

- **Python 3.12, uv, pydantic, typer (CLI), FastAPI (local service and rater app), SQLite.**
- **Metrics and judge:** sacrebleu, COMET (reported only, later), OpenRouter for candidate models, and the Anthropic SDK with Claude Opus 5.5 on the Batches API for the judge.
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
    rater/       rater app: tasks, submissions, quality checks
    cli.py       flywheel ingest | curate | rate | build | train | eval | promote | report | serve
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

Part-time pace. Roughly 11 to 14 weeks for M0 to M6, with a **minimum showable repo at the end of M4** (about week 9). A failed gate delays M5 until a later version passes, so M6 does not wait for M5. Reviewer value ranks: eval report, then training plus gate, then live integration, then AWS. Cuts happen in reverse order.

| # | Milestone | Done when |
|---|---|---|
| M0 | Decisions and skeleton | Open decisions below answered and logged in `docs/decisions/`; verification items checked (LoRA serving, FLORES license); uv project, CI, license, README stub pushed to a public GitHub repo. **Built.** Other corpus licenses are checked when each corpus is added at M3. |
| M1 | Schema, ingest, curation | `flywheel ingest` loads FLORES and a comparisons export into curated events; JSON Schema exported in CI; unit tests cover dedupe, PII scrubbing, leakage protection. **Built.** |
| M2 | Eval harness and baseline | Baseline report on the chosen pair's FLORES devtest for the app's compare-set models and three candidate base models, with chrF++, judge win rates and confidence intervals, plus the first human-rated set, scored on a blind spreadsheet and loaded through the new rating schema. Base model chosen by headroom, from those that fit the cheapest GPU tier, can be exported, and are licensed for self-hosting. Published as the first write-up. **Harness built,** and translation and the judge tested on 3 sentences. The full run and the first human round are next. |
| M3 | Training and registry | A LoRA SFT model trained through Together from a versioned dataset and a checked-in run config, registered with dataset hash, cost and a model card. Then a DPO run on preference pairs. The comparisons adapter marks votes as not available for training until the privacy page names the provider. |
| M4 | Rater app and promotion gates | Rater app live with several raters, picks, corrections and quality checks. Candidates that pass the automated checks are scored blind against their base model. `flywheel promote` runs the pre-registered gate, including the human check, and produces a challenger report, pass or fail, with evidence, plus the positioning report against production models. Cold-start test run and serving host chosen. If time runs short, the human round uses the spreadsheet instead. **Minimum showable repo.** README leads with the result table and report link. |
| M5 | Live integration | Runs once a version passes the gate. About page names the serving host. Challenger added to the app as a new `AIModel` (union, dispatch, provider) named AI Translate Cebuano, in the free compare set for English to Cebuano only, off the `ALL_MODELS` list, behind a flag, with the daily cap and pre-warm in place. Its votes flow back through ingest, and first results are published after 30 days. The loop is closed. If the M4 test fails, this milestone becomes the offline fallback in section 9. |
| M6 | Polish and launch | Architecture diagram, README walkthrough, short demo video, decision log complete, write-up with a "what I would do differently" section, LinkedIn post. Resume updated. |
| M7 | Stretch | AWS deployment with OpenTofu; second training provider; generated TypeScript client and the app pushing events directly; a second language pair through the rater app. |
| Ongoing | Continuous loop | Never finished. New versions train as human examples arrive, each one goes through the gate, and the run history grows. See section 11. |

## Cost estimate

- Development: about $20 to $75 a month (hosted fine-tuning per token, eval API calls).
- One-time experimentation budget: $100 to $200.
- No always-on GPU. COMET runs on the MacBook.
- M2 baseline run: about $5. Translation is about $0.30, scaled from a 3-sentence test, and the judge is about $4.60, scaled from 54 test comparisons.
- Offline evals run on the same small GPU as live serving, at about $0.59 an hour. A run may take about an hour, which has not been measured yet.
- Live serving from M5, ongoing: about $1.50 a month at 5 Cebuano comparisons a day, and about $15 at most at the 50-a-day cap. Free host credits may cover this. These figures assume each request keeps the GPU awake for about 60 seconds, which M4 will measure.
- Ongoing training: paid per run, only when enough new human examples arrive, within a monthly budget set before the loop starts.
- Rater app hosting: a small host that runs only during rating rounds, with a target of under $10 a month. Not priced yet. This assumes volunteer raters; paying them would be a separate budget.
- Re-check provider prices at M0; GPU and fine-tuning prices have moved during 2026.

---

## Risks

| Risk | Mitigation |
|---|---|
| No headroom: base model already matches the compare set | M2 ranks base candidates by headroom before any training. Choose the strongest base that is still clearly below the teachers on Cebuano. |
| Metric validity on Cebuano | chrF++ is the gate metric; COMET is reported as unvalidated; the blind human check can veto a model. |
| Judge is not much better than the candidates | Judge from a different family, run in both orders, and the blind human check can veto a model the judge favored. |
| No native speaker available | The automated results are still published, so M4 stays showable, but the model stays a candidate and does not go live. |
| Careless or inconsistent raters | Hidden known-good and wrong translations, agreement reporting, and exclusion rules set before each round. |
| Rated text leaks into training | Evaluation tasks have no correction box, the training pool never uses FLORES, and the existing leakage check still runs. |
| Rater app overruns its estimate | The app and spreadsheet share a rating format, so the round falls back to the spreadsheet. Label Studio or Argilla are the next option. |
| Too little real feedback data | Baseline and synthetic pairs carry M2 to M4; every manifest and report states human versus synthetic share. |
| Vote bias (names visible) | Votes used for training signal only, never for evaluation. Own-model pick rate reported. |
| Live serving cost | Serverless LoRA is gone, so the challenger runs on a scale-to-zero GPU. English to Cebuano only and a daily cap enforced in the app keep the cost bounded for as long as it runs. |
| Challenger cold starts stall comparisons and collect few votes | The app pre-warms the model when Cebuano is selected. The M4 test confirms this works before anything goes live, and the offline fallback costs nothing. |
| Adapter cannot be exported or self-hosted | Confirm export and the base model's license before choosing the base at M2. Fall back to a short-lived Together endpoint for evals. |
| Corpus licenses and quality | Check each corpus. FLORES-200 is CC-BY-SA and is eval-only. NLLB-200 weights are CC-BY-NC, so no NLLB in anything commercial. No Cebuano Wikipedia derivatives. |
| Timeline | Minimum showable repo at M4. M5 to M7 are additive; each can be cut without hollowing out what is already published. |
| A new version is worse | It stays offline, users keep the current version, and it is logged in the run history. The next version trains on more data. The gate catching it is the system working. |
| Repeated runs overfit the test sets | Tuning uses FLORES dev, devtest is used only at the gate, and part of the human-rated set is replaced each round. |
| Rater time runs out | Only candidates that pass the automated checks reach raters. |
| User text reaches an undisclosed party | Raters see public text only. Votes reach training, and the live model reaches its host, only after the about page names the provider. |

---

## Open decisions

1. **Repo name.** Decided: `feedback-flywheel`.
2. **First language pair.** English to Cebuano. Tier C in the community model plan, in FLORES-200, and sanity-checkable in Cebu. The metric caveats above are accepted as part of the story.
3. **Base model.** Chosen at M2 by headroom from Together's LoRA fine-tuning list. Candidates include Qwen3.5 4B and 9B, Gemma 4 26B A4B and Llama 3.1 8B. Gemma 4 31B IT is on that list too and is also a teacher in the app's free compare set, but a teacher has no headroom over itself, so a smaller Gemma is the cleaner distillation story. A small base is preferred because it must fit the cheapest GPU tier. Confirm that the adapter can be exported and that the license allows self-hosting.
4. **License.** Decided: Apache 2.0. It covers the code only; datasets keep their own licenses.
5. **Native speakers for the human-rated set.** At least one person for about 100 sentences, scored twice a few weeks apart for consistency. More raters are welcome, and the app supports them from M4. Needed for M2 and for the gate at M4.
6. **Live serving host.** Modal or RunPod, chosen at M4 from the cold-start test. See `docs/decisions/0005`.
7. **Rater app host.** Chosen at M4. See `docs/decisions/0007`.
8. **Real user text.** Decided: raters see public text only, and real votes reach training only after the privacy page names the provider. See `docs/decisions/0009`.
9. **Loop settings.** The retrain count, the monthly training budget and the champion vote target. Set before the model goes live.

## Resume line when shipped

> Built and open-sourced a human-feedback model improvement pipeline in Python (preference curation, LoRA SFT and DPO, a native-speaker rating app, blind evaluation with bootstrap confidence intervals, pre-registered promotion gates) and ran it as a continuous loop behind translate.moretosay.org, where gate-approved fine-tuned versions compete in the product's compare view. [N] human votes, [X]-point chrF++ gain over base on FLORES-200 Cebuano, 95% CI [a, b].

Fill in the bracketed numbers from the M4 report. If the gain is zero or negative, the line says so and leads with the gate instead. If the offline fallback is used, replace the live claim with the blind human evaluation.
