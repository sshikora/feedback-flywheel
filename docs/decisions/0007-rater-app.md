# 0007: Native-speaker rater app, with a spreadsheet first

**Status:** Accepted, 2026-11-01

**Context.** The plan said about 100 sentences would be scored by seth and one native speaker, but not how. It named a "human-rating import" and nothing else. That left four gaps:

- The set was scheduled for M2, but the first trained model arrives at M3, so no round covered a model that might reach users.
- The set was called the tie-breaker, but the gate had no human check, so the scores could not change any outcome.
- A spreadsheet suits one speaker and one language. With several raters or languages, blinding, assignment and quality checks become manual work.
- Votes and ratings only choose among outputs that models already wrote, so they cannot teach a model anything the compare set does not know.

**Decision.** Build a small web app for native-speaker raters, and use a spreadsheet until it exists.

- **Rating format first.** M2 adds the rating records to the schema. A spreadsheet CSV and the app both feed the same import, so the evaluation code does not care which one was used.
- **Spreadsheet at M2.** The baseline round uses a blind sheet from `flywheel rate export`, so the first public result does not wait for app work.
- **App at M4.** Raters open a private invite link and work through one task at a time. The language pair comes from config.
- **Several raters and corrections in the first version.** Adding them later would change the data format. A second language needs no new code, but it does not run until Cebuano has worked end to end.
- **Two separate pools.** The evaluation pool is a frozen set of FLORES devtest sentences, scored only, and it never reaches training. The training pool comes from the public corpora cleared at M3. Raters pick the better of two outputs or correct a translation, and those events go through ingest and curation like any other feedback.
- **A human check in the gate.** Candidates that pass the automated checks are scored blind against the version they must beat (`docs/decisions/0008`). Neither adequacy nor fluency may fall below that version beyond the confidence interval. With about 100 sentences this works as a veto, not as proof of improvement. Candidates that miss the check, or have no round, stay candidates and do not reach the app.
- **Quality checks.** The queue hides known-good and deliberately wrong translations. Each round's exclusion rule and minimum agreement are set before scoring starts. At least one rater scores the evaluation set twice, a few weeks apart.
- **A separate service.** The rater app is its own FastAPI app, sharing schemas with the pipeline. It has only task and submit routes, so the local API's event and registry routes are never exposed publicly.
- **Rater privacy.** Raters get a pseudonymous `rater_id` and a random invite token, stored hashed. The app stores no names or email addresses. Ratings, corrections and rater ids stay out of the repo, as `docs/decisions/0003` requires. Raters agree to terms that cover training use and aggregate publication before they start.
- **Public text only.** Raters never see user text, as `docs/decisions/0009` requires.

**Alternatives considered.** Spreadsheets alone are cheaper, but they keep blinding manual and collect no corrections in a usable form. Label Studio or Argilla would work and remain the answer if the build overruns the estimate, but the task is narrow and the app is part of what this project demonstrates. Keeping the human set as a report only would save rater time, but a native speaker would have no say before a model reaches users.

**Consequences.** Corrections are the one signal that can go beyond the distillation ceiling, because they add text no model wrote. They are also slow to collect. Hundreds or more are probably needed before they change a measured result, which is an expectation and not a measurement, and they reach only the training run after they are collected, so the first challenger does not benefit.

The app adds an estimated one to two part-time weeks and a small hosting cost, since raters need a public URL. It runs only during rating rounds. If time runs short, the M4 round falls back to the spreadsheet with no change to the schema or the gate.

The gate now depends on finding a rater. If none is available, the automated results are still published and M4 stays showable, but no model goes live.
