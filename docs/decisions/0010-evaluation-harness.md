# 0010: How the evaluation harness runs

**Status:** Accepted, 2026-11-01

**Context.** M2 needs scores that describe what app users actually get, without spending more than the result is worth. Several choices were left open by the plan.

**Decision.**

- **Candidates run through OpenRouter, with the app's prompt and request settings.** That includes the routing rule that sends text only to hosts that neither store nor train on it. FLORES is public, so this is for a fair measurement and not for privacy. Gemini is the one difference. The app calls Google directly with thinking set to low. Through OpenRouter it refuses reasoning off, and at its default level it cost ten times as much in a 3-sentence test, so it runs with reasoning effort set to low to match the app.
- **The judge uses the Anthropic SDK with Claude Opus 5.5.** It is a different family from every candidate, and the config refuses to load otherwise. It runs on the Batches API at half price, and a rerun resumes the saved batch instead of paying twice.
- **The judge sees no reference.** chrF++ already measures closeness to the reference. A reference-free judge adds a second view, and it still works on real inputs that have no reference.
- **Each base meets each compare model on 200 sentences, in both orders.** Comparing every model with every other on all 1,012 sentences would take about 30,000 calls. Only the base-against-teacher pairs inform the choice of base model. Identical answers tie and an answer beats a failure, so neither costs a call.
- **Refusals and failures are counted, not hidden.** A sentence the judge could not answer leaves the win rate and appears as unusable. Server-side refusal fallbacks are not available on the Batches API.
- **A failed translation scores as an empty answer.** Leaving it out would reward a model for failing on hard sentences.
- **Intervals are paired.** Two models are resampled on the same sentences, so a hard sentence counts against both sides of a difference.
- **Headroom is measured against the best compare model.** That model sets the ceiling for what distillation can teach. The strongest base clearly below it is recommended, subject to the GPU, export and license checks.
- **Paid commands show a plan and stop unless given `--yes`.** Translation also has a hard spending limit, checked by every worker, so overspend is limited to requests already in flight.
- **Sheets contain no model names, and the key stays with the operator.** Row ids and row order change each round, while the sentences stay the same. Cells that a spreadsheet could read as formulas are escaped.

**Not built yet.** COMET is deferred. It brings a heavy dependency, it is unvalidated for Cebuano, and the M2 result does not depend on it.

**Consequences.** Six models on 100 sentences, plus 10 hidden checks, makes 610 rows per round. `flywheel rate export --models` narrows a round to the finalists, which is the practical choice for a volunteer. Models are not deterministic, so a rerun of the translations can differ. Stored outputs are the record, and scores can always be recomputed from them for free.
