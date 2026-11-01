# 0008: Keep the loop running after launch

**Status:** Accepted, 2026-11-01. Replaces the fixed 30-day window in `docs/decisions/0005`.

**Context.** The plan treated the live stage as a one-off: run for 30 days, publish, switch off. It also compared every model only with its own base. The goal is a model that keeps improving, so a single run does not fit, and neither does treating a failed gate as the end of the work.

More data usually helps, but not reliably. Votes stop helping once the model catches up with the models it learns from. Corrections can keep helping, because they add new knowledge. Any new data can still make a version worse, so improvement has to be measured for each version.

**Decision.**

- **The gate stays, and it applies to every version.** The first version must beat its base model. Later versions must beat the current live version.
- **A failed version is held back.** It stays offline, users keep the current version, and it is logged. The next attempt needs new data, or a config change recorded before the run. If no version has passed yet, the live stage waits.
- **Retraining follows data, not a calendar.** A run starts when enough new human examples have cleared curation, at a count set in config, within a monthly training budget.
- **A passing version stays live.** It remains on free English to Cebuano comparisons under the daily cap until a better version replaces it or it is switched off. First results are published after 30 days.
- **One public run history.** A page generated from the registry shows each version's data, scores and whether it shipped. A failed run is one row in that history, not a separate announcement.
- **Test sets are protected.** Tuning uses FLORES dev, and FLORES devtest is used only at the gate. Part of the human-rated set is replaced each round, and reports state how many times each set has been used.
- **The public name is AI Translate Cebuano.** The app always shows model names. This one uses the product's own name, says what the model does and implies no third-party brand. "Test" was rejected because it would probably pull votes down.
- **Free comparisons only.** Pro comparisons accept anything in the app's `ALL_MODELS`, so the model stays off that list. Otherwise Pro users could select it for any language.

**Alternatives considered.** Retrying straight away after a failure is faster, but it costs more per pass and repeated tries against one test set make the scores less trustworthy. Running 30-day rounds per version is cheaper, but votes stop between rounds and the loop slows down. Dropping the gate would remove the only evidence that the model is actually improving.

**Consequences.** Live serving becomes an ongoing cost, estimated at $1.50 to $15 a month from `docs/decisions/0005`, plus training runs. The daily cap still bounds serving costs for as long as the model runs. Rater time also becomes an ongoing need, so only candidates that pass the automated checks reach raters.

M5 now depends on a version passing, which can delay it, so M6 does not wait for M5. At current traffic, reaching a vote target may take a long time. Until then, vote results are reported as inconclusive, and replacing a version depends on the gate rather than on votes.
