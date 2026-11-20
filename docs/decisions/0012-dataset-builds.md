# 0012: How training datasets are built

**Status:** Accepted, 2026-11-04

**Context.** M3 trains from versioned datasets, and every model must trace back to exactly what it learned from. Several choices in turning curated events into SFT examples and DPO pairs were left open by the plan.

**Decision.**

- **Only curated training events come in.** Builders read `curated/train.jsonl` and nothing else, so held-out and eval text cannot reach a dataset by a second path.
- **A snapshot is named by the hash of its examples.** The folder is `<kind>-<first 12 hex characters>`, and the manifest holds the full SHA-256. Rebuilding the same content changes nothing, and a snapshot whose examples were edited is refused. A training run that names a hash names exactly one dataset.
- **SFT keeps contested events, and DPO does not, by default.** When voters on one input chose different answers, the majority answer is still a reasonable target. Its rejected answers, though, were someone else's choice, so DPO would teach the model to avoid an answer people preferred.
- **Pairs come from the same comparison.** Each answer voters passed over becomes one pair against the chosen one. Failed answers, and answers that match the chosen one after normalizing, make no pair. Models in an agreement group are never set against each other.
- **Close pairs are flagged, not dropped.** A pair is close when chrF++ between chosen and rejected is at least 70. In the M2 baseline, Gemini 3.8 Flash and DeepSeek V4.1 Flash answers to the same sentence had a median of 70.2, so about half of real pairs may be close. The data card reports the share, since the rejected answer in a close pair is often also correct.
- **Excluding a model is a recorded setting.** `exclude_models` drops any text written only by the named models. When an agreement group includes an excluded model, the text comes from one that is not excluded. This is how the Gemini terms question in `0011` gets settled at M3, as a setting the manifest records.
- **The data card is generated from the manifest,** so the two cannot disagree.

**Not built yet.** Corrections, from the rater app, become SFT examples and DPO pairs at M4. Public parallel corpora join as synthetic-origin events when the first one is cleared.

**Consequences.** Snapshots hold user text, so they live in `data/` and are never committed. Every build records the curation settings and the hash of the curated file it read. Changing a dataset setting produces a new snapshot and leaves the old one untouched.
