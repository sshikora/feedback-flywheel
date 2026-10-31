# 0003: No user data or corpora in the repo

**Status:** Accepted, 2026-10-31

**Context.** The repo is public and the app's votes contain text people typed. FLORES-200 is CC-BY-SA 4.0, which would attach share-alike terms to anything redistributed with it.

**Decision.** `data/` and `.cache/` are ignored by git, along with all JSONL files except test fixtures. Fixtures are synthetic. FLORES-200 is downloaded at runtime and pinned by SHA-256. Scrubbing runs at ingest, so unscrubbed text is never written by the pipeline. Curation reports hold counts only.

**Consequences.** Anyone can clone and run the pipeline, but real results come from data that stays private. Published reports contain aggregates, plus benchmark sentences under their own license.
