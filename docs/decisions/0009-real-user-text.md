# 0009: Keep real user text from raters and undisclosed providers

**Status:** Accepted, 2026-11-01. Replaces the real voted inputs in the fallback in `docs/decisions/0005`.

**Context.** The plan assumed consent was already covered, because the app says votes and compared text are kept to improve translation quality. Checking the privacy section on `/about` showed that the purpose is covered, but not every recipient:

- The third-party section lists translation providers. It names no training provider, so uploading votes to Together at M3 would be undisclosed.
- It says nothing about people reading compared text, and the offline fallback had a rater scoring real voted inputs.
- It says votes cannot be linked to anyone and cannot be found or deleted on request, so a vote cannot be withdrawn later.
- The scrubber has been validated on FLORES, but it has never run on a real export.

**Decision.**

- **Raters see public text only.** The rater app and the offline fallback use FLORES and cleared public corpora, never user text.
- **Real votes reach training only after the privacy page names the training provider,** and then only votes saved after that change. Early models train on public corpora, synthetic pairs and rater input.
- **The serving host is named on the page before the first live request,** and it must meet the page's promise not to store content or use it for training.

**Alternatives considered.** Treating human review and training upload as covered by "improve translation quality" is simplest, but it stretches what users were told. Using older votes once the page changes would keep more data, but those votes were saved under the old wording. Never uploading real votes has the lowest risk, but it weakens the human feedback story the project is built around.

**Consequences.** Early models contain less vote data, and the fallback no longer measures real inputs. Older votes can still be used locally for aggregate vote statistics, which the page already allows.

The comparisons adapter currently sets `consent.training` to true for every vote. That must change before the first upload at M3. Curation already drops events without training consent, so no other pipeline change is needed for this rule.
