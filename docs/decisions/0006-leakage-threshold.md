# 0006: Near-duplicate threshold of 0.6

**Status:** Accepted, 2026-10-31

**Context.** Near-duplicates of eval text must not reach training. The first default was a trigram Jaccard similarity of 0.8, which was a guess. It was measured on real FLORES-200 English sentences. Random word replacements on 300 devtest sentences measure recall. The 997 dev sentences, checked against devtest, measure false matches.

| Threshold | 1 word changed | 2 words changed | 3 words changed | False matches |
|---|---|---|---|---|
| 0.9 | 46.3% | 6.0% | 0.3% | 0 of 997 |
| 0.8 | 97.0% | 58.7% | 19.7% | 0 of 997 |
| 0.7 | 99.7% | 92.0% | 67.0% | 0 of 997 |
| 0.6 | 100.0% | 98.7% | 91.7% | 0 of 997 |

**Decision.** Use 0.6. The costs are lopsided: a leak inflates every later evaluation result, while a false match loses one training example.

**Consequences.** FLORES sentences are longer than the app's inputs, which are capped at 300 characters. Short texts could see more false matches at this threshold, and the `leak_fixed_eval` count in each curation report is where that would show. Re-measure with real votes once there are enough.

Reproduce with `scripts/measure_leakage_threshold.py`. It uses a fixed seed, so it prints these figures exactly.
