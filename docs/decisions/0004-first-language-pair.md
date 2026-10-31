# 0004: English to Cebuano first

**Status:** Accepted, 2026-10-31

**Context.** Head languages leave no room to improve on frontier models. Cebuano is a low-resource language in FLORES-200, the translation app supports it, and its output can be sanity-checked locally in Cebu.

**Decision.** Start with English to Cebuano and design the evaluation around its weaknesses.

**Consequences.** Two of the usual metrics are weak here. COMET's XLM-R backbone does not cover Cebuano, so its scores are reported as unvalidated. Frontier models are only moderately competent in Cebuano, so an LLM judge cannot be the last word. chrF++ is the gate metric, and a human-rated set is the tie-breaker. Cebuano Wikipedia is largely bot-generated, so corpora derived from it are excluded.
