# 0005: How to serve a challenger now serverless LoRA is gone

**Status:** Accepted, 2026-10-31, subject to the cold-start test at M4. M0 to M3 are unaffected. The fixed window is replaced by `0008`, and the fallback's real voted inputs by `0009`.

**Context.** The plan assumed per-token LoRA serving with no always-on GPU. Checking this at M0 showed that assumption is false. Together's docs state that serverless LoRA inference, including Serverless Multi-LoRA, has been discontinued. Adapters now run on dedicated endpoints, billed per minute per running replica. Idle shutdown is opt-in through an inactive timeout of 30 to 1,440 minutes.

Together's docs list one H100 at $3.99 an hour, while third-party guides quote up to $6.49. At $3.99, an always-on endpoint costs about $2,900 a month. That is ruled out for a project with no paying customers.

**Decision.** Serve the challenger on a scale-to-zero GPU host, with these limits:

- **Host:** Modal or RunPod, chosen at M4 from a real cold-start measurement.
- **Weights:** Fine-tune on Together, then export the adapter and merge it into the base model for self-hosting.
- **Cebuano only:** The challenger runs only on English to Cebuano comparisons.
- **Daily cap:** The app stops calling the challenger after a set number of requests a day, starting at 50. Comparisons then continue with the other models. The cap is enforced in the app, so the spending limit does not depend on any host feature.
- **Pre-warm:** The app wakes the model when a user selects Cebuano as the target language, so it is ready before they press Compare.
- **Small base model:** Choose a base that fits the cheapest GPU tier.
- **Fixed window:** Run live for 30 days, publish the results, then switch it off.

Offline evaluation uses the same host. A short-lived Together dedicated endpoint is the fallback if adapter export or self-hosting turns out to be blocked.

**Expected cost.** About $0.59 an hour for a small GPU. If each request keeps the GPU awake for about 60 seconds, a request costs about one cent. The 60 seconds is an assumption until M4.

| Traffic | Monthly cost |
|---|---|
| About 5 Cebuano comparisons a day | About $1.50 |
| At the cap of 50 a day | About $15 at most |

Pre-warm requests add to this, and each one counts against the daily cap. Modal may also offer free monthly credits of about $30, but sources conflict on the amount.

**Why cold starts matter.** At low traffic, nearly every request finds the model asleep. The app gives each model 12 seconds and returns all answers together, so a sleeping challenger would hold up the whole comparison for the full 12 seconds, then show as failed and collect no vote. Raising the timeout would make that wait longer. Pre-warming avoids both problems, and it needs a small app change.

**Test at M4.** On the chosen base model, measure the time from a cold start to the first answer, and the time for a warm answer. The live plan goes ahead if a pre-warmed model reliably answers within 12 seconds.

**Fallback if the test fails.** Skip live hosting, at no ongoing cost. Run the challenger offline on real voted inputs, have the Cebuano rater score the results blind, and record the demo locally through Ollama. The pipeline, evaluation and gates all stand. The "ran it live" claim comes off the resume line, and the write-up says why.

**Options considered.**

- **A scheduled window on a Together endpoint:** Two hours a day costs about $8 a day and only collects votes inside the window.
- **Groq LoRA inference:** Groq documents uploaded adapters, but its page could not be read to confirm plan access or base model support. Worth checking at M4.

**Sources.** Modal's rates come from third-party sources, because modal.com could not be read directly. Re-check them at M4.

- [Together: LoRA vs. full fine-tuning](https://docs.together.ai/docs/lora-inference)
- [Together: fine-tuning supported models](https://docs.together.ai/docs/fine-tuning/supported-models)
- [Together: serve multiple LoRA adapters on one endpoint](https://docs.together.ai/docs/dedicated-endpoints/v1/lora-adapter)
- [Together: dedicated endpoints overview](https://docs.together.ai/docs/dedicated-endpoints/overview)
- [RunPod: serverless pricing](https://docs.runpod.io/serverless/pricing)
- [Spheron: Modal GPU pricing 2026](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/)
- [xpay: Modal pricing snapshot](https://www.xpay.sh/agent-ready-index/modal/)
- [Groq: LoRA inference](https://console.groq.com/docs/lora)
