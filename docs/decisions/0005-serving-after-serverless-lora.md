# 0005: How to serve a challenger now serverless LoRA is gone

**Status:** Proposed, 2026-10-31. Needs a decision before M5. M0 to M4 are unaffected.

**Context.** The plan assumed per-token LoRA serving with no always-on GPU. Checking this at M0 showed that assumption is false. Together's docs state that serverless LoRA inference, including Serverless Multi-LoRA, has been discontinued. Adapters now run on dedicated endpoints, billed per minute per running replica. Idle shutdown is opt-in through an inactive timeout of 30 to 1,440 minutes.

Together's docs list one H100 at $3.99 an hour, while third-party guides quote up to $6.49. At $3.99, an always-on endpoint costs about $2,900 a month, far outside this project's budget.

**What still works.** Fine-tuning itself is unaffected, and DPO is supported. Offline evaluation fits a dedicated endpoint well: start it, run the eval sets, let it shut down. That costs a few hours of GPU per run.

**Options for live serving at M5.**

1. **Scale-to-zero serverless GPU** with exported weights, on a platform such as Modal or RunPod. Pay only while running. Cold starts may exceed the app's 12 second per-model timeout, but the compare view already shows a failed model as a small muted line, so this degrades gracefully.
2. **A scheduled test window** on a Together endpoint, for example two hours a day with idle shutdown. Simple, and about $8 a day, but it only collects votes inside the window.
3. **Another LoRA host.** Groq documents LoRA inference with uploaded adapters. Plan access and base model support have not been checked.

**Recommendation.** Option 1, verified with a real cold-start measurement at M4. It keeps the cost model closest to the original intent. It also requires that adapters can be exported from the training provider and that the base model's license allows self-hosting, which should be confirmed before the base model is chosen at M2.

**Sources.**
- [Together: LoRA vs. full fine-tuning](https://docs.together.ai/docs/lora-inference)
- [Together: fine-tuning supported models](https://docs.together.ai/docs/fine-tuning/supported-models)
- [Together: serve multiple LoRA adapters on one endpoint](https://docs.together.ai/docs/dedicated-endpoints/v1/lora-adapter)
- [Together: dedicated endpoints overview](https://docs.together.ai/docs/dedicated-endpoints/overview)
- [Groq: LoRA inference](https://console.groq.com/docs/lora)
