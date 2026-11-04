# 0014: Training on Together

**Status:** Accepted, 2026-11-04

**Context.** M3 trains a LoRA model through Together, and `0005` already rules out Together for serving. Together offers two products. Fine-tuning runs managed jobs billed per token. Post-training has your own code drive the training loop on GPUs you provision, billed per GPU-minute. The plan's provider interface submits a job, checks it and downloads the result, which matches fine-tuning.

**Decision.**

- **Use Together's fine-tuning API over plain HTTP.** It is the same pattern as the OpenRouter client, with no SDK dependency, and every request is visible in one file.
- **Train with the app's prompt.** Each example is the prompt the app sends, from `flywheel/prompts.py`, with the chosen answer as the reply. The eval harness builds its prompt from the same module. When a user left the source language on auto, the app asks for JSON with the detected language, so those examples teach that answer. Examples the prompt cannot express, such as an unsupported language or an unknown detected source, are skipped and counted.
- **Price before paying.** `flywheel train start` writes and uploads the training file, waits for Together's validation and gets a price. It starts the job only with `--yes`. Together prices only uploaded, validated files, so the upload happens without `--yes`. The upload is free, and the command says that it happened. A run config's `max_cost_usd` refuses a job whose estimate is higher.
- **One job per run.** Creating a job is billed, so it is never retried. The job id is saved before anything else, and once it is saved, no command creates another job for that run. An edited run config or snapshot is refused before upload or submission.
- **A completed job registers itself.** `flywheel train status` registers the model as a candidate when the job completes. The cost recorded is Together's dollar estimate, because the unit of the job's own `total_price` field is not documented. The first real job will settle that.
- **Download the merged weights.** The merged checkpoint is self-contained and serves on vLLM without a separate adapter step. It is about 19 GB for Qwen3.5 9B.

**Cost.** For Qwen3.5 9B, LoRA SFT costs $0.34 per million tokens and DPO $0.84, with a $4.00 minimum per job. A first dataset of a few thousand short sentence pairs is likely to fall under the minimum, so a run should cost about $4. Together also requires a $100 minimum credit balance.

**Not covered.** Real votes reach Together only under `0009`: after the privacy page names it, and only for votes cast after that change. The question of training on Gemini-written answers from `0011` is settled with `exclude_models` when the first dataset is built, and `train start` lists which models wrote the answers to learn.

**Sources.**

- [Together: fine-tuning quickstart](https://docs.together.ai/docs/fine-tuning/quickstart)
- [Together: create a fine-tuning job](https://docs.together.ai/reference/post-fine-tunes)
- [Together: estimate price](https://docs.together.ai/reference/post-fine-tunes-estimate-price)
- [Together: data preparation](https://docs.together.ai/docs/fine-tuning/data-preparation)
- [Together: deploy or download](https://docs.together.ai/docs/fine-tuning/deployment)
- [Together: fine-tuning pricing](https://docs.together.ai/docs/fine-tuning/pricing) and [rates](https://www.together.ai/pricing)
