# 0016: Evaluate trained models on Modal

**Status:** Accepted, 2026-11-05. Updated 2026-11-06 after the first runs. Applies `0005`'s plan for offline evaluation. Live serving is still chosen at M4.

**Context.** The first training run, `qwen3.5-9b-sft-opus-1`, finished on Together. The eval harness only reached models through OpenRouter, which cannot serve our weights, and Together serves trained models only on dedicated endpoints billed while they run. `0005` planned offline evaluation on the same scale-to-zero host as live serving.

**Decision.**

- **Batch, not a server.** `flywheel eval translate` sends every missing sentence for a self-hosted model to one Modal function. It loads the weights with vLLM, answers in one batch and exits, so nothing is left running.
- **The GPU is an L4.** It holds Qwen3.5 9B unquantized, see `0011`. Modal charges $0.000222 a second, about $0.80 an hour, and its free Starter plan includes $30 of compute a month.
- **The timeout caps the spend.** The function stops after 30 minutes, so one model costs at most about $0.40. The command refuses a model whose worst case does not fit the config's `max_cost_usd`. Each record's cost is the GPU time split evenly across its sentences.
- **The base runs on the same host.** The M2 baseline ran Qwen3.5 9B through OpenRouter, where the provider and its settings are not ours. The trained model is compared with its base run here with the same settings, so a difference comes from training alone.
- **Settings follow the model card.** Thinking is off, as in training and as in the app. Sampling uses Qwen's recommended non-thinking settings: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, with a fixed seed. Answers stop at 4,096 tokens, the same stop as OpenRouter requests.
- **Weights go straight to Modal.** A base model comes from Hugging Face. For a trained model, this machine asks Together for its short-lived storage link and passes only the link, so the Together key never leaves this machine and 19 GB are never uploaded from it. Weights are kept in a Modal volume, so each model is fetched once.
- **Work outlives the connection.** The app is deployed, and each model's work is one call started on Modal, with its id saved in the run folder. The first attempt from this machine was cancelled when the laptop lost its connection for a few minutes. Now the call carries on, and the next run collects it instead of paying again.
- **The model runs eagerly.** The weights take 16.7 GB of the L4's 22 GB, and vLLM ran out of memory capturing CUDA graphs. It runs without them, at most 64 sentences at a time.
- **Configs name the host.** A model in an eval config takes `host = "modal"` and `weights = "hf:<repo>"` or `"run:<training run>"`.

**Found during this.** `flywheel train download` saved an empty file and reported success. Together answers with a redirect to its storage host, which the client did not follow. It now follows the redirect, which drops the API key because the host differs, and refuses an empty download. The first real download then stalled at 5.6 GB and failed after 10 minutes with no way to continue. It now resumes from its partial file, and Modal's fetch retries and resumes too.

**Consequences.**

- The evaluation report was written for the baseline, which ranked base models by headroom. A trained model against its base needs its own report: the chrF++ gain with its interval, and the judge win rate against the 55% gate.
- Live serving needs a web endpoint, a cold start test and a daily cap, and is still M4.
- Modal's rate is read from its pricing page on 2026-11-05 and is kept in `flywheel/evals/modal_host.py`.

**Sources.**

- [Modal pricing](https://modal.com/pricing)
- [Qwen3.5 9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Together: SFT quickstart, downloading the weights](https://docs.together.ai/docs/post-training/quickstart-sft)
