# 0011: Base model checks for export, license and GPU fit

**Status:** Checks done, 2026-11-02. The base model is still chosen at M2, after the first human round. Changes the GPU tier in `0005`.

**Context.** The M2 baseline ranked three candidate base models. Before one is chosen, the plan requires three checks: the trained adapter can be exported, the license allows self-hosting, and the model fits the cheapest GPU tier. These checks are free and need no training run.

**Export.** All three are on Together's LoRA fine-tuning list with both SFT and DPO: `google/gemma-4-26B-A4B-it`, `Qwen/Qwen3.5-9B` and `meta-llama/Meta-Llama-3.1-8B-Instruct-Reference`. Together documents two downloads. The merged variant is the base model with the adapter folded in, ready for any inference framework. The adapter-only variant uses a layout PEFT cannot load, so the pipeline downloads the merged model. The docs state no model restriction on downloads, which is only proven by a first real download at M3. Together needs a first credit purchase of at least $15, and a balance that covers each job. (This first said $100, which Together's billing docs do not support. Corrected 2026-11-05.)

**License.**

| Model | License | Self-hosting | Conditions |
|---|---|---|---|
| Gemma 4 26B A4B | Apache 2.0 | Allowed | None beyond Apache 2.0. Gemma 4 replaced the earlier Gemma terms. |
| Qwen3.5 9B | Apache 2.0 | Allowed | None beyond Apache 2.0 |
| Llama 3.1 8B | Llama 3.1 Community License | Allowed | A model made with it must have a name beginning with "Llama", and the site must show "Built with Llama". Downloads are gated behind a request. |

The Llama naming rule conflicts with the planned name, AI Translate Cebuano.

**GPU fit.** The plan's $0.59 an hour is Modal's T4, a 16 GB card. Modal's L4, with 24 GB, costs about $0.80 an hour. Weight sizes below are parameters times bytes per weight, before the cache and runtime that also need memory.

| Model | Parameters | 16-bit | 8-bit | 4-bit | T4, 16 GB | L4, 24 GB |
|---|---|---|---|---|---|---|
| Gemma 4 26B A4B | 25.8B | 51.6 GB | 25.8 GB | 12.9 GB | No | 4-bit only |
| Qwen3.5 9B | 9.65B | 19.3 GB | 9.7 GB | 4.8 GB | Quantized only, unverified | Yes, unquantized |
| Llama 3.1 8B | 8.03B | 16.1 GB | 8.0 GB | 4.0 GB | Quantized only | Yes, unquantized |

- **Gemma 4 26B A4B** uses only about 4B parameters per token, but all 25.8B must sit in memory. Unquantized it needs an A100 or H100. On an L4 it runs only as a 4-bit model, so the gate would have to score the quantized model, and quantizing our own merged model is an extra step with community tooling.
- **Qwen3.5 9B** fits an L4 unquantized. Only 8 of its 32 layers keep a growing cache, so short translations leave room. On a T4 it would need quantizing, and the T4 has no bfloat16 support. vLLM supports the T4, but whether its kernels for Qwen3.5's linear-attention layers run on that card was not confirmed.
- **Llama 3.1 8B** fits an L4 unquantized with room to spare.

**Decision.**

- **The cheapest tier is the L4, not the T4.** No candidate fits a T4 unquantized. At the plan's 60 seconds per request, a request costs about 1.3 cents. That is about $2 a month at 5 comparisons a day and about $20 at most at the cap of 50.
- **Qwen3.5 9B now leads.** It fits the L4 unquantized, is licensed Apache 2.0, and has clear headroom: 45.74 chrF++ against 61.70 for the best compare model, and 0 to 2% judge wins. Gemma 4 26B A4B scored 59.54 and won 49% against DeepSeek and 53% against Gemma 4 31B, so it leaves little room for distillation, and it fits the L4 only at 4-bit.
- **Llama 3.1 8B is the reserve.** It passes all three checks, but it scored lowest, had 18 runaway answers, and would need a name beginning with "Llama".
- **The human round still decides.** Round m2-a scores Gemini 3.8 Flash, Gemma 4 26B A4B and Qwen3.5 9B. If the raters find Qwen3.5 9B much better or worse than its chrF++ suggests, that changes the choice.

**Found during these checks.** Google's Gemini API terms say "You may not use the Services to develop models that compete with the Services." Training on votes where the chosen answer came from Gemini could fall under that clause. This is a training data question, not a base model one, and it is settled at M3, before the first upload. One option is to drop Gemini-written answers from training targets and keep them for evaluation. Gemma 4 31B is Apache 2.0, and DeepSeek's terms were not checked.

**Sources.**

- [Together: fine-tuning supported models](https://docs.together.ai/docs/fine-tuning/supported-models)
- [Together: SFT quickstart, downloading the weights](https://docs.together.ai/docs/post-training/quickstart-sft)
- [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)
- [Qwen3.5 9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Llama 3.1 license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)
- [vLLM: GPU installation and supported hardware](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- [vLLM recipe: Gemma 4 26B A4B](https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it)
- [Gemini API additional terms](https://ai.google.dev/gemini-api/terms)
- Modal's rates come from third-party snapshots, because modal.com could not be read directly: [usagepricing.com](https://www.usagepricing.com/blueprint/activity/modal-2026-07-14-b300-sandbox-notebooks) and [computeprices.com](https://computeprices.com/providers/modal/gpus/t4).
