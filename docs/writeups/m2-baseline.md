# How far behind are small open models at English to Cebuano?

*Draft. The native-speaker results are still to come.*

Feedback Flywheel turns human feedback on translations into fine-tuned models, and checks each new model against what it replaces before it ships. Before training anything, it has to answer one question: is there anything to learn? A small model that already matches the large ones has nothing to gain from distillation, and the promotion gate could never pass. This baseline measures that gap for English to Cebuano.

## Setup

- **Sentences:** all 1,012 sentences of the FLORES-200 devtest set, English to Cebuano. FLORES is public, and none of it reaches training.
- **Models:** the three models in the free compare set of [translate.moretosay.org](https://translate.moretosay.org), plus three smaller open models that could be fine-tuned and self-hosted cheaply. Each ran through OpenRouter with the app's own prompt and settings, so the scores describe what app users get.
- **chrF++:** a character-level match against the human reference. Every range is a 95% paired bootstrap interval over 2,000 resamples.
- **Judge:** Claude Opus 5.5 compared each small model with each app model on 200 sentences. It saw no reference and no model names, and judged each pair in both orders. Claude is from a different family than every candidate, so it has no model of its own in the race.
- **Cost:** $4.96 in total. Translation cost $0.57, and the judge cost $4.39 at batch prices.

## Results

| Model | Role | chrF++, 95% CI | Failed |
|---|---|---|---|
| Gemini 3.8 Flash | App model | 61.70 [61.08, 62.33] | 1 |
| DeepSeek V4.1 Flash | App model | 60.24 [59.59, 60.87] | 0 |
| Gemma 4 31B | App model | 59.72 [59.07, 60.36] | 0 |
| Gemma 4 26B A4B | Candidate | 59.54 [58.90, 60.20] | 0 |
| Qwen3.5 9B | Candidate | 45.74 [45.18, 46.36] | 0 |
| Llama 3.1 8B | Candidate | 37.15 [36.49, 37.84] | 18 |

How often the judge preferred the small model, with a split decision counted as a tie:

| Candidate | vs Gemini 3.8 Flash | vs DeepSeek V4.1 Flash | vs Gemma 4 31B |
|---|---|---|---|
| Gemma 4 26B A4B | 27% [21, 33] | 49% [43, 55] | 53% [47, 59] |
| Qwen3.5 9B | 0% | 1.8% [0.3, 3.7] | 2.2% [0.5, 4.3] |
| Llama 3.1 8B | 0% | 0% | 0% |

The two measures agree. Gemma 4 26B A4B is close to the app's models and roughly even with two of them. Qwen3.5 9B and Llama 3.1 8B are far behind, and the judge almost never preferred them.

The judge was consistent. The model shown first won exactly half the time, and none of its 3,562 comparisons failed. Swapping the order changed its decision on 26 to 34 of 200 sentences when Gemma 4 26B A4B was involved, and on at most one sentence for the other two. Close calls split. Clear ones don't.

## One answer nearly decided the ranking

In the first full run, Gemini 3.8 Flash answered one sentence by repeating a word for 174,826 characters. That single answer cost $0.25 of the $0.57 translation bill. Corpus chrF++ adds up character counts across all sentences, so it also cut Gemini's score from 61.69 to 50.62, and DeepSeek looked like the best model.

The harness now counts any answer longer than five times its source as a failure, under the same rule for every model, and requests carry a 4,096 token cap. Failures are scored as empty answers and are not retried, because retrying only the bad answers would flatter a model. Llama 3.1 8B had 18 such answers, and its score includes them.

The general lesson is that one broken output can move a corpus-level metric more than the real differences between models. Checking output lengths is cheap, and it should happen before reading any table.

## Choosing a base model

The plan ranks candidates by headroom: the strongest small model that is still clearly below the best app model.

| Candidate | Gap to Gemini 3.8 Flash, chrF++ |
|---|---|
| Gemma 4 26B A4B | 2.16 [1.66, 2.65] |
| Qwen3.5 9B | 15.96 [15.31, 16.58] |
| Llama 3.1 8B | 24.55 [23.78, 25.25] |

On scores alone, Gemma 4 26B A4B is the pick: strongest, and still measurably behind. Three free checks changed that ([decision 0011](../decisions/0011-base-model-checks.md)):

- **GPU fit.** Gemma 4 26B A4B uses about 4B parameters per token, but all 25.8B must sit in memory. On a 24 GB L4, the cheapest card any candidate fits, it runs only after quantizing to 4 bits. Qwen3.5 9B fits unquantized.
- **Headroom that counts.** A 2-point gap, and a judge that calls Gemma 4 26B A4B even with two of the three teachers, leaves little for distillation to teach.
- **License.** Both Gemma 4 and Qwen3.5 are Apache 2.0. Llama 3.1 requires any model built from it to carry "Llama" at the start of its name.

Qwen3.5 9B leads: it fits the card without quantizing, and it has a 16-point gap to close. That is a large gap. It gives fine-tuning room to show a real gain, and it also means the first versions will still lose to every app model. The write-up for each version will say so.

## Where chrF++ falls short

chrF++ rewards matching the reference's characters, not being right. On the sentence below, Gemma 4 26B A4B scored 33.8 and Gemini 3.8 Flash scored 75.3:

| | Text |
|---|---|
| English | Check the label for specific first aid instructions for that specific poison. |
| Reference | Susiha ang timaan sa piho nga mga panudlo sa first aid alang sa piho nga hilo. |
| Gemini 3.8 Flash | Tan-awa ang marka alang sa piho nga mga panudlo sa first aid alang nianang piho nga hilo. |
| Gemma 4 26B A4B | Tan-awa ang label para sa espesipikong mga instruksyon sa first aid para sa maong hilo. |

Gemma's answer uses English and Spanish loanwords ("label", "espesipikong", "instruksyon") where the reference uses Cebuano ones. Whether a reader finds it worse, or just more casual, is a question for a native speaker, and the metric cannot answer it.

## Native-speaker scores

> **To come.** A native Cebuano speaker is scoring 100 sentences from Gemini 3.8 Flash, Gemma 4 26B A4B and Qwen3.5 9B. The sheet hides which model wrote each translation, and 10 hidden checks screen for careless rating. Adequacy and fluency are scored from 1 to 5, and a second round a few weeks later measures consistency.

## Limits

- **One test set.** FLORES sentences come from Wikimedia news articles, travel guides and children's books. App users write shorter and more casual text.
- **The judge is not a Cebuano expert.** Frontier models are only moderately competent in Cebuano, which is why the human round can veto a model at the gate.
- **Runs vary.** These models are not deterministic, so a rerun can differ. The stored outputs are the record, and every score can be recomputed from them for free.

## What's next

M3 trains the first version: a LoRA fine-tune of the chosen base on public parallel text and, once the app's privacy page names the training provider, on the votes of users who chose a translation. Every version goes through the same gate: beat what it replaces on chrF++ beyond the interval, win at least 55% with the judge, show no regression on a guard set, and pass a blind human check.
