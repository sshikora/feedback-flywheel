# 0017: Synthetic pairs: the teacher and the English text

**Status:** Accepted, 2026-11-12. Source text is loaded. Teacher translation comes next, after its cost is confirmed.

**Context.** The first trained model tied with its base, as expected from 1,781 corpus pairs (`0016`). The volume has to come from synthetic pairs, plan step 3, which needs two things: a teacher model whose terms allow training on its output, and English text we may use. `0011` left open whether answers written by Gemini may train.

**The teacher.** The three strongest models in the M2 baseline, checked on 2026-11-12:

| Model | chrF++ (M2) | May its output train our model? |
|---|---|---|
| Gemini 3.8 Flash | 61.7 | Not safely. Google's API terms say "You may not use the Services to develop models that compete with the Services." "Compete" is not defined, and a Cebuano translation model could count. |
| DeepSeek V4.1 Flash | 60.2 | Yes. DeepSeek's API terms say inputs and outputs may be used for "training other models (such as model distillation)", and assign outputs to the user. The weights are MIT. |
| Gemma 4 31B | 59.7 | Yes. The weights are Apache 2.0 and the model card puts no limit on outputs. None of its OpenRouter providers is Google, so Google's API terms do not apply. |

**The English text.** The teacher translates English sentences. The text must allow derived works, commercially, without a share-alike licence that would bind the dataset.

| Source | Licence | Verdict |
|---|---|---|
| English Wikinews | CC BY 2.5, and CC BY 4.0 from 2024-12-16 on. Text from before 2005-09-25 is in the public domain | Yes. A third of FLORES is Wikinews, so this is the closest legal match to the benchmark's news domain. The site went read-only in May 2026, so the dump is fixed. |
| Tatoeba English | CC BY 2.0 FR | Yes. Short everyday sentences, like the app's inputs, which are at most 300 characters. |
| Wikipedia, Wikivoyage | CC BY-SA 4.0 | No: share-alike would carry to the dataset. |
| Voice of America | Public domain for VOA's own text | No: its pages mix in AP, Reuters and AFP copy, which is not free and is not marked reliably. |

**Decision.**

- **DeepSeek V4.1 Flash is the teacher.** It is 1.5 chrF++ behind Gemini and its terms allow exactly this use. About $0.35 for 10,000 sentences.
- **Gemma 4 31B is a second teacher** for preference pairs, so DPO has two answers to choose between. Only full-precision providers are used, because its OpenRouter providers differ in quantization.
- **Gemini never writes a training target.** This settles `0011`'s open question for votes too: a dataset config lists Gemini models in `exclude_models`, so a vote for a Gemini answer does not train. Gemini stays in evaluation.
- **The Claude judge evaluates and does not label.** Anthropic's commercial terms (section D.4) bar accessing the services "to build a competing product or service, including to train competing AI models". So, by the same rule as for Gemini, judge verdicts do not rank training pairs. Plan step 3 ranked teacher answers with chrF++ and the judge, but new source text has no reference for chrF++. How DPO pairs are ranked is decided with the first DPO run.
- **Load Wikinews and Tatoeba English** with `flywheel ingest source <name>`. Each download is pinned by sha256 and read without extracting it, as the corpora in `0015` are. A source sentence is stored as a `SourceText`, not an event: it becomes a training pair only once a teacher translates it.
- **Only published Wikinews articles.** Drafts never passed review and may hold pasted text, so an article needs the `{{publish}}` template. Markup, lists, footnotes and the sources section are removed, and paragraphs are split into sentences.
- **Keep sentences the app could receive.** A sentence is kept at 30 to 300 characters for Wikinews and 10 to 300 for Tatoeba. It must start with a capital or digit, end with a full stop, question or exclamation mark, and have balanced quotes, which drops sentences cut out of a longer quotation. Exact duplicates are dropped.
- **Leave FLORES's own articles out.** The FLORES archive lists the page behind every sentence. The 207 Wikinews articles among them are skipped whole, before the sentence-level leak check runs as well.
- **Draw a sample at load.** The stored file is exactly what goes to the teacher. Sentences are ordered by a seeded hash of their text, so the draw does not depend on file order, and a larger draw keeps every sentence of a smaller one. At most two come from any one article, to spread the sample over more stories.

**First load, 5,000 sentences from each source.**

| | Wikinews | Tatoeba |
|---|---|---|
| Fit to translate | 263,853 | 2,073,079 |
| FLORES articles skipped | 201 of 207. The other 6 are unpublished, so skipped anyway | |
| Drawn sentences that leaked into FLORES | 0 | 0 |
| Mean length | 128 characters | 39 characters |
| Articles drawn from | 4,306 | |

Both checks earn their place. Without skipping articles, about 640 Wikinews sentences match FLORES text, from 196 of its 207 articles. 614 of them are in those articles, and the other 27 are in other articles that repeat the same story, which only the sentence check catches.

Wikinews drops most for unbalanced quotes, 23,050 sentences, nearly all cut from quotations that run over several sentences. Tatoeba drops 63,229 exact duplicates.

**Consequences.** The next step translates the 10,000 sentences with DeepSeek, stored as synthetic events. The event schema needs a signal kind for a teacher's answer, because `reference` means a human translation. Curation's spam and leak checks then apply as for any event, and a native speaker should read a sample before training. Every data card already shows the human and synthetic share.

Tatoeba's English has a house style: 26% of the draw names Tom or Mary, and 7% uses one contributor's recurring names, such as Yanni and Ziri. Names pass through translation, so this skews little, but the share is worth capping if Tatoeba grows.

Both downloads are pinned to dated files. Wikimedia removes dated dumps after a few months, so a fresh clone will need a new pin then. A cached file keeps working.

**Sources.**

- [Gemini API additional terms](https://ai.google.dev/gemini-api/terms)
- [DeepSeek Open Platform terms of service](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html) and [terms of use](https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html)
- [DeepSeek V4.1 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) and [Gemma 4 31B model card](https://huggingface.co/google/gemma-4-31b-it)
- [OpenRouter terms of service](https://openrouter.ai/terms)
- [Anthropic commercial terms](https://www.anthropic.com/legal/commercial-terms)
- [Wikinews licence update](https://meta.wikimedia.org/wiki/Requests_for_comment/Update_standard_license_for_Wikinews) and [the Signpost on Wikinews closing](https://signpost.news/2026-05-22/Special_report)
- [English Wikinews dump, 2026-11-01](https://dumps.wikimedia.org/enwikinews/20261101/)
- [Tatoeba: terms of use](https://tatoeba.org/eng/terms_of_use), and its [English sentences through OPUS](https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/mono/en.txt.gz)
- [Wikivoyage copyleft](https://en.wikivoyage.org/wiki/Wikivoyage:Copyleft) and [VOA terms of use](https://www.voanews.com/terms-use-and-privacy-notice)
- [FLORES repository](https://github.com/openlanguagedata/flores)
