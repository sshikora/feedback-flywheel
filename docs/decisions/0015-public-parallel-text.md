# 0015: Public parallel text for the first training data

**Status:** Accepted, 2026-11-05

**Context.** The plan's first training data is public parallel text, with the license checked for each corpus and nothing derived from Cebuano Wikipedia. Votes cannot train until the privacy page names Together (`0009`). The model may serve in the app and its weights may be published, so the license must allow commercial use and derived works. The translations must also be written by people. Mined or machine-made pairs would teach the errors the model is meant to fix.

OPUS lists 11 English–Cebuano corpora. Each was checked on its OPUS page and its own package files on 2026-11-05.

| Corpus | Pairs | License | Origin | Verdict |
|---|---|---|---|---|
| NLLB | 14.8M | Not checked | Mined from the web by sentence embeddings | No: machine-aligned, and Cebuano web text is largely bot-written Wikipedia |
| CCMatrix | 963k | Not checked | Mined from Common Crawl | No: as for NLLB |
| XLEnt | 208k | Not checked | Named-entity pairs | No: names, not sentences |
| bible-uedin | 62k | CC0 1.0. The King James Version, World English Bible and 1917 *Ang Biblia* texts are public domain | Human translations | Yes, but kept out of the first build |
| WikiMatrix | 29k | Not checked | Mined from Wikipedia | No: Cebuano Wikipedia |
| wikimedia | 6.9k | CC BY-SA 4.0 | Wikipedia's Content Translation tool, which starts from machine translation | No: Cebuano Wikipedia |
| translatewiki | 2.4k | CC BY 3.0 | Volunteer translations of software interfaces | Yes |
| QED | 522 | Research only | Subtitles | No: research only |
| Tatoeba | 424 | CC BY 2.0 FR | Volunteer sentence translations | Yes |
| TED2020 | 100 | TED's usage policy: CC BY-NC-ND | Subtitles | No: non-commercial, no derived works |
| Ubuntu | None listed | Not checked | Software interface | No pairs to load |

Corpora rejected on origin were not license-checked. Google's SMOL is professionally translated and CC BY 4.0, but it has no Cebuano.

**Decision.**

- **Load Tatoeba, translatewiki and bible-uedin**, with `flywheel ingest opus <corpus>`. Each download is pinned by sha256 and read from the zip without extracting it, as FLORES is.
- **A corpus pair is a reference, not a vote.** It becomes a feedback event with a new signal kind, `reference`, whose one candidate is the human translation. Curation treats it like any event, so the FLORES leak check, dedupe and the held-out split all apply. Vote statistics and the brand-bias note leave references out. They train SFT and make no DPO pairs, because there is no rejected answer.
- **Drop markup at load.** Interface strings carry wiki markup, HTML and placeholders such as `$1`, and a model trained on them would emit them. Any pair containing them is skipped and counted. That removes 414 of 2,355 translatewiki pairs, and 117 bible-uedin pairs, which are Psalm headings in escaped angle brackets.
- **Credit every corpus.** CC BY needs attribution. Each snapshot's manifest and data card list the license and attribution of every corpus in it, and `ingest opus` prints them.
- **Keep the Bible out of the first build**, with `exclude_sources = ["opus-bible-uedin"]` in the dataset config. Its register is far from what app users type: 19% of its English lines use "thee", "thou", "hath" or "unto", and its Cebuano is from 1917. It would also be 96% of the data. Whether it helps is a question for a second run, judged at the gate, and not for the first.
- **Make the leak check scale.** Comparing every text that shares a trigram could not finish on 64,000 events: a Bible verse shared a trigram with 1,520 of the 4,018 FLORES texts, and the held-out comparison grows with the square of the event count. The index now uses the size, prefix and position filters from the set similarity join literature (AllPairs and PPJoin). They are exact, and a test checks that the index finds the same matches as comparing everything, at three thresholds. They cut a verse's candidates from about 1,500 to about 120, and all three corpora now curate in about 10 minutes.

**First curation.** All three corpora, 64,420 events, checked against FLORES dev and devtest:

| Source | Loaded | In training after curation |
|---|---|---|
| Tatoeba | 424 | 335 |
| translatewiki | 1,941 | 1,446 |
| bible-uedin | 62,055 | 47,760 |

No corpus text matched FLORES. 6,411 events were held out, and 7,079 were dropped as near-duplicates of held-out text. Most of those are Bible verses, because each Cebuano verse appears twice, once for each English version, so when one copy is held out the other is caught. 1,268 were exact duplicates and 121 were rejected as spam, mostly interface strings left untranslated.

**Consequences.** Without the Bible, the cleared public text is 1,781 training pairs, mostly short. That is enough for a first run, and under Together's $4 minimum, but not enough to teach much. The volume has to come from the plan's later sources: synthetic pairs from teachers whose terms allow training on their outputs, then votes and rater corrections.

Quality is unvetted. Tatoeba's translations are loose in places ("Wait for a few seconds." is "Wetsa."), and at least one translatewiki label is wrong: "prev" is given as "kataposan", which means "last". A native speaker should spot-check a sample before the first run.

Tatoeba has 424 pairs over 391 English sentences. Dedupe keeps one translation per sentence, so alternative translations are lost and the sentence is marked contested.

**Sources.**

- [OPUS corpus listing for ceb-en](https://opus.nlpl.eu/opusapi/?source=ceb&target=en&preprocessing=moses&version=latest)
- OPUS pages for [Tatoeba](https://opus.nlpl.eu/legacy/Tatoeba.php), [translatewiki](https://opus.nlpl.eu/legacy/translatewiki.php), [bible-uedin](https://opus.nlpl.eu/legacy/bible-uedin.php), [QED](https://opus.nlpl.eu/legacy/QED.php), [TED2020](https://opus.nlpl.eu/legacy/TED2020.php) and [wikimedia](https://opus.nlpl.eu/legacy/wikimedia.php)
- [Tatoeba: terms of use](https://tatoeba.org/eng/terms_of_use)
- [translatewiki.net: about](https://translatewiki.net/wiki/Project:About)
- [Christodouloupoulos and Steedman: the Bible corpus](https://github.com/christos-c/bible-corpus)
- [Google SMOL](https://huggingface.co/datasets/google/smol)
- Bayardo, Ma and Srikant (2007), *Scaling up all pairs similarity search*, and Xiao et al. (2011), *Efficient similarity joins for near-duplicate detection*
