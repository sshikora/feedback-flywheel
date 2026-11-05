"""The translation app's prompt, shared by evaluation and training.

Matches `buildPlainPrompt` and `toneInstruction` in the app's
providers/_enriched.ts, so a model is scored on, and trained with, the text the
app actually sends. When the user leaves the source language on auto, the app
asks for JSON with the detected language, and a trained model must answer that way.
"""

import json

# App language codes to the names its prompt uses.
LANGUAGE_NAMES = {"en": "English", "ceb": "Cebuano"}

_TONES = {
    "casual": " Use a casual, conversational tone.",
    "formal": " Use a formal, professional tone.",
}


def translation_prompt(
    text: str, tgt_lang: str, src_lang: str | None, tone: str | None = None
) -> str:
    """The prompt for a known source language, or the detection prompt for None."""
    target = LANGUAGE_NAMES[tgt_lang]
    suffix = _TONES.get(tone or "", "")
    if src_lang is not None:
        return (
            f"Translate the following text from {LANGUAGE_NAMES[src_lang]} to {target}."
            f"{suffix} Return only the translated text with no explanation or commentary."
            f"\n\n{text}"
        )
    return (
        f"Translate the following text to {target} and identify the input's language.{suffix} "
        "Return ONLY a single JSON object (no prose, no markdown fences) with this shape:\n"
        "\n"
        "{\n"
        f'  "translatedText": "<translation in {target}>",\n'
        '  "detectedSourceLanguage": '
        '"<ISO 639-1 code of the input language, e.g. en, fr, ja, zh>"\n'
        "}\n"
        "\n"
        "Text to translate:\n"
        f"{text}"
    )


def detection_answer(translation: str, detected: str) -> str:
    """The answer the app expects to the detection prompt."""
    return json.dumps(
        {"translatedText": translation, "detectedSourceLanguage": detected}, ensure_ascii=False
    )
