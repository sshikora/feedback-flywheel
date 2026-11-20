"""Write snapshots in the provider's training format.

Each example is one turn: the app's prompt as the user message, and the answer
the app expects as the assistant message. Loss falls on the answer only.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flywheel.prompts import LANGUAGE_NAMES, detection_answer, translation_prompt
from flywheel.schema import DpoPair, SftExample


class Unformattable(Exception):
    """This example cannot be written in the app's prompt format."""


@dataclass(frozen=True)
class TrainingFile:
    path: Path
    lines: int
    sha256: str
    skipped: Counter[str]


def sft_line(example: SftExample) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": _prompt(example)},
            {"role": "assistant", "content": _answer(example, example.target)},
        ]
    }


def dpo_line(pair: DpoPair) -> dict[str, Any]:
    return {
        "input": {"messages": [{"role": "user", "content": _prompt(pair)}]},
        "preferred_output": [{"role": "assistant", "content": _answer(pair, pair.chosen)}],
        "non_preferred_output": [{"role": "assistant", "content": _answer(pair, pair.rejected)}],
    }


def write_training_file(examples: Sequence[SftExample | DpoPair], path: Path) -> TrainingFile:
    lines: list[str] = []
    skipped: Counter[str] = Counter()
    for example in examples:
        try:
            record = sft_line(example) if isinstance(example, SftExample) else dpo_line(example)
        except Unformattable as err:
            skipped[str(err)] += 1
            continue
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    content = "".join(lines).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return TrainingFile(path, len(lines), hashlib.sha256(content).hexdigest(), skipped)


def _prompt(example: SftExample | DpoPair) -> str:
    languages = [example.tgt_lang] if example.auto_detect else [example.src_lang, example.tgt_lang]
    if any(code not in LANGUAGE_NAMES for code in languages):
        raise Unformattable("unsupported_language")
    source = None if example.auto_detect else example.src_lang
    return translation_prompt(example.source, example.tgt_lang, source, example.tone)


def _answer(example: SftExample | DpoPair, text: str) -> str:
    if not example.auto_detect:
        return text
    if example.src_lang == "und":
        raise Unformattable("unknown_source_language")
    return detection_answer(text, example.src_lang)
