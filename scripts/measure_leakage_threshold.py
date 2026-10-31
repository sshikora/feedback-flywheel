"""Measure how well each near-duplicate threshold catches edited eval sentences.

Recall: random word replacements on devtest sentences, which should all match.
False matches: dev sentences checked against devtest, which are unrelated text.

Usage: uv run python scripts/measure_leakage_threshold.py [data_dir]
Expects `flywheel ingest flores` to have run for both the dev and devtest splits.
"""

import random
import sys
from pathlib import Path

from flywheel.curate.leakage import LeakageIndex
from flywheel.schema import EvalSegment
from flywheel.storage import read_jsonl

THRESHOLDS = (0.9, 0.8, 0.7, 0.6)
EDITS = (1, 2, 3)
SAMPLE_SIZE = 300
SEED = 7
REPLACEMENTS = ("market", "river", "people", "merkado", "suba", "tawo")


def replace_words(text: str, count: int, rng: random.Random) -> str:
    words = text.split()
    for _ in range(count):
        words[rng.randrange(len(words))] = rng.choice(REPLACEMENTS)
    return " ".join(words)


def main(data_dir: Path) -> None:
    dev = read_jsonl(data_dir / "eval" / "flores200-dev-en-ceb.jsonl", EvalSegment)
    devtest = read_jsonl(data_dir / "eval" / "flores200-devtest-en-ceb.jsonl", EvalSegment)
    rng = random.Random(SEED)
    sample = rng.sample(devtest, SAMPLE_SIZE)

    print("threshold | recall at 1, 2, 3 edits | false matches")
    for threshold in THRESHOLDS:
        index = LeakageIndex((segment.source for segment in devtest), threshold)
        recall = [
            sum(index.matches(replace_words(s.source, edits, rng)) for s in sample) / SAMPLE_SIZE
            for edits in EDITS
        ]
        false_matches = sum(index.matches(segment.source) for segment in dev)
        cells = ", ".join(f"{value:.1%}" for value in recall)
        print(f"{threshold} | {cells} | {false_matches}/{len(dev)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data"))
