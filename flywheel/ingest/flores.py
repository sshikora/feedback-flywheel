"""Load FLORES-200 reference pairs for evaluation.

FLORES-200 is CC-BY-SA 4.0, so it is downloaded at runtime and never committed.
Its download is pinned by hash. Files are read straight from the archive by exact
name, so nothing in the archive is ever extracted to disk.
"""

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

from flywheel.schema import EvalSegment

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
ARCHIVE_NAME = "flores200_dataset.tar.gz"
SPLITS = ("dev", "devtest")

# App language codes to FLORES codes. Add a line to support another language.
FLORES_CODES = {"en": "eng_Latn", "ceb": "ceb_Latn"}


def fetch_archive(cache_dir: Path) -> Path:
    archive = cache_dir / ARCHIVE_NAME
    if not archive.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(archive.name + ".part")
        with urllib.request.urlopen(FLORES_URL, timeout=120) as response:
            partial.write_bytes(response.read())
        os.replace(partial, archive)
    _verify(archive)
    return archive


def _verify(archive: Path) -> None:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != FLORES_SHA256:
        raise ValueError(f"{archive} has sha256 {digest}, expected {FLORES_SHA256}")


def load_segments(archive: Path, src: str, tgt: str, split: str) -> list[EvalSegment]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    for code in (src, tgt):
        if code not in FLORES_CODES:
            raise ValueError(f"no FLORES code for {code!r}, add it to FLORES_CODES")

    with tarfile.open(archive, "r:gz") as tar:
        sources = _read_lines(tar, FLORES_CODES[src], split)
        references = _read_lines(tar, FLORES_CODES[tgt], split)
    if len(sources) != len(references):
        raise ValueError(f"misaligned {split}: {len(sources)} vs {len(references)} lines")

    return [
        EvalSegment(
            id=f"flores200:{split}:{src}-{tgt}:{index}",
            corpus="flores200",
            split=split,
            src_lang=src,
            tgt_lang=tgt,
            source=source,
            reference=reference,
        )
        for index, (source, reference) in enumerate(zip(sources, references, strict=True))
    ]


def _read_lines(tar: tarfile.TarFile, code: str, split: str) -> list[str]:
    member = tar.extractfile(f"./flores200_dataset/{split}/{code}.{split}")
    if member is None:
        raise ValueError(f"{code}.{split} is not a regular file")
    return member.read().decode("utf-8").splitlines()
