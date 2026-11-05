"""Load human-translated parallel text from OPUS as training events.

Only corpora that passed the license and origin check in docs/decisions/0015 are
listed. Each download is pinned by hash and read straight from the zip, so nothing is
extracted to disk. A corpus pair becomes an event with a reference signal: one
candidate, the human translation, which nobody voted on.
"""

import hashlib
import os
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from flywheel.schema import (
    Candidate,
    Consent,
    EventContext,
    FeedbackEvent,
    ReferenceSignal,
    TranslationInput,
)

PAIR = ("ceb", "en")

# Wiki markup, HTML, placeholders and escapes. Interface strings carry them, and a
# model trained on them would learn to emit them.
_MARKUP = re.compile(r"<[^>]*>|\{\{|\}\}|\[\[|\]\]|''|\$\d|&\w+;|%[sd]|\\n")


@dataclass(frozen=True)
class Corpus:
    opus_name: str
    version: str
    released: datetime
    sha256: str
    license: str
    attribution: str

    @property
    def url(self) -> str:
        return (
            f"https://object.pouta.csc.fi/OPUS-{self.opus_name}/{self.version}"
            f"/moses/{'-'.join(PAIR)}.txt.zip"
        )

    @property
    def archive_name(self) -> str:
        return f"{self.opus_name}-{self.version}.{'-'.join(PAIR)}.zip"


_OPUS = "Distributed through OPUS (Tiedemann, 2012)."

CORPORA = {
    "tatoeba": Corpus(
        opus_name="Tatoeba",
        version="v2026-07-08",
        released=datetime(2026, 7, 8, tzinfo=UTC),
        sha256="6a57dba2cb15cd4147d8e86f3a05ac077dea48aa7213e0c3a8139614bda6f580",
        license="CC BY 2.0 FR",
        attribution=f"Sentences and translations by the contributors to tatoeba.org. {_OPUS}",
    ),
    "translatewiki": Corpus(
        opus_name="translatewiki",
        version="v2026-07-01",
        released=datetime(2026, 7, 1, tzinfo=UTC),
        sha256="6e8dc56b4a0b12d1b14582c2a5211636f95c99d31719f38dca542144ff8ca7b3",
        license="CC BY 3.0",
        attribution=f"Interface translations by the translators at translatewiki.net. {_OPUS}",
    ),
    "bible-uedin": Corpus(
        opus_name="bible-uedin",
        version="v1",
        released=datetime(2019, 8, 9, tzinfo=UTC),
        sha256="d1961077237513281df997aeff87e25b40a48f957349fa2e2bc4696764a4b788",
        license="CC0 1.0; the King James Version, World English Bible and 1917 Ang Biblia "
        "texts are in the public domain",
        attribution="Aligned by Christodouloupoulos and Steedman (2015), A massively parallel "
        f"corpus: the Bible in 100 languages. {_OPUS}",
    ),
}


@dataclass(frozen=True)
class CorpusImport:
    events: list[FeedbackEvent]
    skipped: Counter[str]


def source_name(name: str) -> str:
    return f"opus-{name}"


def licenses(sources: dict[str, int]) -> dict[str, dict[str, str]]:
    """License and attribution for every OPUS corpus among these event sources."""
    by_source = {source_name(name): corpus for name, corpus in CORPORA.items()}
    return {
        source: {"license": corpus.license, "attribution": corpus.attribution, "url": corpus.url}
        for source, corpus in sorted(by_source.items())
        if source in sources
    }


def corpus(name: str) -> Corpus:
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}, expected one of {sorted(CORPORA)}")
    return CORPORA[name]


def fetch_archive(name: str, cache_dir: Path) -> Path:
    spec = corpus(name)
    archive = cache_dir / "opus" / spec.archive_name
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(archive.name + ".part")
        with urllib.request.urlopen(spec.url, timeout=120) as response:
            partial.write_bytes(response.read())
        os.replace(partial, archive)
    _verify(archive, spec)
    return archive


def _verify(archive: Path, spec: Corpus) -> None:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != spec.sha256:
        raise ValueError(f"{archive} has sha256 {digest}, expected {spec.sha256}")


def load_events(archive: Path, name: str, src: str = "en", tgt: str = "ceb") -> CorpusImport:
    spec = corpus(name)
    if {src, tgt} != set(PAIR):
        raise ValueError(f"{name} is pinned for {'-'.join(PAIR)}, not {src}-{tgt}")
    with zipfile.ZipFile(archive) as zipped:
        sources = _read_lines(zipped, spec, src)
        targets = _read_lines(zipped, spec, tgt)
    if len(sources) != len(targets):
        raise ValueError(f"misaligned {name}: {len(sources)} vs {len(targets)} lines")

    reference = f"human:{source_name(name)}"
    events: list[FeedbackEvent] = []
    skipped: Counter[str] = Counter()
    for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
        source, target = source.strip(), target.strip()
        if not source or not target:
            skipped["empty"] += 1
            continue
        if _MARKUP.search(source) or _MARKUP.search(target):
            skipped["markup"] += 1
            continue
        events.append(
            FeedbackEvent(
                id=f"opus:{name}:{spec.version}:{src}-{tgt}:{index}",
                source=source_name(name),
                origin="human",
                created_at=spec.released,
                input=TranslationInput(text=source, src_lang=src, tgt_lang=tgt),
                candidates=[Candidate(label="reference", model_id=reference, output=target)],
                signal=ReferenceSignal(chosen=[reference]),
                context=EventContext(blind=True),
                consent=Consent(training=True, publish=True),
            )
        )
    return CorpusImport(events, skipped)


def _read_lines(zipped: zipfile.ZipFile, spec: Corpus, lang: str) -> list[str]:
    return zipped.read(f"{spec.opus_name}.{'-'.join(PAIR)}.{lang}").decode("utf-8").splitlines()
