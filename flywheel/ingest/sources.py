"""Load English source text for a teacher model to translate.

Only sources that passed the license check in docs/decisions/0017 are listed. Each
download is pinned by hash and read without extracting it. A source sentence is not a
training pair: it becomes one only once a teacher translates it.

A sample is drawn at load, not later, so the stored file is exactly what will be sent
to the teacher. The draw orders sentences by a seeded hash of their text, so it does
not depend on file order, and growing the sample keeps every sentence already drawn.
Sentences that leak into the FLORES eval sets are dropped before they count.
"""

import bz2
import gzip
import hashlib
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote

import mwparserfromhell

from flywheel.curate.leakage import LeakageIndex
from flywheel.schema import SourceText
from flywheel.text import normalize_text

MAX_CHARS = 300  # the app's input limit
MAX_PER_ARTICLE = 2  # spreads a sample over more stories


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    archive_name: str
    sha256: str
    released: datetime
    license: str
    attribution: str
    min_chars: int


SOURCES = {
    "wikinews": Source(
        name="wikinews",
        url="https://dumps.wikimedia.org/enwikinews/20261101/"
        "enwikinews-20261101-pages-articles.xml.bz2",
        archive_name="enwikinews-20261101-pages-articles.xml.bz2",
        sha256="2afeb4605bf9bafe7dc099c308980012dcaeab440ffa4145fba4305f8259f4fa",
        released=datetime(2026, 11, 1, tzinfo=UTC),
        license="CC BY 2.5, and CC BY 4.0 for text from 2024-12-16 on. Text from before "
        "2005-09-25 is in the public domain",
        attribution="News articles by the contributors to English Wikinews "
        "(en.wikinews.org), from the 2026-11-01 dump. Each sentence keeps its article title.",
        min_chars=30,
    ),
    "tatoeba": Source(
        name="tatoeba",
        url="https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/mono/en.txt.gz",
        archive_name="Tatoeba-v2026-07-08.en.txt.gz",
        sha256="c346ffae6848cd92d3037c1437f1befba66882c3f38bc999b7fa502feea4df12",
        released=datetime(2026, 7, 8, tzinfo=UTC),
        license="CC BY 2.0 FR",
        attribution="Sentences by the contributors to tatoeba.org. "
        "Distributed through OPUS (Tiedemann, 2012).",
        min_chars=10,
    ),
}

# Templates that stand for their own text. {{w|Page|shown}} links to Wikipedia.
_INLINE_TEMPLATES = {"w"}
# Links to these namespaces are media, categories or other wikis, not words in a sentence.
_NOT_TEXT = re.compile(r"^\s*:?\s*(file|image|media|category|[a-z]{2,3}(-[a-z]+)?):", re.I)
# Published articles carry {{publish}}. Drafts never passed review and may hold pasted text.
_PUBLISHED = re.compile(r"\{\{\s*publish(ed)?\s*[|}]", re.I)
# Everything after these headings is a list of sources and links.
_BACK_MATTER = re.compile(
    r"^==+\s*(sources?|related news|external links|see also|references|sister links)\s*==+",
    re.I | re.M,
)
_NOT_PROSE = ("=", "*", "#", ":", ";", "{|", "|", "!")
# Tags whose contents are footnotes, media or code, not part of the sentence around them.
_HIDDEN_TAGS = frozenset(
    "ref references gallery imagemap math chem timeline score syntaxhighlight source "  # noqa: SIM905
    "pre code table poem".split()
)

# Straight and curly quotes, and brackets, that open or close a sentence.
_OPENING = '"\u201c\u2018('
_CLOSING = "\"'\u201d\u2019)"
# A sentence ends at . ! or ? and any closing quote or bracket, before a capital.
_BOUNDARY = re.compile(rf"(?<=[.!?])([{_CLOSING}]*)\s+(?=[{_OPENING}]?[A-Z0-9])")
# Words that end in a full stop without ending a sentence.
_ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof st jr sr gen gov sen rep lt col sgt capt cmdr adm maj rev hon pres "  # noqa: SIM905
    "inc ltd co corp no vs etc approx est dept univ jan feb mar apr jun jul aug sep sept oct "
    "nov dec mt ft fig op vol".split()
)
_LEFTOVER = re.compile(r"[{}\[\]|<>=_\\]|https?:|www\.|&\w+;")
_STARTS = re.compile(rf"^[{_OPENING}]?[A-Z0-9]")
_ENDS = re.compile(rf"[.!?][{_CLOSING}]*$")


@dataclass(frozen=True)
class SourceImport:
    texts: list[SourceText]
    skipped: Counter[str]
    available: int


def source(name: str) -> Source:
    if name not in SOURCES:
        raise ValueError(f"unknown source {name!r}, expected one of {sorted(SOURCES)}")
    return SOURCES[name]


def fetch_archive(name: str, cache_dir: Path) -> Path:
    spec = source(name)
    archive = cache_dir / "sources" / spec.archive_name
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(archive.name + ".part")
        with urllib.request.urlopen(spec.url, timeout=120) as response, partial.open("wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        os.replace(partial, archive)
    _verify(archive, spec)
    return archive


def _verify(archive: Path, spec: Source) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise ValueError(f"{archive} has sha256 {digest.hexdigest()}, expected {spec.sha256}")


def wikinews_title(url: str) -> str | None:
    """The article title in an en.wikinews.org link, as the dump writes it."""
    prefix = "https://en.wikinews.org/wiki/"
    if not url.startswith(prefix):
        return None
    return unquote(url.removeprefix(prefix)).replace("_", " ")


def load(
    name: str,
    archive: Path,
    size: int,
    seed: str,
    leak_index: LeakageIndex,
    exclude_documents: Iterable[str] = (),
) -> SourceImport:
    """Draw `size` sentences that do not leak, skipping the excluded documents."""
    spec = source(name)
    skipped: Counter[str] = Counter()
    excluded = set(exclude_documents)
    found: dict[str, SourceText] = {}
    raw = _wikinews(archive, excluded, skipped) if name == "wikinews" else _tatoeba(archive)
    for text in raw:
        reason = sentence_problem(text.text, spec.min_chars)
        if reason is not None:
            skipped[reason] += 1
            continue
        key = normalize_text(text.text)
        if key in found:
            skipped["duplicate"] += 1
            continue
        found[key] = text

    drawn: list[SourceText] = []
    per_document: Counter[str] = Counter()
    for key in sorted(found, key=lambda key: _rank(seed, key)):
        if len(drawn) == size:
            break
        text = found[key]
        if text.document is not None and per_document[text.document] >= MAX_PER_ARTICLE:
            continue
        if leak_index.matches(text.text):
            skipped["leaks into FLORES"] += 1
            continue
        per_document[text.document or ""] += 1
        drawn.append(text)
    return SourceImport(drawn, skipped, available=len(found))


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{key}".encode()).hexdigest()


def sentence_problem(text: str, min_chars: int) -> str | None:
    """Why a sentence is unfit to translate, or None when it is fit."""
    if len(text) < min_chars:
        return "too short"
    if len(text) > MAX_CHARS:
        return "too long"
    if _LEFTOVER.search(text):
        return "markup"
    if not _STARTS.search(text) or not _ENDS.search(text):
        return "not a whole sentence"
    if text.count('"') % 2 or text.count("\u201c") != text.count("\u201d"):
        return "unbalanced quotes"
    return None


def _tatoeba(archive: Path) -> Iterator[SourceText]:
    with gzip.open(archive, "rt", encoding="utf-8") as lines:
        for index, line in enumerate(lines):
            yield SourceText(
                id=f"tatoeba:v2026-07-08:en:{index}",
                corpus="tatoeba",
                document=None,
                text=line.strip(),
            )


def _wikinews(archive: Path, excluded: set[str], skipped: Counter[str]) -> Iterator[SourceText]:
    for page_id, title, wikitext in _articles(archive):
        if not _PUBLISHED.search(wikitext):
            skipped["unpublished articles"] += 1
            continue
        if title in excluded:
            skipped["FLORES articles"] += 1
            continue
        for index, sentence in enumerate(split_sentences(article_text(wikitext))):
            yield SourceText(
                id=f"wikinews:20261101:{page_id}:{index}",
                corpus="wikinews",
                document=title,
                text=sentence,
            )


def _articles(archive: Path) -> Iterator[tuple[str, str, str]]:
    """Main-namespace pages that are not redirects: id, title and wikitext."""
    with bz2.open(archive) as stream:
        for _, element in ET.iterparse(stream):
            if not element.tag.endswith("}page"):
                continue
            page = {child.tag.rpartition("}")[2]: child.text or "" for child in element}
            if page.get("ns") == "0" and "redirect" not in page:
                text = next((e.text for e in element.iter() if e.tag.endswith("}text")), None)
                yield page["id"], page["title"], text or ""
            element.clear()


def article_text(wikitext: str) -> str:
    """The prose of an article, one paragraph per line, without markup."""
    body = _BACK_MATTER.split(wikitext, maxsplit=1)[0]
    prose = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith(_NOT_PROSE)
    )
    code = mwparserfromhell.parse(prose)
    for template in code.filter_templates(recursive=False):
        name = str(template.name).strip().lower()
        params = [p for p in template.params if not p.showkey]
        if name in _INLINE_TEMPLATES and params:
            code.replace(template, str(params[-1].value).strip())
        else:
            code.remove(template)
    for tag in code.filter_tags(recursive=False):
        if str(tag.tag).strip().lower() in _HIDDEN_TAGS:
            code.remove(tag)
    for link in code.filter_wikilinks(recursive=False):
        if _NOT_TEXT.match(str(link.title)):
            code.remove(link)
    text = code.strip_code(normalize=True, collapse=True)
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def split_sentences(text: str) -> list[str]:
    """Split each paragraph at sentence ends, keeping abbreviations and initials whole."""
    sentences: list[str] = []
    for paragraph in text.splitlines():
        start = 0
        for boundary in _BOUNDARY.finditer(paragraph):
            end = boundary.end(1)
            last_word = paragraph[start:end].rstrip(_CLOSING).rsplit(maxsplit=1)[-1]
            stem = last_word.rstrip(".!?").lower()
            if last_word.endswith(".") and (
                stem in _ABBREVIATIONS or (len(stem) == 1 and stem.isalpha()) or "." in stem
            ):
                continue
            sentences.append(paragraph[start:end].strip())
            start = boundary.end()
        sentences.append(paragraph[start:].strip())
    return [sentence for sentence in sentences if sentence]
