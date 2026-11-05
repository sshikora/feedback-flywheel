"""A data card per snapshot, generated from its manifest so the two cannot disagree."""

from typing import Any

_KINDS = {
    "sft": ("SFT examples", "an input and the answer to learn"),
    "dpo": ("DPO pairs", "an input, a chosen answer and a rejected one"),
}


def render_card(meta: dict[str, Any]) -> str:
    counts = meta["counts"]
    name = f"{meta['kind']}-{meta['hash'][:12]}"
    label, record = _KINDS[meta["kind"]]
    lines = [
        f"# Data card: {name}",
        "",
        f"- **Contents:** {counts['examples']} {label}, from {counts['events']} curated events.",
        f"- **Each record:** {record}.",
        f"- **Hash:** `{meta['hash']}`",
        f"- **Built:** {meta['created_at']}",
        f"- **Built from:** `{meta['source']['path']}`, sha256 `{meta['source']['sha256']}`",
        f"- **Origin:** {counts['origin']['human']} human and "
        f"{counts['origin']['synthetic']} synthetic events.",
        "",
        "## Composition",
        "",
        _table("Source", counts["sources"]),
        _table("Language pair", counts["language_pairs"]),
    ]
    if meta["kind"] == "sft":
        lines.append(_table("Who wrote the target", counts["target_models"]))
    else:
        lines.append(_table("Model that wrote the chosen answer", counts["chosen_models"]))
        lines.append(_table("Model that wrote the rejected answer", counts["rejected_models"]))
    lines += [
        "## Filters",
        "",
        _table("Setting", {k: _plain(v) for k, v in meta["config"].items()}, "Value"),
        _table("Dropped", counts["dropped"]) if counts["dropped"] else "Nothing was dropped.\n",
        "## Known limits",
        "",
        *_limits(meta),
        "",
    ]
    if meta["licenses"]:
        lines += ["## Licenses", "", *_licenses(meta["licenses"]), ""]
    return "\n".join(lines)


def _limits(meta: dict[str, Any]) -> list[str]:
    counts = meta["counts"]
    limits = []
    if counts["votes"] and counts["blind_share"] < 1:
        unblind = round((1 - counts["blind_share"]) * 100)
        limits.append(
            f"- {unblind}% of votes ({counts['votes']}) were cast with model names visible, "
            "so they carry brand bias. They are training signal, never evaluation."
        )
    if counts["contested_events"]:
        limits.append(
            f"- {counts['contested_events']} events are contested: voters on the same input "
            "chose different answers, or a corpus gave more than one translation."
        )
    if meta["kind"] == "dpo" and counts["close_pairs"]:
        limits.append(
            f"- {round(counts['close_share'] * 100)}% of pairs are close, with chrF++ of at "
            f"least {meta['config']['close_pair_chrf']} between chosen and rejected. In those, "
            "the rejected answer is often also correct, so the pair is a weak signal."
        )
    return limits or ["- None recorded."]


def _licenses(corpora: dict[str, dict[str, str]]) -> list[str]:
    return [
        f"- **{source}:** {corpus['license']}. {corpus['attribution']}"
        for source, corpus in corpora.items()
    ]


def _table(heading: str, values: dict[str, Any], column: str = "Count") -> str:
    rows = "\n".join(f"| {key} | {value} |" for key, value in values.items())
    return f"| {heading} | {column} |\n|---|---|\n{rows}\n" if values else ""


def _plain(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(value) if value else "none"
    return str(value)
