"""Turn stored outputs into results. Needs no API access, so it can always be rerun."""

import html
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flywheel.evals import metrics
from flywheel.evals.config import EvalConfig
from flywheel.evals.judge import choose_segments, score_pair
from flywheel.evals.metrics import Interval
from flywheel.evals.rating import trusted_ratings
from flywheel.schema import EvalSegment, HumanRating, JudgeVerdict, TranslationRecord

WORST_SENTENCES = 10
TRAITS = ("adequacy", "fluency")


@dataclass(frozen=True)
class EvalInputs:
    config: EvalConfig
    segments: list[EvalSegment]
    translations: dict[tuple[str, str], TranslationRecord]
    verdicts: dict[tuple[str, str, str], JudgeVerdict] = field(default_factory=dict)
    ratings: list[HumanRating] = field(default_factory=list)


def _interval_json(interval: Interval) -> dict[str, float]:
    return {
        "value": round(interval.value, 3),
        "low": round(interval.low, 3),
        "high": round(interval.high, 3),
    }


def stored_outputs(inputs: EvalInputs) -> dict[tuple[str, str], str]:
    return {key: r.output for key, r in inputs.translations.items() if r.output is not None}


def _judge_rows(
    inputs: EvalInputs, covered: list[EvalSegment], outputs: Mapping[tuple[str, str], str]
) -> list[dict[str, Any]]:
    """Samples from the covered sentences, exactly as `flywheel eval judge` does."""
    config = inputs.config
    sample = choose_segments(covered, config.judge.sentences, config.seed)
    rows = []
    for base in config.model_ids("base"):
        for teacher in config.model_ids("compare"):
            pair = score_pair(sample, base, teacher, dict(outputs), inputs.verdicts)
            if not pair.scores:
                continue
            scores = np.array(pair.scores)
            weights = metrics.resample_weights(
                len(scores), config.resamples, f"{config.seed}|judge|{base}|{teacher}"
            )
            rows.append(
                {
                    "base": base,
                    "compare": teacher,
                    "base_win_rate": _interval_json(metrics.mean_interval(scores, weights)),
                    "sentences": len(scores),
                    "split_orders": pair.split_orders,
                    "unusable": pair.unusable,
                }
            )
    return rows


def _human_section(inputs: EvalInputs) -> dict[str, Any] | None:
    if not inputs.ratings:
        return None
    settings = inputs.config.rating
    trusted = trusted_ratings(inputs.ratings, settings.min_check_pass)
    scored = [r for r in trusted if r.check is None]
    if not scored:
        return {"models": [], "excluded_rounds": _excluded(inputs.ratings, trusted)}

    by_sentence: dict[tuple[str, str], list[HumanRating]] = defaultdict(list)
    for rating in scored:
        by_sentence[(rating.model_id, rating.segment_id)].append(rating)

    models = []
    for model in sorted({r.model_id for r in scored}):
        sentences = [key for key in by_sentence if key[0] == model]
        row: dict[str, Any] = {"model": model, "sentences": len(sentences)}
        for trait in TRAITS:
            values = np.array(
                [
                    np.mean([getattr(r, trait) for r in by_sentence[key]], dtype=float)
                    for key in sentences
                ]
            )
            weights = metrics.resample_weights(
                len(values), inputs.config.resamples, f"{inputs.config.seed}|human|{model}|{trait}"
            )
            row[trait] = _interval_json(metrics.mean_interval(values, weights))
        models.append(row)

    return {
        "models": models,
        "raters": sorted({r.rater_id for r in scored}),
        "rounds": sorted({r.round for r in scored}),
        "agreement": _agreement(scored),
        "min_agreement": settings.min_agreement,
        "excluded_rounds": _excluded(inputs.ratings, trusted),
    }


def _excluded(all_ratings: list[HumanRating], trusted: list[HumanRating]) -> list[str]:
    keep = {(r.rater_id, r.round) for r in trusted}
    return sorted({f"{r.rater_id}/{r.round}" for r in all_ratings} - {"/".join(k) for k in keep})


def _agreement(scored: list[HumanRating]) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for trait in TRAITS:
        between: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        within: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for r in scored:
            score = float(getattr(r, trait))
            between[(r.model_id, r.segment_id)][r.rater_id].append(score)
            within[(r.rater_id, r.model_id, r.segment_id)].append(score)
        units = [[float(np.mean(v)) for v in raters.values()] for raters in between.values()]
        results[f"{trait}_between_raters"] = metrics.agreement(units)
        results[f"{trait}_within_rater"] = metrics.agreement(list(within.values()))
    return results


def build_report(inputs: EvalInputs) -> dict[str, Any]:
    config = inputs.config
    outputs = stored_outputs(inputs)
    covered = {segment_id for _, segment_id in inputs.translations}
    segments = [s for s in inputs.segments if s.id in covered]
    if not segments:
        raise ValueError("no translations stored for this run")
    references = [s.reference for s in segments]
    weights = metrics.resample_weights(len(segments), config.resamples, f"{config.seed}|chrf")

    stats = {
        m.id: metrics.chrf_stats([outputs.get((m.id, s.id), "") for s in segments], references)
        for m in config.models
    }
    models: list[dict[str, Any]] = []
    for model in config.models:
        records = [r for key, r in inputs.translations.items() if key[0] == model.id]
        models.append(
            {
                "model": model.id,
                "role": model.role,
                "chrf": _interval_json(metrics.chrf_interval(stats[model.id], weights)),
                "failed": len(segments) - sum(r.output is not None for r in records),
                "cost_usd": round(sum(r.cost_usd for r in records), 4),
            }
        )
    models.sort(key=lambda row: row["chrf"]["value"], reverse=True)

    teacher: str = next(row["model"] for row in models if row["role"] == "compare")
    headroom: list[dict[str, Any]] = []
    for row in models:
        if row["role"] != "base":
            continue
        gap = metrics.chrf_difference(stats[teacher], stats[row["model"]], weights)
        headroom.append(
            {
                "model": row["model"],
                "chrf": row["chrf"]["value"],
                "gap_to_best_compare": _interval_json(gap),
                "has_headroom": gap.low > 0,
            }
        )
    usable = [row for row in headroom if row["has_headroom"]]

    worst: dict[str, list[dict[str, Any]]] = {}
    for base in config.model_ids("base"):
        ranked = sorted(
            range(len(segments)),
            key=lambda i: metrics.chrf(stats[base][i]) - metrics.chrf(stats[teacher][i]),
        )
        worst[base] = [
            {
                "source": segments[i].source,
                "reference": segments[i].reference,
                "output": outputs.get((base, segments[i].id)),
                "best_compare_output": outputs.get((teacher, segments[i].id)),
                "chrf": round(metrics.chrf(stats[base][i]), 1),
                "best_compare_chrf": round(metrics.chrf(stats[teacher][i]), 1),
            }
            for i in ranked[:WORST_SENTENCES]
        ]

    return {
        "run": config.name,
        "eval_set": config.eval_set,
        "sentences": len(segments),
        "sentences_in_set": len(inputs.segments),
        "partial": len(segments) < len(inputs.segments),
        "resamples": config.resamples,
        "best_compare": teacher,
        "models": models,
        "headroom": headroom,
        "recommended_base": usable[0]["model"] if usable else None,
        "judge": {
            "model": config.judge.model,
            "pairs": _judge_rows(inputs, segments, outputs),
            "input_tokens": sum(v.input_tokens for v in inputs.verdicts.values()),
            "output_tokens": sum(v.output_tokens for v in inputs.verdicts.values()),
        },
        "human": _human_section(inputs),
        "worst_sentences": worst,
        "translation_cost_usd": round(sum(row["cost_usd"] for row in models), 4),
    }


STYLE = """
:root { color-scheme: light dark; --bg: #fff; --fg: #1c1c1c; --muted: #666; --line: #ddd; }
@media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #eee; --muted: #aaa;
  --line: #3a3a3a; } }
body { background: var(--bg); color: var(--fg); margin: 0 auto; max-width: 960px;
  padding: 24px 16px; font: 16px/1.5 system-ui, sans-serif; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 28px; }
th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left;
  vertical-align: top; }
td.number { font-variant-numeric: tabular-nums; white-space: nowrap; }
.note { color: var(--muted); }
.warning { border-left: 4px solid #c77700; padding: 8px 12px; }
"""


def _text(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{_text(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="number">{cell}</td>' if i >= numeric_from else f"<td>{cell}</td>"
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _range(interval: Mapping[str, float], digits: int = 2) -> str:
    return (
        f"{interval['value']:.{digits}f} "
        f"[{interval['low']:.{digits}f}, {interval['high']:.{digits}f}]"
    )


def _base_finding(report: dict[str, Any]) -> str:
    if report["recommended_base"]:
        return (
            f"Strongest base with headroom: {_text(report['recommended_base'])}. It must still "
            "fit the cheapest GPU, be exportable, and be licensed for self-hosting."
        )
    return (
        "No candidate base sits clearly below the best compare model, so the gate could not "
        "pass with these."
    )


def _human_html(human: dict[str, Any] | None) -> str:
    if human is None:
        return "<h2>Human ratings</h2><p class='note'>No rounds have been scored yet.</p>"
    rows = [
        [
            _text(row["model"]),
            _range(row["adequacy"]),
            _range(row["fluency"]),
            str(row["sentences"]),
        ]
        for row in human["models"]
    ]
    agreement = "".join(
        f"<li>{_text(name.replace('_', ' '))}: "
        f"{'not enough data' if value is None else f'{value:.2f}'}</li>"
        for name, value in human.get("agreement", {}).items()
    )
    excluded = ", ".join(human["excluded_rounds"]) or "none"
    return (
        "<h2>Human ratings</h2>"
        + _table(["Model", "Adequacy (1-5)", "Fluency (1-5)", "Sentences"], rows)
        + f"<p>Agreement, Krippendorff's alpha, flagged below "
        f"{human.get('min_agreement', 0):.3f}:</p><ul>{agreement}</ul>"
        + f"<p class='note'>Rounds left out after failing hidden checks: {_text(excluded)}.</p>"
    )


def render_html(report: dict[str, Any]) -> str:
    title = f"Evaluation report: {report['run']}"
    parts = [
        f"<h1>{_text(title)}</h1>",
        f"<p>{_base_finding(report)}</p>",
        f"<p class='note'>{report['sentences']} sentences from {_text(report['eval_set'])}. "
        "Every range is a 95% bootstrap interval. A difference inside its interval is a tie. "
        f"Translation cost ${report['translation_cost_usd']:.2f}.</p>",
    ]
    if report["partial"]:
        parts.append(
            f"<p class='warning'>Partial run: {report['sentences']} of "
            f"{report['sentences_in_set']} sentences. Do not cite these numbers.</p>"
        )

    parts.append("<h2>chrF++ on the benchmark</h2>")
    parts.append(
        _table(
            ["Model", "Role", "chrF++", "Failed"],
            [
                [_text(m["model"]), _text(m["role"]), _range(m["chrf"]), str(m["failed"])]
                for m in report["models"]
            ],
            numeric_from=2,
        )
    )
    parts.append("<h2>Headroom</h2>")
    parts.append(
        f"<p class='note'>Gap to {_text(report['best_compare'])}, the best compare "
        "model. Training on its answers cannot teach more than it knows.</p>"
    )
    parts.append(
        _table(
            ["Base model", "chrF++", "Gap", "Headroom"],
            [
                [
                    _text(h["model"]),
                    f"{h['chrf']:.2f}",
                    _range(h["gap_to_best_compare"]),
                    "yes" if h["has_headroom"] else "no",
                ]
                for h in report["headroom"]
            ],
        )
    )
    parts.append("<h2>Blind judge</h2>")
    judge = report["judge"]
    if judge["pairs"]:
        parts.append(
            f"<p class='note'>Judged by {_text(judge['model'])}, both orders, with no "
            "model names. A rate near 50% is a tie.</p>"
        )
        parts.append(
            _table(
                ["Base", "Against", "Base win rate", "Sentences", "Orders split"],
                [
                    [
                        _text(p["base"]),
                        _text(p["compare"]),
                        _range(p["base_win_rate"]),
                        str(p["sentences"]),
                        str(p["split_orders"]),
                    ]
                    for p in judge["pairs"]
                ],
                numeric_from=2,
            )
        )
    else:
        parts.append("<p class='note'>The judge has not run yet.</p>")
    parts.append(_human_html(report["human"]))

    parts.append("<h2>Worst sentences against the best compare model</h2>")
    for model, sentences in report["worst_sentences"].items():
        parts.append(f"<h3>{_text(model)}</h3>")
        parts.append(
            _table(
                ["Source", "Reference", "This model", "Best compare", "chrF++"],
                [
                    [
                        _text(s["source"]),
                        _text(s["reference"]),
                        _text(s["output"]) or "<em>failed</em>",
                        _text(s["best_compare_output"]) or "<em>failed</em>",
                        f"{s['chrf']} vs {s['best_compare_chrf']}",
                    ]
                    for s in sentences
                ],
                numeric_from=4,
            )
        )
    parts.append(
        "<p class='note'>FLORES-200 text is CC-BY-SA 4.0. Sentence-level chrF++ is noisy and "
        "only ranks examples for reading.</p>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_text(title)}</title><style>{STYLE}</style></head>"
        f"<body>{''.join(parts)}</body></html>\n"
    )
