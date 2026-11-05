"""A model card per registered model, generated from the registry and the run config."""

from typing import Any

from flywheel.schema import ModelRecord, StatusChange
from flywheel.train.config import RunConfig


def render_card(
    record: ModelRecord,
    config: RunConfig,
    history: list[StatusChange],
    dataset: dict[str, Any] | None,
) -> str:
    hp = config.hyperparameters
    lines = [
        f"# Model card: {record.id}",
        "",
        config.description,
        "",
        f"- **Status:** {record.status}",
        f"- **Base model:** {record.base_model}",
        f"- **Continues from:** {record.start_from or 'the base model'}",
        f"- **Method:** LoRA {record.method.upper()}",
        f"- **Trained on:** {record.provider}"
        + (f", job `{record.provider_job_id}`" if record.provider_job_id else ""),
        f"- **Cost:** ${record.cost_usd:.2f}",
        f"- **Run config:** `{record.run_config}`, sha256 `{record.run_config_hash}`",
        "",
        "## Training data",
        "",
        f"- **Snapshot:** `{config.method}-{record.dataset_hash[:12]}`, "
        f"sha256 `{record.dataset_hash}`",
        *_dataset_lines(dataset),
        "",
        "## Settings",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Epochs | {hp.epochs} |",
        f"| Learning rate | {hp.learning_rate} |",
        f"| Batch size | {hp.batch_size} |",
        f"| LoRA rank | {hp.lora_rank} |",
        f"| LoRA alpha | {hp.lora_alpha} |",
        *([f"| DPO beta | {hp.dpo_beta} |"] if hp.dpo_beta is not None else []),
        f"| Seed | {config.seed} |",
        "",
        "## Gate, set before training",
        "",
        "- chrF++ must beat the version it replaces beyond the 95% interval.",
        f"- The judge must prefer it at least {config.gate.min_judge_win_rate:.0%} of the time.",
        f"- No regression on the guard set `{config.gate.guard_set}`.",
        "- A blind human check must not find it worse."
        if config.gate.human_check
        else "- No human check for this run.",
        "",
        "## Evaluation",
        "",
        f"Report: `{record.eval_report}`"
        if record.eval_report
        else "Not evaluated yet, so this card makes no quality claim.",
        "",
        "## History",
        "",
        "| When | Change | Reason |",
        "|---|---|---|",
        *(f"| {c.at.date().isoformat()} | {_change(c)} | {c.reason} |" for c in history),
        "",
    ]
    return "\n".join(lines)


def _dataset_lines(dataset: dict[str, Any] | None) -> list[str]:
    if dataset is None:
        return ["- The snapshot's manifest was not found, so its contents are not described here."]
    counts = dataset["counts"]
    return [
        f"- **Size:** {counts['examples']} records from {counts['events']} curated events",
        f"- **Human share:** {counts['origin']['human_share']:.0%}",
        "- **Sources:** " + ", ".join(f"{k} ({v})" for k, v in counts["sources"].items()),
        *(
            f"- **Credit for {source}:** {corpus['license']}. {corpus['attribution']}"
            for source, corpus in dataset.get("licenses", {}).items()
        ),
    ]


def _change(change: StatusChange) -> str:
    if change.from_status is None:
        return change.to_status
    if change.from_status == change.to_status:
        return f"held as {change.to_status}"
    return f"{change.from_status} to {change.to_status}"
