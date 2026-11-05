"""Translate an eval set with a self-hosted model on Modal.

The OpenRouter path sends one request per sentence. Here one GPU call answers
every missing sentence for a model, and its time is split evenly into each
record's cost, so the budget check works the same way for both.
"""

from collections.abc import Sequence

from flywheel.evals.config import ModelSpec
from flywheel.evals.translate import build_prompt
from flywheel.schema import EvalSegment, TranslationRecord


class HostError(Exception):
    """Modal refused the work or it failed there. Nothing from this call is stored."""


def worst_case_usd() -> float:
    """What one model can cost at most, since the GPU call stops at its timeout."""
    from flywheel.evals import modal_host

    return modal_host.TIMEOUT_SECONDS * modal_host.GPU_USD_PER_SECOND


def translate_on_modal(
    spec: ModelSpec, segments: Sequence[EvalSegment], url: str | None
) -> list[TranslationRecord]:
    """Fetch the weights if the volume lacks them, then answer every segment."""
    import modal

    from flywheel.evals import modal_host

    kind, _, name = (spec.weights or "").partition(":")
    try:
        with modal_host.app.run():
            if kind == "hf":
                modal_host.fetch.remote(spec.weights_key, repo=name)
            else:
                modal_host.fetch.remote(spec.weights_key, url=url)
            answers, seconds = modal_host.generate.remote(
                spec.weights_key, [build_prompt(segment) for segment in segments]
            )
    except modal.exception.Error as err:
        raise HostError(f"{type(err).__name__}: {err}") from err
    cost = seconds * modal_host.GPU_USD_PER_SECOND / max(len(segments), 1)
    return [
        TranslationRecord(
            segment_id=segment.id,
            model_id=spec.id,
            output=text or None,
            error=None if text else "empty answer",
            cost_usd=cost,
        )
        for segment, (text, _finish) in zip(segments, answers, strict=True)
    ]
