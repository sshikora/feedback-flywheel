"""Translate an eval set with a self-hosted model on Modal.

The OpenRouter path sends one request per sentence. Here one Modal call answers
every missing sentence for a model, and its GPU time is split evenly into each
record's cost, so the budget check works the same way for both. Starting a call
and collecting it are separate, so a call outlives a lost connection.
"""

from collections.abc import Sequence

from flywheel.evals.config import ModelSpec
from flywheel.evals.translate import build_prompt
from flywheel.schema import EvalSegment, TranslationRecord


class HostError(Exception):
    """Modal refused the work or it failed there. Nothing from this call is stored."""


class StillRunning(Exception):
    """This machine lost contact with Modal. The call carries on there."""


def worst_case_usd() -> float:
    """What one model can cost at most, since the GPU call stops at its timeout."""
    from flywheel.evals import modal_host

    return modal_host.TIMEOUT_SECONDS * modal_host.GPU_USD_PER_SECOND


def start_on_modal(spec: ModelSpec, segments: Sequence[EvalSegment], url: str | None) -> str:
    """Deploy the app and start one call for every segment. Returns the call's id."""
    import modal

    from flywheel.evals import modal_host

    kind, _, name = (spec.weights or "").partition(":")
    source = {"repo": name} if kind == "hf" else {"url": url}
    try:
        modal_host.app.deploy()
        call = modal_host.evaluate.spawn(
            spec.weights_key, [build_prompt(segment) for segment in segments], **source
        )
    except modal.exception.Error as err:
        raise HostError(f"{type(err).__name__}: {err}") from err
    return str(call.object_id)


def collect_from_modal(
    call_id: str, spec: ModelSpec, segments: Sequence[EvalSegment]
) -> list[TranslationRecord]:
    """Wait for a call and turn its answers into records."""
    import modal

    from flywheel.evals import modal_host

    try:
        answers, seconds = modal.FunctionCall.from_id(call_id).get()
    except (OSError, TimeoutError, modal.exception.ConnectionError) as err:
        raise StillRunning(f"{type(err).__name__}: {err}") from err
    except Exception as err:
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
