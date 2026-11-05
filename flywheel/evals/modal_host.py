"""Run self-hosted models over an eval set on a Modal GPU.

Weights are fetched once into a Modal volume: from Hugging Face for a base
model, or from a short-lived link Together signs for a trained one, so the
Together key never leaves this machine. The GPU function runs vLLM in batch
mode and exits, so nothing is left running, and its timeout caps the spend.

Both models in a comparison run here with the same settings, so a difference in
scores comes from the weights and not from the host.
"""

import shlex
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import modal

GPU = "L4"
GPU_USD_PER_SECOND = 0.000222  # Modal's L4 rate on 2026-11-05, about $0.80 an hour
TIMEOUT_SECONDS = 1800  # a run stops here, which caps one model at about $0.40 of GPU time
MAX_MODEL_LEN = 8192
MAX_TOKENS = 4096  # the same stop as OpenRouter requests, for a model that repeats itself
SEED = 1

# Qwen's recommended settings for non-thinking mode, from the Qwen3.5 model card.
SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

WEIGHTS = "/weights"

app = modal.App("flywheel-eval")
volume = modal.Volume.from_name("flywheel-weights", create_if_missing=True)
fetch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd")
    .uv_pip_install("huggingface_hub==2.3.0")
    .add_local_python_source("flywheel")
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.32.0")
    .add_local_python_source("flywheel")
)


class Generation(NamedTuple):
    text: str
    finish_reason: str | None


@app.function(image=fetch_image, volumes={WEIGHTS: volume}, timeout=3600)
def fetch(key: str, repo: str | None = None, url: str | None = None) -> bool:
    """Put weights in the volume under `key`. Returns False when they were already there."""
    target = Path(WEIGHTS) / key
    done = target / ".complete"
    if done.exists():
        return False
    target.mkdir(parents=True, exist_ok=True)
    if repo is not None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=target)
    elif url is not None:
        command = f"curl -fsSL {shlex.quote(url)} | zstd -d | tar -x -C {shlex.quote(str(target))}"
        subprocess.run(["bash", "-o", "pipefail", "-c", command], check=True)
    else:
        raise ValueError("fetch needs a repo or a url")
    done.touch()
    volume.commit()
    return True


@app.function(image=gpu_image, gpu=GPU, volumes={WEIGHTS: volume}, timeout=TIMEOUT_SECONDS)
def generate(key: str, prompts: list[str]) -> tuple[list[Generation], float]:
    """Answer each prompt as a single user turn. Returns the answers and seconds spent."""
    started = time.monotonic()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=f"{WEIGHTS}/{key}",
        max_model_len=MAX_MODEL_LEN,
        language_model_only=True,
        seed=SEED,
    )
    params = SamplingParams(max_tokens=MAX_TOKENS, seed=SEED, **SAMPLING)
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    results = llm.chat(
        conversations,
        params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    answers = [
        Generation(result.outputs[0].text.strip(), result.outputs[0].finish_reason)
        for result in results
    ]
    return answers, time.monotonic() - started
