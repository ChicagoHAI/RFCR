"""
modal_qwen25_7b.py — Deploy Qwen2.5-7B-Instruct on Modal via vLLM.

Single A10G-24GB is sufficient for Qwen2.5-7B in bfloat16 (~15GB VRAM).
Endpoint is a drop-in replacement for OpenRouter: set VLLM_BASE_URL_2 and
VLLM_MODEL_2 to route llm_client calls to this deployment automatically
(uses slot 2 so it can run alongside modal_mistral7b simultaneously).

Usage:
    # Step 1 — download weights once:
    modal run modal_qwen25_7b.py::download_weights

    # Step 2 — deploy:
    modal deploy modal_qwen25_7b.py

    # Step 3 — set env vars:
    VLLM_BASE_URL_2=https://<workspace>--qwen25-7b-instruct-serve.modal.run/v1/chat/completions
    VLLM_MODEL_2=qwen2.5-7b-instruct

Serves at: https://<workspace>--qwen25-7b-instruct-serve.modal.run
"""

import modal

HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID    = "qwen2.5-7b-instruct"   # model name passed to --served-model-name
VLLM_PORT   = 8000

app = modal.App("qwen25-7b-instruct")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "huggingface_hub[cli]",
        "fastapi",
        "uvicorn",
    )
)

volume = modal.Volume.from_name("qwen25-7b-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/weights": volume},
)
def download_weights():
    """Pre-download weights into the persistent volume (run once)."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir="/weights",
        ignore_patterns=["*.pt", "original/*"],
    )
    print(f"Downloaded {HF_MODEL_ID} → /weights")


@app.function(
    image=image,
    gpu="A10G",  # L40S preferred; requires Modal payment method
    timeout=7200,
    min_containers=1,
    volumes={"/weights": volume},
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve():
    import subprocess

    subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model",          HF_MODEL_ID,
            "--download-dir",   "/weights",
            "--dtype",          "bfloat16",
            "--max-model-len",  "8192",
            "--max-num-seqs",   "64",
            "--host",           "0.0.0.0",
            "--port",           str(VLLM_PORT),
            "--served-model-name", MODEL_ID,
            "--trust-remote-code",
        ],
    )
