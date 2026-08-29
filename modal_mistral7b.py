"""
modal_mistral7b.py — Deploy mistral-7b-instruct-v0.3 on Modal via vLLM.

Single A10G-24GB is sufficient for Mistral-7B in bfloat16 (~14GB VRAM).
Endpoint is a drop-in replacement for OpenRouter: set VLLM_BASE_URL and
VLLM_MODEL to route llm_client calls to this deployment automatically.

Usage:
    # Step 1 — download weights once:
    modal run modal_mistral7b.py::download_weights

    # Step 2 — deploy:
    modal deploy modal_mistral7b.py

    # Step 3 — set env vars (or use utils/modal_backend.py pattern):
    VLLM_BASE_URL=https://<workspace>--mistral-7b-instruct-serve.modal.run/v1/chat/completions
    VLLM_MODEL=mistral-7b-instruct-v0.3

Serves at: https://<workspace>--mistral-7b-instruct-serve.modal.run
"""

import modal

HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_ID    = "mistral-7b-instruct-v0.3"   # served model name exposed via API
VLLM_PORT   = 8000

app = modal.App("mistral-7b-instruct")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "huggingface_hub[cli]",
        "fastapi",
        "uvicorn",
    )
)

volume = modal.Volume.from_name("mistral-7b-weights", create_if_missing=True)


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
