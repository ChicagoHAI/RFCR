"""
llm_client.py — Unified LLM call utility for all ICRefine pipelines.

Returns LLMResponse(content, thinking) for every call so post-think
is always available. content = structured output after reasoning;
thinking = full internal CoT trace (empty for non-reasoning models).

Per Heddaya et al. (ACL 2026), post-think preserves deductive markers
at 25× higher density than externally prompted summaries.
"""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv
from .run_logger import get_logger

load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL        = "https://api.openai.com/v1/chat/completions"
MAX_TOKENS        = int(os.environ.get("ICR_MAX_TOKENS", 16_000))
MAX_RETRIES       = 6
RETRY_BASE_DELAY  = 10.0
VLLM_READ_TIMEOUT = 1800  # local inference can be slow — 30 min per request

_OPENAI_PREFIXES  = ("gpt-4", "gpt-3", "o1", "o3", "o4")
_OPENAI_REASONING = ("o1", "o3", "o4")  # o-series: no temperature, max_completion_tokens, reasoning_effort

# Models that expose internal reasoning traces (thinking / reasoning_content fields).
# Non-reasoning models need explicit justification elicitation in the scoring prompt.
_REASONING_MODEL_SUBSTRINGS = ("deepseek-r1", "r1-0528", "claude-3-7-sonnet")


def is_reasoning_model(model: str) -> bool:
    """Return True for models that expose internal CoT traces (thinking/reasoning_content)."""
    m = model.lower().split("/")[-1]
    return (
        any(m.startswith(p) for p in _OPENAI_REASONING)
        or any(s in m for s in _REASONING_MODEL_SUBSTRINGS)
    )


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content:  str   # post-think: structured output after internal reasoning
    thinking: str   # full CoT trace from the reasoning field (empty if unavailable)


# ---------------------------------------------------------------------------
# API key + endpoint resolution
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("Error: OPENROUTER_API_KEY environment variable is not set.")
    return key


def _resolve_endpoint(model: str) -> tuple[str, str, bool, bool]:
    """
    Return (url, api_key, is_openai, is_vllm) for the given model.

    Routing priority:
      1. vLLM — when VLLM_BASE_URL + VLLM_MODEL are set and model matches.
      2. OpenAI direct — when OPENAI_API_KEY is set and model starts with a
         known OpenAI prefix.
      3. OpenRouter — fallback for everything else.
    """
    vllm_url   = os.environ.get("VLLM_BASE_URL", "")
    vllm_model = os.environ.get("VLLM_MODEL", "")
    if vllm_url and vllm_model and model == vllm_model:
        return vllm_url, os.environ.get("VLLM_API_KEY", ""), False, True

    # Secondary vLLM endpoint — used by ensemble scoring for the second model.
    # Set VLLM_BASE_URL_2 and VLLM_MODEL_2 to route a specific model to a
    # different vLLM server (e.g. different node/port on the cluster).
    vllm_url_2   = os.environ.get("VLLM_BASE_URL_2", "")
    vllm_model_2 = os.environ.get("VLLM_MODEL_2", "")
    if vllm_url_2 and vllm_model_2 and model == vllm_model_2:
        return vllm_url_2, os.environ.get("VLLM_API_KEY_2", ""), False, True

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    bare = model.removeprefix("openai/")
    if openai_key and bare.startswith(_OPENAI_PREFIXES):
        return OPENAI_URL, openai_key, True, False

    return OPENROUTER_URL, os.environ.get("OPENROUTER_API_KEY", ""), False, False


# ---------------------------------------------------------------------------
# Core call — returns LLMResponse
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    reasoning_effort: str | None = "low",
    seed: int | None = 42,
    _log_label: str = "",
) -> LLMResponse:
    """
    Send a single prompt and return LLMResponse(content, thinking).

    content  = message["content"]            — post-think structured output
    thinking = message["reasoning_content"]  — full internal CoT (vLLM)
             = message["reasoning"]          — full internal CoT (OpenRouter)

    For models that do not expose reasoning (e.g. gpt-4o), thinking will be "".
    reasoning_effort: "low" | "medium" | "high" | None. Sent as
      {"reasoning": {"effort": ...}} for OpenRouter reasoning models. Omitted
      for vLLM and OpenAI (not supported).
    """
    _t0 = time.monotonic()
    url, resolved_key, is_openai, is_vllm = _resolve_endpoint(model)
    model_name = model.removeprefix("openai/") if is_openai else model
    is_openai_reasoning = is_openai and any(model_name.startswith(p) for p in _OPENAI_REASONING)

    payload: dict = {
        "model":    model_name,
        "messages": [{"role": "user", "content": prompt}],
    }
    # o-series (o1/o3/o4) use max_completion_tokens and don't support temperature
    if is_openai_reasoning:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"]  = max_tokens
        payload["temperature"] = temperature
    if seed is not None and not is_openai_reasoning:
        payload["seed"] = seed
    # reasoning_effort: OpenRouter uses {"reasoning": {"effort": ...}};
    # OpenAI o-series uses top-level "reasoning_effort"
    if reasoning_effort is not None:
        if is_openai_reasoning:
            payload["reasoning_effort"] = reasoning_effort
        elif not is_openai and not is_vllm:
            payload["reasoning"] = {"effort": reasoning_effort}
    if not is_openai and not is_vllm:
        payload["provider"] = {"allow_fallbacks": True}

    headers = {"Content-Type": "application/json"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    if not is_openai and not is_vllm:
        headers["HTTP-Referer"] = "https://github.com/sair-evaluation"
        headers["X-Title"]      = "SAIR ICRefine"

    read_timeout = VLLM_READ_TIMEOUT if is_vllm else 300
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=(10, read_timeout)
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    print(
                        f"\n  [retry] HTTP {resp.status_code} — backing off {delay:.0f}s "
                        f"(attempt {attempt}/{MAX_RETRIES})",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
            resp.raise_for_status()

            message = resp.json()["choices"][0]["message"]
            content = (message.get("content") or "").strip()

            if is_vllm:
                thinking = (message.get("reasoning_content") or "").strip()
                # DeepSeek-R1 sometimes puts the structured answer (VERDICT/REASONING/etc.)
                # inside reasoning_content and emits only a short informal sentence in content.
                # Fall back to thinking when: content is empty OR content has no VERDICT line.
                if thinking and (not content or "VERDICT:" not in content.upper()):
                    content  = thinking
                    thinking = ""
                # vLLM may also return <think>...</think> inline in content instead of
                # separating it into reasoning_content. Strip the think block and keep
                # only the text after </think> as the structured answer.
                if "<think>" in content:
                    post_think = re.split(r"</think>", content, flags=re.IGNORECASE)
                    if len(post_think) > 1:
                        thinking = re.sub(r"<think>", "", post_think[0], flags=re.IGNORECASE).strip()
                        content  = post_think[-1].strip()
                    else:
                        # <think> opened but never closed — model was cut off mid-think;
                        # try to find VERDICT inside the think block as last resort.
                        thinking = re.sub(r"<think>", "", content, flags=re.IGNORECASE).strip()
                        content  = thinking
            else:
                thinking = (message.get("reasoning") or "").strip()
                # gpt-oss-120b via OpenRouter sometimes puts everything in "reasoning"
                if not content and thinking:
                    content  = thinking
                    thinking = ""

            _resp = LLMResponse(content=content, thinking=thinking)
            _log = get_logger()
            if _log is not None:
                _log.log_llm_call(
                    model=model,
                    prompt=prompt,
                    response=content,
                    thinking=thinking,
                    elapsed_ms=(time.monotonic() - _t0) * 1000,
                    label=_log_label,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return _resp

        except requests.HTTPError as exc:
            # 4xx (except 429) are client errors — retrying won't help.
            if exc.response is not None and 400 <= exc.response.status_code < 500 \
                    and exc.response.status_code != 429:
                raise RuntimeError(
                    f"LLM call failed (HTTP {exc.response.status_code}, not retrying): {exc}"
                ) from exc
            if attempt < MAX_RETRIES:
                print(
                    f"\n  [retry] {type(exc).__name__} — backing off {delay:.0f}s "
                    f"(attempt {attempt}/{MAX_RETRIES}): {exc}",
                    file=sys.stderr, flush=True,
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(
                    f"LLM call failed after {MAX_RETRIES} retries: {exc}"
                ) from exc
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                print(
                    f"\n  [retry] {type(exc).__name__} — backing off {delay:.0f}s "
                    f"(attempt {attempt}/{MAX_RETRIES}): {exc}",
                    file=sys.stderr, flush=True,
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(
                    f"LLM call failed after {MAX_RETRIES} retries: {exc}"
                ) from exc

    raise RuntimeError("Unexpected exit from retry loop.")


# ---------------------------------------------------------------------------
# Parallel batch call — returns list[LLMResponse | None]
# ---------------------------------------------------------------------------

def call_llm_batch(
    prompts: list[str],
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    concurrency: int = 10,
    progress_label: str = "",
    reasoning_effort: str | None = "low",
    seed: int | None = 42,
) -> list[LLMResponse | None]:
    """
    Call the LLM for each prompt in parallel using a thread pool.

    Returns a list of LLMResponse in the same order as prompts.
    Entries are None where the call failed after all retries.
    """
    results: list[LLMResponse | None] = [None] * len(prompts)
    done = 0

    def _call(idx: int, prompt: str) -> tuple[int, LLMResponse | None]:
        try:
            return idx, call_llm(prompt, model, api_key, temperature, max_tokens,
                                 reasoning_effort, seed, _log_label=progress_label)
        except Exception as exc:
            print(f"\n  [batch] item {idx} error: {exc}", file=sys.stderr)
            return idx, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for i, p in enumerate(prompts):
            futures[pool.submit(_call, i, p)] = i
            if i > 0 and i % concurrency == 0:
                time.sleep(1.0)  # brief pause every batch to avoid burst 429s
        for future in as_completed(futures):
            idx, resp = future.result()
            results[idx] = resp
            done += 1
            label = f"  {progress_label} " if progress_label else "  "
            print(f"{label}{done}/{len(prompts)}", flush=True, file=sys.stderr)
    return results
