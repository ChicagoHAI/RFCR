"""
run_logger.py — Structured per-run logging for training and eval runs.

Every run (pipeline training or eval script) creates one RunLogger and registers
it as the global active logger.  All downstream calls to call_llm() and
score_batch() then write to it automatically — no logger object needs to be
threaded through call stacks.

Directory layout
----------------
    runs/logs/train/<task>_<run_id>/   ← pipeline training runs
    runs/logs/eval/<run_id>/           ← eval script runs

    Each directory contains:
        run_config.json   — all CLI args and invocation metadata (written at init)
        llm_calls.jsonl   — one entry per LLM request (full prompt + full response)
        scoring.jsonl     — one entry per score_batch call (aggregate stats)

Usage (call once at process start)
-----------------------------------
    from utils.run_logger import RunLogger, set_logger
    logger = RunLogger("runs/logs/train/web_of_lies", config=vars(args))
    set_logger(logger)
    # All subsequent call_llm() / score_batch() calls are now logged.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

def make_run_id(label: str = "") -> str:
    """
    Return a unique run ID stamped to microsecond UTC precision.
    Format: <label>_YYYYMMDD_HHMMSS_ffffff  (or just YYYYMMDD_HHMMSS_ffffff)
    Microseconds make collisions essentially impossible even on fast machines.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{label}_{ts}" if label else ts


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

class RunLogger:
    """
    Creates a unique timestamped log directory and provides JSONL append writers
    for LLM calls and scoring passes.  All public methods are thread-safe.
    """

    def __init__(
        self,
        log_base: str | Path,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.run_id  = run_id or make_run_id()
        self.log_dir = Path(log_base) / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._llm_path   = self.log_dir / "llm_calls.jsonl"
        self._score_path = self.log_dir / "scoring.jsonl"

        self._llm_lock   = threading.Lock()
        self._score_lock = threading.Lock()

        if config is not None:
            self._write_config(config)

    # ── Config dump ──────────────────────────────────────────────────────────

    def _write_config(self, config: dict[str, Any]) -> None:
        payload = {
            "run_id":    self.run_id,
            "timestamp": _utcnow(),
            "log_dir":   str(self.log_dir),
            "config":    {k: _jsonify(v) for k, v in config.items()},
        }
        (self.log_dir / "run_config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── LLM call ─────────────────────────────────────────────────────────────

    def log_llm_call(
        self,
        *,
        model:       str,
        prompt:      str,
        response:    str,
        thinking:    str = "",
        elapsed_ms:  float,
        label:       str = "",
        temperature: float = 0.0,
        max_tokens:  int = 0,
    ) -> None:
        """Append one entry to llm_calls.jsonl. Called once per call_llm() invocation."""
        entry = {
            "ts":           _utcnow(),
            "label":        label,
            "model":        model,
            "elapsed_ms":   round(elapsed_ms, 1),
            "prompt_chars": len(prompt),
            "resp_chars":   len(response),
            "temperature":  temperature,
            "max_tokens":   max_tokens,
            "prompt":       prompt,
            "response":     response,
            "thinking":     thinking,
        }
        with self._llm_lock:
            with open(self._llm_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Scoring pass ─────────────────────────────────────────────────────────

    def log_scoring(
        self,
        *,
        label:     str,
        model:     str,
        n_correct: int,
        n_total:   int,
        accuracy:  float,
    ) -> None:
        """Append one entry to scoring.jsonl. Called once per score_batch() invocation."""
        entry = {
            "ts":        _utcnow(),
            "label":     label,
            "model":     model,
            "n_correct": n_correct,
            "n_total":   n_total,
            "accuracy":  round(accuracy, 4),
        }
        with self._score_lock:
            with open(self._score_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Global active logger
# ---------------------------------------------------------------------------

_active_logger: RunLogger | None = None
_global_lock = threading.Lock()


def set_logger(logger: RunLogger) -> None:
    """Register *logger* as the process-wide active logger."""
    global _active_logger
    with _global_lock:
        _active_logger = logger


def get_logger() -> RunLogger | None:
    """Return the active logger, or None if none has been set."""
    return _active_logger


def clear_logger() -> None:
    """Deactivate the global logger (mainly for tests)."""
    global _active_logger
    with _global_lock:
        _active_logger = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonify(v: Any) -> Any:
    """Recursively convert non-JSON-serializable types to strings."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(val) for k, val in v.items()}
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)
