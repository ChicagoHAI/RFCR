"""Candidate-output cache and vote helpers for SF-CR replay experiments."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PARSER_VERSION = "formal_valid_invalid_parser_v2"


def stable_hash(text: str | bytes) -> str:
    data = text if isinstance(text, bytes) else str(text).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def candidate_text_hash(candidate: dict) -> str:
    parts = [
        str(candidate.get("rule", "")),
        str(candidate.get("use_when", "")),
        str(candidate.get("do_not_use_when", "")),
        str(candidate.get("check", "")),
        str(candidate.get("micro_example", "")),
        str(candidate.get("decision_template", "")),
    ]
    return stable_hash("\n\n".join(parts))


def make_candidate_prompt_hash(anchor_text: str, candidate: dict, item: dict) -> str:
    prompt_material = {
        "anchor": anchor_text,
        "candidate_text_hash": candidate_text_hash(candidate),
        "item_id": str(item.get("_sfcr_id") or item.get("id") or ""),
        "item_input": str(item.get("input") or item.get("question") or item),
    }
    return stable_hash(json.dumps(prompt_material, sort_keys=True, ensure_ascii=False))


def prompt_template_hash(task_spec, rf_scoring_flag: bool) -> str:
    builder = getattr(task_spec, "build_scoring_prompt_rf", None) if rf_scoring_flag else None
    if builder is None:
        builder = getattr(task_spec, "build_scoring_prompt", None)
    try:
        material = inspect.getsource(builder)
    except Exception:
        material = repr(builder)
    return stable_hash(material)


def make_protocol_signature(
    *,
    task: str,
    scorer_model: str,
    anchor_hash: str,
    rf_scoring_flag: bool,
    partition_routed_flag: bool,
    cot_first: bool,
    reasoning_effort: str | None,
    prompt_template_hash_value: str,
    concurrency: int,
    parser_version: str = PARSER_VERSION,
) -> dict:
    return {
        "task": task,
        "scorer_model": scorer_model,
        "anchor_hash": anchor_hash,
        "rf_scoring": bool(rf_scoring_flag),
        "partition_routed_scoring": bool(partition_routed_flag),
        "cot_first": bool(cot_first),
        "reasoning_effort": reasoning_effort,
        "prompt_template_hash": prompt_template_hash_value,
        "concurrency": int(concurrency),
        "parser_version": parser_version,
    }


def protocol_signature_hash(signature: dict) -> str:
    return stable_hash(json.dumps(signature, sort_keys=True, ensure_ascii=False))


def protocol_mismatch_reasons(baseline: dict | None, candidate: dict | None) -> list[str]:
    if not baseline or not candidate:
        return ["missing_protocol_signature"]
    reasons = []
    for key in (
        "task",
        "scorer_model",
        "anchor_hash",
        "rf_scoring",
        "partition_routed_scoring",
        "cot_first",
        "reasoning_effort",
        "prompt_template_hash",
        "parser_version",
    ):
        if baseline.get(key) != candidate.get(key):
            reasons.append(key)
    return reasons


def assert_protocol_compatible(
    baseline: dict | None,
    candidate: dict | None,
    *,
    allow_mixed_protocol_comparison: bool = False,
) -> None:
    reasons = protocol_mismatch_reasons(baseline, candidate)
    if reasons and not allow_mixed_protocol_comparison:
        raise ValueError("Scoring protocol mismatch: " + ",".join(reasons))


def make_cache_key(
    *,
    task: str,
    item_id: str,
    scorer_model: str,
    anchor_hash: str,
    candidate_id: str,
    candidate_hash: str,
    prompt_hash: str,
    scoring_concurrency: int,
    rf_scoring_flag: bool,
    partition_routed_flag: bool,
    cot_first: bool = True,
    reasoning_effort: str | None = "low",
    prompt_template_hash_value: str = "",
    parser_version: str = PARSER_VERSION,
    repeat_index: int = 0,
) -> str:
    material = {
        "task": task,
        "item_id": item_id,
        "scorer_model": scorer_model,
        "anchor_hash": anchor_hash,
        "candidate_id": candidate_id,
        "candidate_text_hash": candidate_hash,
        "prompt_hash": prompt_hash,
        "scoring_concurrency": scoring_concurrency,
        "rf_scoring_flag": rf_scoring_flag,
        "partition_routed_flag": partition_routed_flag,
        "cot_first": bool(cot_first),
        "reasoning_effort": reasoning_effort,
        "prompt_template_hash": prompt_template_hash_value,
        "parser_version": parser_version,
        "repeat_index": repeat_index,
    }
    return stable_hash(json.dumps(material, sort_keys=True))


class CandidateOutputCache:
    """Small JSONL cache keyed by candidate text and item prompt material."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.rows: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("cache_key")
                if key:
                    self.rows[key] = row

    def get(self, key: str) -> dict | None:
        row = self.rows.get(key)
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def put(self, row: dict) -> None:
        key = row["cache_key"]
        payload = {
            **row,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.rows[key] = payload
        self.writes += 1
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def stats(self) -> dict:
        return {
            "path": str(self.path) if self.path else "",
            "entries": len(self.rows),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }


def build_cache_row(
    *,
    cache_key: str,
    task: str,
    item: dict,
    candidate_id: str,
    scorer_model: str,
    prompt_hash: str,
    correct: bool,
    annotated_item: dict,
    protocol_signature: dict | None = None,
    repeat_index: int = 0,
) -> dict:
    return {
        "cache_key": cache_key,
        "task": task,
        "item_id": str(item.get("_sfcr_id") or item.get("id") or ""),
        "candidate_id": candidate_id,
        "scorer_model": scorer_model,
        "prompt_hash": prompt_hash,
        "protocol_signature": protocol_signature or {},
        "protocol_signature_hash": protocol_signature_hash(protocol_signature or {}),
        "repeat_index": repeat_index,
        "answer": annotated_item.get("predicted"),
        "gold": annotated_item.get("expected"),
        "correct": bool(correct),
        "parse_error": annotated_item.get("predicted") is None,
        "raw_output": annotated_item.get("raw_response", ""),
    }


def cache_row_to_scored_item(row: dict, item: dict) -> tuple[bool, dict]:
    annotated = {
        **item,
        "predicted": row.get("answer"),
        "expected": row.get("gold"),
        "raw_response": row.get("raw_output", ""),
        "post_think": "",
        "thinking": "",
    }
    return bool(row.get("correct")), annotated


def aggregate_candidate_votes(votes: Iterable[dict]) -> dict:
    rows = list(votes)
    if not rows:
        return {
            "majority_correct": False,
            "candidate_correct_votes": 0,
            "candidate_correct_rate": 0.0,
            "majority_answer": None,
            "candidate_answer_votes": {},
        }
    correct_votes = sum(1 for row in rows if row.get("correct"))
    answer_votes = Counter(row.get("answer") for row in rows)
    majority_answer, _ = answer_votes.most_common(1)[0]
    return {
        "majority_correct": correct_votes > (len(rows) / 2),
        "candidate_correct_votes": correct_votes,
        "candidate_correct_rate": correct_votes / len(rows),
        "majority_answer": majority_answer,
        "candidate_answer_votes": dict(answer_votes),
    }


def rf_scoring_enabled() -> bool:
    return os.environ.get("ICR_USE_RF_SCORING", "").strip().lower() in {"1", "true", "yes", "on"}


def partition_routed_scoring_enabled() -> bool:
    return os.environ.get("ICR_USE_PARTITION_ROUTED_SCORING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
