"""ICR_sfcr/logger.py — Per-example JSONL logging for SFCR runs.

Implements the logging schema specified in section 10 of the experiment plan.
One row is written per (example, model, condition) triple.

Usage:
    log = SFCRLogger(output_dir / "sfcr_log.jsonl", run_id="sfcr_cj_1000")
    log.write_row(...)
    log.close()
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Log row schema (section 10)
# ---------------------------------------------------------------------------

@dataclass
class SFCRLogRow:
    run_id:               str
    seed:                 int
    task:                 str
    dataset:              str
    condition:            str   # "anchor" | "sfcr_global" | "sfcr_routed"
    source_model:         str
    target_model:         str
    example_id:           str
    gold_label:           str
    prediction:           str
    correct:              bool
    prompt_length:        int
    rules_available:      int
    rules_activated:      int
    accepted_rule_ids:    list[str]
    activation_match:     bool
    failure_region_type:  str   # "shared" | "private" | "easy" | "other"
    oracle_mode:          str
    routing_mode:         str


# ---------------------------------------------------------------------------
# Logger class
# ---------------------------------------------------------------------------

class SFCRLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        self._path   = path
        self._run_id = run_id
        self._fh     = open(path, "a", encoding="utf-8")
        self._n      = 0

    def write_row(self, row: SFCRLogRow) -> None:
        self._fh.write(json.dumps(asdict(row)) + "\n")
        self._fh.flush()
        self._n += 1

    def close(self) -> None:
        self._fh.close()

    @property
    def n_rows(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Helpers for building rows
# ---------------------------------------------------------------------------

def rule_id(rule: dict) -> str:
    """Stable short id for a rule (first 8 chars of sha256 of rule text)."""
    return hashlib.sha256(rule["rule"].encode()).hexdigest()[:8]


def failure_region_type(
    item_id: str,
    v_shared_ids: set,
    v_private_ids: set,
    v_easy_ids: set,
) -> str:
    if item_id in v_shared_ids:
        return "shared"
    if item_id in v_private_ids:
        return "private"
    if item_id in v_easy_ids:
        return "easy"
    return "other"


def log_condition(
    logger: SFCRLogger,
    items: list[dict],
    scored_by_model: dict[str, dict[str, bool]],  # model -> {sfcr_id: correct}
    cheatsheet_text: str,
    condition: str,
    accepted_rules: list[dict],
    routing_mode: str,
    oracle_mode: str,
    task: str,
    dataset: str,
    seed: int,
    source_model: str,
    v_shared_ids: set,
    v_private_ids: set,
    v_easy_ids: set,
) -> None:
    """
    Write one log row per (item, model) pair for a given scoring condition.

    `scored_by_model` maps model name → {sfcr_id: correct_bool}.
    Items that appear in `accepted_rules` activation contexts get
    activation_match=True.
    """
    from .activation import matches_use_when

    accepted_ids = [rule_id(r) for r in accepted_rules]

    for model, correctness_map in scored_by_model.items():
        for it in items:
            iid        = it["_sfcr_id"]
            correct    = correctness_map.get(iid, False)
            prediction = str(it.get("predicted", ""))
            gold       = str(it.get("answer", ""))

            # Activation: how many rules match this item's input
            question  = it.get("input", "")
            activated = [
                r for r in accepted_rules
                if matches_use_when(r.get("use_when", ""), question)
            ]
            act_match = len(activated) > 0

            row = SFCRLogRow(
                run_id              = logger._run_id,
                seed                = seed,
                task                = task,
                dataset             = dataset,
                condition           = condition,
                source_model        = source_model,
                target_model        = model,
                example_id          = iid,
                gold_label          = gold,
                prediction          = prediction,
                correct             = correct,
                prompt_length       = len(cheatsheet_text),
                rules_available     = len(accepted_rules),
                rules_activated     = len(activated),
                accepted_rule_ids   = accepted_ids,
                activation_match    = act_match,
                failure_region_type = failure_region_type(
                    iid, v_shared_ids, v_private_ids, v_easy_ids
                ),
                oracle_mode         = oracle_mode,
                routing_mode        = routing_mode,
            )
            logger.write_row(row)
