"""Run Experiment 1: CS-ICL-anchored routed SF-CR evaluation.

This script evaluates frozen SF-CR atoms as an overlay on the existing CS-ICL
surface. It reuses the current strict atom activation decisions, rescores only
activated items with ``CS-ICL cheatsheet + routed atom``, and makes inactive
items inherit the CS-ICL output exactly.

The output is a unified per-item table with four conditions:

* raw
* csicl
* sfcr_raw_anchor
* sfcr_csicl_anchor

The script is intentionally narrow: it does not regenerate or tune atoms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ICR_sfcr.rule_validator import build_cheatsheet_with_rule  # noqa: E402
from scripts.rerun_sfcr_atom_effects_strict import (  # noqa: E402
    ATOM_TASKS,
    DISAMBIG_RULES,
    FORMAL_CANDIDATE_FILES,
    GEOMETRIC_CANDIDATE_FILES,
    OBJECT_SUMMARY,
    WEB_SUMMARY,
    _active_ids_from_effect,
    _as_bool,
    _load_candidate_rules,
    _load_json_rules,
    _load_task_items,
)
from scripts.replay_sfcr_candidates import _rf_inconsistency_reason, _rule_for_memory_format  # noqa: E402
from scripts.run_controlled_full_benchmark_pilot import _parse_prediction, _task_spec_from_cfg  # noqa: E402
from scripts.run_sfcr_emnlp_baseline_surface import _task_cfg  # noqa: E402
from utils.llm_client import call_llm_batch, get_api_key  # noqa: E402


CONDITIONS = ["raw", "csicl", "sfcr_raw_anchor", "sfcr_csicl_anchor"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row if not isinstance(row.get(k), (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_wide(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["task"], row["item_id"]): row for row in _read_jsonl(path)}


def _load_effects(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["task"], row["item_id"]): row for row in _read_jsonl(path)}


def _load_selected_atom_set(path: str | Path | None) -> tuple[set[tuple[str, str]] | None, list[dict]]:
    if not path:
        return None, []
    selected_path = _resolve(path)
    data = json.loads(selected_path.read_text(encoding="utf-8"))
    atoms = data if isinstance(data, list) else data.get("atoms", [])
    selected: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for atom in atoms:
        task = atom.get("task")
        atom_id = atom.get("atom_id") or atom.get("id")
        if not task or not atom_id:
            continue
        selected.add((task, atom_id))
        rows.append({"task": task, "atom_id": atom_id, **{k: v for k, v in atom.items() if k not in {"task", "atom_id", "id"}}})
    return selected, rows


def _load_atom_rules() -> dict[str, dict[str, dict]]:
    formal = _load_candidate_rules(
        FORMAL_CANDIDATE_FILES,
        "formal_fallacies",
        [
            "ff_rf_invalid_converse_bundle_v1",
            "ff_role_universal_negative_complement_strict_v1",
            "ff_role_valid_neither_exhaustive_contrapositive_v1",
        ],
        "rule_check_decision_template",
    )
    geometric = _load_candidate_rules(
        GEOMETRIC_CANDIDATE_FILES,
        "geometric_shapes",
        [
            "gen_geo_1_true_gpt41mini_v1",
            "gen_geo_6_false_gpt41mini_repair_v3",
            "gen_geo_9_false_gpt41mini_repair_v3",
        ],
        "rule_check",
    )
    object_rule = json.loads(_resolve(OBJECT_SUMMARY).read_text(encoding="utf-8"))["candidate_rule"]
    object_rule = _rule_for_memory_format(object_rule, "rule_check")
    disambig = {
        cid: _rule_for_memory_format(rule, "rule_check")
        for cid, rule in _load_json_rules(os.environ.get("SFCR_DISAMBIG_RULES", DISAMBIG_RULES)).items()
    }
    web_summary = json.loads(_resolve(WEB_SUMMARY).read_text(encoding="utf-8"))
    web_rule = _load_json_rules(web_summary["candidate_file"])[web_summary["candidate_id"]]
    web_rule = _rule_for_memory_format(web_rule, "rule_check")
    return {
        "formal_fallacies": formal,
        "geometric_shapes": geometric,
        "object_counting": {object_rule["id"]: object_rule},
        "disambiguation_qa": disambig,
        "web_of_lies": {web_rule["id"]: web_rule},
    }


def _task_spec(task: str):
    return _task_spec_from_cfg(_task_cfg(task))


def _build_eval_prompt(task: str, cheatsheet: str, item: dict) -> tuple[str, str]:
    spec = _task_spec(task)
    eval_fn = getattr(spec, "build_eval_prompt", None)
    if eval_fn is not None:
        return eval_fn(cheatsheet, item), "task_spec.build_eval_prompt"
    return spec.build_scoring_prompt(cheatsheet, item, False), "task_spec.build_scoring_prompt"


def _parse_and_score(task: str, raw: str, item: dict) -> tuple[Any, bool, bool]:
    spec = _task_spec(task)
    pred = _parse_prediction(spec, raw)
    parse_error = pred is None
    correct = False if pred is None else bool(spec.is_correct(pred, item))
    return pred, correct, parse_error


def _active_rule_ids(task: str, effect: dict) -> list[str]:
    ids = _active_ids_from_effect(effect)
    if ids:
        if task == "web_of_lies":
            return [next(iter(_load_atom_rules()["web_of_lies"]))]
        return ids
    if task == "object_counting" and _as_bool(effect.get("activated")):
        return ["gen_object_counting_quantity_filter_repair_v2"]
    return []


def _render_rules(anchor: str, rules: list[dict]) -> str:
    rendered = anchor
    for rule in rules:
        rendered = build_cheatsheet_with_rule(rendered, rule)
    return rendered


def _effect_label(anchor_correct: bool, condition_correct: bool, activated: bool) -> str:
    if not activated:
        return "unchanged_correct" if anchor_correct else "unchanged_wrong"
    if (not anchor_correct) and condition_correct:
        return "fixed"
    if anchor_correct and not condition_correct:
        return "regressed"
    if anchor_correct and condition_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def _condition_summary(wide_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    groups: dict[str, list[dict]] = {"__aggregate__": wide_rows}
    for row in wide_rows:
        groups.setdefault(row["task"], []).append(row)
    for task, rows in sorted(groups.items()):
        n = len(rows)
        for condition in CONDITIONS:
            correct = sum(int(r[f"{condition}_correct"]) for r in rows)
            out.append(
                {
                    "task": task,
                    "condition": condition,
                    "n_items": n,
                    "correct": correct,
                    "accuracy": correct / n if n else 0.0,
                    "parse_errors": sum(int(r.get(f"{condition}_parse_error", False)) for r in rows),
                    "activated_count": sum(int(r.get(f"{condition}_activated", False)) for r in rows)
                    if condition.startswith("sfcr_")
                    else "",
                }
            )
    return out


def _paired_comparisons(wide_rows: list[dict]) -> list[dict]:
    pairs = [
        ("raw", "csicl"),
        ("raw", "sfcr_raw_anchor"),
        ("csicl", "sfcr_csicl_anchor"),
        ("csicl", "sfcr_raw_anchor"),
        ("sfcr_raw_anchor", "sfcr_csicl_anchor"),
    ]
    out: list[dict] = []
    groups: dict[str, list[dict]] = {"__aggregate__": wide_rows}
    for row in wide_rows:
        groups.setdefault(row["task"], []).append(row)
    for task, rows in sorted(groups.items()):
        n = len(rows)
        for a, b in pairs:
            a_correct = sum(int(r[f"{a}_correct"]) for r in rows)
            b_correct = sum(int(r[f"{b}_correct"]) for r in rows)
            a_only = sum(int(r[f"{a}_correct"] and not r[f"{b}_correct"]) for r in rows)
            b_only = sum(int((not r[f"{a}_correct"]) and r[f"{b}_correct"]) for r in rows)
            out.append(
                {
                    "task": task,
                    "condition_a": a,
                    "condition_b": b,
                    "n_items": n,
                    "condition_a_correct": a_correct,
                    "condition_b_correct": b_correct,
                    "delta_correct_b_minus_a": b_correct - a_correct,
                    "delta_accuracy_b_minus_a": (b_correct - a_correct) / n if n else 0.0,
                    "a_only_correct_count": a_only,
                    "b_only_correct_count": b_only,
                    "both_correct_count": sum(int(r[f"{a}_correct"] and r[f"{b}_correct"]) for r in rows),
                    "both_wrong_count": sum(int((not r[f"{a}_correct"]) and (not r[f"{b}_correct"])) for r in rows),
                }
            )
    return out


def _negative_tail_table(wide_rows: list[dict]) -> list[dict]:
    refs = [
        ("raw", "csicl"),
        ("raw", "sfcr_raw_anchor"),
        ("raw", "sfcr_csicl_anchor"),
        ("csicl", "sfcr_csicl_anchor"),
    ]
    out: list[dict] = []
    groups: dict[str, list[dict]] = {"__aggregate__": wide_rows}
    for row in wide_rows:
        groups.setdefault(row["task"], []).append(row)
    for task, rows in sorted(groups.items()):
        n = len(rows)
        for ref, condition in refs:
            regress = sum(int(r[f"{ref}_correct"] and not r[f"{condition}_correct"]) for r in rows)
            fix = sum(int((not r[f"{ref}_correct"]) and r[f"{condition}_correct"]) for r in rows)
            out.append(
                {
                    "task": task,
                    "reference": ref,
                    "condition": condition,
                    "n_items": n,
                    f"{ref}_correct_condition_wrong": regress,
                    f"{ref}_wrong_condition_correct": fix,
                    f"net_vs_{ref}": fix - regress,
                    "negative_tail_rate": regress / n if n else 0.0,
                    "repair_rate": fix / n if n else 0.0,
                }
            )
    return out


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _paired_bootstrap(wide_rows: list[dict], n_boot: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    groups: dict[str, list[dict]] = {"__aggregate__": wide_rows}
    for row in wide_rows:
        groups.setdefault(row["task"], []).append(row)
    for task, rows in sorted(groups.items()):
        n = len(rows)
        if not n:
            continue
        for a, b in [
            ("raw", "csicl"),
            ("raw", "sfcr_raw_anchor"),
            ("csicl", "sfcr_csicl_anchor"),
            ("sfcr_raw_anchor", "sfcr_csicl_anchor"),
        ]:
            observed = sum(int(r[f"{b}_correct"]) - int(r[f"{a}_correct"]) for r in rows) / n
            samples: list[float] = []
            for _ in range(n_boot):
                total = 0
                for _ in range(n):
                    r = rows[rng.randrange(n)]
                    total += int(r[f"{b}_correct"]) - int(r[f"{a}_correct"])
                samples.append(total / n)
            le_zero = sum(1 for x in samples if x <= 0.0) / n_boot
            ge_zero = sum(1 for x in samples if x >= 0.0) / n_boot
            out.append(
                {
                    "task": task,
                    "condition_a": a,
                    "condition_b": b,
                    "n_items": n,
                    "n_boot": n_boot,
                    "delta_accuracy_observed_b_minus_a": observed,
                    "ci95_low": _percentile(samples, 0.025),
                    "ci95_high": _percentile(samples, 0.975),
                    "bootstrap_p_two_sided_against_zero": min(1.0, 2.0 * min(le_zero, ge_zero)),
                }
            )
    return out


def _parse_audit(wide_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    groups: dict[str, list[dict]] = {"__aggregate__": wide_rows}
    for row in wide_rows:
        groups.setdefault(row["task"], []).append(row)
    for task, items in sorted(groups.items()):
        for condition in CONDITIONS:
            rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "n_items": len(items),
                    "parse_errors": sum(int(r.get(f"{condition}_parse_error", False)) for r in items),
                    "failed_calls": sum(int(r.get(f"{condition}_failed_call", False)) for r in items),
                }
            )
    return rows


def _protocol_audit(wide_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    groups: dict[str, list[dict]] = {"__aggregate_atom_tasks__": [r for r in wide_rows if r["task"] in ATOM_TASKS]}
    for row in wide_rows:
        if row["task"] in ATOM_TASKS:
            groups.setdefault(row["task"], []).append(row)
    for task, items in sorted(groups.items()):
        rows.append(
            {
                "task": task,
                "n_atom_task_items": len(items),
                "sfcr_raw_anchor_activated_count": sum(int(r["sfcr_raw_anchor_activated"]) for r in items),
                "sfcr_csicl_anchor_activated_count": sum(int(r["sfcr_csicl_anchor_activated"]) for r in items),
                "missing_sfcr_effect_rows": sum(int(r.get("missing_sfcr_effect", False)) for r in items),
                "csicl_anchor_inactive_passthrough_count": sum(
                    int((not r["sfcr_csicl_anchor_activated"]) and r["sfcr_csicl_anchor_answer"] == r["csicl_answer"])
                    for r in items
                ),
                "csicl_anchor_baseline_correct_mismatch_count": 0,
                "allow_mixed_protocol_comparison": False,
            }
        )
    return rows


def _load_csicl_text(root: Path, task: str, model_short: str, seed: int) -> str:
    path = root / task / f"csicl_{model_short}_seed{seed}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--eval-model", default="openai/gpt-4.1-mini")
    p.add_argument("--item-universe", default="runs/emnlp_bbh_unified_raw_csicl_sfcr_strict_20260518/unified_per_item_wide.jsonl")
    p.add_argument("--strict-sfcr-effects", default="runs/emnlp_bbh_strict_sfcr_atom_effects_20260518/strict_sfcr_item_effects.jsonl")
    p.add_argument("--csicl-cheatsheet-root", default="runs/emnlp_bbh_full18_csicl_gpt41mini_20260518/cheatsheets")
    p.add_argument("--csicl-model-short", default="gpt-4.1-mini")
    p.add_argument("--csicl-seed", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--rf-consistency-guard", action="store_true", default=True)
    p.add_argument("--selected-atoms-file", default="", help="Optional JSON atom-selection file; non-selected activated atoms inherit the CS-ICL anchor.")
    p.add_argument("--output-dir", default="runs/emnlp_exp1_csicl_anchor_sfcr_gpt41mini")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("ICR_USE_RF_SCORING", "1")
    api_key = get_api_key()
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    item_universe = _load_wide(_resolve(args.item_universe))
    strict_effects = _load_effects(_resolve(args.strict_sfcr_effects))
    rules_by_task = _load_atom_rules()
    selected_atom_set, selected_atom_rows = _load_selected_atom_set(args.selected_atoms_file)
    item_by_task = {task: _load_task_items(task) for task in ATOM_TASKS}
    csicl_root = _resolve(args.csicl_cheatsheet_root)
    csicl_by_task = {
        task: _load_csicl_text(csicl_root, task, args.csicl_model_short, args.csicl_seed)
        for task in ATOM_TASKS
    }

    jobs: list[dict] = []
    wide_rows: list[dict] = []
    for key in sorted(item_universe):
        task, item_id = key
        src = item_universe[key]
        effect = strict_effects.get(key)
        activated = bool(effect and _as_bool(effect.get("activated")))
        active_ids = _active_rule_ids(task, effect) if activated and task in ATOM_TASKS else []
        if selected_atom_set is not None:
            active_ids = [cid for cid in active_ids if (task, cid) in selected_atom_set]
            activated = activated and bool(active_ids)
        if activated:
            rules = rules_by_task[task]
            selected_rules = [rules.get(cid) for cid in active_ids if cid in rules]
            if task == "web_of_lies" and not selected_rules:
                selected_rules = list(rules.values())
            if task == "object_counting" and not selected_rules:
                selected_rules = list(rules.values())
            if not selected_rules:
                raise SystemExit(f"No active rules resolved for {task}/{item_id}: {active_ids}")
            cheatsheet = _render_rules(csicl_by_task[task], selected_rules)
            item = item_by_task[task][item_id]
            prompt, prompt_builder = _build_eval_prompt(task, cheatsheet, item)
            jobs.append(
                {
                    "task": task,
                    "item_id": item_id,
                    "item": item,
                    "prompt": prompt,
                    "prompt_builder": prompt_builder,
                    "active_candidate_ids": ",".join(rule["id"] for rule in selected_rules),
                    "atom_text_hash": _sha(json.dumps(selected_rules, sort_keys=True, ensure_ascii=False)),
                    "anchor_hash": _sha(csicl_by_task[task]),
                }
            )
        wide_rows.append(
            {
                "task": task,
                "item_id": item_id,
                "gold": src.get("gold"),
                "input_snippet": src.get("input_snippet", ""),
                "raw_answer": src.get("raw_answer"),
                "raw_correct": _as_bool(src.get("raw_correct")),
                "raw_parse_error": _as_bool(src.get("raw_parse_error")),
                "raw_failed_call": False,
                "raw_activated": False,
                "csicl_answer": src.get("csicl_answer"),
                "csicl_correct": _as_bool(src.get("csicl_correct")),
                "csicl_parse_error": _as_bool(src.get("csicl_parse_error")),
                "csicl_failed_call": False,
                "csicl_activated": False,
                "sfcr_raw_anchor_answer": src.get("sfcr_routed_answer"),
                "sfcr_raw_anchor_correct": _as_bool(src.get("sfcr_routed_correct")),
                "sfcr_raw_anchor_parse_error": _as_bool(src.get("sfcr_routed_parse_error")),
                "sfcr_raw_anchor_failed_call": False,
                "sfcr_raw_anchor_activated": _as_bool(src.get("sfcr_activated")),
                "sfcr_raw_anchor_atom_ids": src.get("sfcr_atom_ids", ""),
                "sfcr_csicl_anchor_answer": src.get("csicl_answer"),
                "sfcr_csicl_anchor_correct": _as_bool(src.get("csicl_correct")),
                "sfcr_csicl_anchor_parse_error": False,
                "sfcr_csicl_anchor_failed_call": False,
                "sfcr_csicl_anchor_activated": activated,
                "sfcr_csicl_anchor_atom_ids": ",".join(active_ids),
                "missing_sfcr_effect": task in ATOM_TASKS and effect is None,
            }
        )

    (out_dir / "csicl_anchor_prompts_audit.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "task": job["task"],
                    "item_id": job["item_id"],
                    "prompt_hash": _sha(job["prompt"]),
                    "prompt_chars": len(job["prompt"]),
                    "prompt_builder": job["prompt_builder"],
                    "active_candidate_ids": job["active_candidate_ids"],
                    "anchor_hash": job["anchor_hash"],
                    "atom_text_hash": job["atom_text_hash"],
                },
                ensure_ascii=False,
            )
            for job in jobs
        )
        + "\n",
        encoding="utf-8",
    )

    responses = call_llm_batch(
        [job["prompt"] for job in jobs],
        model=args.eval_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        progress_label="exp1-csicl-anchor-sfcr",
        reasoning_effort=args.reasoning_effort,
    )
    scored: dict[tuple[str, str], dict] = {}
    for job, resp in zip(jobs, responses):
        raw = "" if resp is None else resp.content
        pred, correct, parse_error = (None, False, True) if resp is None else _parse_and_score(job["task"], raw, job["item"])
        fallback = False
        fallback_reason = ""
        if args.rf_consistency_guard and job["task"] == "formal_fallacies" and pred is not None:
            fallback_reason = _rf_inconsistency_reason(raw, pred)
            fallback = bool(fallback_reason)
        scored[(job["task"], job["item_id"])] = {
            "answer": pred,
            "correct": correct,
            "parse_error": parse_error,
            "failed_call": resp is None,
            "raw_response": raw[:800],
            "prompt_hash": _sha(job["prompt"]),
            "prompt_builder": job["prompt_builder"],
            "active_candidate_ids": job["active_candidate_ids"],
            "anchor_hash": job["anchor_hash"],
            "atom_text_hash": job["atom_text_hash"],
            "rf_consistency_fallback": fallback,
            "rf_inconsistency_reason": fallback_reason,
        }

    for row in wide_rows:
        key = (row["task"], row["item_id"])
        if key not in scored:
            continue
        s = scored[key]
        if s["rf_consistency_fallback"]:
            row["sfcr_csicl_anchor_answer"] = row["csicl_answer"]
            row["sfcr_csicl_anchor_correct"] = row["csicl_correct"]
        else:
            row["sfcr_csicl_anchor_answer"] = s["answer"]
            row["sfcr_csicl_anchor_correct"] = _as_bool(s["correct"])
        row["sfcr_csicl_anchor_parse_error"] = _as_bool(s["parse_error"])
        row["sfcr_csicl_anchor_failed_call"] = _as_bool(s["failed_call"])
        row["sfcr_csicl_anchor_atom_ids"] = s["active_candidate_ids"]
        row["sfcr_csicl_anchor_prompt_hash"] = s["prompt_hash"]
        row["sfcr_csicl_anchor_prompt_builder"] = s["prompt_builder"]
        row["sfcr_csicl_anchor_raw_response"] = s["raw_response"]
        row["sfcr_csicl_anchor_rf_consistency_fallback"] = s["rf_consistency_fallback"]
        row["sfcr_csicl_anchor_rf_inconsistency_reason"] = s["rf_inconsistency_reason"]

    long_rows: list[dict] = []
    for row in wide_rows:
        for condition in CONDITIONS:
            anchor_type = "raw" if condition == "sfcr_raw_anchor" else "csicl" if condition == "sfcr_csicl_anchor" else condition
            long_rows.append(
                {
                    "task": row["task"],
                    "item_id": row["item_id"],
                    "condition": condition,
                    "eval_model": args.eval_model,
                    "anchor_type": anchor_type,
                    "answer": row[f"{condition}_answer"],
                    "gold": row["gold"],
                    "correct": row[f"{condition}_correct"],
                    "parse_error": row.get(f"{condition}_parse_error", False),
                    "failed_call": row.get(f"{condition}_failed_call", False),
                    "activated": row.get(f"{condition}_activated", False),
                    "atom_id": row.get(f"{condition}_atom_ids", ""),
                    "source_cache": args.item_universe if condition in {"raw", "csicl"} else args.strict_sfcr_effects,
                    "prompt_hash": row.get(f"{condition}_prompt_hash", ""),
                    "parser_version": "task_spec.parse_verdict",
                    "rf_scoring": os.environ.get("ICR_USE_RF_SCORING") == "1",
                    "reasoning_effort": args.reasoning_effort,
                    "retry_policy": "utils.llm_client.call_llm_batch_default",
                    "input_snippet": row.get("input_snippet", ""),
                }
            )
        row["sfcr_raw_anchor_effect_vs_raw"] = _effect_label(
            row["raw_correct"], row["sfcr_raw_anchor_correct"], row["sfcr_raw_anchor_activated"]
        )
        row["sfcr_csicl_anchor_effect_vs_csicl"] = _effect_label(
            row["csicl_correct"], row["sfcr_csicl_anchor_correct"], row["sfcr_csicl_anchor_activated"]
        )

    condition_summary = _condition_summary(wide_rows)
    paired = _paired_comparisons(wide_rows)
    negative_tail = _negative_tail_table(wide_rows)
    bootstrap = _paired_bootstrap(wide_rows, args.bootstrap_samples, args.seed)
    parse_audit = _parse_audit(wide_rows)
    protocol_audit = _protocol_audit(wide_rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "exp1_csicl_anchor_sfcr",
        "eval_model": args.eval_model,
        "item_universe": str(_resolve(args.item_universe)),
        "strict_sfcr_effects": str(_resolve(args.strict_sfcr_effects)),
        "csicl_cheatsheet_root": str(csicl_root),
        "conditions": CONDITIONS,
        "n_items": len(wide_rows),
        "n_scored_activated_items": len(jobs),
        "selected_atoms_file": str(_resolve(args.selected_atoms_file)) if args.selected_atoms_file else "",
        "selected_atoms": selected_atom_rows,
        "concurrency": args.concurrency,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }

    _write_jsonl(out_dir / "per_item_long.jsonl", long_rows)
    _write_csv(out_dir / "per_item_long.csv", long_rows)
    _write_jsonl(out_dir / "per_item_wide.jsonl", wide_rows)
    _write_csv(out_dir / "per_item_wide.csv", wide_rows)
    _write_json(out_dir / "condition_summary.json", condition_summary)
    _write_csv(out_dir / "condition_summary.csv", condition_summary)
    _write_json(out_dir / "paired_comparisons.json", paired)
    _write_csv(out_dir / "paired_comparisons.csv", paired)
    _write_json(out_dir / "paired_bootstrap.json", bootstrap)
    _write_json(out_dir / "negative_tail_table.json", negative_tail)
    _write_csv(out_dir / "negative_tail_table.csv", negative_tail)
    _write_csv(out_dir / "parse_audit.csv", parse_audit)
    _write_json(out_dir / "protocol_audit.json", protocol_audit)
    _write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps({"output_dir": str(out_dir), "n_items": len(wide_rows), "n_scored": len(jobs)}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
