"""Evaluate a routed union of SF-CR memory atoms.

This script is intentionally narrower than the training pipeline. It loads
already-generated candidate atoms, routes each item to zero or more atoms, scores
only activated items with the activated atom text, and reports the final
anchor-relative effect of the routed union.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ICR_sfcr.activation import route_candidate
from ICR_sfcr.rule_validator import build_cheatsheet_with_rule
from ICR_sfcr.scoring_cache import rf_scoring_enabled
from scripts.replay_sfcr_candidates import (
    _item_id,
    _load_candidates,
    _load_task,
    _rf_inconsistency_reason,
    _rule_for_memory_format,
    _tag_items,
)
from utils.data import load_jsonl
from utils.llm_client import call_llm_batch, get_api_key
from utils.scorer import SCORING_MAX_TOKENS


def _candidate_paths(raw_values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                paths.append(Path(part))
    return paths


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row if not isinstance(row.get(k), (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_baseline(path: Path, items: list[dict]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item = row.get("item") or {}
        iid = str(item.get("_sfcr_id") or item.get("id") or row.get("item_id") or "")
        if not iid:
            continue
        rows[iid] = row

    missing = [_item_id(item) for item in items if _item_id(item) not in rows]
    if missing:
        raise SystemExit(f"Baseline cache is missing {len(missing)} items, e.g. {missing[:5]}")
    return rows


def _merge_baseline_item(item: dict, baseline_row: dict) -> tuple[bool, dict]:
    cached_item = baseline_row.get("item") or {}
    merged = {
        **item,
        "predicted": baseline_row.get("answer", cached_item.get("predicted")),
        "expected": baseline_row.get("gold", cached_item.get("expected") or item.get("answer")),
        "raw_response": baseline_row.get("raw_output", cached_item.get("raw_response", "")),
    }
    return bool(baseline_row.get("baseline_correct")), merged


def _render_union_cheatsheet(anchor: str, active_rules: list[dict]) -> str:
    rendered = anchor
    for rule in active_rules:
        rendered = build_cheatsheet_with_rule(rendered, rule)
    return rendered


def _build_prompt(task_spec, cheatsheet: str, item: dict, cot_first: bool) -> str:
    rf_builder = getattr(task_spec, "build_scoring_prompt_rf", None)
    if rf_scoring_enabled() and rf_builder is not None:
        try:
            return rf_builder(cheatsheet, item, cot_first)
        except TypeError:
            return rf_builder(cheatsheet, item)
    return task_spec.build_scoring_prompt(cheatsheet, item, cot_first)


def _score_union_items(items: list[dict], prompts: list[str], args, task_spec, api_key: str) -> dict[str, dict]:
    responses = call_llm_batch(
        prompts,
        model=args.model_score,
        api_key=api_key,
        temperature=0.0,
        max_tokens=SCORING_MAX_TOKENS,
        concurrency=args.concurrency,
        progress_label="sfcr-atom-union",
        reasoning_effort=args.reasoning_effort,
    )
    out: dict[str, dict] = {}
    for item, resp in zip(items, responses):
        iid = _item_id(item)
        if resp is None:
            out[iid] = {
                **item,
                "predicted": None,
                "expected": task_spec.answer_label(item),
                "raw_response": "",
                "parse_error": True,
                "correct": False,
            }
            continue
        predicted = task_spec.parse_verdict(resp.content)
        annotated = {
            **item,
            "predicted": predicted,
            "expected": task_spec.answer_label(item),
            "raw_response": resp.content,
            "post_think": task_spec.extract_post_think(resp.content),
            "thinking": resp.thinking,
            "parse_error": predicted is None,
        }
        annotated["correct"] = task_spec.is_correct(predicted, item)
        out[iid] = annotated
    return out


def _effect_label(baseline_correct: bool, union_correct: bool, activated: bool) -> str:
    if not activated:
        return "unchanged_correct" if baseline_correct else "unchanged_wrong"
    if (not baseline_correct) and union_correct:
        return "fixed"
    if baseline_correct and not union_correct:
        return "regressed"
    if baseline_correct and union_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", default="formal_fallacies", choices=["formal_fallacies", "causal_judgement", "geometric_shapes"])
    p.add_argument(
        "--candidate-file",
        required=True,
        action="append",
        help="Candidate JSON/YAML file. May be passed multiple times or as a comma-separated list.",
    )
    p.add_argument("--candidate-ids", required=True, help="Comma-separated candidate IDs to include in the atom union.")
    p.add_argument("--baseline-cache", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--anchor-cheatsheet", required=True)
    p.add_argument("--model-score", default="openai/gpt-4.1-mini")
    p.add_argument("--activation-routing-mode", default="hybrid", choices=["partition", "feature", "hybrid"])
    p.add_argument("--memory-format", default="rule_check_decision_template", choices=[
        "rule_only",
        "rule_check",
        "rule_check_micro_example",
        "rule_check_decision_template",
    ])
    p.add_argument("--rf-consistency-guard", action="store_true")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--cot-first", action="store_true", default=True)
    p.add_argument("--no-cot-first", dest="cot_first", action="store_false")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = get_api_key()
    task_spec = _load_task(args.task)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(Path(args.dataset))
    if args.limit:
        items = items[: args.limit]
    _tag_items(items)

    baseline_rows = _load_baseline(Path(args.baseline_cache), items)
    baseline_correct_by_id: dict[str, bool] = {}
    baseline_items: dict[str, dict] = {}
    for item in items:
        correct, merged = _merge_baseline_item(item, baseline_rows[_item_id(item)])
        baseline_correct_by_id[_item_id(item)] = correct
        baseline_items[_item_id(item)] = merged

    selected_ids = {x.strip() for x in args.candidate_ids.split(",") if x.strip()}
    loaded = []
    for path in _candidate_paths(args.candidate_file):
        loaded.extend(_load_candidates(path, args.task))
    deduped = {}
    for rule in loaded:
        deduped.setdefault(rule["id"], rule)
    candidates = [
        _rule_for_memory_format(rule, args.memory_format)
        for rule in deduped.values()
        if rule["id"] in selected_ids
    ]
    missing = selected_ids - {rule["id"] for rule in candidates}
    if missing:
        raise SystemExit(f"Missing candidate IDs: {sorted(missing)}")

    anchor_path = Path(args.anchor_cheatsheet)
    if not anchor_path.exists() and anchor_path.with_suffix(".txt").exists():
        anchor_path = anchor_path.with_suffix(".txt")
    anchor_text = anchor_path.read_text(encoding="utf-8")

    active_items: list[dict] = []
    active_prompts: list[str] = []
    active_rules_by_id: dict[str, list[dict]] = {}
    route_log_by_id: dict[str, list[dict]] = {}
    activation_counts = {rule["id"]: 0 for rule in candidates}
    overlap_count = 0

    for item in items:
        iid = _item_id(item)
        active_rules: list[dict] = []
        route_logs: list[dict] = []
        for rule in candidates:
            decision = route_candidate(
                rule,
                item,
                activation_routing_mode=args.activation_routing_mode,
                task_spec=task_spec,
            )
            route_logs.append({
                "candidate_id": rule["id"],
                "activated": decision.activated,
                "vetoed": decision.vetoed,
                "activation_reason": decision.activation_reason,
                "matched_positive_tags": decision.matched_positive_tags,
                "matched_negative_tags": decision.matched_negative_tags,
            })
            if decision.activated:
                active_rules.append(rule)
                activation_counts[rule["id"]] += 1
        active_rules_by_id[iid] = active_rules
        route_log_by_id[iid] = route_logs
        if len(active_rules) > 1:
            overlap_count += 1
        if active_rules:
            active_items.append(item)
            active_prompts.append(_build_prompt(
                task_spec,
                _render_union_cheatsheet(anchor_text, active_rules),
                item,
                args.cot_first,
            ))

    scored_active = _score_union_items(active_items, active_prompts, args, task_spec, api_key) if active_items else {}

    item_rows: list[dict] = []
    fixed_ids: list[str] = []
    regressed_ids: list[str] = []
    parse_error_ids: list[str] = []
    fallback_ids: list[str] = []

    for item in items:
        iid = _item_id(item)
        baseline_correct = baseline_correct_by_id[iid]
        baseline_item = baseline_items[iid]
        active_rules = active_rules_by_id[iid]
        activated = bool(active_rules)
        scored = scored_active.get(iid)
        fallback_reason = ""
        consistency_fallback = False
        if activated and scored is not None:
            if scored.get("parse_error"):
                parse_error_ids.append(iid)
            if args.rf_consistency_guard and args.task == "formal_fallacies":
                fallback_reason = _rf_inconsistency_reason(scored.get("raw_response", ""), scored.get("predicted"))
                if fallback_reason:
                    consistency_fallback = True
                    fallback_ids.append(iid)
            if consistency_fallback:
                union_correct = baseline_correct
                union_answer = baseline_item.get("predicted")
            else:
                union_correct = bool(scored.get("correct"))
                union_answer = scored.get("predicted")
        else:
            union_correct = baseline_correct
            union_answer = baseline_item.get("predicted")

        effect = _effect_label(baseline_correct, union_correct, activated)
        if effect == "fixed":
            fixed_ids.append(iid)
        elif effect == "regressed":
            regressed_ids.append(iid)

        route_logs = route_log_by_id[iid]
        matched_positive = sorted({t for log in route_logs for t in log.get("matched_positive_tags", [])})
        matched_negative = sorted({t for log in route_logs for t in log.get("matched_negative_tags", [])})
        item_rows.append({
            "item_id": iid,
            "activated": activated,
            "active_count": len(active_rules),
            "active_candidate_ids": ",".join(rule["id"] for rule in active_rules),
            "baseline_answer": baseline_item.get("predicted"),
            "union_answer": union_answer,
            "gold_answer": baseline_item.get("expected") or item.get("answer"),
            "baseline_correct": baseline_correct,
            "union_correct": union_correct,
            "effect": effect,
            "parse_error": bool(scored.get("parse_error")) if scored else False,
            "rf_consistency_fallback": consistency_fallback,
            "rf_inconsistency_reason": fallback_reason,
            "matched_positive_tags": ",".join(matched_positive),
            "matched_negative_tags": ",".join(matched_negative),
            "input_snippet": " ".join(str(item.get("input", "")).split())[:240],
        })

    n = len(items)
    baseline_correct_n = sum(1 for v in baseline_correct_by_id.values() if v)
    union_correct_n = baseline_correct_n + len(fixed_ids) - len(regressed_ids)
    summary = {
        "output_dir": str(out_dir),
        "task": args.task,
        "dataset": args.dataset,
        "limit": args.limit,
        "model_score": args.model_score,
        "candidate_ids": [rule["id"] for rule in candidates],
        "activation_routing_mode": args.activation_routing_mode,
        "memory_format": args.memory_format,
        "rf_scoring_enabled": rf_scoring_enabled(),
        "rf_consistency_guard": bool(args.rf_consistency_guard),
        "n_items": n,
        "baseline_correct": baseline_correct_n,
        "baseline_accuracy": baseline_correct_n / n if n else 0.0,
        "union_correct": union_correct_n,
        "union_accuracy": union_correct_n / n if n else 0.0,
        "fixed_count": len(fixed_ids),
        "regression_count": len(regressed_ids),
        "net_count": len(fixed_ids) - len(regressed_ids),
        "activated_count": len(active_items),
        "activation_counts_by_atom": activation_counts,
        "overlap_count": overlap_count,
        "parse_error_count": len(parse_error_ids),
        "rf_consistency_fallback_count": len(fallback_ids),
        "fixed_item_ids": fixed_ids,
        "regressed_item_ids": regressed_ids,
        "parse_error_item_ids": parse_error_ids,
        "rf_consistency_fallback_item_ids": fallback_ids,
    }

    (out_dir / "atom_union_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(out_dir / "atom_union_item_effects.jsonl", item_rows)
    _write_csv(out_dir / "atom_union_item_effects.csv", item_rows)
    _write_csv(out_dir / "atom_union_summary.csv", [summary])
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
