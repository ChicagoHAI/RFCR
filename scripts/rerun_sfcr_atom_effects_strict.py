"""Strict single-process rerun of routed SF-CR atom effects.

This script reruns only the activated items for the currently accepted
atom-enabled BBH tasks. It fixes one raw baseline cache, one item universe, and
one parser/prompt policy for the rerun. Non-activated items inherit the fixed
raw baseline exactly.

The goal is to produce protocol-clean SF-CR item effects that can be consumed by
``build_sfcr_unified_paper_table.py`` without raw-baseline mismatch artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ICR_sfcr.rule_validator import build_cheatsheet_with_rule  # noqa: E402
from scripts.evaluate_sfcr_atom_union import _build_prompt as _sfcr_build_prompt  # noqa: E402
from scripts.evaluate_sfcr_atom_union import _render_union_cheatsheet  # noqa: E402
from scripts.replay_sfcr_candidates import (  # noqa: E402
    _load_candidates,
    _load_task,
    _rf_inconsistency_reason,
    _rule_for_memory_format,
)
from scripts.run_controlled_full_benchmark_pilot import (  # noqa: E402
    _safe_item_id,
    _task_spec_from_cfg,
)
from scripts.run_disambiguation_generated_rule_pilot import (  # noqa: E402
    _candidate_prompt as _disambig_prompt,
)
from scripts.run_disambiguation_generated_rule_pilot import (  # noqa: E402
    _feature_tags as _disambig_features,
)
from scripts.run_disambiguation_generated_rule_pilot import (  # noqa: E402
    _parse_option as _parse_disambig,
)
from scripts.run_object_counting_generated_rule_pilot import (  # noqa: E402
    _candidate_prompt as _object_prompt,
)
from scripts.run_object_counting_generated_rule_pilot import _parse_int  # noqa: E402
from scripts.run_sfcr_emnlp_baseline_surface import _task_cfg  # noqa: E402
from scripts.validate_web_of_lies_signature_routes_heldout import (  # noqa: E402
    _candidate_prompt as _web_prompt,
)
from scripts.validate_web_of_lies_signature_routes_heldout import (  # noqa: E402
    _parse_yesno,
    _signature,
)
from utils.data import load_jsonl  # noqa: E402
from utils.llm_client import call_llm_batch, get_api_key  # noqa: E402
from utils.scorer import SCORING_MAX_TOKENS  # noqa: E402


ATOM_TASKS = [
    "formal_fallacies",
    "geometric_shapes",
    "object_counting",
    "disambiguation_qa",
    "web_of_lies",
]


DEFAULT_EFFECTS = {
    "formal_fallacies": "runs/ff_gpt41mini_safe_atom_union_v3_test100_guard_rerun/atom_union_item_effects.jsonl",
    "geometric_shapes": "runs/geometric_shapes_generated_safe_union_v4_gpt41mini_20260518/atom_union_item_effects.jsonl",
    "object_counting": "runs/object_counting_generated_rule_routed_distractor_gpt41mini_20260518/atom_union_item_effects.jsonl",
    "disambiguation_qa": "runs/disambiguation_generated_rule_routed_repeat_no_their_gpt41mini_20260518/item_effects.jsonl",
    "web_of_lies": "runs/web_of_lies_signature_routes_heldout_gpt41mini_20260518/atom_union_item_effects.jsonl",
}


FORMAL_CANDIDATE_FILES = [
    "runs/ff_gpt41mini_rfprompt_v5_ruleaware_bundle_v3_test100_consistency_guard/candidates.json",
    "experiments/formal_fallacies_safe_atom_union_candidates.json",
    "experiments/formal_fallacies_role_aware_subatoms.json",
]


GEOMETRIC_CANDIDATE_FILES = [
    "runs/geometric_shapes_generated_rules_gpt41mini_small_20260518/generated_rules_repaired_v4.json",
    "runs/geometric_shapes_generated_rules_gpt41mini_small_20260518/generated_rules_repaired_v3.json",
]


OBJECT_SUMMARY = "runs/object_counting_generated_rule_routed_distractor_gpt41mini_20260518/atom_union_summary.json"
DISAMBIG_RULES = "runs/disambiguation_generated_rule_gpt41mini_20260518/generated_rules.json"
WEB_SUMMARY = "runs/web_of_lies_signature_routes_heldout_gpt41mini_20260518/atom_union_summary.json"


@dataclass
class ActiveJob:
    task: str
    item_id: str
    item: dict
    prompt: str
    parser: str
    active_ids: str


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
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _input_key(text: Any) -> str:
    return " ".join(str(text or "").split()).lower()[:180]


def _effect_lookup(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for row in _read_jsonl(path):
        task = row.get("task") or _task_from_item_id(str(row.get("item_id", "")))
        enriched = {**row, "_source_path": str(path)}
        out[(task, str(row["item_id"]))] = enriched
        key = _input_key(row.get("input_snippet") or row.get("input") or row.get("sentence"))
        if key:
            out[(task, f"__input__:{key}")] = enriched
    return out


def _task_from_item_id(item_id: str) -> str:
    for task in ATOM_TASKS:
        if item_id.startswith(task + "_"):
            return task
    return ""


def _lookup_effect(task: str, item_id: str, item: dict, raw_row: dict, effects: dict[tuple[str, str], dict]) -> dict | None:
    row = effects.get((task, item_id))
    if row is not None:
        return row
    key = _input_key(raw_row.get("input_snippet") or item.get("input"))
    if key:
        return effects.get((task, f"__input__:{key}"))
    return None


def _load_raw_cache(baseline_surface_items: Path, formal_effect_path: Path) -> dict[tuple[str, str], dict]:
    raw: dict[tuple[str, str], dict] = {}
    for row in _read_jsonl(baseline_surface_items):
        if row.get("condition") != "raw":
            continue
        raw[(row["task"], row["item_id"])] = {
            "task": row["task"],
            "item_id": row["item_id"],
            "answer": row.get("answer"),
            "gold": row.get("gold"),
            "correct": _as_bool(row.get("correct")),
            "parse_error": _as_bool(row.get("parse_error")),
            "raw_response": row.get("raw_response", ""),
            "input_snippet": row.get("input_snippet", ""),
            "source": str(baseline_surface_items),
        }

    # The full18 raw pilot reused the formal_fallacies atom summary and did not
    # emit formal raw rows. Use the validated formal atom-effect baseline rows;
    # this source had zero mismatch in the previous audit.
    for row in _read_jsonl(formal_effect_path):
        item_id = row["item_id"]
        raw.setdefault(
            ("formal_fallacies", item_id),
            {
                "task": "formal_fallacies",
                "item_id": item_id,
                "answer": row.get("baseline_answer"),
                "gold": row.get("gold_answer"),
                "correct": _as_bool(row.get("baseline_correct")),
                "parse_error": False,
                "raw_response": "",
                "input_snippet": row.get("input_snippet", ""),
                "source": str(formal_effect_path),
            },
        )
    return raw


def _load_task_items(task: str) -> dict[str, dict]:
    cfg = _task_cfg(task)
    task_spec = _task_spec_from_cfg(cfg)
    items = load_jsonl(REPO_ROOT / cfg["test_jsonl"])
    return {_safe_item_id(task, item, idx): item for idx, item in enumerate(items)}


def _load_candidate_rules(paths: list[str], task: str, ids: list[str], memory_format: str) -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    for raw_path in paths:
        path = _resolve(raw_path)
        if not path.exists():
            continue
        for rule in _load_candidates(path, task):
            loaded.setdefault(rule["id"], rule)
    missing = [cid for cid in ids if cid not in loaded]
    if missing:
        raise SystemExit(f"Missing {task} candidate rule(s): {missing}")
    return {cid: _rule_for_memory_format(loaded[cid], memory_format) for cid in ids}


def _load_json_rules(path: str) -> dict[str, dict]:
    data = json.loads(_resolve(path).read_text(encoding="utf-8"))
    return {rule["id"]: rule for rule in (data.get("rules") or data.get("candidates") or [])}


def _render_rule_prompt(prefix: str, rule: dict, item: dict) -> str:
    return f"""\
You are solving a BBH {prefix} item.

SF-CR RULE:
{rule.get('rule', '')}

USE WHEN:
{rule.get('use_when', '')}

DO NOT USE WHEN:
{rule.get('do_not_use_when', '')}

CHECK:
{rule.get('check', '')}

Question:
{item['input']}

Reply with ONLY the final VERDICT line.
"""


def _active_ids_from_effect(effect: dict | None) -> list[str]:
    if not effect:
        return []
    if "active_candidate_ids" in effect and effect.get("active_candidate_ids"):
        return [x.strip() for x in str(effect["active_candidate_ids"]).split(",") if x.strip()]
    if effect.get("candidate_id"):
        return [str(effect["candidate_id"])]
    if effect.get("signature") and _as_bool(effect.get("activated")):
        return [str(effect.get("signature"))]
    return []


def _is_activated(task: str, effect: dict | None) -> bool:
    if not effect:
        return False
    if task == "object_counting":
        return _as_bool(effect.get("routed_activated"))
    return _as_bool(effect.get("activated"))


def _build_formal_prompt(active_ids: list[str], rules: dict[str, dict], item: dict) -> str:
    task_spec = _load_task("formal_fallacies")
    active_rules = [rules[cid] for cid in active_ids]
    cheatsheet = _render_union_cheatsheet("", active_rules)
    return _sfcr_build_prompt(task_spec, cheatsheet, item, True)


def _build_geometric_prompt(active_ids: list[str], rules: dict[str, dict], item: dict) -> str:
    task_spec = _load_task("geometric_shapes")
    active_rules = [rules[cid] for cid in active_ids]
    cheatsheet = _render_union_cheatsheet("", active_rules)
    return _sfcr_build_prompt(task_spec, cheatsheet, item, True)


def _parse_response(task: str, text: str, item: dict) -> tuple[Any, bool]:
    if task in {"formal_fallacies", "geometric_shapes"}:
        task_spec = _load_task(task)
        pred = task_spec.parse_verdict(text)
        return pred, bool(task_spec.is_correct(pred, item)) if pred is not None else False
    if task == "object_counting":
        pred = _parse_int(text)
        return pred, pred == str(item["answer"]).strip() if pred is not None else False
    if task == "disambiguation_qa":
        pred = _parse_disambig(text)
        return pred, pred == str(item["answer"]).strip() if pred is not None else False
    if task == "web_of_lies":
        pred = _parse_yesno(text)
        return pred, pred == str(item["answer"]).strip() if pred is not None else False
    raise KeyError(task)


def _effect_label(raw_correct: bool, candidate_correct: bool, activated: bool) -> str:
    if not activated:
        return "unchanged_correct" if raw_correct else "unchanged_wrong"
    if (not raw_correct) and candidate_correct:
        return "fixed"
    if raw_correct and not candidate_correct:
        return "regressed"
    if raw_correct and candidate_correct:
        return "unchanged_correct"
    return "unchanged_wrong"


def _prepare_jobs(args: argparse.Namespace, raw_cache: dict[tuple[str, str], dict]) -> tuple[list[ActiveJob], dict[str, list[dict]]]:
    all_rows: dict[str, list[dict]] = {task: [] for task in ATOM_TASKS}
    jobs: list[ActiveJob] = []
    effect_maps = {task: _effect_lookup(_resolve(path)) for task, path in DEFAULT_EFFECTS.items()}

    formal_rules = _load_candidate_rules(
        FORMAL_CANDIDATE_FILES,
        "formal_fallacies",
        [
            "ff_rf_invalid_converse_bundle_v1",
            "ff_role_universal_negative_complement_strict_v1",
            "ff_role_valid_neither_exhaustive_contrapositive_v1",
        ],
        "rule_check_decision_template",
    )
    geometric_rules = _load_candidate_rules(
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
    disambig_rules = _load_json_rules(DISAMBIG_RULES)
    web_summary = json.loads(_resolve(WEB_SUMMARY).read_text(encoding="utf-8"))
    web_rule = _load_json_rules(web_summary["candidate_file"])[web_summary["candidate_id"]]

    for task in ATOM_TASKS:
        item_by_id = _load_task_items(task)
        for item_id in sorted(iid for (t, iid) in raw_cache if t == task):
            raw = raw_cache[(task, item_id)]
            item = item_by_id[item_id]
            effect = _lookup_effect(task, item_id, item, raw, effect_maps[task])
            activated = _is_activated(task, effect)
            active_ids = _active_ids_from_effect(effect)
            row = {
                "task": task,
                "item_id": item_id,
                "activated": activated,
                "active_candidate_ids": ",".join(active_ids),
                "baseline_answer": raw.get("answer"),
                "gold_answer": raw.get("gold") or item.get("answer"),
                "baseline_correct": bool(raw.get("correct")),
                "baseline_parse_error": bool(raw.get("parse_error")),
                "baseline_source": raw.get("source"),
                "input_snippet": raw.get("input_snippet") or " ".join(str(item.get("input", "")).split())[:240],
                "raw_cache_protocol": "fixed_full18_raw_cache",
            }
            all_rows[task].append(row)
            if not activated:
                continue

            if task == "formal_fallacies":
                prompt = _build_formal_prompt(active_ids, formal_rules, item)
                parser = "task_spec.parse_verdict"
            elif task == "geometric_shapes":
                prompt = _build_geometric_prompt(active_ids, geometric_rules, item)
                parser = "task_spec.parse_verdict"
            elif task == "object_counting":
                prompt = _object_prompt(object_rule, item)
                parser = "object_counting_integer"
            elif task == "disambiguation_qa":
                cid = active_ids[0]
                prompt = _disambig_prompt(disambig_rules[cid], item)
                parser = "disambiguation_option"
            elif task == "web_of_lies":
                prompt = _web_prompt(web_rule, item)
                parser = "web_of_lies_yes_no"
            else:
                raise KeyError(task)
            jobs.append(ActiveJob(task=task, item_id=item_id, item=item, prompt=prompt, parser=parser, active_ids=",".join(active_ids)))
    return jobs, all_rows


def _score_jobs(jobs: list[ActiveJob], args: argparse.Namespace, api_key: str) -> dict[tuple[str, str], dict]:
    prompts = [job.prompt for job in jobs]
    responses = call_llm_batch(
        prompts,
        model=args.model_score,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        progress_label="sfcr-strict-atom-rerun",
        reasoning_effort=args.reasoning_effort,
    )
    out: dict[tuple[str, str], dict] = {}
    for job, resp in zip(jobs, responses):
        raw = "" if resp is None else resp.content
        pred, correct = (None, False) if resp is None else _parse_response(job.task, raw, job.item)
        fallback = False
        fallback_reason = ""
        if args.rf_consistency_guard and job.task == "formal_fallacies" and pred is not None:
            fallback_reason = _rf_inconsistency_reason(raw, pred)
            fallback = bool(fallback_reason)
        out[(job.task, job.item_id)] = {
            "candidate_answer": pred,
            "candidate_correct": correct,
            "candidate_parse_error": pred is None,
            "candidate_raw_response": raw[:800],
            "parser": job.parser,
            "rf_consistency_fallback": fallback,
            "rf_inconsistency_reason": fallback_reason,
            "active_candidate_ids": job.active_ids,
        }
    return out


def _finalize_rows(all_rows: dict[str, list[dict]], scored: dict[tuple[str, str], dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    by_task: dict[str, list[dict]] = {}
    combined: list[dict] = []
    for task, rows in all_rows.items():
        out_rows = []
        for row in rows:
            item_id = row["item_id"]
            activated = bool(row["activated"])
            raw_correct = bool(row["baseline_correct"])
            scored_row = scored.get((task, item_id))
            if not activated:
                candidate_answer = row["baseline_answer"]
                candidate_correct = raw_correct
                parse_error = False
                raw_response = ""
                parser = "raw_passthrough"
                fallback = False
                fallback_reason = ""
            else:
                candidate_answer = scored_row["candidate_answer"]
                candidate_correct = bool(scored_row["candidate_correct"])
                parse_error = bool(scored_row["candidate_parse_error"])
                raw_response = scored_row["candidate_raw_response"]
                parser = scored_row["parser"]
                fallback = bool(scored_row["rf_consistency_fallback"])
                fallback_reason = scored_row["rf_inconsistency_reason"]
                if fallback:
                    candidate_answer = row["baseline_answer"]
                    candidate_correct = raw_correct
            effect = _effect_label(raw_correct, candidate_correct, activated)
            normalized = {
                **row,
                "union_answer": candidate_answer,
                "union_correct": candidate_correct,
                "candidate_answer": candidate_answer,
                "candidate_correct": candidate_correct,
                "parse_error": parse_error,
                "candidate_parse_error": parse_error,
                "candidate_raw_response": raw_response,
                "parser": parser,
                "effect": effect,
                "rf_consistency_fallback": fallback,
                "rf_inconsistency_reason": fallback_reason,
            }
            out_rows.append(normalized)
            combined.append(normalized)
        by_task[task] = out_rows
    return by_task, combined


def _summary(task: str, rows: list[dict], args: argparse.Namespace) -> dict:
    n = len(rows)
    baseline_correct = sum(1 for r in rows if r["baseline_correct"])
    union_correct = sum(1 for r in rows if r["union_correct"])
    fixed = [r["item_id"] for r in rows if r["effect"] == "fixed"]
    regressed = [r["item_id"] for r in rows if r["effect"] == "regressed"]
    parse_errors = [r["item_id"] for r in rows if r["parse_error"]]
    return {
        "task": task,
        "model_score": args.model_score,
        "n_items": n,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / n if n else 0.0,
        "union_correct": union_correct,
        "union_accuracy": union_correct / n if n else 0.0,
        "fixed_count": len(fixed),
        "regression_count": len(regressed),
        "net_count": len(fixed) - len(regressed),
        "activated_count": sum(1 for r in rows if r["activated"]),
        "parse_error_count": len(parse_errors),
        "fixed_item_ids": fixed,
        "regressed_item_ids": regressed,
        "parse_error_item_ids": parse_errors,
        "protocol_note": "Strict single-process rerun: raw baseline fields come from one fixed raw cache; only activated SF-CR items were rescored.",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--baseline-surface-items", default="runs/emnlp_bbh_full18_csicl_gpt41mini_20260518/per_item_outputs.jsonl")
    p.add_argument("--model-score", default="openai/gpt-4.1-mini")
    p.add_argument("--max-tokens", type=int, default=SCORING_MAX_TOKENS)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--rf-consistency-guard", action="store_true", default=True)
    p.add_argument("--output-dir", default="runs/emnlp_bbh_strict_sfcr_atom_effects_20260518")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("ICR_USE_RF_SCORING", "1")
    api_key = get_api_key()
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_cache = _load_raw_cache(
        _resolve(args.baseline_surface_items),
        _resolve(DEFAULT_EFFECTS["formal_fallacies"]),
    )
    jobs, all_rows = _prepare_jobs(args, raw_cache)
    (out_dir / "strict_rerun_prompts_audit.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "task": job.task,
                    "item_id": job.item_id,
                    "parser": job.parser,
                    "active_candidate_ids": job.active_ids,
                    "prompt_chars": len(job.prompt),
                },
                ensure_ascii=False,
            )
            for job in jobs
        )
        + "\n",
        encoding="utf-8",
    )
    scored = _score_jobs(jobs, args, api_key)
    by_task, combined = _finalize_rows(all_rows, scored)

    summaries = []
    for task, rows in by_task.items():
        task_dir = out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        summary = _summary(task, rows, args)
        summaries.append(summary)
        _write_json(task_dir / "atom_union_summary.json", summary)
        _write_jsonl(task_dir / "atom_union_item_effects.jsonl", rows)
        _write_csv(task_dir / "atom_union_item_effects.csv", rows)
        _write_csv(task_dir / "atom_union_summary.csv", [summary])

    aggregate = {
        "task": "__aggregate_atom_tasks__",
        "model_score": args.model_score,
        "n_items": sum(s["n_items"] for s in summaries),
        "baseline_correct": sum(s["baseline_correct"] for s in summaries),
        "union_correct": sum(s["union_correct"] for s in summaries),
        "fixed_count": sum(s["fixed_count"] for s in summaries),
        "regression_count": sum(s["regression_count"] for s in summaries),
        "net_count": sum(s["net_count"] for s in summaries),
        "activated_count": sum(s["activated_count"] for s in summaries),
        "parse_error_count": sum(s["parse_error_count"] for s in summaries),
        "rescored_activated_items": len(jobs),
    }
    aggregate["baseline_accuracy"] = aggregate["baseline_correct"] / aggregate["n_items"]
    aggregate["union_accuracy"] = aggregate["union_correct"] / aggregate["n_items"]
    manifest = {
        "output_dir": str(out_dir),
        "baseline_surface_items": str(_resolve(args.baseline_surface_items)),
        "model_score": args.model_score,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "rf_consistency_guard": args.rf_consistency_guard,
        "default_effect_sources": DEFAULT_EFFECTS,
        "task_summaries": summaries,
        "aggregate": aggregate,
    }
    _write_json(out_dir / "strict_rerun_manifest.json", manifest)
    _write_json(out_dir / "strict_rerun_summary.json", [aggregate] + summaries)
    _write_csv(out_dir / "strict_rerun_summary.csv", [aggregate] + summaries)
    _write_jsonl(out_dir / "strict_sfcr_item_effects.jsonl", combined)
    _write_csv(out_dir / "strict_sfcr_item_effects.csv", combined)
    print(json.dumps({"output_dir": str(out_dir), "aggregate": aggregate}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
