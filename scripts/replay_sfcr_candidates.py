"""Replay SF-CR candidates with feature/hybrid routing.

This diagnostic script regenerates no candidates.  It scores a fixed small
formal_fallacies split, loads existing manual/generated candidate files, and
revalidates them with partition, feature, or hybrid activation routing.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ICR_partition.training.partition import build_partitions
from ICR_sfcr.activation import route_candidate
from ICR_sfcr.rule_validator import (
    apply_count_gate,
    build_cheatsheet_with_rule,
    make_item_effect_row,
    write_item_effect_logs,
)
from ICR_sfcr.scoring_cache import (
    CandidateOutputCache,
    PARSER_VERSION,
    assert_protocol_compatible,
    build_cache_row,
    cache_row_to_scored_item,
    candidate_text_hash,
    make_cache_key,
    make_candidate_prompt_hash,
    make_protocol_signature,
    partition_routed_scoring_enabled,
    prompt_template_hash,
    protocol_mismatch_reasons,
    protocol_signature_hash,
    rf_scoring_enabled,
    stable_hash,
)
from utils.data import load_jsonl
from utils.llm_client import get_api_key
from utils.scorer import score_batch


_TASK_MAP = {
    "formal_fallacies": ("tasks.bbh_tasks_ext", "FORMAL_FALLACIES_TASK"),
    "causal_judgement": ("tasks.bbh_tasks", "CAUSAL_JUDGEMENT_TASK"),
    "geometric_shapes": ("tasks.bbh_tasks", "GEOMETRIC_TASK"),
}


DECISION_TEMPLATE = """\
1. Write the premise form, e.g. A -> B.
2. Write the claimed conclusion form.
3. Mark whether the direction was reversed, contraposed, or chained.
4. Output VALID only if the conclusion is forced by the premises."""


def _load_task(name: str):
    module_path, attr = _TASK_MAP[name]
    return getattr(importlib.import_module(module_path), attr)


def _item_id(item: dict, idx: int | None = None) -> str:
    return str(item.get("_sfcr_id") or item.get("id") or item.get("idx") or idx)


def _tag_items(items: list[dict]) -> None:
    for i, item in enumerate(items):
        item.setdefault("_sfcr_id", _item_id(item, i))


def _load_json_or_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text) or {}
        except Exception:
            # Tiny fallback for the simple manual_rules_sfcr.yaml shape.
            rows: list[dict] = []
            current: dict | None = None
            current_list_key: str | None = None
            for raw in text.splitlines():
                line = raw.rstrip()
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("- id:"):
                    if current:
                        rows.append(current)
                    current = {"id": stripped.split(":", 1)[1].strip().strip('"')}
                    current_list_key = None
                elif current is not None and (
                    re_match := __import__("re").match(r"([a-zA-Z_]+):\s*(.*)", stripped)
                ):
                    key, val = re_match.group(1), re_match.group(2).strip()
                    if val:
                        current[key] = val.strip('"')
                        current_list_key = None
                    else:
                        current[key] = []
                        current_list_key = key
                elif current is not None and current_list_key and stripped.startswith("- "):
                    current[current_list_key].append(stripped[2:].strip().strip('"'))
            if current:
                rows.append(current)
            return {"rules": rows}
    return json.loads(text)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _load_candidates(path: Path, task: str) -> list[dict]:
    data = _load_json_or_yaml(path)
    rows = data.get("rules") or data.get("candidates") or []
    candidates: list[dict] = []
    for i, row in enumerate(rows, 1):
        if row.get("task") and row.get("task") != task:
            continue
        rule_text = row.get("rule") or row.get("action") or row.get("title") or row.get("candidate_id")
        if not rule_text:
            continue
        activate_if = _as_list(row.get("activate_if"))
        do_not = _as_list(row.get("do_not_activate_if"))
        candidate = {
            "id": row.get("id") or row.get("candidate_id") or f"candidate_{i}",
            "task": task,
            "rule": str(rule_text),
            "use_when": row.get("use_when") or " | ".join(activate_if),
            "do_not_use_when": row.get("do_not_use_when") or " | ".join(do_not),
            "check": row.get("check", ""),
            "micro_example": row.get("micro_example", ""),
            "decision_template": row.get("decision_template", ""),
            "positive_tags": _as_list(row.get("positive_tags")),
            "positive_tag_groups": row.get("positive_tag_groups") or [],
            "negative_tags": _as_list(row.get("negative_tags")),
            "source_path": str(path),
        }
        if row.get("partition"):
            candidate["partition"] = row["partition"]
        if row.get("routing_key") is not None:
            candidate["routing_key"] = row["routing_key"]
        if row.get("partition_key") is not None:
            candidate["partition_key"] = row["partition_key"]
        candidates.append(candidate)
    return candidates


def _rule_for_memory_format(rule: dict, memory_format: str) -> dict:
    out = copy.deepcopy(rule)
    if memory_format == "rule_only":
        out["check"] = ""
        out["micro_example"] = ""
        out["decision_template"] = ""
    elif memory_format == "rule_check":
        out["micro_example"] = ""
        out["decision_template"] = ""
    elif memory_format == "rule_check_micro_example":
        out["decision_template"] = ""
    elif memory_format == "rule_check_decision_template":
        out["decision_template"] = out.get("decision_template") or DECISION_TEMPLATE
    else:
        raise ValueError(f"unknown memory format: {memory_format}")
    out["memory_format"] = memory_format
    return out


def _protocol_signature(args, anchor_text: str, task_spec) -> dict:
    rf = rf_scoring_enabled()
    return make_protocol_signature(
        task=args.task,
        scorer_model=args.model_score,
        anchor_hash=stable_hash(anchor_text),
        rf_scoring_flag=rf,
        partition_routed_flag=partition_routed_scoring_enabled(),
        cot_first=args.cot_first,
        reasoning_effort=args.reasoning_effort,
        prompt_template_hash_value=prompt_template_hash(task_spec, rf),
        concurrency=args.concurrency,
        parser_version=PARSER_VERSION,
    )


def _parser_rule(raw: str, parsed_answer) -> str:
    if parsed_answer is None:
        return ""
    text = raw or ""
    if re.search(r"\bVERDICT\b", text, re.I):
        return "verdict"
    if re.search(r"\bfinal\s+answer\b", text, re.I):
        return "final_answer"
    if re.search(r"\bnot\s+valid\b", text, re.I):
        return "not_valid"
    if re.search(r"\bargument is\b|\bconclusion\b|\btherefore\b", text, re.I):
        return "conclusion_line"
    return "tail_token"


def _rf_inconsistency_reason(raw: str, parsed_answer) -> str:
    """Detect obvious reasoning/verdict contradictions in formal-fallacies RF output."""
    if parsed_answer not in {"valid", "invalid"}:
        return ""
    text = raw or ""
    reasoning = "\n".join(
        line for line in text.splitlines()
        if not line.strip().upper().startswith("VERDICT:")
    ).lower()
    if not reasoning.strip():
        return ""

    invalid_cues = [
        r"\binvalid\b",
        r"\bnot\s+valid\b",
        r"does\s+not\s+follow",
        r"not\s+forced",
        r"not\s+entailed",
        r"unsupported",
        r"revers(?:e|ed|ing)",
        r"violat(?:e|es|ing)",
        r"affirming\s+the\s+consequent",
        r"denying\s+the\s+antecedent",
        r"insufficient",
        r"missing\s+implication",
        r"unresolved\s+disjunct",
    ]
    valid_cues = [
        r"\bvalid\b",
        r"\bfollows\b",
        r"\bentails\b",
        r"\bforced\b",
        r"modus\s+ponens",
        r"modus\s+tollens",
        r"contrapositive",
        r"deductively\s+valid",
    ]
    has_invalid = any(re.search(pat, reasoning) for pat in invalid_cues)
    has_valid = any(re.search(pat, reasoning) for pat in valid_cues)
    if parsed_answer == "valid" and has_invalid:
        return "reasoning_invalid_verdict_valid"
    if parsed_answer == "invalid" and has_valid and not has_invalid:
        return "reasoning_valid_verdict_invalid"
    return ""


def _record_parser_audit(args, *, label: str, item: dict, annotated: dict, protocol_signature: dict, cache_hit: bool) -> None:
    row = {
        "label": label,
        "item_id": _item_id(item),
        "parsed_answer": annotated.get("predicted"),
        "gold_answer": annotated.get("expected") or item.get("answer"),
        "parse_error": annotated.get("predicted") is None,
        "matched_parser_rule": _parser_rule(annotated.get("raw_response", ""), annotated.get("predicted")),
        "raw_output_snippet": (annotated.get("raw_response", "") or "")[:500],
        "cache_hit": bool(cache_hit),
        "protocol_signature_hash": protocol_signature_hash(protocol_signature),
    }
    args._parser_audit.append(row)


def _write_parser_audit(out_dir: Path, rows: list[dict]) -> None:
    _write_csv(out_dir / "parser_audit.csv", rows)
    with (out_dir / "parse_errors.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            if row.get("parse_error"):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _protocol_from_cache_row(row: dict) -> dict | None:
    sig = row.get("protocol_signature")
    return sig if isinstance(sig, dict) and sig else None


def _load_or_score_baseline(args, items, anchor_text, task_spec, api_key):
    cache = Path(args.baseline_cache) if args.baseline_cache else None
    protocol = _protocol_signature(args, anchor_text, task_spec)
    args._baseline_protocol_signature = protocol
    if cache and cache.exists():
        correct, wrong = [], []
        seen_protocol: dict | None = None
        missing_protocol = False
        for line in cache.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_protocol = _protocol_from_cache_row(row)
            if row_protocol is None:
                missing_protocol = True
            elif seen_protocol is None:
                seen_protocol = row_protocol
            item = row["item"]
            item.setdefault("predicted", row.get("answer", item.get("predicted")))
            item.setdefault("expected", row.get("gold", item.get("expected") or item.get("answer")))
            if row_protocol:
                item["_protocol_signature"] = row_protocol
            annotated = {
                **item,
                "predicted": row.get("answer", item.get("predicted")),
                "expected": row.get("gold", item.get("expected") or item.get("answer")),
                "raw_response": row.get("raw_output", item.get("raw_response", "")),
            }
            _record_parser_audit(
                args,
                label="baseline",
                item=item,
                annotated=annotated,
                protocol_signature=row_protocol or {},
                cache_hit=True,
            )
            if row["baseline_correct"]:
                correct.append(item)
            else:
                wrong.append(item)
        if args.require_consistent_scoring_protocol:
            if missing_protocol:
                raise SystemExit(
                    f"Baseline cache {cache} has no protocol signatures; use a fresh cache path or allow mixed protocol comparison."
                )
            assert_protocol_compatible(
                seen_protocol,
                protocol,
                allow_mixed_protocol_comparison=args.allow_mixed_protocol_comparison,
            )
        return correct, wrong

    correct, wrong = score_batch(
        items,
        anchor_text,
        args.model_score,
        api_key,
        concurrency=args.concurrency,
        reasoning_effort=args.reasoning_effort,
        cot_first=args.cot_first,
        progress_label="sfcr-replay-baseline",
        task_spec=task_spec,
    )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", encoding="utf-8") as fh:
            for item in correct:
                _record_parser_audit(args, label="baseline", item=item, annotated=item, protocol_signature=protocol, cache_hit=False)
                fh.write(json.dumps({
                    "baseline_correct": True,
                    "item": {**item, "_protocol_signature": protocol},
                    "answer": item.get("predicted"),
                    "gold": item.get("expected") or item.get("answer"),
                    "parse_error": item.get("predicted") is None,
                    "protocol_signature": protocol,
                    "protocol_signature_hash": protocol_signature_hash(protocol),
                }, ensure_ascii=False) + "\n")
            for item in wrong:
                _record_parser_audit(args, label="baseline", item=item, annotated=item, protocol_signature=protocol, cache_hit=False)
                fh.write(json.dumps({
                    "baseline_correct": False,
                    "item": {**item, "_protocol_signature": protocol},
                    "answer": item.get("predicted"),
                    "gold": item.get("expected") or item.get("answer"),
                    "parse_error": item.get("predicted") is None,
                    "protocol_signature": protocol,
                    "protocol_signature_hash": protocol_signature_hash(protocol),
                }, ensure_ascii=False) + "\n")
    return correct, wrong


def _score_activated(activated: list[dict], rule: dict, anchor_text: str, args, task_spec, api_key, label: str):
    if not activated:
        return {}
    rendered = build_cheatsheet_with_rule(anchor_text, rule)
    protocol = _protocol_signature(args, anchor_text, task_spec)
    if args.require_consistent_scoring_protocol:
        assert_protocol_compatible(
            getattr(args, "_baseline_protocol_signature", None),
            protocol,
            allow_mixed_protocol_comparison=args.allow_mixed_protocol_comparison,
        )
    cache: CandidateOutputCache | None = getattr(args, "_candidate_output_cache", None)
    out = {}
    pending = []
    pending_keys = {}
    for item in activated:
        iid = _item_id(item)
        prompt_hash = make_candidate_prompt_hash(anchor_text, rule, item)
        cache_key = make_cache_key(
            task=args.task,
            item_id=iid,
            scorer_model=args.model_score,
            anchor_hash=stable_hash(anchor_text),
            candidate_id=rule["id"],
            candidate_hash=candidate_text_hash(rule),
            prompt_hash=prompt_hash,
            scoring_concurrency=min(args.concurrency, max(1, len(activated))),
            rf_scoring_flag=rf_scoring_enabled(),
            partition_routed_flag=partition_routed_scoring_enabled(),
            cot_first=args.cot_first,
            reasoning_effort=args.reasoning_effort,
            prompt_template_hash_value=protocol["prompt_template_hash"],
            parser_version=protocol["parser_version"],
        )
        cached = cache.get(cache_key) if cache else None
        if cached is not None:
            correct, annotated = cache_row_to_scored_item(cached, item)
            out[iid] = (correct, annotated)
            _record_parser_audit(args, label=label, item=item, annotated=annotated, protocol_signature=protocol, cache_hit=True)
            args._cache_audit.append({
                "item_id": iid,
                "candidate_id": rule["id"],
                "cache_hit": True,
                "cache_miss_reason": "",
                "protocol_signature_match": True,
            })
        else:
            pending.append(item)
            pending_keys[iid] = (cache_key, prompt_hash)
            args._cache_audit.append({
                "item_id": iid,
                "candidate_id": rule["id"],
                "cache_hit": False,
                "cache_miss_reason": "missing_key",
                "protocol_signature_match": True,
            })

    correct, wrong = score_batch(
        pending,
        rendered,
        args.model_score,
        api_key,
        concurrency=min(args.concurrency, max(1, len(pending))),
        reasoning_effort=args.reasoning_effort,
        cot_first=args.cot_first,
        progress_label=label,
        task_spec=task_spec,
    ) if pending else ([], [])
    for item in correct:
        iid = _item_id(item)
        out[iid] = (True, item)
        _record_parser_audit(args, label=label, item=item, annotated=item, protocol_signature=protocol, cache_hit=False)
        if cache:
            cache_key, prompt_hash = pending_keys[iid]
            cache.put(build_cache_row(
                cache_key=cache_key,
                task=args.task,
                item=item,
                candidate_id=rule["id"],
                scorer_model=args.model_score,
                prompt_hash=prompt_hash,
                correct=True,
                annotated_item=item,
                protocol_signature=protocol,
            ))
    for item in wrong:
        iid = _item_id(item)
        out[iid] = (False, item)
        _record_parser_audit(args, label=label, item=item, annotated=item, protocol_signature=protocol, cache_hit=False)
        if cache:
            cache_key, prompt_hash = pending_keys[iid]
            cache.put(build_cache_row(
                cache_key=cache_key,
                task=args.task,
                item=item,
                candidate_id=rule["id"],
                scorer_model=args.model_score,
                prompt_hash=prompt_hash,
                correct=False,
                annotated_item=item,
                protocol_signature=protocol,
            ))
    return out


def _evaluate_pool(pool_name, items, baseline_correct, rule, partition_key, anchor_text, args, task_spec, api_key, partition_label):
    route_rows = []
    activated = []
    for item in items:
        decision = route_candidate(
            rule,
            item,
            activation_routing_mode=args.activation_routing_mode,
            partition_key=partition_key,
            task_spec=task_spec,
        )
        route_rows.append((item, decision))
        if args.validation_routing_mode == "global" or decision.activated:
            activated.append(item)

    scored = _score_activated(
        activated,
        rule,
        anchor_text,
        args,
        task_spec,
        api_key,
        label=f"sfcr-replay:{partition_label}:{rule['id']}:{pool_name}",
    )

    effect_rows = []
    for item, decision in route_rows:
        iid = _item_id(item)
        exposed = args.validation_routing_mode == "global" or decision.activated
        consistency_reason = ""
        consistency_fallback = False
        if exposed:
            cand_correct, cand_item = scored.get(iid, (False, item))
            if args.rf_consistency_guard and args.task == "formal_fallacies":
                consistency_reason = _rf_inconsistency_reason(
                    cand_item.get("raw_response", ""),
                    cand_item.get("predicted"),
                )
                if consistency_reason:
                    consistency_fallback = True
                    cand_correct = baseline_correct
                    cand_item = {
                        **cand_item,
                        "predicted": item.get("predicted"),
                        "_sfcr_rf_consistency_fallback": True,
                        "_sfcr_rf_inconsistency_reason": consistency_reason,
                    }
        else:
            cand_correct, cand_item = baseline_correct, item
        row = make_item_effect_row(
            candidate_id=rule["id"],
            partition=partition_label,
            item=item,
            pool=pool_name,
            activated=exposed,
            route_decision=decision,
            baseline_correct=baseline_correct,
            candidate_correct=cand_correct,
            baseline_answer=item.get("predicted"),
            candidate_answer=cand_item.get("predicted"),
            gold_answer=item.get("expected") or item.get("answer"),
        )
        row["rf_consistency_fallback"] = consistency_fallback
        row["rf_inconsistency_reason"] = consistency_reason
        effect_rows.append(row)
    return effect_rows


def _evaluate_candidate(rule, pb, correct, anchor_text, args, task_spec, api_key):
    partition_label = pb.label
    pools = [
        ("local_failure", list(pb.failures), False),
        ("local_correct", list(pb.correct_pool), True),
        ("global_correct", list(correct), True),
    ]
    item_rows = []
    start_audit_n = len(args._parser_audit)
    for pool_name, pool_items, base_correct in pools:
        item_rows.extend(
            _evaluate_pool(
                pool_name,
                pool_items,
                base_correct,
                rule,
                tuple(pb.key),
                anchor_text,
                args,
                task_spec,
                api_key,
                partition_label,
            )
        )

    fixed_ids = [r["item_id"] for r in item_rows if r["pool"] == "local_failure" and r["effect"] == "fixed"]
    local_reg_ids = [r["item_id"] for r in item_rows if r["pool"] == "local_correct" and r["effect"] == "regressed"]
    global_reg_ids = [r["item_id"] for r in item_rows if r["pool"] == "global_correct" and r["effect"] == "regressed"]
    activated_not_fixed = [
        r["item_id"]
        for r in item_rows
        if r["pool"] == "local_failure" and r["activated"] and r["effect"] != "fixed"
    ]
    vetoed_shared = [
        r["item_id"]
        for r in item_rows
        if r["pool"] == "local_failure" and r["vetoed"]
    ]

    counts = {
        "fixed_count": len(fixed_ids),
        "n_failures": len(pb.failures),
        "local_regression_count": len(local_reg_ids),
        "n_local_correct": len(pb.correct_pool),
        "routed_global_regression_count": len(global_reg_ids),
        "n_global_correct": len(correct),
    }
    diagnostic_only = args.gate_profile == "formal_falsefalse_deploy" and partition_label != "False_False"
    decision = apply_count_gate(
        counts,
        profile=args.gate_profile,
        diagnostic_only=diagnostic_only,
    )
    candidate_protocol = _protocol_signature(args, anchor_text, task_spec)
    mismatch_reasons = protocol_mismatch_reasons(
        getattr(args, "_baseline_protocol_signature", None),
        candidate_protocol,
    )
    row = {
        **counts,
        "candidate_id": rule["id"],
        "memory_format": rule.get("memory_format", args.memory_format),
        "partition": partition_label,
        "rule": rule.get("rule", ""),
        "use_when": rule.get("use_when", ""),
        "do_not_use_when": rule.get("do_not_use_when", ""),
        "positive_tags": rule.get("positive_tags", []),
        "negative_tags": rule.get("negative_tags", []),
        "activation_routing_mode": args.activation_routing_mode,
        "validation_routing_mode": args.validation_routing_mode,
        "routed_net_count": decision.routed_net_count,
        "local_net_count": decision.local_net_count,
        "accepted": decision.accepted,
        "status_label": decision.status_label,
        "reject_reasons": ",".join(decision.reject_reasons),
        "fixed_item_ids": fixed_ids,
        "local_regressed_item_ids": local_reg_ids,
        "global_regressed_item_ids": global_reg_ids,
        "activated_not_fixed_item_ids": activated_not_fixed,
        "vetoed_shared_item_ids": vetoed_shared,
        "parse_errors": sum(1 for r in args._parser_audit[start_audit_n:] if r.get("parse_error")),
        "rf_consistency_guard": bool(args.rf_consistency_guard),
        "rf_consistency_fallback_count": sum(1 for r in item_rows if r.get("rf_consistency_fallback")),
        "protocol_match": not bool(mismatch_reasons),
        "protocol_mismatch_reasons": ",".join(mismatch_reasons),
        "protocol_signature_hash": protocol_signature_hash(candidate_protocol),
    }
    return row, item_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), list)})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", default="formal_fallacies", choices=sorted(_TASK_MAP))
    p.add_argument("--candidate-file", required=True)
    p.add_argument("--baseline-cache", default="runs/formal_ff_limit50_fixed/baseline_cache.jsonl")
    p.add_argument("--dataset", default="datasets/bbh/formal_fallacies_train.jsonl")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--anchor-cheatsheet", default="runs/global_gate_qwen_rf_gpt41mini_formal_limit50/cheatsheet_phase1_pk_final")
    p.add_argument("--model-score", default="qwen2.5-7b-instruct")
    p.add_argument("--router-type", default="feature", choices=["feature"])
    p.add_argument("--activation-routing-mode", default="hybrid", choices=["partition", "feature", "hybrid"])
    p.add_argument("--validation-routing-mode", default="routed", choices=["global", "routed"])
    p.add_argument("--gate-profile", default="formal_falsefalse_deploy")
    p.add_argument("--max-bins", type=int, default=1)
    p.add_argument("--bin-threshold", type=int, default=3)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--cot-first", action="store_true", default=True)
    p.add_argument("--no-cot-first", dest="cot_first", action="store_false")
    p.add_argument("--candidate-output-cache", default="")
    p.add_argument("--require-consistent-scoring-protocol", action="store_true")
    p.add_argument("--allow-mixed-protocol-comparison", action="store_true")
    p.add_argument("--protocol-signature-output", default="")
    p.add_argument(
        "--memory-format",
        default="rule_check",
        choices=[
            "rule_only",
            "rule_check",
            "rule_check_micro_example",
            "rule_check_decision_template",
        ],
    )
    p.add_argument(
        "--rf-consistency-guard",
        action="store_true",
        help="For formal_fallacies RF outputs, fall back to baseline when reasoning and VERDICT visibly contradict.",
    )
    p.add_argument("--strict-no-parse-errors", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args._parser_audit = []
    args._cache_audit = []
    args._candidate_output_cache = CandidateOutputCache(args.candidate_output_cache) if args.candidate_output_cache else None
    load_dotenv(REPO_ROOT / ".env")
    api_key = get_api_key()
    task_spec = _load_task(args.task)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(Path(args.dataset))
    if args.limit:
        items = items[: args.limit]
    _tag_items(items)
    anchor_path = Path(args.anchor_cheatsheet)
    if not anchor_path.exists() and anchor_path.with_suffix(".txt").exists():
        anchor_path = anchor_path.with_suffix(".txt")
    anchor_text = anchor_path.read_text(encoding="utf-8").strip()
    candidates = [
        _rule_for_memory_format(rule, args.memory_format)
        for rule in _load_candidates(Path(args.candidate_file), args.task)
    ]
    if not candidates:
        raise SystemExit(f"No candidates loaded from {args.candidate_file}")

    correct, wrong = _load_or_score_baseline(args, items, anchor_text, task_spec, api_key)
    bins = build_partitions(
        wrong,
        correct,
        bin_threshold=args.bin_threshold,
        partition_key_fn=task_spec.partition_key,
    )
    selected_bins = sorted(bins.values(), key=lambda b: len(b.failures), reverse=True)[: args.max_bins]
    region_summary = [
        {
            "partition": pb.label,
            "key": list(pb.key),
            "n_failures": len(pb.failures),
            "n_local_correct": len(pb.correct_pool),
            "conditions": task_spec.partition_key_to_conditions(pb.key),
        }
        for pb in selected_bins
    ]

    result_rows = []
    item_rows = []
    for pb in selected_bins:
        for rule in candidates:
            row, rows = _evaluate_candidate(rule, pb, correct, anchor_text, args, task_spec, api_key)
            result_rows.append(row)
            item_rows.extend(rows)

    (out_dir / "candidates.json").write_text(json.dumps({"rules": candidates}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "region_summary.json").write_text(json.dumps(region_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "validation_results.json").write_text(json.dumps(result_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "validation_results.csv", result_rows)
    write_item_effect_logs(item_rows, out_dir)
    _write_parser_audit(out_dir, args._parser_audit)
    _write_jsonl(out_dir / "candidate_output_cache_audit.jsonl", args._cache_audit)
    _write_csv(out_dir / "candidate_output_cache_audit.csv", args._cache_audit)
    protocol_path = Path(args.protocol_signature_output) if args.protocol_signature_output else out_dir / "protocol_signature.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(getattr(args, "_baseline_protocol_signature", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cache = getattr(args, "_candidate_output_cache", None)
    if cache:
        (out_dir / "candidate_output_cache_summary.json").write_text(
            json.dumps(cache.stats(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    total_parse_errors = sum(1 for r in args._parser_audit if r.get("parse_error"))
    ready_rows = [
        r for r in result_rows
        if r.get("partition") == "False_False"
        and r.get("fixed_count", 0) >= 3
        and r.get("local_regression_count", 0) == 0
        and r.get("routed_global_regression_count", 0) == 0
        and r.get("routed_net_count", 0) >= 2
        and r.get("parse_errors", 0) == 0
        and r.get("protocol_match")
    ]

    summary = {
        "output_dir": str(out_dir),
        "baseline_accuracy": len(correct) / len(items) if items else 0.0,
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "n_candidates": len(candidates),
        "n_rows": len(result_rows),
        "n_accepted": sum(1 for r in result_rows if r["accepted"]),
        "total_parse_errors": total_parse_errors,
        "memory_format": args.memory_format,
        "protocol_signature_hash": protocol_signature_hash(getattr(args, "_baseline_protocol_signature", {})),
        "full_limit100_ready": bool(ready_rows),
        "limit100_ready_candidate_ids": [r["candidate_id"] for r in ready_rows],
    }
    (out_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.strict_no_parse_errors and total_parse_errors:
        raise SystemExit(f"Replay produced {total_parse_errors} parse errors; see {out_dir / 'parse_errors.jsonl'}")
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
