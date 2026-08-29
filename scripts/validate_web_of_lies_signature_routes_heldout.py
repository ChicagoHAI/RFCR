"""Held-out validation for fixed web_of_lies signature routes.

This script is deliberately narrower than the exploratory signature-selection
utility. It takes route signatures selected elsewhere, freezes them, and
evaluates them on a held-out dataset slice. No route is selected from the
held-out effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data import load_jsonl
from utils.llm_client import call_llm_batch, get_api_key


TASK = "web_of_lies"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="datasets/bbh/web_of_lies_test.jsonl")
    p.add_argument("--candidate-file", default="runs/web_of_lies_generated_rule_format_repair_mt128_gpt41mini_20260518/generated_rule.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-score", default="openai/gpt-4.1-mini")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--route-signature", action="append", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--reasoning-effort", default="low")
    return p.parse_args()


def _item_id(global_idx: int) -> str:
    return f"{TASK}_{global_idx:04d}"


def _parse_yesno(text: str) -> str | None:
    match = re.search(r"VERDICT\s*:?\s*(YES|NO|Yes|No|yes|no)\b", text or "")
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"
    stripped = (text or "").strip()
    match = re.search(r"\b(YES|NO|Yes|No|yes|no)\b", stripped)
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"
    return None


def _load_rule(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rule = (data.get("rules") or data.get("candidates") or [{}])[0]
    return {
        "id": rule.get("id", "gen_web_of_lies_truth_chain_v1"),
        "rule": rule.get("rule", ""),
        "use_when": rule.get("use_when", ""),
        "do_not_use_when": rule.get("do_not_use_when", ""),
        "check": rule.get("check", ""),
        "micro_example": rule.get("micro_example", ""),
    }


def _baseline_prompt(item: dict) -> str:
    return (
        f"{item['input']}\n\n"
        "Compute the truth chain internally. Reply with ONLY one line:\n"
        "VERDICT: Yes or No"
    )


def _candidate_prompt(rule: dict, item: dict) -> str:
    return f"""\
You are solving a BBH web_of_lies item.

SF-CR RULE:
{rule['rule']}

USE WHEN:
{rule.get('use_when', '')}

DO NOT USE WHEN:
{rule.get('do_not_use_when', '')}

CHECK:
{rule.get('check', '')}

Question:
{item['input']}

Compute the chain internally. Do not print steps, names, truth table, or explanation.
Your entire response must be exactly one of these two lines:
VERDICT: Yes
VERDICT: No
"""


def _signature(item: dict) -> str:
    text = str(item["input"])
    sentences = [s.strip() for s in re.split(r"\.\s*", text) if s.strip()]
    parts: list[str] = []
    for idx, sent in enumerate(sentences):
        if sent.lower().startswith("question:"):
            sent = sent.split(":", 1)[1].strip()
        if sent.lower().startswith("does "):
            continue
        low = sent.lower()
        if idx == 0:
            if " tells the truth" in low:
                parts.append("init_true")
            elif " lies" in low:
                parts.append("init_lie")
            continue
        if " says " in low and " tells the truth" in low:
            parts.append("says_truth")
        elif " says " in low and " lies" in low:
            parts.append("says_lie")
    return "signature_" + "_".join(parts)


def _score(prompts: list[str], args: argparse.Namespace, api_key: str, label: str):
    return call_llm_batch(
        prompts,
        model=args.model_score,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        progress_label=label,
        reasoning_effort=args.reasoning_effort,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row if not isinstance(row.get(k), (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = get_api_key()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = load_jsonl(Path(args.dataset))
    selected = list(enumerate(all_items))[args.start : args.start + args.limit]
    route_signatures = set(args.route_signature or [])
    rule = _load_rule(Path(args.candidate_file))

    baseline_responses = _score([_baseline_prompt(item) for _, item in selected], args, api_key, "web-heldout:baseline")
    baseline_rows: list[dict] = []
    active_pairs: list[tuple[int, dict]] = []
    for (global_idx, item), resp in zip(selected, baseline_responses):
        raw = "" if resp is None else resp.content
        pred = _parse_yesno(raw)
        gold = str(item["answer"]).strip()
        sig = _signature(item)
        active = sig in route_signatures
        if active:
            active_pairs.append((global_idx, item))
        baseline_rows.append(
            {
                "item_id": _item_id(global_idx),
                "global_index": global_idx,
                "signature": sig,
                "activated": active,
                "baseline_answer": pred,
                "gold_answer": gold,
                "baseline_correct": pred == gold,
                "baseline_parse_error": pred is None,
                "baseline_raw_response": raw[:500],
                "input": item["input"],
            }
        )

    candidate_by_id: dict[str, dict] = {}
    if active_pairs:
        candidate_responses = _score(
            [_candidate_prompt(rule, item) for _, item in active_pairs],
            args,
            api_key,
            "web-heldout:candidate",
        )
        for (global_idx, item), resp in zip(active_pairs, candidate_responses):
            raw = "" if resp is None else resp.content
            pred = _parse_yesno(raw)
            candidate_by_id[_item_id(global_idx)] = {
                "candidate_answer": pred,
                "candidate_parse_error": pred is None,
                "candidate_raw_response": raw[:500],
            }

    item_rows: list[dict] = []
    fixed_ids, regressed_ids, parse_error_ids = [], [], []
    for row in baseline_rows:
        iid = row["item_id"]
        activated = bool(row["activated"])
        if activated:
            cand = candidate_by_id.get(iid, {"candidate_answer": None, "candidate_parse_error": True, "candidate_raw_response": ""})
            candidate_answer = cand["candidate_answer"]
            candidate_parse_error = bool(cand["candidate_parse_error"])
            candidate_correct = candidate_answer == row["gold_answer"]
        else:
            candidate_answer = row["baseline_answer"]
            candidate_parse_error = False
            candidate_correct = bool(row["baseline_correct"])
            cand = {"candidate_raw_response": ""}

        if activated and candidate_parse_error:
            parse_error_ids.append(iid)
        if not row["baseline_correct"] and candidate_correct:
            effect = "fixed"
            fixed_ids.append(iid)
        elif row["baseline_correct"] and not candidate_correct:
            effect = "regressed"
            regressed_ids.append(iid)
        elif row["baseline_correct"] and candidate_correct:
            effect = "unchanged_correct"
        else:
            effect = "unchanged_wrong"

        item_rows.append(
            {
                **row,
                "candidate_answer": candidate_answer,
                "candidate_correct": candidate_correct,
                "candidate_parse_error": candidate_parse_error,
                "candidate_raw_response": cand.get("candidate_raw_response", ""),
                "effect": effect,
            }
        )

    n = len(item_rows)
    baseline_correct = sum(1 for r in item_rows if r["baseline_correct"])
    union_correct = baseline_correct + len(fixed_ids) - len(regressed_ids)
    summary = {
        "task": TASK,
        "dataset": args.dataset,
        "slice_start": args.start,
        "limit": args.limit,
        "n_items": n,
        "model_score": args.model_score,
        "candidate_id": rule["id"],
        "candidate_file": args.candidate_file,
        "activation_routing_mode": "frozen_signature_routes_heldout",
        "route_signatures": sorted(route_signatures),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / n if n else 0.0,
        "union_correct": union_correct,
        "union_accuracy": union_correct / n if n else 0.0,
        "fixed_count": len(fixed_ids),
        "regression_count": len(regressed_ids),
        "net_count": len(fixed_ids) - len(regressed_ids),
        "activated_count": sum(1 for r in item_rows if r["activated"]),
        "parse_error_count": len(parse_error_ids),
        "baseline_parse_error_count": sum(1 for r in item_rows if r["baseline_parse_error"]),
        "fixed_item_ids": fixed_ids,
        "regressed_item_ids": regressed_ids,
        "parse_error_item_ids": parse_error_ids,
        "protocol_note": "Route signatures were frozen before this held-out validation slice; no route was selected from held-out effects.",
    }

    (out_dir / "atom_union_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "candidate_rule.json").write_text(json.dumps({"rules": [rule]}, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(out_dir / "atom_union_item_effects.jsonl", item_rows)
    _write_csv(out_dir / "atom_union_item_effects.csv", item_rows)
    _write_csv(out_dir / "atom_union_summary.csv", [summary])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
