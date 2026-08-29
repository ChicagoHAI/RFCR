"""Small generated-rule SF-CR pilot for BBH object_counting.

The script reuses existing raw-baseline item logs, asks a generator model for a
single counting rule from baseline failures, evaluates the rule on the full
100-item test split, and optionally performs one boundary-repair round if the
first candidate regresses baseline-correct items.
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
from utils.llm_client import call_llm, call_llm_batch, get_api_key


NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--baseline-items", default="runs/controlled_full_benchmark_extra_bbh_gpt41mini_full_20260517/pilot_item_results.jsonl")
    p.add_argument("--dataset", default="datasets/bbh/object_counting_test.jsonl")
    p.add_argument("--output-dir", default="runs/object_counting_generated_rule_gpt41mini_20260518")
    p.add_argument("--model-gen", default="openai/gpt-4.1-mini")
    p.add_argument("--model-score", default="openai/gpt-4.1-mini")
    p.add_argument("--n-failures", type=int, default=12)
    p.add_argument("--n-boundaries", type=int, default=12)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--repair", action="store_true", default=True)
    return p.parse_args()


def _read_baseline(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") == "object_counting":
            rows[str(row["item_id"])] = row
    return rows


def _item_id(item: dict, idx: int) -> str:
    return str(item.get("id") or f"object_counting_{idx:04d}")


def _example_line(row: dict) -> str:
    return (
        f"- id={row['item_id']} gold={row['gold']} baseline={row['predicted']} "
        f"input={row['input_snippet']}"
    )


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _parse_int(text: str) -> str | None:
    m = re.search(r"VERDICT\s*:?\s*(-?\d+)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"-?\d+", text.strip())
    return m.group(0) if m else None


def _normalise_candidate(rule: dict, cid: str) -> dict:
    out = {
        "id": rule.get("id") or cid,
        "task": "object_counting",
        "rule": str(rule.get("rule", "")).strip(),
        "use_when": str(rule.get("use_when", "object_counting questions")).strip(),
        "do_not_use_when": str(rule.get("do_not_use_when", "")).strip(),
        "check": str(rule.get("check", "")).strip(),
        "micro_example": str(rule.get("micro_example", "")).strip(),
        "source_of_candidate": rule.get("source_of_candidate", "generated_from_failure_logs"),
        "generator_model": rule.get("generator_model", "openai/gpt-4.1-mini"),
    }
    return out


def _generation_prompt(failures: list[dict], boundaries: list[dict]) -> str:
    failure_block = "\n".join(_example_line(r) for r in failures)
    boundary_block = "\n".join(_example_line(r) for r in boundaries)
    return f"""\
You are generating one conservative SF-CR memory atom for BBH object_counting.

The model often undercounts when an item has an explicit quantity word, and it may count distractor objects outside the requested category.

Failure examples, where the baseline answer was wrong:
{failure_block}

Boundary examples, where the baseline was correct and must not regress:
{boundary_block}

Return JSON only:
{{
  "rules": [
    {{
      "id": "gen_object_counting_quantity_filter_v1",
      "task": "object_counting",
      "rule": "...",
      "use_when": "...",
      "do_not_use_when": "...",
      "check": "...",
      "micro_example": "..."
    }}
  ]
}}

Requirements:
- The rule must tell the scorer to count only objects in the category asked by the question.
- Convert quantity words like two/three/four/five into counts, while a/an/one count as 1.
- Ignore distractor objects outside the asked category.
- Keep the rule procedural and short.
"""


def _repair_prompt(rule: dict, fixed: list[dict], regressed: list[dict], unchanged_wrong: list[dict]) -> str:
    fixed_block = "\n".join(_example_line(r) for r in fixed[:10]) or "(none)"
    reg_block = "\n".join(_example_line(r) for r in regressed[:10]) or "(none)"
    miss_block = "\n".join(_example_line(r) for r in unchanged_wrong[:10]) or "(none)"
    return f"""\
Repair this object_counting SF-CR atom. Keep the same rule family, but reduce regressions.

Current rule:
{json.dumps(rule, indent=2, ensure_ascii=False)}

Fixed examples:
{fixed_block}

Regressed examples:
{reg_block}

Still wrong examples:
{miss_block}

Return JSON only with one rule:
{{
  "rules": [
    {{
      "id": "gen_object_counting_quantity_filter_repair_v2",
      "task": "object_counting",
      "rule": "...",
      "use_when": "...",
      "do_not_use_when": "...",
      "check": "...",
      "micro_example": "..."
    }}
  ]
}}

Requirements:
- The CHECK must force a concrete sum.
- Preserve correct baseline cases.
- Do not introduce broad advice unrelated to object counting.
"""


def _candidate_prompt(rule: dict, item: dict) -> str:
    return f"""\
You are solving a BBH object_counting item.

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

Do the count carefully. Reply with ONLY:
VERDICT: <integer>
"""


def _evaluate(rule: dict, items: list[dict], baseline: dict[str, dict], args: argparse.Namespace, api_key: str, out_dir: Path, label: str) -> dict:
    prompts = [_candidate_prompt(rule, item) for item in items]
    responses = call_llm_batch(
        prompts,
        model=args.model_score,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        progress_label=f"object-counting:{label}",
        reasoning_effort="low",
    )
    rows = []
    fixed_ids, regressed_ids, parse_ids = [], [], []
    for idx, (item, resp) in enumerate(zip(items, responses)):
        iid = _item_id(item, idx)
        b = baseline[iid]
        raw = "" if resp is None else resp.content
        pred = None if resp is None else _parse_int(raw)
        parse_error = pred is None
        cand_correct = pred == str(item["answer"]).strip()
        base_correct = bool(b["correct"])
        if parse_error:
            parse_ids.append(iid)
        if (not base_correct) and cand_correct:
            effect = "fixed"
            fixed_ids.append(iid)
        elif base_correct and not cand_correct:
            effect = "regressed"
            regressed_ids.append(iid)
        elif base_correct and cand_correct:
            effect = "unchanged_correct"
        else:
            effect = "unchanged_wrong"
        rows.append({
            "item_id": iid,
            "baseline_answer": b.get("predicted"),
            "candidate_answer": pred,
            "gold": str(item["answer"]).strip(),
            "baseline_correct": base_correct,
            "candidate_correct": cand_correct,
            "parse_error": parse_error,
            "effect": effect,
            "input_snippet": b.get("input_snippet", item["input"])[:300],
            "raw_response": raw[:500],
        })
    n = len(items)
    base_correct_n = sum(1 for r in baseline.values() if r["correct"])
    cand_correct_n = base_correct_n + len(fixed_ids) - len(regressed_ids)
    summary = {
        "label": label,
        "task": "object_counting",
        "model_score": args.model_score,
        "n_items": n,
        "baseline_correct": base_correct_n,
        "baseline_accuracy": base_correct_n / n,
        "candidate_correct": cand_correct_n,
        "candidate_accuracy": cand_correct_n / n,
        "fixed_count": len(fixed_ids),
        "regression_count": len(regressed_ids),
        "net_count": len(fixed_ids) - len(regressed_ids),
        "parse_error_count": len(parse_ids),
        "fixed_item_ids": fixed_ids,
        "regressed_item_ids": regressed_ids,
        "parse_error_item_ids": parse_ids,
        "candidate_id": rule["id"],
    }
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / f"{label}_item_effects.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / f"{label}_item_effects.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {"summary": summary, "rows": rows}


def main() -> None:
    args = _parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = get_api_key()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _read_baseline(Path(args.baseline_items))
    items = load_jsonl(Path(args.dataset))
    for idx, item in enumerate(items):
        iid = _item_id(item, idx)
        if iid not in baseline:
            raise SystemExit(f"Missing baseline row for {iid}")

    baseline_rows = [baseline[_item_id(item, idx)] for idx, item in enumerate(items)]
    failures = [r for r in baseline_rows if not r["correct"]][: args.n_failures]
    boundaries = [r for r in baseline_rows if r["correct"]][: args.n_boundaries]

    prompt = _generation_prompt(failures, boundaries)
    (out_dir / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
    resp = call_llm(prompt, model=args.model_gen, api_key=api_key, temperature=0.2, max_tokens=1200)
    raw = "" if resp is None else resp.content
    (out_dir / "generation_raw.txt").write_text(raw, encoding="utf-8")
    rule = _normalise_candidate((_extract_json(raw).get("rules") or [{}])[0], "gen_object_counting_quantity_filter_v1")
    rule["generator_model"] = args.model_gen
    (out_dir / "generated_rule_v1.json").write_text(json.dumps({"rules": [rule]}, indent=2, ensure_ascii=False), encoding="utf-8")

    first = _evaluate(rule, items, baseline, args, api_key, out_dir, "v1")
    final_rule = rule
    final = first

    if args.repair and first["summary"]["regression_count"] > 0:
        by_id = {row["item_id"]: row for row in baseline_rows}
        fixed = [by_id[i] for i in first["summary"]["fixed_item_ids"] if i in by_id]
        regressed = [by_id[i] for i in first["summary"]["regressed_item_ids"] if i in by_id]
        unchanged_wrong_ids = [row["item_id"] for row in first["rows"] if row["effect"] == "unchanged_wrong"]
        unchanged_wrong = [by_id[i] for i in unchanged_wrong_ids if i in by_id]
        repair_prompt = _repair_prompt(rule, fixed, regressed, unchanged_wrong)
        (out_dir / "repair_prompt_v2.txt").write_text(repair_prompt, encoding="utf-8")
        resp2 = call_llm(repair_prompt, model=args.model_gen, api_key=api_key, temperature=0.1, max_tokens=1200)
        raw2 = "" if resp2 is None else resp2.content
        (out_dir / "repair_raw_v2.txt").write_text(raw2, encoding="utf-8")
        final_rule = _normalise_candidate((_extract_json(raw2).get("rules") or [{}])[0], "gen_object_counting_quantity_filter_repair_v2")
        final_rule["generator_model"] = args.model_gen
        final_rule["parent_candidate_id"] = rule["id"]
        (out_dir / "generated_rule_v2.json").write_text(json.dumps({"rules": [final_rule]}, indent=2, ensure_ascii=False), encoding="utf-8")
        final = _evaluate(final_rule, items, baseline, args, api_key, out_dir, "v2")

    report = {
        "output_dir": str(out_dir),
        "generated_rule_v1": rule,
        "final_rule": final_rule,
        "v1_summary": first["summary"],
        "final_summary": final["summary"],
    }
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["final_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
