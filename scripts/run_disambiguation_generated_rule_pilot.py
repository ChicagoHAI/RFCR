"""Generated-rule SF-CR pilot for BBH disambiguation_qa.

This is a small controlled expansion experiment. It generates several narrow
pronoun-reference memory atoms from cached baseline failures, evaluates them on
the 100-item split, and then performs an offline feature-routing analysis to
find zero-regression sub-atoms worth replaying.
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


TASK = "disambiguation_qa"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--baseline-items", default="runs/controlled_full_benchmark_pilot_gpt41mini_limit100_20260517/pilot_item_results.jsonl")
    p.add_argument("--dataset", default="datasets/bbh/disambiguation_qa_test.jsonl")
    p.add_argument("--output-dir", default="runs/disambiguation_generated_rule_gpt41mini_20260518")
    p.add_argument("--model-gen", default="openai/gpt-4.1-mini")
    p.add_argument("--model-score", default="openai/gpt-4.1-mini")
    p.add_argument("--n-failures", type=int, default=18)
    p.add_argument("--n-boundaries", type=int, default=18)
    p.add_argument("--n-rules", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--min-route-fixes", type=int, default=1)
    return p.parse_args()


def _read_baseline(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") == TASK:
            rows[str(row["item_id"])] = row
    return rows


def _item_id(item: dict, idx: int) -> str:
    return str(item.get("id") or f"{TASK}_{idx:04d}")


def _sentence(text: str) -> str:
    m = re.search(r"Sentence:\s*(.*?)\s*Options:", text, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else re.sub(r"\s+", " ", text).strip()


def _options(text: str) -> str:
    m = re.search(r"Options:\s*(.*)$", text, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _short_example(row: dict, item: dict) -> str:
    return (
        f"- id={row['item_id']} gold={row['gold']} baseline={row['predicted']} "
        f"sentence={_sentence(item['input'])} options={_options(item['input'])}"
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


def _parse_option(text: str) -> str | None:
    m = re.search(r"VERDICT\s*:?\s*\(?([ABC])\)?", text, re.I)
    if m:
        return f"({m.group(1).upper()})"
    m = re.search(r"\(([ABC])\)", text.strip(), re.I)
    return f"({m.group(1).upper()})" if m else None


def _normalise_candidate(rule: dict, idx: int, model_gen: str) -> dict:
    return {
        "id": str(rule.get("id") or f"gen_disambig_rule_{idx}_v1"),
        "task": TASK,
        "rule": str(rule.get("rule", "")).strip(),
        "use_when": str(rule.get("use_when", "")).strip(),
        "do_not_use_when": str(rule.get("do_not_use_when", "")).strip(),
        "check": str(rule.get("check", "")).strip(),
        "activation_cues": list(rule.get("activation_cues") or []),
        "negative_cues": list(rule.get("negative_cues") or []),
        "micro_example": str(rule.get("micro_example", "")).strip(),
        "source_of_candidate": rule.get("source_of_candidate", "generated_from_failure_logs"),
        "generator_model": model_gen,
    }


def _generation_prompt(failures: list[tuple[dict, dict]], boundaries: list[tuple[dict, dict]], n_rules: int) -> str:
    failure_block = "\n".join(_short_example(row, item) for row, item in failures)
    boundary_block = "\n".join(_short_example(row, item) for row, item in boundaries)
    return f"""\
You are generating narrow SF-CR memory atoms for BBH disambiguation_qa.

Task: decide the antecedent of a pronoun, or answer Ambiguous.

Failure examples where the baseline was wrong:
{failure_block}

Boundary examples where the baseline was correct and must not regress:
{boundary_block}

Return JSON only:
{{
  "rules": [
    {{
      "id": "gen_disambig_specific_family_v1",
      "task": "disambiguation_qa",
      "rule": "...",
      "use_when": "...",
      "do_not_use_when": "...",
      "activation_cues": ["short literal cue"],
      "negative_cues": ["short literal veto"],
      "check": "...",
      "micro_example": "..."
    }}
  ]
}}

Requirements:
- Return exactly {n_rules} rules.
- Each rule must be narrow and tied to a recurring linguistic pattern.
- Avoid generic advice like "consider both interpretations"; that often preserves ambiguity errors.
- Prefer rules that distinguish when only one option is semantically licensed.
- Include literal activation cues that can be checked in the sentence text.
- Keep each rule short enough to prepend to a prompt.
"""


def _candidate_prompt(rule: dict, item: dict) -> str:
    return f"""\
You are solving a BBH disambiguation_qa item.

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

Reply with ONLY:
VERDICT: (A), (B), or (C)
"""


def _feature_tags(item: dict) -> list[str]:
    sent = _sentence(item["input"]).lower()
    tags = set()
    pronouns = {
        "he": r"\bhe\b",
        "she": r"\bshe\b",
        "they": r"\bthey\b",
        "his": r"\bhis\b",
        "her": r"\bher\b",
        "their": r"\btheir\b",
        "it": r"\bit\b",
    }
    for name, pat in pronouns.items():
        if re.search(pat, sent):
            tags.add(f"pronoun_{name}")
    if " because " in sent:
        tags.add("because_clause")
    if " but " in sent:
        tags.add("but_clause")
    if " after " in sent:
        tags.add("after_clause")
    if " otherwise " in sent:
        tags.add("otherwise_clause")
    if " told " in sent and " that " in sent:
        tags.add("told_that")
    if " told " in sent and re.search(r"\bshould\b|\bshouldn't\b|should not", sent):
        tags.add("advice_should")
    if " warned " in sent:
        tags.add("warning_event")
    if " helped " in sent and " because " in sent:
        tags.add("helped_because")
    if " asked" in sent:
        tags.add("asked_event")
    if " bought " in sent and "needed one" in sent:
        tags.add("needed_one")
    if " sent a message " in sent or "reply" in sent:
        tags.add("message_reply")
    if " discuss " in sent or "discussed" in sent:
        tags.add("discussion")
    if re.search(r"\bwas unable to\b|\bunable to\b", sent):
        tags.add("unable_to")
    if re.search(r"\bargued with\b|\byelled at\b|\bcomplained to\b", sent):
        tags.add("conflict_event")
    if re.search(r"\bcorrected\b|\bunderstood\b", sent):
        tags.add("knowledge_asymmetry")
    if re.search(r"\bcar\b|\bmicroscope\b|\bbook\b|\bdesign\b|\bproblem\b|\bfloor\b|\bwindow\b", sent):
        tags.add("concrete_object_or_issue")
    return sorted(tags)


def _evaluate(rule: dict, items: list[dict], baseline: dict[str, dict], args: argparse.Namespace, api_key: str, out_dir: Path, label: str) -> dict:
    prompts = [_candidate_prompt(rule, item) for item in items]
    responses = call_llm_batch(
        prompts,
        model=args.model_score,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        progress_label=f"disambiguation:{label}",
        reasoning_effort="low",
    )
    rows = []
    fixed_ids, regressed_ids, parse_ids = [], [], []
    for idx, (item, resp) in enumerate(zip(items, responses)):
        iid = _item_id(item, idx)
        b = baseline[iid]
        raw = "" if resp is None else resp.content
        pred = None if resp is None else _parse_option(raw)
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
        rows.append(
            {
                "candidate_id": rule["id"],
                "item_id": iid,
                "baseline_answer": b.get("predicted"),
                "candidate_answer": pred,
                "gold": str(item["answer"]).strip(),
                "baseline_correct": base_correct,
                "candidate_correct": cand_correct,
                "parse_error": parse_error,
                "effect": effect,
                "features": _feature_tags(item),
                "sentence": _sentence(item["input"]),
                "options": _options(item["input"]),
                "raw_response": raw[:500],
            }
        )
    n = len(items)
    base_correct_n = sum(1 for r in baseline.values() if r["correct"])
    cand_correct_n = base_correct_n + len(fixed_ids) - len(regressed_ids)
    summary = {
        "label": label,
        "task": TASK,
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


def _feature_route_candidates(evaluations: list[dict], min_fixes: int) -> list[dict]:
    candidates = []
    for ev in evaluations:
        rule = ev["rule"]
        rows = ev["result"]["rows"]
        all_tags = sorted({tag for row in rows for tag in row["features"]})
        for tag in all_tags:
            active = [row for row in rows if tag in row["features"]]
            fixed = [row for row in active if row["effect"] == "fixed"]
            reg = [row for row in active if row["effect"] == "regressed"]
            parse = [row for row in active if row["parse_error"]]
            if len(fixed) >= min_fixes and not reg and not parse:
                candidates.append(
                    {
                        "candidate_id": rule["id"],
                        "route_feature": tag,
                        "activated_count": len(active),
                        "fixed_count": len(fixed),
                        "regression_count": len(reg),
                        "net_count": len(fixed) - len(reg),
                        "fixed_item_ids": [row["item_id"] for row in fixed],
                        "regressed_item_ids": [row["item_id"] for row in reg],
                    }
                )
    return sorted(candidates, key=lambda x: (-x["fixed_count"], x["activated_count"], x["candidate_id"], x["route_feature"]))


def _greedy_zero_regression_union(evaluations: list[dict], route_candidates: list[dict], items: list[dict], baseline: dict[str, dict]) -> dict:
    rows_by_rule = {
        ev["rule"]["id"]: {row["item_id"]: row for row in ev["result"]["rows"]}
        for ev in evaluations
    }
    selected = []
    active_by_item: dict[str, dict] = {}
    fixed_ids: set[str] = set()

    for cand in route_candidates:
        rule_rows = rows_by_rule[cand["candidate_id"]]
        trial = dict(active_by_item)
        for iid, row in rule_rows.items():
            if cand["route_feature"] in row["features"] and iid not in trial:
                trial[iid] = row
        regressions = [iid for iid, row in trial.items() if row["effect"] == "regressed"]
        if regressions:
            continue
        trial_fixed = {iid for iid, row in trial.items() if row["effect"] == "fixed"}
        if not (trial_fixed - fixed_ids):
            continue
        active_by_item = trial
        fixed_ids = trial_fixed
        selected.append(cand)

    n = len(items)
    base_correct_n = sum(1 for row in baseline.values() if row["correct"])
    regressed_ids = [iid for iid, row in active_by_item.items() if row["effect"] == "regressed"]
    parse_ids = [iid for iid, row in active_by_item.items() if row["parse_error"]]
    summary = {
        "task": TASK,
        "n_items": n,
        "baseline_correct": base_correct_n,
        "baseline_accuracy": base_correct_n / n,
        "union_correct": base_correct_n + len(fixed_ids) - len(regressed_ids),
        "union_accuracy": (base_correct_n + len(fixed_ids) - len(regressed_ids)) / n,
        "fixed_count": len(fixed_ids),
        "regression_count": len(regressed_ids),
        "net_count": len(fixed_ids) - len(regressed_ids),
        "activated_count": len(active_by_item),
        "parse_error_count": len(parse_ids),
        "fixed_item_ids": sorted(fixed_ids),
        "regressed_item_ids": sorted(regressed_ids),
        "parse_error_item_ids": sorted(parse_ids),
        "selected_routes": selected,
    }
    item_rows = []
    for idx, item in enumerate(items):
        iid = _item_id(item, idx)
        routed = active_by_item.get(iid)
        if routed is None:
            b = baseline[iid]
            item_rows.append(
                {
                    "item_id": iid,
                    "activated": False,
                    "candidate_id": None,
                    "route_feature": None,
                    "baseline_correct": bool(b["correct"]),
                    "candidate_correct": bool(b["correct"]),
                    "effect": "baseline_only",
                    "features": _feature_tags(item),
                    "sentence": _sentence(item["input"]),
                }
            )
        else:
            item_rows.append(
                {
                    "item_id": iid,
                    "activated": True,
                    "candidate_id": routed["candidate_id"],
                    "route_feature": ",".join(
                        route["route_feature"]
                        for route in selected
                        if route["candidate_id"] == routed["candidate_id"] and route["route_feature"] in routed["features"]
                    ),
                    "baseline_correct": routed["baseline_correct"],
                    "candidate_correct": routed["candidate_correct"],
                    "effect": routed["effect"],
                    "features": routed["features"],
                    "sentence": routed["sentence"],
                }
            )
    return {"summary": summary, "rows": item_rows}


def main() -> None:
    args = _parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = get_api_key()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _read_baseline(Path(args.baseline_items))
    items = load_jsonl(Path(args.dataset))
    item_by_id = {_item_id(item, idx): item for idx, item in enumerate(items)}
    for idx, item in enumerate(items):
        iid = _item_id(item, idx)
        if iid not in baseline:
            raise SystemExit(f"Missing baseline row for {iid}")

    baseline_pairs = [(baseline[_item_id(item, idx)], item) for idx, item in enumerate(items)]
    failures = [(row, item) for row, item in baseline_pairs if not row["correct"]][: args.n_failures]
    boundaries = [(row, item) for row, item in baseline_pairs if row["correct"]][: args.n_boundaries]

    prompt = _generation_prompt(failures, boundaries, args.n_rules)
    (out_dir / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
    resp = call_llm(prompt, model=args.model_gen, api_key=api_key, temperature=0.25, max_tokens=2200)
    raw = "" if resp is None else resp.content
    (out_dir / "generation_raw.txt").write_text(raw, encoding="utf-8")

    payload = _extract_json(raw)
    rules = [_normalise_candidate(rule, idx, args.model_gen) for idx, rule in enumerate(payload.get("rules") or [], start=1)]
    rules = rules[: args.n_rules]
    if not rules:
        raise SystemExit("Generator returned no rules")
    (out_dir / "generated_rules.json").write_text(json.dumps({"rules": rules}, indent=2, ensure_ascii=False), encoding="utf-8")

    evaluations = []
    for idx, rule in enumerate(rules, start=1):
        result = _evaluate(rule, items, baseline, args, api_key, out_dir, f"rule{idx}")
        evaluations.append({"rule": rule, "result": result})

    summaries = [ev["result"]["summary"] | {"rule": ev["rule"]} for ev in evaluations]
    (out_dir / "candidate_summaries.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    route_candidates = _feature_route_candidates(evaluations, args.min_route_fixes)
    (out_dir / "feature_route_candidates.json").write_text(json.dumps(route_candidates, indent=2, ensure_ascii=False), encoding="utf-8")

    union = _greedy_zero_regression_union(evaluations, route_candidates, items, baseline)
    union["summary"]["output_dir"] = str(out_dir)
    union["summary"]["model_score"] = args.model_score
    union["summary"]["model_gen"] = args.model_gen
    union["summary"]["candidate_ids"] = [rule["id"] for rule in rules]
    union["summary"]["activation_routing_mode"] = "posthoc_feature_zero_regression"
    union["summary"]["memory_format"] = "rule_check"
    (out_dir / "safe_routed_union_summary.json").write_text(json.dumps(union["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "safe_routed_union_item_effects.jsonl").open("w", encoding="utf-8") as fh:
        for row in union["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "output_dir": str(out_dir),
        "generated_rules": rules,
        "candidate_summaries": summaries,
        "feature_route_candidates": route_candidates,
        "safe_routed_union_summary": union["summary"],
    }
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(union["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
