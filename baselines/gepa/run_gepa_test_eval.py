"""Frozen-prompt test evaluation for the GEPA baseline (ARR rebuttal).

Usage: <repo>/.venv/bin/python run_gepa_test_eval.py <task> [--max-tokens 64]

Mirrors rebuttal_experiments/promptwizard/run_pw_test_eval.py:
- Uses the frozen {task}/optimized_prompt.txt (sha256 recorded; must match the
  hash frozen at optimization time in gepa_opt_summary.json).
- Runs once on the task's 100 test items via the repo's LLM machinery
  (utils.llm_client.call_llm_batch: temperature 0.0, seed 42, model
  openai/gpt-4.1-mini) and scores with the repo's task specs/parsers
  (_task_cfg -> _task_spec_from_cfg -> _parse_prediction / spec.is_correct).
  No scoring reimplemented.
- Prompt composition: GEPA's DefaultAdapter evaluated candidates as
  [system=optimized prompt, user=item input]. The repo client sends a single
  user message, so the optimized prompt is prepended to the item input
  ("{prompt}\n\n{input}") — the same system-prompt handling documented for the
  PromptWizard baseline.
- Compares per-item against the locked CS-ICL anchor cache
  (experiment_reports_and_results/E3_route_provenance_manifest/
   main_eligible_atom_tasks_csicl_cache_canonical.jsonl):
  fix = anchor-wrong -> GEPA-correct; regression = anchor-correct -> GEPA-wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ICR = REPO / "ICRefine"
sys.path.insert(0, str(ICR))

from scripts.run_controlled_full_benchmark_pilot import _parse_prediction, _task_spec_from_cfg  # noqa: E402
from scripts.run_sfcr_emnlp_baseline_surface import _task_cfg  # noqa: E402
from utils.data import load_jsonl  # noqa: E402
from utils.llm_client import call_llm_batch, get_api_key  # noqa: E402

ANCHOR_CACHE = (REPO / "experiment_reports_and_results" / "E3_route_provenance_manifest"
                / "main_eligible_atom_tasks_csicl_cache_canonical.jsonl")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    p.add_argument("--suffix", default="", help="artifact suffix for discarded probe runs")
    args = p.parse_args()
    task = args.task

    out_dir = HERE / task
    frozen = (out_dir / "optimized_prompt.txt").read_text(encoding="utf-8")
    opt_summary = json.loads((out_dir / "gepa_opt_summary.json").read_text())
    assert sha256(frozen) == opt_summary["optimized_prompt_sha256"], "frozen prompt hash mismatch"

    test_path = ICR / "datasets" / "bbh" / f"{task}_test.jsonl"
    items = load_jsonl(test_path)
    assert len(items) == 100, f"expected 100 test items, got {len(items)}"
    anchor = {r["item_id"]: r for r in
              (json.loads(l) for l in ANCHOR_CACHE.read_text().splitlines() if l.strip())
              if r["task"] == task}
    assert len(anchor) == 100

    spec = _task_spec_from_cfg(_task_cfg(task))
    prompts = [f"{frozen}\n\n{item['input']}" for item in items]

    t0 = time.time()
    responses = call_llm_batch(
        prompts,
        model=args.model,
        api_key=get_api_key(),
        temperature=0.0,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        progress_label=f"gepa-eval-{task}",
    )
    wall = time.time() - t0

    rows = []
    n_correct = n_parsed = n_fix = n_reg = n_failed = 0
    max_resp_chars = 0
    for item, prompt, resp in zip(items, prompts, responses):
        raw = "" if resp is None else resp.content
        max_resp_chars = max(max_resp_chars, len(raw))
        pred = None if resp is None else _parse_prediction(spec, raw)
        verdict_parsed = bool(raw) and spec.parse_verdict(raw) is not None
        correct = False if pred is None else bool(spec.is_correct(pred, item))
        a = anchor[item["id"]]
        fix = (not a["csicl_correct"]) and correct
        reg = a["csicl_correct"] and not correct
        n_correct += int(correct)
        n_parsed += int(verdict_parsed)
        n_fix += int(fix)
        n_reg += int(reg)
        n_failed += int(resp is None)
        rows.append({
            "task": task,
            "item_id": item["id"],
            "gold": item["answer"],
            "gepa_answer": pred,
            "gepa_correct": correct,
            "verdict_parsed": verdict_parsed,
            "failed_call": resp is None,
            "anchor_answer": a["csicl_answer"],
            "anchor_correct": a["csicl_correct"],
            "fix": fix,
            "regression": reg,
            "prompt_hash": sha256(prompt)[:16],
            "raw_response": raw[:800],
        })

    sfx = args.suffix
    with (out_dir / f"test_eval_per_item{sfx}.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    anchor_correct = sum(int(a["csicl_correct"]) for a in anchor.values())
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "model": args.model,
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "n_items": len(rows),
        "gepa_correct": n_correct,
        "gepa_accuracy": n_correct / len(rows),
        "anchor_correct": anchor_correct,
        "anchor_accuracy": anchor_correct / len(rows),
        "fix": n_fix,
        "regression": n_reg,
        "net": n_fix - n_reg,
        "delta_pp": (n_correct - anchor_correct) / len(rows) * 100,
        "verdict_parse_rate": n_parsed / len(rows),
        "failed_calls": n_failed,
        "max_response_chars": max_resp_chars,
        "optimized_prompt_sha256": sha256(frozen),
        "seed_prompt_sha256": opt_summary["seed_prompt_sha256"],
        "best_is_seed": opt_summary.get("best_is_seed"),
        "eval_wall_seconds": round(wall, 1),
        "anchor_cache": str(ANCHOR_CACHE.relative_to(REPO)),
        "scorer": "ICRefine task specs via _parse_prediction/spec.is_correct",
        "prompt_composition": "optimized prompt (GEPA system message) prepended to item input as a single user message (repo client has no system channel; same handling as PW baseline)",
    }
    (out_dir / f"test_eval_summary{sfx}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
