"""Frozen-prompt test evaluation for the ProTeGi baseline (ARR rebuttal).

Usage: .venv/bin/python run_protegi_test_eval.py <task> [--max-tokens 64]

- Freezes the ProTeGi-optimized cheatsheet (copied + sha256-hashed here)
  together with the seed (locked CS-ICL anchor) cheatsheet and the verbatim
  ProTeGi config.
- Runs it once on the task's 100 test items via the repo's LLM machinery
  (utils.llm_client.call_llm_batch: temperature 0.0, seed 42, model
  openai/gpt-4.1-mini) and scores with the repo's task specs/parsers
  (_task_cfg -> _task_spec_from_cfg -> _parse_prediction / spec.is_correct).
  No scoring logic is reimplemented.
- Prompt construction is EXACTLY the locked anchor presentation:
  task_spec.build_eval_prompt(optimized_cheatsheet, item) — the same
  verdict-only wrapper and max_tokens=64 as the locked CS-ICL anchor eval
  (run_sfcr_csicl_anchor_eval.py). This is the like-for-like comparison:
  ProTeGi refined the cheatsheet inside the wrapper; the wrapper is unchanged.
- Compares per-item against the locked CS-ICL anchor cache
  (experiment_reports_and_results/E3_route_provenance_manifest/
   main_eligible_atom_tasks_csicl_cache_canonical.jsonl):
  fix = anchor-wrong -> ProTeGi-correct; regression = anchor-correct ->
  ProTeGi-wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ICR = REPO / "ICRefine"
PROTEGI_BASE = Path("<PROTEGI_WORKDIR>")

sys.path.insert(0, str(ICR))

from scripts.run_controlled_full_benchmark_pilot import _parse_prediction, _task_spec_from_cfg  # noqa: E402
from scripts.run_sfcr_emnlp_baseline_surface import _task_cfg  # noqa: E402
from utils.data import load_jsonl  # noqa: E402
from utils.llm_client import call_llm_batch, get_api_key  # noqa: E402

ANCHOR_CACHE = (REPO / "experiment_reports_and_results" / "E3_route_provenance_manifest"
                / "main_eligible_atom_tasks_csicl_cache_canonical.jsonl")

LOCKED_CS = (ICR / "runs" / "emnlp_bbh_full18_csicl_gpt41mini_20260518" / "cheatsheets"
             / "{task}" / "csicl_gpt-4.1-mini_seed1000.txt")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=14)
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    args = p.parse_args()
    task = args.task

    run_dir = PROTEGI_BASE / "runs" / task
    optimized = (run_dir / "optimized_cheatsheet.txt").read_text()
    final_sel = json.loads((run_dir / "final_selection.json").read_text())
    config_used = json.loads((run_dir / "config_used.json").read_text())
    locked_seed = Path(str(LOCKED_CS).format(task=task)).read_text()
    assert sha256(locked_seed) == config_used["seed_cheatsheet_sha256"], \
        "seed cheatsheet does not match the locked CS-ICL anchor file"
    assert sha256(optimized) == final_sel["optimized_cheatsheet_sha256"]

    out_dir = HERE / task
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- freeze artifacts ---------------------------------------------------
    (out_dir / "optimized_prompt.txt").write_text(optimized)
    (out_dir / "seed_prompt.txt").write_text(locked_seed)
    shutil.copy(run_dir / "config_used.json", out_dir / "config_used.json")
    shutil.copy(run_dir / "final_selection.json", out_dir / "final_selection.json")

    # ---- test items + anchor -----------------------------------------------
    items = load_jsonl(ICR / "datasets" / "bbh" / f"{task}_test.jsonl")
    assert len(items) == 100, f"expected 100 test items, got {len(items)}"
    anchor = {r["item_id"]: r for r in
              (json.loads(l) for l in ANCHOR_CACHE.read_text().splitlines() if l.strip())
              if r["task"] == task}
    assert len(anchor) == 100

    spec = _task_spec_from_cfg(_task_cfg(task))
    prompts = [spec.build_eval_prompt(optimized, item) for item in items]

    t0 = time.time()
    responses = call_llm_batch(
        prompts,
        model=args.model,
        api_key=get_api_key(),
        temperature=0.0,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        progress_label=f"protegi-eval-{task}",
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
            "protegi_answer": pred,
            "protegi_correct": correct,
            "verdict_parsed": verdict_parsed,
            "failed_call": resp is None,
            "anchor_answer": a["csicl_answer"],
            "anchor_correct": a["csicl_correct"],
            "fix": fix,
            "regression": reg,
            "prompt_hash": sha256(prompt)[:16],
            "raw_response": raw[:800],
        })

    with (out_dir / "test_eval_per_item.jsonl").open("w") as fh:
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
        "protegi_correct": n_correct,
        "protegi_accuracy": n_correct / len(rows),
        "anchor_correct": anchor_correct,
        "anchor_accuracy": anchor_correct / len(rows),
        "fix": n_fix,
        "regression": n_reg,
        "net": n_fix - n_reg,
        "delta_pp": (n_correct - anchor_correct) / len(rows) * 100,
        "verdict_parse_rate": n_parsed / len(rows),
        "failed_calls": n_failed,
        "max_response_chars": max_resp_chars,
        "optimized_prompt_sha256": sha256(optimized),
        "seed_prompt_sha256": sha256(locked_seed),
        "cheatsheet_changed_by_protegi": sha256(optimized) != sha256(locked_seed),
        "protegi_selected_train_acc": final_sel["selected_train_acc"],
        "protegi_optimization_calls": final_sel["n_llm_calls"],
        "protegi_optimization_est_cost_usd": final_sel["est_cost_usd"],
        "eval_wall_seconds": round(wall, 1),
        "anchor_cache": str(ANCHOR_CACHE.relative_to(REPO)),
        "scorer": "ICRefine task specs via _parse_prediction/spec.is_correct",
        "prompt_builder": "task_spec.build_eval_prompt (identical wrapper to locked anchor eval)",
    }
    (out_dir / "test_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
