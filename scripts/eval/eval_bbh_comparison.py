"""
eval_bbh_comparison.py — Compare our refined cheat sheet against cheat-sheet ICL baseline.

Scores both cheat sheets on the held-out test split using our TaskSpec-based scorer,
then prints side-by-side accuracy + cost-efficiency tables.

Cost model (gpt-4.1-2025-04-14 list prices):
  Input : $2.00 / 1M tokens
  Output: $8.00 / 1M tokens

CS-ICL cost: one generation call (all train items concatenated → cheat sheet).
ICR-Hybrid cost: estimated from run log (scoring passes × items × prompt size +
                 generation calls × output size).

Usage:
    python3 eval_bbh_comparison.py \
        --run-dir runs/bbh_rerun \
        --cs-icl-dir ../cheat-sheet-icl/data/cheat_prompt \
        --model openai/gpt-4.1-2025-04-14 \
        --concurrency 16
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / "ICR_partition" / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data import load_jsonl
from utils.llm_client import get_api_key
from utils.scorer import score_batch


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "magma": {
        "task_flag":   "magma",
        "module":      "tasks.magma",
        "attr":        "MAGMA_TASK",
        "test_jsonl":  "datasets/magma_test.jsonl",
        "train_n":     100,
        "cs_icl_dir":  "magma",
        "cs_icl_file": "gen_gpt-4.1-mini_0_1000.txt",
    },
    "boolean_expressions": {
        "task_flag":   "bbh_boolean",
        "module":      "tasks.bbh_boolean",
        "attr":        "BBH_BOOLEAN_TASK",
        "test_jsonl":  "datasets/bbh/boolean_expressions_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_boolean_expressions",
        "cs_icl_file": "gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "causal_judgement": {
        "task_flag":   "causal_judgement",
        "module":      "tasks.bbh_tasks",
        "attr":        "CAUSAL_JUDGEMENT_TASK",
        "test_jsonl":  "datasets/bbh/causal_judgement_test.jsonl",
        "train_n":     100,
        "cs_icl_dir":  "bbh_causal_judgement",
        "cs_icl_file": "gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "disambiguation_qa": {
        "task_flag":   "disambiguation_qa",
        "module":      "tasks.bbh_tasks",
        "attr":        "DISAMBIGUATION_TASK",
        "test_jsonl":  "datasets/bbh/disambiguation_qa_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_disambiguation_qa",
        "cs_icl_file": "gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "geometric_shapes": {
        "task_flag":   "geometric_shapes",
        "module":      "tasks.bbh_tasks",
        "attr":        "GEOMETRIC_TASK",
        "test_jsonl":  "datasets/bbh/geometric_shapes_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_geometric_shapes",
        "cs_icl_file": "gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "sports_understanding": {
        "task_flag":   "sports_understanding",
        "module":      "tasks.bbh_tasks",
        "attr":        "SPORTS_TASK",
        "test_jsonl":  "datasets/bbh/sports_understanding_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_sports_understanding",
        "cs_icl_file": "gen_gpt-4.1-2025-04-14_0_1000.txt",
    },
    "formal_fallacies": {
        "task_flag":   "formal_fallacies",
        "module":      "tasks.bbh_tasks_ext",
        "attr":        "FORMAL_FALLACIES_TASK",
        "test_jsonl":  "datasets/bbh/formal_fallacies_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_formal_fallacies",
        "cs_icl_file": "gen_gpt-4.1-mini_0_1000.txt",
    },
    "logical_deduction_three": {
        "task_flag":   "logical_deduction_three",
        "module":      "tasks.bbh_tasks_ext",
        "attr":        "LOGICAL_DEDUCTION_3_TASK",
        "test_jsonl":  "datasets/bbh/logical_deduction_three_objects_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_logical_deduction_three_objects",
        "cs_icl_file": "gen_gpt-4.1-mini_0_1000.txt",
    },
    "web_of_lies": {
        "task_flag":   "web_of_lies",
        "module":      "tasks.bbh_tasks_ext",
        "attr":        "WEB_OF_LIES_TASK",
        "test_jsonl":  "datasets/bbh/web_of_lies_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_web_of_lies",
        "cs_icl_file": "gen_gpt-4.1-mini_0_1000.txt",
    },
    "date_understanding": {
        "task_flag":   "date_understanding",
        "module":      "tasks.bbh_tasks_ext",
        "attr":        "DATE_UNDERSTANDING_TASK",
        "test_jsonl":  "datasets/bbh/date_understanding_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_date_understanding",
        "cs_icl_file": "gen_gpt-4.1-mini_0_1000.txt",
    },
    "navigate": {
        "task_flag":   "navigate",
        "module":      "tasks.bbh_tasks_ext",
        "attr":        "NAVIGATE_TASK",
        "test_jsonl":  "datasets/bbh/navigate_test.jsonl",
        "train_n":     150,
        "cs_icl_dir":  "bbh_navigate",
        "cs_icl_file": "gen_gpt-4.1-mini_0_1000.txt",
    },
    "snarks": {
        "task_flag":   "snarks",
        "module":      "tasks.bbh_tasks_ext",
        "attr":        "SNARKS_TASK",
        "test_jsonl":  "datasets/bbh/snarks_test.jsonl",
        "train_n":     107,
        "cs_icl_dir":  "bbh_snarks",
        "cs_icl_file": "gen_gpt-4.1-mini_0_1000.txt",
    },
}

# gpt-4.1-2025-04-14 list prices ($/1M tokens)
_PRICE_IN  = 2.00
_PRICE_OUT = 8.00

# Approximate token sizes
_AVG_ITEM_TOKENS      = 250   # avg tokens per train item (input + reason + answer)
_CS_ICL_GEN_OUT       = 2000  # avg output tokens for CS-ICL cheat sheet generation
_CS_ICL_EVAL_IN       = 2000  # avg tokens per test scoring call (cheat sheet + item)
_CS_ICL_EVAL_OUT      = 100   # avg output tokens per test scoring call

_ICR_SCORE_IN         = 800   # avg tokens per ICR scoring call (cheat sheet + item)
_ICR_SCORE_OUT        = 150   # avg output tokens per scoring call
_ICR_GEN_IN           = 3000  # avg input tokens per case-study / rule-patch generation
_ICR_GEN_OUT          = 800   # avg output tokens per generation call


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def _tok_cost(in_tok: int, out_tok: int) -> float:
    return in_tok * _PRICE_IN / 1e6 + out_tok * _PRICE_OUT / 1e6


def _estimate_csicl_cost(train_n: int, test_n: int) -> dict:
    """One-shot cheat sheet generation + test scoring."""
    gen_in   = train_n * _AVG_ITEM_TOKENS
    gen_out  = _CS_ICL_GEN_OUT
    eval_in  = test_n * _CS_ICL_EVAL_IN
    eval_out = test_n * _CS_ICL_EVAL_OUT
    gen_cost  = _tok_cost(gen_in,  gen_out)
    eval_cost = _tok_cost(eval_in, eval_out)
    return {
        "generation_usd": round(gen_cost,  4),
        "eval_usd":       round(eval_cost, 4),
        "total_usd":      round(gen_cost + eval_cost, 4),
        "total_tokens":   gen_in + gen_out + eval_in + eval_out,
    }


def _estimate_icr_cost_from_log(log_path: Path, train_n: int, test_n: int) -> dict:
    """Parse run.log to count scoring passes and generation calls, then price them."""
    if not log_path.exists():
        return {"total_usd": None, "note": "log not found"}

    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Count full scoring passes (bootstrap + phase1 init + phase2 iter 0)
    full_score_passes  = len(re.findall(r"Scoring all \d+ items", text))
    # Count rescore passes (active-bin items only — smaller)
    rescore_matches    = re.findall(r"Re-scoring (\d+) active-bin items", text)
    rescore_items      = sum(int(x) for x in rescore_matches)
    # Count generation calls (candidates + rule patches)
    gen_calls          = len(re.findall(
        r"(Generating \d+ candidates|rule-patch.*targeting|bootstrap.*calling LLM)", text
    ))

    scoring_in  = full_score_passes * train_n * _ICR_SCORE_IN + rescore_items * _ICR_SCORE_IN
    scoring_out = (full_score_passes * train_n + rescore_items) * _ICR_SCORE_OUT
    gen_in      = gen_calls * _ICR_GEN_IN
    gen_out     = gen_calls * _ICR_GEN_OUT
    # Final eval on test set
    eval_in     = test_n * _ICR_SCORE_IN
    eval_out    = test_n * _ICR_SCORE_OUT

    total_in  = scoring_in  + gen_in  + eval_in
    total_out = scoring_out + gen_out + eval_out

    return {
        "full_score_passes":  full_score_passes,
        "rescore_items":      rescore_items,
        "gen_calls":          gen_calls,
        "scoring_usd":        round(_tok_cost(scoring_in, scoring_out), 4),
        "generation_usd":     round(_tok_cost(gen_in, gen_out), 4),
        "eval_usd":           round(_tok_cost(eval_in, eval_out), 4),
        "total_usd":          round(_tok_cost(total_in, total_out), 4),
        "total_tokens":       total_in + total_out,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _load_task(module: str, attr: str):
    return getattr(importlib.import_module(module), attr)


def _score_cheatsheet(items, cs_text, task_spec, model, api_key, concurrency, label):
    print(f"  [{label}] scoring {len(items)} test items ...", file=sys.stderr)
    # Use verdict-only eval prompt when available — no reasoning scaffold,
    # lower token cost, cleaner accuracy signal on the held-out test set.
    eval_fn = getattr(task_spec, "build_eval_prompt", None)
    if eval_fn is not None:
        from utils.llm_client import call_llm_batch
        prompts = [eval_fn(cs_text, item) for item in items]
        responses = call_llm_batch(
            prompts, model=model, api_key=api_key,
            temperature=0.0, max_tokens=32,
            concurrency=concurrency,
            progress_label=label,
            reasoning_effort=None,
        )
        def _check(item, resp):
            if resp is None:
                return False
            # parse_verdict requires "VERDICT: ..." prefix; bare completions skip that.
            # Fall back to raw stripped content so "Yes", "No", "valid" etc. still match.
            predicted = task_spec.parse_verdict(resp.content)
            if predicted is None:
                predicted = resp.content.strip()
            return task_spec.is_correct(predicted, item)

        correct = sum(_check(item, resp) for item, resp in zip(items, responses))
        acc = correct / len(items) if items else 0.0
    else:
        correct_items, _ = score_batch(
            items, cs_text, model, api_key,
            concurrency=concurrency,
            reasoning_effort=None,
            cot_first=True,
            task_spec=task_spec,
        )
        acc = len(correct_items) / len(items) if items else 0.0
    print(f"  [{label}] acc = {acc:.1%}", file=sys.stderr)
    return acc


def _load_cs_icl(cs_icl_base: Path, cfg: dict) -> str | None:
    p = cs_icl_base / cfg["cs_icl_dir"] / cfg["cs_icl_file"]
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def _load_our_cs(run_dir: Path, task_name: str) -> str | None:
    for name in ("cheatsheet_final.txt", "cheatsheet_current.txt"):
        p = run_dir / task_name / name
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",     default="runs/bbh_rerun")
    p.add_argument("--cs-icl-dir",  default="../cheat-sheet-icl/data/cheat_prompt")
    p.add_argument("--model",       default="openai/gpt-4.1-2025-04-14")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--tasks",       nargs="+", default=list(TASKS.keys()))
    p.add_argument("--no-ours",     action="store_true")
    p.add_argument("--no-csicl",    action="store_true")
    args = p.parse_args()

    api_key    = get_api_key()
    from utils.run_logger import RunLogger, make_run_id, set_logger
    _run_logger = RunLogger(log_base="runs/logs/eval", run_id=make_run_id("bbh_comparison"), config=vars(args))
    set_logger(_run_logger)
    print(f"[log] {_run_logger.log_dir}", file=sys.stderr)
    run_dir    = Path(args.run_dir)
    cs_icl_dir = Path(args.cs_icl_dir)

    results = {}

    for task_name in args.tasks:
        if task_name not in TASKS:
            print(f"[warn] unknown task {task_name!r} — skipping", file=sys.stderr)
            continue

        cfg        = TASKS[task_name]
        task_spec  = _load_task(cfg["module"], cfg["attr"])
        test_items = load_jsonl(Path(cfg["test_jsonl"]))
        train_n    = cfg["train_n"]
        test_n     = len(test_items)

        print(f"\n{'='*55}", file=sys.stderr)
        print(f"  {task_name}  (train={train_n}  test={test_n})", file=sys.stderr)
        print(f"{'='*55}", file=sys.stderr)

        row: dict = {
            "n_train": train_n, "n_test": test_n,
            "cs_icl_acc": None, "ours_acc": None,
            "cs_icl_cost": None, "ours_cost": None,
        }

        # CS-ICL
        if not args.no_csicl:
            cs_text = _load_cs_icl(cs_icl_dir, cfg)
            if cs_text is None:
                print("  [cs-icl] cheat sheet not found", file=sys.stderr)
            else:
                row["cs_icl_acc"]  = _score_cheatsheet(
                    test_items, cs_text, task_spec,
                    args.model, api_key, args.concurrency, "cs-icl",
                )
                row["cs_icl_cost"] = _estimate_csicl_cost(train_n, test_n)

        # Ours
        if not args.no_ours:
            our_cs = _load_our_cs(run_dir, task_name)
            if our_cs is None:
                print(f"  [ours] cheat sheet not found in {run_dir / task_name}", file=sys.stderr)
            else:
                row["ours_acc"]  = _score_cheatsheet(
                    test_items, our_cs, task_spec,
                    args.model, api_key, args.concurrency, "ours",
                )
                row["ours_cost"] = _estimate_icr_cost_from_log(
                    run_dir / task_name / "run.log", train_n, test_n,
                )

        results[task_name] = row

    # ── Accuracy table ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ACCURACY (test split)")
    print(f"{'Task':<25} {'N_test':>6}  {'CS-ICL':>8}  {'Ours':>8}  {'Delta':>8}")
    print("-"*70)
    for task_name, row in results.items():
        cs  = f"{row['cs_icl_acc']:.1%}" if row["cs_icl_acc"] is not None else "N/A"
        our = f"{row['ours_acc']:.1%}"   if row["ours_acc"]   is not None else "N/A"
        if row["cs_icl_acc"] is not None and row["ours_acc"] is not None:
            delta = f"{row['ours_acc'] - row['cs_icl_acc']:+.1%}"
        else:
            delta = "N/A"
        print(f"{task_name:<25} {row['n_test']:>6}  {cs:>8}  {our:>8}  {delta:>8}")
    print("="*70)

    # ── Cost-efficiency table ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("COST EFFICIENCY  (gpt-4.1 estimated @ $2/M in, $8/M out)")
    print(f"{'Task':<25} {'CS-ICL $':>10}  {'Ours $':>10}  {'$/pp (CS-ICL)':>14}  {'$/pp (Ours)':>12}")
    print("-"*70)
    for task_name, row in results.items():
        csicl_usd = row["cs_icl_cost"]["total_usd"] if row["cs_icl_cost"] else None
        ours_usd  = row["ours_cost"]["total_usd"]   if row["ours_cost"]   else None
        csicl_acc = row["cs_icl_acc"]
        ours_acc  = row["ours_acc"]

        # cost-per-percentage-point vs zero-shot (no cheat sheet, assume ~50% baseline)
        # We use accuracy gain over zero-shot = 50% as denominator
        zero_shot_assumed = 0.50
        csicl_str  = f"${csicl_usd:.4f}" if csicl_usd is not None else "N/A"
        ours_str   = f"${ours_usd:.4f}"  if ours_usd  is not None else "N/A"

        if csicl_usd and csicl_acc:
            gain_cs = max(csicl_acc - zero_shot_assumed, 0.001)
            cpp_cs  = f"${csicl_usd / (gain_cs * 100):.4f}"
        else:
            cpp_cs  = "N/A"

        if ours_usd and ours_acc:
            gain_ou = max(ours_acc - zero_shot_assumed, 0.001)
            cpp_ou  = f"${ours_usd / (gain_ou * 100):.4f}"
        else:
            cpp_ou  = "N/A"

        print(f"{task_name:<25} {csicl_str:>10}  {ours_str:>10}  {cpp_cs:>14}  {cpp_ou:>12}")
    print("="*70)
    print("  $/pp = estimated cost per percentage-point accuracy gain over 50% baseline")

    # ── Save — merge into existing results so per-task calls accumulate ───────
    out_path = run_dir / "comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Field-level merge: only overwrite a field if the new value is not None
    for task_name, row in results.items():
        if task_name not in existing:
            existing[task_name] = row
        else:
            for k, v in row.items():
                if v is not None:
                    existing[task_name][k] = v
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nDetailed results → {out_path}  ({len(existing)} tasks total)")


if __name__ == "__main__":
    main()
