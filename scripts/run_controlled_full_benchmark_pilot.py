"""Run a controlled full-benchmark pilot for ICRefine/SF-CR.

This is not a broad training sweep. It gives a low-cost readiness check for
expanding from the validated formal_fallacies SF-CR atom union to a larger BBH
benchmark surface:

* formal_fallacies can reuse a cached SF-CR atom-union summary;
* other BBH tasks are scored as raw/empty-cheatsheet model baselines;
* every run writes task-level, item-level, and cost-estimate artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eval_bbh_comparison import TASKS as BBH_COMPARISON_TASKS
from utils.data import load_jsonl
from utils.llm_client import call_llm_batch, get_api_key


DEFAULT_TASKS = [
    name for name in BBH_COMPARISON_TASKS
    if name != "magma"
]

EXTRA_BBH_TASKS = {
    "logical_deduction_five_objects": {
        "test_jsonl": "datasets/bbh/logical_deduction_five_objects_test.jsonl",
        "answer_type": "mc",
        "verdict_format": "(A), (B), (C), (D), or (E)",
    },
    "logical_deduction_seven_objects": {
        "test_jsonl": "datasets/bbh/logical_deduction_seven_objects_test.jsonl",
        "answer_type": "mc",
        "verdict_format": "(A), (B), (C), (D), (E), (F), or (G)",
    },
    "object_counting": {
        "test_jsonl": "datasets/bbh/object_counting_test.jsonl",
        "answer_type": "number",
        "verdict_format": "an integer",
    },
    "penguins_in_a_table": {
        "test_jsonl": "datasets/bbh/penguins_in_a_table_test.jsonl",
        "answer_type": "mc",
        "verdict_format": "(A), (B), (C), (D), or (E)",
    },
    "reasoning_about_colored_objects": {
        "test_jsonl": "datasets/bbh/reasoning_about_colored_objects_test.jsonl",
        "answer_type": "mc",
        "verdict_format": "one option letter in parentheses, such as (A)",
    },
    "temporal_sequences": {
        "test_jsonl": "datasets/bbh/temporal_sequences_test.jsonl",
        "answer_type": "mc",
        "verdict_format": "(A), (B), (C), or (D)",
    },
    "tracking_shuffled_objects_three_objects": {
        "test_jsonl": "datasets/bbh/tracking_shuffled_objects_three_objects_test.jsonl",
        "answer_type": "mc",
        "verdict_format": "(A), (B), or (C)",
    },
}

DEFAULT_TASKS.extend([name for name in EXTRA_BBH_TASKS if name not in DEFAULT_TASKS])


class GenericBBHTask:
    """Small scoring-only adapter for local BBH files without TaskSpec modules."""

    def __init__(self, answer_type: str, verdict_format: str):
        self.answer_type = answer_type
        self.verdict_format = verdict_format

    def build_eval_prompt(self, cheatsheet: str, item: dict) -> str:
        return (
            f"=== CHEATSHEET ===\n{cheatsheet}\n=== END CHEATSHEET ===\n\n"
            f"{item['input']}\n\n"
            "Reply with ONLY the verdict. Do not explain.\n"
            f"VERDICT: {self.verdict_format}"
        )

    def build_scoring_prompt(self, cheatsheet: str, item: dict, cot_first: bool = False) -> str:
        return self.build_eval_prompt(cheatsheet, item)

    def parse_verdict(self, text: str) -> str | None:
        import re

        if self.answer_type == "number":
            labelled = re.search(r"VERDICT\s*:?\s*(-?\d+)", text, re.IGNORECASE)
            if labelled:
                return labelled.group(1)
            m = re.search(r"-?\d+", text.strip())
            return m.group(0) if m else None

        labelled = re.search(r"VERDICT\s*:?\s*\(?([A-Z])\)?", text, re.IGNORECASE)
        if labelled:
            return f"({labelled.group(1).upper()})"
        m = re.search(r"\(([A-Z])\)", text.strip(), re.IGNORECASE)
        return f"({m.group(1).upper()})" if m else None

    def is_correct(self, predicted: str | None, item: dict) -> bool:
        return predicted is not None and predicted.strip().upper() == str(item["answer"]).strip().upper()

    def answer_label(self, item: dict) -> str:
        return str(item["answer"]).strip()


def _load_task(module: str, attr: str):
    return getattr(importlib.import_module(module), attr)


def _task_spec_from_cfg(cfg: dict):
    if "module" in cfg:
        return _load_task(cfg["module"], cfg["attr"])
    return GenericBBHTask(cfg["answer_type"], cfg["verdict_format"])


def _safe_item_id(task: str, item: dict, idx: int) -> str:
    return str(item.get("id") or item.get("_sfcr_id") or f"{task}_{idx:04d}")


def _select_items(items: list[dict], limit: int, sample_mode: str, seed: int) -> list[dict]:
    if not limit or limit >= len(items):
        return list(items)
    if sample_mode == "head":
        return list(items[:limit])
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(items)), limit))
    return [items[i] for i in idxs]


def _parse_prediction(task_spec, text: str) -> str | None:
    predicted = task_spec.parse_verdict(text)
    if predicted is None:
        stripped = text.strip()
        if stripped:
            predicted = stripped
    return predicted


def _estimate_tokens(text: str) -> int:
    # Conservative enough for spend planning, without requiring tokenizer deps.
    return max(1, (len(text) + 3) // 4)


def _estimate_cost(input_tokens: int, output_tokens: int, price_in: float, price_out: float) -> float:
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _score_raw_task(task_name: str, cfg: dict, args, api_key: str) -> tuple[dict, list[dict]]:
    task_spec = _task_spec_from_cfg(cfg)
    all_items = load_jsonl(REPO_ROOT / cfg["test_jsonl"])
    items = _select_items(all_items, args.limit_per_task, args.sample_mode, args.seed)
    eval_fn = getattr(task_spec, "build_eval_prompt", None)
    prompt_mode = "eval_prompt"
    if eval_fn is not None:
        prompts = [eval_fn(args.baseline_cheatsheet, item) for item in items]
    else:
        prompt_mode = "scoring_prompt_fallback"
        prompts = [task_spec.build_scoring_prompt(args.baseline_cheatsheet, item, False) for item in items]
    input_tokens = sum(_estimate_tokens(p) for p in prompts)
    if args.dry_run:
        responses = [None] * len(prompts)
    else:
        responses = call_llm_batch(
            prompts,
            model=args.model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            progress_label=f"pilot:{task_name}",
            reasoning_effort=args.reasoning_effort,
        )

    item_rows: list[dict] = []
    correct = 0
    parse_errors = 0
    failed_calls = 0
    output_tokens = 0
    for idx, (item, resp) in enumerate(zip(items, responses)):
        gold = task_spec.answer_label(item)
        raw = "" if resp is None else resp.content
        if resp is None:
            failed_calls += 1
            predicted = None
            is_correct = False
        else:
            output_tokens += _estimate_tokens(raw)
            predicted = _parse_prediction(task_spec, raw)
            if predicted is None:
                parse_errors += 1
                is_correct = False
            else:
                is_correct = bool(task_spec.is_correct(predicted, item))
        correct += int(is_correct)
        item_rows.append({
            "task": task_name,
            "suite": "bbh",
            "item_id": _safe_item_id(task_name, item, idx),
            "method": "raw_model_empty_cheatsheet",
            "model": args.model,
            "predicted": predicted,
            "gold": gold,
            "correct": is_correct,
            "parse_error": predicted is None,
            "failed_call": resp is None,
            "input_tokens_est": _estimate_tokens(prompts[idx]),
            "output_tokens_est": 0 if resp is None else _estimate_tokens(raw),
            "input_snippet": " ".join(str(item.get("input", "")).split())[:240],
            "raw_response": raw[:500],
        })

    n = len(items)
    cost = _estimate_cost(input_tokens, output_tokens, args.price_in_per_1m, args.price_out_per_1m)
    row = {
        "task": task_name,
        "suite": "bbh",
        "method": "raw_model_empty_cheatsheet",
        "model": args.model,
        "dataset": cfg["test_jsonl"],
        "n_total_test": len(all_items),
        "n_items": n,
        "limit_per_task": args.limit_per_task,
        "baseline_correct": correct,
        "baseline_accuracy": correct / n if n else 0.0,
        "controlled_correct": correct,
        "controlled_accuracy": correct / n if n else 0.0,
        "delta_correct": 0,
        "delta_accuracy": 0.0,
        "parse_error_count": parse_errors,
        "failed_call_count": failed_calls,
        "api_calls": 0 if args.dry_run else n,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 6),
        "prompt_mode": prompt_mode,
        "note": f"No task-specific SF-CR atoms evaluated in this pilot; prompt_mode={prompt_mode}.",
    }
    return row, item_rows


def _load_formal_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    n = int(summary["n_items"])
    baseline_correct = int(summary["baseline_correct"])
    controlled_correct = int(summary["union_correct"])
    return {
        "task": "formal_fallacies",
        "suite": "bbh",
        "method": "cached_sfcr_atom_union_v3",
        "model": summary.get("model_score", ""),
        "dataset": summary.get("dataset", ""),
        "n_total_test": n,
        "n_items": n,
        "limit_per_task": "cached_full_formal",
        "baseline_correct": baseline_correct,
        "baseline_accuracy": float(summary["baseline_accuracy"]),
        "controlled_correct": controlled_correct,
        "controlled_accuracy": float(summary["union_accuracy"]),
        "delta_correct": controlled_correct - baseline_correct,
        "delta_accuracy": float(summary["union_accuracy"]) - float(summary["baseline_accuracy"]),
        "parse_error_count": int(summary.get("parse_error_count", 0)),
        "failed_call_count": 0,
        "api_calls": 0,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "activated_count": int(summary.get("activated_count", 0)),
        "fixed_count": int(summary.get("fixed_count", 0)),
        "regression_count": int(summary.get("regression_count", 0)),
        "net_count": int(summary.get("net_count", 0)),
        "source_summary": str(path),
        "note": "Reused validated formal_fallacies SF-CR routed atom-union summary.",
    }


def _aggregate(rows: list[dict]) -> dict:
    n = sum(int(row["n_items"]) for row in rows)
    baseline_correct = sum(int(row["baseline_correct"]) for row in rows)
    controlled_correct = sum(int(row["controlled_correct"]) for row in rows)
    input_tokens = sum(int(row.get("estimated_input_tokens", 0)) for row in rows)
    output_tokens = sum(int(row.get("estimated_output_tokens", 0)) for row in rows)
    return {
        "n_tasks": len(rows),
        "n_items": n,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / n if n else 0.0,
        "controlled_correct": controlled_correct,
        "controlled_accuracy": controlled_correct / n if n else 0.0,
        "delta_correct": controlled_correct - baseline_correct,
        "delta_accuracy": (controlled_correct - baseline_correct) / n if n else 0.0,
        "parse_error_count": sum(int(row.get("parse_error_count", 0)) for row in rows),
        "failed_call_count": sum(int(row.get("failed_call_count", 0)) for row in rows),
        "api_calls": sum(int(row.get("api_calls", 0)) for row in rows),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd", 0.0)) for row in rows), 6),
    }


def _agieval_status() -> str:
    patterns = ["datasets/agieval", "datasets/agieval_*", "datasets/*agieval*"]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(REPO_ROOT.glob(pattern))
    return "present" if matches else "not_found"


def _write_report(path: Path, args, rows: list[dict], aggregate: dict) -> None:
    lines = [
        "# Controlled Full-Benchmark Pilot",
        "",
        f"- time: {datetime.now().isoformat(timespec='seconds')}",
        f"- model: `{args.model}`",
        f"- limit_per_task: `{args.limit_per_task}`",
        f"- sample_mode: `{args.sample_mode}`",
        f"- seed: `{args.seed}`",
        f"- formal_fallacies mode: `{args.formal_mode}`",
        f"- AGIEval local datasets: `{_agieval_status()}`",
        "",
        "## Aggregate",
        "",
        f"- tasks: {aggregate['n_tasks']}",
        f"- items: {aggregate['n_items']}",
        f"- baseline accuracy: {aggregate['baseline_accuracy']:.3f}",
        f"- controlled accuracy: {aggregate['controlled_accuracy']:.3f}",
        f"- delta: {aggregate['delta_accuracy']:+.3f} ({aggregate['delta_correct']:+d} items)",
        f"- parse errors: {aggregate['parse_error_count']}",
        f"- failed calls: {aggregate['failed_call_count']}",
        f"- API calls charged in this pilot: {aggregate['api_calls']}",
        f"- estimated input tokens: {aggregate['estimated_input_tokens']}",
        f"- estimated output tokens: {aggregate['estimated_output_tokens']}",
        f"- estimated API cost USD: {aggregate['estimated_cost_usd']:.6f}",
        "",
        "## Per Task",
        "",
        "| task | n | baseline | controlled | delta | parse errors | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {n_items} | {baseline_accuracy:.3f} | {controlled_accuracy:.3f} | "
            "{delta_accuracy:+.3f} | {parse_error_count} | {note} |".format(**row)
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Only `formal_fallacies` currently uses the validated SF-CR routed atom union.",
        "- Other tasks are raw model baselines with an empty cheatsheet, included to test full-benchmark scoring stability and cost.",
        "- A true BBH/AGIEval full benchmark requires task-specific candidate generation or imported cheatsheets/atoms for each task.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    p.add_argument("--limit-per-task", type=int, default=30)
    p.add_argument("--sample-mode", choices=["random", "head"], default="random")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--baseline-cheatsheet", default="")
    p.add_argument("--formal-mode", choices=["reuse", "skip"], default="reuse")
    p.add_argument(
        "--formal-summary",
        default="runs/ff_gpt41mini_safe_atom_union_v3_test100_guard_rerun/atom_union_summary.json",
    )
    p.add_argument("--price-in-per-1m", type=float, default=0.40)
    p.add_argument("--price-out-per-1m", type=float, default=1.60)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "runs" / (
        "controlled_full_benchmark_pilot_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = "" if args.dry_run else get_api_key()
    task_rows: list[dict] = []
    item_rows: list[dict] = []

    for task_name in args.tasks:
        if task_name == "formal_fallacies":
            if args.formal_mode == "skip":
                continue
            formal_path = REPO_ROOT / args.formal_summary
            if not formal_path.exists():
                print(f"[warn] missing formal summary: {formal_path}", file=sys.stderr)
                continue
            row = _load_formal_summary(formal_path)
            task_rows.append(row)
            print(f"[formal_fallacies] reused {formal_path}", file=sys.stderr)
            continue

        if task_name in BBH_COMPARISON_TASKS:
            cfg = BBH_COMPARISON_TASKS[task_name]
        elif task_name in EXTRA_BBH_TASKS:
            cfg = EXTRA_BBH_TASKS[task_name]
        else:
            print(f"[warn] unknown task {task_name!r}; skipping", file=sys.stderr)
            continue
        row, rows = _score_raw_task(task_name, cfg, args, api_key)
        task_rows.append(row)
        item_rows.extend(rows)
        print(
            f"[{task_name}] acc={row['baseline_accuracy']:.3f} "
            f"parse={row['parse_error_count']} failed={row['failed_call_count']}",
            file=sys.stderr,
        )

    aggregate = _aggregate(task_rows)
    payload = {
        "config": vars(args),
        "aggregate": aggregate,
        "tasks": task_rows,
    }
    (out_dir / "pilot_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "pilot_results.csv", task_rows)
    _write_jsonl(out_dir / "pilot_item_results.jsonl", item_rows)
    _write_csv(out_dir / "pilot_item_results.csv", item_rows)
    _write_report(out_dir / "controlled_full_benchmark_pilot_report.md", args, task_rows, aggregate)
    print(json.dumps({"output_dir": str(out_dir), "aggregate": aggregate}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
