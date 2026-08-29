"""Run EMNLP-oriented BBH baseline calibration for SF-CR.

This script evaluates per-task CS-ICL cheat sheets on the same BBH task surface
used by the controlled SF-CR pilot. It writes protocol-stamped per-item outputs
so CS-ICL, raw baseline, and controlled SF-CR rows can be compared without
mixing incompatible item lists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_controlled_full_benchmark_pilot import (  # noqa: E402
    BBH_COMPARISON_TASKS,
    DEFAULT_TASKS,
    EXTRA_BBH_TASKS,
    _estimate_cost,
    _estimate_tokens,
    _parse_prediction,
    _safe_item_id,
    _select_items,
    _task_spec_from_cfg,
)
from utils.data import load_jsonl  # noqa: E402
from utils.llm_client import call_llm, call_llm_batch, get_api_key  # noqa: E402


GEN_PROMPT = """\
Create a cheat sheet based on the examples below. You will be asked to answer
questions similar to these examples during the test, without being allowed to
refer to the examples at that time. Your task here is to make a cheat sheet that
will help you answer such problems correctly. First, carefully read the examples
below and identify which ones are most difficult to answer.

{dataset_str}

Now, create a cheat sheet to help solve the difficult examples. Exclude content
that is easy, and include only specific, detailed points that address the
challenging examples.
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _task_cfg(task: str) -> dict:
    if task in BBH_COMPARISON_TASKS:
        return dict(BBH_COMPARISON_TASKS[task])
    if task in EXTRA_BBH_TASKS:
        return dict(EXTRA_BBH_TASKS[task])
    raise KeyError(task)


def _train_path_from_test(test_jsonl: str) -> Path:
    p = Path(test_jsonl)
    name = p.name
    if not name.endswith("_test.jsonl"):
        raise ValueError(f"Cannot infer train path from {test_jsonl}")
    return REPO_ROOT / p.with_name(name.replace("_test.jsonl", "_train.jsonl"))


def _format_train_item(item: dict) -> str:
    reason = str(item.get("reason", "")).strip()
    if reason:
        return f"Question: {item['input']}\nReasoning: {reason}\nAnswer: {item['answer']}"
    return f"Question: {item['input']}\nAnswer: {item['answer']}"


def _build_csicl_prompt(train_items: list[dict], seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    items = list(train_items)
    rng.shuffle(items)
    formatted = [_format_train_item(it) for it in items]
    dataset_str = "\n\n".join(formatted)
    show_examples = "\n###\n".join(formatted[:2])
    prompt = GEN_PROMPT.format(dataset_str=dataset_str)
    suffix = f"\n\nFollow the format of the examples below in your response.\n\n{show_examples}"
    return prompt, suffix


def _generation_meta(task: str, cfg: dict, args: argparse.Namespace, generated_text: str) -> dict:
    train_path = _train_path_from_test(cfg["test_jsonl"])
    train_items = load_jsonl(train_path)
    n = min(args.train_n, len(train_items)) if args.train_n else len(train_items)
    prompt, _ = _build_csicl_prompt(train_items[:n], args.csicl_seed)
    input_tokens = _estimate_tokens(prompt)
    output_tokens = _estimate_tokens(generated_text)
    return {
        "train_path": str(train_path.relative_to(REPO_ROOT)),
        "n_train": n,
        "generation_prompt_chars": len(prompt),
        "generation_input_tokens": input_tokens,
        "generation_output_tokens": output_tokens,
        "generation_estimated_cost_usd": round(
            _estimate_cost(input_tokens, output_tokens, args.price_in_per_1m, args.price_out_per_1m),
            6,
        ),
    }


def _load_or_generate_csicl(task: str, cfg: dict, args: argparse.Namespace, api_key: str) -> tuple[str, dict]:
    out = Path(args.output_dir)
    model_short = args.generator_model.split("/")[-1].replace(":", "_")
    cs_path = out / "cheatsheets" / task / f"csicl_{model_short}_seed{args.csicl_seed}.txt"
    meta_path = cs_path.with_suffix(".json")
    if cs_path.exists() and not args.regenerate_cheatsheets:
        cs_text = cs_path.read_text(encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "generation_estimated_cost_usd" not in meta:
            meta.update(_generation_meta(task, cfg, args, cs_text))
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return cs_text, meta

    train_path = _train_path_from_test(cfg["test_jsonl"])
    train_items = load_jsonl(train_path)
    n = min(args.train_n, len(train_items)) if args.train_n else len(train_items)
    prompt, suffix = _build_csicl_prompt(train_items[:n], args.csicl_seed)
    print(f"[csicl-gen] {task}: n_train={n} prompt_chars={len(prompt)}", file=sys.stderr)
    resp = call_llm(
        prompt,
        model=args.generator_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.generation_max_tokens,
    )
    cs_text = resp.content.strip() + suffix
    cs_path.parent.mkdir(parents=True, exist_ok=True)
    cs_path.write_text(cs_text, encoding="utf-8")
    meta = {
        "task": task,
        "condition": "csicl",
        "generator_model": args.generator_model,
        "seed": args.csicl_seed,
        "train_path": str(train_path.relative_to(REPO_ROOT)),
        "n_train": n,
        "prompt_hash": _sha(prompt),
        "cheatsheet_hash": _sha(cs_text),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta.update(
        {
            "generation_prompt_chars": len(prompt),
            "generation_input_tokens": _estimate_tokens(prompt),
            "generation_output_tokens": _estimate_tokens(resp.content),
            "generation_estimated_cost_usd": round(
                _estimate_cost(
                    _estimate_tokens(prompt),
                    _estimate_tokens(resp.content),
                    args.price_in_per_1m,
                    args.price_out_per_1m,
                ),
                6,
            ),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return cs_text, meta


def _score_condition(
    task: str,
    cfg: dict,
    condition: str,
    cheatsheet: str,
    args: argparse.Namespace,
    api_key: str,
    meta: dict,
) -> tuple[dict, list[dict]]:
    task_spec = _task_spec_from_cfg(cfg)
    all_items = load_jsonl(REPO_ROOT / cfg["test_jsonl"])
    items = _select_items(all_items, args.limit_per_task, args.sample_mode, args.seed)
    eval_fn = getattr(task_spec, "build_eval_prompt", None)
    prompt_builder = "task_spec.build_eval_prompt"
    if eval_fn is not None:
        prompts = [eval_fn(cheatsheet, item) for item in items]
    else:
        prompt_builder = "task_spec.build_scoring_prompt"
        prompts = [task_spec.build_scoring_prompt(cheatsheet, item, False) for item in items]
    print(f"[score] {condition}:{task} n={len(items)}", file=sys.stderr)
    responses = call_llm_batch(
        prompts,
        model=args.eval_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.eval_max_tokens,
        concurrency=args.concurrency,
        progress_label=f"{condition}:{task}",
        reasoning_effort=args.reasoning_effort,
    )
    rows: list[dict] = []
    correct = 0
    parse_errors = 0
    failed_calls = 0
    input_tokens = 0
    output_tokens = 0
    protocol_signature = _sha(
        json.dumps(
            {
                "condition": condition,
                "eval_model": args.eval_model,
                "generator_model": args.generator_model,
                "task": task,
                "cheatsheet_hash": meta.get("cheatsheet_hash", _sha(cheatsheet)),
                "prompt_builder": prompt_builder,
                "eval_max_tokens": args.eval_max_tokens,
            },
            sort_keys=True,
        )
    )
    for idx, (item, prompt, resp) in enumerate(zip(items, prompts, responses)):
        input_tokens += _estimate_tokens(prompt)
        gold = task_spec.answer_label(item)
        raw = "" if resp is None else resp.content
        if resp is None:
            pred = None
            is_correct = False
            failed_calls += 1
        else:
            output_tokens += _estimate_tokens(raw)
            pred = _parse_prediction(task_spec, raw)
            if pred is None:
                parse_errors += 1
                is_correct = False
            else:
                is_correct = bool(task_spec.is_correct(pred, item))
        correct += int(is_correct)
        rows.append(
            {
                "condition": condition,
                "task": task,
                "item_id": _safe_item_id(task, item, idx),
                "model": args.eval_model,
                "answer": pred,
                "gold": gold,
                "correct": is_correct,
                "parse_error": pred is None,
                "failed_call": resp is None,
                "prompt_hash": _sha(prompt),
                "protocol_signature": protocol_signature,
                "atom_id": "",
                "activated": "",
                "raw_response": raw[:500],
                "input_snippet": " ".join(str(item.get("input", "")).split())[:240],
            }
        )
    n_items = len(items)
    summary = {
        "condition": condition,
        "task": task,
        "model": args.eval_model,
        "generator_model": args.generator_model if condition == "csicl" else "",
        "n_items": n_items,
        "correct": correct,
        "accuracy": correct / n_items if n_items else 0.0,
        "parse_errors": parse_errors,
        "failed_calls": failed_calls,
        "api_calls": n_items,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(
            _estimate_cost(input_tokens, output_tokens, args.price_in_per_1m, args.price_out_per_1m),
            6,
        ),
        "generation_input_tokens": int(meta.get("generation_input_tokens", 0)) if condition == "csicl" else 0,
        "generation_output_tokens": int(meta.get("generation_output_tokens", 0)) if condition == "csicl" else 0,
        "generation_estimated_cost_usd": float(meta.get("generation_estimated_cost_usd", 0.0))
        if condition == "csicl"
        else 0.0,
        "protocol_signature": protocol_signature,
        "prompt_builder": prompt_builder,
        "cheatsheet_hash": meta.get("cheatsheet_hash", _sha(cheatsheet)),
    }
    summary["total_estimated_cost_usd"] = round(
        summary["estimated_cost_usd"] + summary["generation_estimated_cost_usd"],
        6,
    )
    return summary, rows


def _load_raw_from_pilot(args: argparse.Namespace, task_filter: set[str]) -> tuple[list[dict], list[dict]]:
    p = Path(args.raw_pilot)
    if not p.exists():
        return [], []
    data = json.loads(p.read_text(encoding="utf-8"))
    summaries = []
    for row in data["tasks"]:
        if row["task"] in task_filter:
            summaries.append(
                {
                    "condition": "raw",
                    "task": row["task"],
                    "model": row.get("model", args.eval_model),
                    "generator_model": "",
                    "n_items": int(row["n_items"]),
                    "correct": int(row["baseline_correct"]),
                    "accuracy": float(row["baseline_accuracy"]),
                    "parse_errors": int(row.get("parse_error_count", 0)),
                    "failed_calls": int(row.get("failed_call_count", 0)),
                    "api_calls": int(row.get("api_calls", 0)),
                    "estimated_input_tokens": int(row.get("estimated_input_tokens", 0)),
                    "estimated_output_tokens": int(row.get("estimated_output_tokens", 0)),
                    "estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0)),
                    "generation_input_tokens": 0,
                    "generation_output_tokens": 0,
                    "generation_estimated_cost_usd": 0.0,
                    "total_estimated_cost_usd": float(row.get("estimated_cost_usd", 0.0)),
                    "protocol_signature": "raw_from_pilot",
                    "cheatsheet_hash": "",
                }
            )
    item_rows = []
    item_path = p.with_name("pilot_item_results.jsonl")
    if item_path.exists():
        with item_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("task") not in task_filter:
                    continue
                item_rows.append(
                    {
                        "condition": "raw",
                        "task": row.get("task"),
                        "item_id": row.get("item_id"),
                        "model": row.get("model"),
                        "answer": row.get("predicted"),
                        "gold": row.get("gold"),
                        "correct": row.get("correct"),
                        "parse_error": row.get("parse_error"),
                        "failed_call": row.get("failed_call"),
                        "prompt_hash": "",
                        "protocol_signature": "raw_from_pilot",
                        "atom_id": "",
                        "activated": "",
                        "raw_response": row.get("raw_response", ""),
                        "input_snippet": row.get("input_snippet", ""),
                    }
                )
    return summaries, item_rows


def _aggregate(summaries: list[dict]) -> list[dict]:
    out = []
    by_cond: dict[str, list[dict]] = {}
    for row in summaries:
        by_cond.setdefault(row["condition"], []).append(row)
    for cond, rows in sorted(by_cond.items()):
        n = sum(int(r["n_items"]) for r in rows)
        correct = sum(int(r["correct"]) for r in rows)
        out.append(
            {
                "condition": cond,
                "task": "__aggregate__",
                "model": rows[0].get("model", ""),
                "generator_model": rows[0].get("generator_model", ""),
                "n_items": n,
                "correct": correct,
                "accuracy": correct / n if n else 0.0,
                "parse_errors": sum(int(r["parse_errors"]) for r in rows),
                "failed_calls": sum(int(r["failed_calls"]) for r in rows),
                "api_calls": sum(int(r["api_calls"]) for r in rows),
                "estimated_input_tokens": sum(int(r["estimated_input_tokens"]) for r in rows),
                "estimated_output_tokens": sum(int(r["estimated_output_tokens"]) for r in rows),
                "estimated_cost_usd": round(sum(float(r["estimated_cost_usd"]) for r in rows), 6),
                "generation_input_tokens": sum(int(r.get("generation_input_tokens", 0)) for r in rows),
                "generation_output_tokens": sum(int(r.get("generation_output_tokens", 0)) for r in rows),
                "generation_estimated_cost_usd": round(
                    sum(float(r.get("generation_estimated_cost_usd", 0.0)) for r in rows),
                    6,
                ),
                "total_estimated_cost_usd": round(
                    sum(float(r.get("total_estimated_cost_usd", r["estimated_cost_usd"])) for r in rows),
                    6,
                ),
                "protocol_signature": "aggregate",
                "cheatsheet_hash": "",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--conditions", nargs="+", default=["raw", "csicl"], choices=["raw", "csicl"])
    p.add_argument("--eval-model", default="openai/gpt-4.1-mini")
    p.add_argument("--generator-model", default="openai/gpt-4.1-mini")
    p.add_argument("--limit-per-task", type=int, default=0)
    p.add_argument("--sample-mode", choices=["head", "random"], default="head")
    p.add_argument("--seed", type=int, default=20260518)
    p.add_argument("--csicl-seed", type=int, default=1000)
    p.add_argument("--train-n", type=int, default=150)
    p.add_argument("--generation-max-tokens", type=int, default=4000)
    p.add_argument("--eval-max-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--price-in-per-1m", type=float, default=0.40)
    p.add_argument("--price-out-per-1m", type=float, default=1.60)
    p.add_argument("--raw-pilot", default="runs/controlled_full_bbh_gpt41mini_full_20260518/pilot_results.json")
    p.add_argument("--regenerate-cheatsheets", action="store_true")
    p.add_argument("--output-dir", default="runs/emnlp_bbh_baseline_surface_gpt41mini_20260518")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    args.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key()
    task_filter = set(args.tasks)

    summaries: list[dict] = []
    item_rows: list[dict] = []
    if "raw" in args.conditions:
        raw_summaries, raw_items = _load_raw_from_pilot(args, task_filter)
        summaries.extend(raw_summaries)
        item_rows.extend(raw_items)

    if "csicl" in args.conditions:
        for task in args.tasks:
            cfg = _task_cfg(task)
            cs_text, meta = _load_or_generate_csicl(task, cfg, args, api_key)
            summary, rows = _score_condition(task, cfg, "csicl", cs_text, args, api_key, meta)
            summaries.append(summary)
            item_rows.extend(rows)

    aggregate_rows = _aggregate(summaries)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "aggregate": aggregate_rows,
        "tasks": summaries,
    }
    (out_dir / "surface_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "surface_summary.csv", aggregate_rows + summaries)
    _write_jsonl(out_dir / "per_item_outputs.jsonl", item_rows)
    _write_csv(
        out_dir / "parse_audit.csv",
        [
            {
                "condition": r["condition"],
                "task": r["task"],
                "parse_errors": r["parse_errors"],
                "failed_calls": r["failed_calls"],
                "protocol_signature": r["protocol_signature"],
            }
            for r in summaries
        ],
    )
    print(json.dumps({"output_dir": str(out_dir), "aggregate": aggregate_rows}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
