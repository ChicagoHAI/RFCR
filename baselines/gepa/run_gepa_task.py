"""GEPA baseline optimization for one BBH main task (ARR rebuttal).

Usage: .venv/bin/python run_gepa_task.py <task>   (run from rebuttal_experiments/gepa/,
                                                    inside the FRESH gepa venv)

Protocol (mirrors rebuttal_experiments/promptwizard/report.md discipline):
- SEED = the locked CS-ICL cheatsheet
  (ICRefine/runs/emnlp_bbh_full18_csicl_gpt41mini_20260518/cheatsheets/{task}/
   csicl_gpt-4.1-mini_seed1000.txt), embedded in the exact prompt shape the
  locked evaluator uses (tasks/utils._make_eval_prompt for dq/gs/ff,
  GenericBBHTask.build_eval_prompt for object_counting) minus the per-item
  input slot. GEPA's DefaultAdapter sends the candidate as the system message
  and the item input as the user message.
- Train data: ICRefine/datasets/bbh/{task}_train.jsonl ONLY (150 items).
  Following the gepa README Quick Start pattern (gepa.examples.aime), the train
  file is shuffled with random.Random(0) and split half/half into
  trainset (75) / valset (75). Test files are never read here.
- GEPA: standalone `gepa` pip package, gepa.optimize() with default settings.
  Deviations from pure defaults (all documented in gepa_config_used.json):
    * adapter = gepa's own DefaultAdapter, instantiated explicitly so we can
      pass a custom Evaluator (exact-match via the repo's task parsers, per
      protocol) and max_litellm_workers=16 (library default 10; raised per
      user instruction "high parallelism where the library exposes it").
    * max_metric_calls = 680, derived from the documented DSPy GEPA
      'light' auto-budget formula (dspy.teleprompt.gepa.GEPA.auto_budget with
      num_preds=1, num_candidates=6 ['light' => n=6], valset_size=75,
      minibatch_size=35, full_eval_steps=5):
        N = int(max(2*(1*2)*log2(6), 1.5*6)) = 10
        total = 75 (seed full eval) + 6*5 (30) + 10*35 (350)
                + ((10+1)//5 + 1) * 75 (225) = 680
      (The standalone library has no default for max_metric_calls — the
      parameter is required; its README Quick Start uses 150, which with a
      75-item valset would allow at most one candidate to be val-evaluated.)
    * task model AND reflection model: openai/gpt-4.1-mini (user instruction).
  Everything else is the library default: candidate_selection_strategy=
  'pareto', frontier_type='instance', skip_perfect_score=True,
  batch_sampler='epoch_shuffled', reflection_minibatch_size=3 (default),
  module_selector='round_robin', use_merge=False, perfect_score=1.0,
  seed=0 (library default), raise_on_exception=True, cache_evaluation=False,
  val_evaluation_policy=None (=> full_eval). Task-LM calls go through
  litellm.batch_completion with DefaultAdapter's defaults: NO temperature,
  seed, or max_tokens parameters are sent (OpenAI server defaults apply,
  i.e. temperature 1.0) — this is the library's default behavior, recorded
  as-is. Reflection calls likewise send no sampling params.
- Metric: exact-match correctness via the repo's own task parsers
  (scripts.run_controlled_full_benchmark_pilot._parse_prediction +
   task_spec.is_correct), score 1.0/0.0. No scoring reimplemented.
- Every LLM request/response (task + reflection) is logged to
  {task}/llm_calls.jsonl by thin wrappers around litellm.batch_completion /
  litellm.completion (logging only — the GEPA algorithm is untouched), for the
  zero-leak audit and cost accounting. Budget guard aborts if estimated spend
  for this process exceeds $30.
"""

from __future__ import annotations

import json
import math
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ICR = REPO / "ICRefine"
sys.path.insert(0, str(ICR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ICR / ".env")

from scripts.run_controlled_full_benchmark_pilot import _parse_prediction, _task_spec_from_cfg  # noqa: E402
from scripts.run_sfcr_emnlp_baseline_surface import _task_cfg  # noqa: E402
from utils.data import load_jsonl  # noqa: E402

import litellm  # noqa: E402
import gepa  # noqa: E402
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter, EvaluationResult  # noqa: E402

TASK_LM = "openai/gpt-4.1-mini"
REFLECTION_LM = "openai/gpt-4.1-mini"
MAX_METRIC_CALLS = 680
MAX_LITELLM_WORKERS = 16
GEPA_SEED = 0            # gepa.optimize default
SPLIT_SEED = 0           # random.Random(0), mirroring gepa.examples.aime.init_dataset
BUDGET_ABORT_USD = 30.0
PRICE_IN, PRICE_OUT = 0.40, 1.60  # gpt-4.1-mini USD per 1M tokens

CSICL_ROOT = ICR / "runs" / "emnlp_bbh_full18_csicl_gpt41mini_20260518" / "cheatsheets"

# Exact prompt shapes of the locked evaluator (tasks/utils._make_eval_prompt and
# GenericBBHTask.build_eval_prompt), with the "{input}\n\n" slot removed — the
# item input becomes the user message (DefaultAdapter contract).
VERDICT_FORMATS = {
    "disambiguation_qa": "(A), (B), or (C)",
    "geometric_shapes": "(A) through (J)",
    "formal_fallacies": "valid or invalid",
    "object_counting": "an integer",
}


def build_seed_prompt(task: str, cheatsheet: str) -> str:
    fmt = VERDICT_FORMATS[task]
    if task == "object_counting":  # GenericBBHTask.build_eval_prompt shape
        tail = f"Reply with ONLY the verdict. Do not explain.\nVERDICT: {fmt}"
    else:  # tasks/utils._make_eval_prompt shape
        tail = f"Reply with ONLY the verdict — no explanation, no reasoning.\nVERDICT: {fmt}"
    return f"=== CHEATSHEET ===\n{cheatsheet}\n=== END CHEATSHEET ===\n\n{tail}"


class ExactMatchRepoParserEvaluator:
    """Score 1.0 iff the repo task spec judges the parsed prediction correct."""

    def __init__(self, task: str, spec):
        self.task = task
        self.spec = spec
        self.fmt = VERDICT_FORMATS[task]

    def __call__(self, data: dict, response: str) -> EvaluationResult:
        item = data["_item"]
        pred = _parse_prediction(self.spec, response or "")
        correct = pred is not None and bool(self.spec.is_correct(pred, item))
        gold = data["answer"]
        if correct:
            feedback = (
                f"The generated response is correct. The parsed verdict matches the gold answer '{gold}'."
            )
        else:
            verdict = self.spec.parse_verdict(response or "")
            shown = verdict if verdict is not None else "NONE (no parseable 'VERDICT:' line)"
            feedback = (
                f"The generated response is incorrect. Parsed verdict: {shown}. "
                f"The correct answer is '{gold}'. The response must contain a final line "
                f"of the form 'VERDICT: <answer>' where <answer> is {self.fmt}."
            )
        return EvaluationResult(score=1.0 if correct else 0.0, feedback=feedback, objective_scores=None)


class CallLogger:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.n_calls = 0
        self.tok_in = 0
        self.tok_out = 0
        self.fh = path.open("a", encoding="utf-8")

    def log(self, kind: str, model: str, messages, response_content: str, usage, elapsed_ms: float):
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        with self.lock:
            self.n_calls += 1
            self.tok_in += pt
            self.tok_out += ct
            self.fh.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": kind,
                "model": model,
                "messages": messages,
                "response": response_content,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "elapsed_ms": round(elapsed_ms, 1),
            }, ensure_ascii=False) + "\n")
            self.fh.flush()
            cost = (self.tok_in * PRICE_IN + self.tok_out * PRICE_OUT) / 1e6
            if cost > BUDGET_ABORT_USD:
                raise SystemExit(f"BUDGET GUARD: estimated spend ${cost:.2f} > ${BUDGET_ABORT_USD}")

    @property
    def est_cost(self) -> float:
        return (self.tok_in * PRICE_IN + self.tok_out * PRICE_OUT) / 1e6


def install_logging_wrappers(logger: CallLogger):
    """Logging-only wrapper; GEPA algorithm and litellm behavior untouched.

    Only litellm.completion is wrapped: litellm.batch_completion (used by
    DefaultAdapter for task calls) internally dispatches every sub-request
    through the litellm.completion package attribute, so wrapping completion
    alone captures each API call exactly once (wrapping both double-logged
    task calls in the first dq run; that log was deduplicated post-hoc).
    Task calls are distinguishable by their [system, user] message structure
    (the candidate prompt is the system message); reflection calls are a
    single user message.
    """
    orig_completion = litellm.completion

    def logged_completion(*args, **kwargs):
        t0 = time.monotonic()
        resp = orig_completion(*args, **kwargs)
        elapsed = (time.monotonic() - t0) * 1000
        msgs = kwargs.get("messages", [])
        model = kwargs.get("model") or (args[0] if args else "?")
        kind = "task" if any(m.get("role") == "system" for m in msgs) else "reflection"
        content = resp.choices[0].message.content
        logger.log(kind, str(model), msgs, content or "", getattr(resp, "usage", None), elapsed)
        return resp

    litellm.completion = logged_completion


def main() -> None:
    task = sys.argv[1]
    assert task in VERDICT_FORMATS, task
    out_dir = HERE / task
    out_dir.mkdir(parents=True, exist_ok=True)

    # transport robustness only (PW precedent: retry/backoff wrapper accepted)
    litellm.num_retries = 5

    cheatsheet = (CSICL_ROOT / task / "csicl_gpt-4.1-mini_seed1000.txt").read_text(encoding="utf-8")
    seed_prompt = build_seed_prompt(task, cheatsheet)
    (out_dir / "seed_prompt.txt").write_text(seed_prompt, encoding="utf-8")

    train_items = load_jsonl(ICR / "datasets" / "bbh" / f"{task}_train.jsonl")
    assert len(train_items) == 150, f"expected 150 train items, got {len(train_items)}"
    spec = _task_spec_from_cfg(_task_cfg(task))

    def to_datainst(item: dict) -> dict:
        return {
            "input": item["input"],
            "additional_context": {},
            "answer": str(item["answer"]).strip(),
            "_item": item,
            "_id": item.get("id", ""),
        }

    shuffled = list(train_items)
    random.Random(SPLIT_SEED).shuffle(shuffled)  # mirrors gepa.examples.aime.init_dataset
    half = len(shuffled) // 2
    trainset = [to_datainst(it) for it in shuffled[:half]]
    valset = [to_datainst(it) for it in shuffled[half:]]

    logger = CallLogger(out_dir / "llm_calls.jsonl")
    install_logging_wrappers(logger)

    evaluator = ExactMatchRepoParserEvaluator(task, spec)
    adapter = DefaultAdapter(model=TASK_LM, evaluator=evaluator, max_litellm_workers=MAX_LITELLM_WORKERS)

    import importlib.metadata as md
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "gepa_version": md.version("gepa"),
        "litellm_version": md.version("litellm"),
        "python": sys.version.split()[0],
        "task_lm": TASK_LM,
        "reflection_lm": REFLECTION_LM,
        "adapter": "gepa.adapters.default_adapter.DefaultAdapter",
        "evaluator": "ExactMatchRepoParserEvaluator (repo task parsers: _parse_prediction + spec.is_correct; 1.0/0.0)",
        "max_litellm_workers": MAX_LITELLM_WORKERS,
        "max_litellm_workers_note": "library default is 10; raised to 16 per user parallelism instruction",
        "litellm_batch_completion_kwargs": {},
        "task_call_sampling": "library default: no temperature/seed/max_tokens sent (OpenAI defaults, temperature 1.0)",
        "litellm_num_retries": 5,
        "max_metric_calls": MAX_METRIC_CALLS,
        "max_metric_calls_derivation": (
            "DSPy GEPA 'light' auto-budget formula (auto_budget, num_preds=1, num_candidates=6, "
            "valset_size=75, minibatch_size=35, full_eval_steps=5): N=int(max(2*2*log2(6), 9))=10; "
            "total = 75 + 30 + 350 + ((10+1)//5 + 1)*75 = 680. Standalone gepa requires an explicit "
            "max_metric_calls (no default); its README Quick Start value of 150 would permit at most "
            "one full val eval beyond the seed with a 75-item valset."
        ),
        "seed_candidate_components": ["system_prompt"],
        "seed_prompt_source": str((CSICL_ROOT / task / "csicl_gpt-4.1-mini_seed1000.txt").relative_to(REPO)),
        "seed_prompt_shape": (
            "locked evaluator prompt template (tasks/utils._make_eval_prompt; GenericBBHTask for "
            "object_counting) with the per-item input slot removed; item input sent as the user "
            "message per DefaultAdapter contract"
        ),
        "trainset_size": len(trainset),
        "valset_size": len(valset),
        "split": "random.Random(0).shuffle over the 150-item train file, first half train / second half val (gepa README Quick Start pattern)",
        "train_file": f"ICRefine/datasets/bbh/{task}_train.jsonl",
        "gepa_seed": GEPA_SEED,
        "optimize_defaults_used": {
            "candidate_selection_strategy": "pareto",
            "frontier_type": "instance",
            "skip_perfect_score": True,
            "batch_sampler": "epoch_shuffled",
            "reflection_minibatch_size": "None -> 3 (library default)",
            "perfect_score": 1.0,
            "module_selector": "round_robin",
            "use_merge": False,
            "max_merge_invocations": 5,
            "raise_on_exception": True,
            "cache_evaluation": False,
            "val_evaluation_policy": "None -> full_eval",
            "reflection_prompt_template": "None (library default InstructionProposalSignature)",
        },
    }
    (out_dir / "gepa_config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    t0 = time.time()
    result = gepa.optimize(
        seed_candidate={"system_prompt": seed_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        seed=GEPA_SEED,
        run_dir=str(out_dir / "gepa_run"),
        display_progress_bar=False,
    )
    wall = time.time() - t0

    best = result.best_candidate
    best_text = best["system_prompt"] if isinstance(best, dict) else best
    (out_dir / "optimized_prompt.txt").write_text(best_text, encoding="utf-8")

    import hashlib
    summary = {
        "task": task,
        "wall_seconds": round(wall, 1),
        "num_candidates": result.num_candidates,
        "best_idx": result.best_idx,
        "val_aggregate_scores": result.val_aggregate_scores,
        "seed_val_score": result.val_aggregate_scores[0],
        "best_val_score": result.val_aggregate_scores[result.best_idx],
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "parents": result.parents,
        "n_llm_calls_logged": logger.n_calls,
        "prompt_tokens": logger.tok_in,
        "completion_tokens": logger.tok_out,
        "est_cost_usd": round(logger.est_cost, 4),
        "seed_prompt_sha256": hashlib.sha256(seed_prompt.encode()).hexdigest(),
        "optimized_prompt_sha256": hashlib.sha256(best_text.encode()).hexdigest(),
        "best_is_seed": best_text == seed_prompt,
    }
    (out_dir / "gepa_opt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
