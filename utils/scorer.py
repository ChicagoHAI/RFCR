"""
scorer.py — Unified scorer for all ICRefine pipelines.

Each scored item carries:
  predicted    : "TRUE" | "FALSE" | None
  expected     : "TRUE" | "FALSE"
  post_think   : REASONING section extracted from the model's structured output
  thinking     : full internal CoT trace (empty for non-reasoning models)
  raw_response : the full content string

Per Heddaya et al. (ACL 2026), post_think preserves deductive markers at
25× higher density than externally prompted summaries — it is the right
signal for identifying what went wrong in a failure.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .data import is_true
from .llm_client import LLMResponse, call_llm, call_llm_batch, is_reasoning_model
from .run_logger import get_logger
from .task_spec import TaskSpec

SCORING_MAX_TOKENS = int(os.environ.get("ICR_SCORING_MAX_TOKENS", "8192"))
# Default matches the competition setting; local vLLM runs can lower this in
# .env to keep prompt + completion within the served model's max_model_len.

# Appended to scoring prompts for non-reasoning models to elicit a genuine
# step-by-step justification rather than a one-line label.  The text goes
# after the format spec so it overrides any brevity bias without changing
# the expected output structure.
_JUSTIFICATION_SUFFIX = (
    "\n\nIMPORTANT: In your REASONING, explain your thinking step by step — "
    "what you checked first, how you ruled out alternatives, and exactly "
    "where your reasoning led you to your prediction. "
    "Do not just restate the verdict; show the specific steps you took."
)


def _use_rf_scoring() -> bool:
    return os.environ.get("ICR_USE_RF_SCORING", "").strip().lower() in {"1", "true", "yes", "on"}


def _use_partition_routed_scoring() -> bool:
    return os.environ.get("ICR_USE_PARTITION_ROUTED_SCORING", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _resolve_cheatsheet_texts(
    ts: TaskSpec,
    cheatsheet_text: str | object,
    items: list[dict],
) -> list[str]:
    if hasattr(cheatsheet_text, "render_for_partition_key") and _use_partition_routed_scoring():
        texts: list[str] = []
        for item in items:
            try:
                texts.append(cheatsheet_text.render_for_partition_key(ts.partition_key(item)))
            except Exception:
                texts.append(cheatsheet_text.render())
        return texts
    if hasattr(cheatsheet_text, "render") and not isinstance(cheatsheet_text, str):
        text = cheatsheet_text.render()
    else:
        text = str(cheatsheet_text)
    return [text] * len(items)


def _build_scoring_prompts(
    ts: TaskSpec,
    cheatsheet_text: str | list[str],
    items: list[dict],
    model: str,
    cot_first: bool,
) -> list[str]:
    cheatsheet_texts = (
        cheatsheet_text
        if isinstance(cheatsheet_text, list)
        else [cheatsheet_text] * len(items)
    )
    rf_builder = getattr(ts, "build_scoring_prompt_rf", None)
    if _use_rf_scoring() and rf_builder is not None:
        prompts = []
        for cs_text, item in zip(cheatsheet_texts, items):
            try:
                prompts.append(rf_builder(cs_text, item, cot_first))
            except TypeError:
                prompts.append(rf_builder(cs_text, item))
        return prompts

    _use_cot = cot_first or not is_reasoning_model(model)
    _suffix = _JUSTIFICATION_SUFFIX if not is_reasoning_model(model) else ""
    return [
        ts.build_scoring_prompt(cs_text, item, _use_cot) + _suffix
        for cs_text, item in zip(cheatsheet_texts, items)
    ]


# ---------------------------------------------------------------------------
# Default task spec (lazy import to avoid circular dependency)
# ---------------------------------------------------------------------------

def _default_task() -> TaskSpec:
    from tasks.magma import MAGMA_TASK
    return MAGMA_TASK


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    accuracy: float
    correct: list[dict] = field(default_factory=list)
    wrong:   list[dict] = field(default_factory=list)
    errors:  list[dict] = field(default_factory=list)
    n_total: int = 0

    def summary(self) -> str:
        return (
            f"accuracy={self.accuracy:.1%}  "
            f"correct={len(self.correct)}  "
            f"wrong={len(self.wrong)}  "
            f"parse_errors={len(self.errors)}  "
            f"total={self.n_total}"
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_batch(
    items: list[dict],
    cheatsheet_text: str,
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    progress_label: str = "scoring",
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
    task_spec: TaskSpec | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Score items against the current cheatsheet in parallel.

    Returns (correct_items, wrong_items) — both annotated with predicted,
    expected, post_think, thinking, and raw_response.
    Parse errors are counted as wrong.

    task_spec: domain-specific logic (defaults to MAGMA_TASK for backward compat).
    cot_first: use the COT-first scoring prompt variant so the model writes its
               reasoning before stating the verdict.
    """
    ts = task_spec or _default_task()
    # Non-reasoning models produce no internal thinking trace; elicit a genuine
    # step-by-step justification via prompt so post_think carries useful signal
    # for the case study generator.
    prompts = _build_scoring_prompts(
        ts, _resolve_cheatsheet_texts(ts, cheatsheet_text, items), items, model, cot_first
    )
    responses = call_llm_batch(
        prompts,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=SCORING_MAX_TOKENS,
        concurrency=concurrency,
        progress_label=progress_label,
        reasoning_effort=reasoning_effort,
    )

    correct, wrong = [], []
    n_parse_errors = 0

    for item, resp in zip(items, responses):
        if resp is None:
            annotated = {
                **item,
                "predicted":    None,
                "expected":     ts.answer_label(item),
                "post_think":   "",
                "thinking":     "",
                "raw_response": "",
            }
            wrong.append(annotated)
            n_parse_errors += 1
            continue

        predicted  = ts.parse_verdict(resp.content)
        post_think = ts.extract_post_think(resp.content)

        annotated = {
            **item,
            "predicted":    predicted,
            "expected":     ts.answer_label(item),
            "post_think":   post_think,
            "thinking":     resp.thinking,
            "raw_response": resp.content,
        }

        if not ts.is_correct(predicted, item):
            if predicted is None:
                n_parse_errors += 1
            wrong.append(annotated)
        else:
            correct.append(annotated)

    if n_parse_errors:
        print(
            f"\n  [scorer] {n_parse_errors} parse errors (no recognisable answer) — "
            f"counted as wrong.",
            file=sys.stderr,
        )
        shown = 0
        for it in wrong:
            if it.get("predicted") is None and shown < 3:
                raw = it.get("raw_response", "")
                print(
                    f"\n  [parse-debug] raw_response (first 300 chars):\n"
                    f"  {repr(raw[:300])}",
                    file=sys.stderr,
                )
                shown += 1

    _log = get_logger()
    if _log is not None:
        n_total   = len(correct) + len(wrong)
        n_correct = len(correct)
        _log.log_scoring(
            label=progress_label,
            model=model,
            n_correct=n_correct,
            n_total=n_total,
            accuracy=n_correct / n_total if n_total else 0.0,
        )

    return correct, wrong


def _score_batch_ordered(
    items: list[dict],
    cheatsheet_text: str,
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
    progress_label: str = "scoring",
    task_spec: TaskSpec | None = None,
) -> list[tuple[bool | None, dict]]:
    """
    Like score_batch but returns results in the original item order as
    list[(is_correct, annotated_item)].  is_correct is None on parse error.
    Internal helper — used by score_batch_ensemble.
    """
    ts = task_spec or _default_task()
    prompts = _build_scoring_prompts(
        ts, _resolve_cheatsheet_texts(ts, cheatsheet_text, items), items, model, cot_first
    )
    responses = call_llm_batch(
        prompts,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=SCORING_MAX_TOKENS,
        concurrency=concurrency,
        progress_label=progress_label,
        reasoning_effort=reasoning_effort,
    )
    results = []
    for item, resp in zip(items, responses):
        if resp is None:
            annotated = {
                **item,
                "predicted":    None,
                "expected":     ts.answer_label(item),
                "post_think":   "",
                "thinking":     "",
                "raw_response": "",
            }
            results.append((None, annotated))
            continue
        predicted  = ts.parse_verdict(resp.content)
        post_think = ts.extract_post_think(resp.content)
        annotated  = {
            **item,
            "predicted":    predicted,
            "expected":     ts.answer_label(item),
            "post_think":   post_think,
            "thinking":     resp.thinking,
            "raw_response": resp.content,
        }
        correct = ts.is_correct(predicted, item) if predicted is not None else None
        results.append((correct, annotated))
    return results


def score_batch_ensemble(
    items: list[dict],
    cheatsheet_text: str,
    models: list[str],
    weights: list[float],
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
    task_spec: TaskSpec | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Score items with multiple models in parallel and return weighted (correct, wrong).

    An item is "correct" only if ALL models agree it is correct.
    An item is "wrong" if ANY model fails it; the item carries a ``_wrong_weight``
    field ∈ (0, 1] — the normalised sum of weights of models that failed it.

    This propagates into weighted fix_rate inside _mini_eval_full:
      • weight=1.0 — both models wrong  (consensus failure, highest priority)
      • weight=0.5 — one model wrong    (single-model failure, lower priority)

    Post-think traces from all failing models are concatenated with a divider
    so the case-study generator sees richer failure reasoning.
    Structured fields (predicted, expected) come from models[0] (primary).

    weights: relative contribution of each model (normalised internally to sum=1).
    """
    from concurrent.futures import ThreadPoolExecutor as _TPE

    assert len(models) == len(weights) >= 1, "models and weights must be same length ≥ 1"
    total_w      = sum(weights)
    norm_weights = [w / total_w for w in weights]

    # Run all models in parallel — each spawns its own inner thread pool over items.
    with _TPE(max_workers=len(models)) as pool:
        futures = [
            pool.submit(
                _score_batch_ordered,
                items, cheatsheet_text, m, api_key,
                concurrency, temperature, reasoning_effort, cot_first,
                f"scoring[{m.split('/')[-1]}]",
                task_spec,
            )
            for m in models
        ]
        all_ordered: list[list[tuple]] = [f.result() for f in futures]

    correct: list[dict] = []
    wrong:   list[dict] = []
    n_parse_errors = 0

    for i in range(len(items)):
        wrong_weight    = 0.0
        reasoning_parts: list[str] = []
        primary_ann: dict | None   = None

        for j, (model_results, nw) in enumerate(zip(all_ordered, norm_weights)):
            is_correct, ann = model_results[i]
            if j == 0:
                primary_ann = ann
            if is_correct is None:
                wrong_weight   += nw
                n_parse_errors += 1
            elif not is_correct:
                wrong_weight += nw
            think = ann.get("post_think", "")
            if think:
                label = models[j].split("/")[-1]
                reasoning_parts.append(f"[{label}]\n{think}")

        merged = {
            **primary_ann,
            "_wrong_weight": round(wrong_weight, 4),
            "post_think":    "\n\n---\n\n".join(reasoning_parts),
        }

        if wrong_weight == 0.0:
            correct.append(merged)
        else:
            wrong.append(merged)

    if n_parse_errors:
        print(
            f"\n  [ensemble] {n_parse_errors} parse errors across {len(models)} models — "
            f"counted as wrong.",
            file=sys.stderr,
        )

    return correct, wrong


def score_items_streaming(
    items: list[dict],
    get_cheatsheet: Callable[[], str],
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
    max_tokens: int = SCORING_MAX_TOKENS,
    seed: int | None = 42,
    task_spec: TaskSpec | None = None,
) -> Iterator[dict]:
    """
    Sliding-window scorer: yields one annotated item dict as each request
    completes. Always keeps `concurrency` requests in-flight so vLLM never
    idles between batches or during case-study generation.

    get_cheatsheet() is called immediately before each new submission, so
    any cheatsheet update made during a yield (e.g. adding a case study)
    is automatically picked up for the next queued request.

    Yielded dict keys: predicted, expected, post_think, thinking,
    raw_response — plus all original item fields.
    """
    ts = task_spec or _default_task()
    items_iter = iter(items)
    pending: dict[Future, dict] = {}

    def _submit_next(pool: ThreadPoolExecutor) -> bool:
        try:
            item = next(items_iter)
        except StopIteration:
            return False
        current_cheatsheet = get_cheatsheet()
        prompt = _build_scoring_prompts(
            ts,
            _resolve_cheatsheet_texts(ts, current_cheatsheet, [item]),
            [item],
            model,
            cot_first,
        )[0]
        f = pool.submit(
            call_llm, prompt, model, api_key, temperature, max_tokens, reasoning_effort, seed
        )
        pending[f] = item
        return True

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            if not _submit_next(pool):
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                item = pending.pop(f)
                try:
                    resp = f.result()
                    predicted  = ts.parse_verdict(resp.content)
                    post_think = ts.extract_post_think(resp.content)
                    thinking   = resp.thinking
                    raw        = resp.content
                except Exception:
                    predicted = post_think = thinking = raw = ""
                    predicted = None

                yield {
                    **item,
                    "predicted":    predicted,
                    "expected":     ts.answer_label(item),
                    "post_think":   post_think,
                    "thinking":     thinking,
                    "raw_response": raw,
                }

                # Submit next AFTER yield so get_cheatsheet() sees any update
                # the caller made while processing the yielded item.
                _submit_next(pool)


def test_cheatsheet(
    cheatsheet_text: str,
    val_items: list[dict],
    model: str,
    api_key: str,
    concurrency: int = 10,
    temperature: float = 0.0,
    reasoning_effort: str | None = "low",
    cot_first: bool = False,
    task_spec: TaskSpec | None = None,
) -> TestResult:
    """Score cheatsheet_text on the full val_items set. Returns a TestResult."""
    print(f"  Testing on {len(val_items)} items with {model} ...", file=sys.stderr)
    correct, wrong = score_batch(
        val_items, cheatsheet_text, model, api_key,
        concurrency, temperature,
        reasoning_effort=reasoning_effort, cot_first=cot_first,
        task_spec=task_spec,
    )
    scored   = len(correct) + len(wrong)
    accuracy = len(correct) / scored if scored > 0 else 0.0
    return TestResult(accuracy=accuracy, correct=correct, wrong=wrong, n_total=len(val_items))
