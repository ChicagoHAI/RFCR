"""ICR_sfcr/failure_regions.py — Compute failure regions under an anchor cheatsheet.

Failure regions are defined relative to the anchor CS-ICL cheatsheet C0 on a
given set of items:

  F_s        source model failures
  F_j        proxy model j failures
  K_j        proxy model j correct
  V_shared   F_s ∩ (union_j F_j)      — source fails AND at least one proxy fails
  V_private  F_s ∩ (intersect_j K_j)  — source fails BUT all proxies succeed
  V_easy     K_s ∩ (intersect_j K_j)  — source AND all proxies succeed

Soft-probability mode (n_evals > 1):
  Score each model n_evals times at eval_temperature and compute
  p_M(x) = fraction of evals where M answers incorrectly.
  Regions are then defined by thresholds:
    V_shared  = {x: p_s(x) >= tau_s  AND  max_j p_j(x) >= tau_p}
    V_private = {x: p_s(x) >= tau_s  AND  max_j p_j(x) <= tau_low}
    V_easy    = {x: p_s(x) <= tau_low AND  max_j p_j(x) <= tau_low}
  At n_evals=1 (default), p_M ∈ {0, 1} and tau_s=0.5 reproduces the original
  binary behaviour exactly.

Called twice in the pipeline:
  1. On rule_gen items  — to drive rule generation
  2. On gate items      — as baseline for candidate validation

Skip conditions (set skip_reason):
  |V_shared| < 3           no shared failures to refine
  source_acc > 0.95        ceiling — refinement unnecessary
  source_acc < 0.35        floor   — overlap signal unreliable
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from utils.scorer import score_batch
from utils.task_spec import TaskSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item_id(item: dict, idx: int) -> str:
    return str(item.get("id", f"_idx_{idx}"))


def _tag_ids(items: list[dict]) -> None:
    """Attach _sfcr_id to each item in-place (idempotent)."""
    for i, it in enumerate(items):
        it.setdefault("_sfcr_id", _item_id(it, i))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FailureRegions:
    # Annotated item lists (source-model annotation as primary)
    F_s: list[dict]
    V_shared: list[dict]
    V_private: list[dict]
    V_easy: list[dict]

    # Per-model id sets (for gate-split region membership lookup)
    per_model_correct: dict[str, set] = field(default_factory=dict)
    per_model_wrong: dict[str, set]   = field(default_factory=dict)

    # Per-model annotated items: model -> {sfcr_id: annotated_item}
    per_model_annotated: dict[str, dict] = field(default_factory=dict)

    # Soft-mode failure probabilities: model -> {sfcr_id: p_M(x)}
    # At n_evals=1 these are 0.0 or 1.0 (hard binary)
    p_by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    source_accuracy: float = 0.0
    jaccard_matrix: dict   = field(default_factory=dict)
    skip_reason: str | None = None


# ---------------------------------------------------------------------------
# Multi-eval scoring helper
# ---------------------------------------------------------------------------

def _score_model_multi(
    model: str,
    items: list[dict],
    anchor_cheatsheet: str,
    api_key: str,
    task_spec: TaskSpec,
    concurrency: int,
    label: str,
    n_evals: int,
    eval_temperature: float,
) -> tuple[dict[str, float], dict[str, dict]]:
    """
    Score `model` on `items` for `n_evals` rounds.

    Returns:
        p_M   : {sfcr_id: fraction_of_evals_where_model_is_wrong}
        ann   : {sfcr_id: annotated_item}  (from first eval that sees the item)
    """
    wrong_counts: dict[str, int] = defaultdict(int)
    ann: dict[str, dict] = {}
    short = model.split("/")[-1]
    temp = eval_temperature if n_evals > 1 else 0.0

    for eval_idx in range(n_evals):
        progress = (
            f"[{label}] {short} (eval {eval_idx + 1}/{n_evals})"
            if n_evals > 1 else
            f"[{label}] {short}"
        )
        c, w = score_batch(
            items,
            anchor_cheatsheet,
            model,
            api_key,
            concurrency=concurrency,
            temperature=temp,
            progress_label=progress,
            reasoning_effort=None,
            cot_first=True,
            task_spec=task_spec,
        )
        for it in c:
            iid = it["_sfcr_id"]
            ann.setdefault(iid, it)
        for it in w:
            iid = it["_sfcr_id"]
            wrong_counts[iid] += 1
            ann.setdefault(iid, it)

    all_ids = {it["_sfcr_id"] for it in items}
    p_M = {iid: wrong_counts[iid] / n_evals for iid in all_ids}
    return p_M, ann


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_failure_regions(
    items: list[dict],
    anchor_cheatsheet: str,
    source_model: str,
    proxy_models: list[str],
    api_key: str,
    task_spec: TaskSpec,
    concurrency: int = 30,
    label: str = "regions",
    n_evals: int = 1,
    eval_temperature: float = 0.7,
    tau_s: float = 0.5,
    tau_p: float = 0.5,
    tau_low: float = 0.33,
) -> FailureRegions:
    """
    Score source + all proxy models on `items` under `anchor_cheatsheet`
    in parallel, then compute V_shared / V_private / V_easy.

    `n_evals` > 1 enables soft-probability mode: each model is scored
    n_evals times and p_M(x) = fraction_of_evals_wrong is used with
    thresholds (tau_s, tau_p, tau_low) to define regions. At n_evals=1
    this degenerates to the original hard-binary behaviour.

    `label` is used in progress messages to distinguish rule_gen vs gate passes.
    """
    _tag_ids(items)
    all_models = [source_model] + proxy_models

    # ── Score all models concurrently (each model scored n_evals times) ───
    p_by_model: dict[str, dict[str, float]] = {}
    per_model_annotated: dict[str, dict[str, dict]] = {}

    def _score_one(model: str):
        return model, *_score_model_multi(
            model, items, anchor_cheatsheet, api_key, task_spec,
            concurrency, label, n_evals, eval_temperature,
        )

    with ThreadPoolExecutor(max_workers=len(all_models)) as pool:
        for model, p_M, ann in pool.map(_score_one, all_models):
            p_by_model[model] = p_M
            per_model_annotated[model] = ann

    # ── Derive binary correct/wrong sets from probabilities ───────────────
    all_ids = {it["_sfcr_id"] for it in items}

    per_model_correct: dict[str, set] = {}
    per_model_wrong: dict[str, set]   = {}
    for model in all_models:
        p_M = p_by_model[model]
        per_model_wrong[model]   = {iid for iid in all_ids if p_M.get(iid, 0.0) >= tau_s}
        per_model_correct[model] = all_ids - per_model_wrong[model]

    # ── Compute region id sets ─────────────────────────────────────────────
    p_s = p_by_model[source_model]
    F_s_ids = per_model_wrong[source_model]   # p_s(x) >= tau_s
    K_s_ids = per_model_correct[source_model]  # p_s(x) < tau_s
    source_accuracy = len(K_s_ids) / len(items) if items else 0.0

    # V_shared: source fails AND at least one proxy fails
    proxy_fail_union: set = set()
    proxy_correct_inter: set | None = None
    for pm in proxy_models:
        p_pm = p_by_model[pm]
        # "proxy fails" = p_j(x) >= tau_p
        pm_fail = {iid for iid in all_ids if p_pm.get(iid, 0.0) >= tau_p}
        # "proxy consistently correct" = p_j(x) <= tau_low
        pm_correct_strict = {iid for iid in all_ids if p_pm.get(iid, 0.0) <= tau_low}
        proxy_fail_union |= pm_fail
        if proxy_correct_inter is None:
            proxy_correct_inter = pm_correct_strict.copy()
        else:
            proxy_correct_inter &= pm_correct_strict
    if proxy_correct_inter is None:
        proxy_correct_inter = set()

    V_shared_ids  = F_s_ids & proxy_fail_union
    # V_private: source fails AND all proxies are consistently correct
    V_private_ids = F_s_ids & proxy_correct_inter
    # V_easy: source consistently correct AND all proxies consistently correct
    source_correct_strict = {iid for iid in all_ids if p_s.get(iid, 0.0) <= tau_low}
    V_easy_ids = source_correct_strict & proxy_correct_inter

    # ── Recover annotated items ────────────────────────────────────────────
    src_ann = per_model_annotated[source_model]

    def _recover(id_set: set) -> list[dict]:
        return [src_ann[iid] for iid in id_set if iid in src_ann]

    F_s       = _recover(F_s_ids)
    V_shared  = _recover(V_shared_ids)
    V_private = _recover(V_private_ids)
    V_easy    = _recover(V_easy_ids)

    # ── Jaccard matrix ─────────────────────────────────────────────────────
    jaccard_matrix = {}
    for pm in proxy_models:
        pm_fail = per_model_wrong[pm]
        union   = F_s_ids | pm_fail
        inter   = F_s_ids & pm_fail
        jac = len(inter) / len(union) if union else 0.0
        src_short = source_model.split("/")[-1]
        pm_short  = pm.split("/")[-1]
        jaccard_matrix[(src_short, pm_short)] = jac

    # ── Skip guard ────────────────────────────────────────────────────────
    skip_reason = None
    if len(V_shared) < 3:
        skip_reason = f"|V_shared|={len(V_shared)} < 3 — no shared failures to refine"
    elif source_accuracy > 0.95:
        skip_reason = f"source accuracy {source_accuracy:.1%} at ceiling — SFCR not applicable"
    elif source_accuracy < 0.35:
        skip_reason = f"source accuracy {source_accuracy:.1%} at floor — overlap signal unreliable"

    # ── Summary ────────────────────────────────────────────────────────────
    mode_tag = f"n_evals={n_evals}" if n_evals > 1 else "hard-binary"
    print(
        f"[{label}] {mode_tag}  source_acc={source_accuracy:.1%}  "
        f"|F_s|={len(F_s)}  |V_shared|={len(V_shared)}  "
        f"|V_private|={len(V_private)}  |V_easy|={len(V_easy)}"
    )
    for (s, p), jac in jaccard_matrix.items():
        print(f"[{label}] Jaccard({s}↔{p}) = {jac:.3f}")
    if skip_reason:
        print(f"[{label}] SKIP: {skip_reason}")

    return FailureRegions(
        F_s=F_s,
        V_shared=V_shared,
        V_private=V_private,
        V_easy=V_easy,
        per_model_correct=per_model_correct,
        per_model_wrong=per_model_wrong,
        per_model_annotated=per_model_annotated,
        p_by_model=p_by_model,
        source_accuracy=source_accuracy,
        jaccard_matrix=jaccard_matrix,
        skip_reason=skip_reason,
    )
