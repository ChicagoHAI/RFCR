"""
ICR_sfcr/pipeline.py — Shared-Failure Conservative Refinement (SFCR) pipeline.

Algorithm:
  1. Split training data into rule_gen / gate (disjoint).
  2. Score source + proxy models on rule_gen under anchor cheatsheet C0.
     Optionally use n_eval_seeds > 1 for soft probability estimation.
  3. Compute failure regions: V_shared, V_private, V_easy.
  4. Skip guard — exit with anchor unchanged if conditions are unmet.
  5. Cluster V_shared into subtypes; generate 2-3 candidates per subtype.
  6. Pre-compute gate baseline (score all models on gate under anchor).
  7. Validate each candidate via U_LCB / count-aware gate on the gate split.
     Rejected candidates enter the repair loop (--repair-attempts controls depth).
  8. Accept up to max_accepted rules (default 3).
  9. Write outputs: accepted_rules.json, final_cheatsheet_{mode}.txt, sfcr_log.jsonl.

Usage:
    python -m ICR_sfcr.pipeline \\
        --task              causal_judgement \\
        --dataset           datasets/bbh/causal_judgement_train_labeled.jsonl \\
        --anchor-cheatsheet CS_ICL_Initial_Prompt/bbh_causal_judgement/gen_gpt-4.1-mini_0_1000.txt \\
        --output-dir        runs/sfcr_cj_1000 \\
        --model-source      openai/gpt-4.1-mini \\
        --models-proxy      openai/gpt-4.1,google/gemini-2.0-flash,meta-llama/llama-3.3-70b \\
        --held-out-target   claude \\
        --oracle-mode       label_only \\
        --routing-mode      routed \\
        --seed              1000 \\
        --concurrency       30

    # Soft probability regions (3 repeated evals per model):
    python -m ICR_sfcr.pipeline ... --n-eval-seeds 3

    # Enable repair loop (1 repair attempt per rejected candidate):
    python -m ICR_sfcr.pipeline ... --repair-attempts 1

    # Disable subtype clustering (flat-pool generation, original behaviour):
    python -m ICR_sfcr.pipeline ... --no-subtypes

    # With full oracle CoT (for ablation):
    python -m ICR_sfcr.pipeline ... --oracle-mode full_cot

    # Global prepend mode (no routing):
    python -m ICR_sfcr.pipeline ... --routing-mode global
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data import load_jsonl
from utils.llm_client import get_api_key
from tasks.registry import get_task, TASK_REGISTRY

from .activation import activation_summary, build_cheatsheet
from .failure_regions import compute_failure_regions
from .logger import SFCRLogger, log_condition, rule_id
from .rule_generator import generate_candidates, repair_candidate
from .rule_validator import GateBaseline, compute_gate_baseline, validate_candidates
from .splits import make_splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Shared-Failure Conservative Refinement (SFCR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    ap.add_argument("--task",               required=False,
                    help=f"Task name. Known: {', '.join(sorted(TASK_REGISTRY))}")
    ap.add_argument("--dataset",            required=False,
                    help="Path to training .jsonl file")
    ap.add_argument("--anchor-cheatsheet",  required=False,
                    help="Path to anchor CS-ICL cheatsheet .txt file")
    ap.add_argument("--output-dir",         required=False,
                    help="Directory for output files")
    ap.add_argument("--model-source",       required=False,
                    help="Source model used for failure elicitation (e.g. openai/gpt-4.1-mini)")
    ap.add_argument("--models-proxy",       required=False,
                    help="Comma-separated proxy model IDs for U_LCB validation")

    # Optional: protocol
    ap.add_argument("--held-out-target",    default="",
                    help="Family substring to exclude from U_LCB acceptance (leave-one-out)")
    ap.add_argument("--oracle-mode",        default="label_only",
                    choices=["none", "label_only", "compressed", "full_cot"],
                    help="Information provided to the rule generator")
    ap.add_argument("--routing-mode",       default="routed",
                    choices=["global", "routed"],
                    help="How accepted rules are applied at inference")

    # Optional: failure region estimation
    ap.add_argument("--n-eval-seeds",       type=int, default=1,
                    help="Number of repeated evals per model for soft probability regions. "
                         "1 = hard binary (original). 3 = majority-vote soft regions.")
    ap.add_argument("--eval-temperature",   type=float, default=0.7,
                    help="Temperature used for repeated evals when --n-eval-seeds > 1")
    ap.add_argument("--tau-s",              type=float, default=0.5,
                    help="Source failure threshold for soft regions (p_s >= tau_s → fails)")
    ap.add_argument("--tau-p",              type=float, default=0.5,
                    help="Proxy failure threshold for V_shared (max_j p_j >= tau_p)")
    ap.add_argument("--tau-low",            type=float, default=0.33,
                    help="'Consistently correct' threshold (p <= tau_low)")

    # Optional: split sizes
    ap.add_argument("--rule-gen-n",         type=int, default=60)
    ap.add_argument("--gate-n",             type=int, default=40)
    ap.add_argument("--seed",               type=int, default=1000)

    # Optional: generation
    ap.add_argument("--n-candidates",       type=int, default=8)
    ap.add_argument("--candidates-per-subtype", type=int, default=3,
                    help="Candidates to generate per failure subtype")
    ap.add_argument("--temperatures",       default="0.2,0.5,0.8",
                    help="Comma-separated temperatures for candidate generation")
    ap.add_argument("--model-gen",          default="",
                    help="Model for rule generation (defaults to --model-source)")
    ap.add_argument("--max-rule-chars",     type=int, default=800)
    ap.add_argument("--no-subtypes",        action="store_true",
                    help="Disable subtype clustering; use flat-pool generation (original behaviour)")

    # Optional: repair loop
    ap.add_argument("--repair-attempts",    type=int, default=0,
                    help="Repair attempts per rejected candidate (0 = disabled)")

    # Optional: validation weights
    ap.add_argument("--lambda-w",           type=float, default=1.0,
                    help="Penalty weight on private regression in U_LCB")
    ap.add_argument("--mu-w",              type=float, default=1.0,
                    help="Penalty weight on easy regression in U_LCB")
    ap.add_argument("--nu-w",              type=float, default=0.05,
                    help="Penalty weight on rule length in U_LCB")
    ap.add_argument("--max-accepted",       type=int, default=3)
    ap.add_argument("--private-act-ceil",   type=float, default=0.10)
    ap.add_argument("--reg-easy-ceil",      type=float, default=0.05)

    # Optional: infra
    ap.add_argument("--concurrency",        type=int, default=30)
    ap.add_argument("--check-backends",     action="store_true",
                    help="Check local vLLM and OpenRouter connectivity, then exit")
    ap.add_argument("--generator-timeout",  type=float, default=30.0)
    ap.add_argument("--allow-skip-unavailable-generators", action="store_true")
    ap.add_argument("--activation-routing-mode", default="partition",
                    choices=["partition", "feature", "hybrid"])
    ap.add_argument("--min-net-fixes", type=int, default=0,
                    help="Optional explicit net-count gate for validator count mode")

    return ap.parse_args()



def _check_backends(args: argparse.Namespace) -> None:
    import requests

    load_dotenv(Path(__file__).parent.parent / ".env")
    print("[check] local vLLM endpoints")

    def _models_url(url: str) -> str:
        url = url.rstrip("/")
        for suffix in ("/chat/completions", "/completions"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
        if not url.endswith("/v1") and "/v1/" in url:
            url = url.split("/v1/", 1)[0] + "/v1"
        return url.rstrip("/") + "/models"

    for url in (
        os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        os.environ.get("VLLM_BASE_URL_2", "http://127.0.0.1:8001/v1"),
        os.environ.get("VLLM_BASE_URL_3", "http://127.0.0.1:8002/v1"),
    ):
        try:
            r = requests.get(_models_url(url), timeout=5)
            names = [m.get("id") for m in r.json().get("data", [])] if r.ok else []
            print(f"  {_models_url(url)}: {r.status_code} {names[:3]}")
        except Exception as exc:
            print(f"  {url}: ERROR {exc}")

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("[check] OpenRouter: skipped, OPENROUTER_API_KEY is not set")
        return
    payload = {
        "model": os.environ.get("ICR_CHECK_GENERATOR_MODEL", "openai/gpt-4.1-mini"),
        "messages": [{"role": "user", "content": "Reply exactly OK."}],
        "max_tokens": 16,
    }
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    direct = requests.Session()
    direct.trust_env = False
    try:
        r = direct.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=args.generator_timeout)
        print(f"[check] OpenRouter direct: {r.status_code}")
    except Exception as exc:
        print(f"[check] OpenRouter direct: ERROR {exc}")
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=args.generator_timeout)
        print(f"[check] OpenRouter env/proxy: {r.status_code}")
    except Exception as exc:
        print(f"[check] OpenRouter env/proxy: ERROR {exc}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    if args.check_backends:
        _check_backends(args)
        return
    missing = [name for name in ('task', 'dataset', 'anchor_cheatsheet', 'output_dir', 'model_source', 'models_proxy') if not getattr(args, name)]
    if missing:
        raise SystemExit('Missing required arguments unless --check-backends is used: ' + ', '.join(missing))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key      = get_api_key()
    task_spec    = get_task(args.task)
    anchor_text  = Path(args.anchor_cheatsheet).read_text(encoding="utf-8").strip()
    all_items    = load_jsonl(Path(args.dataset))
    proxy_models = [m.strip() for m in args.models_proxy.split(",") if m.strip()]
    gen_model    = args.model_gen or args.model_source
    temperatures = [float(t) for t in args.temperatures.split(",")]
    use_subtypes = not args.no_subtypes

    print(f"\n{'='*60}")
    print(f" SFCR  task={args.task}  seed={args.seed}")
    print(f" source={args.model_source}")
    print(f" proxies={[m.split('/')[-1] for m in proxy_models]}")
    print(f" oracle_mode={args.oracle_mode}  routing_mode={args.routing_mode}")
    print(f" n_eval_seeds={args.n_eval_seeds}  use_subtypes={use_subtypes}")
    print(f" repair_attempts={args.repair_attempts}")
    print(f"{'='*60}\n")

    # ── 1. Split ──────────────────────────────────────────────────────────
    splits = make_splits(
        all_items,
        rule_gen_n=args.rule_gen_n,
        gate_n=args.gate_n,
        seed=args.seed,
    )

    from .failure_regions import _tag_ids
    _tag_ids(splits.rule_gen)
    _tag_ids(splits.gate)

    # ── 2-3. Failure regions on rule_gen ─────────────────────────────────
    print("[pipeline] Computing failure regions on rule_gen split...")
    rg_regions = compute_failure_regions(
        items=splits.rule_gen,
        anchor_cheatsheet=anchor_text,
        source_model=args.model_source,
        proxy_models=proxy_models,
        api_key=api_key,
        task_spec=task_spec,
        concurrency=args.concurrency,
        label="rule_gen",
        n_evals=args.n_eval_seeds,
        eval_temperature=args.eval_temperature,
        tau_s=args.tau_s,
        tau_p=args.tau_p,
        tau_low=args.tau_low,
    )

    # ── 4. Skip guard ─────────────────────────────────────────────────────
    if rg_regions.skip_reason:
        print(f"\n[pipeline] SKIP — {rg_regions.skip_reason}")
        print("[pipeline] Writing anchor cheatsheet unchanged.")
        for mode in ("global", "routed"):
            out = output_dir / f"final_cheatsheet_{mode}.txt"
            out.write_text(anchor_text, encoding="utf-8")
        _write_summary(output_dir, args, accepted_rules=[], skipped=True,
                       skip_reason=rg_regions.skip_reason, regions=rg_regions)
        return

    # ── 5. Generate candidates ────────────────────────────────────────────
    print(f"\n[pipeline] Generating candidates "
          f"(use_subtypes={use_subtypes}, n_candidates={args.n_candidates})...")
    candidates = generate_candidates(
        V_shared=rg_regions.V_shared,
        V_private=rg_regions.V_private,
        anchor_cheatsheet=anchor_text,
        model=gen_model,
        api_key=api_key,
        n_candidates=args.n_candidates,
        temperatures=temperatures,
        oracle_mode=args.oracle_mode,
        max_rule_chars=args.max_rule_chars,
        use_subtypes=use_subtypes,
        candidates_per_subtype=args.candidates_per_subtype,
    )

    if not candidates:
        print("[pipeline] No valid candidates generated — writing anchor unchanged.")
        for mode in ("global", "routed"):
            (output_dir / f"final_cheatsheet_{mode}.txt").write_text(anchor_text, encoding="utf-8")
        _write_summary(output_dir, args, accepted_rules=[], skipped=False,
                       skip_reason="generation produced no valid candidates",
                       regions=rg_regions)
        return

    # ── 6. Gate baseline ─────────────────────────────────────────────────
    print("\n[pipeline] Pre-computing gate baseline...")
    gate_baseline = compute_gate_baseline(
        gate_items=splits.gate,
        anchor_cheatsheet=anchor_text,
        source_model=args.model_source,
        proxy_models=proxy_models,
        api_key=api_key,
        task_spec=task_spec,
        concurrency=args.concurrency,
    )

    # ── 7. Build repair function (if requested) ───────────────────────────
    repair_fn = None
    if args.repair_attempts > 0:
        def repair_fn(rule: dict, failure_profile: dict) -> dict | None:
            return repair_candidate(
                rule=rule,
                failure_profile=failure_profile,
                anchor_cheatsheet=anchor_text,
                model=gen_model,
                api_key=api_key,
                max_rule_chars=args.max_rule_chars,
                max_attempts=args.repair_attempts,
            )

    # ── 8. Validate candidates ────────────────────────────────────────────
    print(f"\n[pipeline] Validating {len(candidates)} candidates "
          f"(repair_attempts={args.repair_attempts})...")
    val_results = validate_candidates(
        candidates=candidates,
        gate_items=splits.gate,
        gate_baseline=gate_baseline,
        anchor_cheatsheet=anchor_text,
        source_model=args.model_source,
        proxy_models=proxy_models,
        held_out_target=args.held_out_target or None,
        api_key=api_key,
        task_spec=task_spec,
        concurrency=args.concurrency,
        lambda_w=args.lambda_w,
        mu_w=args.mu_w,
        nu_w=args.nu_w,
        max_accepted=args.max_accepted,
        private_activation_ceiling=args.private_act_ceil,
        reg_easy_ceiling=args.reg_easy_ceil,
        max_rule_chars=args.max_rule_chars,
        repair_fn=repair_fn,
        repair_attempts=args.repair_attempts,
    )

    # ── 9. Collect accepted rules ─────────────────────────────────────────
    accepted_rules = [r.rule for r in val_results if r.accepted]
    print(f"\n[pipeline] Accepted {len(accepted_rules)} rule(s).")

    # ── 10. Write outputs ─────────────────────────────────────────────────
    # accepted_rules.json
    rules_path = output_dir / "accepted_rules.json"
    rules_path.write_text(
        json.dumps(
            [
                {
                    **r.rule,
                    "rule_id":                rule_id(r.rule),
                    "u_lcb":                  round(r.u_lcb, 6),
                    "private_activation_rate": round(r.private_activation_rate, 4),
                    "reg_easy_worst":         round(r.reg_easy_worst, 4),
                    "count_gate_used":        r.count_gate_used,
                    "repaired":               r.rule.get("repaired", False),
                }
                for r in val_results if r.accepted
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # validation_results.json (all candidates with stats)
    val_path = output_dir / "validation_results.json"
    val_path.write_text(
        json.dumps(
            [
                {
                    "rule_id":                rule_id(r.rule),
                    "rule":                   r.rule.get("rule", "")[:120],
                    "subtype_idx":            r.rule.get("subtype_idx"),
                    "repaired":               r.rule.get("repaired", False),
                    "accepted":               r.accepted,
                    "u_lcb":                  round(r.u_lcb, 6),
                    "reject_reason":          r.reject_reason,
                    "count_gate_used":        r.count_gate_used,
                    "private_activation_rate": round(r.private_activation_rate, 4),
                    "reg_easy_worst":         round(r.reg_easy_worst, 4),
                    "per_proxy": {
                        pm.split("/")[-1]: {
                            "delta_shared":            round(s.delta_shared, 4),
                            "reg_private":             round(s.reg_private, 4),
                            "reg_easy":                round(s.reg_easy, 4),
                            "fixed_shared_count":      s.fixed_shared_count,
                            "reg_easy_count":          s.reg_easy_count,
                            "reg_private_count":       s.reg_private_count,
                            "private_activation_count": s.private_activation_count,
                            "activation_count":        s.activation_count,
                            "n_shared":                s.n_shared,
                            "n_private":               s.n_private,
                            "n_easy":                  s.n_easy,
                        }
                        for pm, s in r.per_proxy_stats.items()
                    },
                }
                for r in val_results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # Final cheatsheets
    for mode in ("global", "routed"):
        cs = build_cheatsheet(anchor_text, accepted_rules, mode=mode)
        (output_dir / f"final_cheatsheet_{mode}.txt").write_text(cs, encoding="utf-8")

    # JSONL log
    logger = SFCRLogger(output_dir / "sfcr_log.jsonl", run_id=args.output_dir)

    rg_shared_ids  = {it["_sfcr_id"] for it in rg_regions.V_shared}
    rg_private_ids = {it["_sfcr_id"] for it in rg_regions.V_private}
    rg_easy_ids    = {it["_sfcr_id"] for it in rg_regions.V_easy}

    anchor_cb_by_model = {
        model: {iid: cb for iid, cb in cb_map.items()}
        for model, cb_map in gate_baseline.correct_by_model.items()
    }
    log_condition(
        logger=logger,
        items=splits.gate,
        scored_by_model=anchor_cb_by_model,
        cheatsheet_text=anchor_text,
        condition="anchor",
        accepted_rules=[],
        routing_mode=args.routing_mode,
        oracle_mode=args.oracle_mode,
        task=args.task,
        dataset=args.dataset,
        seed=args.seed,
        source_model=args.model_source,
        v_shared_ids=rg_shared_ids,
        v_private_ids=rg_private_ids,
        v_easy_ids=rg_easy_ids,
    )

    logger.close()

    act_summary = activation_summary(accepted_rules, splits.gate)
    (output_dir / "activation_summary.json").write_text(
        json.dumps(act_summary, indent=2), encoding="utf-8"
    )

    _write_summary(
        output_dir, args,
        accepted_rules=accepted_rules,
        skipped=False,
        skip_reason=None,
        regions=rg_regions,
        val_results=val_results,
    )

    print(f"\n{'='*60}")
    print(f" SFCR complete — {len(accepted_rules)} rule(s) accepted")
    print(f" Output dir: {output_dir}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------

def _write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    accepted_rules: list[dict],
    skipped: bool,
    skip_reason: str | None,
    regions,
    val_results: list | None = None,
) -> None:
    summary = {
        "task":            args.task,
        "seed":            args.seed,
        "oracle_mode":     args.oracle_mode,
        "routing_mode":    args.routing_mode,
        "model_source":    args.model_source,
        "models_proxy":    args.models_proxy,
        # Generator provenance (rebuttal E design: manifest must record the
        # generator). Mirrors the resolution rule in main(): --model-gen
        # falls back to --model-source when unset.
        "model_gen":       args.model_gen or args.model_source,
        "held_out_target": args.held_out_target,
        "n_eval_seeds":    args.n_eval_seeds,
        "use_subtypes":    not args.no_subtypes,
        "repair_attempts": args.repair_attempts,
        "skipped":         skipped,
        "skip_reason":     skip_reason,
        "source_accuracy": round(regions.source_accuracy, 4),
        # Region sizes and denominators
        "v_shared_size":   len(regions.V_shared),
        "v_private_size":  len(regions.V_private),
        "v_easy_size":     len(regions.V_easy),
        "f_s_size":        len(regions.F_s),
        "jaccard_matrix":  {f"{k[0]}↔{k[1]}": round(v, 4)
                            for k, v in regions.jaccard_matrix.items()},
        "n_accepted":      len(accepted_rules),
        "accepted_rules":  [r.get("rule", "")[:80] for r in accepted_rules],
    }
    if val_results is not None:
        summary["n_candidates"] = len(val_results)
        summary["n_rejected"]   = sum(1 for r in val_results if not r.accepted)
        summary["n_count_gate"] = sum(1 for r in val_results if r.count_gate_used)
        summary["n_repaired"]   = sum(1 for r in val_results if r.rule.get("repaired"))

    (output_dir / "sfcr_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
