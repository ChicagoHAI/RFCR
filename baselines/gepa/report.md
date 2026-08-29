# GEPA Baseline on the 4 BBH Main Tasks (ARR Rebuttal)

**Generated:** 2026-07-12. Reviewer-facing prompt-optimization comparison: **GEPA**
(reflective prompt evolution, ICLR 2026; standalone `gepa` pip library) seeded with the
**existing locked CS-ICL anchor prompt** and optimized per task on the **train split only**,
frozen, then evaluated once on each task's 100 test items and compared per-item against the
locked CS-ICL anchor. Framing: "newest-SOTA optimizer *refining* the anchor vs RFCR
*repairing* the anchor," identically seeded.

## Methodology

For each of the four main tasks (disambiguation_qa, geometric_shapes, formal_fallacies,
object_counting) we ran the standalone GEPA library (`gepa==0.1.1` from PyPI, in a fresh
venv at `rebuttal_experiments/gepa/.venv`; `litellm==1.83.7`) via its public
`gepa.optimize()` API with its own `DefaultAdapter`. The **seed candidate** is the locked
CS-ICL cheatsheet (`ICRefine/runs/emnlp_bbh_full18_csicl_gpt41mini_20260518/cheatsheets/
{task}/csicl_gpt-4.1-mini_seed1000.txt`) embedded verbatim in the exact prompt shape the
locked evaluator uses (`tasks/utils._make_eval_prompt` for dq/gs/ff;
`GenericBBHTask.build_eval_prompt` for object_counting) with the per-item input slot
removed — the item input is sent as the user message per the DefaultAdapter contract
(candidate = system message). Data: `ICRefine/datasets/bbh/{task}_train.jsonl` ONLY
(150 items), shuffled with `random.Random(0)` and split half/half into GEPA
trainset (75) / valset (75), mirroring the library's README Quick Start pattern
(`gepa.examples.aime.init_dataset`). Test files were never read during optimization.

**Metric** (protocol step 3): exact-match correctness through the repo's own task parsers
— `scripts.run_controlled_full_benchmark_pilot._parse_prediction` + `task_spec.is_correct`
— wrapped in a GEPA `Evaluator` returning score 1.0/0.0 plus a train-only textual feedback
line. No scoring logic reimplemented. Task model AND reflection model: **gpt-4.1-mini**.

The best candidate per task was frozen (SHA-256, recorded at optimization time in
`gepa_opt_summary.json` and re-verified before eval), then evaluated once on the 100 test
items through the repo's machinery (`utils.llm_client.call_llm_batch`, temperature 0.0,
seed 42, concurrency 12, max_tokens 64, model openai/gpt-4.1-mini). Fix/regression is
per-item vs the locked CS-ICL anchor cache
(`experiment_reports_and_results/E3_route_provenance_manifest/main_eligible_atom_tasks_csicl_cache_canonical.jsonl`):
fix = anchor-wrong -> GEPA-correct; regression = anchor-correct -> GEPA-wrong.

## GEPA configuration (documented verbatim; see per-task `gepa_config_used.json`)

- Library: `gepa==0.1.1` (PyPI; github.com/gepa-ai/gepa), `litellm==1.83.7`, Python 3.14.5.
- `gepa.optimize(seed_candidate={"system_prompt": <seed>}, trainset=75, valset=75,
  adapter=DefaultAdapter(model="openai/gpt-4.1-mini", evaluator=<exact-match repo-parser
  evaluator>, max_litellm_workers=16), reflection_lm="openai/gpt-4.1-mini",
  max_metric_calls=680, seed=0, run_dir=<task>/gepa_run)`.
- All other settings are library defaults: `candidate_selection_strategy="pareto"`,
  `frontier_type="instance"`, `skip_perfect_score=True`, `batch_sampler="epoch_shuffled"`,
  `reflection_minibatch_size=3` (default), `module_selector="round_robin"`,
  `use_merge=False`, `perfect_score=1.0`, `raise_on_exception=True`,
  `cache_evaluation=False`, `val_evaluation_policy=None` (=> full_eval), default reflection
  prompt template. `seed=0` is the library's default seed (recorded; the only RNG GEPA
  exposes).
- **Budget**: the standalone library has **no default** for `max_metric_calls` (required
  parameter; its README Quick Start uses 150, which with a 75-item valset would fund at
  most one candidate evaluation beyond the seed). Per protocol we used the documented DSPy
  GEPA **'light' preset** budget formula (`dspy.teleprompt.gepa.GEPA.auto_budget`,
  `num_preds=1`, `num_candidates=6` ['light' => n=6], `valset_size=75`,
  `minibatch_size=35`, `full_eval_steps=5`):
  `N = int(max(2*(1*2)*log2(6), 1.5*6)) = 10`;
  `total = 75 + 6*5 + 10*35 + ((10+1)//5 + 1)*75 = 680`.
  The engine's stopper is checked between steps, so actual metric calls slightly exceed
  680 (681/735/732/738 per task below).
- Task-LM calls go through `litellm.batch_completion` with the adapter's defaults: **no
  temperature/seed/max_tokens sent** (OpenAI server defaults apply, i.e. temperature 1.0)
  — this is the library's default behavior, recorded as-is. Reflection calls likewise.
- `max_litellm_workers=16` (library default 10) — parallelism knob only, raised per
  pre-registered instruction ("high parallelism where the library exposes it").
- Single run per task; no reruns, no selection among runs.

## Results (frozen GEPA prompt vs locked CS-ICL anchor, 100 test items/task)

| Task | GEPA test acc | CS-ICL anchor acc | Fix | Regression | Net | Delta pp | Best cand. |
|---|---:|---:|---:|---:|---:|---:|---|
| disambiguation_qa | 80.0% | 83.0% | 2 | 5 | -3 | -3.0 | **seed** (idx 0 of 6) |
| geometric_shapes | 49.0% | 56.0% | 1 | 8 | -7 | -7.0 | **seed** (idx 0 of 7) |
| formal_fallacies | 66.0% | 67.0% | 2 | 3 | -1 | -1.0 | **seed** (idx 0 of 8; cand 6 tied) |
| object_counting | 67.0% | 68.0% | 11 | 12 | -1 | -1.0 | rewrite (idx 5 of 8) |
| **Pooled (400)** | **65.5%** | **68.5%** | **16** | **28** | **-12** | **-3.0** | |

- Paired task-stratified bootstrap (10,000 resamples, seed 20260712), GEPA-vs-anchor pooled
  accuracy delta: **-3.0 pp, 95% CI [-6.25, +0.25]**, two-sided bootstrap p = 0.071
  (`paired_bootstrap_pooled.json`).
- Verdict parse rate **100%** on all 4 official eval runs (400/400) at **max_tokens 64**
  (the anchor protocol's own cap — no deviation needed; longest response 21 chars); 0
  failed calls.

### The headline finding

**On 3 of 4 tasks GEPA could not improve on the CS-ICL anchor at all**: every reflective
mutation it proposed scored at or below the seed on GEPA's own validation split, so its
returned best candidate is the **unchanged seed prompt** (`best_is_seed=true`; optimized
== seed SHA-256). Its proposals replaced the anchor's ~9-14k-char case-study cheatsheet
with ~3.5-4.6k-char generic instruction rewrites, which scored far below the seed
(e.g. dq: seed 0.987 on val vs mutations 0.72-0.84; gs: seed 0.653 vs 0.40-0.60).
On object_counting GEPA adopted a full rewrite (val 0.787 vs seed 0.667) that did **not**
transfer to test: 67% vs anchor 68% (11 fix / 12 regression, net -1).

### Key contrast with RFCR's conservative profile

On the same 400 items, the RFCR repaired route (documented E3 manifest,
`rebuttal_experiments/analysis/regression_profile/report.md`) achieves **11 fix / 0
regression**. GEPA — the newest reflective prompt-evolution optimizer, seeded with the
very same anchor and given a 'light'-preset budget (~680 metric calls/task) — returns the
anchor unchanged on 3/4 tasks and on the 4th produces a whole-prompt rewrite with a
symmetric fix/regression profile (11/12). Pooled: 16 fix / 28 regression (net -12; the
seed-returning tasks' deltas are eval-protocol noise, see honest note 2). This supports
the paper's claim from a second angle: given an already-strong deployed prompt surface,
whole-prompt reflective evolution either cannot beat it or trades fixes for an equal
regression tail, whereas RFCR's gated route makes strictly conservative repairs.

## Optimization-time zero-leak audit

Per task (`{task}/leak_audit.json`): every logged LLM request message and response (task +
reflection), every candidate prompt GEPA constructed (`gepa_run/candidates.json`), plus the
frozen seed/optimized prompts, scanned for normalized distinctive test-item text (same
normalization/marker approach as `rebuttal_experiments/e_qwen/audit_e_qwen.py`).

| Task | LLM calls audited | Docs audited | Candidates | Test-item matches | Train detector hits | Train/test overlap |
|---|---:|---:|---:|---:|---:|---:|
| disambiguation_qa | 695 | 2,079 | 6 | **0** | 1,369 | 0 |
| geometric_shapes | 767 | 2,278 | 7 | **0** | 1,120 | 0 |
| formal_fallacies | 750 | 2,242 | 8 | **0** | 1,040 | 0 |
| object_counting | 758 | 2,264 | 8 | **0** | 1,696 | 0 |

## Calls / cost / runtime

| Phase | Calls | Est. cost (USD) |
|---|---:|---:|
| Optimization (4 tasks: task-LM 681+735+732+738, reflection 14+32+18+20) | 2,970 | ~$1.84 |
| Test evals (4 x 100 official) + gs locked-composition diagnostic (100) + 2 smoke calls | 502 | ~$0.5 |
| **Total** | **~3,470** | **~$2.3** |

Optimization wall-clock: 5.4-11.2 min per task (task-LM minibatch/val evals run through
`litellm.batch_completion` at 16 workers; reflection calls are sequential).

## Honest notes (things that make the comparison imperfect)

1. **System-prompt composition.** GEPA's DefaultAdapter evaluates candidates as
   [system = candidate prompt, user = item input]. The locked anchor evaluator used a
   single user message with the verdict instruction *after* the item input. The seed
   preserves the locked template's text verbatim but relocates the input to the user slot;
   at test time the frozen prompt is prepended to the input as one user message (the repo
   client has no system channel — the same handling documented for the PromptWizard
   baseline).
2. **Eval-protocol noise dominates 3 of 4 rows.** For dq/gs/ff the frozen prompt is
   content-identical to the anchor cheatsheet, so their deltas (-3/-7/-1) measure fresh
   API calls + composition, not GEPA. A diagnostic rerun of geometric_shapes with the
   anchor content in the **exact locked composition**
   (`geometric_shapes/diagnostic_locked_composition.json`, not an official eval) scored
   51% vs the 56% cache (2 fix / 7 reg) — i.e. most of the gs -7 is fresh-call
   nondeterminism on boundary items, ~2 items attributable to composition. This echoes the
   repo-documented 1-2-item (here up to ~5-7 on gs) temp-0 nondeterminism; the anchor side
   comes from the locked cache exactly as pre-registered, fresh-call noise applies to the
   GEPA side only.
3. **Optimization-time sampling.** DefaultAdapter sends no sampling parameters on task
   calls (OpenAI default temperature 1.0, no seed) — the library default, kept as-is.
   GEPA's accept/reject decisions and val scores are therefore noisier than the temp-0
   test eval; the frozen prompt was selected under temp-1 behavior.
4. **Budget choice.** The standalone library requires `max_metric_calls` (no default). We
   used the DSPy 'light' preset formula (680; derivation above) rather than the README
   Quick Start's 150, which would have funded ~1 candidate. This is the more generous,
   more favorable-to-GEPA reading of "default budget."
5. **Integration adaptations (documented, algorithm untouched):** (a) custom `Evaluator`
   passed through the adapter's own public `evaluator` parameter (repo-parser exact match,
   per protocol); (b) `litellm.completion` wrapped for per-call audit logging + cost
   accounting, and `litellm.num_retries=5` for transport robustness (PW precedent);
   (c) `max_litellm_workers` 10 -> 16 (parallelism only). The first (dq) run double-logged
   task calls (batch_completion dispatches through `litellm.completion`, which was also
   wrapped); the dq log was deduplicated post-hoc (raw kept as
   `llm_calls_raw_doublelogged.jsonl.bak`) and the wrapper fixed for the other tasks —
   logging only, no effect on optimization.
6. **formal_fallacies tie.** Candidate 6 tied the seed on val (0.7467); GEPA's
   `best_idx` argmax returns the first maximum, i.e. the seed. Recorded as-is.
7. **object_counting probe-as-official.** The oc eval was launched with a `_mt64_probe`
   filename suffix to check for truncation at max_tokens=64; it parsed 100% (max response
   19 chars) and was adopted as the single official run — no rerun, no selection.
8. **Train split usage.** GEPA saw only the 150-item train file (75 train / 75 val, split
   seed 0 per the README pattern). The seed cheatsheet itself was built from the same
   train split by the locked CS-ICL run — identical seeding to the anchor, per protocol.
9. **Comparison asymmetry.** Anchor answers come from the locked cache (not re-run),
   exactly as pre-registered; GEPA eval answers are fresh API calls (see note 2).

## Deliverables

- `{task}/seed_prompt.txt`, `{task}/optimized_prompt.txt` (SHA-256 frozen in
  `gepa_opt_summary.json` before any test call and re-verified at eval),
  `{task}/gepa_config_used.json` (verbatim settings incl. defaults),
  `{task}/gepa_opt_summary.json` (val scores, lineage, calls, cost),
  `{task}/llm_calls.jsonl` (full optimization-time request/response log),
  `{task}/gepa_run/` (library-native run dir: candidates.json, run_log, state),
  `{task}/test_eval_per_item.jsonl`, `{task}/test_eval_summary.json`,
  `{task}/leak_audit.json`; `geometric_shapes/diagnostic_locked_composition.json`.
- `summary.csv`, `paired_bootstrap_pooled.json`, this `report.md`.
- Scripts: `run_gepa_task.py` (optimization driver), `run_gepa_test_eval.py`,
  `audit_gepa_leak.py`, `analyze_gepa.py`.
