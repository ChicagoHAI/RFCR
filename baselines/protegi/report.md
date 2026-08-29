# ProTeGi seeded-anchor baseline — report

Method: ProTeGi / APO (Pryzant et al., EMNLP 2023), the local LMOps
`prompt_optimization` BBH adaptation, SEEDED with the locked CS-ICL
cheatsheet per task (identical prompt wrapper to the locked evaluator).
Optimization on `datasets/bbh/{task}_train.jsonl` only; single run per task,
config verbatim in `{task}/config_used.json` (4 rounds, beam via UCB
evaluator, gpt-4.1-mini as task and gradient model, max_threads 12-16).
Frozen (SHA-256) best prompt evaluated once on the 100 test items
(temp 0.0, seed 42, max_tokens 64, repo parsers, locked E3 anchor cache).
Zero-leak audits over every optimization prompt/response: 0 test-item
matches on all four tasks (9,262-plus train sanity hits each; details in
`{task}/leak_audit.json`).

## Results (frozen ProTeGi prompt vs locked CS-ICL anchor)

| Task | ProTeGi | Anchor | Fix | Reg | Net | Changed from seed |
|---|---:|---:|---:|---:|---:|---|
| disambiguation_qa | 82.0% | 83.0% | 1 | 2 | -1 | NO (seed retained) |
| geometric_shapes | 51.0% | 56.0% | 5 | 10 | -5 | yes |
| formal_fallacies | 69.0% | 67.0% | 10 | 8 | +2 | yes |
| object_counting | 80.0% | 68.0% | 16 | 4 | +12 | yes |
| **Pooled (400)** | **70.5%** | **68.5%** | **32** | **24** | **+8** | |

Pooled task-stratified paired bootstrap (10k, seed 20260712):
**+2.00 pp, 95% CI [-1.50, +5.50]**, two-sided p=0.309.

## Reading

- disambiguation_qa: the beam search RETAINED THE SEED (identical SHA-256)
  after 4 rounds / 7,075 calls — no candidate beat the cheatsheet on train.
  The -1 test delta is fresh-call-vs-cache noise on identical content.
- The three adopted rewrites transfer unevenly: object_counting +12 is a
  real gain; formal_fallacies +2 pays 8 regressions for 10 fixes;
  geometric_shapes -5 (5 fixes / 10 regressions) — an adopted rewrite that
  looked better on train and harmed test.
- Aggregate: +2.0 pp — almost exactly RFCR's +2.75 pp dev figure — but with
  **24 regressions where RFCR has 0**. Matched net gain, unmatched harm
  profile: this is the sharpest single row of the baseline comparison.

## Cost

Optimization: 28,816 calls, est. $13.79; evals ~400
calls. Written by the coordinating session after the worker agent completed
all evals/audits but stalled writing this report; all numbers recomputed
from the per-item files.
