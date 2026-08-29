# RFCR: Robust Failure-Conservative Repair

Code and frozen artifacts for **Robust Failure-Conservative Repair (RFCR)**, a
conservative method for refining prompt memories (distilled cheat sheets) from
cross-model failures. RFCR generates candidate rule atoms from shared failures,
validates them with a conservative acceptance gate, routes them at inference
time, and preserves exact no-op passthrough when no safe atom activates.

This repository accompanies the paper and is provided so that the method, its
evaluation pipeline, and the frozen evidence behind the reported numbers can be
inspected and re-run.

## Repository layout

| Path | Contents |
|---|---|
| `ICR_sfcr/` | The RFCR method: failure regions, rule generation, the ULCB acceptance gate, activation-boundary repair, and routed activation. |
| `scripts/` | Evaluation pipeline. Main entry point: `scripts/run_sfcr_csicl_anchor_eval.py` (frozen atoms evaluated as a routed overlay on a CS-ICL anchor). |
| `tasks/` | BBH task specifications, prompt builders, and answer parsers. |
| `utils/` | Shared LLM client, scorer, data loaders, and task-spec helpers. |
| `datasets/bbh/` | The frozen per-task BBH train/test splits used throughout. |
| `runs/` | Frozen inputs the reported numbers depend on: the seed-1000 CS-ICL cheat sheets and the generated rule sets. |
| `artifacts/E3_route_provenance_manifest/` | Hash-locked 400-item main surface: item IDs, per-item CS-ICL answers, RFCR outputs, and selected atoms. |
| `baselines/` | ProTeGi and GEPA prompt-optimization baselines: drivers, per-task configs, frozen prompts, and result summaries. |
| `modal_qwen25_7b.py`, `modal_mistral7b.py` | Optional vLLM serving scripts for the two open-weight proxy models. |

The three small modules `ICR_partition/`, `ICR_reasoning/`, `ICR_rules/` contain
only the leaf utilities (a partition helper, an oracle-CSV loader, and a
rule-set dataclass) that the evaluation pipeline imports.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

Model access is configured in `.env`:

- `OPENROUTER_API_KEY` — routes most cloud models through OpenRouter.
- `OPENAI_API_KEY` — if set, `gpt-4*` / `o1*` / `o3*` / `o4*` models go directly
  to the OpenAI API.

The open-weight proxy models (`qwen2.5-7b-instruct`,
`mistral-7b-instruct-v0.3`) are served with vLLM. `modal_qwen25_7b.py` and
`modal_mistral7b.py` serve them on Modal, or run any OpenAI-compatible vLLM
endpoint locally and point `VLLM_BASE_URL` / `VLLM_BASE_URL_2` at it. When
scoring against these endpoints set `ICR_SCORING_MAX_TOKENS=2048` (their context
window is 8192).

## Reproducing the main result

The main result evaluates the frozen atom package as a routed overlay on the
locked CS-ICL anchor, on the 400-item BBH surface (disambiguation QA, formal
fallacies, geometric shapes, object counting):

```bash
export SFCR_DISAMBIG_RULES=runs/token_expansion_disambig_gpt41mini_20260522/generated_rules.json
python -m scripts.run_sfcr_csicl_anchor_eval \
  --eval-model openai/gpt-4.1-mini \
  --csicl-cheatsheet-root runs/emnlp_bbh_full18_csicl_gpt41mini_20260518/cheatsheets \
  --csicl-model-short gpt-4.1-mini --csicl-seed 1000 \
  --reasoning-effort low --max-tokens 64 --bootstrap-samples 10000
```

Decode is temperature 0.0, `max_tokens` 64. Activated items are rescored with
`CS-ICL cheat sheet + routed atom`; inactive items inherit the CS-ICL answer
byte-for-byte. `SFCR_DISAMBIG_RULES` must point at the `20260522` rule file —
the disambiguation atom IDs exist with different text in an earlier file.

The frozen 400-item surface (item IDs, CS-ICL answers, RFCR outputs, selected
atoms) with SHA-256 integrity anchors is in
`artifacts/E3_route_provenance_manifest/`.

## Baselines

`baselines/protegi/` and `baselines/gepa/` contain the drivers, per-task
configurations, frozen optimized prompts, leak audits, and result summaries for
the two prompt-optimization baselines. Both are seeded with the same locked
CS-ICL cheat sheet RFCR refines and optimize on the train split only.

## License

MIT. Copyright (c) 2026 ChicagoHAI. See `LICENSE`.
