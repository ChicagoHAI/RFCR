# E3 Route Provenance Manifest

Date: 2026-05-23

## Purpose

This manifest makes the repaired-route result auditable. It records which atoms
were selected, how route definitions are represented, which item universe and
CS-ICL cache were used, and which parser/prompt-builder/retry policy produced
the final strict rerun.

## Compact Protocol Audit

| Field | Value |
|---|---|
| `run_id` | `disambig_needed_one_told_that_strict_csicl_rerun_20260522` |
| `item_surface` | `main_eligible_atom_tasks` |
| `n_items` | `400` |
| `item_ids_hash` | `06e51b82716d83a9faa8887a91fe36c2d5a41081e970605e4af719bf693c9bb9` |
| `csicl_cache_hash` | `84cbbf2d460d0f5e2925d12d0834d755efb3377089bfbb3b2c3b41d2f48f4c23` |
| `atom_file_hash` | `f2a90971412c403bcdbc17ba428bb41d26fd3fc3b2e41082151ddabb0002ccd9` |
| `route_file_hash` | `1d241f4526c2cbd460083f7646af446c3fd6a5f70a421a4da49ffb9fe97cf1c0` |
| `parser_version` | `task_spec.parse_verdict via scripts.run_controlled_full_benchmark_pilot._parse_prediction` |
| `prompt_builder_version` | `task_spec.build_eval_prompt` |
| `retry_policy` | `utils.llm_client.call_llm_batch_default` |
| `model_version` | `openai/gpt-4.1-mini` |
| `decode_config` | `temperature=0.0; max_tokens=64; reasoning_effort=low; top_p=provider default` |
| `bootstrap_samples` | `10000` |
| `frozen_before_eval` | `True` |
| `selected_atoms_frozen_at_utc` | `2026-05-22T15:50:05.034257+00:00` |
| `strict_effects_frozen_at_utc` | `2026-05-22T15:50:05.034257+00:00` |
| `final_run_created_at` | `2026-05-22T15:52:01` |

## Surface Hashes

| Surface | n | item_ids_hash | csicl_cache_hash | sfcr_outputs_hash |
|---|---:|---|---|---|
| `main_eligible_atom_tasks` | 400 | `06e51b82716d83a9faa8887a91fe36c2d5a41081e970605e4af719bf693c9bb9` | `84cbbf2d460d0f5e2925d12d0834d755efb3377089bfbb3b2c3b41d2f48f4c23` | `7494a9558de9f22d45809d309e7ae8a043fa302fbd1ba4cd3b4a08cba5a62a38` |
| `full18` | 1716 | `86e76590590f448761699c7dbc315d25922bea248629e959e08a9e21369486ad` | `47d4ac8c6fa52cb44a7a1f250054840db71c7174ea16557892b6ff470f6207a0` | `92338851b5a334542d0507d9c7a78c05d21e6cccf86cf9807b74cc4734b4352b` |

## Selected Atoms

| Task | Atom ID | Anchor | Source | Route |
|---|---|---|---|---|
| `disambiguation_qa` | `gen_disambig_specific_family_v1_2` | `csicl` | `focused_repair_20260522` | `needed_one_or_told_that` |
| `formal_fallacies` | `ff_role_universal_negative_complement_strict_v1` | `csicl` | `locked_20260519` | `` |
| `geometric_shapes` | `gen_geo_1_true_gpt41mini_v1` | `csicl` | `locked_20260519` | `` |
| `geometric_shapes` | `gen_geo_6_false_gpt41mini_repair_v3` | `csicl` | `locked_20260519` | `` |


## Freeze Audit

The selected atom file and strict effects file were written before the final
rerun manifest/output files:

- selected atoms frozen at: `2026-05-22T15:50:05.034257+00:00`
- strict effects frozen at: `2026-05-22T15:50:05.034257+00:00`
- final run manifest created at: `2026-05-22T15:52:01`
- frozen-before-eval flag: `True`

## Protocol Audit Summary

| Task | n | activated | inactive passthrough | missing effect rows | baseline mismatch |
|---|---:|---:|---:|---:|---:|
| `__aggregate_atom_tasks__` | 500 | 39 | 461 | 0 | 0 |
| `disambiguation_qa` | 100 | 15 | 85 | 0 | 0 |
| `formal_fallacies` | 100 | 4 | 96 | 0 | 0 |
| `geometric_shapes` | 100 | 20 | 80 | 0 | 0 |
| `object_counting` | 100 | 0 | 100 | 0 | 0 |
| `web_of_lies` | 100 | 0 | 100 | 0 | 0 |


## Source State Caveat

Git HEAD at manifest build time: `1f57c2d96ae5549fa68c7ef48e83e33b9669d279`.

The server tree is dirty/untracked because it contains experiment scripts and
local patches. To make this auditable, the manifest includes SHA256 hashes for
all source files used by the parser, prompt builder, route feature extraction,
and task wrappers. The exact list is stored in:

- `reports/sfcr_e3_route_provenance_manifest_20260523/source_script_hashes.json`

## Files Written

- `sfcr_e3_route_provenance_manifest.json`
- `route_definition_bundle.json`
- `source_script_hashes.json`
- `main_eligible_atom_tasks_item_ids.tsv`
- `main_eligible_atom_tasks_csicl_cache_canonical.jsonl`
- `main_eligible_atom_tasks_sfcr_outputs_canonical.jsonl`
- `full18_item_ids.tsv`
- `full18_csicl_cache_canonical.jsonl`
- `full18_sfcr_outputs_canonical.jsonl`
- `e3_compact_protocol_audit.csv`
- `table_e3_protocol_audit.tex`

## Claim Supported

The repaired-route result is a strict paired comparison rather than a mixture of
caches, parsers, item universes, or prompt builders.

## Claim Not Supported

This manifest does not prove cross-model zero-regression transfer and does not
turn AGIEval into a main benchmark claim.
