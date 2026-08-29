"""utils/task_spec.py — Abstract task specification for ICRefine.

A TaskSpec bundles all domain-specific logic needed to adapt ICRefine to any
binary (or multi-class) classification task beyond magma equational implication.

The four coupling surfaces:
  1. Scoring prompt — build_scoring_prompt, parse_verdict, extract_post_think
  2. Answer checking — is_correct, answer_label
  3. Partitioning   — partition_key, partition_key_to_conditions
  4. Generation     — format_failure, generation_prompt_template

Everything else in the loop (binning, gating, archive, crossover, rule-patch)
is task-agnostic and does not need to change.

Usage
-----
    from utils.task_spec import TaskSpec
    from tasks.magma import MAGMA_TASK

    result = run_partition_loop(..., task_spec=MAGMA_TASK)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class TaskSpec:
    """
    Domain-specific configuration for one ICRefine task.

    All callables must be pure (no shared mutable state) — they are called
    from worker threads without synchronisation.
    """

    # ── Scoring prompt ────────────────────────────────────────────────────────
    # Returns the complete prompt string sent to the evaluation model.
    # cot_first: if True, request REASONING before VERDICT in the response.
    build_scoring_prompt: Callable[[str, dict, bool], str]
    # (cheatsheet_text: str, item: dict, cot_first: bool) -> str

    # ── Answer checking ───────────────────────────────────────────────────────
    # Returns True if `predicted` is the correct answer for `item`.
    # Must return False when predicted is None (parse failure).
    is_correct: Callable[[str | None, dict], bool]
    # (predicted: str | None, item: dict) -> bool

    # Returns the ground-truth answer as a display string for annotated items.
    # E.g. "TRUE"/"FALSE" for magma, "(A)" for MCQA, "cat, dog" for sorting.
    answer_label: Callable[[dict], str]
    # (item: dict) -> str

    # Extracts the predicted answer string from a raw LLM response.
    # Returns None on parse failure (no recognisable answer found).
    parse_verdict: Callable[[str], str | None]
    # (raw_content: str) -> str | None

    # Extracts the reasoning / post-think section from a raw LLM response.
    # Falls back to the full content string if no distinct section found.
    extract_post_think: Callable[[str], str]
    # (raw_content: str) -> str

    # ── Partitioning ─────────────────────────────────────────────────────────
    # Returns a hashable tuple for structural binning. Items sharing the same
    # key go into the same PartitionBin and are solved concurrently.
    partition_key: Callable[[dict], tuple]
    # (item: dict) -> tuple

    # Converts a partition key back into human-readable ACTIVATE IF conditions
    # that are injected at the top of every generated case study for that bin.
    partition_key_to_conditions: Callable[[tuple], list[str]]
    # (key: tuple) -> list[str]

    # ── Case study generation ─────────────────────────────────────────────────
    # Format one failure item for the failure_lines block in the generation prompt.
    # The item dict may contain enriched fields set by the training loop:
    #   post_think     — model's wrong reasoning trace
    #   oracle_nearest — nearest-neighbour oracle item dict
    #   _oracle_exact  — exact oracle reasoning string (pre-baked by the loop)
    format_failure: Callable[[dict], str]
    # (item: dict) -> str

    # Case study generation prompt template.
    # Required {}-format placeholders:
    #   {roadmap}              — current cheatsheet reasoning roadmap
    #   {case_studies}         — current cheatsheet case studies (rendered)
    #   {already_covered}      — bullet list of patterns already in the cheatsheet
    #   {failure_lines}        — formatted failure items (from format_failure)
    #   {polarity_instruction} — polarity / failure-type directive
    #   {retry_context}        — previous-attempt context (empty on first try)
    generation_prompt_template: str

    # ── Optional overrides ────────────────────────────────────────────────────
    # Override the polarity instruction string builder.
    # Signature: (polarity: str, failure_type: str, divergence_step: str) -> str
    # If None, generate_candidates falls back to the built-in magma instructions.
    build_polarity_instruction: Callable[[str, str, str], str] | None = None

    # Task identifier — used in log messages and update_log entries.
    task_name: str = "unnamed"

    # ── Rule patching (ICR_rules integration) ────────────────────────────────
    # Build the SAIR-style scoring prompt from a rendered rule-set template text
    # and one item. If None, falls back to jinja2 {{ equation1 }}/{{ equation2 }}.
    # Signature: (template_text: str, item: dict) -> str
    build_rule_scoring_prompt: Callable[[str, dict], str] | None = None

    # Extract the dominant triggered rule ID from a model reasoning trace.
    # Returns None if no recognizable rule ID is found.
    # If None, falls back to the magma identify_triggered_rule parser.
    # Signature: (reasoning_text: str) -> str | None
    identify_triggered_rule: Callable[[str], str | None] | None = None

    # Build the full rule-patch generation prompt from raw rule + failure data.
    # If None, generate_rule_patch falls back to the built-in RULE_PATCH_PROMPT (magma).
    # Signature: (target_rule, rule_set, failures: list[dict],
    #             correct_pool: list[dict], oracle: dict) -> str
    build_rule_patch_prompt: Callable[..., str] | None = None

    # Regex pattern string used to parse rule IDs in patch LLM responses.
    # If None, uses the built-in magma pattern (TR-\w+|FR-\w+|...).
    rule_id_regex: str | None = None

    # Bootstrap an initial rule set from failure examples (called once before Phase 1).
    # If None, --auto-rule-init is not available for this task.
    # Signature: (failures: list[dict], model: str, api_key: str) -> RuleSet
    bootstrap_ruleset: Callable[..., object] | None = None

    # ── Concrete-example generation (alternative to abstract rules + case studies) ──
    # When set, the pipeline calls this instead of bootstrap_ruleset to seed
    # prior_knowledge with named concrete scenario examples (CS-ICL style).
    # Phase 1 rule-patching is skipped. Phase 2 uses concrete_cs_gen_fn if set.
    # Signature: (failures: list[dict], model: str, api_key: str) -> str
    bootstrap_cheatsheet_fn: Callable[..., str] | None = None

    # When set, Phase 2 calls this to generate a raw named-scenario text section
    # for a failure cluster instead of the structured CaseStudy generator.
    # The section is gated on fix_rate and regression before being appended to
    # cheatsheet.prior_knowledge.
    # Signature: (failures: list[dict], cheatsheet_text: str,
    #             model: str, api_key: str) -> str | None
    concrete_cs_gen_fn: Callable[..., str | None] | None = None

    # ── Evaluation-only prompt ────────────────────────────────────────────────
    # Minimal verdict-only prompt used during test-set evaluation (no REASONING
    # instruction).  The model sees the cheatsheet and question, then outputs
    # only the verdict label — no reasoning scaffold, lower token cost, cleaner
    # signal.
    # Signature: (cheatsheet_text: str, item: dict) -> str
    # If None, eval falls back to build_scoring_prompt(cot_first=False).
    build_eval_prompt: Callable[[str, dict], str] | None = None

    # ── Reasoning-free / label-only prompt surfaces ──────────────────────────
    # Several task modules expose lightweight prompt builders used by newer
    # pipeline variants.  They are optional here so older loops keep using the
    # canonical build_scoring_prompt/parse_verdict path.
    build_scoring_prompt_rf: Callable[..., str] | None = None
    build_label_prompt: Callable[..., str] | None = None
    parse_label: Callable[[str], str | None] | None = None
    build_pass1_label_prompt: Callable[..., str] | None = None
    parse_pass1_label: Callable[[str], str | None] | None = None

    # Optional metadata used by CS-ICL / rule-patching helpers.
    label_field: str | None = None
    format_for_csicl: Callable[[dict], str] | None = None
    patch_domain: str | None = None
    patch_verdict_format: str | None = None
    patch_rule_style: str | None = None
