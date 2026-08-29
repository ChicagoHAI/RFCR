"""tasks/disambiguation_qa.py — TaskSpec for the disambiguation_qa BBH task."""

from __future__ import annotations

import re

from utils.task_spec import TaskSpec
from tasks.utils import (
    _make_eval_prompt,
    _parse_mc,
    _mc_correct,
    _mc_label,
    _extract_reasoning,
    _format_failure,
    _rule_score_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_DISAMBIG_SCORING = """\
You are resolving pronoun antecedents.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: (A), (B), or (C)  ← FIRST LINE.\
REASONING: Think step by step. Identify the candidates, which pronoun appears, and the pragmatic context.
"""

_DISAMBIG_SCORING_COT = """\
You are resolving pronoun antecedents.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think through who each candidate referent is, which pronouns appear, and the pragmatic context.

VERDICT: (A), (B), or (C)  ← FIRST LINE.\
REASONING: Think step by step. Identify the candidates, which pronoun appears, and the pragmatic context.
"""


def _disambig_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _DISAMBIG_SCORING_COT if cot else _DISAMBIG_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


_DISAMBIG_RF = """\
You are resolving pronoun antecedents.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Answer the question by following the provided cheatsheet. Ensure that your response ends with VERDICT: (A), (B), or (C)"""


def _disambiguation_qa_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _DISAMBIG_RF.format(cheatsheet=cs, question=item["input"])


def _parse_disambiguation_qa_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            m = re.search(r"\(([ABC])\)", val)
            if m:
                return f"({m.group(1)})"
    return None


_DQ_LABEL_PROMPT = """\
Analyze this pronoun disambiguation question and identify the primary resolution strategy.

QUESTION:
{question}

Identify which ONE strategy primarily determines the correct pronoun antecedent:
- subjecthood: the pronoun refers to the grammatical subject of the main clause (subject preference)
- recency: the pronoun refers to the most recently mentioned entity in context
- gender_agreement: the pronoun is resolved purely by matching grammatical gender/number
- thematic_role: the pronoun refers to the agent/patient/theme based on thematic role assignment
- world_knowledge: resolution requires real-world knowledge about typical behavior or roles
- genuinely_ambiguous: two or more cues conflict with genuinely equal strength; answer is "Ambiguous"
- other: resolution depends on a different strategy not listed above

Output EXACTLY two lines:
RESOLUTION_STRATEGY: <strategy>
REASON: <one sentence explanation>"""


def _build_dq_label_prompt(item: dict) -> str:
    return _DQ_LABEL_PROMPT.format(question=item.get("input", "").strip())


_DQ_STRATEGIES = {
    "subjecthood", "recency", "gender_agreement", "thematic_role",
    "world_knowledge", "genuinely_ambiguous", "other",
}


def _parse_dq_label(text: str) -> str:
    for line in text.strip().splitlines():
        if line.upper().startswith("RESOLUTION_STRATEGY:"):
            label = line.split(":", 1)[1].strip().lower()
            if label in _DQ_STRATEGIES:
                return label
            for s in _DQ_STRATEGIES:
                if s in label:
                    return s
    return "other"


def _disambig_partition_key(item: dict) -> tuple:
    # Prefer LLM semantic label injected by label pre-pass
    label = item.get("semantic_label")
    if label and label != "(unknown)":
        return (label,)
    t = item["input"].lower()
    pronoun = ("they" if "they " in t or "their " in t else
               "she" if " she " in t or " her " in t else "he")
    has_ambig_opt = "ambiguous" in t
    return (pronoun, has_ambig_opt)


def _disambig_key_to_conds(key: tuple) -> list[str]:
    if len(key) == 1:
        return [f"resolution strategy: {key[0]}"]
    pronoun, has_ambig = key
    conds = [f"pronoun is '{pronoun}'"]
    if has_ambig:
        conds.append("one option is 'Ambiguous'")
    return conds


def _disambig_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = (
        f"POLARITY — model selected wrong option (majority correct answer is {polarity}):\n"
        "TYPE A: model lacks pragmatic knowledge about typical pronoun resolution conventions.\n"
        "TYPE B: model has the knowledge but misjudges which entity is the antecedent."
    )
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up. Show the next step.\n\n" + base
    return base


def _disambig_identify_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(DQ-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(DQ-\w+)", reasoning)
    return m.group(1) if m else None


_DISAMBIG_GEN_PROMPT = (
    "You are an expert in pronoun resolution helping a model that fails on disambiguation questions.\n"
    "The task: given a sentence and 2-3 candidate referents, decide which entity a pronoun refers to.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. List the candidate referents and their answer labels (A), (B), (C).\n"
    "  2. What pronoun is being resolved? What gender/number constraints apply?\n"
    "  3. Which resolution cue determines the correct answer?\n"
    "     subjecthood preference / recency (most recent mention) / thematic role / gender agreement / world knowledge\n"
    "  4. Is 'Ambiguous' ever correct here? (only when two cues genuinely conflict with equal strength)\n"
    "  5. Did the model apply the wrong cue (TYPE A) or apply the right cue to the wrong entity (TYPE B)?\n\n"
    "STRUCTURAL VOCABULARY — use these exact terms in ACTIVATE IF:\n"
    "    pronoun: he / she / they / him / her / them / his / her\n"
    "    resolution_cue: subjecthood / recency / gender_agreement / thematic_role / world_knowledge\n"
    "    has_ambiguous_option: one answer option is 'Ambiguous'\n"
    "    n_candidates: number of named referent candidates (2 or 3)\n"
    "    answer_is_A / answer_is_B / answer_is_C\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy sentence text or names from the failures above.\n"
    "  • SUPPORT examples must be freshly invented sentences that illustrate the cue pattern.\n"
    "  • The teaching note should apply to any pronoun resolution question of this structural type.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short title naming the cue, e.g. 'Subjecthood Beats Recency'] ===\n"
    "FAILURE_TYPE: A (model uses wrong resolution cue) or B (right cue, wrong entity)\n"
    "ACTIVATE IF:\n"
    "  - [pronoun and resolution_cue from vocabulary above]\n"
    "DO NOT ACTIVATE IF: [case where the referent is unambiguous and model is correct]\n"
    "COMMON WRONG MOVE: [which entity the model picks and which cue it incorrectly prioritises]\n"
    "NEXT CHECK: [the cue to apply → answer is (A), (B), or (C)]\n"
    "WHY THIS WORKS: [1-2 sentences on the linguistic/pragmatic principle]\n"
    "SUPPORT:\n"
    "  • [example sentence + candidates]  |  Answer: (A)/(B)/(C)  — [cue note]\n"
    "{retry_context}"
)

DISAMBIGUATION_TASK = TaskSpec(
    build_scoring_prompt=_disambig_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_disambig_partition_key,
    partition_key_to_conditions=_disambig_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_DISAMBIG_GEN_PROMPT,
    build_polarity_instruction=_disambig_polarity,
    task_name="disambiguation_qa",
    build_scoring_prompt_rf=_disambiguation_qa_scoring_prompt_rf,
    build_rule_scoring_prompt=_rule_score_prompt,
    identify_triggered_rule=_disambig_identify_rule,
    rule_id_regex=r"(DQ-\w+)",
    bootstrap_ruleset=None,  # MC tasks: use Phase 2 only by default
    build_eval_prompt=_make_eval_prompt("(A), (B), or (C)"),
    build_label_prompt=_build_dq_label_prompt,
    parse_label=_parse_dq_label,
)
