"""tasks/snarks.py — TaskSpec for the snarks BBH task."""

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
    _bootstrap_ruleset,
    _generic_polarity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_SNARKS_SCORING = """\
You are identifying which of two statements is sarcastic.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

A sarcastic statement says the opposite of what it literally means, \
using positive language in a negative context or vice versa.

VERDICT: (A) or (B)  ← FIRST LINE.\
REASONING: Think step by step. Explain what makes the sarcastic option sarcastic (context mismatch, \
exaggerated praise, literal reading that contradicts common sense) and why the other is literal. \
Explicitly name any cheatsheet rules you are applying. If no rule matched, say so.
"""

_SNARKS_SCORING_COT = """\
You are identifying which of two statements is sarcastic.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: look for a mismatch between the literal meaning and the context — \
positive wording used for something bad, or sincere praise that would be absurd in context.

VERDICT: (A) or (B)  ← FIRST LINE.\
REASONING: Think step by step. For each option, identify whether there is a context mismatch. \
Explain what makes the sarcastic option sarcastic and the other literal. \
Explicitly name any cheatsheet rules you are applying. If no rule matched, say so.
"""

_SNARKS_SCORING_COT_RF = """\
You are identifying which of two statements is sarcastic.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output format - use plain text only, no markdown bold or headers:
REASONING: apply the cheatsheet rules step by step - for each option explain whether there is a mismatch between the literal meaning and the context, then identify which is sarcastic.
VERDICT: (A)

(Replace (A) above with (B) if statement (B) is the sarcastic one. VERDICT must be the last line, must be followed immediately by (A) or (B) on the same line, and must appear exactly once.)"""


def _snarks_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _SNARKS_SCORING_COT if cot else _SNARKS_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _snarks_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _SNARKS_SCORING_COT_RF.format(cheatsheet=cs, question=item["input"])


def _parse_snarks_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if "(A)" in val:
                return "(A)"
            if "(B)" in val:
                return "(B)"
    return None


def _snarks_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    # Detect common sarcasm signal types
    has_praise_bad = any(w in t for w in ["terrible", "awful", "horrible", "worst", "useless", "idiot"])
    has_ironic_positive = any(w in t for w in ["genius", "brilliant", "amazing", "great job", "well done"])
    return (has_praise_bad, has_ironic_positive)


def _snarks_key_to_conds(key: tuple) -> list[str]:
    praise_bad, ironic_pos = key
    conds = []
    if praise_bad:
        conds.append("one option uses clearly negative terms (terrible, awful, worst)")
    if ironic_pos:
        conds.append("one option uses exaggerated praise (genius, brilliant, great job)")
    if not conds:
        conds.append("sarcasm signal is subtle — context mismatch rather than explicit negative terms")
    return conds


_SNARKS_GEN_PROMPT = (
    "You are an expert in pragmatics and irony helping a model that fails at sarcasm detection.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What sarcasm signal is present in the correct answer that the model missed?\n"
    "     Think in terms of: context mismatch, exaggerated praise, self-evident absurdity, \n"
    "     literal reading that contradicts common sense\n"
    "  2. What made the wrong option look sarcastic to the model instead?\n"
    "  3. What everyday analogy captures the sarcasm pattern here?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy the specific statements or sentences from the failures above.\n"
    "  • SUPPORT examples must be freshly invented sarcastic statements illustrating the pattern.\n"
    "  • The teaching note should apply to any statement pair with this sarcasm structure.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Sarcasm Pattern]: [Memorable Name] ===\n"
    "FAILURE_TYPE: A (missed the sarcasm signal) or B (right signal, wrong option)\n"
    "ACTIVATE IF:\n"
    "  - scenario feels like: [describe the context mismatch pattern in plain language]\n"
    "  - the giveaway is: [what makes the sarcastic option recognizable once you see it]\n"
    "DO NOT ACTIVATE IF: [the case where the positive/negative wording is genuinely literal]\n"
    "COMMON WRONG MOVE: [what the model picks instead and why it's fooled]\n"
    "NEXT CHECK: [plain question to identify the sarcasm → (A) or (B)]\n"
    "WHY THIS WORKS: [1-2 sentences on the irony/context mismatch, in everyday terms]\n"
    "SUPPORT:\n"
    "  • [concrete sarcastic sentence example]  |  Answer: (X)  — [brief note on the mismatch]\n"
    "{retry_context}"
)


def _snarks_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="snarks — identifying which of two statements is sarcastic",
        rule_prefix="SK",
        concepts="context mismatch, irony, exaggerated praise for negative situation, "
                 "literal vs intended meaning, self-evident absurdity",
        verdict_fmt="(A) or (B)",
        ruleset_intro="Find the statement where the literal wording contradicts the implied meaning. Rules:",
        ruleset_footer="\nIf no rule applies, ask: which statement would be absurd or insincere if taken literally? "
                       "That is the sarcastic one.\n\nVERDICT: (A) or (B)\n"
                       "REASONING: Explain the literal-vs-intended meaning mismatch.",
        section_title="SNARKS RULES",
    )


SNARKS_TASK = TaskSpec(
    build_scoring_prompt=_snarks_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_snarks_partition_key,
    partition_key_to_conditions=_snarks_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_SNARKS_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="snarks",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_snarks_bootstrap,
    build_eval_prompt=_make_eval_prompt("(A) or (B)"),
)
