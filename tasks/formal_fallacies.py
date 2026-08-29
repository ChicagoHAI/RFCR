"""tasks/formal_fallacies.py — TaskSpec for the formal_fallacies BBH task."""

from __future__ import annotations

import re

from utils.task_spec import TaskSpec
from tasks.utils import (
    _make_eval_prompt,
    _parse_valid_invalid,
    _valid_invalid_correct,
    _valid_invalid_label,
    _extract_reasoning,
    _format_failure,
    _rule_score_prompt,
    _bootstrap_ruleset,
    _generic_polarity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_FF_SCORING = """\
You are evaluating whether a deductive argument is valid or invalid.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: valid or invalid  ← FIRST LINE.\
REASONING: Think step by step. State the logical form (e.g. modus ponens, affirming the consequent), \
explain why the conclusion does or does not follow from the premises, and explicitly name any \
cheatsheet rules you are applying. If no rule matched, say so.
"""

_FF_SCORING_COT = """\
You are evaluating whether a deductive argument is valid or invalid.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: identify the logical form of the argument, then decide if \
the conclusion follows necessarily from the premises.

VERDICT: valid or invalid  ← FIRST LINE.\
REASONING: Think step by step. State the logical form, explain why the conclusion does or does not \
follow from the premises, and explicitly name any cheatsheet rules you are applying. \
If no rule matched, say so.
"""

_FF_SCORING_COT_RF = """\
You are evaluating whether a deductive argument is valid or invalid.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output exactly two plain-text lines. Do not use markdown, bullets, headings, or LaTeX.
Before writing the verdict, first check whether any ADDITIONAL RULE in the cheatsheet matches
the argument. If a rule matches, the REASONING line must name that rule and apply its CHECK.
If a matching rule explicitly says to output VALID or INVALID, obey that instruction.
Line 1: REASONING: one sentence under 45 words naming the decisive logical form or missing implication.
Line 2 must be exactly one of these two strings, without the words "Line 2":
VERDICT: VALID
VERDICT: INVALID

Choose VERDICT: INVALID when the conclusion is not forced by the premises."""


def _ff_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _FF_SCORING_COT if cot else _FF_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _formal_fallacies_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _FF_SCORING_COT_RF.format(cheatsheet=cs, question=item["input"])


def _parse_formal_fallacies_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val.startswith("VALID"):
                return "valid"
            if val.startswith("INVALID"):
                return "invalid"
    return None


def _ff_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    has_not_x_is_y = bool(re.search(r"whoever is not .+ is ", t))
    has_whoever    = "whoever" in t
    has_something  = "something" in t or "someone" in t
    return (has_not_x_is_y, has_whoever or has_something)


def _ff_key_to_conds(key: tuple) -> list[str]:
    has_not_x, has_whoever = key
    conds = []
    if has_not_x:
        conds.append("argument uses 'whoever is not X is Y' structure (common invalid converse)")
    if has_whoever:
        conds.append("argument uses universal quantifier (whoever / something / someone)")
    return conds or ["argument is a syllogism with stated premises and a conclusion"]


_FF_GEN_PROMPT = (
    "You are an expert in formal logic helping a model that keeps making errors on deductive validity.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What named logical fallacy (or valid form) is this?\n"
    "     Common invalids: affirming the consequent, denying the antecedent, illicit conversion\n"
    "     Common valids:   modus ponens, modus tollens, hypothetical syllogism\n"
    "  2. What is the model confusing? (e.g. thinks a converse is equivalent to original)\n"
    "  3. What one-sentence rule would have prevented the error?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT LOGICAL PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy premise text or predicate names (e.g. 'cousin of Chris') from the failures above.\n"
    "  • SUPPORT examples must be freshly invented simple syllogisms using generic placeholders (A, B, X, Y).\n"
    "  • The teaching note should apply to any argument of this logical form.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Fallacy Name or Valid Form] ===\n"
    "FAILURE_TYPE: A (wrong logical form identified) or B (right form, wrong validity verdict)\n"
    "ACTIVATE IF:\n"
    "  - argument structure looks like: [describe the surface form — e.g. 'if A then B; therefore if not-A then not-B']\n"
    "  - the conclusion attempts to: [what the argument is trying to derive]\n"
    "DO NOT ACTIVATE IF: [the structurally similar case that IS valid]\n"
    "COMMON WRONG MOVE: [what the model wrongly concludes and why]\n"
    "NEXT CHECK: [the one logical test to apply → valid or invalid]\n"
    "WHY THIS WORKS: [1-2 sentences on why the logical form (in)validates the conclusion]\n"
    "SUPPORT:\n"
    "  • [concrete everyday syllogism example]  |  Answer: valid/invalid  — [brief note]\n"
    "{retry_context}"
)


def _ff_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="formal deductive logic (valid vs invalid arguments)",
        rule_prefix="FF",
        concepts="modus ponens, modus tollens, affirming the consequent, denying the antecedent, "
                 "illicit conversion, hypothetical syllogism",
        verdict_fmt="valid or invalid",
        ruleset_intro="Evaluate the argument's logical form. Rules (apply first match):",
        ruleset_footer="\nIf no rule applies, check whether the conclusion follows necessarily "
                       "from the premises by logical form alone.\n\nVERDICT: valid or invalid\n"
                       "REASONING: Name the logical form applied, then explain.",
        section_title="FORMAL FALLACY RULES",
    )


FORMAL_FALLACIES_TASK = TaskSpec(
    build_scoring_prompt=_ff_scoring_prompt,
    is_correct=_valid_invalid_correct,
    answer_label=_valid_invalid_label,
    parse_verdict=_parse_valid_invalid,
    extract_post_think=_extract_reasoning,
    partition_key=_ff_partition_key,
    partition_key_to_conditions=_ff_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_FF_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="formal_fallacies",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_ff_bootstrap,
    build_eval_prompt=_make_eval_prompt("valid or invalid"),
    build_scoring_prompt_rf=_formal_fallacies_scoring_prompt_rf,
)
