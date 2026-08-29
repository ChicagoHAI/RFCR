"""tasks/logical_deduction.py — TaskSpec for the logical_deduction_three_objects BBH task."""

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

_LD3_SCORING = """\
You are solving a logical ordering puzzle with three objects.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: (A), (B), or (C)  ← FIRST LINE.\
REASONING: Think step by step. Build the full ordering chain from the constraints, explicitly name any \
cheatsheet rules you are applying, and then state which option matches. If no rule matched, say so.
"""

_LD3_SCORING_COT = """\
You are solving a logical ordering puzzle with three objects.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Build the full left-to-right (or ordered) sequence from the constraints, then answer.

VERDICT: (A), (B), or (C)  ← FIRST LINE.\
REASONING: Think step by step. Build the full ordering chain from the constraints, explicitly name any \
cheatsheet rules you are applying, and then state which option matches. If no rule matched, say so.
"""


def _ld3_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _LD3_SCORING_COT if cot else _LD3_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


_LD3_SCORING_COT_RF = """\
You are solving a logical ordering puzzle with three objects.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output format - use plain text only, no markdown bold or headers:
REASONING: apply the cheatsheet rules step by step, building the full ordering from the constraints, then select the matching option.
VERDICT: (A)

(Replace (A) above with (B) or (C) as appropriate. VERDICT must be the last line, must be followed immediately by the option on the same line, and must appear exactly once.)"""


def _logical_deduction_three_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _LD3_SCORING_COT_RF.format(cheatsheet=cs, question=item["input"])


def _parse_logical_deduction_three_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            m = re.search(r"\(([ABC])\)", val)
            if m:
                return f"({m.group(1)})"
    return None


_LD3_RELATIONS = {
    "left":    "positional (left/right)",
    "right":   "positional (left/right)",
    "older":   "temporal/comparative (older/newer)",
    "newer":   "temporal/comparative (older/newer)",
    "heavier": "weight comparison (heavier/lighter)",
    "lighter": "weight comparison (heavier/lighter)",
    "taller":  "size comparison (taller/shorter)",
    "shorter": "size comparison (taller/shorter)",
    "larger":  "size comparison (larger/smaller)",
    "smaller": "size comparison (larger/smaller)",
    "faster":  "speed comparison (faster/slower)",
    "cheaper": "cost comparison (cheaper/more expensive)",
}


def _ld3_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    for kw, rel_type in _LD3_RELATIONS.items():
        if kw in t:
            return (rel_type,)
    return ("general ordering",)


def _ld3_key_to_conds(key: tuple) -> list[str]:
    (rel,) = key
    return [f"ordering relation is {rel}"]


_LD3_GEN_PROMPT = (
    "You are an expert in logical deduction helping a model that fails at ordering puzzles with three objects.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What step in building the ordering chain did the model get wrong?\n"
    "     (e.g. misread 'A is to the right of B' as 'B is to the right of A', "
    "or stopped after processing only 2 of 3 constraints)\n"
    "  2. What is the minimal ordering chain the model should have built?\n"
    "  3. What one check would have caught the error?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy specific object names or constraint text from the failures above.\n"
    "  • SUPPORT examples must use generic placeholder names (A, B, C or Object 1, 2, 3).\n"
    "  • The teaching note should apply to any 3-object ordering puzzle with this constraint structure.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Ordering Error Type] ===\n"
    "FAILURE_TYPE: A (misread a constraint direction) or B (correct chain, wrong position read-off)\n"
    "ACTIVATE IF:\n"
    "  - the puzzle involves: [describe the relation type and what the model typically does wrong]\n"
    "  - the error pattern is: [what specific mistake the model makes in these failures]\n"
    "DO NOT ACTIVATE IF: [the case where the model handles this relation correctly]\n"
    "COMMON WRONG MOVE: [exactly how the model misreads or misapplies the constraint]\n"
    "NEXT CHECK: [the specific constraint to re-read carefully → which object is leftmost/smallest/etc.]\n"
    "WHY THIS WORKS: [1-2 sentences on the correct chain-building approach]\n"
    "SUPPORT:\n"
    "  • [mini 3-object ordering example]  |  Answer: (X)  — [brief note]\n"
    "{retry_context}"
)


def _ld3_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="logical deduction — ordering three objects from pairwise constraints",
        rule_prefix="LD",
        concepts="constraint chaining, transitive ordering, reading direction of comparisons, "
                 "positional vs comparative relations",
        verdict_fmt="(A), (B), or (C)",
        ruleset_intro="Build the full ordering from all constraints, then answer. Rules:",
        ruleset_footer="\nIf no rule applies, write out all constraints as a chain "
                       "and read off the answer.\n\nVERDICT: (A), (B), or (C)\n"
                       "REASONING: Show the constraint chain leading to the answer.",
        section_title="LOGICAL DEDUCTION RULES",
    )


LOGICAL_DEDUCTION_3_TASK = TaskSpec(
    build_scoring_prompt=_ld3_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_ld3_partition_key,
    partition_key_to_conditions=_ld3_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_LD3_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="logical_deduction_three_objects",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_ld3_bootstrap,
    build_eval_prompt=_make_eval_prompt("(A), (B), or (C)"),
)
