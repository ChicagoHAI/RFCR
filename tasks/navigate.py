"""tasks/navigate.py — TaskSpec for the navigate BBH task."""

from __future__ import annotations

import re

from utils.task_spec import TaskSpec
from tasks.utils import (
    _make_eval_prompt,
    _parse_yesno,
    _yesno_correct,
    _yesno_label,
    _extract_reasoning,
    _format_failure,
    _rule_score_prompt,
    _bootstrap_ruleset,
    _generic_polarity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_NAV_SCORING = """\
You are determining whether a sequence of movement instructions returns to the starting point.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Always face forward. Track net displacement: forward/backward on one axis, left/right on the other.
You return to start only if BOTH net displacements are zero.

VERDICT: Yes or No  ← FIRST LINE.\
REASONING: Think step by step. Show the running totals for each axis (forward/backward and left/right) \
as you process each instruction. Explicitly name any cheatsheet rules you are applying. \
If no rule matched, say so.
"""

_NAV_SCORING_COT = """\
You are determining whether a sequence of movement instructions returns to the starting point.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: track forward/backward total and left/right total separately.
Return to start iff both totals are zero.

VERDICT: Yes or No  ← FIRST LINE.\
REASONING: Think step by step. Show the running totals for each axis as you process each instruction, \
explicitly name any cheatsheet rules you are applying, then state whether both totals are zero. \
If no rule matched, say so.
"""


def _nav_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _NAV_SCORING_COT if cot else _NAV_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


_NAV_SCORING_COT_RF = """\
You are determining whether a sequence of movement instructions returns to the starting point.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output format - use plain text only, no markdown bold or headers:
REASONING: track forward/backward total and left/right total separately through each instruction, applying the cheatsheet rules, then determine whether both totals are zero.
VERDICT: YES

(Replace YES above with NO if you do not return to the starting point. VERDICT must be the last line, must be followed immediately by YES or NO on the same line, and must appear exactly once.)"""


def _navigate_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _NAV_SCORING_COT_RF.format(cheatsheet=cs, question=item["input"])


def _parse_navigate_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val.startswith("YES"):
                return "Yes"
            if val.startswith("NO"):
                return "No"
    return None


def _nav_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    # Compute net forward/back and left/right from the instructions
    fb = 0
    lr = 0
    for m in re.finditer(r"take (\d+) steps? (forward|backward|left|right)", t):
        n = int(m.group(1))
        d = m.group(2)
        if d == "forward":    fb += n
        elif d == "backward": fb -= n
        elif d == "left":     lr -= n
        elif d == "right":    lr += n
    fb_balanced = (fb == 0)
    lr_balanced = (lr == 0)
    return (fb_balanced, lr_balanced)


def _nav_key_to_conds(key: tuple) -> list[str]:
    fb_bal, lr_bal = key
    conds = []
    conds.append("forward/backward steps are balanced (net=0)" if fb_bal
                 else "forward/backward steps are NOT balanced")
    conds.append("left/right steps are balanced (net=0)" if lr_bal
                 else "left/right steps are NOT balanced")
    return conds


_NAV_GEN_PROMPT = (
    "You are an expert in spatial reasoning helping a model that fails at navigation return-to-start problems.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "KEY INSIGHT: Track two independent axes. Forward/backward is one axis; left/right is another.\n"
    "Both must sum to zero to return to start. 'Always face forward' means turning is irrelevant.\n\n"
    "DIAGNOSE:\n"
    "  1. Which axis did the model compute wrong?\n"
    "  2. Did it forget to track one axis? Miscalculate a subtraction? Confuse forward with right?\n"
    "  3. What check would have caught the error?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy specific instruction sequences from the failures above.\n"
    "  • SUPPORT examples must use freshly invented short step sequences.\n"
    "  • The teaching note should apply to any navigation sequence with this axis-error structure.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Navigation Error Type] ===\n"
    "FAILURE_TYPE: A (wrong axis calculation) or B (correct totals, wrong conclusion)\n"
    "ACTIVATE IF:\n"
    "  - the error pattern is: [describe what the model does wrong — e.g. 'only checks one axis']\n"
    "  - the instruction sequence has: [describe the structure that triggers the error]\n"
    "DO NOT ACTIVATE IF: [the simple case where steps obviously cancel]\n"
    "COMMON WRONG MOVE: [exactly what the model miscalculates]\n"
    "NEXT CHECK: [compute forward−backward total AND left−right total → both zero = Yes, else No]\n"
    "WHY THIS WORKS: [1-2 sentences on two-axis independence]\n"
    "SUPPORT:\n"
    "  • [short instruction sequence]  |  Answer: Yes/No  — [axis totals]\n"
    "{retry_context}"
)


def _nav_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="navigation — does a sequence of steps return to the starting point?",
        rule_prefix="NV",
        concepts="two-axis tracking (forward/back, left/right), net displacement, "
                 "both axes must be zero to return to start, 'always face forward' means no turning",
        verdict_fmt="Yes or No",
        ruleset_intro="Track forward/backward and left/right totals independently. Rules:",
        ruleset_footer="\nIf no rule applies, sum forward−backward and left−right separately. "
                       "Return to start iff both sums are zero.\n\n"
                       "VERDICT: Yes or No\nREASONING: Show both axis totals (forward−backward and left−right).",
        section_title="NAVIGATE RULES",
    )


NAVIGATE_TASK = TaskSpec(
    build_scoring_prompt=_nav_scoring_prompt,
    is_correct=_yesno_correct,
    answer_label=_yesno_label,
    parse_verdict=_parse_yesno,
    extract_post_think=_extract_reasoning,
    partition_key=_nav_partition_key,
    partition_key_to_conditions=_nav_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_NAV_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="navigate",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_nav_bootstrap,
    build_eval_prompt=_make_eval_prompt("Yes or No"),
)
