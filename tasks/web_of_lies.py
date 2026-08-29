"""tasks/web_of_lies.py — TaskSpec for the web_of_lies BBH task."""

from __future__ import annotations

import os as _os
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

_WOL_SCORING = """\
You are tracing a chain of truth-tellers and liars.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Rules: A truth-teller's claim about someone is accurate. A liar's claim is the opposite.
Track each person's actual truth-value through the chain.

VERDICT: Yes or No  ← FIRST LINE.\
REASONING: Trace each step of the chain (person → truth-value, showing how each "lies" flips and \
each "tells the truth" preserves). Explicitly name any cheatsheet rules you are applying. \
If no rule matched, say so.
"""

_WOL_SCORING_COT = """\
You are tracing a chain of truth-tellers and liars.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: start from the first given fact and propagate through each claim.

VERDICT: Yes or No  ← FIRST LINE.\
REASONING: Trace each step of the chain (person → truth-value, showing how each "lies" flips and \
each "tells the truth" preserves). Explicitly name any cheatsheet rules you are applying. \
If no rule matched, say so.
"""

_WOL_SCORING_COT_RF = """\
You are tracing a chain of truth-tellers and liars.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output format - use plain text only, no markdown bold or headers:
REASONING: apply the cheatsheet protocol step by step, starting from the given fact and propagating through each claim.
VERDICT: YES

(Replace YES above with NO if the final person does not tell the truth. VERDICT must be the last line, must be followed immediately by YES or NO on the same line, and must appear exactly once.)"""


def _wol_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _WOL_SCORING_COT if cot else _WOL_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _web_of_lies_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _WOL_SCORING_COT_RF.format(cheatsheet=cs, question=item["input"])


def _parse_web_of_lies_rf(text: str):
    """Parse verdict from reasoning-first format (VERDICT on last line)."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val.startswith("YES"):
                return "Yes"
            if val.startswith("NO"):
                return "No"
    return None


def _wol_partition_key(item: dict) -> tuple:
    # Count number of people in the chain (count "says" occurrences)
    n_says = item["input"].lower().count(" says ")
    chain_len = "short" if n_says <= 2 else "medium" if n_says <= 4 else "long"
    # Does the chain start with a liar?
    starts_with_lie = "lies." in item["input"].split(".")[0].lower() or \
                      "lies\n" in item["input"].lower()[:100]
    return (chain_len, starts_with_lie)


def _wol_key_to_conds(key: tuple) -> list[str]:
    chain_len, starts_lie = key
    conds = [f"chain length is {chain_len} ({{'short': '≤2', 'medium': '3–4', 'long': '5+'}}[chain_len] + ' claims')"]
    if starts_lie:
        conds.append("chain starts with someone who lies")
    else:
        conds.append("chain starts with someone who tells the truth")
    return conds


_WOL_V3 = (
    "You are an expert in propositional logic helping a model that fails at truth-chain problems.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "KEY INSIGHT: Each 'lies' in the chain flips the truth-value; each 'tells the truth' keeps it.\n"
    "Count the number of 'lies' in the chain. Even number of lies → same as the initial truth-value. "
    "Odd number → flipped.\n\n"
    "DIAGNOSE:\n"
    "  1. Where in the chain did the model lose track of the truth-value?\n"
    "  2. Did it miscount 'lies'? Misread 'tells the truth'? Stop too early?\n"
    "  3. What rule would prevent this?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy person names from the failures above.\n"
    "  • SUPPORT examples must use generic placeholder names (Person A, Person B, etc.).\n"
    "  • The teaching note should apply to any truth-chain of this length and polarity structure.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Chain Error Type] ===\n"
    "FAILURE_TYPE: A (lost track mid-chain) or B (correct tracking, wrong final verdict)\n"
    "ACTIVATE IF:\n"
    "  - chain looks like: [describe the pattern — e.g. 'starts with liar, 3+ says claims']\n"
    "  - the model's error is: [what specific step it got wrong]\n"
    "DO NOT ACTIVATE IF: [the simpler case where the model traces correctly]\n"
    "COMMON WRONG MOVE: [exactly where the model drops the flip or counts wrong]\n"
    "NEXT CHECK: [count the 'lies' claims; even=same, odd=flipped → Yes or No]\n"
    "WHY THIS WORKS: [1-2 sentences on parity tracking]\n"
    "SUPPORT:\n"
    "  • [short chain example: A lies, B says A tells truth → B lies → No]  |  Answer: No  — [note]\n"
    "{retry_context}"
)

_WOL_V4 = (
    "You are an expert in propositional logic helping a model that fails at truth-chain problems.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "KEY INSIGHT: Each 'lies' in the chain flips the truth-value; each 'tells the truth' keeps it.\n"
    "Count the number of 'lies' in the chain. Even number of lies → same as the initial truth-value. "
    "Odd number → flipped.\n\n"
    "DIAGNOSE:\n"
    "  1. Where in the chain did the model lose track of the truth-value?\n"
    "  2. Did it miscount 'lies'? Misread 'tells the truth'? Stop too early?\n"
    "  3. What rule would prevent this?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy person names from the failures above.\n"
    "  • SUPPORT examples must use generic placeholder names (Person A, Person B, etc.).\n"
    "  • The teaching note should apply to any truth-chain of this length and polarity structure.\n\n"
    "ACTIVATE IF — REQUIRED CONSTRAINTS:\n"
    "  Conditions must describe a REASONING FAILURE MODE that is identifiable WITHOUT knowing the answer.\n"
    "  • DO NOT include the expected answer (Yes or No) as a condition. The answer is unknown at inference\n"
    "    time; encoding it is circular and causes the case study to overfit to the training distribution.\n"
    "  • DO NOT write conditions tied to a specific chain length from these failures. Describe the\n"
    "    structural error class (e.g. 'chain contains an odd number of lies claims but model fails\n"
    "    to flip'), not a raw count that may not recur in unseen examples.\n"
    "  • A valid ACTIVATE IF condition must apply to a broad class of truth-chain problems, not only\n"
    "    to the specific individuals or chain configurations present in the training failures above.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Chain Error Type] ===\n"
    "FAILURE_TYPE: A (lost track mid-chain) or B (correct tracking, wrong final verdict)\n"
    "ACTIVATE IF:\n"
    "  - chain structure: [describe the STRUCTURAL pattern, e.g. 'chain contains multiple consecutive lies']\n"
    "  - reasoning error: [what specific step fails — do not name the expected answer]\n"
    "DO NOT ACTIVATE IF: [the simpler case where the model traces correctly]\n"
    "COMMON WRONG MOVE: [exactly where the model drops the flip or counts wrong]\n"
    "NEXT CHECK: [count the 'lies' claims; even=same, odd=flipped → Yes or No]\n"
    "WHY THIS WORKS: [1-2 sentences on parity tracking]\n"
    "SUPPORT:\n"
    "  • [short chain example: A lies, B says A tells truth → B lies → No]  |  Answer: No  — [note]\n"
    "{retry_context}"
)

WOL_PROMPTS: dict[str, str] = {"v3": _WOL_V3, "v4": _WOL_V4}
_WOL_LATEST = "v3"
_wol_env = _os.environ.get("ICR_GEN_PROMPT_VERSION", "").strip()
_WOL_GEN_PROMPT = WOL_PROMPTS[_wol_env if _wol_env in WOL_PROMPTS else _WOL_LATEST]


def _wol_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="web of lies — tracing truth-value through chains of truth-tellers and liars",
        rule_prefix="WL",
        concepts="truth propagation, lie flipping, parity of lies in chain, "
                 "chain length, starting truth-value",
        verdict_fmt="Yes or No",
        ruleset_intro="Trace the chain: each 'lies' flips the truth-value, each 'tells the truth' keeps it. Rules:",
        ruleset_footer="\nIf no rule applies, trace each person in order and track truth-value flips.\n\n"
                       "VERDICT: Yes or No\nREASONING: Trace each step in the chain and explain the truth-value flips.",
        section_title="WEB OF LIES RULES",
    )


WEB_OF_LIES_TASK = TaskSpec(
    build_scoring_prompt=_wol_scoring_prompt,
    is_correct=_yesno_correct,
    answer_label=_yesno_label,
    parse_verdict=_parse_yesno,
    extract_post_think=_extract_reasoning,
    partition_key=_wol_partition_key,
    partition_key_to_conditions=_wol_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_WOL_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="web_of_lies",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_wol_bootstrap,
    build_eval_prompt=_make_eval_prompt("Yes or No"),
)
