"""tasks/date_understanding.py — TaskSpec for the date_understanding BBH task."""

from __future__ import annotations

import os as _os
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

_DU_SCORING = """\
You are solving a date arithmetic or calendar reasoning question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: (A)–(F)  ← FIRST LINE.\
REASONING: Think step by step. State the starting date, apply the operation step by step \
(add/subtract days or months, find day of week, convert format), explicitly name any cheatsheet \
rules you are applying, and then identify the matching option. If no rule matched, say so.
"""

_DU_SCORING_COT = """\
You are solving a date arithmetic or calendar reasoning question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: determine the starting date, apply the operation (add/subtract days or months, \
find day of week, convert format), and match to the options.

VERDICT: (A)–(F)  ← FIRST LINE.\
REASONING: Think step by step. State the starting date, apply the operation step by step, explicitly \
name any cheatsheet rules you are applying, and then identify the matching option. If no rule matched, \
say so.
"""

_DU_SCORING_COT_RF = """\
You are solving a date arithmetic or calendar reasoning question.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output format - use plain text only, no markdown bold or headers:
REASONING: apply the cheatsheet protocol step by step - state the starting date, apply each operation (add/subtract days or months, find day of week, convert format), and identify the matching option.
VERDICT: (A)

(Replace (A) above with the correct option letter (A)-(F). VERDICT must be the last line, must be followed immediately by the option on the same line, and must appear exactly once.)"""


def _du_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _DU_SCORING_COT if cot else _DU_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


def _date_understanding_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _DU_SCORING_COT_RF.format(cheatsheet=cs, question=item["input"])


def _parse_date_understanding_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            m = re.search(r"\(([A-F])\)", val)
            if m:
                return f"({m.group(1)})"
    return None


def _du_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    if "day of the week" in t or "what day" in t:
        op = "weekday"
    elif "month" in t and ("ago" in t or "later" in t or "from now" in t):
        op = "month_arithmetic"
    elif "ago" in t or "later" in t or "from now" in t:
        op = "day_arithmetic"
    elif "mm/dd/yyyy" in t or "format" in t:
        op = "format_conversion"
    elif "yesterday" in t or "tomorrow" in t:
        op = "relative_day"
    else:
        op = "general"
    return (op,)


def _du_key_to_conds(key: tuple) -> list[str]:
    (op,) = key
    labels = {
        "weekday":          "question asks for the day of the week",
        "month_arithmetic": "question involves adding or subtracting months",
        "day_arithmetic":   "question involves adding or subtracting days",
        "format_conversion":"question involves converting between date formats",
        "relative_day":     "question uses relative terms like yesterday or tomorrow",
        "general":          "general date reasoning question",
    }
    return [labels.get(op, f"date operation: {op}")]


_DU_V3 = (
    "You are an expert in calendar arithmetic helping a model that fails on date reasoning questions.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What date operation is being performed? (add days, add months, find weekday, convert format)\n"
    "  2. What is the starting date (explicit or implied)?\n"
    "  3. What specific arithmetic error did the model make?\n"
    "     (e.g. off by one in month count, wrong year for leap year, confused UK/US format)\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT reproduce specific dates, names, or scenario text from the failures above.\n"
    "  • SUPPORT examples must use freshly invented dates that illustrate the arithmetic pattern.\n"
    "  • The teaching note should apply to any date question of this operation type.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Date Operation Type: Error Description] ===\n"
    "FAILURE_TYPE: A (wrong starting date) or B (right start, wrong arithmetic)\n"
    "ACTIVATE IF:\n"
    "  - question involves: [describe the operation type and the typical error pattern]\n"
    "  - the error looks like: [what the model computes wrong]\n"
    "DO NOT ACTIVATE IF: [the simpler date question this model handles correctly]\n"
    "COMMON WRONG MOVE: [the specific arithmetic mistake with example]\n"
    "NEXT CHECK: [the step-by-step arithmetic to verify → which option matches]\n"
    "WHY THIS WORKS: [1-2 sentences on the correct procedure]\n"
    "SUPPORT:\n"
    "  • [concrete date example]  |  Answer: (X)  — [brief note]\n"
    "{retry_context}"
)

_DU_V4 = (
    "You are an expert in calendar arithmetic helping a model that fails on date reasoning questions.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE:\n"
    "  1. What date operation is being performed? (add days, add months, find weekday, convert format)\n"
    "  2. What is the starting date (explicit or implied)?\n"
    "  3. What specific arithmetic error did the model make?\n"
    "     (e.g. off by one in month count, wrong year for leap year, confused UK/US format)\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT reproduce specific dates, names, or scenario text from the failures above.\n"
    "  • SUPPORT examples must use freshly invented dates that illustrate the arithmetic pattern.\n"
    "  • The teaching note should apply to any date question of this operation type.\n\n"
    "ACTIVATE IF — REQUIRED CONSTRAINTS:\n"
    "  Conditions must be identifiable from the question structure alone, without knowing the answer.\n"
    "  • DO NOT include the correct answer option (A)–(F) as a condition. The answer is unknown at\n"
    "    inference time; encoding it is circular and causes the case study to overfit to the training set.\n"
    "  • DO NOT write conditions tied to specific calendar dates or scenario details from these failures.\n"
    "    Describe the operation type and error class (e.g. 'month-boundary arithmetic', 'format ambiguity'),\n"
    "    not a concrete date that will not recur verbatim in unseen questions.\n"
    "  • A valid ACTIVATE IF condition must apply to any date question involving this operation type,\n"
    "    not only to the particular dates and phrasing present in the training failures above.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Date Operation Type: Error Description] ===\n"
    "FAILURE_TYPE: A (wrong starting date) or B (right start, wrong arithmetic)\n"
    "ACTIVATE IF:\n"
    "  - question involves: [OPERATION TYPE — e.g. 'add N months crossing year boundary']\n"
    "  - error pattern: [what structural mistake the model makes — do not name the correct answer option]\n"
    "DO NOT ACTIVATE IF: [the simpler date question this model handles correctly]\n"
    "COMMON WRONG MOVE: [the specific arithmetic mistake with example]\n"
    "NEXT CHECK: [the step-by-step arithmetic to verify → which option matches]\n"
    "WHY THIS WORKS: [1-2 sentences on the correct procedure]\n"
    "SUPPORT:\n"
    "  • [concrete date example]  |  Answer: (X)  — [brief note]\n"
    "{retry_context}"
)

DU_PROMPTS: dict[str, str] = {"v3": _DU_V3, "v4": _DU_V4}
_DU_LATEST = "v3"
_du_env = _os.environ.get("ICR_GEN_PROMPT_VERSION", "").strip()
_DU_GEN_PROMPT = DU_PROMPTS[_du_env if _du_env in DU_PROMPTS else _DU_LATEST]


def _du_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="date understanding — calendar arithmetic and date format reasoning",
        rule_prefix="DU",
        concepts="adding/subtracting days and months, day-of-week calculation, "
                 "UK vs US date formats, month lengths, leap years",
        verdict_fmt="(A) through (F)",
        ruleset_intro="Identify the starting date and operation, then apply arithmetic. Rules:",
        ruleset_footer="\nIf no rule applies, determine the starting date from context, "
                       "apply the requested operation carefully, and match to the options.\n\n"
                       "VERDICT: (A)–(F)\nREASONING: Show the date arithmetic step by step.",
        section_title="DATE UNDERSTANDING RULES",
    )


DATE_UNDERSTANDING_TASK = TaskSpec(
    build_scoring_prompt=_du_scoring_prompt,
    is_correct=_mc_correct,
    answer_label=_mc_label,
    parse_verdict=_parse_mc,
    extract_post_think=_extract_reasoning,
    partition_key=_du_partition_key,
    partition_key_to_conditions=_du_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_DU_GEN_PROMPT,
    build_polarity_instruction=_generic_polarity,
    task_name="date_understanding",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_du_bootstrap,
    build_eval_prompt=_make_eval_prompt("(A), (B), (C), (D), (E), or (F)"),
)
