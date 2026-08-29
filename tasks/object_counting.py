"""tasks/object_counting.py — TaskSpec for the object_counting BBH task.

Items are "I have X ... How many <category> do I have?" questions with integer answers.
The model outputs a plain integer verdict.
"""

from __future__ import annotations

import re

from utils.task_spec import TaskSpec
from tasks.utils import (
    _make_eval_prompt,
    _extract_reasoning,
    _format_failure,
    _rule_score_prompt,
    _bootstrap_ruleset,
    _trivial_key,
    _trivial_conds,
    _generic_polarity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_OC_SCORING = """\
You are solving an object counting problem.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Convert all word-numbers to digits first (e.g. "four" → 4, "two" → 2). \
Then count only items that belong to the requested category. \
Sum the individual counts.

VERDICT: <integer>  ← FIRST LINE, a plain integer only, no units or punctuation.
REASONING: Show each item's count and the running total.\
"""

_OC_SCORING_COT = """\
You are solving an object counting problem.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: list each item with its quantity, filter by the requested category, \
then sum.

VERDICT: <integer>  ← FIRST LINE, a plain integer only.
REASONING: Show each item and running total, applying any cheatsheet rules.\
"""


def _oc_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _OC_SCORING_COT if cot else _OC_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


# ─────────────────────────────────────────────────────────────────────────────
# Parser / correctness
# ─────────────────────────────────────────────────────────────────────────────

def _parse_oc_verdict(content: str) -> str | None:
    # "VERDICT: 8" or "**VERDICT:** 8"
    m = re.search(r"\*{0,2}VERDICT\*{0,2}:?\*{0,2}\s*\*{0,2}(\d+)\*{0,2}", content, re.IGNORECASE)
    if m:
        return m.group(1)
    # First-line bare integer (model follows "FIRST LINE" literally)
    m_head = re.search(r"^\s*(\d+)\s*$", content[:60], re.MULTILINE)
    if m_head:
        return m_head.group(1)
    # Fallback: last standalone integer in tail
    nums = re.findall(r"\b(\d+)\b", content[-300:])
    return nums[-1] if nums else None


def _oc_correct(predicted: str | None, item: dict) -> bool:
    return predicted is not None and predicted.strip() == item["answer"].strip()


def _oc_label(item: dict) -> str:
    return item["answer"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Partition key — bucket by answer magnitude
# ─────────────────────────────────────────────────────────────────────────────

def _oc_partition_key(item: dict) -> tuple:
    try:
        n = int(item["answer"].strip())
    except (ValueError, KeyError):
        return ("unknown",)
    if n <= 5:
        bucket = "small"
    elif n <= 15:
        bucket = "medium"
    else:
        bucket = "large"
    # Also flag whether counting requires category filtering
    q = item["input"].lower()
    needs_filter = bool(re.search(
        r"how many (musical instruments?|fruits?|vegetables?|animals?|foods?|"
        r"toys?|clothes?|items? of clothing|pieces? of furniture|"
        r"electronics?|machines?|tools?|vehicles?|sports? equipment)",
        q,
    ))
    return (bucket, needs_filter)


def _oc_key_to_conds(key: tuple) -> list[str]:
    bucket, needs_filter = key
    conds = [f"answer is in the '{bucket}' range"]
    if needs_filter:
        conds.append("question asks to count a specific category (not all objects)")
    else:
        conds.append("question asks to count all objects or uses a broad category")
    return conds


# ─────────────────────────────────────────────────────────────────────────────
# Polarity instruction
# ─────────────────────────────────────────────────────────────────────────────

def _oc_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    if failure_type == "ABANDONMENT":
        return "STRATEGY — ABANDONMENT: model gave up or produced no integer. Show how to parse and count."
    if polarity.upper() in ("YES", "TRUE"):
        return (
            "POLARITY — UNDERCOUNT: model gave a number smaller than the correct answer. "
            "Diagnose whether it missed items (e.g. skipped a word-number like 'four stoves') "
            "or excluded items that belong to the requested category."
        )
    return (
        "POLARITY — OVERCOUNT: model gave a number larger than the correct answer. "
        "Diagnose whether it counted items from the wrong category, double-counted, "
        "or failed to filter out items that don't belong."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generation prompt
# ─────────────────────────────────────────────────────────────────────────────

_OC_GEN_PROMPT = (
    "You are an expert in arithmetic and category reasoning helping a model that fails at object-counting.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "KEY INSIGHT: Object counting has two independent error sources:\n"
    "  (A) PARSING ERROR — missing or miscounting word-numbers "
    "(e.g. 'four stoves' counts as 4, not 1; 'two lamps' counts as 2, not 1).\n"
    "  (B) FILTERING ERROR — counting items from the wrong category "
    "(e.g. including stoves when the question asks for musical instruments).\n\n"
    "DIAGNOSE:\n"
    "  1. Which error type occurred — parsing (A) or filtering (B), or both?\n"
    "  2. Which specific item was miscounted or wrongly included/excluded?\n"
    "  3. What one-line check would have caught it?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN, not memorise the specific failures.\n"
    "  • DO NOT copy item lists from the failures above.\n"
    "  • SUPPORT examples must use freshly invented short item lists.\n"
    "  • The teaching note must apply to any similar list-counting question.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short error-type title] ===\n"
    "FAILURE_TYPE: A (parsing) or B (filtering)\n"
    "ACTIVATE IF:\n"
    "  - [precise structural condition e.g. 'input contains word-numbers AND asks for a specific category']\n"
    "DO NOT ACTIVATE IF: [closest case where the model's approach is correct]\n"
    "COMMON WRONG MOVE: [exactly what the model miscounts or mislabels]\n"
    "NEXT CHECK: [concrete step to get the right integer]\n"
    "WHY THIS WORKS: [1-2 sentences]\n"
    "SUPPORT:\n"
    "  • [short invented item list]  |  Answer: <integer>  — [note on the error avoided]\n"
    "{retry_context}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _oc_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="object counting — count items of a specific category from a list",
        rule_prefix="OC",
        concepts="word-number conversion (four → 4), category filtering, summing individual counts",
        verdict_fmt="<integer>",
        ruleset_intro="Object counting rules:",
        ruleset_footer="\nIf no rule applies, parse each item's quantity, filter to the right category, sum.\n\n"
                       "VERDICT: <integer>\nREASONING: Show per-item counts and running total.",
        section_title="OBJECT COUNTING RULES",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TaskSpec
# ─────────────────────────────────────────────────────────────────────────────

OBJECT_COUNTING_TASK = TaskSpec(
    build_scoring_prompt=_oc_scoring_prompt,
    is_correct=_oc_correct,
    answer_label=_oc_label,
    parse_verdict=_parse_oc_verdict,
    extract_post_think=_extract_reasoning,
    partition_key=_oc_partition_key,
    partition_key_to_conditions=_oc_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_OC_GEN_PROMPT,
    build_polarity_instruction=_oc_polarity,
    task_name="object_counting",
    build_rule_scoring_prompt=_rule_score_prompt,
    bootstrap_ruleset=_oc_bootstrap,
    build_eval_prompt=_make_eval_prompt("<integer>"),
)
