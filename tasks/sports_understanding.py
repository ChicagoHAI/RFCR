"""tasks/sports_understanding.py — TaskSpec for the sports_understanding BBH task."""

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
)


# ─────────────────────────────────────────────────────────────────────────────
# Known athletes by sport
# ─────────────────────────────────────────────────────────────────────────────

_SPORTS_KNOWN = {
    "hockey": ["lindholm", "carlson", "ovechkin", "crosby", "mackinnon", "hedman",
               "stamkos", "pastrnak", "draisaitl", "point", "matthews"],
    "basketball": ["lebron", "curry", "durant", "harden", "giannis", "doncic",
                   "kawhi", "paul", "lillard", "tatum", "embiid", "jokic"],
    "football": ["mahomes", "brady", "rodgers", "wilson", "stafford", "burrow",
                 "jackson", "prescott", "murray", "herbert", "allen"],
    "baseball": ["trout", "betts", "judge", "arenado", "freeman", "devers",
                 "guerrero", "alvarez", "soto", "goldschmidt"],
    "soccer": ["messi", "ronaldo", "neymar", "mbappe", "salah", "lewandowski",
               "benzema", "de bruyne", "kante", "modric"],
    "tennis": ["federer", "djokovic", "nadal", "murray", "tsitsipas", "zverev",
               "medvedev", "alcaraz", "swiatek", "osaka"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_SPORTS_SCORING = """\
You are judging whether a sports sentence is plausible.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: YES or NO  ← FIRST LINE.\
REASONING: Think step by step. Identify the athlete, sport, and whether the action is possible.
"""

_SPORTS_SCORING_COT = """\
You are judging whether a sports sentence is plausible.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think step by step: identify the athlete, sport, and whether the action described is possible in that sport.

VERDICT: YES or NO  ← FIRST LINE.\
REASONING: Think step by step. Identify the athlete, sport, and whether the action is possible.
"""


def _sports_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _SPORTS_SCORING_COT if cot else _SPORTS_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


_SPORTS_RF = """\
You are judging whether a sports sentence is plausible.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Output format - use plain text only, no markdown bold or headers:
REASONING: identify the athlete, sport, and apply the cheatsheet rules to determine whether the action described is plausible.
VERDICT: YES

(Replace YES above with NO if the sentence is not plausible. VERDICT must be the last line, must be followed immediately by YES or NO on the same line, and must appear exactly once.)"""


def _sports_understanding_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _SPORTS_RF.format(cheatsheet=cs, question=item["input"])


def _parse_sports_understanding_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val.startswith("YES"):
                return "YES"
            if val.startswith("NO"):
                return "NO"
    return None


def _sports_partition_key(item: dict) -> tuple:
    t = item["input"].lower()
    sport = "other"
    for s, players in _SPORTS_KNOWN.items():
        if any(p in t for p in players) or s in t:
            sport = s
            break
    return (sport,)


def _sports_key_to_conds(key: tuple) -> list[str]:
    (sport,) = key
    return [f"sentence is about {sport}"]


def _sports_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = {
        "YES": ("POLARITY — FALSE NEGATIVE (model said NO, correct is YES):\n"
                "The action IS plausible but model said it isn't. "
                "Likely model doesn't know the player's sport or misidentified the action."),
        "NO":  ("POLARITY — FALSE POSITIVE (model said YES, correct is NO):\n"
                "The action is NOT plausible but model said it is. "
                "Likely model confused sports terminology (e.g. 'beat the buzzer' is basketball, not hockey)."),
    }.get(polarity.upper(), "Diagnose TYPE A (missing sport knowledge) or TYPE B (wrong action/sport mapping).")
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up. Show the next step.\n\n" + base
    return base


def _sports_identify_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(SP-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(SP-\w+)", reasoning)
    return m.group(1) if m else None


_SPORTS_INTRO = ("You are judging whether sports sentences are plausible.\n"
                 "Apply these rules in order (stop at the first match):")
_SPORTS_FOOTER = (
    "\nIf no rule applies, identify the athlete's sport and judge whether the described "
    "action is physically and legally possible in that sport.\n\n"
    "VERDICT: YES or NO\n"
    "REASONING: Begin with the principle applied (if any matched) or reason from the sport's rules."
)


def _sports_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="sports plausibility judgment",
        rule_prefix="SP",
        concepts="sport-specific actions and terminology, athlete-sport mapping, "
                 "what actions are possible in which sports",
        verdict_fmt="YES or NO",
        ruleset_intro=_SPORTS_INTRO,
        ruleset_footer=_SPORTS_FOOTER,
        section_title="SPORTS PLAUSIBILITY RULES",
    )


_SPORTS_GEN_PROMPT = (
    "You are an expert in sports rules helping a model that fails on sports plausibility questions.\n"
    "The task: decide if a sentence about an athlete's action is plausible given their sport.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. Which sport is mentioned? Is the athlete's name a reliable signal for their sport?\n"
    "  2. What action is described? (scoring / movement / equipment use / position / rule violation)\n"
    "  3. Is this action physically or rule-wise possible in that sport?\n"
    "  4. Did the model fail because it doesn't know the sport's rules (TYPE A),\n"
    "     or because it confused rules from a different sport (TYPE B)?\n\n"
    "STRUCTURAL VOCABULARY — use these exact terms in ACTIVATE IF:\n"
    "    sport: hockey / basketball / football / baseball / soccer / tennis / golf / swimming / other\n"
    "    action_type: scoring / movement / equipment / position / rule_violation\n"
    "    error: unknown_sport_rule (TYPE A) / cross_sport_confusion (TYPE B)\n"
    "    answer_is_yes: the sentence IS plausible in that sport\n"
    "    answer_is_no: the sentence is NOT plausible in that sport\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy athlete names or sentence text from the failures above.\n"
    "  • SUPPORT examples must describe the sport-rule pattern using generic placeholder names.\n"
    "  • The teaching note should apply to any athlete in this sport on any similar question.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [short title naming the sport and action type] ===\n"
    "FAILURE_TYPE: A (model doesn't know this sport's rule) or B (confuses rules from another sport)\n"
    "ACTIVATE IF:\n"
    "  - [sport and action_type from vocabulary above]\n"
    "DO NOT ACTIVATE IF: [case where the action is clearly possible/impossible and model is correct]\n"
    "COMMON WRONG MOVE: [which sport rule the model incorrectly applies]\n"
    "NEXT CHECK: [the specific sport rule to verify → sentence IS or IS NOT plausible → YES or NO]\n"
    "WHY THIS WORKS: [1-2 sentences on the sport-specific rule]\n"
    "SUPPORT:\n"
    "  • [athlete name + action sentence]  |  Answer: YES/NO  — [sport rule note]\n"
    "{retry_context}"
)

SPORTS_TASK = TaskSpec(
    build_scoring_prompt=_sports_scoring_prompt,
    is_correct=_yesno_correct,
    answer_label=_yesno_label,
    parse_verdict=_parse_yesno,
    extract_post_think=_extract_reasoning,
    partition_key=_sports_partition_key,
    partition_key_to_conditions=_sports_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_SPORTS_GEN_PROMPT,
    build_polarity_instruction=_sports_polarity,
    task_name="sports_understanding",
    build_rule_scoring_prompt=_rule_score_prompt,
    identify_triggered_rule=_sports_identify_rule,
    rule_id_regex=r"(SP-\w+)",
    bootstrap_ruleset=_sports_bootstrap,
    build_eval_prompt=_make_eval_prompt("YES or NO"),
)
