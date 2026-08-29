"""tasks/causal_judgement.py — TaskSpec for the causal_judgement BBH task."""

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
# Scoring prompts
# ─────────────────────────────────────────────────────────────────────────────

_CAUSAL_SCORING = """\
You are answering causal reasoning questions from the perspective of a typical person.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

VERDICT: YES or NO  ← FIRST LINE.\
REASONING: Think step by step and explain your causal reasoning.
"""

_CAUSAL_SCORING_COT = """\
You are answering causal reasoning questions from the perspective of a typical person.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Think through the causal structure step by step, then give your verdict.

VERDICT: YES or NO  ← FIRST LINE.\
REASONING: Think step by step and explain your causal reasoning.
"""


def _causal_scoring_prompt(cs: str, item: dict, cot: bool = False) -> str:
    t = _CAUSAL_SCORING_COT if cot else _CAUSAL_SCORING
    return t.format(cheatsheet=cs, question=item["input"])


_CAUSAL_RF = """\
You are answering causal reasoning questions from the perspective of a typical person.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

{question}

Answer the question by following the provided cheatsheet. Ensure that your response ends with VERDICT: YES or NO"""


def _causal_judgement_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _CAUSAL_RF.format(cheatsheet=cs, question=item["input"])


def _parse_causal_judgement_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val.startswith("YES"):
                return "YES"
            if val.startswith("NO"):
                return "NO"
    return None


_CJ_LABEL_PROMPT = """\
Analyze this causal reasoning scenario and identify the causal structure type.

SCENARIO:
{scenario}

Identify the PRIMARY causal reasoning pattern from the list below. Choose exactly one.
- overdetermination: two or more independent sufficient causes; candidate may be redundant
- preemption: candidate would have caused outcome but a faster independent cause acted first
- background_condition: candidate is a routine expected factor; a different abnormal factor is the real cause
- counterfactual_dependence: outcome depends specifically on candidate's action; counterfactuals are explicit
- double_prevention: candidate prevented something that would have prevented the outcome (two negations)
- joint_causation: multiple factors all required together; neither alone is sufficient
- proximate_vs_distal: question turns on whether immediate vs background/distal actor caused outcome
- intentional_causation: question turns on whether agent acted deliberately toward the outcome
- other: scenario does not fit the above categories

Output EXACTLY two lines:
CAUSAL_PATTERN: <pattern name>
REASON: <one sentence explanation>"""


def _build_cj_label_prompt(item: dict) -> str:
    return _CJ_LABEL_PROMPT.format(scenario=item.get("input", "").strip())


_CJ_PATTERNS = {
    "overdetermination", "preemption", "background_condition",
    "counterfactual_dependence", "double_prevention", "joint_causation",
    "proximate_vs_distal", "intentional_causation", "other",
}


def _parse_cj_label(text: str) -> str:
    for line in text.strip().splitlines():
        if line.upper().startswith("CAUSAL_PATTERN:"):
            label = line.split(":", 1)[1].strip().lower()
            if label in _CJ_PATTERNS:
                return label
            # fuzzy: check if any known pattern is a substring
            for p in _CJ_PATTERNS:
                if p in label:
                    return p
    return "other"


def _causal_partition_key(item: dict) -> tuple:
    # Prefer LLM semantic label injected by --pass1-partition
    label = item.get("semantic_label")
    if label and label != "(unknown)":
        return (label,)
    t = item["input"].lower()
    has_cf = any(w in t for w in ["would have", "could have", "had not", "hadn't", "if not"])
    has_od = ("both" in t or "each" in t) and ("sufficient" in t or "alone" in t or "independently" in t)
    has_pp = "prevent" in t and "prevent" in t[t.index("prevent") + 7:]
    return (has_cf, has_od or has_pp)


def _causal_key_to_conds(key: tuple) -> list[str]:
    if len(key) == 1:
        # Semantic label key
        return [f"causal pattern: {key[0]}"]
    has_cf, has_complex = key
    conds = []
    conds.append("scenario involves counterfactual reasoning" if has_cf
                 else "no explicit counterfactuals in scenario")
    if has_complex:
        conds.append("scenario involves overdetermination or double-prevention")
    return conds


def _causal_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = {
        "YES": ("POLARITY — FALSE NEGATIVE (model said NO, correct is YES):\n"
                "Model failed to recognise causation. Identify the missing causal principle."),
        "NO":  ("POLARITY — FALSE POSITIVE (model said YES, correct is NO):\n"
                "Model incorrectly attributed causation. Focus on overdetermination, "
                "preemption, double-prevention, or proximate vs distal cause."),
    }.get(polarity.upper(),
          "Diagnose whether failures are TYPE A (missing principle) or TYPE B (wrong application).")
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up. Show the next step.\n\n" + base
    return base


def _causal_identify_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(CJ-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(CJ-\w+)", reasoning)
    return m.group(1) if m else None


_CAUSAL_INTRO = ("You are answering causal judgment questions as a typical person.\n"
                 "Apply these rules in order (stop at the first match):")
_CAUSAL_FOOTER = (
    "\nIf no rule applies, reason from first principles: identify the proximate cause "
    "and consider whether the outcome depends counterfactually on the actor's action.\n\n"
    "VERDICT: YES or NO\n"
    "REASONING: Begin with the principle applied (if any matched) or explain from first principles."
)


def _causal_bootstrap(failures, model, api_key):
    return _bootstrap_ruleset(
        failures, model, api_key,
        task_desc="causal reasoning and judgment",
        rule_prefix="CJ",
        concepts="proximate vs distal causation, overdetermination, preemption, "
                 "double-prevention, counterfactual dependence, typical person's intuition",
        verdict_fmt="YES or NO",
        ruleset_intro=_CAUSAL_INTRO,
        ruleset_footer=_CAUSAL_FOOTER,
        section_title="CAUSAL JUDGMENT RULES",
    )


_CAUSAL_CONCRETE_BOOTSTRAP_PROMPT = """\
You are creating a cheat sheet for a language model to answer causal judgment questions.

The model answers from the perspective of a typical person.

Below are {n} questions the model answered WRONG:
{failure_lines}

Your task: write a set of named concrete scenario examples that teach the key causal \
reasoning patterns a typical person uses. Each example is a short named scenario \
(like "Two Wires", "Two Gardeners") with a clear causal structure, verdict, and explanation.

Cover these causal types (use the failures above to determine which matter most):
1. Overdetermination — two independent causes each sufficient alone
2. Preemption — one cause takes over before another can act
3. Double prevention — X prevents Y which would have prevented the outcome
4. Proximate vs distal causation — immediate vs background cause
5. Counterfactual dependence — outcome would not have occurred without this action
6. Joint sufficiency — two factors both required; neither alone caused the outcome

For each scenario, use this exact format:

=== [Causal Type]: [Scenario Name] ===
Scenario: <1-2 sentence concrete description>
Causal structure: <what role each actor/factor plays>
Verdict: YES / NO
Why a typical person says this: <1 sentence>
Apply when: <structural cue to recognise this pattern>

Write 4-6 scenarios. Choose concrete everyday situations. \
Do NOT reuse the failure examples above — invent fresh scenarios.
"""


def _causal_concrete_bootstrap(failures: list[dict], model: str, api_key: str) -> str:
    from utils.llm_client import call_llm
    failure_lines = "\n".join(
        f"  [{i}] {it.get('input', '?')[:250].strip()}"
        f"\n      Expected: {it.get('answer','?').strip()}  Got: {it.get('predicted','?')}"
        for i, it in enumerate(failures[:15], 1)
    )
    prompt = _CAUSAL_CONCRETE_BOOTSTRAP_PROMPT.format(
        n=min(len(failures), 15),
        failure_lines=failure_lines,
    )
    response = call_llm(prompt, model=model, api_key=api_key, max_tokens=1200, temperature=0.3)
    return response.content.strip()


_CAUSAL_CONCRETE_CS_PROMPT = """\
You are improving a cheat sheet used by a language model to answer causal judgment questions.

The model answers from the perspective of a typical person.

=== EXISTING CHEAT SHEET ===
{cheatsheet}
=== END CHEAT SHEET ===

Below are {n} questions the model answered WRONG in a specific failure cluster:
{failure_lines}

Your task: write ONE new named concrete scenario example (like "Two Wires", "Traffic Light") \
that teaches the causal reasoning pattern the model is missing in these failures.

Requirements:
- The scenario must be concrete and everyday (not abstract)
- It must be different from any scenarios already in the cheat sheet
- It must directly address the error pattern shown in the failures above
- It should generalise to novel scenarios, not just fix these specific items

Use this exact format:

=== [Causal Type]: [Scenario Name] ===
Scenario: <1-2 sentence concrete description>
Causal structure: <what role each actor/factor plays>
Verdict: YES / NO
Why a typical person says this: <1 sentence>
Apply when: <structural cue to recognise this pattern>
"""


def _causal_concrete_cs_gen(
    failures: list[dict],
    cheatsheet_text: str,
    model: str,
    api_key: str,
) -> str | None:
    from utils.llm_client import call_llm
    failure_lines = "\n".join(
        f"  [{i}] {it.get('input', '?')[:250].strip()}"
        f"\n      Expected: {it.get('answer','?').strip()}  Got: {it.get('predicted','?')}"
        + (f"\n      Wrong reasoning: {it.get('post_think','')[:200]}" if it.get("post_think") else "")
        for i, it in enumerate(failures[:8], 1)
    )
    prompt = _CAUSAL_CONCRETE_CS_PROMPT.format(
        n=min(len(failures), 8),
        cheatsheet=cheatsheet_text[:3000],
        failure_lines=failure_lines,
    )
    response = call_llm(prompt, model=model, api_key=api_key, max_tokens=500, temperature=0.4)
    text = response.content.strip()
    return text if text else None


_CAUSAL_GEN_PROMPT = (
    "You are an expert in causal reasoning helping a model that keeps failing on causal judgment questions.\n"
    "The model answers from the perspective of a typical person.\n\n"
    "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
    "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
    "Your case study MUST address a gap NOT covered above.\n"
    "=== END ALREADY COVERED ===\n\n"
    "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
    "=== YOUR TASK ===\n{polarity_instruction}\n\n"
    "DIAGNOSE the failures by answering:\n"
    "  1. What everyday story shape do these failures share?\n"
    "     Think in terms of concrete situations: two people both trying, one blocking another,\n"
    "     a backup that wasn't needed, a chain of events where something was prevented, etc.\n"
    "  2. What question is a typical person really asking when they judge causation here?\n"
    "  3. What intuition is the model missing — not as an abstract label, but as a plain feeling\n"
    "     a non-expert would have about who is really responsible?\n\n"
    "TRANSFERABILITY REQUIREMENT:\n"
    "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
    "  • DO NOT copy or reference specific input text, names, or scenario details from the failures above.\n"
    "  • SUPPORT examples must be freshly invented everyday situations that illustrate the pattern.\n"
    "  • The teaching note should be useful to any model on any similar question — not just these items.\n\n"
    "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
    "=== CASE STUDY: [Causal Pattern]: [Memorable Scenario Name] ===\n"
    "FAILURE_TYPE: A (model used wrong causal intuition) or B (right intuition, wrong actor)\n"
    "ACTIVATE IF:\n"
    "  - scenario feels like: [1 plain sentence — what shared story shape makes this case study relevant]\n"
    "  - the question is asking: [what the typical person is really judging — e.g. 'whether X's action made a difference']\n"
    "DO NOT ACTIVATE IF: [the superficially similar case where the model's usual reasoning is correct]\n"
    "COMMON WRONG MOVE: [what the model incorrectly concludes and why, in plain language]\n"
    "NEXT CHECK: [a plain question a typical person would ask to resolve this → YES or NO]\n"
    "WHY THIS WORKS: [1-2 sentences on the everyday intuition — avoid abstract causal theory labels]\n"
    "SUPPORT:\n"
    "  • [concrete everyday scenario]  |  Answer: YES/NO  — [brief plain-language note]\n"
    "{retry_context}"
)

CAUSAL_JUDGEMENT_TASK = TaskSpec(
    build_scoring_prompt=_causal_scoring_prompt,
    is_correct=_yesno_correct,
    answer_label=_yesno_label,
    parse_verdict=_parse_yesno,
    extract_post_think=_extract_reasoning,
    partition_key=_causal_partition_key,
    partition_key_to_conditions=_causal_key_to_conds,
    format_failure=_format_failure,
    generation_prompt_template=_CAUSAL_GEN_PROMPT,
    build_polarity_instruction=_causal_polarity,
    task_name="causal_judgement",
    build_scoring_prompt_rf=_causal_judgement_scoring_prompt_rf,
    build_label_prompt=_build_cj_label_prompt,
    parse_label=_parse_cj_label,
    build_rule_scoring_prompt=_rule_score_prompt,
    identify_triggered_rule=_causal_identify_rule,
    rule_id_regex=r"(CJ-\w+)",
    bootstrap_ruleset=_causal_bootstrap,
    bootstrap_cheatsheet_fn=_causal_concrete_bootstrap,
    concrete_cs_gen_fn=_causal_concrete_cs_gen,
    build_eval_prompt=_make_eval_prompt("YES or NO"),
    build_pass1_label_prompt=_build_cj_label_prompt,
    parse_pass1_label=_parse_cj_label,
    patch_domain="causal reasoning and everyday causal intuition",
    patch_verdict_format="→  YES or →  NO  (two spaces before YES/NO)",
    patch_rule_style="causal patterns a typical person would apply: overdetermination, "
                     "preemption, proximate vs distal cause, double prevention, "
                     "joint causation, counterfactual dependence, background condition",
)
