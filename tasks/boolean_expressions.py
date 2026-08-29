"""tasks/boolean_expressions.py — TaskSpec for the boolean_expressions BBH task."""

from __future__ import annotations

import re

from utils.task_spec import TaskSpec
from utils.data import is_true
from utils.parser import parse_response as _parse_response, normalize as _normalize
from ICR_rules.rules.rule import RuleSet


# ---------------------------------------------------------------------------
# Scoring prompts
# ---------------------------------------------------------------------------

_SCORING_PROMPT = """\
You are evaluating a boolean expression.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

Evaluate the following boolean expression and determine whether it is TRUE or FALSE.
The expression uses Python-style boolean operators: "not", "and", "or", and the literals True and False.
Parentheses are evaluated first.

Expression: {expression}

CRITICAL INSTRUCTION: The VERY FIRST LINE of your response must be either:
  VERDICT: TRUE
  VERDICT: FALSE
Do NOT write anything before this line. Not a single word. Start with VERDICT immediately.
After the verdict line you may show your step-by-step evaluation.
Even if you are uncertain, you MUST commit to a verdict.

Output format:
VERDICT: TRUE or FALSE  ← THIS MUST BE YOUR FIRST LINE, NO EXCEPTIONS.
REASONING: evaluate the expression step by step, following operator precedence (not > and > or).\
"""

_SCORING_PROMPT_COT_FIRST = """\
You are evaluating a boolean expression.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

Evaluate the following boolean expression and determine whether it is TRUE or FALSE.
The expression uses Python-style boolean operators: "not", "and", "or", and the literals True and False.
Parentheses are evaluated first. Precedence order (highest to lowest): not, and, or.

Expression: {expression}

Work through the evaluation step by step in your REASONING, then state the VERDICT.

CRITICAL INSTRUCTION: The VERY FIRST LINE of your response must be either:
  VERDICT: TRUE
  VERDICT: FALSE
Do NOT write anything before this line. Not a single word. Start with VERDICT immediately.
After the verdict line you may show your step-by-step evaluation.

Output format:
VERDICT: TRUE or FALSE  ← THIS MUST BE YOUR FIRST LINE, NO EXCEPTIONS.
REASONING: evaluate step by step from innermost parentheses outward, applying not > and > or precedence.\
"""


def _bbh_bool_build_scoring_prompt(cheatsheet_text: str, item: dict, cot_first: bool = False) -> str:
    template = _SCORING_PROMPT_COT_FIRST if cot_first else _SCORING_PROMPT
    return template.format(
        cheatsheet=cheatsheet_text,
        expression=item["input"],
    )


_SCORING_PROMPT_RF = """\
You are evaluating a boolean expression.

=== CHEATSHEET ===
{cheatsheet}
=== END CHEATSHEET ===

Evaluate the following boolean expression and determine whether it is TRUE or FALSE.
The expression uses Python-style boolean operators: "not", "and", "or", and the literals True and False.
Parentheses are evaluated first. Precedence order (highest to lowest): not, and, or.

Expression: {expression}

Output format - use plain text only, no markdown bold or headers:
REASONING: apply the cheatsheet rules, evaluating from innermost parentheses outward using not > and > or precedence.
VERDICT: TRUE

(Replace TRUE above with FALSE if the expression is false. VERDICT must be the last line, must be followed immediately by TRUE or FALSE on the same line, and must appear exactly once.)"""


def _boolean_expressions_scoring_prompt_rf(cs: str, item: dict, cot: bool = True) -> str:
    return _SCORING_PROMPT_RF.format(cheatsheet=cs, expression=item["input"])


def _parse_boolean_expressions_rf(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val.startswith("TRUE"):
                return "TRUE"
            if val.startswith("FALSE"):
                return "FALSE"
    return None


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def _bbh_bool_is_correct(predicted: str | None, item: dict) -> bool:
    if predicted is None:
        return False
    return (predicted == "TRUE") == is_true(item["answer"])


def _bbh_bool_answer_label(item: dict) -> str:
    return "TRUE" if is_true(item["answer"]) else "FALSE"


# ---------------------------------------------------------------------------
# Verdict / post-think parsing
# ---------------------------------------------------------------------------

def _bbh_bool_parse_verdict(content: str) -> str | None:
    return _parse_response(_normalize(content))["verdict"]


def _bbh_bool_extract_post_think(content: str) -> str:
    parsed = _parse_response(_normalize(content))
    return parsed.get("reasoning") or content.strip()


# ---------------------------------------------------------------------------
# Partition key — bucket by which operators appear
# ---------------------------------------------------------------------------

def _bbh_bool_partition_key(item: dict) -> tuple:
    expr = item["input"].lower()
    has_not = "not " in expr
    has_and = " and " in expr
    has_or  = " or " in expr
    return (has_not, has_and, has_or)


def _bbh_bool_partition_key_to_conditions(key: tuple) -> list[str]:
    has_not, has_and, has_or = key
    conds = []
    if has_not:
        conds.append("expression contains NOT operator")
    else:
        conds.append("expression does NOT contain NOT operator")
    if has_and:
        conds.append("expression contains AND operator")
    if has_or:
        conds.append("expression contains OR operator")
    if not has_and and not has_or:
        conds.append("expression uses only NOT (or bare literals)")
    return conds


# ---------------------------------------------------------------------------
# Failure display
# ---------------------------------------------------------------------------

def _bbh_bool_format_failure(item: dict) -> str:
    expected   = "TRUE" if is_true(item.get("answer", False)) else "FALSE"
    predicted  = item.get("predicted", "?")
    post_think = item.get("post_think", "").strip()

    block = (
        f"  Expression: {item.get('input', '?')}\n"
        f"  expected={expected}  predicted={predicted}\n"
        f"  Model's wrong reasoning:\n"
        f"    {post_think if post_think else '(not captured)'}"
    )

    exact = item.get("_oracle_exact", "")
    if exact:
        block += f"\n  Correct reference reasoning:\n    {exact}"

    return block


# ---------------------------------------------------------------------------
# Case study generation prompt
# ---------------------------------------------------------------------------

BBH_BOOLEAN_GENERATION_PROMPT = """\
You are an expert in boolean logic working on automated reasoning evaluation.

You are writing a TEACHING NOTE for a weaker reasoning model that keeps making \
the same mistake when evaluating boolean expressions. Your job is to diagnose WHY \
it fails and teach the exact fix — either a key piece of missing knowledge about \
operator precedence or evaluation rules, or a wrong/missing reasoning pattern.

=== EXISTING CASE STUDIES ===
{case_studies}
=== END CASE STUDIES ===

=== PATTERNS ALREADY COVERED — YOUR CASE STUDY MUST NOT RESTATE THESE ===
{already_covered}
Your new case study MUST address a gap NOT covered above.
=== END ALREADY COVERED ===

The following boolean expressions were ALL predicted INCORRECTLY by the weaker model.
The ground-truth verdict and the weaker model's wrong reasoning are shown.
Where available, a correct reference reasoning trace is shown for contrast.

=== FAILURES WITH INCORRECT MODEL REASONING ===

{failure_lines}

=== YOUR TASK ===

{polarity_instruction}

Step 0 — DIAGNOSE the failure type. Choose exactly one:
  TYPE A — MISSING KNOWLEDGE: The weaker model's reasoning strategy was reasonable
    but it lacks a key fact about operator precedence, evaluation order, or boolean laws.
    Example: model doesn't know "not" binds tighter than "and" which binds tighter than "or".
  TYPE B — WRONG/MISSING REASONING PATTERN: The weaker model has the relevant rules
    but applies them in the wrong order, skips a necessary step, or misreads parentheses.

Step 1 — For TYPE A: State the missing rule precisely in one sentence with an example.
  For TYPE B: Quote or paraphrase the exact wrong move. Name the correct move instead.

Step 2 — Find the CORRECT MOVE: the specific mechanical check that produces the right answer.
  Must be something the model can execute by direct inspection of the expression syntax.

Step 3 — Find the TRIGGER: the precise structural conditions that distinguish these
  expressions from superficially similar ones where the same mistake would not occur.
  Be narrow. Prefer a trigger that fires on 2–3 cases correctly over one that fires broadly.

  FEATURE VOCABULARY — use these exact terms in ACTIVATE IF conditions:
    has_not: expression contains "not"
    has_and: expression contains "and"
    has_or: expression contains "or"
    nested_not: "not not" appears in the expression
    paren_depth: maximum parenthesis nesting depth
    operand_count: number of True/False literals

Step 4 — Find the ANTI-TRIGGER: 1–2 structurally similar cases where the weaker model's
  approach is actually correct and this teaching note should NOT fire.

TRANSFERABILITY REQUIREMENT:
Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.
  • DO NOT copy or reproduce specific boolean expressions from the failures above.
  • SUPPORT examples must be freshly invented expressions that illustrate the same structural pattern.
  • The teaching note should be useful to any model on any similar expression — not just these items.

OUTPUT 1 — CASE STUDY (max 900 chars)
Write the teaching note in EXACTLY this format, with these exact field names:

=== CASE STUDY: [short title — name the missing rule (TYPE A) or the mistaken pattern (TYPE B)] ===
FAILURE_TYPE: A or B
ACTIVATE IF:
  - [condition 1 — one structural fact about the expression]
  - [condition 2 — ...]
  (All conditions must hold. If any is false, do not use this note.)
DO NOT ACTIVATE IF: [1 sentence — the closest structural case where the model's approach is correct]
COMMON WRONG MOVE: [1 sentence — TYPE A: "Does not apply [missing rule]..."; TYPE B: starts with a verb]
NEXT CHECK: [the one mechanical thing to verify instead — end with "If yes → TRUE. If no → FALSE." or similar]
WHY THIS WORKS: [1–2 sentence justification grounded in boolean algebra]
SUPPORT:
  • Expression: ...  |  Answer: TRUE/FALSE  — [brief note]
  • Expression: ...  |  Answer: TRUE/FALSE  — [brief note]
{retry_context}"""


# ---------------------------------------------------------------------------
# Polarity instruction
# ---------------------------------------------------------------------------

def _bbh_bool_polarity_instruction(polarity: str, failure_type: str, divergence_step: str) -> str:
    _p = polarity.strip().upper()
    if _p == "TRUE":
        base = (
            "POLARITY DIRECTIVE — FALSE NEGATIVE bin (model said FALSE, correct answer is TRUE):\n"
            "Prioritize TYPE A (MISSING KNOWLEDGE). The model likely failed to apply a boolean law "
            "that collapses the expression to TRUE — e.g., double negation, De Morgan, or short-circuit. "
            "Your case study should give a concrete trigger: IF [structural condition] THEN the "
            "expression evaluates to TRUE without full expansion."
        )
    elif _p == "FALSE":
        base = (
            "POLARITY DIRECTIVE — FALSE POSITIVE bin (model said TRUE, correct answer is FALSE):\n"
            "Prioritize TYPE B (WRONG REASONING PATTERN). The model likely applied the wrong "
            "operator precedence, missed a negation, or misread parentheses. Identify the exact "
            "wrong step and the correct evaluation order."
        )
    else:
        base = (
            "Diagnose whether these failures are TYPE A (missing boolean rule) or TYPE B "
            "(wrong evaluation order / misread structure), choosing the type that best explains "
            "the majority of cases."
        )
    if failure_type == "ABANDONMENT":
        base = (
            "STRATEGY — ABANDONMENT: model gave up instead of completing the evaluation.\n"
            "These are TYPE B failures. Show where the model stopped and the exact next step.\n\n"
        ) + base
    return base


# ---------------------------------------------------------------------------
# Rule patching (ICR_rules integration)
# ---------------------------------------------------------------------------

_BBH_RULE_ID_REGEX = r"(BR-\w+)"

_BBH_RULE_PATCH_PROMPT = """\
You are an expert in boolean logic improving a decision guide used by a weaker \
reasoning model to evaluate boolean expressions (TRUE/FALSE).

The guide applies a sequence of named rules. One specific rule is causing \
classification errors. Your job: propose the smallest possible patch that fixes \
the failing cases without breaking the correct ones.

=== TARGET RULE ===
Rule ID: {rule_id}
Current text:
{rule_text}

=== FULL RULE GUIDE (read-only context — do not patch other rules) ===
{cheatsheet_context}

=== FAILING CASES (this rule fired → wrong verdict) ===
{failure_lines}

=== CORRECT CASES THIS RULE HANDLES (do not break) ===
{correct_lines}

=== YOUR TASK ===
Choose exactly one patch type:

  TIGHTEN   — add one or two conditions so the rule no longer fires on the failing cases
  SPLIT     — replace with two rules: one for correct cases, one for what was incorrectly matched
  REPLACE   — rewrite the rule entirely with tighter conditions (keep same ID)
  ADD_GUARD — insert a new rule IMMEDIATELY BEFORE this one that catches the
               failing cases and emits the correct verdict (new ID ends in -G, e.g. BR-2-G)

Hard requirements:
  1. Use only features from the guide: has_not, has_and, has_or, nested_not,
     paren_depth, operand_count, operator precedence (not > and > or)
  2. Keep rule text concise — match existing rule style exactly
  3. Every new rule must end with →  TRUE or →  FALSE
  4. For SPLIT/ADD_GUARD: list rules in the order they should appear

Output EXACTLY this format (no extra text before or after):
PATCH_TYPE: <TIGHTEN|SPLIT|REPLACE|ADD_GUARD>
NEW_RULES:
  <rule_id>: <rule_text>
  [<rule_id_B>: <rule_text_B>]
REASONING: <why this patch fixes failures without breaking correct cases>
VERIFY: <for each failing case, state which new condition fails to match>
"""


def _bbh_build_rule_scoring_prompt(template_text: str, item: dict) -> str:
    """Render a BBH rule-set template with the boolean expression substituted."""
    try:
        from jinja2 import Template as _T
        rendered = _T(template_text).render(expression=item["input"])
    except Exception:
        rendered = template_text.replace("{{ expression }}", item["input"])
    if item["input"] not in rendered:
        rendered = rendered.rstrip() + f"\n\nExpression: {item['input']}\n"
    return rendered


def _bbh_identify_triggered_rule(reasoning: str) -> str | None:
    m = re.search(r"RULE CITED:\s*(BR-\w+)", reasoning)
    if m:
        return m.group(1)
    m = re.search(r"\b(BR-\w+)", reasoning)
    return m.group(1) if m else None


def _bbh_format_rule_failures(failures: list[dict], oracle: dict) -> str:
    lines = []
    for i, item in enumerate(failures[:10], 1):
        expr = item.get("input", "?")
        expected = "TRUE" if is_true(item.get("answer", False)) else "FALSE"
        predicted = item.get("predicted", item.get("verdict", "?"))
        lines.append(f"[{i}] Expression: {expr}")
        lines.append(f"    Expected: {expected}  Model predicted: {predicted}")
        reasoning = item.get("reasoning") or item.get("post_think") or ""
        if reasoning:
            snippet = reasoning[-400:].replace("\n", " ").strip()
            lines.append(f"    Wrong reasoning: ...{snippet}")
        oracle_key = (expr,)
        oracle_reasoning = oracle.get(oracle_key, "")
        if oracle_reasoning:
            snippet = oracle_reasoning[:400].replace("\n", " ").strip()
            lines.append(f"    Oracle (correct): {snippet}...")
        lines.append("")
    return "\n".join(lines)


def _bbh_format_rule_correct(correct_pool: list[dict]) -> str:
    lines = []
    for item in list(correct_pool)[:6]:
        expr = item.get("input", "?")
        answer = "TRUE" if is_true(item.get("answer", False)) else "FALSE"
        lines.append(f"  Expression: {expr}  |  Answer: {answer}")
    return "\n".join(lines) if lines else "  (none available)"


def _bbh_build_rule_patch_prompt(target_rule, rule_set, failures, correct_pool, oracle) -> str:
    return _BBH_RULE_PATCH_PROMPT.format(
        rule_id=target_rule.id,
        rule_text=target_rule.text.strip(),
        cheatsheet_context=rule_set.render_decision_guide(),
        failure_lines=_bbh_format_rule_failures(failures, oracle),
        correct_lines=_bbh_format_rule_correct(correct_pool),
    )


# ---------------------------------------------------------------------------
# Rule bootstrap — generate an initial RuleSet from failure examples
# ---------------------------------------------------------------------------

_BBH_BOOTSTRAP_PROMPT = """\
You are an expert in boolean logic. A weaker reasoning model keeps failing to \
evaluate boolean expressions correctly. The expressions use Python-style \
operators: "not", "and", "or", and literals True/False.
Operator precedence (highest to lowest): not > and > or. Parentheses override.

Here are {n} expressions the model predicted INCORRECTLY:

{failure_lines}

Write 6-10 named decision rules (BR-1, BR-2, ...) that capture the patterns \
causing these mistakes. Rules should cover things like precedence traps, \
double-negation, De Morgan equivalences, parenthesis pitfalls, and short-circuit evaluation.

Requirements:
  - Each rule must describe a specific structural condition AND give a verdict
  - Order rules from most specific (narrow conditions) to most general
  - End every rule with  →  TRUE  or  →  FALSE

Output ONLY the rules, one per line, in EXACTLY this format — no headers, no explanations:
BR-1: <condition description> →  <TRUE or FALSE>
BR-2: <condition description> →  <TRUE or FALSE>
...\
"""

_BBH_RULESET_INTRO = """\
You are evaluating a boolean expression using a step-by-step decision guide.
The expression uses Python-style operators: not, and, or, and literals True/False.

Apply the following rules in order. Stop at the FIRST rule whose condition matches.\
"""

_BBH_RULESET_FOOTER = """\

If no rule above applies, evaluate the expression directly using \
Python operator precedence: not (highest) > and > or (lowest). \
Parentheses always override precedence.

VERDICT: TRUE or FALSE
RULE CITED: <rule ID, e.g. BR-3> or NONE if no rule matched
REASONING: Step-by-step evaluation. You MUST start with the rule ID you are \
applying (e.g. "BR-4 applies: not binds tighter than and, so ..."). \
If no rule matched, start with "No rule matched; evaluating by precedence.".\
"""


def _bbh_bootstrap_ruleset(failures: list[dict], model: str, api_key: str):
    """Call an LLM to generate an initial BBH rule set from failure examples."""
    from utils.llm_client import call_llm
    from ICR_rules.rules.rule import Rule, Section, RuleSet, _infer_verdict

    failure_lines = "\n".join(
        f"  [{i}] {item.get('input', '?')}"
        f"  →  expected {'TRUE' if is_true(item.get('answer', False)) else 'FALSE'}"
        f", model said {item.get('predicted', '?')}"
        for i, item in enumerate(failures[:20], 1)
    )
    prompt = _BBH_BOOTSTRAP_PROMPT.format(n=min(len(failures), 20),
                                          failure_lines=failure_lines)

    response = call_llm(prompt, model=model, api_key=api_key,
                        max_tokens=800, temperature=0.3)

    raw_rules: list[tuple[str, str]] = []
    for line in response.content.splitlines():
        line = line.strip()
        m = re.match(r"(BR-\w+):\s*(.+)", line)
        if m:
            raw_rules.append((m.group(1), line))

    if not raw_rules:
        raw_rules = [
            ("BR-1", "BR-1: not binds tighter than and, which binds tighter than or →  TRUE"),
        ]

    rule_objects = [
        Rule(id=rid, section="main", text=rtext, verdict=_infer_verdict(rtext))
        for rid, rtext in raw_rules
    ]
    section = Section(
        name="main",
        title="BOOLEAN EVALUATION RULES",
        preamble="",
        rules=rule_objects,
        postamble="",
    )
    return RuleSet(
        intro=_BBH_RULESET_INTRO,
        sections=[section],
        footer=_BBH_RULESET_FOOTER,
        source_path="",
    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

BBH_BOOLEAN_TASK = TaskSpec(
    build_scoring_prompt=_bbh_bool_build_scoring_prompt,
    is_correct=_bbh_bool_is_correct,
    answer_label=_bbh_bool_answer_label,
    parse_verdict=_bbh_bool_parse_verdict,
    extract_post_think=_bbh_bool_extract_post_think,
    partition_key=_bbh_bool_partition_key,
    partition_key_to_conditions=_bbh_bool_partition_key_to_conditions,
    format_failure=_bbh_bool_format_failure,
    generation_prompt_template=BBH_BOOLEAN_GENERATION_PROMPT,
    build_polarity_instruction=_bbh_bool_polarity_instruction,
    task_name="bbh_boolean_expressions",
    build_rule_scoring_prompt=_bbh_build_rule_scoring_prompt,
    identify_triggered_rule=_bbh_identify_triggered_rule,
    build_rule_patch_prompt=_bbh_build_rule_patch_prompt,
    rule_id_regex=_BBH_RULE_ID_REGEX,
    bootstrap_ruleset=_bbh_bootstrap_ruleset,
)
