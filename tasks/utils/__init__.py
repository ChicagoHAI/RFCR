"""tasks/utils/__init__.py — Shared helpers for all BBH task modules.

Functions here were previously split between tasks/bbh_tasks.py (lines 24-268)
and tasks/bbh_tasks_ext.py (lines 29-140).  They are merged here (duplicates
de-duplicated) so individual task files can import from a single location.
"""

from __future__ import annotations

import re

from utils.task_spec import TaskSpec  # noqa: F401 — re-exported for convenience
from ICR_rules.rules.rule import RuleSet  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Eval-prompt factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_eval_prompt(verdict_format: str):
    """Return a verdict-only eval prompt builder for the given answer format."""
    def _eval(cs: str, item: dict) -> str:
        return (
            f"=== CHEATSHEET ===\n{cs}\n=== END CHEATSHEET ===\n\n"
            f"{item['input']}\n\n"
            f"Reply with ONLY the verdict — no explanation, no reasoning.\n"
            f"VERDICT: {verdict_format}"
        )
    return _eval


# ─────────────────────────────────────────────────────────────────────────────
# Low-level parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_yesno(content: str) -> str | None:
    # Handle "VERDICT: Yes", "**VERDICT:** Yes", "VERDICT: **Yes**"
    m = re.search(r"\*{0,2}VERDICT\*{0,2}:?\*{0,2}\s*\*{0,2}(YES|NO)\*{0,2}", content, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Models that follow "FIRST LINE" literally: bare Yes/No at head without VERDICT label
    m_head = re.search(r"^\s*\*{0,2}(YES|NO)\*{0,2}\b", content[:100], re.IGNORECASE)
    if m_head:
        return m_head.group(1).upper()
    # Last resort: bare Yes/No in tail of response
    m2 = re.search(r"\b(YES|NO)\b", content[-200:], re.IGNORECASE)
    return m2.group(1).upper() if m2 else None


def _parse_mc(content: str) -> str | None:
    # Handle "VERDICT: (A)", "**VERDICT:** (A)", "VERDICT: **(A)**"
    m = re.search(r"\*{0,2}VERDICT\*{0,2}:?\*{0,2}\s*\*{0,2}\(?([A-Z])\)?\*{0,2}", content, re.IGNORECASE)
    if m:
        return f"({m.group(1).upper()})"
    # Handle "FINAL ANSWER: (A)" or "FINAL ANSWER: A" (gemini-style)
    m_fa = re.search(r"FINAL\s+ANSWER\s*:\s*\*{0,2}\(?([A-Z])\)?\*{0,2}", content, re.IGNORECASE)
    if m_fa:
        return f"({m_fa.group(1).upper()})"
    # Models that follow "FIRST LINE" literally: bare (X) at head without VERDICT label
    m_head = re.search(r"^\s*\*{0,2}\(?([A-Z])\)?\*{0,2}\s*$", content[:100], re.IGNORECASE | re.MULTILINE)
    if m_head:
        return f"({m_head.group(1).upper()})"
    # Llama-style: "correct answer is **(X) ..." — require literal (X) parentheses
    m_ca = re.search(r"\bcorrect answer is[^\n]*\(([A-Z])\)", content, re.IGNORECASE)
    if m_ca:
        return f"({m_ca.group(1).upper()})"
    # Last resort: bare option letter in tail of response (extended to 500 chars)
    m2 = re.search(r"\(([A-Z])\)", content[-500:], re.IGNORECASE)
    return f"({m2.group(1).upper()})" if m2 else None


def _extract_reasoning(content: str) -> str:
    # Handle plain "REASONING:" and markdown-bold "**REASONING:**"
    m = re.search(r"\*{0,2}REASONING\*{0,2}:?\*{0,2}\s*(.*)", content, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Correctness / label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _yesno_correct(predicted: str | None, item: dict) -> bool:
    return predicted is not None and predicted.upper() == item["answer"].strip().upper()


def _yesno_label(item: dict) -> str:
    return item["answer"].strip().upper()


def _mc_correct(predicted: str | None, item: dict) -> bool:
    return predicted is not None and predicted.upper() == item["answer"].strip().upper()


def _mc_label(item: dict) -> str:
    return item["answer"].strip().upper()


# ─────────────────────────────────────────────────────────────────────────────
# Formal-fallacies / ext-task helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_valid_invalid(content: str) -> str | None:
    def _line_label(line: str) -> str | None:
        text = re.sub(r"[*_`]+", "", line).strip().lower()
        if not text:
            return None
        if "valid or invalid" in text and not re.search(r"\b(?:answer|verdict|therefore|conclusion|argument)\b", text):
            return None
        if re.search(r"\bnot\s+valid\b", text):
            return "invalid"
        m = re.search(r"\binvalid\b", text)
        if m:
            return "invalid"
        m = re.search(r"\bvalid\b", text)
        if m:
            return "valid"
        return None

    # Handle plain "VERDICT: valid" and markdown-bold "**VERDICT:** valid".
    # Check invalid before valid so "INVALID" cannot be partially consumed.
    m = re.search(
        r"\*{0,2}VERDICT\*{0,2}:?\*{0,2}\s*\*{0,2}(invalid|valid)\*{0,2}",
        content,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()

    # Common local-model fallback: the model ignores VERDICT but gives
    # "Final Answer: Invalid" or "Therefore, the argument is invalid" and then
    # continues with explanation, so a tiny tail-only search misses it.
    labelled = re.finditer(
        r"(?im)^\s*(?:#+\s*)?(?:final\s+answer|answer|final\s+verdict|verdict|conclusion)\s*:?\s*(.*)$",
        content,
    )
    for match in reversed(list(labelled)):
        label = _line_label(match.group(1))
        if label:
            return label

    conclusion_lines = [
        line
        for line in content.splitlines()
        if re.search(r"\b(?:therefore|thus|hence|conclusion|argument is|deductively)\b", line, re.I)
    ]
    for line in reversed(conclusion_lines):
        label = _line_label(line)
        if label:
            return label

    # Last resort: scan the response tail, but prefer the last explicit token.
    matches = list(re.finditer(r"\b(invalid|valid)\b", content[-1000:], re.IGNORECASE))
    return matches[-1].group(1).lower() if matches else None


def _valid_invalid_correct(predicted: str | None, item: dict) -> bool:
    return predicted is not None and predicted.lower() == item["answer"].strip().lower()


def _valid_invalid_label(item: dict) -> str:
    return item["answer"].strip().lower()


def _trivial_key(item: dict) -> tuple:
    return ("all",)


def _trivial_conds(key: tuple) -> list[str]:
    return []


def _generic_polarity(polarity: str, failure_type: str, divergence_step: str) -> str:
    base = {
        "YES":   "POLARITY — FALSE NEGATIVE: model said NO/invalid/wrong option but correct answer is YES/valid/correct option.",
        "NO":    "POLARITY — FALSE POSITIVE: model said YES/valid/correct option but the answer is NO/invalid/other option.",
        "TRUE":  "POLARITY — FALSE NEGATIVE: model output the wrong answer.",
        "FALSE": "POLARITY — FALSE POSITIVE: model output the wrong answer.",
    }.get(polarity.upper(), "Diagnose whether the model applied the wrong rule or misread the structure.")
    if failure_type == "ABANDONMENT":
        base = "STRATEGY — ABANDONMENT: model gave up or produced no verdict. Show the next step.\n\n" + base
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Rule-scoring prompt builder (all tasks use item["input"] as {{ question }})
# ─────────────────────────────────────────────────────────────────────────────

def _rule_score_prompt(template_text: str, item: dict) -> str:
    try:
        from jinja2 import Template as _T
        rendered = _T(template_text).render(question=item["input"])
    except Exception:
        rendered = template_text.replace("{{ question }}", item["input"])
    if item["input"][:40] not in rendered:
        rendered = rendered.rstrip() + f"\n\n{item['input']}\n"
    return rendered


# ─────────────────────────────────────────────────────────────────────────────
# Generation prompt factory
# ─────────────────────────────────────────────────────────────────────────────

def _gen_prompt(
    domain: str, task_desc: str,
    type_a: str, type_b: str,
    feature_vocab: str, verdict_fmt: str,
) -> str:
    """Return a generation prompt template (uses {roadmap}, {case_studies}, etc.)."""
    return (
        f"You are an expert in {domain} working on automated reasoning evaluation.\n\n"
        f"A weaker reasoning model keeps making the same mistake on: {task_desc}.\n"
        "Write a TEACHING NOTE diagnosing WHY it fails and teaching the exact fix.\n\n"
        "=== EXISTING CASE STUDIES ===\n{case_studies}\n=== END CASE STUDIES ===\n\n"
        "=== PATTERNS ALREADY COVERED ===\n{already_covered}\n"
        "Your case study MUST address a gap NOT covered above.\n"
        "=== END ALREADY COVERED ===\n\n"
        "=== FAILURES WITH INCORRECT MODEL REASONING ===\n{failure_lines}\n\n"
        "=== YOUR TASK ===\n{polarity_instruction}\n\n"
        "Step 0 — DIAGNOSE:\n"
        f"  TYPE A — MISSING KNOWLEDGE: {type_a}\n"
        f"  TYPE B — WRONG REASONING PATTERN: {type_b}\n\n"
        "Step 1 — State the missing fact (TYPE A) or wrong move (TYPE B) precisely.\n\n"
        "Step 2 — CORRECT MOVE: the specific check that gives the right answer.\n\n"
        "Step 3 — TRIGGER: precise structural conditions.\n"
        "  FEATURE VOCABULARY — use these exact terms in ACTIVATE IF:\n"
        f"{feature_vocab}\n\n"
        "Step 4 — ANTI-TRIGGER: 1-2 cases where the model's approach is correct.\n\n"
        "TRANSFERABILITY REQUIREMENT:\n"
        "Your case study must encode an ABSTRACT REASONING PATTERN — not memorize the specific failures above.\n"
        "  • DO NOT copy or reference specific input text, names, dates, or numbers from the failures above.\n"
        "  • SUPPORT examples must be freshly invented minimal examples that illustrate the pattern.\n"
        "  • The teaching note should be useful to any model on any similar question — not just these items.\n\n"
        "OUTPUT 1 — CASE STUDY (max 900 chars)\n"
        "=== CASE STUDY: [short title] ===\n"
        "FAILURE_TYPE: A or B\n"
        "ACTIVATE IF:\n"
        "  - [condition 1]\n"
        "DO NOT ACTIVATE IF: [closest case where model is correct]\n"
        "COMMON WRONG MOVE: [1 sentence]\n"
        f"NEXT CHECK: [what to verify → {verdict_fmt}]\n"
        "WHY THIS WORKS: [1-2 sentences]\n"
        "SUPPORT:\n"
        f"  • [example]  |  Answer: {verdict_fmt}  — [brief note]\n"
        "{retry_context}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap factory
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ruleset(
    failures: list[dict],
    model: str,
    api_key: str,
    *,
    task_desc: str,
    rule_prefix: str,
    concepts: str,
    verdict_fmt: str,
    ruleset_intro: str,
    ruleset_footer: str,
    section_title: str,
) -> RuleSet:
    from utils.llm_client import call_llm
    from ICR_rules.rules.rule import Rule, Section, RuleSet as _RS, _infer_verdict

    failure_lines = "\n".join(
        f"  [{i}] {it.get('input', '?')[:200].strip()}"
        f"\n      Expected: {it.get('answer','?').strip()}  Got: {it.get('predicted','?')}"
        + (f"\n      Wrong reasoning: {it.get('post_think','')[:300]}" if it.get("post_think") else "")
        + (f"\n      Correct reasoning: {it.get('reason','')[:300]}" if it.get("reason") else "")
        for i, it in enumerate(failures[:15], 1)
    )
    prompt = (
        f"You are an expert in {task_desc}. A weaker model keeps failing.\n"
        f"Key concepts: {concepts}\n\n"
        f"Incorrectly predicted items:\n{failure_lines}\n\n"
        f"Write 6-10 named rules ({rule_prefix}-1, {rule_prefix}-2, ...) capturing the error patterns.\n"
        "Requirements:\n"
        "  - Each rule: structural condition + verdict\n"
        "  - Order: most specific first\n"
        f"  - End every rule with  →  <verdict>  (format: {verdict_fmt})\n\n"
        f"Output ONLY the rules, one per line:\n"
        f"{rule_prefix}-1: <condition> →  <verdict>\n..."
    )
    response = call_llm(prompt, model=model, api_key=api_key, max_tokens=600, temperature=0.3)

    raw_rules: list[tuple[str, str]] = []
    for line in response.content.splitlines():
        line = line.strip()
        m = re.match(rf"({rule_prefix}-\w+):\s*(.+)", line)
        if m:
            raw_rules.append((m.group(1), line))

    if not raw_rules:
        raw_rules = [(f"{rule_prefix}-1", f"{rule_prefix}-1: fallback rule →  YES")]

    rule_objects = [
        Rule(id=rid, section="main", text=rtext, verdict=_infer_verdict(rtext))
        for rid, rtext in raw_rules
    ]
    section = Section(name="main", title=section_title,
                      preamble="", rules=rule_objects, postamble="")
    return _RS(intro=ruleset_intro, sections=[section],
               footer=ruleset_footer, source_path="")


# ─────────────────────────────────────────────────────────────────────────────
# Failure formatter
# ─────────────────────────────────────────────────────────────────────────────

def _format_failure(item: dict, max_input: int = 300) -> str:
    expected = item.get("answer", "?").strip()
    predicted = item.get("predicted", "?")
    reasoning = (item.get("post_think") or item.get("reasoning") or "").strip()
    block = (
        f"  Input: {item.get('input', '?')[:max_input]}\n"
        f"  expected={expected}  predicted={predicted}\n"
        f"  Model's wrong reasoning:\n"
        f"    {reasoning[:300] if reasoning else '(not captured)'}"
    )
    exact = item.get("_oracle_exact", "")
    if exact:
        block += f"\n  Correct reasoning:\n    {exact[:600]}"
    return block
