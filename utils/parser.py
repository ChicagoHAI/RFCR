"""
parser.py — Parse structured fields from a raw model response.

Expected response format:
    VERDICT: TRUE | FALSE
    REASONING: <text>
    PROOF: <text if TRUE, else empty>
    COUNTEREXAMPLE: <text if FALSE, else empty>

Handles:
  - Inline content on the same line as a header ("PROOF: inline content here")
  - Back-references to REASONING instead of a new proof block
    ("PROOF: as given above" → proof = reasoning text)
  - Markdown bold/italic wrappers around headers ("**VERDICT:** TRUE")
"""

from __future__ import annotations

import re

SECTION_HEADERS: tuple[str, ...] = ("VERDICT", "REASONING", "PROOF", "COUNTEREXAMPLE")

_HEADER_BOUNDARY = re.compile(
    r"^(?:" + "|".join(SECTION_HEADERS) + r"):",
    re.MULTILINE | re.IGNORECASE,
)

_BACK_REFERENCE_PATTERNS = re.compile(
    r"""
    as\s+given\s+above
    | see\s+(?:the\s+)?(?:reasoning|above|proof\s+above)
    | (?:the\s+)?(?:above|preceding)\s+(?:reasoning|argument|analysis)\s+
      (?:is|constitutes|serves\s+as|provides)\s+(?:the\s+)?(?:a\s+)?(?:full\s+)?proof
    | reasoning\s+above\s+constitutes
    | as\s+(?:stated|shown|argued|demonstrated)\s+above
    | (?:full\s+)?proof\s+(?:is\s+)?(?:given|provided|contained)\s+(?:in\s+)?(?:the\s+)?(?:above|reasoning)
    | refer\s+to\s+(?:the\s+)?(?:above\s+)?reasoning
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MD_BOLD_RE = re.compile(
    # Matches all variants the model produces:
    #   **VERDICT:** TRUE   — bold with colon inside
    #   **VERDICT**: TRUE   — bold with colon outside
    #   **VERDICT: FALSE**  — whole token bolded
    #   *VERDICT:* TRUE     — single-star italic
    r"\*{1,2}(VERDICT|REASONING|PROOF|COUNTEREXAMPLE)\*{0,2}:?\*{0,2}:?",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Strip markdown bold/italic from section headers before parsing."""
    return _MD_BOLD_RE.sub(lambda m: m.group(1).upper() + ":", text)


def _extract_section(text: str, label: str) -> str:
    match = re.search(rf"^{label}:\s*(.*)$", text, re.MULTILINE | re.IGNORECASE)
    if not match:
        return ""
    inline = match.group(1).strip()
    tail_start = match.end()
    next_header = _HEADER_BOUNDARY.search(text, tail_start)
    tail = text[tail_start : next_header.start()] if next_header else text[tail_start:]
    tail = tail.strip()
    if inline and tail:
        return inline + "\n" + tail
    return inline or tail


def _resolve_proof(proof: str, reasoning: str) -> str:
    if not proof or _BACK_REFERENCE_PATTERNS.search(proof):
        return reasoning
    return proof


def parse_response(text: str) -> dict:
    """
    Extract VERDICT, REASONING, PROOF, and COUNTEREXAMPLE from *text*.

    Returns a dict with those four keys. VERDICT is "TRUE"/"FALSE" or None
    if the model did not produce a parseable verdict.
    """
    text = normalize(text)
    verdict_match = re.search(r"^VERDICT:\s*(TRUE|FALSE)", text, re.MULTILINE | re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    else:
        # Fallback: model wrote conversational CoT without a VERDICT: line.
        # Strategy: find the LAST clear concluding TRUE/FALSE statement in the
        # full response — the model's final conclusion is most authoritative.
        # Patterns ordered from most to least reliable.
        _CONCLUDE_RE = re.compile(
            r"\b(?:the\s+)?(?:answer|implication|verdict|result|conclusion)\s+is\s+(TRUE|FALSE)"
            r"|\bE1\s+(?:does\s+(?:not\s+)?)?implies?\s+E2[^.]*?\b(TRUE|FALSE)"
            r"|\btherefore[,\s]+(?:the\s+(?:answer|implication)\s+is\s+)?(TRUE|FALSE)"
            r"|\bso[,\s]+(?:the\s+(?:answer|implication)\s+is\s+)?(TRUE|FALSE)"
            r"|\b(TRUE|FALSE)\s*[.]\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        all_matches = list(_CONCLUDE_RE.finditer(text))
        if all_matches:
            # Take the last match — most likely the final conclusion
            last = all_matches[-1]
            verdict = next(g for g in last.groups() if g is not None).upper()
        else:
            verdict = None
    reasoning = _extract_section(text, "REASONING")
    raw_proof = _extract_section(text, "PROOF")
    return {
        "verdict":        verdict,
        "reasoning":      reasoning,
        "proof":          _resolve_proof(raw_proof, reasoning),
        "counterexample": _extract_section(text, "COUNTEREXAMPLE"),
    }


def compute_correct(parsed: dict, item: dict) -> bool | None:
    """Compare the model's verdict against ground-truth. Returns None if unparseable."""
    if parsed["verdict"] is None:
        return None
    return (parsed["verdict"] == "TRUE") == (str(item["answer"]).strip().upper() == "TRUE")


def split_case_studies(text: str) -> list[str]:
    """Split LLM output containing multiple case studies into individual strings."""
    # Lookahead for === (3 equals) so the delimiter itself stays in each part.
    parts = re.split(r"(?====\s*CASE STUDY:)", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]
