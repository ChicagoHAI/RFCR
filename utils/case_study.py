"""
utils/case_study.py — Structured case study record.

Each case study is stored as a rich structured record rather than a flat string.
The human-readable render() output is identical to the existing prompt format so
the scoring pipeline is unaffected.  The structured fields enable:

  - Routing at inference time (match activate_if against current problem features)
  - Per-case precision tracking (n_activations / n_fixes)
  - Machine-readable dedup (compare activate_if lists instead of raw text)
  - Budget-aware rendering (select top-K activated cases)

Field descriptions
------------------
title                : short descriptive label
activate_if          : IDENTIFY conditions — checklist that must ALL be true for
                       this case to fire (parsed from IDENTIFY: section)
do_not_activate_if   : boundary conditions — when NOT to fire (DOES NOT APPLY TO)
action               : what to conclude when activate_if is met (ACTION:)
next_check           : what to do after this fires — either "DONE: TRUE/FALSE" or
                       "PROCEED TO: <step>" for chained routing (NEXT_CHECK:)
common_wrong_move    : what the model typically does wrong here (COMMON_WRONG_MOVE:)
why_this_check_works : mathematical justification (WHY:)
support_examples     : parsed EXAMPLES — list of {e1, e2, answer, note}
    feature_signature    : one-line compact structural tag, e.g. "absorbing→general_L0"
                       (FEATURE_SIGNATURE:)
    routing_key          : optional machine-readable partition key. When set,
                       routed scoring exposes this case only to matching items.
    target_roadmap_aspect: which DT step / roadmap aspect this corrects (TARGET_STEP:)
creation_fix_rate    : fix_rate on the flush bin that produced this case study
historical_fix_rate  : running average updated across ablation/eval passes
n_activations        : total times this case study was matched at inference
n_fixes              : total correct predictions when this case study fired
raw_text             : original LLM output (preserved for fallback / debugging)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaseStudy:
    # Core identity
    title: str

    # Gate fields (structured from IDENTIFY / DOES NOT APPLY TO)
    activate_if: list[str] = field(default_factory=list)
    do_not_activate_if: list[str] = field(default_factory=list)

    # Decision fields
    action: str = ""
    next_check: str = ""           # "DONE: TRUE", "DONE: FALSE", or "PROCEED TO: STEP N"

    # Diagnostic fields
    common_wrong_move: str = ""
    why_this_check_works: str = ""

    # Evidence
    support_examples: list[dict] = field(default_factory=list)
    # Each dict: {e1: str, e2: str, answer: str, note: str}

    # Routing metadata
    feature_signature: str = ""
    routing_key: list = field(default_factory=list)
    target_roadmap_aspect: str = ""
    failure_type: str = ""   # "A" (missing knowledge) or "B" (wrong reasoning pattern)

    # Running statistics
    creation_fix_rate: float = 0.0
    historical_fix_rate: float = 0.0
    n_activations: int = 0
    n_fixes: int = 0

    # Original LLM text — preserved for fallback rendering and debugging
    raw_text: str = ""

    # Roadmap patch produced alongside this case study (from === ROADMAP PATCH === block).
    # ICR_partition applies this to cheatsheet.roadmap when the case study is accepted,
    # making both the example-level (case study) and rule-level (roadmap) corrections.
    roadmap_patch: str = ""

    # When the generator chose MODIFY instead of ADD NEW, this holds the title of
    # the existing case study to replace.  Empty string means ADD NEW (default).
    modification_target: str = ""

    # ------------------------------------------------------------------
    # Rendering — identical format to the existing prompt injection
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Produce the human-readable text for prompt injection.

        Renders in the standard structured case study format understood by the
        scoring model. Metadata fields (feature_signature, common_wrong_move, etc.)
        are NOT rendered into the prompt to preserve the existing token budget.
        """
        if not self.activate_if and not self.action and self.raw_text:
            # Unparsed case study — fall back to raw text
            return self.raw_text.strip()

        lines = [f"=== CASE STUDY: {self.title} ==="]

        if self.activate_if:
            lines.append("IDENTIFY:")
            for cond in self.activate_if:
                lines.append(f"  {cond.lstrip('- •').strip()}")

        if self.action:
            lines.append(f"ACTION: {self.action}")

        if self.why_this_check_works:
            lines.append(f"WHY: {self.why_this_check_works}")

        if self.support_examples:
            lines.append("EXAMPLES:")
            for ex in self.support_examples:
                e1   = ex.get("e1", "")
                e2   = ex.get("e2", "")
                ans  = ex.get("answer", "")
                note = ex.get("note", "")
                line = f"  • E1 = {e1}  |  E2 = {e2}  |  Answer: {ans}"
                if note:
                    line += f"  — {note}"
                lines.append(line)

        if self.do_not_activate_if:
            body = "; ".join(self.do_not_activate_if)
            lines.append(f"DOES NOT APPLY TO: {body}")

        return "\n".join(lines)

    def render_with_metadata(self) -> str:
        """
        Extended render that also includes routing metadata fields.

        Used for inspection / debugging. NOT used in prompt injection.
        """
        base = self.render()
        extras: list[str] = []
        if self.feature_signature:
            extras.append(f"FEATURE_SIGNATURE: {self.feature_signature}")
        if self.common_wrong_move:
            extras.append(f"COMMON_WRONG_MOVE: {self.common_wrong_move}")
        if self.target_roadmap_aspect:
            extras.append(f"TARGET_STEP: {self.target_roadmap_aspect}")
        if self.next_check:
            extras.append(f"NEXT_CHECK: {self.next_check}")
        if self.creation_fix_rate or self.historical_fix_rate:
            extras.append(
                f"STATS: creation_fix_rate={self.creation_fix_rate:.0%}  "
                f"historical_fix_rate={self.historical_fix_rate:.0%}  "
                f"n_activations={self.n_activations}  n_fixes={self.n_fixes}"
            )
        if extras:
            return base + "\n" + "\n".join(extras)
        return base

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "title":                self.title,
            "activate_if":          self.activate_if,
            "do_not_activate_if":   self.do_not_activate_if,
            "action":               self.action,
            "next_check":           self.next_check,
            "common_wrong_move":    self.common_wrong_move,
            "why_this_check_works": self.why_this_check_works,
            "support_examples":     self.support_examples,
            "feature_signature":    self.feature_signature,
            "routing_key":          self.routing_key,
            "target_roadmap_aspect": self.target_roadmap_aspect,
            "failure_type":         self.failure_type,
            "creation_fix_rate":    self.creation_fix_rate,
            "historical_fix_rate":  self.historical_fix_rate,
            "n_activations":        self.n_activations,
            "n_fixes":              self.n_fixes,
            "raw_text":             self.raw_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaseStudy":
        """Load from the JSON sidecar dict (all fields may be present)."""
        return cls(
            title=d.get("title", "untitled"),
            activate_if=d.get("activate_if", []),
            do_not_activate_if=d.get("do_not_activate_if", []),
            action=d.get("action", ""),
            next_check=d.get("next_check", ""),
            common_wrong_move=d.get("common_wrong_move", ""),
            why_this_check_works=d.get("why_this_check_works", ""),
            support_examples=d.get("support_examples", []),
            feature_signature=d.get("feature_signature", ""),
            routing_key=d.get("routing_key", []),
            target_roadmap_aspect=d.get("target_roadmap_aspect", ""),
            failure_type=d.get("failure_type", ""),
            creation_fix_rate=d.get("creation_fix_rate", 0.0),
            historical_fix_rate=d.get("historical_fix_rate", 0.0),
            n_activations=d.get("n_activations", 0),
            n_fixes=d.get("n_fixes", 0),
            raw_text=d.get("raw_text", ""),
        )

    @classmethod
    def from_text(cls, text: str) -> "CaseStudy":
        """
        Parse a case study from LLM-generated text.

        Handles both the legacy format (IDENTIFY/ACTION/WHY/EXAMPLES/DOES NOT APPLY TO)
        and the new extended format that also includes FEATURE_SIGNATURE, COMMON_WRONG_MOVE,
        TARGET_STEP, NEXT_CHECK.

        If parsing fails for any field, that field gets an empty default — the raw_text
        is always preserved so nothing is lost.
        """
        text = text.strip()
        title   = _extract_title_from_text(text)
        raw     = text

        # --- Trigger fields: new names take priority over old ---
        activate_if = (
            _parse_list_field(text, "ACTIVATE IF")
            or _parse_list_field(text, "IDENTIFY")
        )
        do_not_apply = (
            _parse_scalar_field(text, "DO NOT ACTIVATE IF")
            or _parse_scalar_field(text, "DOES NOT APPLY TO")
        )
        do_not_activate_if = [do_not_apply] if do_not_apply else []

        # --- Reasoning-move fields: new names take priority over old ---
        common_wrong_move = (
            _parse_scalar_field(text, "COMMON WRONG MOVE")
            or _parse_scalar_field(text, "COMMON_WRONG_MOVE")
        )
        next_check = (
            _parse_scalar_field(text, "NEXT CHECK")
            or _parse_scalar_field(text, "NEXT_CHECK")
            or _parse_scalar_field(text, "NEXT CHECK IF FIRES")
        )
        why = (
            _parse_scalar_field(text, "WHY THIS WORKS")
            or _parse_scalar_field(text, "WHY")
        )
        # SUPPORT is the new name for EXAMPLES; both parsed as structured examples
        support_examples = _parse_examples_from_field(text, "SUPPORT") or _parse_examples(text)

        # ACTION is the old classification verdict — kept for backward compat
        action = _parse_scalar_field(text, "ACTION")

        # Metadata fields
        feature_signature     = (
            _parse_scalar_field(text, "FEATURE_SIGNATURE")
            or _parse_scalar_field(text, "FEATURE SIGNATURE")
        )
        target_roadmap_aspect = (
            _parse_scalar_field(text, "TARGET_STEP")
            or _parse_scalar_field(text, "TARGET STEP")
        )
        failure_type_raw = (
            _parse_scalar_field(text, "FAILURE_TYPE")
            or _parse_scalar_field(text, "FAILURE TYPE")
        )
        # Normalise to "A" or "B"; default empty means unknown (treated as B)
        failure_type = ""
        if failure_type_raw:
            t = failure_type_raw.strip().upper()
            if t.startswith("A"):
                failure_type = "A"
            elif t.startswith("B"):
                failure_type = "B"

        # Legacy fallbacks: PATTERN/RULE map onto activate_if/action
        if not activate_if:
            pattern = _parse_scalar_field(text, "PATTERN")
            if pattern:
                activate_if = [pattern]
        if not action:
            rule = _parse_scalar_field(text, "RULE")
            if rule:
                action = rule

        return cls(
            title=title,
            activate_if=activate_if,
            do_not_activate_if=do_not_activate_if,
            action=action,
            next_check=next_check,
            common_wrong_move=common_wrong_move,
            why_this_check_works=why,
            support_examples=support_examples,
            feature_signature=feature_signature,
            target_roadmap_aspect=target_roadmap_aspect,
            failure_type=failure_type,
            raw_text=raw,
        )

    # ------------------------------------------------------------------
    # Completeness check
    # ------------------------------------------------------------------

    def is_complete(self) -> tuple[bool, list[str]]:
        """
        Return (True, []) if the case study has all required fields,
        otherwise (False, [list of missing field names]).

        Required fields:
          - activate_if   : at least one trigger condition
          - why_this_check_works : non-empty justification
          - support_examples    : at least one example
        """
        missing = []
        if not self.activate_if:
            missing.append("ACTIVATE IF / IDENTIFY")
        if not self.why_this_check_works:
            missing.append("WHY THIS WORKS / WHY")
        if not self.support_examples:
            missing.append("SUPPORT / EXAMPLES")
        return (len(missing) == 0), missing

    # ------------------------------------------------------------------
    # Stats helpers (called by loop.py / maintenance.py)
    # ------------------------------------------------------------------

    def record_activation(self, fixed: bool) -> None:
        """Update running stats when this case study is tested on an item."""
        self.n_activations += 1
        if fixed:
            self.n_fixes += 1
        total = self.n_activations
        if total > 0:
            # Blend creation_fix_rate with observed rate (equal weight each)
            observed = self.n_fixes / total
            self.historical_fix_rate = (self.creation_fix_rate + observed) / 2


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Headers that delimit fields within a case study block
_FIELD_HEADERS = [
    # New vocabulary (teaching-move format)
    "ACTIVATE IF", "DO NOT ACTIVATE IF",
    "COMMON WRONG MOVE", "NEXT CHECK", "WHY THIS WORKS", "SUPPORT",
    # Old vocabulary (classification format) — kept for backward compat
    "IDENTIFY", "ACTION", "WHY", "EXAMPLES", "DOES NOT APPLY TO",
    # Metadata fields (both formats)
    "FEATURE_SIGNATURE", "FEATURE SIGNATURE",
    "COMMON_WRONG_MOVE", "COMMON WRONG MOVE",
    "TARGET_STEP", "TARGET STEP",
    "NEXT_CHECK", "NEXT CHECK IF FIRES",
    # Legacy (ICR_naive / ICR_reasoning old format)
    "PATTERN", "RULE", "EXCEPTIONS",
]

_HEADER_RE = re.compile(
    r"^(" + "|".join(re.escape(h) for h in _FIELD_HEADERS) + r")\s*:(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_title_from_text(text: str) -> str:
    """Pull title from '=== CASE STUDY: <title> ===' header line."""
    m = re.search(r"===\s*CASE STUDY\s*:\s*(.+?)\s*===", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: first meaningful line
    for line in text.splitlines():
        line = line.strip().lstrip("=- ").rstrip("=- ").strip()
        if line and not line.lower().startswith("case study"):
            return line[:72] + ("..." if len(line) > 72 else "")
    return "untitled"


def _field_spans(text: str) -> dict[str, str]:
    """
    Extract all field values from a case study block.

    Returns a dict mapping canonical field name → field body text.
    """
    spans: dict[str, str] = {}
    matches = list(_HEADER_RE.finditer(text))
    for idx, m in enumerate(matches):
        header  = m.group(1).strip().upper()
        inline  = m.group(2).strip()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body    = text[m.end() : end_pos].strip()
        value   = (inline + "\n" + body).strip() if inline else body
        spans[header] = value
    return spans


def _parse_scalar_field(text: str, field_name: str) -> str:
    """Extract a single-value field by name."""
    spans = _field_spans(text)
    return spans.get(field_name.upper(), "").strip()


def _parse_list_field(text: str, field_name: str) -> list[str]:
    """
    Parse a field whose body is a list of bullet-pointed conditions.

    Accepts leading bullets: -, •, *, numbers.
    Returns a list of stripped condition strings (empty items removed).
    """
    body = _parse_scalar_field(text, field_name)
    if not body:
        return []
    items: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        # Strip bullet markers: -, •, *, or numbered lists (1. 2) etc.)
        line = re.sub(r"^[-•*]\s*", "", line).strip()      # bare dash/bullet
        line = re.sub(r"^\d+[.)]\s*", "", line).strip()    # numbered list
        if line:
            items.append(line)
    return items


def _parse_examples_from_field(text: str, field_name: str) -> list[dict]:
    """
    Parse bullet-pointed examples from a named field into list[{e1, e2, answer, note}].

    Handles formats:
      • E1 = ...  |  E2 = ...  |  Answer: TRUE/FALSE  — optional note
      • E1 = ...  |  E2 = ...  |  TRUE — optional note  (no "Answer:" prefix)
    """
    body = _parse_scalar_field(text, field_name)
    if not body:
        return []

    examples: list[dict] = []
    for line in body.splitlines():
        line = line.strip().lstrip("•-* ").strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue

        e1 = re.sub(r"^E1\s*=\s*", "", parts[0], flags=re.IGNORECASE).strip()
        e2 = re.sub(r"^E2\s*=\s*", "", parts[1], flags=re.IGNORECASE).strip()
        answer = ""
        note   = ""

        if len(parts) >= 3:
            ans_raw  = parts[2]
            ans_note = re.split(r"\s+[—–-]\s+", ans_raw, maxsplit=1)
            ans_str  = re.sub(r"^Answer\s*:\s*", "", ans_note[0], flags=re.IGNORECASE).strip()
            answer   = ans_str.upper() if ans_str.upper() in ("TRUE", "FALSE") else ans_str
            note     = ans_note[1].strip() if len(ans_note) > 1 else ""

        examples.append({"e1": e1, "e2": e2, "answer": answer, "note": note})

    return examples


def _parse_examples(text: str) -> list[dict]:
    """Parse EXAMPLES field (old format). Falls back to empty list."""
    return _parse_examples_from_field(text, "EXAMPLES")
