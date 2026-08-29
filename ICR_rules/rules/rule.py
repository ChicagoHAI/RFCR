"""
Rule, RulePatch, Section, RuleSet — structured representation of a neurico-style cheatsheet.

The cheatsheet is modelled as an ordered list of Sections, each containing
fixed scaffold text and a list of mutable Rule objects. Disabling or patching
a rule re-renders the cheatsheet without touching anything else.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple


DIVIDER = "-" * 45


@dataclass
class Rule:
    id: str                        # "TR-1", "FR-1", "M3", "FG-2", "CPLEMMA", etc.
    section: str                   # "step0", "step0b", "step2", "step3", "step4", "step5"
    text: str                      # full rule text as it appears (may be multi-line)
    verdict: Optional[str] = None  # "TRUE" | "FALSE" | None
    enabled: bool = True

    def disable(self) -> Rule:
        return replace(self, enabled=False)

    def enable(self) -> Rule:
        return replace(self, enabled=True)

    def with_text(self, new_text: str) -> Rule:
        return replace(self, text=new_text)


@dataclass
class RulePatch:
    target_rule_id: str                   # ID of the rule being patched
    patch_type: str                       # TIGHTEN | SPLIT | REPLACE | ADD_GUARD
    new_rules: List[Tuple[str, str]]      # [(new_id, new_text), ...]
    reasoning: str                        # LLM's explanation
    verify: str = ""                      # LLM's self-verification against failures
    bin_key: Optional[str] = None        # partition bin that triggered this patch
    bin_fix_rate: float = 0.0            # fraction of bin failures fixed after patch


@dataclass
class Section:
    name: str       # "STEP 0", "STEP 0B", "STEP 4", etc.
    title: str      # e.g. "STEP 4: STRUCTURAL RULES"
    preamble: str   # fixed text before the first rule
    rules: List[Rule]
    postamble: str  # fixed text after the last rule (usually empty)

    def render(self) -> str:
        parts = []
        parts.append(f"{DIVIDER}\n{self.title}\n{DIVIDER}")
        if self.preamble.strip():
            parts.append(self.preamble.rstrip())
        rule_lines = []
        for rule in self.rules:
            if rule.enabled:
                rule_lines.append(rule.text.rstrip())
        if rule_lines:
            parts.append("\n".join(rule_lines))
        if self.postamble.strip():
            parts.append(self.postamble.rstrip())
        return "\n".join(parts)


@dataclass
class RuleSet:
    intro: str             # jinja2 preamble + GROUND RULES + MANDATORY (before STEP 0)
    sections: List[Section]
    footer: str            # RESPONSE FORMAT + CASE PATTERNS + CRITICAL INSTRUCTION
    source_path: str = ""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Full cheatsheet text — usable as a jinja2 template (contains {{ equation1 }})."""
        parts = [self.intro.rstrip()]
        for section in self.sections:
            parts.append(section.render())
        parts.append(self.footer.rstrip())
        return "\n\n".join(p for p in parts if p.strip())

    def render_for_sair(self, equation1: str, equation2: str) -> str:
        """Render as a complete SAIR evaluation prompt for a specific problem."""
        template = self.render()
        return (template
                .replace("{{ equation1 }}", equation1)
                .replace("{{ equation2 }}", equation2))

    def render_decision_guide(self) -> str:
        """Strip the jinja2 preamble (first 3 lines). Returns the decision guide
        portion for use as {cheatsheet} in ICR's SCORING_PROMPT."""
        full = self.render()
        lines = full.split("\n")
        skip = {"You are a mathematician", "Your task is to determine", "Use the following decision guide"}
        start = 0
        skipped = 0
        for i, line in enumerate(lines):
            if any(line.strip().startswith(s) for s in skip):
                start = i + 1
                skipped += 1
            if skipped >= 3:
                break
        # Skip leading blank lines
        while start < len(lines) and not lines[start].strip():
            start += 1
        return "\n".join(lines[start:])

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def all_rules(self) -> List[Rule]:
        return [r for s in self.sections for r in s.rules]

    def enabled_rules(self) -> List[Rule]:
        return [r for r in self.all_rules() if r.enabled]

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        for rule in self.all_rules():
            if rule.id == rule_id:
                return rule
        return None

    def get_section(self, section_name: str) -> Optional[Section]:
        for s in self.sections:
            if s.name == section_name:
                return s
        return None

    def byte_size(self) -> int:
        return len(self.render().encode("utf-8"))

    def rule_ids(self) -> List[str]:
        return [r.id for r in self.all_rules()]

    # ------------------------------------------------------------------
    # Mutation (returns new RuleSet, never mutates in place)
    # ------------------------------------------------------------------

    def disable_rule(self, rule_id: str) -> RuleSet:
        new_sections = []
        for section in self.sections:
            new_rules = [r.disable() if r.id == rule_id else r for r in section.rules]
            new_sections.append(Section(section.name, section.title, section.preamble, new_rules, section.postamble))
        return RuleSet(self.intro, new_sections, self.footer, source_path="")

    def enable_rule(self, rule_id: str) -> RuleSet:
        new_sections = []
        for section in self.sections:
            new_rules = [r.enable() if r.id == rule_id else r for r in section.rules]
            new_sections.append(Section(section.name, section.title, section.preamble, new_rules, section.postamble))
        return RuleSet(self.intro, new_sections, self.footer, source_path="")

    def apply_patch(self, patch: RulePatch) -> RuleSet:
        """Apply a RulePatch. Returns a new RuleSet with the target rule replaced.
        Clears source_path so the scorer uses rendered text, not the original file."""
        new_sections = []
        for section in self.sections:
            new_rules: List[Rule] = []
            for rule in section.rules:
                if rule.id == patch.target_rule_id:
                    if patch.patch_type in ("TIGHTEN", "REPLACE"):
                        nid, ntext = patch.new_rules[0]
                        verdict = _infer_verdict(ntext)
                        new_rules.append(Rule(id=nid, section=rule.section, text=ntext, verdict=verdict))
                    elif patch.patch_type in ("SPLIT", "ADD_GUARD"):
                        for nid, ntext in patch.new_rules:
                            verdict = _infer_verdict(ntext)
                            new_rules.append(Rule(id=nid, section=rule.section, text=ntext, verdict=verdict))
                else:
                    new_rules.append(rule)
            new_sections.append(Section(section.name, section.title, section.preamble, new_rules, section.postamble))
        return RuleSet(self.intro, new_sections, self.footer, source_path="")

    def summary(self) -> str:
        enabled = sum(1 for r in self.all_rules() if r.enabled)
        total = len(self.all_rules())
        kb = self.byte_size() / 1024
        return f"RuleSet: {enabled}/{total} rules enabled, {kb:.1f} KB"


def _infer_verdict(rule_text: str) -> Optional[str]:
    if "\u2192  TRUE" in rule_text or "->  TRUE" in rule_text or "→  TRUE" in rule_text:
        return "TRUE"
    if "\u2192  FALSE" in rule_text or "->  FALSE" in rule_text or "→  FALSE" in rule_text:
        return "FALSE"
    return None
