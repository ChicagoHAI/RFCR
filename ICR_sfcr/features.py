"""Lightweight feature tags for SF-CR formal-fallacies routing.

These heuristics are deliberately conservative.  They are not a full logic
parser; they provide stable activation evidence for routed validation and
per-item diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


FORMAL_FALLACY_TAGS = frozenset(
    {
        "universal_negative",
        "no_x_is_y",
        "all_x_are_y",
        "some_x_are_y",
        "complement_class",
        "non_y_class",
        "converse_quantifier_conversion",
        "universal_negative_complement_conversion",
        "exhaustive_partition_claim",
        "explicit_exhaustive_alternatives",
        "valid_syllogism",
        "categorical_syllogism",
        "predicate_swap",
        "illicit_conversion",
        "conditional_statement",
        "converse_inference",
        "inverse_inference",
        "modus_ponens",
        "modus_tollens",
        "contrapositive_valid",
        "chain_argument",
        "broken_chain",
        "disjunction",
    }
)


@dataclass
class FeatureResult:
    tags: set[str] = field(default_factory=set)
    evidence: dict[str, list[str]] = field(default_factory=dict)

    def add(self, tag: str, evidence: str) -> None:
        self.tags.add(tag)
        if evidence:
            self.evidence.setdefault(tag, []).append(evidence[:220])


def _text(obj: str | dict) -> str:
    if isinstance(obj, dict):
        return str(obj.get("input") or obj.get("question") or obj)
    return str(obj)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _find(pattern: str, text: str) -> list[str]:
    return [m.group(0) for m in re.finditer(pattern, text, flags=re.I)]


def extract_formal_fallacy_feature_result(item_or_text: str | dict) -> FeatureResult:
    """Extract formal-fallacies routing tags from an item or text string."""
    raw = _text(item_or_text)
    t = _norm(raw)
    out = FeatureResult()

    universal_negative_patterns = [
        r"\bno\s+[a-z][a-z0-9 '\-]{0,80}?\s+(?:is|are|was|were)\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}",
        r"\bnone of (?:the )?[a-z][a-z0-9 '\-]{1,80}\s+(?:is|are|was|were)\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}",
        r"\bevery\s+[a-z][a-z0-9 '\-]{1,80}\s+(?:is|are|was|were)\s+not\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}",
        r"\bwhoever\s+is\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}\s+is\s+not\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}",
        r"\bbeing\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}\s+is\s+sufficient\s+for\s+not\s+being\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}",
    ]
    for pat in universal_negative_patterns:
        for ev in _find(pat, t):
            out.add("universal_negative", ev)
            out.add("no_x_is_y", ev)

    if _find(r"\bevery\s+[a-z][a-z0-9 '\-]{1,80}\s+(?:is|are)\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}", t):
        out.add("all_x_are_y", "every/all categorical statement")
    if _find(r"\bsome\s+[a-z][a-z0-9 '\-]{1,80}\s+(?:is|are)\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}", t):
        out.add("some_x_are_y", "some categorical statement")

    complement_patterns = [
        r"\bnon[- ][a-z][a-z0-9'\-]+",
        r"\bnot\s+(?:a |an |the |being )?[a-z][a-z0-9 '\-]{1,80}",
        r"\bwhoever\s+is\s+not\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}",
        r"\bthings?\s+that\s+are\s+not\s+(?:a |an |the )?[a-z][a-z0-9 '\-]{1,80}",
        r"\boutside\s+(?:of\s+)?(?:the\s+)?[a-z][a-z0-9 '\-]{1,80}",
    ]
    for pat in complement_patterns:
        for ev in _find(pat, t):
            out.add("complement_class", ev)
            out.add("non_y_class", ev)

    universal_conclusion = bool(
        re.search(
            r"\b(?:therefore|hence|conclude|conclusion|it follows|in consequence|we may conclude|from this follows)\b"
            r".{0,220}\b(?:all|every|any|whoever)\b.{0,120}\b(?:not|non[- ]|outside)\b",
            t,
        )
        or re.search(r"\bwhoever\s+is\s+not\b.{0,120}\bis\s+(?:a |an |the )?", t)
        or re.search(r"\bif\s+not\b.{0,120}\bthen\b", t)
    )
    conclusion_cue = bool(
        re.search(
            r"\b(?:therefore|hence|conclude|conclusion|it follows|in consequence|we may conclude|from this follows|so, necessarily)\b",
            t,
        )
    )
    if "universal_negative" in out.tags and "complement_class" in out.tags and (universal_conclusion or conclusion_cue):
        out.add("universal_negative_complement_conversion", "universal negative plus complement/negated conclusion cue")
        out.add("converse_quantifier_conversion", "converted exclusion into complement-class or negated conclusion")
        out.add("illicit_conversion", "universal negative complement/negation conversion")

    exhaustive_patterns = [
        r"\beither\b.{0,120}\bor\b",
        r"\bexactly one of\b",
        r"\bonly two (?:kinds|types|classes|categories|alternatives)\b",
        r"\ball (?:objects|things|people|items|cases|entities)?\s*(?:are|is)?\s*either\b",
        r"\beverything\s+is\s+either\b",
        r"\bexhaust(?:s|ive|ively| all alternatives)\b",
    ]
    for pat in exhaustive_patterns:
        for ev in _find(pat, t):
            out.add("exhaustive_partition_claim", ev)
            out.add("explicit_exhaustive_alternatives", ev)

    if re.search(r"\b(?:if|then|only if|whenever|sufficient for|necessary for)\b", t):
        out.add("conditional_statement", "conditional cue")
    if re.search(r"\bif\b.{0,80}\bthen\b.{0,160}\bif\b.{0,80}\bthen\b", t):
        out.add("chain_argument", "multiple conditional links")
    if re.search(r"\b(?:or|either)\b", t):
        out.add("disjunction", "or/either cue")
    if re.search(r"\bnot\b.{0,80}\btherefore\b.{0,80}\bnot\b", t):
        out.add("inverse_inference", "negated antecedent/consequent cue")
    if re.search(r"\bnot\b.{0,80}\bif\b|\bif\b.{0,80}\bnot\b", t):
        out.add("contrapositive_valid", "contrapositive-like cue")
    if "universal_negative" in out.tags or "all_x_are_y" in out.tags or "some_x_are_y" in out.tags:
        out.add("categorical_syllogism", "categorical quantifier cue")
    if re.search(r"\bconverse\b|\bconverted?\b", t):
        out.add("converse_inference", "converse wording")
    if re.search(r"\bmodus ponens\b", t):
        out.add("modus_ponens", "modus ponens")
        out.add("valid_syllogism", "modus ponens")
    if re.search(r"\bmodus tollens\b", t):
        out.add("modus_tollens", "modus tollens")
        out.add("valid_syllogism", "modus tollens")

    return out


def extract_formal_fallacy_features(item_or_text: str | dict) -> set[str]:
    """Return just the tag set for callers that do not need evidence."""
    return extract_formal_fallacy_feature_result(item_or_text).tags


def infer_candidate_tags(rule: dict) -> tuple[set[str], set[str]]:
    """Infer tags for older rule files that do not yet carry tag metadata."""
    explicit_pos = set(rule.get("positive_tags") or [])
    explicit_neg = set(rule.get("negative_tags") or [])
    if explicit_pos or explicit_neg:
        return explicit_pos, explicit_neg

    text = " ".join(
        str(rule.get(k, ""))
        for k in ("id", "rule", "use_when", "do_not_use_when", "check", "feature_signature")
    ).lower()
    pos = set()
    neg = set()
    if (
        "universal negative" in text
        or "no x is y" in text
        or "no x are y" in text
        or "non-y" in text
        or "non y" in text
        or "complement" in text
    ):
        pos.add("universal_negative_complement_conversion")
    if "exhaustive" in text or "either x or y" in text or "either" in text:
        neg.add("exhaustive_partition_claim")
        neg.add("explicit_exhaustive_alternatives")
    if not neg and "universal_negative_complement_conversion" in pos:
        neg.update(
            {
                "exhaustive_partition_claim",
                "explicit_exhaustive_alternatives",
                "valid_syllogism",
                "modus_ponens",
                "modus_tollens",
            }
        )
    return pos, neg

# ---------------------------------------------------------------------------
# Coverage-expansion feature refinements
# ---------------------------------------------------------------------------

FORMAL_FALLACY_TAGS = FORMAL_FALLACY_TAGS | frozenset(
    {
        "all_non_y_claim",
        "third_category_possible",
        "explicit_two_category_partition",
        "explicit_exhaustive_partition",
        "valid_categorical_syllogism",
        "illicit_categorical_conversion",
        "premise_universal_negative",
        "premise_no_x_is_y",
        "premise_none_x_are_y",
        "conclusion_universal_complement_membership",
        "conclusion_all_non_y_are_x",
        "conclusion_if_not_y_then_x",
        "conclusion_non_y_implies_x",
        "no_explicit_exhaustive_partition",
        "invalid_complement_conversion_candidate",
        "conclusion_merely_restates_no_x_is_y",
        "conclusion_standard_valid_chain",
        "premise_conditional_exclusion",
        "conclusion_negated_membership",
        "conclusion_complement_membership_normalized",
        "valid_contraposition_like",
        "invalid_inverse_or_complement_like",
        "normalized_necessary_sufficient_relation",
        "normalized_none_of_this_relation",
        "normalized_not_every_relation",
        "direct_exhaustive_binary_conclusion",
        "none_of_this_reverse_membership",
        "weak_disjunction",
        "valid_disjunctive_syllogism",
        "exhaustive_case_split",
        "premise_conclusion_gap",
        "affirming_consequent",
        "denying_antecedent",
        "negated_disjunction",
        "not_c_or_k_form",
        "conditional_with_negated_disjunction",
        "multi_premise_chain",
        "disjunctive_chain",
        "valid_contrapositive_surface",
        "valid_de_morgan_contrapositive",
        "valid_common_consequent_disjunctive_chain",
        "neither_consequent_converse",
        "valid_existential_de_morgan_witness",
        "valid_neither_contrapositive",
        "valid_nobody_neither_necessary_chain",
    }
)

_BASE_extract_formal_fallacy_feature_result = extract_formal_fallacy_feature_result


def extract_formal_fallacy_feature_result(item_or_text: str | dict) -> FeatureResult:
    """Extract formal-fallacy tags, including coverage-expansion tags."""
    out = _BASE_extract_formal_fallacy_feature_result(item_or_text)
    raw = _text(item_or_text)
    t = _norm(raw)

    all_non_y = bool(
        re.search(
            r"\b(?:all|every|any|whoever|whatever|everything)\b.{0,120}\b(?:non[- ]|not\s+(?:a |an |the |being )?)",
            t,
        )
        or re.search(r"\bwhoever\s+is\s+not\b.{0,120}\bis\b", t)
    )
    if all_non_y:
        out.add("all_non_y_claim", "universal claim over complement/negated class")

    if "exhaustive_partition_claim" in out.tags:
        out.add("explicit_two_category_partition", "explicit exhaustive/two-category wording")

    if "universal_negative_complement_conversion" in out.tags and "exhaustive_partition_claim" not in out.tags:
        out.add("third_category_possible", "no explicit exhaustive partition; third category remains possible")
        out.add("illicit_categorical_conversion", "categorical exclusion converted too strongly")

    if (
        re.search(r"\ball\b.{0,80}\bare\b.{0,120}\ball\b.{0,80}\bare\b.{0,120}\btherefore\b.{0,120}\ball\b", t)
        or re.search(r"\bevery\b.{0,80}\bis\b.{0,120}\bevery\b.{0,80}\bis\b.{0,120}\btherefore\b.{0,120}\bevery\b", t)
    ):
        out.add("valid_categorical_syllogism", "all A are B; all B are C style chain")
        out.add("valid_syllogism", "valid categorical chain")

    return out


def extract_formal_fallacy_features(item_or_text: str | dict) -> set[str]:
    """Return just the tag set for callers that do not need evidence."""
    return extract_formal_fallacy_feature_result(item_or_text).tags


def assign_formal_fallacy_subtype_from_tags(tags: set[str] | list[str]) -> str:
    tags = set(tags)
    if "universal_negative_complement_conversion" in tags:
        return "universal_negative_complement"
    if "converse_inference" in tags and "conditional_statement" in tags:
        return "conditional_converse"
    if "inverse_inference" in tags and "conditional_statement" in tags:
        return "conditional_inverse"
    if "modus_ponens" in tags or "modus_tollens" in tags or "contrapositive_valid" in tags:
        return "valid_conditional"
    if "valid_categorical_syllogism" in tags or "categorical_syllogism" in tags:
        return "categorical_syllogism"
    if "some_x_are_y" in tags or "all_x_are_y" in tags:
        return "quantifier_scope"
    return "other"


def assign_formal_fallacy_subtype(item_or_text: str | dict) -> str:
    return assign_formal_fallacy_subtype_from_tags(
        extract_formal_fallacy_features(item_or_text)
    )

# ---------------------------------------------------------------------------
# Role-aware formal-fallacies refinements
# ---------------------------------------------------------------------------

_ROLE_BASE_extract_formal_fallacy_feature_result = extract_formal_fallacy_feature_result

_CONCLUSION_CUE_RE = re.compile(
    r"\b(?:therefore|hence|conclude|conclusion|it follows|in consequence|we may conclude|from this follows|so, necessarily)\b",
    flags=re.I,
)


def _premise_and_conclusion(text: str) -> tuple[str, str]:
    match = _CONCLUSION_CUE_RE.search(text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.end() :]


def _has_universal_negative(text: str) -> bool:
    return bool(
        re.search(r"\bno\s+[^.]{1,80}?\s+(?:is|are|was|were)\s+[^.]{1,80}", text)
        or re.search(r"\bnone of (?:the )?[^.]{1,80}?\s+(?:is|are|was|were)\s+[^.]{1,80}", text)
        or re.search(r"\bno member of [^.]{1,80}?\s+(?:is|are|was|were)\s+[^.]{1,80}", text)
        or re.search(r"\bbeing [^.]{1,80}\s+is\s+(?:sufficient|necessary)\s+for\s+not being\b", text)
        or re.search(r"\bnot being [^.]{1,80}\s+is\s+(?:sufficient|necessary)\s+for\s+not being\b", text)
        or re.search(r"\bnobody is neither\b", text)
    )


def _has_none_x_are_y(text: str) -> bool:
    return bool(re.search(r"\bnone of (?:the )?[^.]{1,80}?\s+(?:is|are|was|were)\s+[^.]{1,80}", text))


def _has_universal_complement_conclusion(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:all|every|any|anything|anyone|whoever|those who|everything)\b"
            r"[^.]{0,120}\b(?:non[- ]|not\s+(?:a |an |the |being )?|are not|is not)"
            r"[^.]{0,120}\b(?:is|are|then)\b",
            text,
        )
        or re.search(r"\bif\b[^.]{0,80}\bnot\b[^.]{0,80}\bthen\b", text)
        or re.search(r"\bwhatever\b[^.]{0,120}\bnone of this\b[^.]{0,160}\bis\b", text)
        or re.search(r"\bbeing [^.]{1,100}\s+is\s+necessary\s+for\s+not being\b", text)
    )


def _has_all_non_y_are_x(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:all|every|anything|everything|whoever|those who)\b"
            r"[^.]{0,120}\b(?:non[- ]|not\s+(?:a |an |the |being )?)"
            r"[^.]{0,120}\b(?:is|are)\b",
            text,
        )
        or re.search(r"\bwhatever\b[^.]{0,120}\bnone of this\b[^.]{0,160}\bis\b", text)
    )


def _has_if_not_y_then_x(text: str) -> bool:
    return bool(
        re.search(r"\bif\b[^.]{0,100}\bnot\b[^.]{0,100}\bthen\b", text)
        or re.search(r"\bbeing [^.]{1,100}\s+is\s+necessary\s+for\s+not being\b", text)
    )


def _has_exhaustive_partition(text: str) -> bool:
    return bool(
        re.search(r"\bevery(?:thing| object| animal| item| case| entity)?\b[^.]{0,120}\beither\b[^.]{0,120}\bor\b", text)
        or re.search(r"\bonly two (?:categories|classes|kinds|types|alternatives)\b", text)
        or re.search(r"\bexactly one of\b", text)
        or re.search(r"\bno third (?:category|class|kind|type|alternative)\b", text)
        or re.search(r"\ball [^.]{1,80}\bare either\b[^.]{0,120}\band not both\b", text)
    )


def _has_valid_categorical_chain(text: str) -> bool:
    return bool(
        re.search(
            r"\ball [^.]{1,60}\s+are\s+[^.]{1,60}\.\s*all [^.]{1,60}\s+are\s+[^.]{1,60}\.\s*(?:therefore|hence|so|we may conclude)",
            text,
        )
        or re.search(
            r"\bevery [^.]{1,60}\s+is\s+[^.]{1,60}\.\s*every [^.]{1,60}\s+is\s+[^.]{1,60}\.\s*(?:therefore|hence|so|we may conclude)",
            text,
        )
    )


def _has_conditional_converse(text: str) -> bool:
    return bool(re.search(r"\bif\b.{1,120}\bthen\b.{1,220}\btherefore\b.{0,120}\bif\b.{1,120}\bthen\b", text))


def _normalizes_necessary_sufficient(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:being|not being) [^.]{1,120}\s+is\s+(?:necessary|sufficient)\s+for\s+(?:being|not being)\b",
            text,
        )
    )


def _has_conditional_exclusion(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:being|not being) [^.]{1,120}\s+is\s+(?:necessary|sufficient)\s+for\s+not being\b",
            text,
        )
        or re.search(
            r"\bnot being [^.]{1,120}\s+is\s+(?:necessary|sufficient)\s+for\s+(?:being|not being)\b",
            text,
        )
        or re.search(r"\bnobody is neither\b", text)
        or re.search(r"\bneither\b[^.]{1,160}\bnor\b", text)
    )


def _has_negated_membership_conclusion(text: str) -> bool:
    return bool(
        re.search(r"\bit is not the case that\b", text)
        or re.search(r"\bnot every\b", text)
        or re.search(r"\bthere is somebody\b[^.]{0,160}\bnot\b", text)
        or re.search(r"\bthere is no\b", text)
        or re.search(r"\bwe may conclude:?\s*(?:it is false that|it is not the case that|not every)\b", text)
    )


def _has_normalized_complement_conclusion(text: str) -> bool:
    return bool(
        _has_universal_complement_conclusion(text)
        or _has_negated_membership_conclusion(text)
        or re.search(r"\bwhatever\b[^.]{0,180}\bnone of this\b", text)
        or re.search(r"\bnone of this:?\b[^.]{0,180}\bis\b", text)
        or re.search(r"\bnot being [^.]{1,120}\s+is\s+necessary\s+for\s+not being\b", text)
        or re.search(r"\bbeing [^.]{1,120}\s+is\s+necessary\s+for\s+not being\b", text)
    )


def _has_valid_contraposition_like(text: str) -> bool:
    return bool(
        re.search(r"\bif\b[^.]{0,100}\bthen\b[^.]{0,140}\btherefore\b[^.]{0,120}\bif not\b", text)
        or re.search(r"\bnot being [^.]{1,120}\s+is\s+necessary\s+for\s+not being\b", text)
    )


def _has_direct_exhaustive_binary_conclusion(premise: str, conclusion: str) -> bool:
    multi_premise_cues = [
        r"\bto start with\b",
        r"\bmoreover\b",
        r"\bfurthermore\b",
        r"\bin addition\b",
        r"\bnow,?\b",
        r"\ball this entails\b",
        r"\bevery [^.]{1,120}\b",
        r"\bmore than one premise\b",
    ]
    if any(re.search(pat, premise) for pat in multi_premise_cues):
        return False
    return bool(
        re.search(r"\bnobody is neither\b[^.]{1,160}\bnor\b", premise)
        and re.search(r"\bbeing [^.]{1,120}\s+is\s+necessary\s+for\s+not being\b", conclusion)
    )


def _has_none_of_this_reverse_membership(premise: str, conclusion: str) -> bool:
    return bool(
        re.search(r"\bevery [^.]{1,120}\s+is\s+neither\b[^.]{1,180}\bnor\b", premise)
        and re.search(r"\bwhatever\b[^.]{0,120}\bnone of this\b[^.]{0,200}\bis\b", conclusion)
    )


def _has_weak_disjunction_gap(premise: str, conclusion: str) -> bool:
    return bool(
        (
            re.search(r"\b(?:not [^.]{1,80}\s+or\s+not [^.]{1,80}|[^.]{1,80}\s+or\s+[^.]{1,80})", premise)
            and re.search(r"\b(?:therefore|it is not the case|we may conclude|not being|not every)\b", conclusion)
        )
        or (
            re.search(r"\b(?:either\b[^.]{1,120}\bor\b|[^.]{1,80}\s+or\s+[^.]{1,80})", premise)
            and conclusion.strip()
            and not re.search(r"\bor\b|\beither\b", conclusion)
        )
        or re.search(r"\b[a-z]\s+or\s+[a-z]\s*\.?\s*(?:therefore|so|hence)\s+[a-z]\b", premise + " " + conclusion)
    )


def _has_valid_contrapositive_surface(premise: str, conclusion: str) -> bool:
    return bool(
        re.search(r"\bnot being [^.]{1,120}\s+is\s+sufficient\s+for\s+not being\b", premise)
        and (
            re.search(r"\beveryone who is\b[^.]{1,160}\bis\b", conclusion)
            or re.search(r"\bwhoever is\b[^.]{1,160}\bis\b", conclusion)
            or re.search(r"\bnot [^.]{1,80}\b[^.]{0,80}\btherefore\b[^.]{0,80}\bnot\b", premise + " " + conclusion)
        )
    )


def _has_valid_de_morgan_contrapositive(premise: str, conclusion: str) -> bool:
    return bool(
        (
            re.search(r"\bnot\b[^.]{1,100}\bor\s+not\b", premise)
            and re.search(r"\bevery\b[^.]{1,120}\bis\b[^.]{1,120}\band\b", premise)
            and re.search(r"\bit is not the case that\b|\bnot being\b|\bis not\b", conclusion)
        )
        or (
            re.search(r"\bsome\b[^.]{1,160}\bnot\b[^.]{1,100}\bor\s+not\b", premise)
            and re.search(r"\b(?:every|being)\b[^.]{1,160}\b(?:sufficient for|is)\b", premise)
            and re.search(r"\bnot every\b", conclusion)
        )
    )


def _has_valid_common_consequent_disjunctive_chain(premise: str, conclusion: str) -> bool:
    return bool(
        re.search(r"\bsufficient for not being\b", premise)
        and re.search(r"\bsufficient for not being\b", premise[re.search(r"\bsufficient for not being\b", premise).end():])
        and re.search(r"\bevery\b[^.]{1,180}\bor\b", premise)
        and re.search(r"\bsufficient for not being\b", conclusion)
    )


def _has_valid_existential_de_morgan_witness(premise: str, conclusion: str) -> bool:
    return bool(
        re.search(r"\bsome\b[^.]{1,160}\bis\b", premise)
        and re.search(r"\beveryone who is both\b[^.]{1,180}\bis not\b", premise)
        and re.search(r"\bsome\b[^.]{1,180}\bnot\b[^.]{1,120}\bor\s+not\b", conclusion)
    )


def _has_negated_disjunction_converse(premise: str, conclusion: str) -> bool:
    return bool(
        (
            re.search(r"\bno [^.]{1,100}\s+is\s+[^.]{1,100}\s+or\s+[^.]{1,100}", premise)
            or re.search(r"\bif [^.]{1,100}\bthen\b[^.]{0,80}\bnot\s*\([^)]{1,120}\bor[^)]{1,120}\)", premise)
            or re.search(r"\bnot\s*\([^)]{1,120}\bor[^)]{1,120}\)", premise)
            or re.search(r"\bevery [^.]{1,120}\s+is\s+neither\b[^.]{1,180}\bnor\b", premise)
        )
        and re.search(r"\bit is (?:false|not the case) that\b|\btherefore\b[^.]{0,120}\bnot\b", conclusion)
    )


def _has_neither_consequent_converse(premise: str, conclusion: str) -> bool:
    return bool(
        re.search(r"\bevery [^.]{1,120}\s+is\s+neither\b[^.]{1,180}\bnor\b", premise)
        and len(re.findall(r"\b(?:is|are|was|were)\s+not\b|\bit is false that\b|\bit is not the case that\b", premise)) >= 2
        and re.search(r"\b(?:it is false that|it is not the case that|not being|is not)\b", conclusion)
    )


def _has_valid_neither_contrapositive(premise: str, conclusion: str) -> bool:
    return bool(
        re.search(r"\bevery [^.]{1,120}\s+is\s+neither\b[^.]{1,180}\bnor\b", premise)
        and len(re.findall(r"\b(?:is|are|was|were)\s+not\b|\bit is false that\b|\bit is not the case that\b", premise)) < 2
        and re.search(r"\b(?:it is false that|it is not the case that|not being|is not)\b", conclusion)
    )


def _has_valid_nobody_neither_necessary_chain(premise: str, conclusion: str) -> bool:
    return bool(
        re.search(r"\bnobody is neither\b[^.]{1,180}\bnor\b", premise)
        and re.search(r"\beveryone who is not\b[^.]{1,180}\bis not both\b", premise)
        and re.search(r"\bbeing [^.]{1,120}\s+is\s+necessary\s+for\s+not being\b", premise)
        and re.search(r"\bnobody is neither\b[^.]{1,180}\bnor\b", conclusion)
    )


def _has_multi_premise_chain(text: str) -> bool:
    cues = sum(
        1
        for pat in (
            r"\bfirst(?: of all| premise)?\b",
            r"\bsecond(?: premise)?\b",
            r"\bto start with\b",
            r"\bnext\b",
            r"\bmoreover\b",
            r"\bnow\b",
            r"\ball this entails\b",
        )
        if re.search(pat, text)
    )
    return cues >= 3 and bool(re.search(r"\b(?:or|either|necessary|sufficient|if|then)\b", text))


def extract_formal_fallacy_feature_result(item_or_text: str | dict) -> FeatureResult:
    """Extract formal-fallacy tags with role-aware premise/conclusion evidence."""
    out = _ROLE_BASE_extract_formal_fallacy_feature_result(item_or_text)
    raw = _text(item_or_text)
    t = _norm(raw)
    premise, conclusion = _premise_and_conclusion(t)
    conclusion_text = conclusion or t

    if _has_universal_negative(premise):
        out.add("premise_universal_negative", premise[:220])
        out.add("premise_no_x_is_y", premise[:220])
    if _has_none_x_are_y(premise):
        out.add("premise_none_x_are_y", premise[:220])

    if _has_universal_complement_conclusion(conclusion_text):
        out.add("conclusion_universal_complement_membership", conclusion_text[:220])
        out.add("conclusion_non_y_implies_x", conclusion_text[:220])
    if _has_all_non_y_are_x(conclusion_text):
        out.add("conclusion_all_non_y_are_x", conclusion_text[:220])
    if _has_if_not_y_then_x(conclusion_text):
        out.add("conclusion_if_not_y_then_x", conclusion_text[:220])

    if _has_exhaustive_partition(t) or "exhaustive_partition_claim" in out.tags:
        out.add("explicit_exhaustive_partition", "explicit exhaustive partition")
        out.add("explicit_two_category_partition", "explicit exhaustive/two-category wording")
    else:
        out.add("no_explicit_exhaustive_partition", "no explicit exhaustive partition")
        out.add("third_category_possible", "no explicit exhaustive partition; third category remains possible")

    if _has_valid_categorical_chain(t):
        out.add("valid_categorical_syllogism", "all/every categorical chain")
        out.add("valid_syllogism", "all/every categorical chain")
        out.add("conclusion_standard_valid_chain", "standard categorical chain conclusion")

    if (
        "premise_universal_negative" in out.tags
        and "conclusion_universal_complement_membership" in out.tags
        and "explicit_exhaustive_partition" not in out.tags
    ):
        out.add("universal_negative_complement_conversion", "role-aware universal negative plus complement conclusion")
        out.add("invalid_complement_conversion_candidate", "premise No X is Y; conclusion all/not-Y implies X")
        out.add("illicit_categorical_conversion", "role-aware complement conversion")

    if (
        "premise_universal_negative" in out.tags
        and "conclusion_universal_complement_membership" not in out.tags
        and re.search(r"\b(?:no|not)\b", conclusion_text)
    ):
        out.add("conclusion_merely_restates_no_x_is_y", conclusion_text[:220])

    if _has_conditional_converse(t):
        out.add("conditional_statement", "if/then conditional")
        out.add("converse_inference", "if A then B; therefore if B then A")

    if _normalizes_necessary_sufficient(t):
        out.add("normalized_necessary_sufficient_relation", "necessary/sufficient relation")
    if re.search(r"\bnone of this\b", t):
        out.add("normalized_none_of_this_relation", "none-of-this complement relation")
    if re.search(r"\bnot every\b", t):
        out.add("normalized_not_every_relation", "not-every existential negation")

    if _has_conditional_exclusion(premise):
        out.add("premise_conditional_exclusion", premise[:220])
    if _has_negated_membership_conclusion(conclusion_text):
        out.add("conclusion_negated_membership", conclusion_text[:220])
    if _has_normalized_complement_conclusion(conclusion_text):
        out.add("conclusion_complement_membership_normalized", conclusion_text[:220])
    if _has_valid_contraposition_like(t):
        out.add("valid_contraposition_like", "contraposition-like normalized form")
    if _has_direct_exhaustive_binary_conclusion(premise, conclusion_text):
        out.add("direct_exhaustive_binary_conclusion", "direct nobody-neither A/B entails not-B -> A")
        out.add("valid_contraposition_like", "direct exhaustive binary implication")
    if _has_none_of_this_reverse_membership(premise, conclusion_text):
        out.add("none_of_this_reverse_membership", "every X is neither A nor B; therefore every neither-A-nor-B is X")
        out.add("invalid_inverse_or_complement_like", "none-of-this reverse membership")
        out.add("invalid_complement_conversion_candidate", "none-of-this reverse membership")
    if _has_weak_disjunction_gap(premise, conclusion_text):
        out.add("weak_disjunction", "disjunctive premise does not force the conclusion")
        out.add("premise_conclusion_gap", "premise remains compatible with alternatives")
    if _has_valid_contrapositive_surface(premise, conclusion_text):
        out.add("contrapositive_valid", "not-A sufficient for not-B licenses B -> A")
        out.add("valid_contrapositive_surface", "not-A sufficient for not-B licenses B -> A")
        out.add("modus_tollens", "contrapositive/modus-tollens surface form")
    if _has_valid_de_morgan_contrapositive(premise, conclusion_text):
        out.add("contrapositive_valid", "not-X or not-Y plus Z -> X and Y licenses not Z")
        out.add("valid_de_morgan_contrapositive", "De Morgan contrapositive")
        out.add("modus_tollens", "De Morgan contrapositive/modus tollens")
    if _has_valid_common_consequent_disjunctive_chain(premise, conclusion_text):
        out.add("valid_syllogism", "disjunctive branches share the same consequent")
        out.add("valid_disjunctive_syllogism", "disjunctive branches share the same consequent")
        out.add("valid_common_consequent_disjunctive_chain", "A -> C, B -> C, X -> A or B entails X -> C")
    if _has_valid_existential_de_morgan_witness(premise, conclusion_text):
        out.add("valid_syllogism", "existential witness plus not-both universal licenses not-A or not-B")
        out.add("valid_disjunctive_syllogism", "existential De Morgan witness")
        out.add("valid_existential_de_morgan_witness", "some X is P; all A and B are not P; therefore some X is not A or not B")
    if _has_negated_disjunction_converse(premise, conclusion_text):
        out.add("negated_disjunction", "negated disjunction or no-H-is-C-or-K structure")
        out.add("conditional_with_negated_disjunction", "conditional with negated disjunction")
        out.add("not_c_or_k_form", "not C and not K / not(C or K) form")
        out.add("converse_inference", "uses the reverse direction of a negated-disjunction conditional")
    if _has_neither_consequent_converse(premise, conclusion_text):
        out.add("neither_consequent_converse", "Z -> neither A nor B; not A and not B used to infer not Z")
        out.add("negated_disjunction", "neither-consequent converse")
        out.add("conditional_with_negated_disjunction", "neither-consequent conditional")
        out.add("not_c_or_k_form", "neither A nor B consequent")
        out.add("converse_inference", "uses the reverse direction of a neither-consequent conditional")
    if _has_valid_neither_contrapositive(premise, conclusion_text):
        out.add("contrapositive_valid", "Z -> neither A nor B plus A/B licenses not Z")
        out.add("valid_neither_contrapositive", "neither-consequent contrapositive")
        out.add("modus_tollens", "neither-consequent modus tollens")
    if _has_valid_nobody_neither_necessary_chain(premise, conclusion_text):
        out.add("valid_syllogism", "nobody-neither plus not-both and necessary-condition chain")
        out.add("valid_nobody_neither_necessary_chain", "valid nobody-neither necessary-condition chain")
        out.add("contrapositive_valid", "valid proof by contradiction over neither/not-both chain")
    if _has_multi_premise_chain(t):
        out.add("multi_premise_chain", "several premise cues in one argument")
        out.add("chain_argument", "multi-premise chain")
        if "disjunction" in out.tags:
            out.add("disjunctive_chain", "multi-premise chain with disjunction")

    if (
        "premise_conditional_exclusion" in out.tags
        and "conclusion_complement_membership_normalized" in out.tags
        and "explicit_exhaustive_partition" not in out.tags
        and "valid_categorical_syllogism" not in out.tags
    ):
        out.add("invalid_inverse_or_complement_like", "normalized exclusion premise plus complement/negated conclusion")
        out.add("invalid_complement_conversion_candidate", "normalized complement/negated-membership candidate")

    return out


def extract_formal_fallacy_features(item_or_text: str | dict) -> set[str]:
    """Return just the tag set for callers that do not need evidence."""
    return extract_formal_fallacy_feature_result(item_or_text).tags


def extract_formal_fallacy_role_diagnostics(item_or_text: str | dict) -> dict:
    result = extract_formal_fallacy_feature_result(item_or_text)
    raw = _text(item_or_text)
    t = _norm(raw)
    premise, conclusion = _premise_and_conclusion(t)
    return {
        "features": sorted(result.tags),
        "premise_spans": [premise[:220]] if premise else [],
        "conclusion_spans": [conclusion[:220]] if conclusion else [],
        "detected_x": "",
        "detected_y": "",
        "exhaustive_partition_evidence": "; ".join(result.evidence.get("explicit_exhaustive_partition", [])),
    }


def assign_formal_fallacy_subtype_from_tags(tags: set[str] | list[str]) -> str:
    tags = set(tags)
    if "invalid_complement_conversion_candidate" in tags or "universal_negative_complement_conversion" in tags:
        return "universal_negative_complement"
    if "invalid_inverse_or_complement_like" in tags:
        return "categorical_invalid_conversion"
    if "modus_tollens" in tags:
        return "modus_tollens_valid"
    if "modus_ponens" in tags:
        return "modus_ponens_valid"
    if "converse_inference" in tags and "conditional_statement" in tags:
        return "conditional_converse"
    if "inverse_inference" in tags and "conditional_statement" in tags:
        return "conditional_inverse"
    if "valid_categorical_syllogism" in tags or "conclusion_standard_valid_chain" in tags:
        return "categorical_valid_chain"
    if "illicit_categorical_conversion" in tags:
        return "categorical_invalid_conversion"
    if "categorical_syllogism" in tags:
        return "categorical_syllogism"
    return "other"


def assign_formal_fallacy_subtype(item_or_text: str | dict) -> str:
    return assign_formal_fallacy_subtype_from_tags(
        extract_formal_fallacy_features(item_or_text)
    )
