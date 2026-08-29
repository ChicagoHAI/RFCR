"""ICR_sfcr/activation.py — USE WHEN routing and cheatsheet construction.

Two inference modes:
  global  — anchor + ALL accepted rules prepended unconditionally
  routed  — anchor + only rules whose USE WHEN trigger matches the input question

V1 routing uses simple keyword overlap: tokenise the USE WHEN clause,
strip English stop words, and check whether any content term appears in
the lowercased question text.  The plan explicitly calls for "a simple
keyword or symbolic trigger" — this is not meant to be a retrieval system.
"""
from __future__ import annotations

import re

# Minimal English stop-word list — enough to filter article/preposition noise
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "as", "into", "through", "during",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both", "either",
    "this", "that", "these", "those", "it", "its", "if", "then", "than",
    "when", "where", "which", "who", "what", "how", "there", "their",
    "they", "we", "you", "he", "she", "i", "my", "your", "our", "his",
    "her", "its", "about", "between", "each", "more", "most", "other",
    "some", "such", "only", "same", "also", "any", "all",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _keywords(text: str) -> set[str]:
    """Extract lowercased non-stop-word tokens from text."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def matches_use_when(use_when: str, question_text: str) -> bool:
    """
    Return True if the USE WHEN clause has any keyword overlap with question_text.
    An empty USE WHEN clause always matches (unconditional rule).
    """
    if not use_when.strip():
        return True
    trigger_kws = _keywords(use_when)
    if not trigger_kws:
        return True
    question_kws = _keywords(question_text)
    return bool(trigger_kws & question_kws)


def _rule_block(rule: dict) -> str:
    lines = [f"RULE: {rule['rule']}"]
    if rule.get("use_when"):
        lines.append(f"USE WHEN: {rule['use_when']}")
    if rule.get("do_not_use_when"):
        lines.append(f"DO NOT USE WHEN: {rule['do_not_use_when']}")
    if rule.get("check"):
        lines.append(f"CHECK: {rule['check']}")
    return "\n".join(lines)


def build_cheatsheet(
    anchor: str,
    accepted_rules: list[dict],
    mode: str = "global",
    question_text: str = "",
) -> str:
    """
    Construct the final cheatsheet for a given question.

    mode="global"  — append all accepted rules unconditionally
    mode="routed"  — append only rules whose USE WHEN matches question_text
    """
    if not accepted_rules:
        return anchor

    if mode == "global":
        active = accepted_rules
    elif mode == "routed":
        active = [
            r for r in accepted_rules
            if matches_use_when(r.get("use_when", ""), question_text)
        ]
    else:
        raise ValueError(f"Unknown routing mode: {mode!r}. Use 'global' or 'routed'.")

    if not active:
        return anchor

    rule_section = "\n\n--- ADDITIONAL RULES ---\n" + "\n\n".join(
        _rule_block(r) for r in active
    )
    return anchor.rstrip() + rule_section


def activation_summary(
    accepted_rules: list[dict],
    items: list[dict],
) -> dict:
    """
    Return per-rule activation statistics over a set of items.
    Used for logging and reporting.
    """
    summary = {}
    for i, rule in enumerate(accepted_rules):
        use_when = rule.get("use_when", "")
        hits = sum(
            1 for it in items
            if matches_use_when(use_when, it.get("input", ""))
        )
        summary[f"rule_{i}"] = {
            "rule_prefix":   rule["rule"][:60],
            "use_when":      use_when[:80],
            "activation_n":  hits,
            "activation_pct": hits / len(items) if items else 0.0,
        }
    return summary

# ---------------------------------------------------------------------------
# Feature/tag routed activation for SF-CR follow-up experiments
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any

from .features import (
    extract_formal_fallacy_feature_result,
    infer_candidate_tags,
)


@dataclass
class RouteDecision:
    activated: bool
    activation_reason: str
    matched_positive_tags: list[str] = field(default_factory=list)
    matched_negative_tags: list[str] = field(default_factory=list)
    vetoed: bool = False
    item_tags: list[str] = field(default_factory=list)


def _parse_partition_label(label: str) -> tuple | None:
    if not label:
        return None
    parts = label.split("_")
    if all(p in {"True", "False"} for p in parts):
        return tuple(p == "True" for p in parts)
    return tuple(parts)


def _candidate_partition_key(rule: dict, fallback: tuple | None = None) -> tuple | None:
    raw = rule.get("routing_key") or rule.get("partition_key")
    if raw is None and rule.get("partition"):
        return _parse_partition_label(str(rule.get("partition")))
    if raw is None:
        return fallback
    if isinstance(raw, tuple):
        return raw
    if isinstance(raw, list):
        return tuple(raw)
    if isinstance(raw, str):
        return _parse_partition_label(raw)
    return fallback


def _positive_tag_groups(rule: dict) -> list[set[str]]:
    raw_groups = rule.get("positive_tag_groups") or []
    groups: list[set[str]] = []
    for group in raw_groups:
        if isinstance(group, (list, tuple, set)):
            tags = {str(t) for t in group if str(t)}
        else:
            tags = {str(group)}
        if tags:
            groups.append(tags)
    return groups


def route_candidate(
    rule: dict,
    item: dict,
    activation_routing_mode: str = "feature",
    partition_key: tuple | None = None,
    task_spec: Any | None = None,
) -> RouteDecision:
    """Route one candidate to one item under partition, feature, or hybrid mode."""
    mode = activation_routing_mode
    if mode not in {"partition", "feature", "hybrid"}:
        raise ValueError(f"Unknown activation_routing_mode={mode!r}")

    item_text = str(item.get("input") or item.get("question") or item)
    feature_result = extract_formal_fallacy_feature_result(item_text)
    item_tags = feature_result.tags

    cand_pos, cand_neg = infer_candidate_tags(rule)
    cand_pos.update(rule.get("positive_tags") or [])
    cand_neg.update(rule.get("negative_tags") or [])
    tag_groups = _positive_tag_groups(rule)

    matched_pos = sorted(t for t in cand_pos if t in item_tags)
    matched_neg = sorted(t for t in cand_neg if t in item_tags)
    vetoed = bool(matched_neg)

    if tag_groups:
        matched_group = next((g for g in tag_groups if g.issubset(item_tags)), None)
        feature_ok = matched_group is not None
        if matched_group is not None:
            matched_pos = sorted(matched_group)
            feature_reason = "matched positive tag group: " + ",".join(matched_pos)
        else:
            missing = ["+".join(sorted(g - item_tags)) for g in tag_groups]
            feature_reason = "missing all positive tag groups: " + " | ".join(missing)
    elif cand_pos:
        feature_ok = cand_pos.issubset(item_tags)
        feature_reason = (
            "matched positive tags: " + ",".join(matched_pos)
            if feature_ok
            else "missing positive tags: " + ",".join(sorted(cand_pos - item_tags))
        )
    else:
        use_when = rule.get("use_when", "")
        feature_ok = matches_use_when(use_when, item_text)
        feature_reason = "USE WHEN keyword route" if feature_ok else "USE WHEN did not match"

    if vetoed:
        feature_ok = False
        feature_reason = "vetoed by negative tags: " + ",".join(matched_neg)

    part_ok = True
    part_reason = "no partition constraint"
    target_key = _candidate_partition_key(rule, partition_key)
    if target_key is not None and task_spec is not None:
        try:
            actual_key = tuple(task_spec.partition_key(item))
            part_ok = actual_key == tuple(target_key)
            part_reason = f"partition actual={actual_key} target={tuple(target_key)}"
        except Exception as exc:
            part_ok = False
            part_reason = f"partition error: {exc}"

    if mode == "partition":
        activated = part_ok
        reason = part_reason
    elif mode == "feature":
        activated = feature_ok
        reason = feature_reason
    else:
        activated = part_ok and feature_ok
        reason = f"{part_reason}; {feature_reason}"

    return RouteDecision(
        activated=activated,
        activation_reason=reason,
        matched_positive_tags=matched_pos,
        matched_negative_tags=matched_neg,
        vetoed=vetoed,
        item_tags=sorted(item_tags),
    )
