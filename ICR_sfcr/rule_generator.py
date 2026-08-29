"""ICR_sfcr/rule_generator.py — Generate structured rule candidates from failure regions.

Each candidate has the structured format:
  RULE:           The rule text itself.
  USE WHEN:       Conditions that must hold for the rule to apply.
  DO NOT USE WHEN: Conditions that exclude the rule (anti-activation).
  CHECK:          A self-check the model should perform before applying the rule.

v2 additions:
  cluster_subtypes()   — LLM-based grouping of V_shared into 2-4 failure subtypes.
  generate_candidates() — Cluster-first, then generate 2-3 candidates per subtype
                          with private examples shown as explicit boundary cases.
  repair_candidate()   — Narrow a failed candidate given its rejection profile.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from utils.llm_client import call_llm
from utils.task_spec import TaskSpec

from .prompts import (
    COMPRESSED_RATIONALE_PROMPT,
    RULE_GENERATION_PROMPT,
    RULE_GENERATION_PROMPT_SUBTYPE,
    RULE_REPAIR_PROMPT,
    SUBTYPE_CLUSTER_PROMPT,
)

_MAX_RULE_CHARS = 800
_ITEM_DISPLAY_LIMIT = 8  # max items per block in the generation prompt


# ---------------------------------------------------------------------------
# Item formatting
# ---------------------------------------------------------------------------

def _compress_rationale(reasoning: str, model: str, api_key: str) -> str:
    prompt = COMPRESSED_RATIONALE_PROMPT.format(reasoning=reasoning[:2000])
    resp = call_llm(prompt, model=model, api_key=api_key, temperature=0.0, max_tokens=120)
    return resp.content.strip() if resp else reasoning[:120]


def _format_item(item: dict, oracle_mode: str, model: str, api_key: str) -> str:
    text   = item.get("input", str(item))
    answer = item.get("answer", "?")
    reason = item.get("reason", "")

    if oracle_mode == "none":
        return f"Question: {text}"
    if oracle_mode == "label_only":
        return f"Question: {text}\nAnswer: {answer}"
    if oracle_mode == "compressed" and reason:
        summary = _compress_rationale(reason, model, api_key)
        return f"Question: {text}\nAnswer: {answer}\nKey insight: {summary}"
    if oracle_mode == "full_cot" and reason:
        return f"Question: {text}\nAnswer: {answer}\nReasoning: {reason}"
    return f"Question: {text}\nAnswer: {answer}"


def _format_block(
    items: list[dict],
    oracle_mode: str,
    model: str,
    api_key: str,
    max_items: int = _ITEM_DISPLAY_LIMIT,
) -> str:
    if not items:
        return "(none)"
    selected = items[:max_items]

    if oracle_mode == "compressed":
        with ThreadPoolExecutor(max_workers=min(len(selected), 8)) as pool:
            formatted = list(
                pool.map(
                    lambda it: _format_item(it, oracle_mode, model, api_key),
                    selected,
                )
            )
    else:
        formatted = [_format_item(it, oracle_mode, model, api_key) for it in selected]

    return "\n\n".join(f"[{i+1}] {f}" for i, f in enumerate(formatted))


def _format_block_boundary(
    items: list[dict],
    max_items: int = _ITEM_DISPLAY_LIMIT,
) -> str:
    """Format private items as explicit boundary/counterexamples."""
    if not items:
        return "(none)"
    selected = items[:max_items]
    parts = []
    for i, it in enumerate(selected):
        text   = it.get("input", str(it))
        answer = it.get("answer", "?")
        parts.append(
            f"[{i+1}] BOUNDARY — rule must NOT fire here\n"
            f"Question: {text}\nAnswer: {answer}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Rule parser
# ---------------------------------------------------------------------------

_LABEL_PATTERN = re.compile(
    r"(?:^|\n)(RULE|USE WHEN|DO NOT USE WHEN|CHECK)\s*:(.*?)(?=\n(?:RULE|USE WHEN|DO NOT USE WHEN|CHECK)\s*:|$)",
    re.DOTALL | re.IGNORECASE,
)


def parse_rule_text(raw: str) -> dict | None:
    """
    Parse structured RULE / USE WHEN / DO NOT USE WHEN / CHECK output.
    Returns None if RULE section is missing or empty.
    """
    sections: dict[str, str] = {}
    for m in _LABEL_PATTERN.finditer(raw):
        key = m.group(1).upper().strip()
        val = m.group(2).strip()
        sections[key] = val

    rule = sections.get("RULE", "").strip()
    if not rule:
        return None

    return {
        "rule":             rule,
        "use_when":         sections.get("USE WHEN", "").strip(),
        "do_not_use_when":  sections.get("DO NOT USE WHEN", "").strip(),
        "check":            sections.get("CHECK", "").strip(),
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# Subtype clustering
# ---------------------------------------------------------------------------

def cluster_subtypes(
    V_shared: list[dict],
    model: str,
    api_key: str,
    max_subtypes: int = 4,
) -> list[dict]:
    """
    Group V_shared examples into 2-max_subtypes failure subtypes using an LLM.

    Returns a list of subtype dicts:
        {label: str, description: str, items: list[dict]}

    Falls back to a single subtype containing all items if clustering fails or
    V_shared is too small to split meaningfully.
    """
    if len(V_shared) < 4:
        # Too few items to cluster — single subtype
        return [{"label": "all shared failures", "description": "", "items": V_shared}]

    block = _format_block(V_shared, oracle_mode="label_only", model=model, api_key=api_key,
                          max_items=min(len(V_shared), 20))
    prompt = SUBTYPE_CLUSTER_PROMPT.format(v_shared_block=block)

    resp = call_llm(prompt, model=model, api_key=api_key, temperature=0.0, max_tokens=800)
    if resp is None:
        return [{"label": "all shared failures", "description": "", "items": V_shared}]

    raw = resp.content.strip()
    # Extract JSON — strip any markdown fences
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return [{"label": "all shared failures", "description": "", "items": V_shared}]

    try:
        parsed = json.loads(json_match.group(0))
        subtypes_raw = parsed.get("subtypes", [])
    except (json.JSONDecodeError, KeyError):
        return [{"label": "all shared failures", "description": "", "items": V_shared}]

    if not subtypes_raw:
        return [{"label": "all shared failures", "description": "", "items": V_shared}]

    # Build item lists from indices
    subtypes: list[dict] = []
    assigned: set[int] = set()
    for st in subtypes_raw[:max_subtypes]:
        indices = [i for i in st.get("indices", []) if isinstance(i, int) and 0 <= i < len(V_shared)]
        if not indices:
            continue
        assigned.update(indices)
        subtypes.append({
            "label":       st.get("label", "subtype"),
            "description": st.get("description", ""),
            "items":       [V_shared[i] for i in indices],
        })

    # Any unassigned items go into the first subtype (or a new one)
    unassigned = [V_shared[i] for i in range(len(V_shared)) if i not in assigned]
    if unassigned:
        if subtypes:
            subtypes[0]["items"].extend(unassigned)
        else:
            subtypes = [{"label": "all shared failures", "description": "", "items": V_shared}]

    # Guard: must have at least one non-empty subtype
    subtypes = [st for st in subtypes if st["items"]]
    if not subtypes:
        return [{"label": "all shared failures", "description": "", "items": V_shared}]

    print(f"[cluster] {len(subtypes)} subtypes: " +
          ", ".join(f'"{st["label"]}"({len(st["items"])})' for st in subtypes))
    return subtypes


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    V_shared: list[dict],
    V_private: list[dict],
    anchor_cheatsheet: str,
    model: str,
    api_key: str,
    n_candidates: int = 8,
    temperatures: list[float] | None = None,
    oracle_mode: str = "label_only",
    max_rule_chars: int = _MAX_RULE_CHARS,
    use_subtypes: bool = True,
    candidates_per_subtype: int = 3,
) -> list[dict]:
    """
    Generate up to `n_candidates` structured rule candidates from V_shared.

    When `use_subtypes=True` (default):
      - Cluster V_shared into subtypes.
      - Generate `candidates_per_subtype` candidates per subtype, using a
        subtype-targeted prompt that shows private items as boundary cases.
      - Each candidate is tagged with its `subtype_idx` and `subtype_items`
        (sfcr_ids) so the validator can compute per-subtype delta_shared.

    When `use_subtypes=False`:
      - Fall back to original flat-pool generation.

    Candidates are deduplicated by normalised RULE text.
    Each returned dict has keys: rule, use_when, do_not_use_when, check,
    raw_text, temperature, subtype_idx (int), subtype_items (list[str] | None).
    """
    if temperatures is None:
        temperatures = [0.2, 0.5, 0.8]

    private_block = _format_block_boundary(V_private)
    seen: set[str] = set()
    candidates: list[dict] = []

    if not use_subtypes:
        # ── Original flat-pool mode ────────────────────────────────────────
        v_shared_block = _format_block(V_shared, oracle_mode, model, api_key)
        prompt = RULE_GENERATION_PROMPT.format(
            anchor_cheatsheet=anchor_cheatsheet,
            v_shared_block=v_shared_block,
            v_private_block=private_block,
        )
        candidates = _generate_from_prompt(
            prompt, model, api_key, n_candidates, temperatures,
            max_rule_chars, seen,
            subtype_idx=0, subtype_items=None,
        )
        return candidates

    # ── Subtype-clustered mode ─────────────────────────────────────────────
    subtypes = cluster_subtypes(V_shared, model, api_key)
    total_target = min(n_candidates, len(subtypes) * candidates_per_subtype)

    for st_idx, subtype in enumerate(subtypes):
        st_items    = subtype["items"]
        st_desc     = subtype["description"] or subtype["label"]
        other_items = [it for st in subtypes for it in st["items"] if st is not subtype]

        subtype_block = _format_block(st_items, oracle_mode, model, api_key,
                                      max_items=_ITEM_DISPLAY_LIMIT)
        other_block   = _format_block(other_items, "label_only", model, api_key,
                                      max_items=_ITEM_DISPLAY_LIMIT // 2)

        prompt = RULE_GENERATION_PROMPT_SUBTYPE.format(
            anchor_cheatsheet=anchor_cheatsheet,
            subtype_description=st_desc,
            v_subtype_block=subtype_block,
            v_other_block=other_block if other_items else "(none)",
            v_private_block=private_block,
        )

        subtype_item_ids = [it["_sfcr_id"] for it in st_items if "_sfcr_id" in it]
        new_cands = _generate_from_prompt(
            prompt, model, api_key, candidates_per_subtype, temperatures,
            max_rule_chars, seen,
            subtype_idx=st_idx, subtype_items=subtype_item_ids,
        )
        candidates.extend(new_cands)
        print(f"[gen] subtype {st_idx+1}/{len(subtypes)} "
              f'"{subtype["label"]}": {len(new_cands)} candidate(s)')

        if len(candidates) >= total_target:
            break

    return candidates[:n_candidates]


def _generate_from_prompt(
    prompt: str,
    model: str,
    api_key: str,
    n_target: int,
    temperatures: list[float],
    max_rule_chars: int,
    seen: set[str],
    subtype_idx: int,
    subtype_items: list[str] | None,
) -> list[dict]:
    """Repeatedly call the LLM until n_target unique candidates are collected."""
    candidates: list[dict] = []
    attempt = 0
    max_attempts = n_target * 6

    while len(candidates) < n_target and attempt < max_attempts:
        temp = temperatures[attempt % len(temperatures)]
        attempt += 1

        resp = call_llm(prompt, model=model, api_key=api_key,
                        temperature=temp, max_tokens=700)
        if resp is None:
            continue

        parsed = parse_rule_text(resp.content)
        if parsed is None:
            continue
        if len(parsed["rule"]) > max_rule_chars:
            continue

        key = _normalise(parsed["rule"])
        if key in seen:
            continue

        seen.add(key)
        parsed["raw_text"]      = resp.content
        parsed["temperature"]   = temp
        parsed["subtype_idx"]   = subtype_idx
        parsed["subtype_items"] = subtype_items
        candidates.append(parsed)
        print(
            f"[gen] candidate {len(candidates)}/{n_target}  "
            f"temp={temp:.1f}  len={len(parsed['rule'])} chars"
        )

    if len(candidates) < n_target:
        print(
            f"[gen] WARNING: only generated {len(candidates)}/{n_target} "
            f"unique candidates after {attempt} attempts"
        )

    return candidates


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------

def repair_candidate(
    rule: dict,
    failure_profile: dict,
    anchor_cheatsheet: str,
    model: str,
    api_key: str,
    max_rule_chars: int = _MAX_RULE_CHARS,
    max_attempts: int = 2,
) -> dict | None:
    """
    Attempt to narrow a rejected candidate using its failure profile.

    `failure_profile` keys (all optional):
        reject_reason         : str   — why it was rejected
        mis_triggered_easy    : list  — easy items that regressed (Q/A dicts)
        mis_triggered_private : list  — private items that were activated (Q/A dicts)
        no_gain_proxies       : list  — proxy model names that showed no delta_shared

    Returns a new parsed rule dict if repair succeeds, None otherwise.
    The returned dict preserves `subtype_idx` and `subtype_items` from the original.
    """
    reject_reason     = failure_profile.get("reject_reason", "unknown rejection")
    mis_easy          = failure_profile.get("mis_triggered_easy", [])
    mis_private       = failure_profile.get("mis_triggered_private", [])
    no_gain_proxies   = failure_profile.get("no_gain_proxies", [])

    all_mis = mis_private + mis_easy
    if all_mis:
        mis_lines = "\n".join(
            f"  [{i+1}] Q: {it.get('input', '')[:120]}  A: {it.get('answer', '')}"
            for i, it in enumerate(all_mis[:6])
        )
        mis_triggered_section = (
            "Examples where the rule incorrectly triggered "
            "(narrow USE WHEN to exclude these):\n" + mis_lines
        )
    else:
        mis_triggered_section = ""

    if no_gain_proxies:
        no_gain_section = (
            "Models that showed no improvement: "
            + ", ".join(no_gain_proxies)
            + "\nConsider whether the rule is too abstract to transfer."
        )
    else:
        no_gain_section = ""

    prompt = RULE_REPAIR_PROMPT.format(
        rule=rule.get("rule", ""),
        use_when=rule.get("use_when", ""),
        do_not_use_when=rule.get("do_not_use_when", ""),
        check=rule.get("check", ""),
        reject_reason=reject_reason,
        mis_triggered_section=mis_triggered_section,
        no_gain_section=no_gain_section,
    )

    for attempt in range(max_attempts):
        resp = call_llm(
            prompt,
            model=model,
            api_key=api_key,
            temperature=0.3 + 0.2 * attempt,
            max_tokens=700,
        )
        if resp is None:
            continue

        parsed = parse_rule_text(resp.content)
        if parsed is None:
            continue
        if len(parsed["rule"]) > max_rule_chars:
            continue
        # Reject if the repair made no real change to USE WHEN
        if _normalise(parsed.get("use_when", "")) == _normalise(rule.get("use_when", "")):
            continue

        parsed["raw_text"]      = resp.content
        parsed["temperature"]   = 0.3 + 0.2 * attempt
        parsed["subtype_idx"]   = rule.get("subtype_idx", 0)
        parsed["subtype_items"] = rule.get("subtype_items")
        parsed["repaired"]      = True
        print(f"[repair] attempt {attempt+1}: new USE WHEN = {parsed['use_when'][:80]!r}")
        return parsed

    print(f"[repair] failed after {max_attempts} attempts — discarding candidate")
    return None
