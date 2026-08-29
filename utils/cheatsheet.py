"""
cheatsheet.py — Cheatsheet data structure.

A Cheatsheet has two parts:
  roadmap      — a structured reasoning roadmap (generated once at init)
  case_studies — a growing list of CaseStudy records (added during training)

render() produces the full text injected into the evaluation prompt, enforcing
size limits so the total stays within ~10 kb:
  - Roadmap is hard-capped at ROADMAP_MAX_CHARS.
  - Each case study is hard-capped at CASE_STUDY_MAX_CHARS (applied to render()).
  - Case studies are included newest-first (most recent failures are most relevant)
    until TOTAL_RENDER_MAX_CHARS would be exceeded.

The JSON sidecar always stores the full untruncated structured content so nothing
is lost.  Each case study is stored as a rich structured dict (see CaseStudy.to_dict()).

Persistence: save() writes <path>.txt (rendered) and <path>.json (full structured).
Backward compatibility: load() accepts the old format (case_studies as list[str])
and transparently wraps each string with CaseStudy.from_text().
JSON backward compat: old sidecar files with key "decision_tree" are loaded into roadmap.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from utils.case_study import CaseStudy


# ---------------------------------------------------------------------------
# Query feature extraction — for routing case studies at inference time
# ---------------------------------------------------------------------------

# Sentinel form value returned for items that don't have equation fields (e.g. BBH tasks).
# tokens() returns frozenset() for this value so relevance falls back to fix_rate only.
_NULL_FORM = "NONE"


class QueryFeatures(NamedTuple):
    """Structural features extracted from an (E1, E2) query pair."""
    form_e1:   str   # TRIVIAL | SINGLETON | ABSORBING | STANDARD | GENERAL | NONE
    form_e2:   str
    l_e1:      int   # number of * on the left side of = in E1
    l_e2:      int   # number of * on the left side of = in E2
    vars_e1:   int   # distinct variable count in E1
    vars_e2:   int   # distinct variable count in E2
    depth_e1:  int   # total * count in E1 (nesting depth proxy)
    depth_e2:  int   # total * count in E2

    def signature(self) -> str:
        """
        Compact structural tag in the same format used by CaseStudy.feature_signature.
        e.g. "absorbing→general_L0" or "standard_vars3→standard_vars2_L1"
        """
        f1 = self.form_e1.lower()
        f2 = self.form_e2.lower()
        return f"{f1}_vars{self.vars_e1}→{f2}_vars{self.vars_e2}_L{self.l_e2}"

    def tokens(self) -> frozenset[str]:
        """
        Token set for Jaccard similarity against CaseStudy.feature_signature tokens.
        Includes form names, variable-count markers, and left-op-count markers.
        Returns frozenset() for NONE-form items (non-equation tasks) so relevance
        falls back entirely to fix_rate with jaccard=0.
        """
        if self.form_e1 == _NULL_FORM:
            return frozenset()
        return frozenset([
            self.form_e1.lower(),
            self.form_e2.lower(),
            f"vars{self.vars_e1}",
            f"vars{self.vars_e2}",
            f"L{self.l_e2}",
            f"L{self.l_e1}",
        ])


def extract_query_features(item: dict) -> QueryFeatures:
    """
    Extract structural features from an evaluation item dict.

    Reads item["equation1"] and item["equation2"] (strings like "x = y * z").
    For items without equation fields (e.g. BBH tasks), returns a null-sentinel
    QueryFeatures whose tokens() is empty — relevance ranking uses fix_rate only.
    Pure string parsing — no LLM calls, no regex-heavy dependencies.
    """
    if "equation1" not in item:
        return QueryFeatures(
            form_e1=_NULL_FORM, form_e2=_NULL_FORM,
            l_e1=0, l_e2=0, vars_e1=0, vars_e2=0, depth_e1=0, depth_e2=0,
        )
    e1_raw = str(item["equation1"]).strip()
    e2_raw = str(item.get("equation2", "")).strip()
    return _features_from_pair(e1_raw, e2_raw)


def _features_from_pair(e1_raw: str, e2_raw: str) -> QueryFeatures:
    def _split_eq(eq: str) -> tuple[str, str]:
        """Split 'LHS = RHS' at the first bare '=' (not inside parens)."""
        depth = 0
        for i, ch in enumerate(eq):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "=" and depth == 0:
                return eq[:i].strip(), eq[i + 1:].strip()
        # Fallback: split at first '='
        parts = eq.split("=", 1)
        return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (eq.strip(), "")

    def _is_bare_var(s: str) -> bool:
        """True if the string is a single variable (letters/digits, no *)."""
        return bool(re.fullmatch(r"[a-zA-Z]\w*", s.replace(" ", "")))

    def _vars(s: str) -> set[str]:
        """Return the set of distinct single-letter variable names in an expression."""
        return set(re.findall(r"\b([a-zA-Z])\b", s))

    def _star_count(s: str) -> int:
        return s.count("*")

    def _classify(lhs: str, rhs: str) -> str:
        if lhs == rhs:
            return "TRIVIAL"
        lhs_bare = _is_bare_var(lhs)
        rhs_bare = _is_bare_var(rhs)
        if lhs_bare and rhs_bare:
            return "SINGLETON"
        if lhs_bare:
            # Variable on left — is it in the RHS?
            return "STANDARD" if lhs in _vars(rhs) else "ABSORBING"
        if rhs_bare:
            # Variable on right — is it in the LHS?
            return "STANDARD" if rhs in _vars(lhs) else "ABSORBING"
        return "GENERAL"

    lhs1, rhs1 = _split_eq(e1_raw)
    lhs2, rhs2 = _split_eq(e2_raw)

    return QueryFeatures(
        form_e1  = _classify(lhs1, rhs1),
        form_e2  = _classify(lhs2, rhs2),
        l_e1     = _star_count(lhs1),
        l_e2     = _star_count(lhs2),
        vars_e1  = len(_vars(e1_raw)),
        vars_e2  = len(_vars(e2_raw)),
        depth_e1 = _star_count(e1_raw),
        depth_e2 = _star_count(e2_raw),
    )


def _sig_tokens(sig: str) -> frozenset[str]:
    """Tokenise a feature_signature string by splitting on non-alphanumeric chars."""
    return frozenset(t for t in re.split(r"[^a-zA-Z0-9]+", sig) if t)


def _relevance_score(cs: CaseStudy, qf: QueryFeatures) -> float:
    """
    Score a case study's relevance to a query, in [0.0, 1.0].

    Two components:
      - Structural similarity (weight 0.7): Jaccard overlap between the query's
        structural tokens and the case study's feature_signature tokens, plus a
        keyword bonus from activate_if text.
        TYPE A case studies (missing knowledge) use a narrow E1-form signature
        (e.g. "absorbing") that applies to an entire form class — a query that
        shares even one token gets a full structural match so the lemma is always
        surfaced when the E1 form condition holds.
      - historical_fix_rate (weight 0.3): more accurate case studies rank higher
        when structural similarity is tied.

    Cases with no feature_signature get jaccard=0 but still compete on fix rate.
    """
    q_tokens = qf.tokens()

    # --- Jaccard against feature_signature ---
    if cs.feature_signature:
        cs_tokens = _sig_tokens(cs.feature_signature)
        union = q_tokens | cs_tokens
        intersection = q_tokens & cs_tokens
        if cs.failure_type == "A" and intersection:
            # TYPE A: any token hit on the lemma scope means full relevance —
            # the knowledge gap applies whenever the E1 form condition matches.
            jaccard = 1.0
        else:
            jaccard = len(intersection) / len(union) if union else 0.0
    else:
        jaccard = 0.0

    # --- Keyword bonus from activate_if conditions ---
    # Checks whether form names (e.g. "absorbing", "general") appear in the
    # activate_if text — catches cases whose feature_signature is absent or
    # uses different conventions.
    keyword_bonus = 0.0
    if cs.activate_if:
        activate_text = " ".join(cs.activate_if).lower()
        hits = sum(1 for tok in q_tokens if tok in activate_text)
        keyword_bonus = min(hits / max(len(q_tokens), 1), 0.4)  # cap at 0.4

    structural = min(jaccard + keyword_bonus * (1 - jaccard), 1.0)

    # --- Blend with fix rate ---
    fix_rate = cs.historical_fix_rate or cs.creation_fix_rate
    return 0.7 * structural + 0.3 * fix_rate


# ---------------------------------------------------------------------------
# Size budget  (all in characters)
# ---------------------------------------------------------------------------

ROADMAP_MAX_CHARS      = 2_500   # ~600 tokens — covers 8-10 detailed steps
CASE_STUDY_MAX_CHARS   =   600   # ~150 tokens — one focused rule + 2-3 examples
TOTAL_RENDER_MAX_CHARS = 50_000  # ~14 k tokens — safe for 28 k-token vLLM context

# Backward-compat alias — external code that imported DECISION_TREE_MAX_CHARS still works
DECISION_TREE_MAX_CHARS = ROADMAP_MAX_CHARS

CASE_STUDIES_HEADER  = "=== CASE STUDIES ==="
CASE_STUDY_DIVIDER   = "--- Case Study {n}: {title} ---"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

PRIOR_KNOWLEDGE_HEADER = "=== PRIOR KNOWLEDGE ==="
ROADMAP_HEADER         = "=== REASONING ROADMAP ==="


@dataclass
class Cheatsheet:
    roadmap: str
    case_studies: list[CaseStudy] = field(default_factory=list)
    # Optional frozen prior knowledge (e.g. NeuriCo prompt or CS-ICL bootstrap).
    # Rendered before the roadmap and never modified by ICR.
    # When prior_knowledge_segments is non-empty this field is ignored at render
    # time — use it only as a fallback / backward-compat string.
    prior_knowledge: str = ""
    # Structured segments of prior_knowledge.  Each dict has keys:
    #   id      : str   — stable identifier, e.g. "seg_0"
    #   title   : str   — first heading in the block
    #   content : str   — full text of the segment (including its heading)
    #   enabled : bool  — False means the segment is excluded from render
    # Populated by gen_icr_bootstrap.py; empty in all older Cheatsheet files.
    prior_knowledge_segments: list[dict] = field(default_factory=list)
    # When True, render() skips ALL character caps: roadmap is not truncated,
    # every case study is included in full, and the total budget is unlimited.
    # Use --no-render-limit in ICR_select/pipeline.py to enable.
    # Not persisted to JSON — must be set explicitly after load().
    no_limit: bool = field(default=False, compare=False, repr=False)
    # Hard cap on total rendered chars. When set, overrides TOTAL_RENDER_MAX_CHARS.
    # Useful for cheatsheet-size ablations (--max-cheatsheet-chars).
    # Not persisted to JSON — must be set explicitly after load().
    render_max_chars: int | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Accept list[str] or list[dict] for backward compatibility.
        normalized: list[CaseStudy] = []
        for cs in self.case_studies:
            if isinstance(cs, str):
                normalized.append(CaseStudy.from_text(cs))
            elif isinstance(cs, dict):
                normalized.append(CaseStudy.from_dict(cs))
            else:
                normalized.append(cs)
        self.case_studies = normalized

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """
        Produce the full cheatsheet text ready for injection into a prompt.

        Selection strategy (two passes):
          1. Bin representatives — for each distinct feature_signature, include
             the highest-fix-rate case study.  This guarantees every structural
             bin that has at least one case study is represented in the render.
          2. Remaining budget — fill with the remaining case studies sorted by
             fix rate descending, so the weakest ones are dropped when the
             character budget (TOTAL_RENDER_MAX_CHARS) runs out.

        Fix rate is historical_fix_rate when available, falling back to
        creation_fix_rate.  Case studies with no feature_signature are grouped
        together as one "unclassified" bin.

        For inference-time routing, use render_for_query(item, top_k) instead.
        """
        if not self.case_studies:
            return self._render_with_selection([])

        def _fix_rate(cs: CaseStudy) -> float:
            return cs.historical_fix_rate or cs.creation_fix_rate or 0.0

        # Group by feature_signature — each distinct signature is one bin
        by_sig: dict[str, list[tuple[int, CaseStudy]]] = defaultdict(list)
        for i, cs in enumerate(self.case_studies):
            key = cs.feature_signature or "__unclassified__"
            by_sig[key].append((i, cs))

        # Pass 1: best representative from each bin, sorted by fix rate
        representatives: list[tuple[int, CaseStudy]] = []
        rest: list[tuple[int, CaseStudy]] = []
        for group in by_sig.values():
            group_sorted = sorted(group, key=lambda p: _fix_rate(p[1]), reverse=True)
            representatives.append(group_sorted[0])
            rest.extend(group_sorted[1:])

        representatives.sort(key=lambda p: _fix_rate(p[1]), reverse=True)
        rest.sort(key=lambda p: _fix_rate(p[1]), reverse=True)

        return self._render_with_selection(representatives + rest)

    def render_for_query(
        self,
        item: dict,
        top_k: int = 3,
    ) -> str:
        """
        Render the cheatsheet with only the top-k most relevant case studies
        for this specific query item.

        Instead of dumping all recent cases globally (render()), this method:
          1. Extracts structural features from item["equation1"]/["equation2"].
          2. Scores every case study by relevance to those features using
             token Jaccard on feature_signature + activate_if keyword hits,
             blended with historical_fix_rate.
          3. Renders the top-k highest-scoring cases within the character budget.

        The decision tree and prior knowledge sections are identical to render().
        The format is identical — this is a drop-in replacement in the scorer.

        Parameters
        ----------
        item    : evaluation item dict with "equation1" and "equation2" keys.
        top_k   : maximum number of case studies to include (default: 3).
                  The budget cap applies independently — fewer may be included
                  if the top-k cases are long.
        """
        qf = extract_query_features(item)
        return self._render_with_selection(
            selected=self._select_top_k(qf, top_k),
        )

    def render_for_partition_key(
        self,
        partition_key: tuple | list,
        include_unrouted: bool = False,
    ) -> str:
        """
        Render only case studies whose machine-readable routing_key matches.

        Prior knowledge is always rendered. Case studies with no routing_key are
        excluded by default so routed evaluation can measure conservative local
        exposure rather than global-prepend behavior.
        """
        key = list(partition_key)
        selected: list[tuple[int, CaseStudy]] = []
        for i, cs in enumerate(self.case_studies):
            cs_key = getattr(cs, "routing_key", []) or []
            if cs_key == key or (include_unrouted and not cs_key):
                selected.append((i, cs))
        return self._render_with_selection(selected)

    def _select_top_k(self, qf: QueryFeatures, top_k: int) -> list[tuple[int, CaseStudy]]:
        """
        Return up to top_k (original_index, CaseStudy) pairs ranked by relevance.
        Original index is preserved so the Case Study N display number is stable.
        """
        if not self.case_studies:
            return []
        scored = sorted(
            enumerate(self.case_studies),
            key=lambda pair: _relevance_score(pair[1], qf),
            reverse=True,
        )
        return scored[:top_k]

    def _render_with_selection(
        self,
        selected: list[tuple[int, CaseStudy]],
    ) -> str:
        """
        Shared render core: DT + prior knowledge + a caller-provided list of
        (original_index, CaseStudy) pairs, within the standard character budget.

        When self.no_limit is True, all character caps are bypassed: the roadmap
        is never truncated, every case study is rendered in full, and the total
        budget is unlimited.
        """
        _no_limit = self.no_limit

        parts: list[str] = []
        pk = self._render_prior_knowledge()
        if pk:
            parts += [PRIOR_KNOWLEDGE_HEADER, "", pk, ""]

        _total_cap = TOTAL_RENDER_MAX_CHARS if self.render_max_chars is None else self.render_max_chars
        budget = float("inf") if _no_limit else _total_cap - len("\n".join(parts))

        if selected:
            header_block = ["", CASE_STUDIES_HEADER]
            if not _no_limit:
                budget -= len("\n".join(header_block))
            cs_blocks: list[list[str]] = []

            for orig_idx, cs in selected:
                display_n = orig_idx + 1   # 1-based, preserves original numbering
                body  = cs.render() if _no_limit else _truncate(cs.render(), CASE_STUDY_MAX_CHARS)
                block = [
                    "",
                    CASE_STUDY_DIVIDER.format(n=display_n, title=cs.title),
                    body,
                ]
                block_size = len("\n".join(block))
                if block_size > budget:
                    break
                cs_blocks.append(block)
                if not _no_limit:
                    budget -= block_size

            if cs_blocks:
                parts += header_block
                for block in cs_blocks:
                    parts += block

        return "\n".join(parts)

    def render_size(self) -> int:
        """Return the character count of the rendered cheatsheet."""
        return len(self.render())

    def _render_prior_knowledge(self) -> str:
        """
        Return the prior-knowledge text to inject into the prompt.

        If prior_knowledge_segments is populated, join only the enabled segments
        (preserving the original --- separators between them).
        Falls back to the raw prior_knowledge string for backward compatibility.
        """
        if self.prior_knowledge_segments:
            enabled = [s["content"] for s in self.prior_knowledge_segments if s.get("enabled", True)]
            return "\n\n---\n\n".join(enabled).strip()
        return self.prior_knowledge.strip()

    # ------------------------------------------------------------------
    # Prior-knowledge segment ablation
    # ------------------------------------------------------------------

    def disable_prior_segment(self, segment_id: str) -> bool:
        """Disable a prior-knowledge segment by id. Returns True if found."""
        for seg in self.prior_knowledge_segments:
            if seg["id"] == segment_id:
                seg["enabled"] = False
                return True
        return False

    def enable_prior_segment(self, segment_id: str) -> bool:
        """Re-enable a prior-knowledge segment by id. Returns True if found."""
        for seg in self.prior_knowledge_segments:
            if seg["id"] == segment_id:
                seg["enabled"] = True
                return True
        return False

    def prior_segment_ids(self) -> list[str]:
        """Return all segment ids in order."""
        return [s["id"] for s in self.prior_knowledge_segments]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_case_study(self, cs: "CaseStudy | str") -> None:
        """Append a new case study. Accepts CaseStudy or plain text string."""
        if isinstance(cs, str):
            cs = CaseStudy.from_text(cs)
        self.case_studies.append(cs)
        rendered_size = self.render_size()
        if rendered_size > TOTAL_RENDER_MAX_CHARS:
            print(
                f"  [cheatsheet] rendered size {rendered_size:,} chars — oldest case "
                f"studies will be excluded from render but remain in JSON.",
                file=sys.stderr,
            )

    def replace_case_study(self, target_title: str, new_cs: "CaseStudy") -> bool:
        """
        Replace the first case study whose title fuzzy-matches target_title.
        Returns True if a match was found and replaced, False otherwise.
        """
        target = target_title.lower().strip()
        for i, cs in enumerate(self.case_studies):
            cs_title = cs.title.lower().strip()
            if cs_title == target or target in cs_title or cs_title in target:
                self.case_studies[i] = new_cs
                return True
        return False

    def patch_roadmap(self, patch_text: str) -> None:
        """
        Append a roadmap patch block produced by the reasoning-aware case study generator.

        Patches are appended after the existing roadmap text.  If the combined
        length exceeds ROADMAP_MAX_CHARS, render() will truncate the oldest
        content (from the beginning of the roadmap), so newly added rules survive.
        """
        patch_text = patch_text.strip()
        if not patch_text:
            return
        self.roadmap = self.roadmap.rstrip() + "\n\n" + patch_text
        rm_len = len(self.roadmap)
        if rm_len > ROADMAP_MAX_CHARS:
            print(
                f"  [cheatsheet] roadmap is {rm_len:,} chars after patch — "
                f"oldest content will be truncated in render (cap={ROADMAP_MAX_CHARS}).",
                file=sys.stderr,
            )

    # Backward-compat alias
    def patch_decision_tree(self, patch_text: str) -> None:
        self.patch_roadmap(patch_text)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save the cheatsheet.
        <path>.txt  — rendered output (size-limited, what goes into prompts)
        <path>.json — full structured content (no truncation; case studies as rich dicts)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.with_suffix(".txt").write_text(self.render(), encoding="utf-8")
        payload: dict = {
            "roadmap":         self.roadmap,
            "case_studies":    [cs.to_dict() for cs in self.case_studies],
            # Always write prior_knowledge as the current rendered text so older
            # loaders can read it without understanding segments.
            "prior_knowledge": self._render_prior_knowledge(),
        }
        if self.prior_knowledge_segments:
            payload["prior_knowledge_segments"] = self.prior_knowledge_segments
        path.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Cheatsheet":
        """
        Load from the .json sidecar written by save().

        Backward compatible: if case_studies entries are plain strings (old format),
        each is parsed with CaseStudy.from_text().
        """
        json_path = Path(path).with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8"))

        raw_studies = data.get("case_studies", [])
        case_studies: list[CaseStudy] = []
        for entry in raw_studies:
            if isinstance(entry, str):
                case_studies.append(CaseStudy.from_text(entry))
            elif isinstance(entry, dict):
                case_studies.append(CaseStudy.from_dict(entry))
            # else: skip malformed entries

        # Backward compat: old sidecars used "decision_tree" key
        rm = data.get("roadmap") or data.get("decision_tree", "")
        return cls(
            roadmap=rm,
            case_studies=case_studies,
            prior_knowledge=data.get("prior_knowledge", ""),
            prior_knowledge_segments=data.get("prior_knowledge_segments", []),
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> str:
        rendered = self.render_size()
        dt_shown = min(len(self.roadmap), ROADMAP_MAX_CHARS)
        return (
            f"case_studies={len(self.case_studies)}  "
            f"rendered={rendered:,} chars  "
            f"dt_chars={dt_shown}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """
    Truncate *text* to at most *max_chars* characters, breaking at a line
    boundary where possible so no line is cut mid-sentence.
    Appends '[truncated]' if content was removed.
    """
    if len(text) <= max_chars:
        return text

    # Try to break at the last newline before the limit
    cut = text.rfind("\n", 0, max_chars)
    if cut == -1 or cut < max_chars // 2:
        # No good line break — cut at word boundary
        cut = text.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars

    return text[:cut].rstrip() + "\n[truncated]"
