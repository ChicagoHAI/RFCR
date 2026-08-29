"""
ICR_partition/training/partition.py — Structural partitioning of items into bins.

Each item is assigned a PartitionKey computed entirely from equation syntax —
no LLM calls.  Items with the same key share a structurally homogeneous failure
mode and are solved independently.

PartitionKey = (form_e1, form_e2, depth_bucket_e1, expected_answer, e1_proj_class, sep_fires)
  form_*         : TRIVIAL | SINGLETON | ABSORBING | STANDARD | GENERAL
  depth_bucket   : 0 (no * operators), 1 (exactly one), 2 (two or more)
  expected_answer: "TRUE" | "FALSE"
  e1_proj_class  : "canon_l" | "canon_r" | "left_proj" | "right_proj" | "nested" | "other"
    canon_l    — E1 normalizes to a canonical left-projection form (CANON-L1 or CANON-L2)
    canon_r    — E1 normalizes to CANON-R1 (right-projection)
    left_proj  — E1 = x * A where A has no x (syntactic left-proj, non-canonical)
    right_proj — E1 = A * x where A has no x (syntactic right-proj, non-canonical)
    nested     — x appears multiple times or non-terminally in E1's RHS
    other      — non-STANDARD form, or not applicable
  sep_fires      : "LP" | "RP" | "SET" | "XOR" | "AB" | "none"
    The first separator invariant that holds for E1 but not E2.
    "none" means no separator fires (most pairs).
    For FALSE bins, sep_fires != "none" means the failure has a known structural cause
    and the case study generator can write a precise separator rule.

This yields up to 5 × 5 × 3 × 2 × 6 × 6 = 5400 possible keys; in practice a
2000-item dataset will populate 50–150 keys (most sep_fires values are "none").

Design rationale
----------------
* e1_proj_class now distinguishes canonical collapse forms (canon_l / canon_r) from
  syntactic projection forms.  Case studies for canon_l bins can state the exact
  projection lemma; case studies for left_proj bins cover the broader syntactic class.

* sep_fires sub-partitions FALSE bins by which structural invariant is violated.
  A failure in the "LP" bin means the model missed "LP(E1)=TRUE ∧ LP(E2)=FALSE → FALSE".
  The generator sees a homogeneous failure set and writes a single targeted rule.
"""

from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from utils.cheatsheet import QueryFeatures, extract_query_features
from utils.data import is_true
from utils.equation_features import (
    compute_features as _compute_eq_features,
    first_separator,
    detect_collapse,
)

if TYPE_CHECKING:
    from utils.case_study import CaseStudy


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PartitionKey = tuple[str, str, int, str, str, str, str, str]
# (form_e1, form_e2, depth_bucket_e1, expected_answer, e1_proj_class, sep_fires,
#  top_shape_e1, xtop_e1)
# top_shape_e1: "v-m"|"m-v"|"m-m"|"none"   — only set for STANDARD/nested E1
# xtop_e1:      "left"|"right"|"both"|"none" — only set for STANDARD/nested E1


# ---------------------------------------------------------------------------
# Key computation — pure structural, O(1), no LLM
# ---------------------------------------------------------------------------

def _depth_bucket(depth: int) -> int:
    """Coarsen raw depth into 0 / 1 / 2+ bucket."""
    return min(depth, 2)


def _e1_proj_class(equation: str) -> str:
    """
    Sub-classify a STANDARD form E1 by how its lone variable is anchored in
    the RHS.  Pure syntactic — no LLM calls.

    Returns (in priority order):
      "canon_l"    — E1 normalizes to a canonical left-projection form (CANON-L1/L2)
      "canon_r"    — E1 normalizes to the canonical right-projection form (CANON-R1)
      "left_proj"  — x = x * A  where A contains no x (syntactic, non-canonical)
      "right_proj" — x = A * x  where A contains no x (syntactic, non-canonical)
      "nested"     — x appears multiple times or non-terminally in the RHS
      "other"      — non-STANDARD form, no '*' in RHS, or parse failed
    """
    # Check canonical collapse forms first (most specific)
    collapse = detect_collapse(equation)
    if collapse == "left_proj":
        return "canon_l"
    if collapse == "right_proj":
        return "canon_r"

    parts = equation.strip().split("=", 1)
    if len(parts) != 2:
        return "other"
    lhs, rhs = parts[0].strip(), parts[1].strip()
    # Must be a bare single variable on the left
    if not re.match(r"^[a-z]$", lhs):
        return "other"
    # Must have at least one * in the RHS (rules out TRIVIAL / SINGLETON)
    if "*" not in rhs:
        return "other"
    v = lhs
    # Must be STANDARD form: lone variable appears somewhere in RHS
    if v not in rhs.replace(" ", ""):
        return "other"

    # left_proj: rhs = v * A,  A has no v
    m = re.match(rf"^{v}\s*\*\s*(.+)$", rhs)
    if m and v not in m.group(1).replace(" ", ""):
        return "left_proj"

    # right_proj: rhs = A * v  (v is the very last token), A has no v
    m = re.match(rf"^(.*\S)\s*\*\s*{v}\s*$", rhs)
    if m and v not in m.group(1).replace(" ", ""):
        return "right_proj"

    return "nested"


def item_partition_key(item: dict) -> PartitionKey:
    """
    Compute the partition key for one item.  Pure structural — no LLM calls.
    Falls back to ("GENERAL","GENERAL",2,polarity,"other","none","none","none")
    on any parse error.
    """
    polarity = "TRUE" if is_true(item.get("answer", False)) else "FALSE"
    try:
        qf: QueryFeatures = extract_query_features(item)
        proj = _e1_proj_class(item.get("equation1", ""))
        e1_feat = _compute_eq_features(item.get("equation1", ""))
        e2_feat = _compute_eq_features(item.get("equation2", ""))
        sep = first_separator(e1_feat, e2_feat)
        # STEP 0B features: only meaningful for nested bare E1
        top_shape = e1_feat.top_shape if (e1_feat.bare and proj == "nested") else "none"
        xtop      = e1_feat.xtop      if (e1_feat.bare and proj == "nested") else "none"
        return (qf.form_e1, qf.form_e2, _depth_bucket(qf.depth_e1),
                polarity, proj, sep, top_shape, xtop)
    except Exception:
        return ("GENERAL", "GENERAL", 2, polarity, "other", "none", "none", "none")


def partition_label(key: PartitionKey) -> str:
    """Human-readable label for a partition key.

    Handles the full 8-element magma key natively; falls back to a compact
    str() representation for any other key shape (e.g. BBH boolean (bool, bool, bool)).
    """
    if len(key) != 8:
        # Generic fallback for non-magma tasks
        return "_".join(str(v) for v in key)
    form_e1, form_e2, depth_b, polarity, proj_class, sep_fires, top_shape, xtop = key
    depth_str = ("d0", "d1", "d2+")[depth_b]
    label = f"{form_e1}→{form_e2}_{depth_str}_{polarity}"
    if proj_class != "other":
        label += f"_{proj_class}"
    if sep_fires != "none":
        label += f"_sep{sep_fires}"
    # Sub-partition nested bins by topShape and xTop
    if top_shape != "none":
        label += f"_ts{top_shape}"
    if xtop != "none":
        label += f"_xt{xtop}"
    return label


_FORM_DESC = {
    "TRIVIAL":   "x = x (tautology)",
    "SINGLETON": "x = y (forces all elements equal)",
    "ABSORBING": "one side is a variable absent from the other side",
    "STANDARD":  "lone variable appears on both sides",
    "GENERAL":   "both sides contain * operations",
}

_DEPTH_DESC = {
    0: "no * operators (depth 0)",
    1: "exactly one * operator (depth 1)",
    2: "two or more * operators (depth 2+)",
}

_PROJ_DESC = {
    "canon_l":    "E1 normalizes to a canonical left-projection law (x = x*(y*(z*(x*y))) or x = x*((y*z)*(z*z))): every product collapses to its left factor",
    "canon_r":    "E1 normalizes to the canonical right-projection law (x = (((y*z)*x)*z)*x): every product collapses to its right factor",
    "left_proj":  "E1 has the form x = x * A where A contains no x (left-projection structure: forces x*y = x in any satisfying magma)",
    "right_proj": "E1 has the form x = A * x where A contains no x (right-projection structure: forces x*y = y in any satisfying magma)",
    "nested":     "x appears multiple times or non-terminally in E1's RHS (not a clean projection form)",
}

_SEP_DESC = {
    "LP":  "LP(E1)=TRUE but LP(E2)=FALSE — leftmost-variable invariant violated → always FALSE",
    "RP":  "RP(E1)=TRUE but RP(E2)=FALSE — rightmost-variable invariant violated → always FALSE",
    "SET": "SET(E1)=TRUE but SET(E2)=FALSE — variable-set invariant violated → always FALSE",
    "XOR": "XOR(E1)=TRUE but XOR(E2)=FALSE — parity invariant violated → always FALSE",
    "AB":  "AB(E1)=TRUE but AB(E2)=FALSE — exact-count invariant violated → always FALSE",
}

_TOPSHAPE_DESC = {
    "v-m": "topShape(E1)=v-m — top split is (variable * product): bare variable on left of top *",
    "m-v": "topShape(E1)=m-v — top split is (product * variable): bare variable side on right of top *",
    "m-m": "topShape(E1)=m-m — top split is (product * product): bare variable is inside both children",
}

_XTOP_DESC = {
    "left":  "xTop(E1)=left  — bare variable x appears only in the left child of the top product split",
    "right": "xTop(E1)=right — bare variable x appears only in the right child of the top product split",
    "both":  "xTop(E1)=both  — bare variable x appears in both children of the top product split",
}


def partition_key_to_conditions(key: PartitionKey) -> list[str]:
    """
    Convert a PartitionKey to a list of structural ACTIVATE IF conditions
    expressed in the same plain-English format the LLM uses.

    These are injected as the first conditions in every generated case study,
    guaranteeing that the case study only fires on items from this partition.
    """
    form_e1, form_e2, depth_b, polarity, proj_class, sep_fires, top_shape, xtop = key
    conditions = [
        f"E1 is {form_e1} form ({_FORM_DESC.get(form_e1, form_e1.lower())})",
        f"E2 is {form_e2} form ({_FORM_DESC.get(form_e2, form_e2.lower())})",
        f"E1 has {_DEPTH_DESC.get(depth_b, 'unknown depth')}",
        f"Expected answer is {polarity}",
    ]
    if proj_class in _PROJ_DESC:
        conditions.append(_PROJ_DESC[proj_class])
    if sep_fires in _SEP_DESC:
        conditions.append(_SEP_DESC[sep_fires])
    if top_shape in _TOPSHAPE_DESC:
        conditions.append(_TOPSHAPE_DESC[top_shape])
    if xtop in _XTOP_DESC:
        conditions.append(_XTOP_DESC[xtop])
    return conditions


# ---------------------------------------------------------------------------
# PartitionBin
# ---------------------------------------------------------------------------

# Reservoir cap for the designated correct pool per partition.
# Smaller than the global CORRECT_POOL_MAX (40) used in ICR_select because
# items are already structurally homogeneous — fewer are needed for a
# representative regression check.
CORRECT_POOL_PER_PARTITION_MAX = 50

# Maximum archived candidates per partition bin.
# Each entry is (fix_rate, CaseStudy), sorted descending by fix_rate.
# Caps memory footprint while keeping the highest-scoring failed candidates
# for crossover and re-evaluation in future outer iterations.
ARCHIVE_MAX = 8


@dataclass
class PartitionBin:
    """
    All failures sharing the same structural partition key, paired with a
    designated correct pool drawn exclusively from the same structural class.

    correct_pool is used for regression checks: a case study generated for
    this partition should not regress items in the same structural class.
    Because the pool is structurally matched to the candidate's ACTIVATE IF
    conditions, this check is both tighter (fewer false passes) and cheaper
    (smaller pool) than a global reservoir.

    candidate_archive stores the best (fix_rate, CaseStudy) pairs from
    previous outer iterations that failed a gate.  Between iterations the
    failure set shifts, so archived candidates may now pass.  The top-2
    archive entries are also used to produce a crossover candidate each round.
    """
    key:               PartitionKey
    failures:          list[dict] = field(default_factory=list)
    correct_pool:      list[dict] = field(default_factory=list)
    solved:            bool       = False   # retired — skip in future iterations
    n_flushes:         int        = 0       # case studies accepted for this partition
    candidate_archive: list       = field(default_factory=list)
    # list[tuple[float, CaseStudy]] — sorted descending by fix_rate, capped at ARCHIVE_MAX

    @property
    def label(self) -> str:
        return partition_label(self.key)

    def add_correct(self, item: dict) -> None:
        """Reservoir-sample item into correct_pool (bounded at CORRECT_POOL_PER_PARTITION_MAX)."""
        n = len(self.correct_pool)
        if n < CORRECT_POOL_PER_PARTITION_MAX:
            self.correct_pool.append(item)
        else:
            # Reservoir sampling: replace a random existing slot
            j = random.randrange(n + 1)
            if j < CORRECT_POOL_PER_PARTITION_MAX:
                self.correct_pool[j] = item

    def archive_candidate(self, fix_rate: float, cs: "CaseStudy") -> None:
        """
        Add a failed candidate to the archive, keeping the top ARCHIVE_MAX by
        fix_rate.  Used for evolutionary re-evaluation and crossover.
        """
        self.candidate_archive.append((fix_rate, cs))
        self.candidate_archive.sort(key=lambda x: -x[0])
        if len(self.candidate_archive) > ARCHIVE_MAX:
            self.candidate_archive = self.candidate_archive[:ARCHIVE_MAX]

    def __len__(self) -> int:
        return len(self.failures)


# ---------------------------------------------------------------------------
# Build partitions from a scored pass
# ---------------------------------------------------------------------------

def build_partitions(
    wrong_items:         list[dict],
    correct_items:       list[dict],
    bin_threshold:       int = 3,
    partition_key_fn=None,
) -> dict[PartitionKey, PartitionBin]:
    """
    Route wrong and correct items into PartitionBins keyed by structural class.

    partition_key_fn: callable (item -> tuple) that computes the bin key.
    Defaults to item_partition_key (magma-specific) for backward compat.
    Pass task_spec.partition_key to use a different task's binning logic.

    A bin is only created when its failure count >= bin_threshold (not enough
    failures to form a meaningful teaching batch → skip).  Correct items are
    always routed to their partition's correct_pool regardless of whether a bin
    exists for that key.

    Returns the dict of active (failure-count >= bin_threshold) bins only.
    """
    _key_fn = partition_key_fn or item_partition_key

    # Build correct pools for all structural classes first
    correct_pools: dict[PartitionKey, list[dict]] = {}
    for item in correct_items:
        key = _key_fn(item)
        correct_pools.setdefault(key, []).append(item)

    # Group failures by partition key
    failure_groups: dict[PartitionKey, list[dict]] = {}
    for item in wrong_items:
        key = _key_fn(item)
        failure_groups.setdefault(key, []).append(item)

    bins: dict[PartitionKey, PartitionBin] = {}
    for key, failures in sorted(failure_groups.items(), key=lambda kv: -len(kv[1])):
        if len(failures) < bin_threshold:
            continue
        pb = PartitionBin(key=key, failures=list(failures))
        for item in correct_pools.get(key, []):
            pb.add_correct(item)
        bins[key] = pb

    return bins


# ---------------------------------------------------------------------------
# Refresh partitions after a re-score pass
# ---------------------------------------------------------------------------

def refresh_partitions(
    bins:                 dict[PartitionKey, PartitionBin],
    new_wrong:            list[dict],
    new_correct:          list[dict],
    retirement_threshold: int = 2,
    log_fn=None,
    partition_key_fn=None,
) -> None:
    """
    Update bin.failures from a fresh scoring pass and retire bins whose
    remaining failure count fell below retirement_threshold.

    Called after each outer iteration.  Mutates bins in-place.

    partition_key_fn: same function passed to build_partitions — must match.
    Defaults to item_partition_key (magma-specific) for backward compat.
    """
    _key_fn = partition_key_fn or item_partition_key

    # Re-index fresh wrong items by partition key
    fresh_wrong: dict[PartitionKey, list[dict]] = {}
    for item in new_wrong:
        key = _key_fn(item)
        fresh_wrong.setdefault(key, []).append(item)

    # Absorb new correct items into partition correct pools
    for item in new_correct:
        key = _key_fn(item)
        if key in bins:
            bins[key].add_correct(item)

    # Update failure lists and check retirement
    for key, pb in bins.items():
        if pb.solved:
            continue
        old_n = len(pb.failures)
        pb.failures = fresh_wrong.get(key, [])
        new_n = len(pb.failures)
        if new_n < retirement_threshold:
            pb.solved = True
            if log_fn:
                log_fn(
                    f"  [partition:{pb.label}] retired — "
                    f"failures {old_n} → {new_n} < threshold={retirement_threshold}"
                )
        elif log_fn and new_n != old_n:
            log_fn(f"  [partition:{pb.label}] failures {old_n} → {new_n}")


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def partition_summary(bins: dict[PartitionKey, PartitionBin]) -> list[dict]:
    """Return a serialisable per-bin summary for logging."""
    rows = []
    for pb in sorted(bins.values(), key=lambda b: -len(b)):
        rows.append({
            "partition":    pb.label,
            "failures":     len(pb.failures),
            "correct_pool": len(pb.correct_pool),
            "n_flushes":    pb.n_flushes,
            "solved":       pb.solved,
        })
    return rows


def print_partition_table(
    bins:    dict[PartitionKey, PartitionBin],
    title:   str = "PARTITION SUMMARY",
    file=sys.stderr,
) -> None:
    print(f"\n{'='*65}", file=file)
    print(title, file=file)
    print(f"{'='*65}", file=file)
    print(f"  {'Partition':<38} {'Fail':>4} {'Corr':>5} {'Flush':>5} {'Status'}", file=file)
    print(f"  {'-'*63}", file=file)
    for pb in sorted(bins.values(), key=lambda b: -len(b)):
        status = "retired" if pb.solved else "active"
        print(
            f"  {pb.label:<38} {len(pb.failures):>4} "
            f"{len(pb.correct_pool):>5} {pb.n_flushes:>5}  {status}",
            file=file,
        )
    active = sum(1 for pb in bins.values() if not pb.solved)
    print(f"\n  active={active}  total={len(bins)}", file=file)
