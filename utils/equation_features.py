"""
utils/equation_features.py — Structural feature computation for equation pairs.

Used by:
  - ICR_partition/training/partition.py  (partition key enrichment)
  - utils/cheatsheet.py                  (QueryFeatures extension)

Feature definitions — base 9
------------------------------
  size(E)     total variable occurrences on both sides
  vars(E)     distinct variable letters
  imb(E)      Σ_v |left_count(v) − right_count(v)|   (imbalance)
  bare(E)     TRUE if exactly one side is a single variable with no *
  lp(E)       TRUE if leftmost variable matches on both sides
  rp(E)       TRUE if rightmost variable matches on both sides
  set_eq(E)   TRUE if both sides use the exact same set of distinct variables
  xor(E)      TRUE if every variable has the same parity on both sides
  ab(E)       TRUE if every variable has the exact same total count on both sides

STEP 0B features (bare laws only — "none"/False for non-bare)
--------------------------------------------------------------
  lx(E)       leftmost variable on the product side equals the bare variable x
  rx(E)       rightmost variable on the product side equals the bare variable x
  xtop(E)     where x appears in the top product split: "left"|"right"|"both"|"none"
  top_shape(E) shape of the top product split: "v-m"|"m-v"|"m-m"|"none"
  square(E)   product side contains a subterm u*u (same variable both sides of one *)
  rhs_vars(E) distinct variable count on the product side

Separator tests
---------------
first_separator(e1, e2) → the first invariant that holds for E1 but not E2:
  "LP" | "RP" | "SET" | "XOR" | "AB" | "none"

Collapse detection
------------------
detect_collapse(eq) → "left_proj" | "right_proj" | "none"
Matches E1 against three canonical bare laws using exact normalization.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Per-equation features
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EqFeatures:
    # Base 9 features
    size:      int
    vars:      int
    imb:       int
    bare:      bool
    lp:        bool
    rp:        bool
    set_eq:    bool
    xor:       bool
    ab:        bool
    # STEP 0B features (meaningful only for bare laws)
    lx:        bool = False   # leftmost product-side var == bare var
    rx:        bool = False   # rightmost product-side var == bare var
    xtop:      str  = "none"  # "left"|"right"|"both"|"none"
    top_shape: str  = "none"  # "v-m"|"m-v"|"m-m"|"none"
    square:    bool = False   # product side contains u*u subterm
    rhs_vars:  int  = 0       # distinct vars on product side


# ---------------------------------------------------------------------------
# Per-pair derived info
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairFeatures:
    e1:            EqFeatures
    e2:            EqFeatures
    sep_fires:     str   # "LP"|"RP"|"SET"|"XOR"|"AB"|"none"
    collapse_type: str   # "left_proj"|"right_proj"|"none"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _split_eq(eq: str) -> tuple[str, str]:
    """Split 'L = R' at the first bare '=' (not inside parentheses)."""
    depth = 0
    for i, ch in enumerate(eq):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "=" and depth == 0:
            return eq[:i].strip(), eq[i + 1:].strip()
    parts = eq.split("=", 1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (eq.strip(), "")


def _var_list(expr: str) -> list[str]:
    """Variable letters in left-to-right order."""
    return re.findall(r"\b([a-zA-Z])\b", expr)


def _is_bare(expr: str) -> bool:
    """True if the expression is a single variable with no *."""
    return bool(re.fullmatch(r"[a-zA-Z]", expr.replace(" ", "")))


# ---------------------------------------------------------------------------
# STEP 0B expression-tree helpers
# ---------------------------------------------------------------------------

def _strip_outer_parens(expr: str) -> str:
    """
    Remove one layer of outer parentheses if they wrap the entire expression.
    E.g. "(x * y)" → "x * y",  "(x * y) * z" → unchanged.
    """
    expr = expr.strip()
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0:
            # If the first '(' closes before the end, outer parens don't wrap all
            if i < len(expr) - 1:
                return expr
            # It wraps all → strip
            return expr[1:-1].strip()
    return expr


def _top_split(expr: str) -> tuple[str, str] | None:
    """
    Find the first top-level '*' (depth 0 after stripping outer parens) and
    return (left_child, right_child).  Returns None if no such '*' exists.
    """
    expr = _strip_outer_parens(expr)
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "*" and depth == 0:
            return expr[:i].strip(), expr[i + 1:].strip()
    return None


def _has_star(expr: str) -> bool:
    """True if the expression contains any '*' operator."""
    return "*" in expr


def _compute_top_shape(product_side: str) -> str:
    """
    Classify the top-level split of a product-side expression.
      "v-m"  — (variable) * (product)
      "m-v"  — (product) * (variable)
      "m-m"  — (product) * (product)
      "none" — no top-level split found (e.g. bare variable or empty)
    """
    split = _top_split(product_side)
    if split is None:
        return "none"
    left, right = split
    left_is_var  = _is_bare(left)
    right_is_var = _is_bare(right)
    if left_is_var and not right_is_var:
        return "v-m"
    if not left_is_var and right_is_var:
        return "m-v"
    if not left_is_var and not right_is_var:
        return "m-m"
    # Both vars (x = x * y case) — unusual but handle gracefully
    return "v-v"


def _compute_xtop(product_side: str, bare_var: str) -> str:
    """
    Determine where the bare variable appears in the top-level split.
      "left"  — bare_var only in left child
      "right" — bare_var only in right child
      "both"  — bare_var in both children
      "none"  — no top-level split, or bare_var absent
    """
    split = _top_split(product_side)
    if split is None:
        return "none"
    left, right = split
    # Check using word-boundary regex to avoid matching substrings
    pattern = re.compile(rf"\b{re.escape(bare_var)}\b")
    in_left  = bool(pattern.search(left))
    in_right = bool(pattern.search(right))
    if in_left and in_right:
        return "both"
    if in_left:
        return "left"
    if in_right:
        return "right"
    return "none"


def _compute_square(product_side: str) -> bool:
    """
    True if the product side contains any subterm of the form u*u
    (same single variable on both sides of one '*').
    Matches e.g. "z * z", "y*y" at any depth.
    """
    return bool(re.search(r"\b([a-zA-Z])\s*\*\s*\1\b", product_side))


def _compute_step0b_features(
    eq: str,
) -> dict:
    """
    Compute STEP 0B features for a bare law.
    Returns a dict with keys: lx, rx, xtop, top_shape, square, rhs_vars.
    Returns all-default values if the equation is not bare.
    """
    defaults = dict(lx=False, rx=False, xtop="none", top_shape="none",
                    square=False, rhs_vars=0)
    lhs, rhs = _split_eq(eq)
    lhs_bare = _is_bare(lhs)
    rhs_bare = _is_bare(rhs)

    if not (lhs_bare or rhs_bare):
        return defaults

    # Ensure bare variable is on the left
    if rhs_bare and not lhs_bare:
        lhs, rhs = rhs, lhs

    bare_var = lhs.strip()

    # product side = rhs
    product_side = rhs
    vars_on_product = _var_list(product_side)

    if not vars_on_product:
        return defaults

    lx        = (vars_on_product[0] == bare_var)
    rx        = (vars_on_product[-1] == bare_var)
    rhs_vars  = len(set(vars_on_product))
    xtop      = _compute_xtop(product_side, bare_var)
    top_shape = _compute_top_shape(product_side)
    square    = _compute_square(product_side)

    return dict(lx=lx, rx=rx, xtop=xtop, top_shape=top_shape,
                square=square, rhs_vars=rhs_vars)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(eq: str) -> EqFeatures:
    """Compute all structural features for one equation string."""
    lhs, rhs = _split_eq(eq)
    lv = _var_list(lhs)
    rv = _var_list(rhs)
    distinct = set(lv + rv)

    lc = Counter(lv)
    rc = Counter(rv)

    size  = len(lv) + len(rv)
    v_cnt = len(distinct)
    imb   = sum(abs(lc.get(v, 0) - rc.get(v, 0)) for v in distinct)
    bare  = _is_bare(lhs) or _is_bare(rhs)
    lp    = bool(lv and rv and lv[0] == rv[0])
    rp    = bool(lv and rv and lv[-1] == rv[-1])
    set_e = set(lv) == set(rv)
    xor   = all((lc.get(v, 0) % 2) == (rc.get(v, 0) % 2) for v in distinct)
    ab    = all(lc.get(v, 0) == rc.get(v, 0) for v in distinct)

    # STEP 0B features
    s0b = _compute_step0b_features(eq) if bare else {}

    return EqFeatures(
        size=size, vars=v_cnt, imb=imb, bare=bare,
        lp=lp, rp=rp, set_eq=set_e, xor=xor, ab=ab,
        lx=s0b.get("lx", False),
        rx=s0b.get("rx", False),
        xtop=s0b.get("xtop", "none"),
        top_shape=s0b.get("top_shape", "none"),
        square=s0b.get("square", False),
        rhs_vars=s0b.get("rhs_vars", 0),
    )


# ---------------------------------------------------------------------------
# Separator tests
# ---------------------------------------------------------------------------

_SEP_ORDER = ("LP", "RP", "SET", "XOR", "AB")


def first_separator(e1: EqFeatures, e2: EqFeatures) -> str:
    """Return the name of the first separator invariant that fires, or 'none'."""
    checks = {
        "LP":  (e1.lp,     e2.lp),
        "RP":  (e1.rp,     e2.rp),
        "SET": (e1.set_eq, e2.set_eq),
        "XOR": (e1.xor,    e2.xor),
        "AB":  (e1.ab,     e2.ab),
    }
    for name in _SEP_ORDER:
        a, b = checks[name]
        if a and not b:
            return name
    return "none"


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------

# Canonical bare laws after renaming (all spaces removed).
# Convention: bare variable → x; others by first appearance in product side → y, z, w, u, v
_CANON_LEFT: frozenset[str] = frozenset({
    "x=x*(y*(z*(x*y)))",    # CANON-L1
    "x=x*((y*z)*(z*z))",   # CANON-L2
})
_CANON_RIGHT: frozenset[str] = frozenset({
    "x=(((y*z)*x)*z)*x",   # CANON-R1
})


def _normalize_bare_law(eq: str) -> str | None:
    """
    Normalize a bare law for canonical comparison.
    Returns the normalized string (spaces stripped) or None if eq is not bare.
    """
    lhs, rhs = _split_eq(eq)
    lhs_bare = _is_bare(lhs)
    rhs_bare = _is_bare(rhs)

    if not (lhs_bare or rhs_bare):
        return None

    # Ensure the bare variable is on the left
    if rhs_bare and not lhs_bare:
        lhs, rhs = rhs, lhs

    bare_var = lhs.strip()
    mapping: dict[str, str] = {bare_var: "x"}
    rename_seq = iter(["y", "z", "w", "u", "v"])

    for v in re.findall(r"[a-zA-Z]", rhs):
        if v not in mapping:
            try:
                mapping[v] = next(rename_seq)
            except StopIteration:
                mapping[v] = v  # too many vars — won't match any canon

    def _apply(s: str) -> str:
        return re.sub(r"[a-zA-Z]", lambda m: mapping.get(m.group(), m.group()), s)

    return (_apply(lhs) + "=" + _apply(rhs)).replace(" ", "")


def detect_collapse(eq: str) -> str:
    """
    Returns 'left_proj', 'right_proj', or 'none'.
    Only fires on the exact canonical source forms after normalization.
    """
    norm = _normalize_bare_law(eq)
    if norm is None:
        return "none"
    if norm in _CANON_LEFT:
        return "left_proj"
    if norm in _CANON_RIGHT:
        return "right_proj"
    return "none"


# ---------------------------------------------------------------------------
# Pair-level API
# ---------------------------------------------------------------------------

def compute_pair_features(e1_str: str, e2_str: str) -> PairFeatures:
    """Compute all structural features for an (E1, E2) pair."""
    f1 = compute_features(e1_str)
    f2 = compute_features(e2_str)
    return PairFeatures(
        e1=f1,
        e2=f2,
        sep_fires=first_separator(f1, f2),
        collapse_type=detect_collapse(e1_str),
    )
