"""
utils/oracle_index.py — Nearest-neighbour oracle lookup for disagreement mining.

For each student-wrong item, this module finds the structurally most similar
oracle entry — a different equation pair that the teacher (oracle) answered
correctly and whose form is closest to the student's failing pair.

This enables "disagreement bin" routing in the training loop:

    teacher-correct / student-wrong → disagree_bin  (high-priority flush)
    teacher-wrong   / student-wrong → both_wrong_bin (deprioritized)

Only disagreement items carry a distillable teacher signal.  Items where both
fail are collected separately and only flushed when the disagreement bin is too
small to trigger on its own.

Similarity measure
------------------
Structural Jaccard on QueryFeatures.tokens():
  {form_e1, form_e2, "vars<N>", "L<N>"} for both the query and oracle entry.

Union size is at most 6 tokens per side (12 total), so Jaccard is fast.
A match is accepted when Jaccard ≥ min_similarity (default 0.25).

Usage
-----
    from utils.oracle_index import OracleIndex
    from ICR_reasoning.core.oracle import load_oracle_csv

    oracle = load_oracle_csv("oracle.csv")
    idx    = OracleIndex(oracle)

    entry, sim = idx.find_nearest(item) or (None, 0.0)
    if entry:
        item["oracle_nearest"] = entry.to_dict()
        item["oracle_sim"]     = sim
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import NamedTuple

from utils.cheatsheet import QueryFeatures, _features_from_pair


# ---------------------------------------------------------------------------
# OracleEntry
# ---------------------------------------------------------------------------

class OracleEntry(NamedTuple):
    """One oracle record — a correct teacher answer for an equation pair."""
    eq1:       str
    eq2:       str
    reasoning: str
    features:  QueryFeatures

    def to_dict(self) -> dict:
        return {
            "eq1":       self.eq1,
            "eq2":       self.eq2,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# OracleIndex
# ---------------------------------------------------------------------------

@dataclass
class OracleIndex:
    """
    Index of oracle entries searchable by structural similarity.

    Parameters
    ----------
    oracle          : OracleDict — (eq1, eq2) → reasoning_text
    min_similarity  : Jaccard threshold below which no match is returned.
                      Default 0.25 — requires at least one shared form token.
    """
    _entries:       list[OracleEntry]
    _min_sim:       float

    def __init__(
        self,
        oracle: dict[tuple[str, str], str],
        min_similarity: float = 0.25,
    ) -> None:
        self._min_sim = min_similarity
        self._entries = []
        for (eq1, eq2), reasoning in oracle.items():
            try:
                features = _features_from_pair(eq1, eq2)
            except Exception:
                continue
            self._entries.append(OracleEntry(eq1=eq1, eq2=eq2,
                                              reasoning=reasoning,
                                              features=features))
        print(
            f"  [oracle_index] Built index with {len(self._entries)} entries "
            f"(min_sim={min_similarity})",
            file=sys.stderr,
        )

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def find_nearest(
        self,
        item: dict,
    ) -> tuple[OracleEntry, float] | None:
        """
        Return the (OracleEntry, similarity) pair whose structural features
        are most similar to item["equation1"] / item["equation2"].

        Returns None if no entry meets min_similarity or the index is empty.

        Exact-key matches (same equation pair) are excluded — the existing
        oracle exact-match path already handles those.  This method is for
        cross-pair structural neighbours.
        """
        if not self._entries:
            return None

        try:
            qf = _features_from_pair(
                str(item.get("equation1", "")).strip(),
                str(item.get("equation2", "")).strip(),
            )
        except Exception:
            return None

        q_tokens   = qf.tokens()
        item_eq1   = str(item.get("equation1", "")).strip()
        item_eq2   = str(item.get("equation2", "")).strip()

        best_entry: OracleEntry | None = None
        best_sim   = -1.0

        for entry in self._entries:
            # Skip exact-key matches — let the existing oracle path handle those
            if entry.eq1 == item_eq1 and entry.eq2 == item_eq2:
                continue

            cs_tokens = entry.features.tokens()
            union     = q_tokens | cs_tokens
            sim       = len(q_tokens & cs_tokens) / len(union) if union else 0.0

            if sim > best_sim:
                best_sim   = sim
                best_entry = entry

        if best_entry is None or best_sim < self._min_sim:
            return None

        return (best_entry, best_sim)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def coverage(self, items: list[dict]) -> float:
        """
        Fraction of items in a list that have at least one oracle nearest match.
        Useful for logging at startup.
        """
        if not items:
            return 0.0
        matched = sum(1 for it in items if self.find_nearest(it) is not None)
        return matched / len(items)
