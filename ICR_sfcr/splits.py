"""ICR_sfcr/splits.py — Training-data split for SFCR.

Splits training items into two disjoint sets:
  rule_gen  — used to compute failure regions and generate candidate rules
  gate      — held out; used exclusively to validate candidate rules

The anchor cheatsheet is assumed to have been generated from a separate
bootstrap pool (e.g. CS_ICL_Initial_Prompt files). That pool is not
re-partitioned here; we only split what's passed in as `items`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SFCRSplits:
    rule_gen: list[dict]
    gate: list[dict]


def make_splits(
    items: list[dict],
    rule_gen_n: int = 60,
    gate_n: int = 40,
    seed: int = 1000,
) -> SFCRSplits:
    """
    Shuffle items and slice into rule_gen / gate. The two halves are disjoint.

    If `rule_gen_n + gate_n > len(items)` both sizes are scaled proportionally
    to fit, preserving their ratio, with a floor of 3 each.
    """
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    total = rule_gen_n + gate_n
    if total > n:
        scale = n / total
        rule_gen_n = max(3, int(rule_gen_n * scale))
        gate_n = max(3, int(gate_n * scale))
        print(
            f"[splits] dataset has {n} items — scaled to "
            f"rule_gen={rule_gen_n}, gate={gate_n}"
        )

    rule_gen = shuffled[:rule_gen_n]
    gate = shuffled[rule_gen_n : rule_gen_n + gate_n]

    print(
        f"[splits] rule_gen={len(rule_gen)}  gate={len(gate)}  "
        f"unused={n - len(rule_gen) - len(gate)}  seed={seed}"
    )
    return SFCRSplits(rule_gen=rule_gen, gate=gate)
