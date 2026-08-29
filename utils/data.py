"""
data.py — Dataset loading, sampling, splitting utilities, and shared training primitives.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

def is_true(answer) -> bool:
    """Normalise a JSON bool or string answer to Python bool."""
    if isinstance(answer, bool):
        return answer
    return str(answer).strip().upper() == "TRUE"

# Backward-compatible alias — prefer is_true in new code.
_is_true = is_true


# ---------------------------------------------------------------------------
# Shared training primitive
# ---------------------------------------------------------------------------

@dataclass
class FailureBin:
    """Accumulates wrong items until the threshold is reached, then flushes."""
    threshold: int
    _items: list[dict] = field(default_factory=list, init=False)

    def add(self, item: dict) -> None:
        self._items.append(item)

    def is_full(self) -> bool:
        return len(self._items) >= self.threshold

    def flush(self) -> list[dict]:
        items = self._items[:]
        self._items.clear()
        return items

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class DisagreementBin:
    """
    Priority failure bin for teacher-correct / student-wrong pairs.

    Each item added here must carry an "oracle_nearest" field (dict with
    eq1, eq2, reasoning) attached by the training loop after nearest-neighbour
    oracle lookup.  Items where the oracle has no distillable signal (both
    teacher and student wrong) go to a regular FailureBin instead.

    Flushed before the fallback FailureBin — disagreement items provide a
    richer teaching signal because the prompt includes both wrong student
    reasoning and a structurally similar correct oracle trace.
    """
    threshold: int
    _items: list[dict] = field(default_factory=list, init=False)

    def add(self, item: dict) -> None:
        assert "oracle_nearest" in item, (
            "DisagreementBin.add() requires item['oracle_nearest'] to be set"
        )
        self._items.append(item)

    def is_full(self) -> bool:
        return len(self._items) >= self.threshold

    def flush(self) -> list[dict]:
        items = self._items[:]
        self._items.clear()
        return items

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load all non-empty lines from a .jsonl file."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_instances(
    items: list[dict],
    n_true: int,
    n_false: int,
    seed: int,
) -> list[dict]:
    """
    Return up to n_true TRUE and n_false FALSE examples, shuffled together.
    Prints a warning to stderr if the dataset has fewer examples than requested.
    """
    rng = random.Random(seed)
    true_items  = [it for it in items if is_true(it["answer"])]
    false_items = [it for it in items if not is_true(it["answer"])]

    actual_true  = min(n_true,  len(true_items))
    actual_false = min(n_false, len(false_items))

    if actual_true < n_true:
        print(
            f"Warning: only {len(true_items)} TRUE examples available, requested {n_true}.",
            file=sys.stderr,
        )
    if actual_false < n_false:
        print(
            f"Warning: only {len(false_items)} FALSE examples available, requested {n_false}.",
            file=sys.stderr,
        )

    combined = rng.sample(true_items, actual_true) + rng.sample(false_items, actual_false)
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Train / val / seed splitting
# ---------------------------------------------------------------------------

def split_dataset(
    items: list[dict],
    val_fraction: float,
    seed_examples: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split into (seed, train, val).

    seed_examples items (balanced TRUE/FALSE) are reserved for initial
    cheatsheet generation.  val_fraction of the remainder become the
    validation set.  The rest become the training set for the loop.

    Returns (seed_set, train_set, val_set).
    """
    rng = random.Random(seed)

    true_items  = [it for it in items if is_true(it["answer"])]
    false_items = [it for it in items if not is_true(it["answer"])]

    def _split_one(lst: list) -> tuple[list, list, list]:
        shuffled = lst[:]
        rng.shuffle(shuffled)
        n_seed = min(seed_examples // 2, len(shuffled))
        seed_part = shuffled[:n_seed]
        rest = shuffled[n_seed:]
        n_val = max(1, round(len(rest) * val_fraction)) if val_fraction > 0 else 0
        return seed_part, rest[n_val:], rest[:n_val]

    s_t, tr_t, v_t = _split_one(true_items)
    s_f, tr_f, v_f = _split_one(false_items)

    seed_set  = s_t + s_f
    train_set = tr_t + tr_f
    val_set   = v_t + v_f

    rng.shuffle(seed_set)
    rng.shuffle(train_set)
    rng.shuffle(val_set)

    return seed_set, train_set, val_set
