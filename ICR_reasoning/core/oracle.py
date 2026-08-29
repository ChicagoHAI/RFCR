"""
ICR_reasoning/core/oracle.py — Load GPT-5.4 oracle reasoning for contrast injection.

The oracle dict maps (equation1, equation2) → correct_reasoning_text, sourced from
gpt5.4_normal_default.csv (97.8% accuracy on normal).  When a failure item has a
matching oracle entry, the case study generator sees both the wrong model reasoning
AND a correct reasoning trace — giving it a concrete contrast to learn from.

Only entries where correct == "True" are included.  The REASONING section is extracted
from the full VERDICT/REASONING/PROOF response (the most useful part for the generator,
which already knows the verdict from the item's answer field).
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_reasoning(response: str) -> str:
    """
    Pull out the REASONING block from a VERDICT/REASONING/PROOF response.

    Returns an empty string if the block can't be found (e.g. the model
    responded with only VERDICT+PROOF and no explicit REASONING header).
    """
    # Match from REASONING: up to (but not including) the next section header or end
    m = re.search(
        r"REASONING\s*:\s*\n?(.*?)(?=\n(?:PROOF|VERDICT)\s*:|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # Fallback: try everything after REASONING: to end
    m2 = re.search(r"REASONING\s*:\s*\n?(.*)", response, re.DOTALL | re.IGNORECASE)
    return m2.group(1).strip() if m2 else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

OracleDict = dict[tuple[str, str], str]   # (eq1, eq2) → reasoning


def load_oracle_csv(csv_path: Path | str) -> OracleDict:
    """
    Load oracle reasoning from gpt5.4_normal_default.csv.

    Returns a dict keyed by (equation1, equation2) → reasoning text.
    Only correct entries are included.  When the same (eq1, eq2) pair appears
    multiple times (repeat_id > 0), the first correct entry wins.
    """
    oracle: OracleDict = {}
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"  [oracle] WARNING: CSV not found at {csv_path}", file=sys.stderr)
        return oracle

    loaded = skipped = 0
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("correct", "")).strip().lower() != "true":
                skipped += 1
                continue
            key = (row["equation1"].strip(), row["equation2"].strip())
            if key in oracle:
                continue   # keep first occurrence only
            reasoning = _extract_reasoning(row["response"])
            if reasoning:
                oracle[key] = reasoning
                loaded += 1

    print(
        f"  [oracle] Loaded {loaded} correct reasoning traces from {csv_path.name} "
        f"({skipped} incorrect entries skipped)",
        file=sys.stderr,
    )
    return oracle
