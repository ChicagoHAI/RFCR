"""Aggregate the GEPA baseline results: pooled paired task-stratified bootstrap
CI (GEPA vs locked CS-ICL anchor) and summary.csv.

Usage: python analyze_gepa.py    (mirrors promptwizard/analyze_pw.py)
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ["disambiguation_qa", "geometric_shapes", "formal_fallacies", "object_counting"]
BOOT_N = 10000
BOOT_SEED = 20260712

RFCR_FIX, RFCR_REG = 11, 0


def main():
    per_task = {}
    items_by_task = {}
    for task in TASKS:
        rows = [json.loads(l) for l in (HERE / task / "test_eval_per_item.jsonl").read_text().splitlines()]
        per_task[task] = json.loads((HERE / task / "test_eval_summary.json").read_text())
        items_by_task[task] = rows

    rng = random.Random(BOOT_SEED)
    n_total = sum(len(v) for v in items_by_task.values())
    observed = sum(int(r["gepa_correct"]) - int(r["anchor_correct"])
                   for rows in items_by_task.values() for r in rows) / n_total
    samples = []
    for _ in range(BOOT_N):
        tot = 0
        for rows in items_by_task.values():
            n = len(rows)
            for _ in range(n):
                r = rows[rng.randrange(n)]
                tot += int(r["gepa_correct"]) - int(r["anchor_correct"])
        samples.append(tot / n_total)
    samples.sort()

    def q(p):
        pos = (len(samples) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        return samples[lo] * (1 - (pos - lo)) + samples[hi] * (pos - lo)

    le0 = sum(1 for x in samples if x <= 0) / BOOT_N
    ge0 = sum(1 for x in samples if x >= 0) / BOOT_N
    boot = {
        "n_items": n_total,
        "n_boot": BOOT_N,
        "seed": BOOT_SEED,
        "stratified_by": "task",
        "delta_accuracy_observed_gepa_minus_anchor": observed,
        "ci95_low": q(0.025),
        "ci95_high": q(0.975),
        "bootstrap_p_two_sided_against_zero": min(1.0, 2.0 * min(le0, ge0)),
    }
    (HERE / "paired_bootstrap_pooled.json").write_text(json.dumps(boot, indent=2))

    fields = ["task", "n_items", "gepa_correct", "gepa_accuracy", "anchor_correct",
              "anchor_accuracy", "fix", "regression", "net", "delta_pp",
              "verdict_parse_rate", "failed_calls", "max_tokens"]
    csv_rows = []
    tot = {k: 0 for k in ["n_items", "gepa_correct", "anchor_correct", "fix", "regression", "failed_calls"]}
    for task in TASKS:
        s = per_task[task]
        csv_rows.append({k: s[k] for k in fields if k in s} | {"task": task})
        for k in tot:
            tot[k] += s[k]
    pooled = {
        "task": "POOLED",
        "n_items": tot["n_items"],
        "gepa_correct": tot["gepa_correct"],
        "gepa_accuracy": tot["gepa_correct"] / tot["n_items"],
        "anchor_correct": tot["anchor_correct"],
        "anchor_accuracy": tot["anchor_correct"] / tot["n_items"],
        "fix": tot["fix"],
        "regression": tot["regression"],
        "net": tot["fix"] - tot["regression"],
        "delta_pp": (tot["gepa_correct"] - tot["anchor_correct"]) / tot["n_items"] * 100,
        "verdict_parse_rate": sum(int(r["verdict_parsed"]) for rows in items_by_task.values() for r in rows) / tot["n_items"],
        "failed_calls": tot["failed_calls"],
        "max_tokens": "per-task",
    }
    csv_rows.append(pooled)
    with (HERE / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    print(json.dumps({"pooled": pooled, "bootstrap": boot}, indent=2))
    print(f"\nRFCR contrast: GEPA {tot['fix']} fix / {tot['regression']} reg "
          f"vs RFCR repaired route {RFCR_FIX} fix / {RFCR_REG} reg on the same 400 items")


if __name__ == "__main__":
    main()
