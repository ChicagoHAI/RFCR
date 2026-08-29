"""Zero-leak audit for the GEPA baseline (ARR rebuttal).

Usage: python audit_gepa_leak.py <task>

Same normalization/marker approach as rebuttal_experiments/e_qwen/audit_e_qwen.py:
- normalize whitespace + lowercase; per-task distinctive-substring markers so
  matches are per-item, not per-template.
- Documents audited: every logged LLM request message (task + reflection) and
  every logged response from {task}/llm_calls.jsonl, every candidate prompt
  GEPA constructed ({task}/gepa_run/candidates.json), plus the frozen
  seed_prompt.txt and optimized_prompt.txt.
- Checks: 0 test-item matches anywhere; detector sanity: train items MUST
  match (they are in the optimization data); train/test key overlap must be 0.
Writes {task}/leak_audit.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

TASK = sys.argv[1]
HERE = Path(__file__).resolve().parent
ICR = HERE.parents[1] / "ICRefine"
OUT = HERE / TASK

MARKERS = {
    "disambiguation_qa": "sentence:",
    "geometric_shapes": "path d=",
    "formal_fallacies": "",
    "object_counting": "",
}
MARKER = MARKERS[TASK]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def distinctive(t: str) -> str:
    i = t.find(MARKER)
    return t[i:] if i >= 0 else t


def keys(path: Path) -> list[str]:
    return [distinctive(norm(json.loads(l)["input"]))
            for l in path.read_text().splitlines() if l.strip()]


test_keys = keys(ICR / f"datasets/bbh/{TASK}_test.jsonl")
train_keys = keys(ICR / f"datasets/bbh/{TASK}_train.jsonl")
assert (not MARKER) or all(MARKER in k[: len(MARKER) + 2] for k in test_keys), \
    f"marker {MARKER!r} missing from some test items"
overlap = set(test_keys) & set(train_keys)

# ---- collect documents -------------------------------------------------------
docs: list[tuple[str, str]] = []  # (label, text)
n_llm_calls = 0
with (OUT / "llm_calls.jsonl").open() as fh:
    for i, line in enumerate(fh):
        c = json.loads(line)
        n_llm_calls += 1
        for m in c.get("messages", []):
            docs.append((f"call{i}:{c['kind']}:{m['role']}", m.get("content") or ""))
        docs.append((f"call{i}:{c['kind']}:response", c.get("response") or ""))

candidates = json.loads((OUT / "gepa_run" / "candidates.json").read_text())
cand_list = candidates if isinstance(candidates, list) else [candidates]
for j, cand in enumerate(cand_list):
    if isinstance(cand, dict):
        for name, text in cand.items():
            docs.append((f"candidate{j}:{name}", text))
    else:
        docs.append((f"candidate{j}", str(cand)))

seed_text = (OUT / "seed_prompt.txt").read_text()
opt_text = (OUT / "optimized_prompt.txt").read_text()
docs.append(("frozen:seed_prompt.txt", seed_text))
docs.append(("frozen:optimized_prompt.txt", opt_text))

# ---- scan --------------------------------------------------------------------
test_matches = 0
train_hits = 0
examples = []
for label, text in docs:
    p = norm(text)
    hit = [k for k in test_keys if k in p]
    if hit:
        test_matches += len(hit)
        examples.append({"doc": label, "first_match": hit[0][:120]})
    train_hits += sum(1 for k in train_keys if k in p)

report = {
    "task": TASK,
    "marker": MARKER,
    "n_test_items": len(test_keys),
    "n_llm_calls_audited": n_llm_calls,
    "n_docs_audited": len(docs),
    "n_candidates_audited": len(cand_list),
    "test_item_matches (must be 0)": test_matches,
    "detector_sanity_train_item_matches (must be > 0)": train_hits,
    "train_test_key_overlap (must be 0)": len(overlap),
    "examples": examples[:3],
    "seed_prompt_sha256": hashlib.sha256(seed_text.encode()).hexdigest(),
    "optimized_prompt_sha256": hashlib.sha256(opt_text.encode()).hexdigest(),
}
(OUT / "leak_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False))
