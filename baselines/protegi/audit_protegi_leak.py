"""Zero-leak audit for a ProTeGi optimization run (ARR rebuttal baseline).

Usage: python audit_protegi_leak.py <task>

Same normalization/marker approach as rebuttal_experiments/e_qwen/
audit_e_qwen.py and the PromptWizard audit: every prompt AND every response
ProTeGi produced during optimization (rebuttal_protegi/runs/<task>/
llm_calls.jsonl in the LMOps clone — this includes all minibatch/UCB scoring
prompts, gradient prompts, prompt rewrites, and synonym generations), plus
the frozen artifacts (seed prompt, optimized cheatsheet, per-round results),
is scanned for the distinctive normalized text of every test item.
Test items must match ZERO times; train items must match (>0) as detector
sanity. Also checks train/test key disjointness.

Emits <task>/leak_audit.json next to this script.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICR = HERE.parents[1] / "ICRefine"
PROTEGI_BASE = Path("<PROTEGI_WORKDIR>")

TASK = sys.argv[1]
RUN = PROTEGI_BASE / "runs" / TASK

# task-specific distinctive-substring marker (same as audit_e_qwen.py:
# whole-input keys for ff/oc, sentence:/path d= markers for dq/gs)
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


test_keys = [distinctive(norm(json.loads(l)["input"]))
             for l in (ICR / f"datasets/bbh/{TASK}_test.jsonl").read_text().splitlines() if l.strip()]
train_keys = [distinctive(norm(json.loads(l)["input"]))
              for l in (ICR / f"datasets/bbh/{TASK}_train.jsonl").read_text().splitlines() if l.strip()]
overlap = set(test_keys) & set(train_keys)

# ---- corpus: every optimization-time prompt/response + frozen artifacts ----
docs = []  # (label, normalized_text)
llm_calls = [json.loads(l) for l in (RUN / "llm_calls.jsonl").read_text().splitlines() if l.strip()]
for c in llm_calls:
    i = c["i"]
    docs.append((f"llm_call_{i}_prompt", norm(c["prompt"])))
    for j, r in enumerate(c.get("responses") or []):
        docs.append((f"llm_call_{i}_response_{j}", norm(r or "")))

artifact_paths = [
    PROTEGI_BASE / "seeds" / f"{TASK}_seed_prompt.md",
    PROTEGI_BASE / "data" / TASK / "train.jsonl",
    RUN / "best_candidate.md",
    RUN / "optimized_cheatsheet.txt",
    RUN / "results.txt",
    HERE / TASK / "optimized_prompt.txt",
    HERE / TASK / "seed_prompt.txt",
]
artifact_hashes = {}
for path in artifact_paths:
    if path.exists():
        raw = path.read_text()
        docs.append((f"artifact:{path.name}", norm(raw)))
        artifact_hashes[str(path)] = hashlib.sha256(raw.encode()).hexdigest()

test_hits, train_hits, examples = 0, 0, []
for label, text in docs:
    for k in test_keys:
        if k in text:
            test_hits += 1
            examples.append({"doc": label, "first_match": k[:120]})
    train_hits += sum(1 for k in train_keys if k in text)

report = {
    "task": TASK,
    "protegi_run_dir": str(RUN),
    "n_test_items": len(test_keys),
    "n_train_items": len(train_keys),
    "n_llm_calls_audited": len(llm_calls),
    "n_docs_audited": len(docs),
    "test_item_matches (must be 0)": test_hits,
    "train_test_key_overlap (must be 0)": len(overlap),
    "detector_sanity_train_item_matches (must be > 0)": train_hits,
    "examples": examples[:5],
    "artifact_sha256": artifact_hashes,
}
out = HERE / TASK / "leak_audit.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in report.items() if k != "artifact_sha256"}, indent=2, ensure_ascii=False))
