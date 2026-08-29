"""
utils/results_io.py

Locked read-modify-write for shared JSON results files.
Safe for concurrent processes writing to the same file.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path


def locked_save(results_file: Path, updates: dict) -> dict:
    """Merge `updates` into `results_file` under an exclusive file lock.

    Reads the current file contents, merges in `updates`, writes back,
    and returns the merged dict. Creates the file and parent dirs if needed.
    Safe for multiple concurrent processes writing to the same file.
    """
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        content = fh.read()
        current = json.loads(content) if content.strip() else {}
        current.update(updates)
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(current, indent=2))
        fcntl.flock(fh, fcntl.LOCK_UN)
    return current
