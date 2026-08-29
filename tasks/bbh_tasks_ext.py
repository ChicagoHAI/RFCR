"""tasks/bbh_tasks_ext.py — backward-compatibility shim.

All task implementations have moved to individual files.  This module
re-exports everything that was previously defined here so that existing
scripts using ``from tasks.bbh_tasks_ext import ...`` continue to work.
"""

# ── Task constants ────────────────────────────────────────────────────────────
from tasks.formal_fallacies import FORMAL_FALLACIES_TASK
from tasks.logical_deduction import LOGICAL_DEDUCTION_3_TASK
from tasks.web_of_lies import WEB_OF_LIES_TASK
from tasks.date_understanding import DATE_UNDERSTANDING_TASK
from tasks.navigate import NAVIGATE_TASK
from tasks.snarks import SNARKS_TASK

__all__ = [
    "FORMAL_FALLACIES_TASK",
    "LOGICAL_DEDUCTION_3_TASK",
    "WEB_OF_LIES_TASK",
    "DATE_UNDERSTANDING_TASK",
    "NAVIGATE_TASK",
    "SNARKS_TASK",
]
