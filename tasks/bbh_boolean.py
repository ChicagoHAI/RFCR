"""tasks/bbh_boolean.py — backward-compatibility shim.

The BBH_BOOLEAN_TASK implementation has moved to tasks.boolean_expressions.
This module re-exports it so that ``from tasks.bbh_boolean import BBH_BOOLEAN_TASK``
continues to work without modification.
"""

from tasks.boolean_expressions import BBH_BOOLEAN_TASK

__all__ = ["BBH_BOOLEAN_TASK"]
