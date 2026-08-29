"""tasks/bbh_tasks.py — backward-compatibility shim.

All task implementations have moved to individual files.  This module
re-exports everything that was previously defined here so that existing
scripts using ``from tasks.bbh_tasks import ...`` continue to work.
"""

# ── Task constants ────────────────────────────────────────────────────────────
from tasks.causal_judgement import CAUSAL_JUDGEMENT_TASK
from tasks.sports_understanding import SPORTS_TASK
from tasks.disambiguation_qa import DISAMBIGUATION_TASK
from tasks.movie_recommendation import MOVIE_TASK
from tasks.geometric_shapes import GEOMETRIC_TASK

# ── Shared helpers (now live in tasks.utils) ─────────────────────────────────
from tasks.utils import (
    _make_eval_prompt,
    _parse_yesno,
    _parse_mc,
    _extract_reasoning,
    _yesno_correct,
    _yesno_label,
    _mc_correct,
    _mc_label,
    _rule_score_prompt,
    _gen_prompt,
    _bootstrap_ruleset,
    _format_failure,
)

# ── Private helpers referenced by dev scripts ────────────────────────────────
from tasks.causal_judgement  import _causal_bootstrap, _CAUSAL_GEN_PROMPT
from tasks.sports_understanding import _SPORTS_GEN_PROMPT
from tasks.geometric_shapes  import _GEO_GEN_PROMPT

__all__ = [
    "CAUSAL_JUDGEMENT_TASK",
    "SPORTS_TASK",
    "DISAMBIGUATION_TASK",
    "MOVIE_TASK",
    "GEOMETRIC_TASK",
    "_make_eval_prompt",
    "_parse_yesno",
    "_parse_mc",
    "_extract_reasoning",
    "_yesno_correct",
    "_yesno_label",
    "_mc_correct",
    "_mc_label",
    "_rule_score_prompt",
    "_gen_prompt",
    "_bootstrap_ruleset",
    "_format_failure",
    "_causal_bootstrap",
    "_CAUSAL_GEN_PROMPT",
    "_SPORTS_GEN_PROMPT",
    "_GEO_GEN_PROMPT",
]
