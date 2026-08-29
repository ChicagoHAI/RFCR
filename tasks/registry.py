"""tasks/registry.py — Single source of truth for task name → TaskSpec mapping.

Usage:
    from tasks.registry import get_task
    task_spec = get_task("web_of_lies")
"""
from __future__ import annotations

import importlib
from utils.task_spec import TaskSpec

# Maps canonical task name → (module_path, attribute_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "magma":                           ("tasks.magma",                  "MAGMA_TASK"),
    "boolean_expressions":             ("tasks.boolean_expressions",    "BBH_BOOLEAN_TASK"),
    "causal_judgement":                ("tasks.causal_judgement",       "CAUSAL_JUDGEMENT_TASK"),
    "disambiguation_qa":               ("tasks.disambiguation_qa",      "DISAMBIGUATION_TASK"),
    "geometric_shapes":                ("tasks.geometric_shapes",       "GEOMETRIC_TASK"),
    "movie_recommendation":            ("tasks.movie_recommendation",   "MOVIE_TASK"),
    "sports_understanding":            ("tasks.sports_understanding",   "SPORTS_TASK"),
    "date_understanding":              ("tasks.date_understanding",     "DATE_UNDERSTANDING_TASK"),
    "formal_fallacies":                ("tasks.formal_fallacies",       "FORMAL_FALLACIES_TASK"),
    "logical_deduction_three_objects": ("tasks.logical_deduction",      "LOGICAL_DEDUCTION_3_TASK"),
    "navigate":                        ("tasks.navigate",               "NAVIGATE_TASK"),
    "snarks":                          ("tasks.snarks",                 "SNARKS_TASK"),
    "web_of_lies":                     ("tasks.web_of_lies",            "WEB_OF_LIES_TASK"),
    "object_counting":                 ("tasks.object_counting",        "OBJECT_COUNTING_TASK"),
    "gpqa_diamond":                    ("tasks.gpqa_diamond",           "GPQA_DIAMOND_TASK"),
    "agieval_lsat_ar":                 ("tasks.agieval",                "AGIEVAL_LSAT_AR_TASK"),
    "agieval_lsat_lr":                 ("tasks.agieval",                "AGIEVAL_LSAT_LR_TASK"),
    "agieval_logiqa_en":               ("tasks.agieval",                "AGIEVAL_LOGIQA_EN_TASK"),
    "mmlu_formal_logic":               ("tasks.mmlu",                   "MMLU_FORMAL_LOGIC_TASK"),
    "mmlu_professional_law":           ("tasks.mmlu",                   "MMLU_PROFESSIONAL_LAW_TASK"),
    "mmlu_college_mathematics":        ("tasks.mmlu",                   "MMLU_COLLEGE_MATHEMATICS_TASK"),
    "mmlu_moral_scenarios":            ("tasks.mmlu",                   "MMLU_MORAL_SCENARIOS_TASK"),
    "mmlu_high_school_physics":        ("tasks.mmlu",                   "MMLU_HIGH_SCHOOL_PHYSICS_TASK"),
}

# Short aliases used in eval scripts
_ALIASES: dict[str, str] = {
    "logical_deduction_three": "logical_deduction_three_objects",
}

TASK_REGISTRY = {**_REGISTRY}


def get_task(task_name: str) -> TaskSpec:
    """Return the TaskSpec singleton for *task_name*.

    Raises KeyError if the name is not registered.
    """
    canonical = _ALIASES.get(task_name, task_name)
    if canonical not in _REGISTRY:
        raise KeyError(
            f"Unknown task {task_name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, attr = _REGISTRY[canonical]
    return getattr(importlib.import_module(module_path), attr)
