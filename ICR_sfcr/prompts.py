"""ICR_sfcr/prompts.py — Prompt templates for SFCR rule generation."""

# ---------------------------------------------------------------------------
# Rule generation — flat pool (original)
# ---------------------------------------------------------------------------

RULE_GENERATION_PROMPT = """\
You are improving a compact task cheatsheet used to help a language model solve reasoning problems.

The current cheatsheet is:
<cheatsheet>
{anchor_cheatsheet}
</cheatsheet>

Below are two example sets.

=== SET A — SHARED FAILURES ===
These examples cause mistakes under the current cheatsheet across multiple models.
Write a rule that helps correct them.

{v_shared_block}

=== SET B — BOUNDARY CASES (source-private failures) ===
The source model fails these, but all other evaluated models get them right.
Your rule MUST NOT activate on these examples — they define the boundary of the rule.

{v_private_block}

Instructions:
- Generate exactly one short rule that helps the SHARED FAILURE examples.
- The rule must not trigger on BOUNDARY CASE examples.
- Do not copy, quote, or paraphrase any gold reasoning traces.
- The rule must be task-level, general, and checkable in ≤ 800 characters.
- The rule must complement the existing cheatsheet — do not duplicate content already there.

Output format (use these exact labels, one per line, no extra headers):
RULE:
USE WHEN:
DO NOT USE WHEN:
CHECK:
"""

# ---------------------------------------------------------------------------
# Rule generation — subtype-targeted
# ---------------------------------------------------------------------------

RULE_GENERATION_PROMPT_SUBTYPE = """\
You are improving a compact task cheatsheet used to help a language model solve reasoning problems.

The current cheatsheet is:
<cheatsheet>
{anchor_cheatsheet}
</cheatsheet>

=== SET A — TARGET FAILURE SUBTYPE ===
These examples share a specific failure pattern you must address.
Subtype description: {subtype_description}

{v_subtype_block}

=== SET B — OTHER SHARED FAILURES (context only) ===
These also fail, but for different reasons. Do not write a rule for these.

{v_other_block}

=== SET C — BOUNDARY CASES ===
These examples look structurally similar to SET A but must NOT trigger your rule.
The source model fails them, but other models do not — they mark the edge of where the rule applies.

{v_private_block}

Instructions:
- Write a rule that fixes the TARGET FAILURE SUBTYPE (SET A) examples.
- Your USE WHEN must be narrow enough to exclude the BOUNDARY CASES (SET C).
- Your DO NOT USE WHEN must explicitly reference the distinguishing features of SET C.
- Do not address SET B — other rules will handle those.
- Do not copy, quote, or paraphrase any gold reasoning traces.
- The rule must be task-level, general, and checkable in ≤ 800 characters.

Output format (use these exact labels, one per line, no extra headers):
RULE:
USE WHEN:
DO NOT USE WHEN:
CHECK:
"""

# ---------------------------------------------------------------------------
# Subtype clustering
# ---------------------------------------------------------------------------

SUBTYPE_CLUSTER_PROMPT = """\
You are analyzing a set of reasoning questions that a language model fails to answer correctly.
Your job is to group them into 2-4 distinct failure subtypes based on the shared reasoning pattern or structural feature that causes the error.

Here are the failing examples:

{v_shared_block}

Instructions:
- Identify 2-4 distinct subtypes. More subtypes is not better — only split if the failure cause is genuinely different.
- Each example must be assigned to exactly one subtype.
- Give each subtype a short label (3-6 words) and a one-sentence description of the failure pattern.
- Output valid JSON only, with this exact structure:

{{
  "subtypes": [
    {{
      "label": "short label",
      "description": "one sentence describing the failure pattern",
      "indices": [0, 3, 7]
    }}
  ]
}}

The indices are 0-based positions in the list above (0 = first example shown).
Do not include any text outside the JSON.
"""

# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------

RULE_REPAIR_PROMPT = """\
You previously generated this rule for a reasoning task cheatsheet:

RULE: {rule}
USE WHEN: {use_when}
DO NOT USE WHEN: {do_not_use_when}
CHECK: {check}

It was rejected for this reason: {reject_reason}

{mis_triggered_section}

{no_gain_section}

Your task:
- Keep the core RULE insight — it is directionally correct.
- Make USE WHEN more precise so it does not fire on the problematic examples above.
- Make DO NOT USE WHEN explicitly exclude the distinguishing features of those examples.
- Do not change CHECK unless it is actively harmful.
- Keep the total rule under 800 characters.

Output the revised rule using these exact labels (one per line, no extra headers):
RULE:
USE WHEN:
DO NOT USE WHEN:
CHECK:
"""

# ---------------------------------------------------------------------------
# Compressed rationale (used in oracle_mode="compressed")
# ---------------------------------------------------------------------------

COMPRESSED_RATIONALE_PROMPT = """\
Summarize the key insight from the following reasoning in a single sentence.
The sentence should capture the general principle, not quote specific details.

Reasoning:
{reasoning}

One-sentence insight:"""
