"""Checks for the deterministic guards that keep the ambiguity check from
interrupting a question that has already said what to do or what to cover.
Fictional text only. Run: python tests/test_clarification_guard.py (from app/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import intent_agent as ia  # noqa: E402

failures = []

# Should skip the check: opens with a deliverable and names no bare document.
SKIP_TASK = [
    "Draft a breach-notification clause for a new supplier agreement, using our own precedent.",
    "Draft a short note to the board on our position across the supply agreements.",
    "Write a one-paragraph summary of the dispute outcomes.",
    "Put our service level agreements in date order, oldest first.",
    "List every licence that renews this year.",
    "Please draft a mutual NDA for a new vendor.",
    "Rank the loan agreements by principal.",
]
for q in SKIP_TASK:
    if not ia._states_its_task(q):
        failures.append(("task should skip", q))

# Should NOT skip: a deliverable aimed at an unnamed single document, or no deliverable.
KEEP_ASKING = [
    "Draft a summary of this agreement.",
    "Write a note on that contract.",
    "Summarize this agreement.",
    "What is the notice period?",
    "Which of these is better?",
    "Review the indemnity.",
]
for q in KEEP_ASKING:
    if ia._states_its_task(q):
        failures.append(("task should NOT skip", q))

# Scope stated as a whole population.
WIDE = [
    "Across the arbitration and court matters in this set, what pattern emerges in how costs are dealt with?",
    "Across our data processing agreements, which carry an audit right?",
    "What themes run across all of our supply contracts?",
    "How many licences are in the corpus?",
]
for q in WIDE:
    if not ia._explicitly_corpus_wide(q):
        failures.append(("wide should skip", q))

NOT_WIDE = [
    "What is the notice period?",
    "Across the term, when does the fee escalate?",
    "Summarize this agreement.",
]
for q in NOT_WIDE:
    if ia._explicitly_corpus_wide(q):
        failures.append(("wide should NOT skip", q))

if failures:
    for f in failures:
        print("FAILED:", f)
    raise SystemExit(1)
print("ok - %d clarification-guard checks correct (%d of them negative)" % (
    len(SKIP_TASK) + len(KEEP_ASKING) + len(WIDE) + len(NOT_WIDE), len(KEEP_ASKING) + len(NOT_WIDE)))
