"""
Router precision tests for the playbook-compliance branch (Ask mode).

This branch replaces ordinary retrieval with a playbook report, so a false
positive does not merely answer oddly — it answers a question about the
DOCUMENT with a verdict about a house RULE the question never mentioned.
The negative cases below therefore matter more than the positive ones: every
one of them is a question ordinary retrieval already answers well, and each
contains at least one word the detector looks for, which is why it is here.

The detector is conjunctive by design (a compliance verb AND a reference to
the house standard) precisely so "does the supplier comply with Applicable
Law" cannot reach the playbook.

Run: python tests/test_compliance_router.py   (from the app/ directory)
Zero database, zero network, zero model calls.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.intent_agent import _is_compliance_query as detect  # noqa: E402


COMPLIANCE = [
    "is this NDA in satisfaction of our company rules",
    "Does this agreement comply with our playbook?",
    "Check this contract against our house position",
    "Does the Zephyra SPA meet our standard terms?",
    "Is this consistent with our internal guidelines?",
    "Where does this deviate from the house standard?",
    "Review this NDA against our approved templates",
    "Does this align with company policy?",
    "Is the liability cap acceptable under our playbook?",
    "Does this conform to our firm standards?",
    "Is this in line with our fallback positions?",
    "Assess this agreement against our internal requirements",
]

# Questions ordinary retrieval already answers correctly. Each carries a word
# one half of the detector looks for — the other half is what keeps them out.
MUST_NOT_MATCH = [
    # compliance verb, but the standard is a law or a counterparty duty
    "Does the supplier comply with Applicable Law?",
    "Is the vendor compliant with data protection law?",
    "What are the compliance obligations in this contract?",
    "Which milestones did Nimbus meet on time?",
    "Does this satisfy the conditions precedent?",
    "Does clause 12 meet the definition of Confidential Information?",
    "Summarise the deviation from the milestone schedule",
    # compliance verb + another DOCUMENT, not the house position
    "Does this agreement comply with the MSA?",
    "Compare the NDA against the Share Purchase Agreement",
    # "standard" as an ordinary contract word, no compliance verb
    "What is our standard payment term in this deal?",
    "What is the standard of care for Confidential Information?",
    # plain factual
    "What is the liability cap?",
    "Which contracts do not have a liability cap?",
    "Who are the parties to the agreement?",
]


def run() -> int:
    failures = []
    for q in COMPLIANCE:
        if not detect(q):
            failures.append(f"MISSED: {q!r}")
    for q in MUST_NOT_MATCH:
        if detect(q):
            failures.append(f"STOLEN: {q!r}")

    total = len(COMPLIANCE) + len(MUST_NOT_MATCH)
    if failures:
        print(f"FAIL — {len(failures)} of {total} checks failed:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok — {total} compliance-router checks correct "
          f"({len(MUST_NOT_MATCH)} of them negative)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
