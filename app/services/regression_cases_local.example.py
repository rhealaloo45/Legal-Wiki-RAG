"""Format sample for services/regression_cases_local.py (fictional content).

Copy to regression_cases_local.py and replace with cases verified by hand
against your own corpus. That file is gitignored; this one is not, so nothing
here may name a real document, party or figure.
"""

SEED_CASES: list[dict] = [
    {
        # Scope case: which documents the question must resolve to.
        "name": "scope-example-party-pair",
        "archetype": "abstention",
        "question": "What is the liability cap in the agreement between Acme Widgets Ltd and Globex Corp?",
        "expect_docs": ["Acme Widgets - Globex MSA"],
        "expect_abstain": True,
        "expect_answer": "No numeric liability cap is stated in the agreement.",
        "notes": "Why this case exists and how the expected answer was verified.",
    },
    {
        # Answer case: a verified expected answer for the graded tier.
        "name": "answer-example-notice-period",
        "archetype": "point-lookup",
        "question": "What notice period does the Acme Widgets NDA require for termination?",
        "expect_docs": ["Acme Widgets NDA"],
        "expect_answer": "Thirty days' written notice.",
        "notes": "",
    },
]
