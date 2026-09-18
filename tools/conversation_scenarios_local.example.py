"""Format sample for tools/conversation_scenarios_local.py (fictional content).
Copy to conversation_scenarios_local.py and write scenarios against your own
corpus; that file is gitignored."""

SCENARIOS = [
    {
        "name": "example-follow-up-stays-on-document",
        "why": "A follow-up that names no document must stay on the one the thread pinned.",
        "turns": [
            {"q": "What is the term of the NDA between Acme Widgets Ltd and Globex Corp?",
             "free": False, "must": ["two years"]},
            {"q": "Which agreements expire in the next 90 days?",
             "path": "analytics", "free": True},
        ],
    },
]
