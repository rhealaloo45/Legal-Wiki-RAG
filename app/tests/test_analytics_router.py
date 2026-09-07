"""
Router precision tests for the Phase 4 analytics branches.

These branches answer from SQL with NO retrieval, so a false positive does not
merely answer oddly — it answers a question about one document with a statistic
over the whole corpus. The negative cases below matter more than the positive
ones: every one of them is a question the existing retrieval pipeline already
answers well, and stealing it would be a regression.

Run: python -m tests.test_analytics_router   (from the app/ directory)
Zero database, zero network, zero model calls.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.intent_agent import _is_analytics_query as detect  # noqa: E402


AGGREGATE = [
    "What is the average liability cap across our contracts?",
    "What's the total contract value across all vendor agreements?",
    "Show me the median liability cap",
    "What is the highest liability cap we have agreed?",
    "combined contract value for Tata Power",
    "typical liability cap across the portfolio",
]

GAP = [
    "Which contracts do not have a liability cap?",
    "Which agreements are missing a governing law clause?",
    "List the contracts without a termination provision",
    "Show me documents that lack a liability cap",
    "How many contracts don't have a cap?",
]

TREND = [
    "Has our average liability cap changed over time?",
    "Show the liability cap trend by year",
    "Has the contract value been getting bigger over the years?",
    "liability cap year-on-year",
]

# Questions the retrieval pipeline already answers correctly. Each of these
# contains at least one word the detectors look for, which is exactly why they
# are here — a looser detector eats them.
MUST_NOT_MATCH = [
    # single-document lookups that name a metric word
    "What is the liability cap in the Master Services Agreement dated 12 August 2020?",
    "What is the liability cap in this agreement?",
    "What is the cap under the MSA?",
    "What is the total contract value of that agreement?",
    "What does clause 12 say about the liability cap?",
    # absence questions about ONE document — a document-level question, not a
    # corpus gap query
    "Does the Master Services Agreement dated 12 August 2020 contain a non-compete?",
    # ordinary factual questions
    "What is the governing law of our Master Services Agreements?",
    "How many contracts do we have with Tata Power?",
    "What are the exceptions to the liability cap?",
    "Summarise the termination provisions",
    "Who are the parties to the agreement?",
    # trend-ish words with no metric
    "Has our approach to arbitration changed over time?",
    # named-party questions carrying an op+metric word pair by coincidence —
    # these ask about that party's own document(s), not a corpus statistic.
    # Confirmed live (Sept 2026 grading pass, "Q3"): this exact question
    # returned a one-document corpus statistic and silently dropped its own
    # carve-out half.
    "What is the aggregate liability cap for Apex Suvarna Telecommunications "
    "Private Limited and Nidra Bhandari, and what types of liability are "
    "excluded from that cap?",
    "What is the highest liability cap Apex Meridian Systems Private Limited "
    "has agreed to?",
    # asks for carve-out/exclusion TEXT, not a number — the aggregate branch
    # has no clause text to answer this from even when no party is named.
    "What is the average liability cap across our contracts, and what carve-outs apply?",
]


def run() -> int:
    failures = []

    for q in AGGREGATE:
        got = detect(q)
        if got != "aggregate":
            failures.append(f"{q!r} -> {got!r}, expected 'aggregate'")
    for q in GAP:
        got = detect(q)
        if got != "gap":
            failures.append(f"{q!r} -> {got!r}, expected 'gap'")
    for q in TREND:
        got = detect(q)
        if got != "trend":
            failures.append(f"{q!r} -> {got!r}, expected 'trend'")
    for q in MUST_NOT_MATCH:
        got = detect(q)
        if got:
            failures.append(f"STOLEN: {q!r} -> {got!r}, expected no match")

    total = len(AGGREGATE) + len(GAP) + len(TREND) + len(MUST_NOT_MATCH)
    if failures:
        print(f"FAIL — {len(failures)} of {total} checks failed:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok — {total} analytics-router checks correct "
          f"({len(MUST_NOT_MATCH)} of them negative)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
