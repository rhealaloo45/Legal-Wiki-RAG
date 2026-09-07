"""
Parser tests for value normalisation (§ Phase 3.5c).

Inputs are real values taken from contracts.liability_cap,
obligations.deadline and clauses.typed_value on the live corpus.

The most important assertions here are the NEGATIVE ones: an unreadable value
must come back as `unparsed` and a cross-reference as `reference`, never as a
None that a gap-detection query would read as "this contract has no cap".

Run: python -m tests.test_normalize   (from the app/ directory)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import normalize as N  # noqa: E402


MONEY_CASES = [
    # (input, expected status, expected amount, expected currency)
    ({"amount": 25529937.0, "currency": "INR", "raw": "Rs. 25,529,937"}, N.OK, 25529937.0, "INR"),
    ("Rs. 110,661,324", N.OK, 110661324.0, "INR"),
    ("Rs. 9,284,268,071", N.OK, 9284268071.0, "INR"),
    ("INR 2,169,990,936", N.OK, 2169990936.0, "INR"),
    ("$1,500,000", N.OK, 1500000.0, "USD"),
    ("2.5 crore", N.OK, 25000000.0, None),
    ("50 lakhs", N.OK, 5000000.0, None),
    ("USD 3 million", N.OK, 3000000.0, "USD"),
    ("Except in respect of liability arising from fraud, gross negligence, "
     "wilful misconduct, or breach of confidentiality obligations, each Party's "
     "aggregate liability shall not exceed Rs. 201,212,221.", N.OK, 201212221.0, "INR"),
    # cross-reference: a real value that lives elsewhere, NOT a parse failure
    ("the cap agreed in Schedule IV", N.REFERENCE, None, None),
    ("As set out in Annexure B", N.REFERENCE, None, None),
    # genuinely unreadable — must NOT come back as absent
    ("As agreed between the parties from time to time", N.UNPARSED, None, None),
    ("subject to statutory restrictions", N.UNPARSED, None, None),
    # genuinely empty
    (None, N.ABSENT, None, None),
    ("", N.ABSENT, None, None),

    # --- caps stated as a RULE, not a figure --------------------------------
    # Every one of these parsed as a rupee amount before FORMULA existed:
    # "5% of the Contract Price" became a cap of five rupees on real contracts.
    # A SUM must skip these; a Calculation Agent can resolve them given a base.
    ("maximum aggregate cap of 5% of the Contract Price", N.FORMULA, None, None),
    ("maximum aggregate cap of 10% of the Contract Price (for liquidated damages)",
     N.FORMULA, None, None),
    ("capped at 150% of fees paid in the preceding 12 months", N.FORMULA, None, None),
    ("limited to the total fees paid or payable under the SOW in the twelve (12) "
     "months preceding the event giving rise to liability", N.FORMULA, None, None),
    ("twice the annual charges", N.FORMULA, None, None),

    # --- clause pointers must never be read as amounts ----------------------
    ("Subject to the limitations in Clause 10 of the MSA", N.UNPARSED, None, None),
    ("Liability under this DPA is subject to the limitations in Clause 10 of the MSA",
     N.UNPARSED, None, None),
    # a real figure alongside a clause reference must still be found
    ("Under Clause 12, aggregate liability shall not exceed Rs. 5,00,000",
     N.OK, 500000.0, "INR"),
]

DURATION_CASES = [
    # (input, expected status, expected days, expected business_days)
    ("not less than 90 days' prior written notice", N.OK, 90, False),
    ("30 days", N.OK, 30, False),
    ("thirty (30) days", N.OK, 30, False),
    ("within 72 hours", N.OK, 3.0, False),
    ("for a period of not less than 5 years following expiry", N.OK, 1825, False),
    ("two (2) years", N.OK, 730, False),
    ("15 business days", N.OK, 15, True),
    ("10 working days", N.OK, 10, True),
    ("6 months", N.OK, 180, False),
    # prose with no number at all — unparsed, never absent
    ("prior to commencing work", N.UNPARSED, None, False),
    ("for the duration of the Term", N.UNPARSED, None, False),
    (None, N.ABSENT, None, False),
]

DATE_CASES = [
    ("2020-08-12", N.OK, "2020-08-12"),
    ("12 August 2020", N.OK, "2020-08-12"),
    ("17 January 2019", N.OK, "2019-01-17"),
    ("August 12, 2020", N.OK, "2020-08-12"),
    ("Sept 3 2021", N.OK, "2021-09-03"),
    ("on the Effective Date", N.UNPARSED, None),
    (None, N.ABSENT, None),
]


def run() -> int:
    failures = []

    for raw, status, amount, currency in MONEY_CASES:
        got = N.parse_money(raw)
        label = repr(raw)[:60]
        if got["status"] != status:
            failures.append(f"parse_money({label}) status={got['status']!r}, expected {status!r}")
        elif status == N.OK:
            if got["amount"] != amount:
                failures.append(f"parse_money({label}) amount={got['amount']}, expected {amount}")
            if currency is not None and got["currency"] != currency:
                failures.append(f"parse_money({label}) currency={got['currency']!r}, expected {currency!r}")

    for raw, status, days, business in DURATION_CASES:
        got = N.parse_duration(raw)
        label = repr(raw)[:60]
        if got["status"] != status:
            failures.append(f"parse_duration({label}) status={got['status']!r}, expected {status!r}")
        elif status == N.OK:
            if got["days"] != days:
                failures.append(f"parse_duration({label}) days={got['days']}, expected {days}")
            if got["business_days"] != business:
                failures.append(f"parse_duration({label}) business_days={got['business_days']}, expected {business}")

    for raw, status, iso in DATE_CASES:
        got = N.parse_date(raw)
        label = repr(raw)[:60]
        if got["status"] != status:
            failures.append(f"parse_date({label}) status={got['status']!r}, expected {status!r}")
        elif status == N.OK and got["date"] != iso:
            failures.append(f"parse_date({label}) date={got['date']!r}, expected {iso!r}")

    # The rule the whole module exists for: an unreadable value must never be
    # indistinguishable from an absent one.
    for probe in ("As agreed between the parties", "the cap agreed in Schedule IV"):
        if N.parse_money(probe)["status"] == N.ABSENT:
            failures.append(f"parse_money({probe!r}) reported ABSENT — a gap query "
                            f"would read this as 'no cap exists'")

    total = len(MONEY_CASES) + len(DURATION_CASES) + len(DATE_CASES) + 2
    if failures:
        print(f"FAIL — {len(failures)} of {total} checks failed:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok — {total} normalisation checks correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
