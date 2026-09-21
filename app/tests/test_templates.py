"""Checks for wording fingerprints (services/templates.py). Fictional text only.

Run: python tests/test_templates.py  (from app/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import templates as T  # noqa: E402

failures = []


def check(name, cond):
    if not cond:
        failures.append(name)


# Same wording, different party, figure and place: one fingerprint.
a = "Acme Widgets Ltd shall notify Globex Corp within 48 hours of any incident affecting Customer Data."
b = "Initech Pty Ltd shall notify Hooli Inc within 24 hours of any incident affecting Customer Data."
check("names and numbers masked", T.fingerprint(a) == T.fingerprint(b))

p = "Acme Widgets Ltd shall not store Customer Data outside India without consent and shall keep a record of all locations."
q = "Globex Corp shall not store Customer Data outside Singapore without consent and shall keep a record of all locations."
check("places masked", T.fingerprint(p) == T.fingerprint(q))

# Different commitment: different fingerprint.
c = "Acme Widgets Ltd shall not use Customer Data to train any model and shall use it solely to provide the Services."
check("different wording differs", T.fingerprint(a) != T.fingerprint(c))

# No operative verb, nothing to fingerprint.
check("definition has none", T.fingerprint("Customer Data means all data supplied by the Customer.") is None)
check("empty is none", T.fingerprint("") is None and T.fingerprint(None) is None)

# Page quote lines: only "> " lines of real length.
page = ("Summary of the clause.\n\n**Supporting Quotes:**\n"
        "> Acme Widgets Ltd shall notify Globex Corp within 48 hours of any incident.\n"
        "> too short\n"
        "not a quote line shall notify Globex Corp within 48 hours of any incident affecting it\n")
check("quote lines", T.quote_lines(page) == ["Acme Widgets Ltd shall notify Globex Corp within 48 hours of any incident."])

if failures:
    print("FAILED:", failures)
    raise SystemExit(1)
print("ok - 8 wording-fingerprint checks correct (2 negative)")
