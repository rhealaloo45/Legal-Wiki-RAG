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

# One commitment, fingerprinted the same whether or not the extractor cut an
# unrelated sentence in front of it.
alone = ("Confidentiality obligations shall survive for so long as the information "
         "remains confidential.")
bundled = "This Agreement shall remain effective for three years. " + alone
check("bundled passage still carries the wording",
      set(T.fingerprints(alone)) <= set(T.fingerprints(bundled)))
check("the extra sentence is its own wording", len(T.fingerprints(bundled)) == 2)
check("whole-passage fingerprint still differs", T.fingerprint(alone) != T.fingerprint(bundled))
check("one sentence is one fingerprint", T.fingerprints(alone) == [T.fingerprint(alone)])
check("nothing operative, nothing to fingerprint",
      T.fingerprints("Customer Data means all data supplied by the Customer.") == [])

# A full stop inside an abbreviation or a clause number is not a sentence end.
check("abbreviations survive",
      len(T.wordings("Acme Pte. Ltd. shall pay the fee under Section 3.2 hereof.")) == 1)
check("two sentences are two",
      len(T.wordings("Acme Co. shall indemnify the Buyer. Art. 5 shall apply.")) == 2)

# The same sentence twice in one passage is one fingerprint, not two.
check("repeat within a passage counted once",
      len(T.fingerprints(alone + " " + alone)) == 1)

if failures:
    print("FAILED:", failures)
    raise SystemExit(1)
print("ok - 16 wording-fingerprint checks correct (3 negative)")
