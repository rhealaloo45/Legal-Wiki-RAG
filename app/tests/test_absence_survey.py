"""Checks the anomaly survey's second measure: a wording most of a population
states that one or two of them do not.
Fictional text only. Run: python tests/test_absence_survey.py (from app/)
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import survey, templates  # noqa: E402

failures = []

AI = ("The Processor shall not use Customer Data to train, fine-tune or improve any "
      "general-purpose artificial intelligence or machine learning model.")
CONF = ("Each Party shall keep the other Party's Confidential Information strictly "
        "confidential and shall not disclose such Confidential Information to any "
        "third party except to its employees and professional advisers on a "
        "need-to-know basis.")

# Nine documents. Eight carry both wordings; the ninth carries neither, but its
# text deals with confidentiality in its own words and says nothing at all
# about AI training.
DOCS = ["doc_%d.pdf" % i for i in range(9)]
ODD = DOCS[8]
TEXTS = {d: (AI + " " + CONF).lower() for d in DOCS[:8]}
TEXTS[ODD] = ("each party shall treat information received from the other as "
              "confidential, shall not disclose it to any third party, and shall "
              "restrict access to employees and professional advisers who need it "
              "on a strictly need-to-know basis.")

BY_HASH = {"h_ai": set(DOCS[:8]), "h_conf": set(DOCS[:8])}
UNITS = {d: 20 for d in DOCS}
EXAMPLES = {"h_ai": AI, "h_conf": CONF}


@contextlib.contextmanager
def _conn():
    yield None


def _run(by_hash=BY_HASH, units=UNITS, texts=TEXTS):
    real = (templates.is_populated, templates.docs_by_hash, templates.example_text,
            survey._doc_text, survey.db.get_engine)
    templates.is_populated = lambda w, s: True
    templates.docs_by_hash = lambda w, s, d: (by_hash, units)
    templates.example_text = lambda w, s, h: EXAMPLES.get(h)
    survey._doc_text = lambda c, w, s, d: texts.get(d, "")
    survey.db.get_engine = lambda: type("E", (), {"connect": staticmethod(_conn)})()
    try:
        return survey._absence_findings("w", "s", {d: {"source_doc": d, "parties": [d]}
                                                   for d in texts})
    finally:
        (templates.is_populated, templates.docs_by_hash, templates.example_text,
         survey._doc_text, survey.db.get_engine) = real


findings, thin, reworded = _run()

# The AI wording is reported: the odd document uses none of its vocabulary.
ai = [f for f in findings if "fine-tune" in f[1]]
if len(ai) != 1 or ai[0][0] != 8 or ai[0][2] != [ODD]:
    failures.append(("AI wording must be reported against the one document", findings))

# The confidentiality wording is NOT reported: the same document states it in
# its own words, which the term overlap sees.
if any("Confidential Information" in f[1] for f in findings):
    failures.append(("a rewording must not be reported as missing", findings))
if reworded != 1:
    failures.append(("the rewording is counted", reworded))
if thin:
    failures.append(("no document is thin here", thin))

# A population below the minimum is not surveyed this way at all.
small = {d: TEXTS[d] for d in DOCS[:5]}
f2, _, _ = _run(by_hash={"h_ai": set(list(small)[:4])},
                units={d: 20 for d in small}, texts=small)
if f2:
    failures.append(("population below the minimum is not surveyed", f2))

# A document the extractor barely read is set aside rather than reported.
thin_units = dict(UNITS)
thin_units[ODD] = 1
f3, thin3, _ = _run(units=thin_units)
if thin3 != 1:
    failures.append(("barely-extracted document is set aside", thin3))
if f3:
    failures.append(("nothing is reported once the thin document is set aside", f3))

# A wording only half the population shares is not a shared wording.
f4, _, _ = _run(by_hash={"h_ai": set(DOCS[:4])})
if f4:
    failures.append(("a wording half the population shares is not reported", f4))

# Terms are content words only.
terms = survey._wording_terms(AI)
for expected in ("fine-tune", "artificial", "intelligence", "machine", "learning"):
    if expected not in terms:
        failures.append(("content word kept", expected))
for unwanted in ("shall", "party", "the", "any"):
    if unwanted in terms:
        failures.append(("function word dropped", unwanted))

if failures:
    for f in failures:
        print("FAILED:", f)
    raise SystemExit(1)
print("ok - 12 absence-survey checks correct (4 of them negative)")
