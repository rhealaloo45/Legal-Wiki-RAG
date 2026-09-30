"""Checks that a quote flagged against the retrieved passage is looked for in
the documents themselves before it is called unverified.
Fictional text only. Run: python tests/test_source_citation_check.py (from app/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import wiki  # noqa: E402

failures = []

IN_INDEX = ("The Supplier shall maintain the Records for a period of not less than "
            "seven years following termination.")
IN_FILE_ONLY = ("NOW, THEREFORE, in consideration of the mutual covenants set out "
                "below, the Parties agree as follows.")
FABRICATED = ("The Supplier shall reimburse the Customer for all luncheon expenses "
              "incurred by its directors.")

FILE_TEXT = " ".join([IN_INDEX, IN_FILE_ONLY, "and other recitals of no interest."])
reads = []


def _run(quotes, titles=("Records – Supplier",), docs=("a.pdf",), exists=True):
    del reads[:]
    real = (wiki._active_wiki_id, wiki._db.source_docs_for_titles,
            wiki._db.stored_text_for_docs, wiki.os.path.exists, wiki._read_file)
    wiki._active_wiki_id = lambda: "w"
    wiki._db.source_docs_for_titles = lambda w, s, t: list(docs)
    wiki._db.stored_text_for_docs = lambda w, s, d: IN_INDEX
    wiki.os.path.exists = lambda p: exists
    wiki._read_file = lambda p: (reads.append(p), FILE_TEXT)[1]
    try:
        return wiki._quotes_present_in_sources(list(quotes), "s", list(titles))
    finally:
        (wiki._active_wiki_id, wiki._db.source_docs_for_titles,
         wiki._db.stored_text_for_docs, wiki.os.path.exists, wiki._read_file) = real


# A quote elsewhere in the index is found without the file being opened.
idx, fil = _run([IN_INDEX])
if idx != {IN_INDEX} or fil:
    failures.append(("index quote found in the index", idx, fil))
if reads:
    failures.append(("the file is not read when the index settles it", reads))

# A quote only the original file has is found there, and reported separately.
idx, fil = _run([IN_FILE_ONLY])
if fil != {IN_FILE_ONLY} or idx:
    failures.append(("file-only quote found in the file", idx, fil))
if len(reads) != 1:
    failures.append(("the file is read once", reads))

# A fabricated quote is found nowhere and stays flagged.
idx, fil = _run([FABRICATED])
if idx or fil:
    failures.append(("fabricated quote must not be cleared", idx, fil))

# A mixed set is separated correctly.
idx, fil = _run([IN_INDEX, IN_FILE_ONLY, FABRICATED])
if idx != {IN_INDEX} or fil != {IN_FILE_ONLY}:
    failures.append(("mixed set separated", idx, fil))

# Whitespace and case differences do not matter; the comparison is normalised.
idx, fil = _run(["the supplier SHALL   maintain the Records for a period of not "
                 "less than seven years following termination."])
if not idx:
    failures.append(("normalised comparison", idx, fil))

# Missing file on disk: the index result stands, nothing raises.
idx, fil = _run([IN_FILE_ONLY], exists=False)
if idx or fil:
    failures.append(("missing file leaves the quote flagged", idx, fil))

# No pages, no documents, no quotes: all answer emptily rather than raising.
for args in (([IN_INDEX], ()), ([], ("t",))):
    if _run(args[0], titles=args[1]) != (set(), set()):
        failures.append(("empty input", args))
if _run([IN_INDEX], docs=()) != (set(), set()):
    failures.append(("no source documents", None))

# A database that raises is not allowed to break the answer.
_real = wiki._db.source_docs_for_titles


def _boom(*a, **k):
    raise RuntimeError("database down")


wiki._db.source_docs_for_titles = _boom
try:
    if wiki._quotes_present_in_sources([IN_INDEX], "s", ["t"]) != (set(), set()):
        failures.append(("a failing lookup returns nothing found", None))
finally:
    wiki._db.source_docs_for_titles = _real

# Only a bounded number of quotes is checked, so a long list cannot run away.
many = ["quote number %d that appears nowhere at all" % i for i in range(20)]
idx, fil = _run(many + [IN_INDEX])
if idx or fil:
    failures.append(("only the first few quotes are checked", idx, fil))

if failures:
    for f in failures:
        print("FAILED:", f)
    raise SystemExit(1)
print("ok - 12 source-citation checks correct (5 of them negative)")
