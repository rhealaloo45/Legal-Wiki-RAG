"""Checks that a party-name scan which comes back sitting on its cap is
re-scanned in full before its size is compared or intersected.
Fictional names only. Run: python tests/test_party_scan_cap.py (from app/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import wiki  # noqa: E402

failures = []
calls = []


class _StubDB:
    """Stands in for services.db. ``corpus`` maps a phrase to the documents
    that mention it; the stub truncates to ``cap`` exactly as the real query's
    LIMIT does, so a capped first pass is reproduced rather than described."""

    def __init__(self, corpus):
        self.corpus = corpus

    def find_source_docs_mentioning_phrase(self, wiki_id, session_id, phrase, cap=25):
        docs = sorted(self.corpus.get(phrase, []))
        calls.append((phrase, cap))
        return docs[:cap]


def _run(question, corpus, max_docs=4):
    del calls[:]
    real_db, real_wiki_id, real_canon = wiki._db, wiki._active_wiki_id, wiki._with_canonical_party_names
    wiki._db = _StubDB(corpus)
    wiki._active_wiki_id = lambda: "w"
    wiki._with_canonical_party_names = lambda names: list(names)
    try:
        return wiki._resolve_docs_by_party(question, "s", max_docs=max_docs)
    finally:
        wiki._db, wiki._active_wiki_id, wiki._with_canonical_party_names = (
            real_db, real_wiki_id, real_canon)


BROAD = ["broad_%02d.pdf" % i for i in range(60)]

# 0. The document the question names sits outside the first pass's cap. Only a
#    full scan lets the instrument word in the question find its filename; a
#    capped set leaves the whole question unanswerable.
guarantee = "zz_brackenfold_guarantee_2022.pdf"
scope = _run(
    "What does the guarantee with Brackenfold Logistics Limited say about demand?",
    {"Brackenfold Logistics": ["brackenfold_supply_%02d.pdf" % i for i in range(60)] + [guarantee]},
)
if scope != {guarantee}:
    failures.append(("document beyond the cap must still be findable", scope))

# 1. A name in far more documents than the cap must not beat a genuinely
#    narrow name just because both sets were cut to the same length.
narrow = ["Ashgrove Foods Limited supply 2021.pdf"]
scope = _run(
    "What does the supply agreement between Brackenfold Logistics Limited and "
    "Ashgrove Foods Limited say about termination?",
    {"Brackenfold Logistics": BROAD + narrow,
     "Ashgrove Foods": narrow},
)
if scope != set(narrow):
    failures.append(("broad name must not win on a truncated count", scope))

# 2. The re-scan runs for the capped name and does not run for the narrow one.
wide_calls = [c for c in calls if c[1] > 20]
if [c[0] for c in wide_calls] != ["Brackenfold Logistics"]:
    failures.append(("only the capped name is re-scanned", wide_calls))

# 3. Two names intersect to the one document they share even when the shared
#    document sits beyond the first name's cap.
shared = ["zz_shared_deed.pdf"]
scope = _run(
    "In the deed between Brackenfold Logistics Limited and Harrowgate Rail Limited, "
    "what is the notice period?",
    {"Brackenfold Logistics": BROAD + shared,
     "Harrowgate Rail": ["other_harrowgate.pdf"] + shared},
)
if scope != set(shared):
    failures.append(("intersection must survive a capped first pass", scope))

# 4. A name inside the cap is left alone: no wide query at all.
scope = _run(
    "What does the licence between Ashgrove Foods Limited and Harrowgate Rail Limited cover?",
    {"Ashgrove Foods": ["a.pdf", "b.pdf"],
     "Harrowgate Rail": ["b.pdf", "c.pdf"]},
)
if scope != {"b.pdf"}:
    failures.append(("uncapped intersection unchanged", scope))
if any(c[1] > 20 for c in calls):
    failures.append(("no wide query when nothing hit the cap", calls))

# 5. A name that really does match exactly the cap's worth of documents keeps
#    that set; the re-scan confirms it rather than changing it.
exact = ["exact_%02d.pdf" % i for i in range(20)]
scope = _run(
    "What do the Brackenfold Logistics Limited agreements say about audit rights?",
    {"Brackenfold Logistics": exact},
    max_docs=40,
)
if scope != set(exact):
    failures.append(("exactly-at-cap set preserved", len(scope)))

if failures:
    for f in failures:
        print("FAILED:", f)
    raise SystemExit(1)
print("ok - 7 party-scan-cap checks correct")
