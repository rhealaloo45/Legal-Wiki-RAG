"""Read-only diagnostic: what does the INDEX actually know that identifies Q85's document?

Q85 names its document three ways — a party ("Tata Sons Private Limited"), an
instrument ("services agreement"), and a registered-office block ("Bombay House,
24 Homi Mody Street, Mumbai"). Ground truth says the office block is stated
identically in Service Agreement 2 and Service Agreement 4, which is what makes
the question ambiguous. Phrase search finds it in ONE document, so either the
wiki pages compiled at ingest did not keep it for SA 4, or SA 4 renders it
differently.

This prints what each signal resolves to, so the ambiguity detector can be built
on one that actually discriminates in the index rather than one that only
discriminates in the source PDFs.

Runs SELECTs only. Usage: cd app && python eval/diagnose_q85.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

import config  # noqa: E402
from services import db  # noqa: E402
from services import wiki  # noqa: E402

SID = os.getenv("HARNESS_WIKI_SESSION", "3a66b0ab-a9cc-48f0-a3f3-b0ab863936fe")


def short(d):
    return wiki._norm_doc_name(d)


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


rule("1. Phrase search — which documents state the office block?")
for phrase in ["Bombay House, 24 Homi Mody Street, Mumbai",
               "Bombay House, 24 Homi Mody Street",
               "24 Homi Mody Street",
               "Homi Mody Street",
               "Bombay House"]:
    try:
        hits = db.find_source_docs_mentioning_phrase(SID, phrase, cap=40)
    except Exception as e:
        print(f"  {phrase!r}: ERROR {e}")
        continue
    print(f"\n  {phrase!r} → {len(hits)} document(s)")
    for h in sorted(short(x) for x in hits)[:15]:
        print(f"      {h}")


rule("2. Does ANY page of Service Agreement 2 / 4 contain the address text?")
from sqlalchemy import text  # noqa: E402

engine = db.get_engine()
with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT source_doc, title,
               position('Bombay' in content)  AS bombay_at,
               position('Homi'   in content)  AS homi_at,
               length(content)                AS len
        FROM pages
        WHERE session_id = :sid
          AND title NOT LIKE 'Q:%'
          AND (source_doc ILIKE '%Service Agreement 2%'
            OR source_doc ILIKE '%Service Agreement 4%'
            OR source_doc ILIKE '%SA_02%' OR source_doc ILIKE '%SA_04%')
        ORDER BY source_doc, title
    """), {"sid": SID})
    cur = None
    for r in rows:
        if r.source_doc != cur:
            cur = r.source_doc
            print(f"\n  {short(cur)}")
        mark = []
        if r.bombay_at:
            mark.append(f"Bombay@{r.bombay_at}")
        if r.homi_at:
            mark.append(f"Homi@{r.homi_at}")
        print(f"      {'HIT  ' if mark else '     '} {r.title[:64]:<66} "
              f"{r.len:>7} chars  {' '.join(mark)}")


rule("3. Party + instrument — how many service agreements name Tata Sons?")
try:
    fam_docs = set(db.get_documents_by_family(SID, "Service Agreement"))
    print(f"  Service Agreement family: {len(fam_docs)} document(s)")
except Exception as e:
    fam_docs = set()
    print(f"  family lookup ERROR: {e}")

for name in ["Tata Sons Private Limited", "Tata Sons"]:
    try:
        hits = set(db.find_source_docs_mentioning_phrase(SID, name, cap=200))
    except Exception as e:
        print(f"  {name!r}: ERROR {e}")
        continue
    inter = sorted(short(d) for d in (hits & fam_docs))
    print(f"\n  {name!r} → {len(hits)} document(s) corpus-wide, "
          f"{len(inter)} of them service agreements:")
    for d in inter[:20]:
        print(f"      {d}")


rule("4. What the pipeline's own resolvers currently return for Q85")
Q85 = ("What is the execution date of the services agreement entered into by "
       "Tata Sons Private Limited having its registered office at Bombay House, "
       "24 Homi Mody Street, Mumbai?")
try:
    print(f"  _resolve_docs_by_party      : "
          f"{sorted(short(d) for d in wiki._resolve_docs_by_party(Q85, SID))}")
except Exception as e:
    print(f"  _resolve_docs_by_party      : ERROR {e}")
try:
    _fam, _fd = wiki._question_family_scope(Q85, SID)
    print(f"  _question_family_scope      : {_fam} ({len(_fd)} docs)")
except Exception as e:
    print(f"  _question_family_scope      : ERROR {e}")
try:
    print(f"  resolve_scope               : {wiki.resolve_scope(Q85, SID)['method']}")
except Exception as e:
    print(f"  resolve_scope               : ERROR {e}")

print("\nDone. Paste this whole output back.")
