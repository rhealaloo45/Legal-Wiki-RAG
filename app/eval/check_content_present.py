"""Is the ground-truth answer even IN the wiki?

Separates two failure modes that look identical from the outside: retrieval
picked the wrong pages (fixable in the query pipeline) versus ingest never
produced the page at all (fixable only by re-ingesting the source document).
Scoring a retrieval fix against a question of the second kind is wasted effort,
so this runs first.

For each question, searches the expected source document's pages for the
distinctive strings its ground truth turns on.
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sqlalchemy import text

from services import db

SESSION = os.getenv("CHECK_SESSION", "prodcorpus-46")

# (question, expected source doc substring, regex of what the ground truth needs)
PROBES = [
    ("Q14", "NDA 5",                     r"reasonable standard of commercial care|thirty-six months"),
    ("Q17", "Service Agreement 2",       r"pre-existing|work for hire"),
    ("Q18", "Service Agreement 4",       r"prevail|precedence|takes priority|supersede"),
    ("Q21", "Service Agreement 7",       r"terminate for convenience|fifteen months"),
    ("Q24", "Court Case Document 4",     r"Arbitration and Conciliation Act|milestone"),
    ("Q27", "Court Case Document 7",     r"Trade Marks Act|Code of Civil Procedure"),
    ("Q49", "Court Case Document 2",     r"06 July 2025|6 July 2025|July 2025"),
    ("Q56", "Court Case Document 6",     r"Tata Motors Passenger Vehicles"),
    ("Q58", "Court Case Document 7",     r"Trade Marks Act|Code of Civil Procedure"),
    ("Q61", "Joint Venture Agreement 3", r"18 November 2025|November 2025"),
    ("Q65", "Joint Venture Agreement 7", r"wellness beverage|functional pantry|premium nutrition"),
    ("Q67", "Legal Opinion 2",           r"dynamic injunction|takedown"),
    ("Q70", "Legal Opinion 7",           r"foreground IP|foreground intellectual"),
    ("Q74", "NDA 6",                     r"2045|net-zero|net zero"),
    ("Q76", "NDA 6",                     r"process parameters|metallurgical|furnace"),
    ("Q79", "NDA 7",                     r"Organic India|Capital Foods"),
    ("Q85", "Service Agreement 4",       r"\d{1,2}\s+\w+\s+20\d\d"),
    ("Q92", "Shareholder Agreement 4",   r"NourishNext"),
    ("Q94", "Shareholder Agreement 6",   r"OmniRetail|Trent"),
    ("Q101", "Court Case Document 3",    r"screenshot|transcript|payment instruction"),
    ("Q102", "Tata Brand Judgment 7",    r"investment holding|trust-led|pernicious"),
    ("Q105", "Tata Brand Judgment 5",    r"Western Freeway|Thane|real estate"),
]


def main() -> None:
    engine = db.get_engine()
    missing, present, nodoc = [], [], []
    with engine.connect() as conn:
        for qid, doc, pat in PROBES:
            rows = conn.execute(text("""
                SELECT title, content FROM pages
                WHERE session_id = :s AND source_doc ILIKE :doc
            """), {"s": SESSION, "doc": f"%{doc}%"}).fetchall()
            if not rows:
                nodoc.append(qid)
                print(f"{qid:6} DOC ABSENT      {doc}")
                continue
            hits = [t for t, c in rows if re.search(pat, c or "", re.I)]
            if hits:
                present.append(qid)
                print(f"{qid:6} content present  {doc:28} ({len(rows)} pages) -> {hits[0][:45]}")
            else:
                missing.append(qid)
                print(f"{qid:6} CONTENT MISSING  {doc:28} ({len(rows)} pages) /{pat[:40]}/")

    print(f"\npresent: {len(present)}  missing: {len(missing)}  doc absent: {len(nodoc)}")
    print("retrieval-fixable :", ", ".join(present))
    print("ingest-limited    :", ", ".join(missing + nodoc))
    json.dump({"present": present, "missing": missing, "nodoc": nodoc},
              open(os.path.join(os.environ.get("TEMP", "."), "content_presence.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
