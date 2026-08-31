"""
ingest_quality_check.py — flag under-extracted documents after a bulk ingest

Ingest synthesis (the LLM pass that turns raw OCR'd text into wiki pages) is a
judgment call, not a deterministic parse — it can silently under-extract a
document (skip its stamp/certificate details, leave metadata fields null,
produce far fewer pages than its content warrants) without ever raising an
error or showing up as "failed" in /progress. A 500-doc ingest has no manual
way to catch that at scale, so this script applies two cheap signals learned
from a real case (NDA 2 in a same-day test batch, before its ingest prompt
fix): a document is worth a human look if either

  1. its page_metadata row has very few populated fields (govering_law,
     jurisdiction, parties, matter_reference, etc.) — NDA 2 had only 3 of 11
     populated pre-fix (all 0 content-bearing ones null) despite the source
     text stating governing law, jurisdiction, and a stamp certificate number, or
  2. it produced an unusually small number of wiki pages relative to the
     batch's typical count.

This is a heuristic triage, not a correctness guarantee — a flagged document
needs a human/LLM spot-check against its source; a clean document isn't
proven perfect, just not showing the specific failure signature we already
caught once. Read-only — makes no changes to ingest data.

Usage:
    cd app
    python3 ingest_quality_check.py <session_id>
    python3 ingest_quality_check.py <session_id> --min-fields 3 --min-pages 5
"""

from __future__ import annotations

import argparse
import sys

import config
from services import db

# Metadata columns that carry actual document content (excludes doc_type /
# doc_family, which are always populated by ingest classification and would
# otherwise mask a genuinely empty row).
_CONTENT_FIELDS = [
    "governing_law", "jurisdiction", "effective_date", "termination_notice",
    "liability_cap", "ip_ownership", "parties", "auto_renewal",
    "notice_period", "payment_terms", "matter_reference",
]


def _session_wiki_id(session_id: str) -> str:
    """A session never spans two wikis (see services/wikis.py) — any one
    row's wiki_id is the session's value."""
    from sqlalchemy import text
    engine = db.get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT wiki_id FROM pages WHERE session_id = :sid LIMIT 1"),
            {"sid": session_id},
        ).first()
    return row[0] if row and row[0] else db.DEFAULT_WIKI_ID


def _page_counts_by_doc(wiki_id: str, session_id: str) -> dict[str, int]:
    from sqlalchemy import text
    engine = db.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT source_doc, COUNT(*) AS n
                FROM pages
                WHERE wiki_id = :w AND session_id = :sid AND source_doc <> ''
                GROUP BY source_doc
            """),
            {"w": wiki_id, "sid": session_id},
        )
        return {row.source_doc: row.n for row in rows}


def run(session_id: str, min_fields: int, min_pages: int) -> list[dict]:
    wiki_id = _session_wiki_id(session_id)
    docs = db.get_source_docs(wiki_id, session_id)
    page_counts = _page_counts_by_doc(wiki_id, session_id)

    if not docs:
        print(f"No documents found for session '{session_id}'.")
        return []

    avg_pages = sum(page_counts.values()) / len(page_counts) if page_counts else 0

    flagged = []
    print(f"{'Document':<70} {'Pages':>6} {'Fields':>7}  Flag")
    print("-" * 95)
    for doc in sorted(docs):
        meta = db.get_metadata(wiki_id, session_id, doc)
        field_count = sum(1 for f in _CONTENT_FIELDS if meta.get(f))
        pages = page_counts.get(doc, 0)

        reasons = []
        if field_count <= min_fields:
            reasons.append(f"only {field_count} metadata field(s)")
        if pages < min_pages:
            reasons.append(f"only {pages} wiki page(s)")

        flag = "  <-- " + "; ".join(reasons) if reasons else ""
        short_name = doc[-68:] if len(doc) > 68 else doc
        print(f"{short_name:<70} {pages:>6} {field_count:>7}{flag}")

        if reasons:
            flagged.append({"doc": doc, "pages": pages, "fields": field_count, "reasons": reasons})

    print("-" * 95)
    print(f"{len(docs)} documents, avg {avg_pages:.1f} wiki pages/doc, "
          f"{len(flagged)} flagged for manual spot-check.")
    if flagged:
        print("\nFlagged documents:")
        for f in flagged:
            print(f"  - {f['doc']}  ({'; '.join(f['reasons'])})")

    return flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session_id", help="Session to check")
    parser.add_argument("--min-fields", type=int, default=2,
                         help="Flag docs with this many or fewer populated metadata fields (default: 2)")
    parser.add_argument("--min-pages", type=int, default=5,
                         help="Flag docs with fewer than this many wiki pages (default: 5)")
    args = parser.parse_args()

    if not config.USE_DATABASE:
        print("USE_DATABASE is not enabled — this check requires the Postgres-backed wiki.")
        sys.exit(1)

    run(args.session_id, args.min_fields, args.min_pages)


if __name__ == "__main__":
    main()
