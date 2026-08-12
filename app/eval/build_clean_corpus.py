"""Copy the real (non-decoy) documents of a session into a clean session.

The audit corpus mixes 46 real Tata documents with 448 synthetic ``Test_<TYPE>_<NN>``
fixtures that will not exist in production. Retrieval competes against those
fixtures, so a score measured on the mixed corpus is not a score of the shipped
product. This builds the production-representative session to measure against.

Copies pages and every table retrieval depends on. No re-ingest, no LLM cost.
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sqlalchemy import text

from services import db

SRC = os.getenv("CLEAN_SRC_SESSION", "3a66b0ab-a9cc-48f0-a3f3-b0ab863936fe")
DST = os.getenv("CLEAN_DST_SESSION", "prodcorpus-46")

# Synthetic fixtures are named Test_<TYPE>_<NN>. The real documents live under a
# folder that also contains "Test_" ("Legal AI - Test_NDA (1)_NDA 5..."), so the
# trailing number is what actually separates them — matching bare "Test_" would
# delete the entire corpus.
DECOY = "source_doc ~ 'Test_[A-Za-z]+_[0-9]+'"

TABLES = ("pages", "page_embeddings_azure", "page_metadata", "clause_map",
          "source_positions", "relations")


def main() -> None:
    engine = db.get_engine()
    with engine.begin() as conn:
        for t in TABLES:
            conn.execute(text(f"DELETE FROM {t} WHERE session_id = :d"), {"d": DST})

        # content_tsv is a generated column — omitted so Postgres recomputes it.
        conn.execute(text(f"""
            INSERT INTO pages (session_id, title, content, summary, source_doc,
                               contradiction_flagged, variants, append_count,
                               char_count, last_modified)
            SELECT :d, title, content, summary, source_doc,
                   contradiction_flagged, variants, append_count,
                   char_count, last_modified
            FROM pages
            WHERE session_id = :s AND source_doc IS NOT NULL
              AND title NOT LIKE 'Q:%' AND NOT ({DECOY})
        """), {"s": SRC, "d": DST})

        conn.execute(text("""
            INSERT INTO page_embeddings_azure (session_id, title, embedding, doc_family)
            SELECT :d, e.title, e.embedding, e.doc_family
            FROM page_embeddings_azure e
            WHERE e.session_id = :s
              AND e.title IN (SELECT title FROM pages WHERE session_id = :d)
        """), {"s": SRC, "d": DST})

        # page_metadata is keyed by SOURCE DOC name, not page title — see the
        # pm.title = p.source_doc join in db.backfill_embedding_families.
        conn.execute(text("""
            INSERT INTO page_metadata
            SELECT :d, title, governing_law, jurisdiction, effective_date,
                   termination_notice, liability_cap, ip_ownership, parties,
                   auto_renewal, notice_period, payment_terms, matter_reference,
                   doc_type, doc_family
            FROM page_metadata
            WHERE session_id = :s
              AND title IN (SELECT DISTINCT source_doc FROM pages WHERE session_id = :d)
        """), {"s": SRC, "d": DST})

        conn.execute(text("""
            INSERT INTO clause_map
            SELECT :d, source_doc, clause_num, heading, page_title
            FROM clause_map
            WHERE session_id = :s
              AND source_doc IN (SELECT DISTINCT source_doc FROM pages WHERE session_id = :d)
        """), {"s": SRC, "d": DST})

        conn.execute(text("""
            INSERT INTO source_positions
            SELECT :d, source_doc, page_num, char_start, char_end
            FROM source_positions
            WHERE session_id = :s
              AND source_doc IN (SELECT DISTINCT source_doc FROM pages WHERE session_id = :d)
        """), {"s": SRC, "d": DST})

        # Both endpoints must survive the filter, or the graph gains edges
        # pointing at pages this session does not have.
        conn.execute(text("""
            INSERT INTO relations (session_id, from_title, to_title, label)
            SELECT :d, from_title, to_title, label
            FROM relations
            WHERE session_id = :s
              AND from_title IN (SELECT title FROM pages WHERE session_id = :d)
              AND to_title   IN (SELECT title FROM pages WHERE session_id = :d)
        """), {"s": SRC, "d": DST})

    with engine.connect() as conn:
        for t in TABLES:
            n = conn.execute(text(f"SELECT count(*) FROM {t} WHERE session_id = :d"),
                             {"d": DST}).scalar()
            print(f"  {t:24} {n}")
        print("  distinct source docs    ",
              conn.execute(text("SELECT count(DISTINCT source_doc) FROM pages WHERE session_id = :d"),
                           {"d": DST}).scalar())
        print("  decoy rows remaining    ",
              conn.execute(text(f"SELECT count(*) FROM pages WHERE session_id = :d AND {DECOY}"),
                           {"d": DST}).scalar())
        print("  doc families            ",
              conn.execute(text("SELECT count(DISTINCT doc_family) FROM page_metadata "
                                "WHERE session_id = :d AND doc_family IS NOT NULL"),
                           {"d": DST}).scalar())


if __name__ == "__main__":
    main()
