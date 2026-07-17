"""
reingest_stuck_docs.py — targeted recovery for docs left mid-merge/synthesis
when an ingest run was killed (Docker/machine went down mid-ingest).

Unlike reingest_missing_docs.py (docs with ZERO pages), these docs already
have PARTIAL pages from the interrupted run. Re-running wiki.ingest() as-is
would merge onto that incomplete state rather than rebuild cleanly, so this
first deletes each stuck doc's own pages/metadata/embeddings/relations/
source_positions (scoped by source_doc, and by the page titles that belong to
it), then re-ingests fresh from disk — same end state as a first-time ingest.

Usage:
    cd app
    python reingest_stuck_docs.py <session_id> <doc_name1> [<doc_name2> ...]
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("reingest_stuck_docs")


def reingest_stuck(session_id: str, doc_names: list[str]) -> None:
    import config
    from sqlalchemy import text
    from services import db as _db
    from services import wiki

    # app.py wires TESSERACT_CMD into pytesseract at Flask startup; standalone
    # scripts need to do it themselves or every scanned-PDF page silently
    # fails OCR with "tesseract is not installed".
    if config.TESSERACT_CMD:
        from services.reader import configure_tesseract
        configure_tesseract(config.TESSERACT_CMD)

    engine = _db.get_engine()
    with engine.begin() as conn:
        for doc_name in doc_names:
            titles = [r.title for r in conn.execute(
                text("SELECT title FROM pages WHERE session_id = :sid AND source_doc = :sd"),
                {"sid": session_id, "sd": doc_name},
            )]
            logger.info("%s: clearing %d existing pages", doc_name, len(titles))
            if titles:
                emb_tbl = _db._emb_table_name()
                conn.execute(text(f"DELETE FROM {emb_tbl} WHERE session_id = :sid AND title = ANY(:titles)"),
                             {"sid": session_id, "titles": titles})
                conn.execute(text("DELETE FROM page_metadata WHERE session_id = :sid AND title = ANY(:titles)"),
                             {"sid": session_id, "titles": titles})
                conn.execute(text("DELETE FROM contradictions WHERE session_id = :sid AND page_title = ANY(:titles)"),
                             {"sid": session_id, "titles": titles})
                conn.execute(text("DELETE FROM relations WHERE session_id = :sid AND (from_title = ANY(:titles) OR to_title = ANY(:titles))"),
                             {"sid": session_id, "titles": titles})
            conn.execute(text("DELETE FROM source_positions WHERE session_id = :sid AND source_doc = :sd"),
                         {"sid": session_id, "sd": doc_name})
            conn.execute(text("DELETE FROM pages WHERE session_id = :sid AND source_doc = :sd"),
                         {"sid": session_id, "sd": doc_name})

    for doc_name in doc_names:
        path = os.path.join(config.UPLOAD_PATH, doc_name)
        if not os.path.exists(path):
            logger.error("  MISSING ON DISK, skipping: %s", doc_name)
            continue
        logger.info("Ingesting: %s", doc_name)
        try:
            result = wiki.ingest(path, session_id)
            logger.info("  done — %d pages", (result or {}).get("pages_updated", 0))
        except Exception as e:
            logger.error("  FAILED: %s: %s", doc_name, e)

    logger.info("Reingest complete.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: python reingest_stuck_docs.py <session_id> <doc_name1> [<doc_name2> ...]")
        sys.exit(1)
    reingest_stuck(sys.argv[1], sys.argv[2:])
