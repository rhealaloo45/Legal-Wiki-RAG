"""
reingest_missing_docs.py — one-time targeted recovery for docs that never got
pages during a stalled/interrupted ingest run.

Finds every uploaded file for a session that has no corresponding rows in
`pages`, and runs wiki.ingest() on just those files. Existing pages for other
documents are untouched — _atomic_merge only ever adds/updates pages produced
from the given file's own content, so already-ingested docs are not affected.

Also fixes sessions.json's "files" count back to the true total uploaded
file count (uploading a partial batch through the normal /upload endpoint
would otherwise overwrite it to the small batch size).

Usage:
    cd app
    python reingest_missing_docs.py <session_id>
"""

from __future__ import annotations

import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("reingest_missing_docs")


def _find_missing_docs(conn, session_id: str, uploaded_paths: list[str]) -> list[str]:
    from sqlalchemy import text

    rows = conn.execute(
        text("SELECT DISTINCT source_doc FROM pages WHERE session_id = :sid"),
        {"sid": session_id},
    )
    have = {r.source_doc for r in rows if r.source_doc}

    missing = []
    for path in uploaded_paths:
        doc_name = os.path.basename(path)
        if doc_name not in have:
            missing.append(path)
    return missing


def reingest_missing(session_id: str) -> None:
    import config

    if not config.USE_DATABASE:
        logger.error("DATABASE_URL is not set — this script only applies to PostgreSQL mode.")
        sys.exit(1)

    from services import db as _db
    from services import wiki

    prefix = f"{session_id}_"
    uploaded_paths = [
        os.path.join(config.UPLOAD_PATH, f)
        for f in os.listdir(config.UPLOAD_PATH)
        if f.startswith(prefix)
    ]
    if not uploaded_paths:
        logger.error("No uploaded files found on disk for session %s", session_id)
        sys.exit(1)

    engine = _db.get_engine()
    with engine.connect() as conn:
        missing = _find_missing_docs(conn, session_id, uploaded_paths)

    if not missing:
        logger.info("Nothing missing — every uploaded file already has pages.")
    else:
        logger.info("Missing docs to ingest: %d", len(missing))
        for path in missing:
            doc_name = os.path.basename(path)
            logger.info("Ingesting: %s", doc_name)
            try:
                result = wiki.ingest(path, session_id)
                logger.info("  done — %d pages", (result or {}).get("pages_updated", 0))
            except Exception as e:
                logger.error("  FAILED: %s: %s", doc_name, e)

    # Fix the session's displayed doc count back to the true uploaded total.
    with open(config.SESSIONS_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    if session_id in sessions:
        true_total = len(uploaded_paths)
        old = sessions[session_id].get("files")
        sessions[session_id]["files"] = true_total
        with open(config.SESSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2)
        logger.info("Fixed session doc count: %s -> %d", old, true_total)

    logger.info("Reingest complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python reingest_missing_docs.py <session_id>")
        sys.exit(1)
    reingest_missing(sys.argv[1])
