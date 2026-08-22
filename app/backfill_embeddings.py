"""
backfill_embeddings.py — Phase 3 one-time backfill

Finds every session that has pages in PostgreSQL but is missing embeddings,
then embeds their summaries and stores them in the active embedding
provider's table (see db._emb_table_name — page_embeddings for nvidia,
page_embeddings_azure for azure, etc).

Safe to run multiple times — only processes pages that don't already have
an embedding row (uses NOT IN sub-query).

Usage:
    cd app
    python3 backfill_embeddings.py               # all sessions
    python3 backfill_embeddings.py <session_id>  # single session
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("backfill")


def _get_sessions_with_missing_embeddings(conn, target_session: str | None) -> list[str]:
    from sqlalchemy import text
    from services import db as _db

    if target_session:
        return [target_session]

    tbl = _db._emb_table_name()
    rows = conn.execute(
        text(f"""
            SELECT DISTINCT p.session_id
            FROM pages p
            WHERE NOT EXISTS (
                SELECT 1 FROM {tbl} pe
                WHERE pe.session_id = p.session_id
                  AND pe.title      = p.title
            )
            ORDER BY p.session_id
        """)
    )
    return [r.session_id for r in rows]


def _get_unembedded_pages(conn, session_id: str) -> list[tuple[str, str]]:
    """Return (title, embed_text) for pages missing embeddings in this session."""
    from sqlalchemy import text
    from services import db as _db

    tbl = _db._emb_table_name()
    rows = conn.execute(
        text(f"""
            SELECT title, summary, content
            FROM pages
            WHERE session_id = :sid
              AND title NOT IN (
                  SELECT title FROM {tbl} WHERE session_id = :sid
              )
        """),
        {"sid": session_id},
    )
    result = []
    for row in rows:
        # Prefer summary; fall back to first 400 chars of content
        embed_text = (row.summary or row.content or "")[:400].strip()
        if embed_text:
            result.append((row.title, embed_text))
    return result


def _session_wiki_id(conn, session_id: str) -> str:
    """The wiki_id already stamped on this session's pages — a session never
    spans two wikis (see services/wikis.py), so any one row's value is the
    session's value. Falls back to the default wiki for a session with no
    pages yet (nothing to backfill for it anyway)."""
    from sqlalchemy import text
    from services import db as _db
    row = conn.execute(
        text("SELECT wiki_id FROM pages WHERE session_id = :sid LIMIT 1"),
        {"sid": session_id},
    ).first()
    return row[0] if row and row[0] else _db.DEFAULT_WIKI_ID


def backfill(target_session: str | None = None, batch_size: int = 16) -> dict:
    import config

    if not config.USE_DATABASE:
        logger.error(
            "DATABASE_URL is not set — backfill only applies to PostgreSQL mode. "
            "File-based sessions use BM25+LLM for page selection."
        )
        if __name__ == "__main__":
            sys.exit(1)
        return {"embedded": 0, "errors": 0, "message": "DATABASE_URL not set"}

    from services import db as _db
    from services import embedder as _embedder
    from sqlalchemy import text

    engine = _db.get_engine()

    with engine.connect() as conn:
        sessions = _get_sessions_with_missing_embeddings(conn, target_session)

    if not sessions:
        logger.info("Nothing to backfill — all pages already have embeddings.")
        return {"embedded": 0, "errors": 0, "sessions": 0}

    logger.info("Sessions to backfill: %d", len(sessions))

    total_embedded = 0
    total_skipped  = 0

    for session_id in sessions:
        with engine.connect() as conn:
            pages = _get_unembedded_pages(conn, session_id)
            wiki_id = _session_wiki_id(conn, session_id)

        if not pages:
            logger.info("  [%s] already fully embedded — skipping", session_id)
            continue

        logger.info("  [%s] embedding %d pages ...", session_id, len(pages))
        session_embedded = 0
        session_errors   = 0

        # Process in batches to stay within API rate limits
        for i in range(0, len(pages), batch_size):
            chunk = pages[i : i + batch_size]
            titles = [t for t, _ in chunk]
            texts  = [txt for _, txt in chunk]

            try:
                embeddings = _embedder.embed_batch(texts, is_query=False)
            except Exception as e:
                logger.error(
                    "    Batch %d failed (%s) — skipping %d pages",
                    i // batch_size + 1, e, len(chunk),
                )
                session_errors += len(chunk)
                continue

            for title, embedding in zip(titles, embeddings):
                try:
                    _db.upsert_embedding(wiki_id, session_id, title, embedding)
                    session_embedded += 1
                except Exception as e:
                    logger.error("    upsert_embedding failed for '%s': %s", title, e)
                    session_errors += 1

            logger.info(
                "    batch %d/%d done (%d embedded so far)",
                i // batch_size + 1,
                (len(pages) + batch_size - 1) // batch_size,
                session_embedded,
            )

        total_embedded += session_embedded
        total_skipped  += session_errors

        # Phase 1: stamp doc_family onto the freshly-embedded rows from existing
        # metadata (backfill path doesn't know each page's family at embed time),
        # so family-filtered vector search works for backfilled corpora too.
        try:
            fam_updated = _db.backfill_embedding_families(wiki_id, session_id)
            if fam_updated:
                logger.info("  [%s] doc_family populated on %d embedding rows", session_id, fam_updated)
        except Exception as _fam_err:
            logger.warning("  [%s] doc_family backfill failed: %s", session_id, _fam_err)

        logger.info(
            "  [%s] complete — %d embedded, %d errors",
            session_id, session_embedded, session_errors,
        )

    logger.info(
        "Backfill complete — %d embeddings stored, %d skipped due to errors.",
        total_embedded, total_skipped,
    )
    return {"embedded": total_embedded, "errors": total_skipped, "sessions": len(sessions)}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    backfill(target_session=target)
