"""
backfill_embeddings.py — Phase 3 one-time backfill

Finds every session that has pages in PostgreSQL but is missing embeddings,
then embeds their summaries and stores them in page_embeddings.

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

    if target_session:
        return [target_session]

    rows = conn.execute(
        text("""
            SELECT DISTINCT p.session_id
            FROM pages p
            WHERE NOT EXISTS (
                SELECT 1 FROM page_embeddings pe
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

    rows = conn.execute(
        text("""
            SELECT title, summary, content
            FROM pages
            WHERE session_id = :sid
              AND title NOT IN (
                  SELECT title FROM page_embeddings WHERE session_id = :sid
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


def backfill(target_session: str | None = None, batch_size: int = 16) -> None:
    import config

    if not config.USE_DATABASE:
        logger.error(
            "DATABASE_URL is not set — backfill only applies to PostgreSQL mode. "
            "File-based sessions use BM25+LLM for page selection."
        )
        sys.exit(1)

    from services import db as _db
    from services import embedder as _embedder
    from sqlalchemy import text

    engine = _db.get_engine()

    with engine.connect() as conn:
        sessions = _get_sessions_with_missing_embeddings(conn, target_session)

    if not sessions:
        logger.info("Nothing to backfill — all pages already have embeddings.")
        return

    logger.info("Sessions to backfill: %d", len(sessions))

    total_embedded = 0
    total_skipped  = 0

    for session_id in sessions:
        with engine.connect() as conn:
            pages = _get_unembedded_pages(conn, session_id)

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
                    _db.upsert_embedding(session_id, title, embedding)
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
        logger.info(
            "  [%s] complete — %d embedded, %d errors",
            session_id, session_embedded, session_errors,
        )

    logger.info(
        "Backfill complete — %d embeddings stored, %d skipped due to errors.",
        total_embedded, total_skipped,
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    backfill(target_session=target)
