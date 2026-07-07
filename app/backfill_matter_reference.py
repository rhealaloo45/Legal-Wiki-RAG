"""
backfill_matter_reference.py — one-time backfill for the matter_reference metadata field

matter_reference was added to the ingest metadata schema after some documents
were already ingested, so those documents have no matter_reference value in
page_metadata even though the document itself may state one. This script
finds documents missing the field, runs a small extraction call against a
sample of that document's already-ingested page content (not the original
file — cheap, no re-ingest), and backfills the value if one is found.

Cosmetic metadata completeness, not an answer-correctness fix — safe to run
multiple times (only processes documents where matter_reference is still
NULL) and doesn't touch any other stored data.

Usage:
    cd app
    python3 backfill_matter_reference.py               # all sessions
    python3 backfill_matter_reference.py <session_id>  # single session
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("backfill_matter_reference")

# Cap on how much page content per document to feed the extraction prompt —
# a matter/case/docket number, if present, is almost always in the opening
# pages (caption, header, recitals), so this stays cheap without needing the
# full document.
_MAX_SAMPLE_CHARS = 4000

_EXTRACTION_PROMPT = """\
You are extracting a single metadata field from excerpts of a legal document.

Find the Matter/case/docket/reference number printed in the document's header
or caption — the kind of identifier a law firm or court assigns to a specific
matter (e.g. "Matter No. 2024-0451", "Case No. CV-2023-1122", "File Ref: TL/2024/88").
Do NOT invent one. If no such reference is stated anywhere in the excerpts,
respond with exactly: null

DOCUMENT EXCERPTS:
{excerpts}

Respond with ONLY the reference string verbatim as it appears in the text, or
the word null — nothing else, no explanation."""


def _get_docs_missing_matter_reference(conn, target_session: str | None) -> list[tuple[str, str]]:
    """Return (session_id, source_doc) pairs that have pages but no non-null
    matter_reference in page_metadata."""
    from sqlalchemy import text

    where_session = "AND p.session_id = :sid" if target_session else ""
    params = {"sid": target_session} if target_session else {}

    rows = conn.execute(
        text(f"""
            SELECT DISTINCT p.session_id, p.source_doc
            FROM pages p
            WHERE p.source_doc IS NOT NULL AND p.source_doc != ''
              {where_session}
              AND NOT EXISTS (
                  SELECT 1 FROM page_metadata pm
                  WHERE pm.session_id = p.session_id
                    AND pm.title = p.source_doc
                    AND pm.matter_reference IS NOT NULL
              )
            ORDER BY p.session_id, p.source_doc
        """),
        params,
    )
    return [(r.session_id, r.source_doc) for r in rows]


def _get_sample_text(conn, session_id: str, source_doc: str) -> str:
    """Concatenate early page content for a document, capped at _MAX_SAMPLE_CHARS."""
    from sqlalchemy import text

    rows = conn.execute(
        text("""
            SELECT title, content
            FROM pages
            WHERE session_id = :sid AND source_doc = :doc
            ORDER BY title
        """),
        {"sid": session_id, "doc": source_doc},
    )
    parts = []
    total = 0
    for row in rows:
        chunk = f"## {row.title}\n{row.content}\n\n"
        parts.append(chunk)
        total += len(chunk)
        if total >= _MAX_SAMPLE_CHARS:
            break
    return "".join(parts)[:_MAX_SAMPLE_CHARS]


def backfill(target_session: str | None = None) -> None:
    import config

    if not config.USE_DATABASE:
        logger.error(
            "DATABASE_URL is not set — backfill only applies to PostgreSQL mode."
        )
        sys.exit(1)

    from services import db as _db
    from services import llm as _llm

    engine = _db.get_engine()

    with engine.connect() as conn:
        docs = _get_docs_missing_matter_reference(conn, target_session)

    if not docs:
        logger.info("Nothing to backfill — every document already has a matter_reference.")
        return

    logger.info("Documents to check: %d", len(docs))

    total_found = 0
    total_null = 0
    total_errors = 0

    for session_id, source_doc in docs:
        with engine.connect() as conn:
            sample = _get_sample_text(conn, session_id, source_doc)

        if not sample.strip():
            logger.warning("  [%s] no page content found — skipping", source_doc)
            continue

        try:
            raw, _ = _llm.ask(
                _EXTRACTION_PROMPT.format(excerpts=sample),
                pipeline="wiki",
                max_tokens=config.MAX_TOKENS_MATTER_REFERENCE,
                fast=True,
            )
            value = raw.strip().strip('"').strip()
        except Exception as e:
            logger.error("  [%s] extraction call failed: %s", source_doc, e)
            total_errors += 1
            continue

        if not value or value.lower() == "null":
            total_null += 1
            continue

        try:
            _db.upsert_metadata(session_id, source_doc, {"matter_reference": value})
            total_found += 1
            logger.info("  [%s] matter_reference = %r", source_doc, value)
        except Exception as e:
            logger.error("  [%s] upsert failed: %s", source_doc, e)
            total_errors += 1

    logger.info(
        "Backfill complete — %d found and stored, %d confirmed absent, %d errors.",
        total_found, total_null, total_errors,
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    backfill(target_session=target)
