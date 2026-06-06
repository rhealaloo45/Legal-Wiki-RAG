"""
PostgreSQL abstraction layer — replaces index.json file I/O.

All functions are synchronous (matching the threaded Flask/wiki stack).
Requires: sqlalchemy>=2.0, psycopg2-binary, pgvector.
Activated only when DATABASE_URL is set in the environment.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    import config
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool

    url = config.DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL is not set — cannot initialise PostgreSQL engine")

    _engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    _init_schema(_engine)
    return _engine


def _init_schema(engine) -> None:
    """Create all tables on first connect. Idempotent (IF NOT EXISTS throughout)."""
    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pages (
                id                    BIGSERIAL PRIMARY KEY,
                session_id            TEXT NOT NULL,
                title                 TEXT NOT NULL,
                content               TEXT NOT NULL DEFAULT '',
                summary               TEXT NOT NULL DEFAULT '',
                source_doc            TEXT NOT NULL DEFAULT '',
                contradiction_flagged BOOLEAN NOT NULL DEFAULT FALSE,
                variants              JSONB,
                append_count          INT NOT NULL DEFAULT 0,
                char_count            INT NOT NULL DEFAULT 0,
                last_modified         TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (session_id, title)
            )
        """))

        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS pages_session_source_idx ON pages (session_id, source_doc)"
        ))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS relations (
                id         BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                from_title TEXT NOT NULL,
                to_title   TEXT NOT NULL,
                label      TEXT NOT NULL DEFAULT '',
                UNIQUE (session_id, from_title, to_title, label)
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS page_embeddings (
                session_id TEXT NOT NULL,
                title      TEXT NOT NULL,
                embedding  vector(1536),
                PRIMARY KEY (session_id, title)
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ingest_progress (
                session_id TEXT PRIMARY KEY,
                data       JSONB NOT NULL DEFAULT '{}',
                started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS page_metadata (
                session_id         TEXT NOT NULL,
                title              TEXT NOT NULL,
                governing_law      TEXT,
                jurisdiction       TEXT,
                effective_date     TEXT,
                termination_notice TEXT,
                liability_cap      TEXT,
                ip_ownership       TEXT,
                parties            TEXT,
                auto_renewal       TEXT,
                notice_period      TEXT,
                payment_terms      TEXT,
                PRIMARY KEY (session_id, title)
            )
        """))

        conn.commit()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def get_pages(session_id: str) -> dict[str, dict]:
    """Return all pages for a session as {title: {content, summary, source_doc, ...}}."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT title, content, summary, source_doc, contradiction_flagged, variants "
                 "FROM pages WHERE session_id = :sid"),
            {"sid": session_id},
        )
        pages: dict[str, dict] = {}
        for row in rows:
            page: dict[str, Any] = {
                "content": row.content,
                "summary": row.summary,
                "source_doc": row.source_doc,
            }
            if row.contradiction_flagged:
                page["contradiction_flagged"] = True
            if row.variants is not None:
                page["variants"] = row.variants
            pages[row.title] = page
        return pages


def get_page(session_id: str, title: str) -> dict | None:
    """Return a single page dict or None if it does not exist."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT content, summary, source_doc, contradiction_flagged, variants "
                 "FROM pages WHERE session_id = :sid AND title = :title"),
            {"sid": session_id, "title": title},
        ).fetchone()
        if row is None:
            return None
        page: dict[str, Any] = {
            "content": row.content,
            "summary": row.summary,
            "source_doc": row.source_doc,
        }
        if row.contradiction_flagged:
            page["contradiction_flagged"] = True
        if row.variants is not None:
            page["variants"] = row.variants
        return page


def get_page_titles(session_id: str) -> list[str]:
    """Return only page titles for a session (cheaper than get_pages for file-name lookups)."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT title FROM pages WHERE session_id = :sid"),
            {"sid": session_id},
        )
        return [row.title for row in rows]


def get_all_page_titles_and_content(session_id: str) -> dict[str, str]:
    """Return {title: content} for all pages. Used for the cross-reference pass."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT title, content FROM pages WHERE session_id = :sid"),
            {"sid": session_id},
        )
        return {row.title: row.content for row in rows}


def count_pages(session_id: str) -> int:
    """Return the number of pages stored for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM pages WHERE session_id = :sid"),
            {"sid": session_id},
        ).scalar()
        return result or 0


def count_relations(session_id: str) -> int:
    """Return the number of relations stored for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM relations WHERE session_id = :sid"),
            {"sid": session_id},
        ).scalar()
        return result or 0


def upsert_page(
    session_id: str,
    title: str,
    content: str,
    summary: str,
    source_doc: str,
    contradiction_flagged: bool = False,
    variants: list | None = None,
) -> None:
    """Insert or fully replace a page row."""
    from sqlalchemy import text
    engine = get_engine()
    variants_json = json.dumps(variants) if variants is not None else None
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO pages
                    (session_id, title, content, summary, source_doc,
                     contradiction_flagged, variants, append_count, char_count, last_modified)
                VALUES
                    (:sid, :title, :content, :summary, :source_doc,
                     :cf, CAST(:variants AS jsonb), 1, :char_count, now())
                ON CONFLICT (session_id, title) DO UPDATE SET
                    content               = EXCLUDED.content,
                    summary               = EXCLUDED.summary,
                    source_doc            = EXCLUDED.source_doc,
                    contradiction_flagged = EXCLUDED.contradiction_flagged,
                    variants              = EXCLUDED.variants,
                    append_count          = pages.append_count + 1,
                    char_count            = EXCLUDED.char_count,
                    last_modified         = now()
            """),
            {
                "sid": session_id,
                "title": title,
                "content": content,
                "summary": summary,
                "source_doc": source_doc,
                "cf": contradiction_flagged,
                "variants": variants_json,
                "char_count": len(content),
            },
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------

def get_relations(session_id: str) -> list[dict]:
    """Return all relations for a session as [{from, to, label}]."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT from_title, to_title, label FROM relations WHERE session_id = :sid"),
            {"sid": session_id},
        )
        return [{"from": r.from_title, "to": r.to_title, "label": r.label} for r in rows]


def upsert_relation(session_id: str, from_title: str, to_title: str, label: str) -> None:
    """Insert a single relation if it doesn't exist."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO relations (session_id, from_title, to_title, label)
                VALUES (:sid, :from_title, :to_title, :label)
                ON CONFLICT (session_id, from_title, to_title, label) DO NOTHING
            """),
            {"sid": session_id, "from_title": from_title, "to_title": to_title, "label": label},
        )
        conn.commit()


def bulk_upsert_relations(session_id: str, rels: list[tuple[str, str, str]]) -> None:
    """Batch-insert (from_title, to_title, label) tuples — skips existing rows."""
    if not rels:
        return
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO relations (session_id, from_title, to_title, label)
                VALUES (:sid, :from_title, :to_title, :label)
                ON CONFLICT (session_id, from_title, to_title, label) DO NOTHING
            """),
            [
                {"sid": session_id, "from_title": f, "to_title": t, "label": l}
                for f, t, l in rels
            ],
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Progress store (S5)
# ---------------------------------------------------------------------------

def get_progress(session_id: str) -> dict:
    """Return the current progress dict for a session, or {} if none exists."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT data FROM ingest_progress WHERE session_id = :sid"),
            {"sid": session_id},
        ).fetchone()
        if row is None:
            return {}
        data = row.data
        return dict(data) if data else {}


def set_progress(session_id: str, data: dict) -> None:
    """Upsert the progress dict for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO ingest_progress (session_id, data, updated_at)
                VALUES (:sid, CAST(:data AS jsonb), now())
                ON CONFLICT (session_id) DO UPDATE SET
                    data       = EXCLUDED.data,
                    updated_at = now()
            """),
            {"sid": session_id, "data": json.dumps(data)},
        )
        conn.commit()


def delete_progress(session_id: str) -> None:
    """Remove the progress row for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM ingest_progress WHERE session_id = :sid"),
            {"sid": session_id},
        )
        conn.commit()


def cleanup_old_progress(days: int = 7) -> None:
    """Delete progress rows not updated in the last `days` days."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM ingest_progress WHERE updated_at < now() - make_interval(days => :days)"),
            {"days": days},
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------

def migrate_from_json(session_id: str, json_path: str) -> None:
    """Read an index.json and insert all pages/relations into the DB.

    Safe to call multiple times — ON CONFLICT DO NOTHING skips existing rows.
    """
    from sqlalchemy import text

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_pages = data.get("pages", {})
    raw_rels = data.get("relations", [])

    engine = get_engine()
    with engine.connect() as conn:
        for title, value in raw_pages.items():
            if isinstance(value, str):
                content = value
                summary = source_doc = ""
                contradiction_flagged = False
                variants_json = None
            else:
                content = value.get("content", "")
                summary = value.get("summary", "")
                source_doc = value.get("source_doc", "")
                contradiction_flagged = bool(value.get("contradiction_flagged", False))
                v = value.get("variants")
                variants_json = json.dumps(v) if v else None

            conn.execute(
                text("""
                    INSERT INTO pages
                        (session_id, title, content, summary, source_doc,
                         contradiction_flagged, variants, char_count)
                    VALUES
                        (:sid, :title, :content, :summary, :source_doc,
                         :cf, CAST(:variants AS jsonb), :char_count)
                    ON CONFLICT (session_id, title) DO NOTHING
                """),
                {
                    "sid": session_id,
                    "title": title,
                    "content": content,
                    "summary": summary,
                    "source_doc": source_doc,
                    "cf": contradiction_flagged,
                    "variants": variants_json,
                    "char_count": len(content),
                },
            )

        for rel in raw_rels:
            conn.execute(
                text("""
                    INSERT INTO relations (session_id, from_title, to_title, label)
                    VALUES (:sid, :from_title, :to_title, :label)
                    ON CONFLICT (session_id, from_title, to_title, label) DO NOTHING
                """),
                {
                    "sid": session_id,
                    "from_title": rel.get("from", ""),
                    "to_title": rel.get("to", ""),
                    "label": rel.get("label", ""),
                },
            )

        conn.commit()

    logger.info(
        "Migrated session %s from JSON: %d pages, %d relations",
        session_id, len(raw_pages), len(raw_rels),
    )
