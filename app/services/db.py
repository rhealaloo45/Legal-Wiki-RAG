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


def _emb_table_name() -> str:
    """Vector-table name for the currently active embedding provider.

    Each provider gets its own table (page_embeddings for the legacy nvidia
    4096-dim vectors, page_embeddings_azure for the new 3072-dim ones, etc.)
    so switching providers never touches another provider's already-stored
    vectors — no drop, no forced re-embed, revert by flipping the env var.
    """
    import config
    provider = config.EMBEDDING_PROVIDER
    return "page_embeddings" if provider == "nvidia" else f"page_embeddings_{provider}"


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

    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    _init_schema(engine)
    # Only cache on success — if schema init throws (e.g. a transient DB
    # connectivity issue), the next call should retry from scratch instead
    # of being stuck forever on a schema-less engine.
    _engine = engine
    return _engine


def reset_engine() -> None:
    """Discard the cached engine so the next call to get_engine() re-initialises
    the schema.  Call this whenever the embedding provider changes at runtime so
    the dimension migration check fires with the new vector size."""
    global _engine
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
        _engine = None


# Arbitrary fixed key for the schema-init advisory lock — any int64 works,
# it just needs to be the same constant everywhere it's used.
_SCHEMA_INIT_LOCK_KEY = 727273001


def _init_schema(engine) -> None:
    """Create all tables on first connect. Idempotent (IF NOT EXISTS throughout).

    Multiple gunicorn worker processes can each call this independently on
    boot. Without serializing, concurrent workers race on the same ALTER
    TABLE/CREATE INDEX statements and can deadlock each other (seen live:
    two workers both migrating `pages` at once). A session-level Postgres
    advisory lock makes every worker but one simply wait its turn — the
    winner does the real work, the rest then run a fast no-op pass since
    everything below is IF NOT EXISTS.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _SCHEMA_INIT_LOCK_KEY})
        try:
            _run_schema_statements(conn, text)
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _SCHEMA_INIT_LOCK_KEY})


def _run_schema_statements(conn, text) -> None:
    # (indentation below kept one level deep intentionally — this used to be
    # the body of a `with engine.connect() as conn:` block)
    if True:
        # The app's DB role may not have privilege to CREATE EXTENSION even
        # when it's a no-op (already installed by an admin) — Postgres still
        # enforces the permission check on the attempt itself. Don't let a
        # permission error here abort the whole schema-init transaction and
        # skip every table below.
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as _ext_err:
            logger.warning(
                "Could not create/verify 'vector' extension (may already exist "
                "and this role lacks CREATE EXTENSION privilege — fine if an "
                "admin already ran it): %s", _ext_err,
            )
            conn.rollback()

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

        import config as _cfg
        _emb_dims = _cfg.get_embedding_dimensions()
        _emb_table = _emb_table_name()
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_emb_table} (
                session_id TEXT NOT NULL,
                title      TEXT NOT NULL,
                embedding  vector({_emb_dims}),
                doc_family TEXT,
                PRIMARY KEY (session_id, title)
            )
        """))
        # Dimension migration: if THIS provider's table already exists but was
        # created with a different vector size (e.g. dims changed for the same
        # provider), drop and recreate it. Each provider owns a separate table
        # (see _emb_table_name), so this never touches another provider's data —
        # switching EMBEDDING_PROVIDER just points at a different table.
        try:
            row = conn.execute(text("""
                SELECT (regexp_matches(
                    format_type(a.atttypid, a.atttypmod),
                    'vector\\((\\d+)\\)'
                ))[1]::int AS dim
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = :tbl
                  AND a.attname  = 'embedding'
                  AND NOT a.attisdropped
            """), {"tbl": _emb_table}).fetchone()
            if row and row.dim != _emb_dims:
                logger.warning(
                    "%s dimension mismatch (DB=%d, config=%d) — "
                    "recreating table (embeddings will be regenerated by backfill)",
                    _emb_table, row.dim, _emb_dims,
                )
                conn.execute(text(f"DROP INDEX IF EXISTS {_emb_table}_hnsw_idx"))
                conn.execute(text(f"DROP TABLE {_emb_table}"))
                conn.execute(text(f"""
                    CREATE TABLE {_emb_table} (
                        session_id TEXT NOT NULL,
                        title      TEXT NOT NULL,
                        embedding  vector({_emb_dims}),
                        doc_family TEXT,
                        PRIMARY KEY (session_id, title)
                    )
                """))
        except Exception as _dim_err:
            logger.warning("Could not check %s dimension: %s", _emb_table, _dim_err)
            conn.rollback()

        # doc_family (Phase 1) — added AFTER the dimension-migration block so the
        # guard applies to the final table whether it was freshly created, an old
        # pre-doc_family table, or just dropped+recreated above. Composite btree
        # index makes the family pre-filter cheap before the HNSW ORDER BY.
        try:
            conn.execute(text(f"""
                ALTER TABLE {_emb_table}
                ADD COLUMN IF NOT EXISTS doc_family TEXT
            """))
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {_emb_table}_family_idx
                ON {_emb_table} (session_id, doc_family)
            """))
        except Exception as _emb_fam_err:
            logger.warning("Could not add %s.doc_family column/index (may already exist): %s", _emb_table, _emb_fam_err)
            conn.rollback()

        # HNSW index for sub-5ms cosine similarity search at 140k+ pages.
        # pgvector ≤ 0.6 enforces a hard 2000-dimension limit on HNSW/IVFFlat.
        # If the configured embedding dimension exceeds 2000, skip the index and
        # fall back to an exact sequential scan — still correct, and fast enough
        # until the wiki grows beyond ~50k pages.  Upgrade pgvector to 0.7+
        # (which raises the limit to 16000) to re-enable the index at higher dims.
        if _emb_dims <= 2000:
            try:
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {_emb_table}_hnsw_idx
                    ON {_emb_table}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """))
            except Exception as _idx_err:
                logger.warning("Could not create HNSW index (pgvector limit?): %s", _idx_err)
                conn.rollback()
        else:
            logger.info(
                "EMBEDDING_DIMENSIONS=%d > 2000 — skipping HNSW index "
                "(exact cosine scan used instead; upgrade pgvector ≥ 0.7 to re-enable)",
                _cfg.EMBEDDING_DIMENSIONS,
            )

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
                matter_reference   TEXT,
                doc_type           TEXT,
                doc_family         TEXT,
                PRIMARY KEY (session_id, title)
            )
        """))

        # Existing DBs created before matter_reference was added won't get it
        # from CREATE TABLE IF NOT EXISTS above — add it explicitly.
        try:
            conn.execute(text("""
                ALTER TABLE page_metadata
                ADD COLUMN IF NOT EXISTS matter_reference TEXT
            """))
        except Exception as _matter_ref_err:
            logger.warning("Could not add matter_reference column (may already exist): %s", _matter_ref_err)
            conn.rollback()

        # doc_type / doc_family added later (Phase 0, 20k-doc scale work) — same
        # migration guard so pre-existing session DBs pick up the columns.
        try:
            conn.execute(text("""
                ALTER TABLE page_metadata
                ADD COLUMN IF NOT EXISTS doc_type TEXT
            """))
            conn.execute(text("""
                ALTER TABLE page_metadata
                ADD COLUMN IF NOT EXISTS doc_family TEXT
            """))
        except Exception as _docfam_err:
            logger.warning("Could not add doc_type/doc_family columns (may already exist): %s", _docfam_err)
            conn.rollback()

        # S2: FTS column + GIN index for O(log N) cross-reference (Phase 4)
        # Add generated tsvector column to pages if it doesn't exist yet.
        try:
            conn.execute(text("""
                ALTER TABLE pages
                ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            """))
        except Exception as _tsv_err:
            logger.warning("Could not add content_tsv column (may already exist): %s", _tsv_err)
            conn.rollback()
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS pages_content_tsv_gin_idx
                ON pages USING GIN(content_tsv)
            """))
        except Exception as _gin_err:
            logger.warning("Could not create GIN index (may already exist): %s", _gin_err)
            conn.rollback()

        # S4: Structured contradictions table (Phase 4)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id          BIGSERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL,
                page_title  TEXT NOT NULL,
                claim       TEXT,
                value_a     TEXT,
                source_a    TEXT,
                value_b     TEXT,
                source_b    TEXT,
                detected_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS contradictions_session_idx
            ON contradictions (session_id, page_title)
        """))

        # Chat messages table (conversational UX)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          BIGSERIAL PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                msg_type    TEXT NOT NULL DEFAULT 'text',
                metadata    JSONB,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chat_messages_session_idx
            ON chat_messages (session_id, created_at)
        """))

        # Source positions table (citation exact-location support)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS source_positions (
                session_id  TEXT NOT NULL,
                source_doc  TEXT NOT NULL,
                page_num    INT NOT NULL,
                char_start  INT NOT NULL,
                char_end    INT NOT NULL,
                PRIMARY KEY (session_id, source_doc, page_num)
            )
        """))

        # Clause number -> wiki page mapping. Ingest builds page titles from the
        # source's clause headings but strips the leading number ("5. Return,
        # Destruction..." becomes "Return, Destruction... – ..."), which made
        # every ask-by-clause-number question unanswerable even when the source
        # is numbered. Populated by backfill_clause_map.py from the original
        # PDFs; read at query time to pin retrieval. One clause may map to
        # several pages (ingest splits compound clauses like "Remedies, Term,
        # and Governing Law" into separate topic pages), hence page_title in
        # the primary key.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS clause_map (
                session_id  TEXT NOT NULL,
                source_doc  TEXT NOT NULL,
                clause_num  TEXT NOT NULL,
                heading     TEXT NOT NULL,
                page_title  TEXT NOT NULL,
                PRIMARY KEY (session_id, source_doc, clause_num, page_title)
            )
        """))

        # Per-query trace — stage timings, retrieval detail, LLM calls, for
        # the "how did the app arrive at this answer" debugging view. One row
        # per /query request, linked to the assistant chat_messages row it
        # produced. See services/tracing.py.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_traces (
                id              BIGSERIAL PRIMARY KEY,
                session_id      TEXT NOT NULL,
                wiki_session_id TEXT,
                message_id      BIGINT,
                question        TEXT NOT NULL,
                total_ms        INT NOT NULL DEFAULT 0,
                trace           JSONB NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS query_traces_session_idx
            ON query_traces (session_id, created_at)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS query_traces_message_idx
            ON query_traces (message_id)
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


# ---------------------------------------------------------------------------
# Embeddings (Phase 3 — pgvector search)
# ---------------------------------------------------------------------------

def upsert_embedding(session_id: str, title: str, embedding: list[float],
                     doc_family: str | None = None) -> None:
    """Store or update a page embedding vector.

    doc_family (Phase 1) is denormalized onto the embedding row so
    search_similar_pages can pre-filter by family without a join. Optional and
    backward-compatible — existing callers that omit it store NULL, which the
    unfiltered search path ignores.
    """
    from sqlalchemy import text
    engine = get_engine()
    tbl = _emb_table_name()
    # Format as pgvector literal string: [x1,x2,...] then CAST to vector
    emb_str = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {tbl} (session_id, title, embedding, doc_family)
                VALUES (:sid, :title, CAST(:embedding AS vector), :fam)
                ON CONFLICT (session_id, title) DO UPDATE SET
                    embedding  = EXCLUDED.embedding,
                    doc_family = COALESCE(EXCLUDED.doc_family, {tbl}.doc_family)
            """),
            {"sid": session_id, "title": title, "embedding": emb_str, "fam": doc_family},
        )
        conn.commit()


def search_similar_pages(
    session_id: str, query_embedding: list[float], limit: int = 25,
    doc_family: "str | list[str] | None" = None,
    exclude_cached: bool = False,
) -> list[str]:
    """Return page titles ordered by cosine similarity to query_embedding.

    Uses the HNSW index for sub-5ms lookup even at 140k+ pages.
    Returns [] if no embeddings exist for this session.

    doc_family (Phase 1): when provided (a single family or a list), the search
    is pre-filtered to embedding rows in those families before the ANN ordering.
    This narrows the candidate set for family-scoped questions ("across all
    NDAs") at 20k-doc scale, cutting near-neighbour noise. None = search the
    whole session (backward-compatible default).

    exclude_cached: drop cached "Q:" answer pages from the ranking, the same way
    find_source_docs_mentioning_phrase already does. Must be set whenever the
    caller has filtered those pages out of its own in-memory dict, because the
    caller's post-filter runs AFTER the LIMIT: a cached answer is by nature the
    nearest neighbour of the question that produced it, so the whole top-N is
    cached answers, every one is then discarded, and the vector channel
    contributes nothing at all. Measured on Q101 against the 7,245-embedding
    audit session — all 15 vector hits were "Q:" pages, retrieval silently
    degraded to BM25-only, and the page holding the answer never surfaced.
    """
    from sqlalchemy import text
    engine = get_engine()
    tbl = _emb_table_name()
    emb_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"
    params = {"sid": session_id, "embedding": emb_str, "limit": limit}
    family_clause = ""
    if doc_family:
        families = [doc_family] if isinstance(doc_family, str) else list(doc_family)
        if families:
            family_clause = "AND doc_family = ANY(:families)"
            params["families"] = families
    cached_clause = "AND title NOT LIKE 'Q:%'" if exclude_cached else ""
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT title
                FROM {tbl}
                WHERE session_id = :sid
                {family_clause}
                {cached_clause}
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """),
            params,
        )
        return [row.title for row in rows]


def backfill_embedding_families(session_id: str) -> int:
    """Populate page_embeddings.doc_family from existing metadata, no re-embed.

    Set-based UPDATE joining each embedding row → its page → that page's source
    document → the document's doc_family (Phase 0 metadata). Lets an
    already-embedded corpus gain family filtering without regenerating vectors —
    used by backfill_embeddings.py and safe to run repeatedly. Returns the number
    of rows updated.
    """
    from sqlalchemy import text
    engine = get_engine()
    tbl = _emb_table_name()
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                UPDATE {tbl} pe
                SET doc_family = pm.doc_family
                FROM pages p
                JOIN page_metadata pm
                  ON pm.session_id = p.session_id AND pm.title = p.source_doc
                WHERE pe.session_id = p.session_id
                  AND pe.title = p.title
                  AND pe.session_id = :sid
                  AND pm.doc_family IS NOT NULL
                  AND pe.doc_family IS DISTINCT FROM pm.doc_family
            """),
            {"sid": session_id},
        )
        conn.commit()
        return result.rowcount or 0


def count_embeddings(session_id: str) -> int:
    """Return the number of pages with embeddings stored for a session."""
    from sqlalchemy import text
    engine = get_engine()
    tbl = _emb_table_name()
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT COUNT(*) FROM {tbl} WHERE session_id = :sid"),
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


# ---------------------------------------------------------------------------
# S2: FTS cross-reference helpers (Phase 4)
# ---------------------------------------------------------------------------

def find_pages_mentioning_title(session_id: str, title: str) -> list[str]:
    """Return titles of pages whose content mentions the given title.

    Uses the GIN-indexed content_tsv column for O(log N) lookup instead of
    the O(N) Python substring scan.  Returns [] if the title produces no FTS
    tokens (e.g. a single stop-word title).
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT title FROM pages
                WHERE session_id = :sid
                  AND title      != :title
                  AND content_tsv @@ plainto_tsquery('english', :tokens)
            """),
            {"sid": session_id, "title": title, "tokens": title},
        )
        return [row.title for row in rows]


def find_source_docs_mentioning_phrase(
    session_id: str, phrase: str, cap: int = 25
) -> list[str]:
    """Return the distinct source_docs whose page CONTENT mentions ``phrase``.

    Used to resolve a document by a party name the user typed (e.g. "SteelLoop
    Resource Recovery", "Cold Chain Energy Services") when that name lives only
    in the document body — not in the filename, page-title identifier, or the
    (often redaction-masked) parties metadata. Uses phraseto_tsquery so the
    words must appear ADJACENTLY, which is what makes a multi-word party name
    document-specific instead of matching every doc that happens to share a
    common word. Cached "Q:" answer pages are excluded so a prior answer can't
    masquerade as a source document. GIN-indexed (content_tsv) → O(log N).

    Returns [] when the phrase yields no FTS tokens or matches nothing.
    """
    from sqlalchemy import text
    if not phrase or not phrase.strip():
        return []
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT source_doc FROM pages
                WHERE session_id = :sid
                  AND title NOT LIKE 'Q:%'
                  AND source_doc IS NOT NULL
                  AND content_tsv @@ phraseto_tsquery('english', :phrase)
                LIMIT :cap
            """),
            {"sid": session_id, "phrase": phrase, "cap": cap},
        )
        return [row.source_doc for row in rows]


def find_source_docs_by_title_tokens(
    session_id: str, tokens: list[str], kind_hint: str | None = None,
    cap: int = 25,
) -> list[str]:
    """Return distinct source_docs whose page TITLES contain EVERY token.

    Companion to ``find_source_docs_mentioning_phrase`` above, which searches
    page CONTENT. Content search cannot separate the parties to a matter from
    the parties merely NAMED in one — litigation documents recite the opposing
    side's officers in their procedural/discovery paragraphs, so a content
    search for two adversaries matches every document that quotes that
    boilerplate (measured on this corpus: "Aether" AND "Helios" hit 76 of ~115
    documents, nearly all of them unrelated matters reusing the same recital).

    Page TITLES are the discriminator, because ingest synthesises the matter's
    own short-name into every title it writes for a document ("Parties –
    Aether-Helios (Verified Complaint)", "Signature – Aether v Helios
    (Answer)"). Requiring both parties in the TITLE drops the same corpus from
    76 documents to 20 — the instruments actually BETWEEN those two parties.

    ``kind_hint`` adds one more ILIKE against the title, used to match the
    document-type word the question supplies ("the verified complaint", "the
    answer") against the parenthetical ingest appends to each title; on the
    measured corpus that narrows the 20 to exactly 1. Cached "Q:" answer pages
    are excluded so a prior answer cannot masquerade as a source document.

    Returns [] for fewer than one token or no match, so callers can treat an
    empty result as "no opinion" and fall through unchanged.
    """
    from sqlalchemy import text
    toks = [t.strip() for t in (tokens or []) if t and t.strip()]
    if not toks:
        return []
    # Bounded so a pathological question cannot build an unbounded predicate.
    toks = toks[:4]
    conds = " AND ".join(f"title ILIKE :t{i}" for i in range(len(toks)))
    params: dict = {f"t{i}": f"%{tok}%" for i, tok in enumerate(toks)}
    params.update({"sid": session_id, "cap": cap})
    if kind_hint and kind_hint.strip():
        conds += " AND title ILIKE :kind"
        params["kind"] = f"%{kind_hint.strip()}%"
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT DISTINCT source_doc FROM pages
                WHERE session_id = :sid
                  AND title NOT LIKE 'Q:%'
                  AND source_doc IS NOT NULL
                  AND {conds}
                LIMIT :cap
            """),
            params,
        )
        return [row.source_doc for row in rows]


# ---------------------------------------------------------------------------
# S3: Page compaction helpers (Phase 4)
# ---------------------------------------------------------------------------

def find_pages_due_for_compaction(
    session_id: str, append_threshold: int, char_threshold: int
) -> list[dict]:
    """Return pages that exceed the compaction thresholds."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT title, content, summary, variants, append_count, char_count, source_doc
                FROM pages
                WHERE session_id = :sid
                  AND (
                    append_count >= :at
                    OR (append_count >= 2 AND char_count >= :ct)
                  )
                ORDER BY append_count DESC, char_count DESC
            """),
            {"sid": session_id, "at": append_threshold, "ct": char_threshold},
        )
        return [dict(row._mapping) for row in rows]


def reset_page_after_compaction(
    session_id: str,
    title: str,
    content: str,
    summary: str,
    contradiction_flagged: bool,
) -> None:
    """Replace page content after re-synthesis and reset append_count to 0."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                UPDATE pages
                SET content               = :content,
                    summary               = :summary,
                    contradiction_flagged = :cf,
                    append_count          = 0,
                    char_count            = :char_count,
                    variants              = NULL,
                    last_modified         = now()
                WHERE session_id = :sid AND title = :title
            """),
            {
                "sid": session_id,
                "title": title,
                "content": content,
                "summary": summary,
                "cf": contradiction_flagged,
                "char_count": len(content),
            },
        )
        conn.commit()


# ---------------------------------------------------------------------------
# S4: Structured contradiction storage (Phase 4)
# ---------------------------------------------------------------------------

def upsert_contradiction(
    session_id: str,
    page_title: str,
    claim: str | None,
    value_a: str | None,
    source_a: str | None,
    value_b: str | None,
    source_b: str | None,
) -> None:
    """Record a detected contradiction for a page."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO contradictions
                    (session_id, page_title, claim, value_a, source_a, value_b, source_b)
                VALUES (:sid, :title, :claim, :va, :sa, :vb, :sb)
            """),
            {
                "sid": session_id, "title": page_title,
                "claim": claim, "va": value_a, "sa": source_a,
                "vb": value_b, "sb": source_b,
            },
        )
        conn.commit()


# ---------------------------------------------------------------------------
# C7: Metadata helpers (Phase 4)
# ---------------------------------------------------------------------------

_METADATA_COLUMNS = (
    "governing_law", "jurisdiction", "effective_date", "termination_notice",
    "liability_cap", "ip_ownership", "parties", "auto_renewal",
    "notice_period", "payment_terms", "matter_reference",
    # doc_type: the LLM's inferred free-text type ("NDA", "Master Services Agreement").
    # doc_family: doc_type normalized to a small controlled vocabulary (see
    # wiki._normalize_doc_family) — the queryable grouping key for family-scoped
    # retrieval ("across all NDAs") and metadata-filtered vector search (Phase 1).
    "doc_type", "doc_family",
)


def upsert_metadata(session_id: str, doc_name: str, metadata: dict) -> None:
    """Store document-level metadata extracted at ingest time.

    `doc_name` is used as the `title` key — Review mode looks up by doc_name.
    Only non-None values are written; existing values are preserved for fields
    not present in the new metadata dict.
    """
    if not metadata:
        return
    from sqlalchemy import text
    engine = get_engine()
    # Build dynamic SET clause for non-None fields only
    updates = {k: metadata[k] for k in _METADATA_COLUMNS if metadata.get(k) is not None}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = COALESCE(:{k}, page_metadata.{k})" for k in updates)
    params = {"sid": session_id, "title": doc_name, **updates}
    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT INTO page_metadata (session_id, title, {', '.join(updates)})
                VALUES (:sid, :title, {', '.join(':' + k for k in updates)})
                ON CONFLICT (session_id, title) DO UPDATE SET {set_clause}
            """),
            params,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Chat messages (conversational UX)
# ---------------------------------------------------------------------------

def insert_message(
    session_id: str,
    role: str,
    content: str,
    msg_type: str = "text",
    metadata: dict | None = None,
) -> int:
    """Insert a chat message and return its id."""
    from sqlalchemy import text
    engine = get_engine()
    meta_json = json.dumps(metadata) if metadata is not None else None
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                INSERT INTO chat_messages (session_id, role, content, msg_type, metadata)
                VALUES (:sid, :role, :content, :msg_type, CAST(:metadata AS jsonb))
                RETURNING id
            """),
            {
                "sid": session_id, "role": role, "content": content,
                "msg_type": msg_type, "metadata": meta_json,
            },
        ).fetchone()
        conn.commit()
        return row.id


def insert_trace(
    session_id: str,
    wiki_session_id: str,
    message_id: "int | None",
    question: str,
    total_ms: int,
    trace: dict,
) -> int:
    """Insert a query trace and return its id. See services/tracing.py."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                INSERT INTO query_traces (session_id, wiki_session_id, message_id, question, total_ms, trace)
                VALUES (:sid, :wsid, :mid, :q, :tms, CAST(:trace AS jsonb))
                RETURNING id
            """),
            {
                "sid": session_id, "wsid": wiki_session_id, "mid": message_id,
                "q": question, "tms": total_ms, "trace": json.dumps(trace),
            },
        ).fetchone()
        conn.commit()
        return row.id


def get_trace_by_message_id(message_id: int) -> "dict | None":
    """Return the most recent trace for a chat message, or None if untraced
    (e.g. it predates tracing, or was answered by a fast-path with DB off)."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT id, session_id, wiki_session_id, question, total_ms, trace, created_at
                FROM query_traces WHERE message_id = :mid
                ORDER BY id DESC LIMIT 1
            """),
            {"mid": message_id},
        ).fetchone()
        if not row:
            return None
        return {
            "id": row.id, "session_id": row.session_id, "wiki_session_id": row.wiki_session_id,
            "question": row.question, "total_ms": row.total_ms, "trace": row.trace,
            "created_at": row.created_at.isoformat() if row.created_at else "",
        }


def get_messages(session_id: str, limit: int = 50) -> list[dict]:
    """Return chat messages for a session, oldest first."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, role, content, msg_type, metadata, created_at
                FROM chat_messages
                WHERE session_id = :sid
                ORDER BY created_at ASC
                LIMIT :limit
            """),
            {"sid": session_id, "limit": limit},
        )
        result = []
        for r in rows:
            msg = {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "msg_type": r.msg_type,
                "metadata": r.metadata if r.metadata else {},
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            result.append(msg)
        return result


def get_recent_context(session_id: str, n: int = 5) -> list[dict]:
    """Return the last n messages for building conversation context."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content, msg_type
                FROM (
                    SELECT role, content, msg_type, created_at
                    FROM chat_messages
                    WHERE session_id = :sid AND msg_type IN ('text', 'answer')
                    ORDER BY created_at DESC
                    LIMIT :n
                ) sub
                ORDER BY created_at ASC
            """),
            {"sid": session_id, "n": n},
        )
        return [{"role": r.role, "content": r.content, "msg_type": r.msg_type} for r in rows]


def count_trailing_disambiguations(session_id: str, look_back: int = 6) -> int:
    """How many of this thread's most recent assistant turns were, in an unbroken
    run ending at the newest, a "which document?" prompt.

    A disambiguation prompt is answered by the user's next message, which is
    appended to the original question and re-run through the same matcher. When
    that reply carries no resolvable document token — "same document", "the one
    I just asked about" — the re-run fails identically and asks again, forever.
    Confirmed live: three consecutive prompts on one question, each answered in
    good faith, none resolving. Counting the unbroken run is what lets the caller
    stop asking a question the user has already tried to answer.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT msg_type
                FROM chat_messages
                WHERE session_id = :sid AND role = 'assistant'
                ORDER BY created_at DESC
                LIMIT :n
            """),
            {"sid": session_id, "n": look_back},
        )
        streak = 0
        for r in rows:
            if r.msg_type != "disambiguation":
                break
            streak += 1
        return streak


def get_recent_answer_scope(session_id: str, n: int = 1) -> list[dict]:
    """Return ``{method, docs}`` for the last n assistant answers, newest first.

    Reads the scope decision the query pipeline records in each answer's
    metadata JSONB. Consumed by wiki._carryover_scope: a question that names no
    document can inherit the scope of the document the conversation is
    demonstrably already about.

    Deliberately reads the recorded SCOPE, not ``files_used``. A file count
    cannot tell "scoped to one named reference" from "synthesised across the
    corpus" — in this corpus a single numbered reference routinely resolves to
    two files (a real document plus its zero-padded Test_* sibling), so any
    count-based rule is guesswork. The resolver's own method is the ground truth.

    Answers written before this metadata existed yield method="" and are
    therefore never inherited from.

    Also returns ``files`` (the answer's ``files_used``) for the ONE case the
    recorded scope cannot serve: a "broad"/"default" turn resolves no
    target_docs at all, so a comparative follow-up after a multi-document
    answer ("which agreement has the strictest requirement?") has nothing to
    inherit and silently widens to the whole corpus. ``files`` records what the
    previous answer actually drew on, which is the set such a question refers
    back to. Kept as a SEPARATE key so the count-based reasoning above still
    never governs the single-document carryover path — see
    wiki._carryover_comparative_set for the guards on its use.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT metadata
                FROM chat_messages
                WHERE session_id = :sid
                  AND role = 'assistant'
                  AND msg_type = 'answer'
                ORDER BY created_at DESC
                LIMIT :n
            """),
            {"sid": session_id, "n": n},
        )
        out: list[dict] = []
        for r in rows:
            md = r.metadata if isinstance(r.metadata, dict) else {}
            docs = md.get("scope_docs")
            files = md.get("files_used")
            out.append({
                "method": str(md.get("scope_method") or ""),
                "docs": [str(x) for x in docs if x] if isinstance(docs, list) else [],
                "files": [str(x) for x in files if x] if isinstance(files, list) else [],
            })
        return out


def delete_messages(session_id: str) -> None:
    """Delete all chat messages for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM chat_messages WHERE session_id = :sid"),
            {"sid": session_id},
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Source document helpers
# ---------------------------------------------------------------------------

def get_source_docs(session_id: str) -> list[str]:
    """Return distinct source_doc values for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT source_doc FROM pages
                WHERE session_id = :sid AND source_doc != ''
            """),
            {"sid": session_id},
        )
        return [r.source_doc for r in rows]


# ---------------------------------------------------------------------------
# Source positions (citation exact-location support)
# ---------------------------------------------------------------------------

def store_page_map(session_id: str, source_doc: str, page_map: list[dict]) -> None:
    """Store page-level character positions for a source document."""
    if not page_map:
        return
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        for entry in page_map:
            conn.execute(
                text("""
                    INSERT INTO source_positions (session_id, source_doc, page_num, char_start, char_end)
                    VALUES (:sid, :doc, :pn, :cs, :ce)
                    ON CONFLICT (session_id, source_doc, page_num) DO UPDATE SET
                        char_start = EXCLUDED.char_start,
                        char_end   = EXCLUDED.char_end
                """),
                {
                    "sid": session_id, "doc": source_doc,
                    "pn": entry["page_num"], "cs": entry["char_start"], "ce": entry["char_end"],
                },
            )
        conn.commit()


def get_page_map(session_id: str, source_doc: str) -> list[dict]:
    """Return page positions for a source document."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT page_num, char_start, char_end
                FROM source_positions
                WHERE session_id = :sid AND source_doc = :doc
                ORDER BY page_num
            """),
            {"sid": session_id, "doc": source_doc},
        )
        return [{"page_num": r.page_num, "char_start": r.char_start, "char_end": r.char_end} for r in rows]


def find_quote_position(session_id: str, source_doc: str, quote_text: str) -> dict:
    """Find the page number and character offset of a quote in a source document.

    Loads the document text from the upload path, does a fuzzy match, then maps
    the offset back to a PDF page using the stored page_map.
    """
    import re as _re

    page_map = get_page_map(session_id, source_doc)
    if not page_map:
        return {"found": False, "page_num": 0, "char_offset": 0}

    # We need the full document text to search in. Load from the uploads directory.
    import config as _cfg
    import os as _os
    from services.reader import read_file as _read

    target_path = None
    prefix = f"{session_id}_"
    upload_dir = _cfg.UPLOAD_PATH
    doc_basename = source_doc.replace("/", "_").replace("\\", "_")
    if _os.path.isdir(upload_dir):
        for fname in _os.listdir(upload_dir):
            if fname.startswith(prefix) and fname.endswith(doc_basename):
                target_path = _os.path.join(upload_dir, fname)
                break
        if not target_path:
            for fname in _os.listdir(upload_dir):
                if fname.endswith(doc_basename):
                    target_path = _os.path.join(upload_dir, fname)
                    break

    if not target_path or not _os.path.exists(target_path):
        return {"found": False, "page_num": 0, "char_offset": 0}

    try:
        full_text = _read(target_path)
    except Exception:
        return {"found": False, "page_num": 0, "char_offset": 0}

    # Normalize for fuzzy matching
    norm_text = _re.sub(r'\s+', ' ', full_text.lower())
    norm_quote = _re.sub(r'\s+', ' ', quote_text.strip().lower())

    idx = norm_text.find(norm_quote)
    if idx == -1 and len(norm_quote) > 40:
        idx = norm_text.find(norm_quote[:40])

    if idx == -1:
        return {"found": False, "page_num": 0, "char_offset": 0}

    # Map offset to page number
    matched_page = 1
    for entry in page_map:
        if entry["char_start"] <= idx < entry["char_end"]:
            matched_page = entry["page_num"]
            break
    else:
        if page_map:
            matched_page = page_map[-1]["page_num"]

    return {"found": True, "page_num": matched_page, "char_offset": idx}


def get_metadata(session_id: str, doc_name: str) -> dict:
    """Return the metadata dict for a document, or {} if none stored."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(f"""
                SELECT {', '.join(_METADATA_COLUMNS)}
                FROM page_metadata
                WHERE session_id = :sid AND title = :title
            """),
            {"sid": session_id, "title": doc_name},
        ).fetchone()
        if row is None:
            return {}
        return {k: v for k, v in zip(_METADATA_COLUMNS, row) if v is not None}


def get_documents_by_family(session_id: str, doc_family: str) -> list[str]:
    """Return the doc_name (title) of every document in a given family.

    Used by scope resolution (Phase 2) to answer family-scoped questions
    ("across all NDAs") by resolving the family to its concrete member documents.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT title
                FROM page_metadata
                WHERE session_id = :sid AND doc_family = :fam
            """),
            {"sid": session_id, "fam": doc_family},
        )
        return [row.title for row in rows]


def list_doc_families(session_id: str) -> list[str]:
    """Return the distinct non-null doc_family values present in a session.

    Lets scope resolution know which families actually exist before trying to
    match a question's phrasing against one.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT doc_family
                FROM page_metadata
                WHERE session_id = :sid AND doc_family IS NOT NULL
            """),
            {"sid": session_id},
        )
        return [row.doc_family for row in rows]


def lookup_clause(session_id: str, doc_hint: str, clause_num: str) -> list[dict]:
    """Resolve a clause number to its heading and wiki page(s) for one document.

    doc_hint is whatever name the scope resolver produced — it may be the full
    source_doc or a fragment of it, so match by containment either way. Returns
    [] when the document is unnumbered or the number does not exist, and the
    caller distinguishes those two cases via doc_clause_numbers().
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT heading, page_title
                FROM clause_map
                WHERE session_id = :sid AND clause_num = :num
                  AND (source_doc ILIKE '%' || :doc || '%'
                       OR :doc ILIKE '%' || source_doc || '%')
            """),
            {"sid": session_id, "num": clause_num, "doc": doc_hint},
        )
        return [{"heading": r.heading, "page_title": r.page_title} for r in rows]


def doc_clause_numbers(session_id: str, doc_hint: str) -> list[str]:
    """Every clause number the map knows for a document ([] = unnumbered source)."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT clause_num
                FROM clause_map
                WHERE session_id = :sid
                  AND (source_doc ILIKE '%' || :doc || '%'
                       OR :doc ILIKE '%' || source_doc || '%')
            """),
            {"sid": session_id, "doc": doc_hint},
        )
        nums = [r.clause_num for r in rows]
        # "1" < "10" < "2" under string sort; sort numerically by dotted parts.
        return sorted(nums, key=lambda n: [int(p) for p in n.split(".")])
