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
import re
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


def page_compaction_lock(session_id: str, title: str):
    """Per-page advisory lock guarding compaction (§ 01.6 Concurrency).

    A Postgres advisory lock rather than a Python threading.Lock because
    compaction runs under gunicorn: a thread lock only serializes within one
    worker process, and the race the doc describes is between concurrent
    ingests that may be in different processes entirely.

    Non-blocking — yields False rather than waiting. A caller that can't get
    the lock should skip the page, not queue behind it: whoever holds it is
    about to produce a fresh compaction of that same page, so waiting only to
    redo the work is the outcome worth avoiding.
    """
    from contextlib import contextmanager

    @contextmanager
    def _lock():
        from sqlalchemy import text
        # Two 32-bit keys: a fixed namespace and the page hash. Postgres
        # advisory locks are a flat global space, so the namespace keeps these
        # from colliding with the schema-init lock or anything added later.
        key = _stable_lock_key(f"{session_id}:{title}")
        conn = get_engine().connect()
        acquired = False
        try:
            acquired = bool(conn.execute(
                text("SELECT pg_try_advisory_lock(:ns, :key)"),
                {"ns": _COMPACTION_LOCK_NAMESPACE, "key": key},
            ).scalar())
            yield acquired
        finally:
            try:
                if acquired:
                    conn.execute(text("SELECT pg_advisory_unlock(:ns, :key)"),
                                 {"ns": _COMPACTION_LOCK_NAMESPACE, "key": key})
                    conn.commit()
            finally:
                conn.close()

    return _lock()


_COMPACTION_LOCK_NAMESPACE = 0x4C57  # "LW"


def _stable_lock_key(s: str) -> int:
    """Deterministic signed-32-bit key from a string.

    Not Python's hash(): that is randomized per process by PYTHONHASHSEED, so
    two workers would compute different keys for the same page and neither
    would ever block the other — a lock that silently never locks.
    """
    import hashlib
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=True)


def _crypto():
    """Lazy handle on the encryption helpers.

    Imported lazily so `db` has no import-time dependency on `crypto`, and so
    a deployment with no key configured never touches the cryptography
    package at all.
    """
    from services import crypto
    return crypto


def _question_table_name() -> str:
    """Hypothetical-question vectors, per embedding provider — same
    one-table-per-provider convention as _emb_table_name, so switching
    providers never mixes vector dimensions in one table."""
    import config
    return f"question_embeddings_{config.EMBEDDING_PROVIDER}"


def _clause_table_name() -> str:
    """Clause-level vectors (the Precedent layer), per embedding provider —
    same one-table-per-provider convention as the page and question tables."""
    import config
    return f"clause_embeddings_{config.EMBEDDING_PROVIDER}"


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
                user_id     BIGINT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chat_messages_session_idx
            ON chat_messages (session_id, created_at)
        """))

        # Who wrote this message. Nullable and unread by any query today —
        # the two-role admin/user split and per-user chat isolation are both
        # deferred (target architecture § 01.4 / § 00 Scope).
        #
        # It exists now anyway because that same section warns per-user chat
        # isolation "needs a user_id FK on conversation records from the
        # start ... not retrofitted". The retrofit is only cheap while there
        # is exactly one account, since every existing row provably belongs
        # to it. Once a second person writes messages, unattributed history
        # can't be split apart after the fact.
        try:
            conn.execute(text("""
                ALTER TABLE chat_messages
                ADD COLUMN IF NOT EXISTS user_id BIGINT
            """))
        except Exception as _user_id_err:
            logger.warning("Could not add chat_messages.user_id column (may already exist): %s", _user_id_err)
            conn.rollback()

        # Deliberately NOT a foreign key to users(id): chat_messages predates
        # the users table and holds rows from before auth existed, and this
        # app also runs with AUTH_ENABLED=false where no users row exists at
        # all. A hard FK would make message writes fail in exactly the setups
        # that don't have accounts. Add the constraint alongside the role
        # split, when every writer is guaranteed to be a real user.
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS chat_messages_user_idx
            ON chat_messages (user_id, created_at)
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

        # Single-user auth (target architecture § 01.4). One row in practice
        # this pass. `role` is inert — no code reads it — but the column costs
        # nothing now and saves a migration if the deferred admin/user split is
        # ever built; see services/auth.py for why it's here and not deferred
        # along with the rest of the role system.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id             BIGSERIAL PRIMARY KEY,
                username       TEXT NOT NULL UNIQUE,
                password_hash  TEXT NOT NULL,
                role           TEXT NOT NULL DEFAULT 'admin',
                created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_login_at  TIMESTAMPTZ
            )
        """))

        # Login attempt log — backs the rate limiter AND doubles as an audit
        # trail. Deliberately in the DB rather than in-process memory: gunicorn
        # runs multiple workers, and a per-process counter would let an attacker
        # get N attempts per worker instead of N total.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id          BIGSERIAL PRIMARY KEY,
                username    TEXT,
                ip          TEXT,
                success     BOOLEAN NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS login_attempts_username_idx
            ON login_attempts (username, created_at DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS login_attempts_ip_idx
            ON login_attempts (ip, created_at DESC)
        """))

        # Admin document lifecycle (target architecture § 01.4). One row per
        # archived document; absence of a row means active — deliberately not
        # a NOT NULL status column with a default, so "is anything archived
        # at all" is a cheap existence check rather than a full table scan.
        # See services/documents.py for the archive/delete orchestration and
        # the known limitation around pages merged from multiple documents.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_status (
                session_id   TEXT NOT NULL,
                source_doc   TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'archived',
                archived_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (session_id, source_doc)
            )
        """))

        # Review Queue, first slice (target architecture § 02). One row per
        # extracted clause, additive to the existing ingest LLM call — see
        # wiki.py's INGEST_PROMPT_TEMPLATE/DETAIL_PROMPT_TEMPLATE. `stakes`
        # is computed in Python from clause_type against a fixed high-stakes
        # set (see app.py's _HIGH_STAKES_CLAUSE_TYPES), never LLM-decided —
        # the whole bulk-accept-vs-individual-sign-off split depends on that
        # not being a number the extracting model can quietly game.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS clauses (
                id             BIGSERIAL PRIMARY KEY,
                session_id     TEXT NOT NULL,
                source_doc     TEXT NOT NULL,
                clause_type    TEXT NOT NULL,
                verbatim_text  TEXT NOT NULL,
                typed_value    JSONB,
                confidence     REAL NOT NULL,
                page_num       INT,
                char_start     INT,
                char_end       INT,
                stakes         TEXT NOT NULL DEFAULT 'low',
                review_status  TEXT NOT NULL DEFAULT 'pending',
                resolution     TEXT,
                reviewed_at    TIMESTAMPTZ,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS clauses_session_status_idx
            ON clauses (session_id, review_status)
        """))
        # Canonical clause type (§ Phase 3.5c) — added BESIDE clause_type,
        # which keeps the raw model-chosen label. NULL means "not mapped",
        # which is a real answer here, not a missing value: see
        # services/clause_vocab.py on why a nearest guess is worse.
        conn.execute(text("""
            ALTER TABLE clauses ADD COLUMN IF NOT EXISTS clause_type_canon TEXT
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS clauses_canon_idx
            ON clauses (wiki_id, session_id, clause_type_canon)
            WHERE clause_type_canon IS NOT NULL
        """))

        _init_regression_schema(conn, text)
        _init_backbone_schema(conn, text)

        conn.commit()


# ---------------------------------------------------------------------------
# Accuracy regression suite (target architecture § Phase 3.5a)
# ---------------------------------------------------------------------------

def _init_regression_schema(conn, text) -> None:
    """Stored regression cases, runs and per-case results.

    Three tiers, deliberately separated by what they cost to run:

      scope    — asserts resolve_scope()'s decision only. No pipeline, no
                 LLM, no embedding call. Free, so it can run on every commit,
                 and it covers the failure class this corpus actually keeps
                 hitting (which documents a question resolves to).
      pipeline — runs the real /query pipeline and asserts structural facts
                 about the answer: did it abstain, did it cite, which
                 documents did it read. Costs one query per case.
      graded   — pipeline plus an LLM judge scoring the answer text against
                 a stored expected answer. Costs the query plus the judge.

    A case carries expectations for every tier it participates in; a run
    names the tier it executed, so a cheap scope run and an expensive graded
    run over the same cases stay comparable but never get confused for one
    another.
    """
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS regression_cases (
            id                  BIGSERIAL PRIMARY KEY,
            wiki_id             TEXT NOT NULL,
            session_id          TEXT NOT NULL,
            name                TEXT NOT NULL,
            question            TEXT NOT NULL,
            archetype           TEXT,
            -- tier: scope
            expect_scope_method TEXT,
            expect_docs         JSONB,
            -- tier: pipeline
            expect_abstain      BOOLEAN NOT NULL DEFAULT FALSE,
            must_contain        JSONB,
            must_not_contain    JSONB,
            -- tier: graded
            expect_answer       TEXT,
            notes               TEXT,
            active              BOOLEAN NOT NULL DEFAULT TRUE,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, name)
        )
    """))
    # Names the mechanism that reaches this case's documents when it is NOT
    # resolve_scope - the counting path reads the document index, the
    # Calculation Agent falls back to an identifier lookup. Set, it tells the
    # scope tier there is nothing here for it to assert; the pipeline tier
    # still checks the documents from files_used.
    conn.execute(text("ALTER TABLE regression_cases "
                      "ADD COLUMN IF NOT EXISTS scope_resolved_by TEXT"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS regression_runs (
            id            BIGSERIAL PRIMARY KEY,
            wiki_id       TEXT NOT NULL,
            session_id    TEXT NOT NULL,
            tier          TEXT NOT NULL,
            label         TEXT,
            git_sha       TEXT,
            status        TEXT NOT NULL DEFAULT 'running',
            cases_total   INT NOT NULL DEFAULT 0,
            cases_passed  INT NOT NULL DEFAULT 0,
            cases_failed  INT NOT NULL DEFAULT 0,
            error         TEXT,
            started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at   TIMESTAMPTZ
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS regression_results (
            id                  BIGSERIAL PRIMARY KEY,
            run_id              BIGINT NOT NULL REFERENCES regression_runs(id) ON DELETE CASCADE,
            case_id             BIGINT,
            case_name           TEXT NOT NULL,
            passed              BOOLEAN NOT NULL,
            failures            JSONB,
            actual_scope_method TEXT,
            actual_docs         JSONB,
            answer              TEXT,
            scores              JSONB,
            total_ms            INT,
            trace_id            BIGINT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS regression_results_run_idx
        ON regression_results (run_id)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS regression_runs_scope_idx
        ON regression_runs (wiki_id, session_id, tier, started_at DESC)
    """))
    _init_page_quality_schema(conn, text)


def _init_page_quality_schema(conn, text) -> None:
    """Per-page extraction provenance (§ Phase 3.5b).

    Records only what the reader can actually observe: which engine ran, how
    the page's text was obtained, and how much came back. Deliberately does
    NOT carry a fabricated OCR confidence score — Tesseract can report one via
    image_to_data, but the OCR path here tries several PSM modes behind a retry
    wrapper and keeps the longest result, so there is no single confidence
    figure that honestly describes the output. A column that would be filled
    with a plausible-looking invented number is worse than no column: the whole
    point of this table is telling a reader when not to trust a document.

    `char_count` after extraction, against the same MIN_CHARS_PER_PAGE floor
    the reader uses to decide a page needs OCR, is the signal that actually
    matters — a page still under that floor after OCR ran is a page nobody
    can answer questions from.
    """
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS page_quality (
            id                BIGSERIAL PRIMARY KEY,
            wiki_id           TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            source_doc        TEXT NOT NULL,
            page_num          INT  NOT NULL,
            extraction_method TEXT NOT NULL,
            ocr_engine        TEXT,
            char_count        INT  NOT NULL DEFAULT 0,
            needed_ocr        BOOLEAN NOT NULL DEFAULT FALSE,
            below_floor       BOOLEAN NOT NULL DEFAULT FALSE,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, source_doc, page_num)
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS page_quality_doc_idx
        ON page_quality (wiki_id, session_id, source_doc)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS page_quality_problem_idx
        ON page_quality (wiki_id, session_id) WHERE below_floor
    """))


# ---------------------------------------------------------------------------
# Phase 0 Backbone — wikis + typed per-family tables (target architecture § 03)
# ---------------------------------------------------------------------------

# The wiki every pre-backbone row belongs to. Existing corpora were ingested
# before `wikis` existed, so they can't be attributed to a wiki the admin
# actually chose — they all belong to the one implicit corpus that was there
# all along. Fixed UUID rather than a lookup so backfill is deterministic and
# re-runnable.
DEFAULT_WIKI_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_WIKI_NAME = "Default wiki"

# Tables that predate the backbone and are getting `wiki_id` added rather than
# created with it. Listed explicitly (not discovered) so a table added later
# without a wiki_id is a visible omission here, not a silent isolation hole.
_LEGACY_WIKI_SCOPED_TABLES = (
    "pages",
    "relations",
    "page_metadata",
    "clauses",
    "contradictions",
    "clause_map",
    "document_status",
    "source_positions",
)


def _init_backbone_schema(conn, text) -> None:
    """Phase 0 Backbone tables. Additive only — nothing existing is dropped
    or restructured, per the architecture doc's "What the database gains".

    Every table here carries `wiki_id` at creation, never as a retrofit. The
    legacy tables above get it added + backfilled instead, which is the one
    retrofit the doc explicitly accepts (they existed before the primitive did).
    """
    # --- wikis: the isolation primitive itself -----------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS wikis (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'active',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_by  TEXT
        )
    """))
    conn.execute(text("""
        INSERT INTO wikis (id, name, status, created_by)
        VALUES (:id, :name, 'active', 'system')
        ON CONFLICT (id) DO NOTHING
    """), {"id": DEFAULT_WIKI_ID, "name": DEFAULT_WIKI_NAME})

    # System-level active-wiki pointer (§ Wikis — "switch-based, not
    # simultaneous"). A settings row rather than a wikis.is_active flag:
    # exactly one pointer can exist by construction, so two rows can never
    # both claim active.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        INSERT INTO app_settings (key, value)
        VALUES ('active_wiki_id', :id)
        ON CONFLICT (key) DO NOTHING
    """), {"id": DEFAULT_WIKI_ID})

    # --- documents: one row per document, any family -----------------------
    # Generalized from the original `contracts` design. `schema_version` is
    # what re-ingest's transactional swap keys off (§ 01.4) — it exists from
    # the first row so re-ingest never has to guess at un-stamped history.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS documents (
            id                BIGSERIAL PRIMARY KEY,
            wiki_id           TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            source_doc        TEXT NOT NULL,
            doc_family        TEXT,
            doc_type          TEXT,
            jurisdiction      TEXT,
            parties           JSONB,
            effective_date    TEXT,
            expiry_date       TEXT,
            status            TEXT,
            lifecycle         TEXT NOT NULL DEFAULT 'current',
            role              TEXT,
            binding_status    TEXT,
            family_confidence REAL,
            family_method     TEXT,
            folder_hint       TEXT,
            schema_version    INT NOT NULL DEFAULT 1,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, source_doc)
        )
    """))
    # content_hash — added after the table existed, same migration guard as
    # every other late column here. Not part of the UNIQUE constraint: a
    # constraint would reject a legitimate re-ingest at the DB layer with an
    # opaque error, where the application-level check in wiki.ingest() can
    # instead skip cleanly and say which existing document it matched.
    try:
        conn.execute(text("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS content_hash TEXT
        """))
    except Exception as _hash_err:
        logger.warning("Could not add content_hash column (may already exist): %s", _hash_err)
        conn.rollback()
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS documents_wiki_hash_idx
        ON documents (wiki_id, content_hash)
        WHERE content_hash IS NOT NULL
    """))
    # file_hash — SHA-256 of raw upload bytes, computed before any text
    # extraction/OCR runs. This is the upload-time dedup signal: cheap enough
    # to check on every file before it costs anything. content_hash (above)
    # is a secondary, post-extraction check for same-text-different-bytes.
    try:
        conn.execute(text("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS file_hash TEXT
        """))
    except Exception as _fhash_err:
        logger.warning("Could not add file_hash column (may already exist): %s", _fhash_err)
        conn.rollback()
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS documents_wiki_file_hash_idx
        ON documents (wiki_id, file_hash)
        WHERE file_hash IS NOT NULL
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS documents_wiki_family_idx
        ON documents (wiki_id, doc_family)
    """))

    # --- contracts: Family 1 (+2) --------------------------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS contracts (
            id              BIGSERIAL PRIMARY KEY,
            wiki_id         TEXT NOT NULL,
            document_id     BIGINT,
            session_id      TEXT NOT NULL,
            source_doc      TEXT NOT NULL,
            governing_law   TEXT,
            liability_cap   TEXT,
            term_length     TEXT,
            renewal_terms   TEXT,
            termination     TEXT,
            binding_status  TEXT,
            exclusivity     TEXT,
            typed_value     JSONB,
            confidence      REAL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, source_doc)
        )
    """))

    # --- obligations: Family 1 (+2) ------------------------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS obligations (
            id               BIGSERIAL PRIMARY KEY,
            wiki_id          TEXT NOT NULL,
            document_id      BIGINT,
            session_id       TEXT NOT NULL,
            source_doc       TEXT NOT NULL,
            obligated_party  TEXT,
            entity_id        BIGINT,
            duty             TEXT,
            trigger          TEXT,
            deadline         TEXT,
            notice_period    TEXT,
            consequence      TEXT,
            source_clause_id BIGINT,
            verbatim_text    TEXT,
            page_num         INT,
            confidence       REAL,
            review_status    TEXT NOT NULL DEFAULT 'pending',
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS obligations_wiki_doc_idx
        ON obligations (wiki_id, session_id, source_doc)
    """))

    # --- litigation_facts: Family 3 ------------------------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS litigation_facts (
            id                  BIGSERIAL PRIMARY KEY,
            wiki_id             TEXT NOT NULL,
            document_id         BIGINT,
            session_id          TEXT NOT NULL,
            source_doc          TEXT NOT NULL,
            court               TEXT,
            case_number         TEXT,
            plaintiffs          JSONB,
            defendants          JSONB,
            procedural_posture  TEXT,
            holding             TEXT,
            relief_granted      TEXT,
            disposition         TEXT,
            decided_date        TEXT,
            typed_value         JSONB,
            confidence          REAL,
            review_status       TEXT NOT NULL DEFAULT 'pending',
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, source_doc)
        )
    """))

    # --- authorizations: Family 4 --------------------------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS authorizations (
            id                  BIGSERIAL PRIMARY KEY,
            wiki_id             TEXT NOT NULL,
            document_id         BIGINT,
            session_id          TEXT NOT NULL,
            source_doc          TEXT NOT NULL,
            grantor             TEXT,
            grantee             TEXT,
            scope_of_authority  TEXT,
            limitations         TEXT,
            resolving_body      TEXT,
            effective_date      TEXT,
            expiry_date         TEXT,
            typed_value         JSONB,
            confidence          REAL,
            review_status       TEXT NOT NULL DEFAULT 'pending',
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, source_doc)
        )
    """))

    # --- opinions: Family 5 --------------------------------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS opinions (
            id                   BIGSERIAL PRIMARY KEY,
            wiki_id              TEXT NOT NULL,
            document_id          BIGINT,
            session_id           TEXT NOT NULL,
            source_doc           TEXT NOT NULL,
            addressee            TEXT,
            matters_opined       JSONB,
            assumptions          JSONB,
            qualifications       JSONB,
            conclusion           TEXT,
            reliance_limitation  TEXT,
            opinion_date         TEXT,
            typed_value          JSONB,
            confidence           REAL,
            review_status        TEXT NOT NULL DEFAULT 'pending',
            created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, source_doc)
        )
    """))

    # --- citations: all families --------------------------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS citations (
            id              BIGSERIAL PRIMARY KEY,
            wiki_id         TEXT NOT NULL,
            document_id     BIGINT,
            session_id      TEXT NOT NULL,
            source_doc      TEXT NOT NULL,
            citation_text   TEXT NOT NULL,
            authority_type  TEXT,
            normalized_form TEXT,
            page_title      TEXT,
            anchor_id       BIGINT,
            clause_id       BIGINT,
            page_num        INT,
            confidence      REAL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS citations_wiki_doc_idx
        ON citations (wiki_id, session_id, source_doc)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS citations_normalized_idx
        ON citations (wiki_id, normalized_form)
    """))

    # --- structural_anchors: section/¶ numbering -----------------------------
    # Also the input to structure-aware segmentation (§ 01 Segmentation) — the
    # anchor parse runs before the segment decision, so these rows exist for
    # the same document the segments were cut from.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS structural_anchors (
            id            BIGSERIAL PRIMARY KEY,
            wiki_id       TEXT NOT NULL,
            document_id   BIGINT,
            session_id    TEXT NOT NULL,
            source_doc    TEXT NOT NULL,
            anchor_label  TEXT NOT NULL,
            anchor_kind   TEXT,
            heading_text  TEXT,
            char_start    INT,
            char_end      INT,
            page_num      INT,
            page_title    TEXT,
            ordinal       INT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS structural_anchors_doc_idx
        ON structural_anchors (wiki_id, session_id, source_doc, ordinal)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS structural_anchors_label_idx
        ON structural_anchors (wiki_id, session_id, anchor_label)
    """))

    # --- entities / entity_aliases: canonical party registry -----------------
    # The UNIQUE on (wiki_id, canonical_key) is the hardening item, not an
    # afterthought: canonicalization that races two spellings of "Acme Corp"
    # into two rows defeats the whole point, so the constraint carries it and
    # writes go through an upsert (see upsert_entity).
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS entities (
            id             BIGSERIAL PRIMARY KEY,
            wiki_id        TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            canonical_key  TEXT NOT NULL,
            entity_type    TEXT,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, canonical_key)
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS entity_aliases (
            id         BIGSERIAL PRIMARY KEY,
            wiki_id    TEXT NOT NULL,
            entity_id  BIGINT NOT NULL,
            alias      TEXT NOT NULL,
            alias_key  TEXT NOT NULL,
            source_doc TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, alias_key)
        )
    """))

    # --- tables / figures: § 01.1 -------------------------------------------
    # `tables` and `figures` are the doc's own names and both are unreserved
    # keywords in Postgres, so they're used verbatim rather than renamed —
    # keeping code and architecture doc reading the same.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS tables (
            id           BIGSERIAL PRIMARY KEY,
            wiki_id      TEXT NOT NULL,
            document_id  BIGINT,
            session_id   TEXT NOT NULL,
            source_doc   TEXT NOT NULL,
            page_num     INT,
            page_title   TEXT,
            caption      TEXT,
            columns      JSONB,
            rows         JSONB,
            confidence   REAL,
            extraction_method TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending',
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS tables_wiki_doc_idx
        ON tables (wiki_id, session_id, source_doc)
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS figures (
            id           BIGSERIAL PRIMARY KEY,
            wiki_id      TEXT NOT NULL,
            document_id  BIGINT,
            session_id   TEXT NOT NULL,
            source_doc   TEXT NOT NULL,
            page_num     INT,
            page_title   TEXT,
            figure_kind  TEXT,
            description  TEXT,
            image_ref    TEXT,
            confidence   REAL,
            extraction_method TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending',
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS figures_wiki_doc_idx
        ON figures (wiki_id, session_id, source_doc)
    """))

    # --- document_relations: document-to-document edges ----------------------
    # Its own table rather than rows in `relations`, whose from_title/to_title
    # are *page* titles. Putting document names in that column space would
    # corrupt the page graph the knowledge-graph view and cross-reference pass
    # both walk. The vocabulary is the doc's own: amends, superseded-by,
    # ancillary-to, references, references-unresolved.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS document_relations (
            id             BIGSERIAL PRIMARY KEY,
            wiki_id        TEXT NOT NULL,
            session_id     TEXT NOT NULL,
            from_doc       TEXT NOT NULL,
            to_doc         TEXT,
            to_doc_raw     TEXT NOT NULL,
            label          TEXT NOT NULL,
            resolved       BOOLEAN NOT NULL DEFAULT FALSE,
            match_score    REAL,
            confidence     REAL,
            evidence_text  TEXT,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, from_doc, to_doc_raw, label)
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS document_relations_from_idx
        ON document_relations (wiki_id, session_id, from_doc)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS document_relations_label_idx
        ON document_relations (session_id, label)
    """))

    # --- review_queue: the non-clause flag kinds -----------------------------
    # The shipped clause queue (the `clauses` table) stays exactly as it is —
    # it works, it's tested, and its stakes split is already correct. This
    # table carries the kinds the backbone adds: doc-type misclassification,
    # per-family metadata fields, and low-confidence table/figure extraction.
    # get_review_queue() unions the two so the admin panel keeps showing one
    # queue; two separate queues would be a worse answer than one join.
    # Reader verdicts on answers. Every other score in this system measures
    # whether an answer is faithful to the text it cites; none can see whether
    # it was USEFUL, because that is not a property of the documents. This is
    # the only ground truth for it, and the only way the six-score confidence
    # can ever be calibrated rather than trusted.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS answer_feedback (
            id                   BIGSERIAL PRIMARY KEY,
            wiki_id              TEXT NOT NULL,
            session_id           TEXT NOT NULL,
            message_id           BIGINT,
            question             TEXT NOT NULL,
            answer_excerpt       TEXT,
            verdict              TEXT NOT NULL,
            note                 TEXT,
            scope_method         TEXT,
            confidence_value     INT,
            confidence_governing TEXT,
            files_used           JSONB,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_answer_feedback_wiki
            ON answer_feedback (wiki_id, id DESC)
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS review_queue (
            id             BIGSERIAL PRIMARY KEY,
            wiki_id        TEXT NOT NULL,
            session_id     TEXT NOT NULL,
            source_doc     TEXT NOT NULL,
            item_kind      TEXT NOT NULL,
            item_label     TEXT NOT NULL,
            item_value     TEXT,
            typed_value    JSONB,
            confidence     REAL NOT NULL DEFAULT 0.0,
            stakes         TEXT NOT NULL DEFAULT 'low',
            reason         TEXT,
            page_num       INT,
            review_status  TEXT NOT NULL DEFAULT 'pending',
            resolution     TEXT,
            reviewed_at    TIMESTAMPTZ,
            superseded_at  TIMESTAMPTZ,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS review_queue_session_status_idx
        ON review_queue (session_id, review_status)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS review_queue_doc_idx
        ON review_queue (wiki_id, session_id, source_doc)
    """))

    # --- collections: a named set of documents (Phase 2) --------------------
    # Shared plumbing under Playbooks and the Deviation Dashboard: both need
    # "run this over these documents" and neither should invent its own idea
    # of what a document set is.
    #
    # Wiki-scoped, and the UNIQUE is on (wiki_id, name) rather than name alone:
    # two wikis are separate corpora and each may reasonably have its own
    # "Active NDAs", the same way playbooks are wiki-scoped rather than drawn
    # from a shared default set.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS collections (
            id           BIGSERIAL PRIMARY KEY,
            wiki_id      TEXT NOT NULL,
            session_id   TEXT NOT NULL,
            name         TEXT NOT NULL,
            description  TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, name)
        )
    """))
    # Membership is stored explicitly rather than as a saved filter. A filter
    # re-evaluates on every read, so a playbook run and the dashboard row that
    # records it could silently cover different documents; an explicit list is
    # what makes a recorded run reproducible. Filters are offered as a way to
    # POPULATE the list, not as the list itself.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS collection_documents (
            id            BIGSERIAL PRIMARY KEY,
            collection_id BIGINT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            wiki_id       TEXT NOT NULL,
            session_id    TEXT NOT NULL,
            source_doc    TEXT NOT NULL,
            added_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (collection_id, source_doc)
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS collections_wiki_idx ON collections (wiki_id)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS collection_documents_cid_idx
        ON collection_documents (collection_id)
    """))

    # --- playbooks: house positions per clause type (Phase 2) ---------------
    # Wiki-scoped by design (§ Access & Admin Lifecycle): a firm's own house
    # rules and a client's differing playbook must not cross wikis, so there is
    # no shared default set.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS playbooks (
            id           BIGSERIAL PRIMARY KEY,
            wiki_id      TEXT NOT NULL,
            session_id   TEXT NOT NULL,
            name         TEXT NOT NULL,
            description  TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, name)
        )
    """))
    # One rule per clause type. The three positions are the vocabulary the doc
    # specifies — standard / fallback / unacceptable — and a rule is useless
    # without at least the standard one, which the service enforces.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS playbook_rules (
            id            BIGSERIAL PRIMARY KEY,
            playbook_id   BIGINT NOT NULL REFERENCES playbooks(id) ON DELETE CASCADE,
            wiki_id       TEXT NOT NULL,
            clause_type   TEXT NOT NULL,
            standard      TEXT NOT NULL,
            fallback      TEXT,
            unacceptable  TEXT,
            guidance      TEXT,
            severity      TEXT NOT NULL DEFAULT 'medium',
            ordinal       INT  NOT NULL DEFAULT 0,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (playbook_id, clause_type)
        )
    """))
    # A run records WHICH documents it covered, not just the collection id:
    # collection membership can change afterwards, and a run whose scope can
    # drift is not reproducible evidence. documents_covered is the frozen list.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS playbook_runs (
            id                 BIGSERIAL PRIMARY KEY,
            wiki_id            TEXT NOT NULL,
            session_id         TEXT NOT NULL,
            playbook_id        BIGINT NOT NULL REFERENCES playbooks(id) ON DELETE CASCADE,
            collection_id      BIGINT,
            collection_name    TEXT,
            status             TEXT NOT NULL DEFAULT 'running',
            documents_total    INT  NOT NULL DEFAULT 0,
            documents_done     INT  NOT NULL DEFAULT 0,
            findings_total     INT  NOT NULL DEFAULT 0,
            documents_covered  JSONB,
            error              TEXT,
            started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at        TIMESTAMPTZ
        )
    """))
    # One row per (document, clause type) assessed. verdict is the deviation
    # signal the Phase 3 dashboard aggregates; `missing` is a real verdict, not
    # an absence of one — a contract with no liability cap at all is a finding.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS playbook_findings (
            id            BIGSERIAL PRIMARY KEY,
            run_id        BIGINT NOT NULL REFERENCES playbook_runs(id) ON DELETE CASCADE,
            wiki_id       TEXT NOT NULL,
            source_doc    TEXT NOT NULL,
            clause_type   TEXT NOT NULL,
            clause_id     BIGINT,
            verdict       TEXT NOT NULL,
            severity      TEXT,
            rationale     TEXT,
            redline       TEXT,
            clause_text   TEXT,
            grounded      BOOLEAN,
            confidence    REAL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))
    for stmt in (
        "CREATE INDEX IF NOT EXISTS playbooks_wiki_idx ON playbooks (wiki_id)",
        "CREATE INDEX IF NOT EXISTS playbook_rules_pid_idx ON playbook_rules (playbook_id)",
        "CREATE INDEX IF NOT EXISTS playbook_runs_wiki_idx ON playbook_runs (wiki_id, playbook_id)",
        "CREATE INDEX IF NOT EXISTS playbook_findings_run_idx ON playbook_findings (run_id)",
        "CREATE INDEX IF NOT EXISTS playbook_findings_verdict_idx "
        "ON playbook_findings (wiki_id, verdict, clause_type)",
    ):
        conn.execute(text(stmt))

    # --- prompt library (Phase 3) --------------------------------------------
    # Reusable, wiki-scoped prompt templates — "Also fixing: Prompt library".
    # Distinct from services/rules.py's House Rules: those are global answer-
    # style instructions appended to every prompt automatically. This is a
    # library a person picks FROM for one drafting/query request, so it is
    # wiki-scoped like everything else in this phase rather than a single
    # global file, and stores {{placeholder}} bodies rather than fixed text.
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS prompt_templates (
            id          BIGSERIAL PRIMARY KEY,
            wiki_id     TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            name        TEXT NOT NULL,
            category    TEXT,
            body        TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, name)
        )
    """))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS prompt_templates_wiki_idx ON prompt_templates (wiki_id)"
    ))

    # --- hypothetical-question embeddings (§ 01 stage 06) --------------------
    # The third embedding type, alongside page-level and clause-level. Its own
    # table rather than extra rows in the page table: a question and a page
    # are different things, and mixing them would make every existing
    # page-similarity query silently start returning questions.
    #
    # Vector dimension follows the same per-provider convention as the page
    # tables (see _emb_table_name) so a provider switch never mixes dimensions.
    try:
        import config as _cfg
        _emb_dims = _cfg.get_embedding_dimensions()
        _q_tbl = _question_table_name()
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_q_tbl} (
                id          BIGSERIAL PRIMARY KEY,
                wiki_id     TEXT NOT NULL DEFAULT '{DEFAULT_WIKI_ID}',
                session_id  TEXT NOT NULL,
                title       TEXT NOT NULL,
                question    TEXT NOT NULL,
                embedding   vector({_emb_dims}),
                doc_family  TEXT,
                source_doc  TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (session_id, title, question)
            )
        """))
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS {_q_tbl}_session_idx "
            f"ON {_q_tbl} (session_id, title)"
        ))
    except Exception as _q_err:
        logger.warning("Could not create question-embedding table: %s", _q_err)
        conn.rollback()

    # --- clause embeddings: the Precedent layer (Phase 2) --------------------
    # The embedding type Draft Mode reads from. Drafting needs the CLAUSE that
    # solves a problem, not the page that mentions it: page vectors average a
    # whole topic, so "limitation of liability capped at fees" ranks a page
    # discussing liability generally above the clause that actually says it.
    #
    # Carries doc_family and role so retrieval can scope to role-tagged
    # precedent documents without a join back to `documents` on every search,
    # and keyed on clause_id so a re-ingest that replaces clauses replaces
    # their vectors with them.
    try:
        import config as _cfg2
        _emb_dims2 = _cfg2.get_embedding_dimensions()
        _c_tbl = _clause_table_name()
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_c_tbl} (
                id           BIGSERIAL PRIMARY KEY,
                wiki_id      TEXT NOT NULL DEFAULT '{DEFAULT_WIKI_ID}',
                session_id   TEXT NOT NULL,
                clause_id    BIGINT NOT NULL,
                source_doc   TEXT NOT NULL,
                clause_type  TEXT,
                doc_family   TEXT,
                role         TEXT,
                embedding    vector({_emb_dims2}),
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (clause_id)
            )
        """))
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS {_c_tbl}_scope_idx "
            f"ON {_c_tbl} (wiki_id, session_id, role)"
        ))
    except Exception as _c_err:
        logger.warning("Could not create clause-embedding table: %s", _c_err)
        conn.rollback()

    # --- wiki_id on the legacy tables ---------------------------------------
    # DEFAULT is set so rows written by not-yet-threaded code paths still land
    # in the default wiki rather than NULL (a NULL wiki_id would silently fall
    # out of every wiki-scoped predicate — invisible data loss, not an error).
    for _tbl in _LEGACY_WIKI_SCOPED_TABLES:
        try:
            conn.execute(text(
                f"ALTER TABLE {_tbl} ADD COLUMN IF NOT EXISTS wiki_id TEXT "
                f"NOT NULL DEFAULT '{DEFAULT_WIKI_ID}'"
            ))
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {_tbl}_wiki_idx ON {_tbl} (wiki_id)"
            ))
        except Exception as _wiki_col_err:
            logger.warning(
                "Could not add wiki_id to %s (table may not exist yet): %s",
                _tbl, _wiki_col_err,
            )
            conn.rollback()

    # Embedding tables are per-provider and discovered, not listed — a provider
    # switch creates a new one, and it needs the column too.
    try:
        for _emb_tbl in _page_embedding_tables(conn):
            conn.execute(text(
                f"ALTER TABLE {_emb_tbl} ADD COLUMN IF NOT EXISTS wiki_id TEXT "
                f"NOT NULL DEFAULT '{DEFAULT_WIKI_ID}'"
            ))
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {_emb_tbl}_wiki_idx "
                f"ON {_emb_tbl} (wiki_id)"
            ))
    except Exception as _emb_wiki_err:
        logger.warning("Could not add wiki_id to embedding tables: %s", _emb_wiki_err)
        conn.rollback()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def get_pages(wiki_id: str, session_id: str, include_archived: bool = False) -> dict[str, dict]:
    """Return all pages for a session as {title: {content, summary, source_doc, ...}}.

    include_archived=False (the default, used everywhere retrieval/browsing
    happens) excludes pages whose source_doc is archived — this is THE
    enforcement point for "an archived document drops out of search/chat":
    every caller (wiki index load, hybrid retrieval, /wiki/graph, /wiki/pages)
    funnels through here, so filtering once here covers all of them rather
    than needing the same check re-added at every call site.

    wiki_id is a mandatory predicate, not a display filter (see services/
    wikis.py) — a page written under a different wiki must never surface
    here even if it shares a session_id.
    """
    from sqlalchemy import text
    engine = get_engine()
    archived_clause = "" if include_archived else """
                AND NOT EXISTS (
                    SELECT 1 FROM document_status ds
                    WHERE ds.wiki_id = pages.wiki_id
                      AND ds.session_id = pages.session_id
                      AND ds.source_doc = pages.source_doc
                      AND ds.status = 'archived'
                )"""
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT title, content, summary, source_doc, contradiction_flagged, variants
                FROM pages
                WHERE wiki_id = :w AND session_id = :sid{archived_clause}
            """),
            {"w": wiki_id, "sid": session_id},
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


def get_page_list(wiki_id: str, session_id: str, include_archived: bool = False) -> list[dict]:
    """Lightweight admin listing: title, source_doc, char_count,
    contradiction_flagged, last_modified — for the Wiki page browser.

    Separate from get_pages() (the retrieval-critical path every query
    funnels through) so this admin-only listing can carry extra columns
    without touching that path.
    """
    from sqlalchemy import text
    engine = get_engine()
    archived_clause = "" if include_archived else """
                AND NOT EXISTS (
                    SELECT 1 FROM document_status ds
                    WHERE ds.wiki_id = pages.wiki_id
                      AND ds.session_id = pages.session_id
                      AND ds.source_doc = pages.source_doc
                      AND ds.status = 'archived'
                )"""
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT title, source_doc, char_count, contradiction_flagged, last_modified
                FROM pages
                WHERE wiki_id = :w AND session_id = :sid{archived_clause}
                ORDER BY title
            """),
            {"w": wiki_id, "sid": session_id},
        )
        return [
            {
                "title": r.title,
                "source_doc": r.source_doc,
                "char_count": r.char_count,
                "contradiction_flagged": r.contradiction_flagged,
                "last_modified": r.last_modified.isoformat() if r.last_modified else None,
            }
            for r in rows
        ]


def _page_embedding_tables(conn) -> list[str]:
    """Every page_embeddings* table across every embedding provider ever
    used — see _emb_table_name(): switching EMBEDDING_PROVIDER doesn't
    merge or drop the previous provider's table, so a page mutation that
    only touched the currently-active table would leave a stale row under
    an inactive provider. Shared by rename_page/delete_page/merge_pages,
    same lookup delete_document_data already uses.
    """
    from sqlalchemy import text
    return [r[0] for r in conn.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name LIKE 'page_embeddings%'"
    ))]


def rename_page(wiki_id: str, session_id: str, old_title: str, new_title: str) -> bool:
    """Rename a wiki page and every table that references it by title.

    Returns False if old_title doesn't exist, or new_title is already taken
    — both checked inside the same transaction as the writes to avoid a
    check-then-write race. Embedding vectors are updated in place (title
    column only) — content didn't change, only its label, so no re-embed
    is needed.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pages WHERE wiki_id = :w AND session_id = :sid AND title = :t"),
            {"w": wiki_id, "sid": session_id, "t": old_title},
        ).first()
        if not exists:
            return False
        clash = conn.execute(
            text("SELECT 1 FROM pages WHERE wiki_id = :w AND session_id = :sid AND title = :t"),
            {"w": wiki_id, "sid": session_id, "t": new_title},
        ).first()
        if clash:
            return False

        params = {"w": wiki_id, "sid": session_id, "new": new_title, "old": old_title}
        conn.execute(text("UPDATE pages SET title = :new WHERE wiki_id = :w AND session_id = :sid AND title = :old"), params)
        conn.execute(text("UPDATE relations SET from_title = :new WHERE wiki_id = :w AND session_id = :sid AND from_title = :old"), params)
        conn.execute(text("UPDATE relations SET to_title = :new WHERE wiki_id = :w AND session_id = :sid AND to_title = :old"), params)
        conn.execute(text("UPDATE clause_map SET page_title = :new WHERE wiki_id = :w AND session_id = :sid AND page_title = :old"), params)
        conn.execute(text("UPDATE contradictions SET page_title = :new WHERE wiki_id = :w AND session_id = :sid AND page_title = :old"), params)

        for emb_table in _page_embedding_tables(conn):
            conn.execute(
                text(f'UPDATE "{emb_table}" SET title = :new WHERE wiki_id = :w AND session_id = :sid AND title = :old'),
                params,
            )

        conn.commit()
    return True


def delete_page(wiki_id: str, session_id: str, title: str) -> bool:
    """Delete one wiki page and every row across the schema that
    references it by title. Returns False if the page doesn't exist.

    Closes a real gap delete_document_data has today: that whole-document
    delete only clears clause_map/source_positions by source_doc, never
    contradictions or clause_map by page title (see its own docstring) —
    not retrofitted there, since that path is already tested at the
    document-cascade granularity; this is a new, separate, page-scoped
    delete for the Wiki admin section.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pages WHERE wiki_id = :w AND session_id = :sid AND title = :t"),
            {"w": wiki_id, "sid": session_id, "t": title},
        ).first()
        if not exists:
            return False

        params = {"w": wiki_id, "sid": session_id, "t": title}
        conn.execute(text("DELETE FROM pages WHERE wiki_id = :w AND session_id = :sid AND title = :t"), params)
        conn.execute(text("DELETE FROM relations WHERE wiki_id = :w AND session_id = :sid AND (from_title = :t OR to_title = :t)"), params)
        conn.execute(text("DELETE FROM clause_map WHERE wiki_id = :w AND session_id = :sid AND page_title = :t"), params)
        conn.execute(text("DELETE FROM contradictions WHERE wiki_id = :w AND session_id = :sid AND page_title = :t"), params)

        for emb_table in _page_embedding_tables(conn):
            conn.execute(
                text(f'DELETE FROM "{emb_table}" WHERE wiki_id = :w AND session_id = :sid AND title = :t'),
                params,
            )

        conn.commit()
    return True


def merge_pages(wiki_id: str, session_id: str, source_title: str, target_title: str) -> bool:
    """Absorb source_title's content into target_title, then remove
    source_title. Returns False if either page doesn't exist.

    Relations pointing at source_title are re-pointed to target_title (the
    concept they describe still exists post-merge) rather than deleted,
    then de-duplicated in case the re-point created an exact duplicate edge
    or a self-loop. Target's embedding row is deleted, not re-embedded —
    its content just changed, so the vector is now stale; the existing
    /wiki/backfill_embeddings pass picks it up on its next run like any
    other missing embedding. No LLM/embedding call fires as part of merge.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        src = conn.execute(
            text("SELECT content, summary FROM pages WHERE wiki_id = :w AND session_id = :sid AND title = :t"),
            {"w": wiki_id, "sid": session_id, "t": source_title},
        ).first()
        tgt = conn.execute(
            text("SELECT content, summary FROM pages WHERE wiki_id = :w AND session_id = :sid AND title = :t"),
            {"w": wiki_id, "sid": session_id, "t": target_title},
        ).first()
        if not src or not tgt:
            return False

        merged_content = f"{tgt.content}\n\n{src.content}"
        merged_summary = tgt.summary or src.summary

        conn.execute(
            text("""UPDATE pages SET content = :c, summary = :s, char_count = :cc, last_modified = now()
                     WHERE wiki_id = :w AND session_id = :sid AND title = :t"""),
            {"c": merged_content, "s": merged_summary, "cc": len(merged_content),
             "w": wiki_id, "sid": session_id, "t": target_title},
        )
        conn.execute(
            text("DELETE FROM pages WHERE wiki_id = :w AND session_id = :sid AND title = :t"),
            {"w": wiki_id, "sid": session_id, "t": source_title},
        )

        repoint = {"w": wiki_id, "sid": session_id, "tgt": target_title, "src": source_title}
        conn.execute(text("UPDATE relations SET from_title = :tgt WHERE wiki_id = :w AND session_id = :sid AND from_title = :src"), repoint)
        conn.execute(text("UPDATE relations SET to_title = :tgt WHERE wiki_id = :w AND session_id = :sid AND to_title = :src"), repoint)
        # A re-point can create a self-loop (if source and target were
        # already directly connected) or an exact duplicate of an edge that
        # already existed under target_title — both are noise now, drop them.
        conn.execute(
            text("DELETE FROM relations WHERE wiki_id = :w AND session_id = :sid AND from_title = :tgt AND to_title = :tgt"),
            {"w": wiki_id, "sid": session_id, "tgt": target_title},
        )
        conn.execute(text("""
            DELETE FROM relations a USING relations b
            WHERE a.wiki_id = :w AND b.wiki_id = :w
              AND a.session_id = :sid AND b.session_id = :sid
              AND a.ctid > b.ctid
              AND a.from_title = b.from_title AND a.to_title = b.to_title AND a.label = b.label
        """), {"w": wiki_id, "sid": session_id})

        conn.execute(text("UPDATE clause_map SET page_title = :tgt WHERE wiki_id = :w AND session_id = :sid AND page_title = :src"), repoint)
        conn.execute(text("UPDATE contradictions SET page_title = :tgt WHERE wiki_id = :w AND session_id = :sid AND page_title = :src"), repoint)

        for emb_table in _page_embedding_tables(conn):
            conn.execute(
                text(f'DELETE FROM "{emb_table}" WHERE wiki_id = :w AND session_id = :sid AND title = ANY(:titles)'),
                {"w": wiki_id, "sid": session_id, "titles": [source_title, target_title]},
            )

        conn.commit()
    return True


# ---------------------------------------------------------------------------
# Review Queue — clauses (target architecture § 02, first slice)
# ---------------------------------------------------------------------------

def source_docs_with_title_token(wiki_id: str, session_id: str, token: str) -> list[str]:
    """Which documents carry this token in any of their PAGE TITLES.

    Ingest gives each document a short identifier inside its page titles
    ("Definitions - SA1-Vishesh-Realty (Framework Supply Agreement)"). That
    identifier names 378 documents in this corpus and appears in no filename,
    so scope resolution - which only ever read filenames - could not resolve a
    question that used one, which is exactly what a reader does after seeing it
    in an answer's citations.

    Capped: the caller only accepts a token that identifies ONE document, so
    there is no reason to drag back a long list to discover it is ambiguous.
    """
    from sqlalchemy import text
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT source_doc FROM pages
                WHERE wiki_id = :w AND session_id = :s AND title ILIKE :tok
                LIMIT 5
            """),
            {"w": wiki_id, "s": session_id, "tok": "%" + token + "%"},
        ).fetchall()
    return [r[0] for r in rows if r[0]]


def insert_clauses(wiki_id: str, session_id: str, source_doc: str, clauses: list[dict]) -> int:
    """Persist clauses extracted alongside a document's ingest LLM call.

    Called independently of _atomic_merge/_merge_wiki — clause rows are
    append-only per ingest, no merge-by-title logic like pages have, so
    this never touches that (complex, already-tested) machinery. Silently
    skips any clause missing a usable type/text/confidence rather than
    raising, since a single malformed entry in an LLM's JSON output
    shouldn't drop the rest of a real extraction.
    """
    from sqlalchemy import text
    engine = get_engine()
    inserted = 0
    with engine.connect() as conn:
        for c in clauses:
            clause_type = (c.get("type") or "").strip()
            verbatim_text = (c.get("text") or "").strip()
            confidence = c.get("confidence")
            if not clause_type or not verbatim_text or not isinstance(confidence, (int, float)):
                continue
            stakes = "high" if _is_high_stakes_clause_type(clause_type) else "low"
            # Canonical type is derived here rather than backfilled later, so a
            # newly ingested clause is queryable by canon immediately. NULL
            # when the vocabulary declines — never a nearest guess.
            from services import clause_vocab as _vocab
            canon = _vocab.canonical(clause_type)
            conn.execute(
                text("""
                    INSERT INTO clauses
                        (wiki_id, session_id, source_doc, clause_type, clause_type_canon,
                         verbatim_text, typed_value, confidence, page_num, stakes)
                    VALUES
                        (:w, :sid, :doc, :ctype, :canon, :vtext, :tval, :conf, :page, :stakes)
                """),
                {
                    "w": wiki_id, "sid": session_id, "doc": source_doc, "ctype": clause_type,
                    "canon": canon,
                    # Verbatim clause text is the most sensitive thing this
                    # table holds — it is the client's actual contract wording.
                    # Encrypted at rest; no-op when no key is configured.
                    "vtext": _crypto().encrypt(verbatim_text),
                    "tval": json.dumps(_crypto().encrypt_json(c["typed_value"]))
                            if c.get("typed_value") is not None else None,
                    "conf": float(confidence), "page": c.get("page"), "stakes": stakes,
                },
            )
            inserted += 1
        conn.commit()
    return inserted


# Fixed set drawn directly from the target architecture doc's own Fig. 3
# flow text ("liability · indemnity · termination · lifecycle") — computed
# here in Python, never left to the extracting LLM to self-report, since
# the whole bulk-accept/individual-sign-off split depends on this being a
# rule the model can't quietly route around.
_HIGH_STAKES_CLAUSE_TYPES = {"liability", "indemnity", "termination", "lifecycle"}


def _is_high_stakes_clause_type(clause_type: str) -> bool:
    lowered = clause_type.lower()
    return any(h in lowered for h in _HIGH_STAKES_CLAUSE_TYPES)


# Review Queue item kinds beyond clauses. Doc-type is deliberately always
# high stakes: it decides which schema is applied to the WHOLE document, so a
# wrong call there is not one bad field but the wrong set of fields entirely
# — strictly higher stakes than any single value within the right set.
_ALWAYS_HIGH_STAKES_KINDS = {"doc_type"}


def insert_answer_feedback(wiki_id: str, session_id: str, *, question: str,
                           verdict: str, answer_excerpt: str = "",
                           note: str = "", message_id=None,
                           scope_method: str = "", confidence_value=None,
                           confidence_governing: str = "",
                           files_used=None) -> int:
    """Record a reader's verdict on one answer.

    The only ground truth this system has about usefulness. Every automated
    score here measures whether an answer is faithful to the text it cites,
    and an answer can be perfectly faithful and still miss what the lawyer
    needed - no check in the pipeline can see that, because it is not a
    property of the documents.

    Stored alongside the scores the pipeline produced for the same answer, so
    the six-dimension confidence can eventually be calibrated rather than
    trusted: it currently separates correct from failing answers by 4.8 points
    on the 200-question set, which is real but far too weak to gate on, and
    nothing but human verdicts can say whether that gap is widenable.
    """
    from sqlalchemy import text
    import json as _json
    v = (verdict or "").strip().lower()
    if v not in ("up", "down"):
        raise ValueError("verdict must be 'up' or 'down', got %r" % verdict)
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            INSERT INTO answer_feedback
                (wiki_id, session_id, message_id, question, answer_excerpt,
                 verdict, note, scope_method, confidence_value,
                 confidence_governing, files_used)
            VALUES (:w, :s, :m, :q, :a, :v, :n, :sm, :cv, :cg, CAST(:f AS jsonb))
            RETURNING id
        """), {"w": wiki_id, "s": session_id, "m": message_id,
               "q": (question or "")[:4000], "a": (answer_excerpt or "")[:4000],
               "v": v, "n": (note or "")[:2000], "sm": scope_method or "",
               "cv": confidence_value, "cg": confidence_governing or "",
               "f": _json.dumps(list(files_used or [])[:12])}).fetchone()
        conn.commit()
        return int(row[0])


def get_answer_feedback(wiki_id: str, session_id: str = "",
                        limit: int = 200) -> list[dict]:
    """Recorded verdicts, newest first. Session-scoped when one is given."""
    from sqlalchemy import text
    engine = get_engine()
    sql = ("SELECT id, session_id, question, verdict, note, scope_method, "
           "confidence_value, confidence_governing, created_at "
           "FROM answer_feedback WHERE wiki_id = :w")
    params = {"w": wiki_id, "n": max(1, min(int(limit or 200), 1000))}
    if session_id:
        sql += " AND session_id = :s"
        params["s"] = session_id
    sql += " ORDER BY id DESC LIMIT :n"
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [{"id": r[0], "session_id": r[1], "question": r[2], "verdict": r[3],
             "note": r[4], "scope_method": r[5], "confidence_value": r[6],
             "confidence_governing": r[7], "created_at": str(r[8])} for r in rows]


def feedback_calibration(wiki_id: str) -> dict:
    """Does the six-score confidence predict what readers actually think?

    The question the score cannot answer about itself. Returns the mean
    confidence behind each verdict and the gap between them; a gap near zero
    means the score is decorative, however carefully it was built.
    """
    from sqlalchemy import text
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT verdict, count(*), avg(confidence_value)
            FROM answer_feedback
            WHERE wiki_id = :w AND confidence_value IS NOT NULL
            GROUP BY verdict
        """), {"w": wiki_id}).fetchall()
    by = {r[0]: {"n": int(r[1]), "mean_confidence": float(r[2])} for r in rows}
    out = {"by_verdict": by, "separation": None}
    if "up" in by and "down" in by:
        out["separation"] = round(by["up"]["mean_confidence"]
                                  - by["down"]["mean_confidence"], 1)
    return out


def insert_review_items(wiki_id: str, session_id: str, source_doc: str,
                        items: list[dict]) -> int:
    """Queue non-clause flagged extractions.

    `stakes` is computed here, never taken from the caller's payload and
    never from the model — same rule the clause queue already enforces, for
    the same reason: the bulk-accept split is only meaningful if the thing
    deciding it can't be talked into a different answer.
    """
    if not items:
        return 0
    from sqlalchemy import text
    engine = get_engine()
    inserted = 0
    with engine.connect() as conn:
        for it in items:
            kind = (it.get("item_kind") or "").strip()
            label = (it.get("item_label") or "").strip()
            if not kind or not label:
                continue
            if kind in _ALWAYS_HIGH_STAKES_KINDS:
                stakes = "high"
            elif it.get("high_stakes"):
                stakes = "high"
            else:
                stakes = "low"
            conn.execute(text("""
                INSERT INTO review_queue
                    (wiki_id, session_id, source_doc, item_kind, item_label,
                     item_value, typed_value, confidence, stakes, reason, page_num)
                VALUES
                    (:w, :sid, :doc, :kind, :label, :val, :tval, :conf, :stakes,
                     :reason, :page)
            """), {
                "w": wiki_id, "sid": session_id, "doc": source_doc,
                "kind": kind, "label": label,
                "val": (_crypto().encrypt(str(it["item_value"])[:4000])
                        if it.get("item_value") is not None else None),
                "tval": json.dumps(_crypto().encrypt_json(it["typed_value"]))
                        if it.get("typed_value") is not None else None,
                "conf": float(it.get("confidence") or 0.0),
                "stakes": stakes, "reason": it.get("reason"),
                "page": it.get("page_num"),
            })
            inserted += 1
        conn.commit()
    return inserted


def supersede_review_items(wiki_id: str, session_id: str, source_doc: str) -> int:
    """On re-ingest, move a document's prior pending items to `superseded`
    rather than deleting them.

    The doc is explicit that old resolutions are archived, not dropped: a
    reviewer's judgement on a previous version is evidence about how this
    document was read, and destroying it on every re-ingest would quietly
    erase the audit trail the queue exists to create.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        res = conn.execute(text("""
            UPDATE review_queue
            SET review_status = 'superseded', superseded_at = now()
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc
              AND review_status = 'pending'
        """), {"w": wiki_id, "sid": session_id, "doc": source_doc})
        conn.commit()
        return res.rowcount or 0


def get_review_queue(wiki_id: str, session_id: str) -> list[dict]:
    """Pending review items for one session — clauses and every other flagged
    extraction kind in one queue.

    Sorted by stakes (high first, so items needing individual sign-off surface
    before the bulk-accept pile) then confidence ascending (most likely wrong
    first). `item_kind` tells the caller which store a row came from, and
    resolution routes on it.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        clause_rows = conn.execute(
            text("""
                SELECT id, source_doc, clause_type, verbatim_text, typed_value,
                       confidence, page_num, stakes, created_at
                FROM clauses
                WHERE wiki_id = :w AND session_id = :sid AND review_status = 'pending'
            """),
            {"w": wiki_id, "sid": session_id},
        ).fetchall()
        other_rows = conn.execute(
            text("""
                SELECT id, source_doc, item_kind, item_label, item_value,
                       typed_value, confidence, page_num, stakes, reason, created_at
                FROM review_queue
                WHERE wiki_id = :w AND session_id = :sid AND review_status = 'pending'
            """),
            {"w": wiki_id, "sid": session_id},
        ).fetchall()

    # Decrypt on the way out. decrypt_safe rather than decrypt: one row written
    # under a rotated key should not blank the whole queue — it shows as
    # unreadable and the rest still triages.
    _c = _crypto()
    items = [
        {
            "id": r.id, "item_kind": "clause", "source_doc": r.source_doc,
            "clause_type": r.clause_type,
            "verbatim_text": _c.decrypt_safe(r.verbatim_text, "[unreadable — encryption key mismatch]"),
            "typed_value": _c.decrypt_json(r.typed_value),
            "confidence": r.confidence,
            "page_num": r.page_num, "stakes": r.stakes, "reason": None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in clause_rows
    ]
    items += [
        {
            # `clause_type` / `verbatim_text` are populated for every kind so
            # the shipped Review Queue UI renders these without a special case
            # — the label is what the card shows, the value is its body.
            "id": r.id, "item_kind": r.item_kind, "source_doc": r.source_doc,
            "clause_type": r.item_label,
            "verbatim_text": _c.decrypt_safe(r.item_value, "") or "",
            "typed_value": _c.decrypt_json(r.typed_value),
            "confidence": r.confidence,
            "page_num": r.page_num, "stakes": r.stakes, "reason": r.reason,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in other_rows
    ]
    items.sort(key=lambda i: (i["stakes"] != "high", i["confidence"] or 0.0))
    return items


def resolve_clause(wiki_id: str, session_id: str, clause_id: int, action: str, edited_text: str | None = None) -> bool:
    """Resolve one clause: accept, reject, or edit (accept with a corrected
    verbatim_text). Returns False if the clause doesn't exist, isn't in this
    session, or isn't still pending. Provenance-marked via `resolution`,
    per the doc's "accept, edit, or reject all write a human-confirmed
    marker" requirement.
    """
    from sqlalchemy import text
    engine = get_engine()
    resolution_map = {"accept": "individual_accept", "reject": "individual_reject", "edit": "edited"}
    if action not in resolution_map:
        raise ValueError(f"Unknown action {action!r} — expected accept, reject, or edit")
    review_status = "rejected" if action == "reject" else "approved"
    with engine.connect() as conn:
        params = {
            "w": wiki_id, "sid": session_id, "id": clause_id,
            "status": review_status, "resolution": resolution_map[action],
        }
        if action == "edit":
            if not edited_text or not edited_text.strip():
                raise ValueError("edited_text is required for action=edit")
            result = conn.execute(
                text("""
                    UPDATE clauses SET review_status = :status, resolution = :resolution,
                           verbatim_text = :vtext, reviewed_at = now()
                    WHERE wiki_id = :w AND session_id = :sid AND id = :id AND review_status = 'pending'
                """),
                # A reviewer's corrected wording is as sensitive as the
                # extracted original, so it is encrypted on the same terms.
                {**params, "vtext": _crypto().encrypt(edited_text.strip())},
            )
        else:
            result = conn.execute(
                text("""
                    UPDATE clauses SET review_status = :status, resolution = :resolution,
                           reviewed_at = now()
                    WHERE wiki_id = :w AND session_id = :sid AND id = :id AND review_status = 'pending'
                """),
                params,
            )
        conn.commit()
        return (result.rowcount or 0) > 0


def get_review_documents(wiki_id: str, session_id: str) -> list[dict]:
    """The Review Queue grouped by document, not by flagged item.

    An item-level queue answers "what is doubtful"; it does not answer "is
    this document right", which is the question a reviewer with the source
    text in front of them is actually able to settle. This returns one entry
    per document with pending work: its classification, every extracted
    metadata field with that field's own confidence, and the counts behind
    the summary — so the reviewer reads a document once rather than meeting
    its fields scattered through a list of unrelated cards.
    """
    from sqlalchemy import text
    engine = get_engine()
    _c = _crypto()

    with engine.connect() as conn:
        docs = conn.execute(text("""
            SELECT source_doc, doc_family, doc_type, jurisdiction,
                   family_confidence, family_method, folder_hint,
                   lifecycle, schema_version, created_at
            FROM documents
            WHERE wiki_id = :w AND session_id = :s
            ORDER BY family_confidence ASC NULLS FIRST, created_at DESC
        """), {"w": wiki_id, "s": session_id}).fetchall()

        clause_counts = dict(conn.execute(text("""
            SELECT source_doc, COUNT(*) FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND review_status = 'pending'
            GROUP BY source_doc
        """), {"w": wiki_id, "s": session_id}).fetchall())
        high_counts = dict(conn.execute(text("""
            SELECT source_doc, COUNT(*) FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND review_status = 'pending' AND stakes = 'high'
            GROUP BY source_doc
        """), {"w": wiki_id, "s": session_id}).fetchall())
        item_counts = dict(conn.execute(text("""
            SELECT source_doc, COUNT(*) FROM review_queue
            WHERE wiki_id = :w AND session_id = :s AND review_status = 'pending'
            GROUP BY source_doc
        """), {"w": wiki_id, "s": session_id}).fetchall())
        page_counts = dict(conn.execute(text("""
            SELECT source_doc, COUNT(*) FROM pages
            WHERE wiki_id = :w AND session_id = :s GROUP BY source_doc
        """), {"w": wiki_id, "s": session_id}).fetchall())

        # Family typed rows carry the extracted metadata. Which table holds a
        # given document depends on its family, so they are read together and
        # matched by source_doc rather than joined per family.
        typed: dict[str, tuple] = {}
        for tbl in ("contracts", "litigation_facts", "authorizations", "opinions"):
            try:
                for r in conn.execute(text(
                    f"SELECT source_doc, typed_value, confidence FROM {tbl} "
                    f"WHERE wiki_id = :w AND session_id = :s"
                ), {"w": wiki_id, "s": session_id}).fetchall():
                    typed[r[0]] = (r[1], r[2])
            except Exception as err:
                logger.debug("Could not read %s for review documents: %s", tbl, err)

    out: list[dict] = []
    for d in docs:
        source_doc = d[0]
        raw_typed, row_conf = typed.get(source_doc, (None, None))
        fields = _review_fields(raw_typed, _c)
        pending = int(clause_counts.get(source_doc, 0)) + int(item_counts.get(source_doc, 0))
        out.append({
            "source_doc": source_doc,
            "doc_family": d[1],
            "doc_type": d[2],
            "jurisdiction": d[3],
            "family_confidence": d[4],
            "family_method": d[5],
            "folder_hint": d[6],
            "lifecycle": d[7],
            "schema_version": d[8],
            "created_at": d[9].isoformat() if d[9] else None,
            "fields": fields,
            "metadata_confidence": row_conf,
            "page_count": int(page_counts.get(source_doc, 0)),
            "pending_total": pending,
            "pending_clauses": int(clause_counts.get(source_doc, 0)),
            "pending_high_stakes": int(high_counts.get(source_doc, 0)),
            "pending_items": int(item_counts.get(source_doc, 0)),
            "lowest_confidence": min(
                [f["confidence"] for f in fields if f["confidence"] is not None]
                + ([d[4]] if d[4] is not None else []) or [None]
            ) if (fields or d[4] is not None) else None,
        })
    return out


def _review_fields(raw_typed, crypto_mod) -> list[dict]:
    """Normalize a family row's typed_value into display fields.

    Handles both shapes on purpose: rows written before per-field confidence
    existed carry only values plus a flagged-field list, and those documents
    should still be reviewable rather than showing an empty panel. Their
    fields report a null confidence, which the UI renders as unknown — an
    honest blank rather than a fabricated number.
    """
    if raw_typed is None:
        return []
    decoded = crypto_mod.decrypt_json(raw_typed)
    if isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except Exception:
            return []
    if not isinstance(decoded, dict):
        return []

    per_field = decoded.get("fields")
    if isinstance(per_field, dict) and per_field:
        rows = []
        for name, meta in per_field.items():
            if not isinstance(meta, dict):
                continue
            rows.append({
                "name": name,
                "value": meta.get("value"),
                "raw": meta.get("raw"),
                "confidence": meta.get("confidence"),
                "flagged": bool(meta.get("flagged")),
                "coerced": bool(meta.get("coerced")),
                "reason": meta.get("reason"),
                "high_stakes": bool(meta.get("high_stakes")),
                # Edit provenance — a human-corrected value and a
                # model-extracted one that happen to match are not the same
                # fact, so the UI shows which is which.
                "edited": bool(meta.get("edited")),
                "edited_at": meta.get("edited_at"),
                "previous_value": meta.get("previous_value"),
            })
        rows.sort(key=lambda r: (r["confidence"] is None,
                                 r["confidence"] if r["confidence"] is not None else 1.0))
        return rows

    # Legacy shape — values only.
    validated = decoded.get("validated")
    if not isinstance(validated, dict):
        return []
    flagged = set(decoded.get("flagged_fields") or [])
    return [
        {"name": name, "value": value, "raw": None, "confidence": None,
         "flagged": name in flagged, "coerced": False, "reason": None,
         "high_stakes": False}
        for name, value in validated.items()
    ]


def get_document_review_items(wiki_id: str, session_id: str, source_doc: str) -> list[dict]:
    """Every pending flagged item for one document — clauses and other kinds."""
    all_items = get_review_queue(wiki_id, session_id)
    return [i for i in all_items if i["source_doc"] == source_doc]


def resolve_document(wiki_id: str, session_id: str, source_doc: str, action: str,
                     min_confidence: float = 0.0) -> dict:
    """Approve or reject every pending item for one document.

    Approve still refuses high-stakes clauses below the threshold: reviewing
    a document as a whole is a workflow convenience, not a licence to wave
    through the items the stakes rule exists to protect. Those stay pending
    and are reported back, so the caller can see the document is not finished
    rather than assume it is.
    """
    from sqlalchemy import text
    if action not in ("approve", "reject"):
        raise ValueError(f"Unknown action {action!r} — expected approve or reject")
    engine = get_engine()
    status = "approved" if action == "approve" else "rejected"
    resolution = "document_accepted" if action == "approve" else "document_rejected"

    with engine.connect() as conn:
        if action == "reject":
            clause_where = ""
            item_where = ""
            params = {"w": wiki_id, "sid": session_id, "doc": source_doc,
                      "status": status, "res": resolution}
        else:
            clause_where = " AND (stakes = 'low' OR confidence >= :minc)"
            item_where = " AND (stakes = 'low' OR confidence >= :minc)"
            params = {"w": wiki_id, "sid": session_id, "doc": source_doc, "status": status,
                      "res": resolution, "minc": min_confidence}

        n_clauses = conn.execute(text(f"""
            UPDATE clauses SET review_status = :status, resolution = :res,
                   reviewed_at = now()
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc
              AND review_status = 'pending'{clause_where}
        """), params).rowcount or 0
        n_items = conn.execute(text(f"""
            UPDATE review_queue SET review_status = :status, resolution = :res,
                   reviewed_at = now()
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc
              AND review_status = 'pending'{item_where}
        """), params).rowcount or 0
        remaining = (conn.execute(text("""
            SELECT COUNT(*) FROM clauses WHERE wiki_id = :w AND session_id = :sid
              AND source_doc = :doc AND review_status = 'pending'
        """), {"w": wiki_id, "sid": session_id, "doc": source_doc}).scalar() or 0) + \
            (conn.execute(text("""
            SELECT COUNT(*) FROM review_queue WHERE wiki_id = :w AND session_id = :sid
              AND source_doc = :doc AND review_status = 'pending'
        """), {"w": wiki_id, "sid": session_id, "doc": source_doc}).scalar() or 0)
        conn.commit()

    return {"clauses": n_clauses, "items": n_items, "remaining": int(remaining)}


def resolve_review_item(wiki_id: str, session_id: str, item_id: int, action: str,
                        edited_text: str | None = None) -> bool:
    """Resolve one non-clause review item. Mirrors resolve_clause exactly,
    including the still-pending guard, so a double-submit from the UI can't
    silently overwrite an earlier reviewer's decision."""
    from sqlalchemy import text
    engine = get_engine()
    resolution_map = {"accept": "individual_accept", "reject": "individual_reject",
                      "edit": "edited"}
    if action not in resolution_map:
        raise ValueError(f"Unknown action {action!r} — expected accept, reject, or edit")
    review_status = "rejected" if action == "reject" else "approved"
    with engine.connect() as conn:
        params = {"w": wiki_id, "sid": session_id, "id": item_id, "status": review_status,
                  "resolution": resolution_map[action]}
        if action == "edit":
            if not edited_text or not edited_text.strip():
                raise ValueError("edited_text is required for action=edit")
            result = conn.execute(text("""
                UPDATE review_queue SET review_status = :status, resolution = :resolution,
                       item_value = :val, reviewed_at = now()
                WHERE wiki_id = :w AND session_id = :sid AND id = :id AND review_status = 'pending'
            """), {**params, "val": _crypto().encrypt(edited_text.strip())})
        else:
            result = conn.execute(text("""
                UPDATE review_queue SET review_status = :status, resolution = :resolution,
                       reviewed_at = now()
                WHERE wiki_id = :w AND session_id = :sid AND id = :id AND review_status = 'pending'
            """), params)
        conn.commit()
        return (result.rowcount or 0) > 0


def bulk_accept_review_items(wiki_id: str, session_id: str, min_confidence: float) -> int:
    """Bulk-accept low-stakes non-clause items at or above the threshold.

    Same server-side exclusion as clauses: high stakes is filtered in the
    WHERE clause, so a request body claiming otherwise changes nothing. This
    also means doc-type items are never bulk-acceptable, since they are
    always recorded as high stakes.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            UPDATE review_queue
            SET review_status = 'approved', resolution = 'bulk_accepted', reviewed_at = now()
            WHERE wiki_id = :w AND session_id = :sid AND review_status = 'pending'
              AND stakes = 'low' AND confidence >= :min_conf
        """), {"w": wiki_id, "sid": session_id, "min_conf": min_confidence})
        conn.commit()
        return result.rowcount or 0


def bulk_accept_clauses(wiki_id: str, session_id: str, min_confidence: float) -> int:
    """Accept every pending LOW-stakes clause at or above min_confidence.

    High-stakes clauses are excluded by the WHERE clause itself, not by
    trusting the caller — "individual sign-off only" for liability/
    indemnity/termination/lifecycle is enforced here regardless of what a
    request body claims, per the doc's stakes-tier split.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                UPDATE clauses
                SET review_status = 'approved', resolution = 'bulk_accepted', reviewed_at = now()
                WHERE wiki_id = :w AND session_id = :sid AND review_status = 'pending'
                  AND stakes = 'low' AND confidence >= :min_conf
            """),
            {"w": wiki_id, "sid": session_id, "min_conf": min_confidence},
        )
        conn.commit()
        return result.rowcount or 0


def get_page(wiki_id: str, session_id: str, title: str) -> dict | None:
    """Return a single page dict or None if it does not exist."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT content, summary, source_doc, contradiction_flagged, variants, "
                 "append_count "
                 "FROM pages WHERE wiki_id = :w AND session_id = :sid AND title = :title"),
            {"w": wiki_id, "sid": session_id, "title": title},
        ).fetchone()
        if row is None:
            return None
        page: dict[str, Any] = {
            "content": row.content,
            "summary": row.summary,
            "source_doc": row.source_doc,
            # Required by run_compaction()'s post-lock staleness re-check. Without
            # it that check read .get("append_count", 0) -> 0, which is always
            # below the threshold, so every page was judged "no longer due" and
            # compaction silently never ran while still paying for a lock and a
            # re-read per candidate page on every single ingest.
            "append_count": row.append_count or 0,
        }
        if row.contradiction_flagged:
            page["contradiction_flagged"] = True
        if row.variants is not None:
            page["variants"] = row.variants
        return page


def get_page_titles(wiki_id: str, session_id: str) -> list[str]:
    """Return only page titles for a session (cheaper than get_pages for file-name lookups)."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT title FROM pages WHERE wiki_id = :w AND session_id = :sid"),
            {"w": wiki_id, "sid": session_id},
        )
        return [row.title for row in rows]


def get_all_page_titles_and_content(wiki_id: str, session_id: str) -> dict[str, str]:
    """Return {title: content} for all pages. Used for the cross-reference pass."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT title, content FROM pages WHERE wiki_id = :w AND session_id = :sid"),
            {"w": wiki_id, "sid": session_id},
        )
        return {row.title: row.content for row in rows}


def count_pages(wiki_id: str, session_id: str) -> int:
    """Return the number of pages stored for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM pages WHERE wiki_id = :w AND session_id = :sid"),
            {"w": wiki_id, "sid": session_id},
        ).scalar()
        return result or 0


def count_relations(wiki_id: str, session_id: str) -> int:
    """Return the number of relations stored for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM relations WHERE wiki_id = :w AND session_id = :sid"),
            {"w": wiki_id, "sid": session_id},
        ).scalar()
        return result or 0


# ---------------------------------------------------------------------------
# Document lifecycle (target architecture § 01.4 — Archive / Hard-delete)
#
# See services/documents.py for the orchestration layer (file deletion,
# sessions.json bookkeeping) and the disclosed limitation around pages
# merged from multiple source documents — everything here operates on
# whatever pages.source_doc currently records, which for a merged page is
# only the most recent contributing document, not the full set.
# ---------------------------------------------------------------------------

def get_document_statuses(wiki_id: str, session_id: str) -> dict[str, dict]:
    """Return {source_doc: {status, archived_at}} for every document that has
    ever been archived. A document with no row here is active."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT source_doc, status, archived_at FROM document_status WHERE wiki_id = :w AND session_id = :sid"),
            {"w": wiki_id, "sid": session_id},
        )
        return {
            r.source_doc: {
                "status": r.status,
                "archived_at": r.archived_at.isoformat() if r.archived_at else None,
            }
            for r in rows
        }


def archive_document(wiki_id: str, session_id: str, source_doc: str) -> None:
    """Mark a document archived. Idempotent — archiving twice just refreshes archived_at."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO document_status (wiki_id, session_id, source_doc, status, archived_at)
                VALUES (:w, :sid, :doc, 'archived', now())
                ON CONFLICT (session_id, source_doc)
                DO UPDATE SET status = 'archived', archived_at = now(), wiki_id = :w
            """),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )
        conn.commit()


def unarchive_document(wiki_id: str, session_id: str, source_doc: str) -> None:
    """Restore an archived document to active by removing its status row —
    keeps "row exists" == "archived" a valid invariant everywhere else that
    reads this table, rather than also needing to check a status value."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM document_status WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )
        conn.commit()


def is_document_archived(wiki_id: str, session_id: str, source_doc: str) -> bool:
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT 1 FROM document_status
                WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc AND status = 'archived'
            """),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        ).first()
        return row is not None


def delete_document_pages_exclusive(wiki_id: str, session_id: str,
                                    source_doc: str) -> dict:
    """Delete only the pages this document is the SOLE contributor to.

    For re-ingest, not for document deletion. `pages.source_doc` records the
    LAST document to write a page, so a merged concept page — one several
    documents contributed to — can carry this document's name while most of
    its content came from others. Deleting by source_doc alone therefore
    destroys other documents' work, and re-ingesting this one document cannot
    rebuild it: it only re-contributes its own share.

    Measured on the live corpus when this was found the hard way: four
    documents whose pypdf text is 248-1,860 characters were each credited with
    12-13 pages, and clearing them by source_doc cost 36 pages of merged
    content that no single re-ingest could restore.

    `variants` is the discriminator — it holds one entry per contribution, and
    is NULL exactly on pages written once. Shared pages are left in place for
    wiki.ingest to merge into, which blends rather than swaps; blending a page
    several documents built is the correct trade against deleting the other
    contributors outright.
    """
    from sqlalchemy import text
    engine = get_engine()
    report = {"pages_deleted": 0, "embeddings_deleted": 0, "relations_deleted": 0,
              "shared_pages_kept": 0, "clause_map_deleted": 0,
              "source_positions_deleted": 0}
    with engine.connect() as conn:
        titles = [r.title for r in conn.execute(text("""
            SELECT title FROM pages
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc
              AND (variants IS NULL OR jsonb_array_length(variants) <= 1)
              AND COALESCE(append_count, 0) <= 1
        """), {"w": wiki_id, "sid": session_id, "doc": source_doc})]
        report["shared_pages_kept"] = conn.execute(text("""
            SELECT count(*) FROM pages
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc
              AND NOT ((variants IS NULL OR jsonb_array_length(variants) <= 1)
                       AND COALESCE(append_count, 0) <= 1)
        """), {"w": wiki_id, "sid": session_id, "doc": source_doc}).scalar() or 0

        if titles:
            emb_tables = [r[0] for r in conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'page_embeddings%'"
            ))]
            for emb_table in emb_tables:
                res = conn.execute(text(
                    f'DELETE FROM "{emb_table}" WHERE wiki_id = :w AND session_id = :sid '
                    f'AND title = ANY(:titles)'),
                    {"w": wiki_id, "sid": session_id, "titles": titles})
                report["embeddings_deleted"] += res.rowcount or 0

            res = conn.execute(text("""
                DELETE FROM pages WHERE wiki_id = :w AND session_id = :sid
                  AND title = ANY(:titles)
            """), {"w": wiki_id, "sid": session_id, "titles": titles})
            report["pages_deleted"] = res.rowcount or 0

            res = conn.execute(text("""
                DELETE FROM relations WHERE wiki_id = :w AND session_id = :sid
                  AND (from_title = ANY(:titles) OR to_title = ANY(:titles))
            """), {"w": wiki_id, "sid": session_id, "titles": titles})
            report["relations_deleted"] = res.rowcount or 0

        # Both are per-document by construction, so they carry no other
        # document's work and are safe to clear whole.
        res = conn.execute(text(
            "DELETE FROM clause_map WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
            {"w": wiki_id, "sid": session_id, "doc": source_doc})
        report["clause_map_deleted"] = res.rowcount or 0
        res = conn.execute(text(
            "DELETE FROM source_positions WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
            {"w": wiki_id, "sid": session_id, "doc": source_doc})
        report["source_positions_deleted"] = res.rowcount or 0
        conn.commit()
    return report


def delete_document_data(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """Cascade-delete every DB row cleanly attributable to one document.

    Does NOT touch the uploaded file on disk or sessions.json — see
    services/documents.py, which owns those and calls this for the DB side.
    Does NOT touch `entities` — it's a global canonical-entity registry
    shared across documents (keyed by wiki_id/canonical_key, not
    source_doc); this document's rows in `entity_aliases` are removed,
    which may leave an entity with zero remaining aliases, but that's an
    orphan in a dedup table, not a data-integrity problem worth chasing here.

    Deliberately scoped to what pages.source_doc actually records: a
    shared/merged concept page (a statute, a clause type referenced by
    several documents) holds only its LAST contributing document in that
    column — see wiki.py's merge-guard comment near "silently overwrites
    source_doc". A page this document once contributed to, but no longer
    "owns" in that column, is left untouched. That's a real gap in today's
    schema, not a bug in this function; closing it needs the target
    architecture's per-document structured tables (Phase 0 backbone), not
    built yet. Returns a report of exactly what was removed so the caller
    can say so honestly instead of claiming full removal.
    """
    from sqlalchemy import text
    engine = get_engine()
    report = {
        "pages_deleted": 0, "embeddings_deleted": 0,
        "clause_map_deleted": 0, "source_positions_deleted": 0,
        "relations_deleted": 0, "contradictions_deleted": 0,
        "page_metadata_deleted": 0,
        "clauses_deleted": 0, "clause_embeddings_deleted": 0,
        "question_embeddings_deleted": 0,
        "contracts_deleted": 0, "obligations_deleted": 0,
        "litigation_facts_deleted": 0, "authorizations_deleted": 0,
        "opinions_deleted": 0, "citations_deleted": 0,
        "structural_anchors_deleted": 0, "entity_aliases_deleted": 0,
        "tables_deleted": 0, "figures_deleted": 0,
        "review_queue_deleted": 0, "document_relations_deleted": 0,
        "collection_documents_deleted": 0, "playbook_findings_deleted": 0,
        "documents_deleted": 0,
    }

    # Tables keyed directly by (wiki_id, session_id, source_doc) — the
    # Phase-0-ish typed tables added alongside `documents`, none of them
    # carrying a real FK to it (confirmed: only collection_documents,
    # playbook_rules/runs/findings have any pg_constraint fkey, and none
    # point at `documents`), so deletion order among them doesn't matter.
    simple_tables = [
        ("contracts", "contracts_deleted"),
        ("obligations", "obligations_deleted"),
        ("litigation_facts", "litigation_facts_deleted"),
        ("authorizations", "authorizations_deleted"),
        ("opinions", "opinions_deleted"),
        ("citations", "citations_deleted"),
        ("structural_anchors", "structural_anchors_deleted"),
        ("tables", "tables_deleted"),
        ("figures", "figures_deleted"),
        ("review_queue", "review_queue_deleted"),
        ("collection_documents", "collection_documents_deleted"),
        ("clauses", "clauses_deleted"),
        # defined_terms is DERIVED from the clause rows above by
        # services/defined_terms.build(), but it is still per-document data
        # keyed the same way, and leaving it behind outlived the document:
        # after a 51-document re-ingest, 26 rows still described documents
        # that no longer had a single page. A stale definition is worse than
        # a missing one - the defined-terms fast path answers from this table
        # directly, with no page to contradict it.
        ("defined_terms", "defined_terms_deleted"),
    ]
    # entity_aliases and playbook_findings share the same source_doc filter
    # but have no session_id column.
    wiki_only_tables = [
        ("entity_aliases", "entity_aliases_deleted"),
        ("playbook_findings", "playbook_findings_deleted"),
    ]

    with engine.connect() as conn:
        titles = [r.title for r in conn.execute(
            text("SELECT title FROM pages WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )]

        if titles:
            # Every page_embeddings* table across every provider ever used —
            # switching EMBEDDING_PROVIDER doesn't merge or drop the previous
            # provider's table (see _emb_table_name), so a stale vector under
            # a now-inactive provider would otherwise survive this delete.
            emb_tables = [r[0] for r in conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'page_embeddings%'"
            ))]
            for emb_table in emb_tables:
                result = conn.execute(
                    text(f'DELETE FROM "{emb_table}" WHERE wiki_id = :w AND session_id = :sid AND title = ANY(:titles)'),
                    {"w": wiki_id, "sid": session_id, "titles": titles},
                )
                report["embeddings_deleted"] += result.rowcount or 0

            result = conn.execute(
                text("DELETE FROM pages WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
                {"w": wiki_id, "sid": session_id, "doc": source_doc},
            )
            report["pages_deleted"] = result.rowcount or 0

            # Deleted rather than re-labelled: today's relations.label is a
            # real semantic field (party<->clause, clause<->clause) — the
            # target architecture's "mark as references-unresolved instead
            # of orphaning" treatment is specified for its own future typed
            # edge set, not a licence to overwrite this column's existing
            # meaning. An edge pointing at a title that no longer exists is
            # simply dangling once the page is gone; delete it rather than
            # leave a graph edge to nothing or invent semantics this schema
            # doesn't yet have a field for.
            result = conn.execute(
                text("""
                    DELETE FROM relations
                    WHERE wiki_id = :w AND session_id = :sid
                      AND (from_title = ANY(:titles) OR to_title = ANY(:titles))
                """),
                {"w": wiki_id, "sid": session_id, "titles": titles},
            )
            report["relations_deleted"] = result.rowcount or 0

            result = conn.execute(
                text("""
                    DELETE FROM contradictions
                    WHERE wiki_id = :w AND session_id = :sid AND page_title = ANY(:titles)
                """),
                {"w": wiki_id, "sid": session_id, "titles": titles},
            )
            report["contradictions_deleted"] = result.rowcount or 0

            result = conn.execute(
                text("""
                    DELETE FROM page_metadata
                    WHERE wiki_id = :w AND session_id = :sid AND title = ANY(:titles)
                """),
                {"w": wiki_id, "sid": session_id, "titles": titles},
            )
            report["page_metadata_deleted"] = result.rowcount or 0

        # clause_embeddings* / question_embeddings* — same per-provider
        # table-name discovery as page_embeddings above.
        for prefix, key in (("clause_embeddings", "clause_embeddings_deleted"),
                            ("question_embeddings", "question_embeddings_deleted")):
            emb_tables = [r[0] for r in conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE :pat"
            ), {"pat": f"{prefix}%"})]
            for emb_table in emb_tables:
                result = conn.execute(
                    text(f'DELETE FROM "{emb_table}" WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc'),
                    {"w": wiki_id, "sid": session_id, "doc": source_doc},
                )
                report[key] += result.rowcount or 0

        for table, key in simple_tables:
            result = conn.execute(
                text(f'DELETE FROM "{table}" WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc'),
                {"w": wiki_id, "sid": session_id, "doc": source_doc},
            )
            report[key] = result.rowcount or 0

        for table, key in wiki_only_tables:
            result = conn.execute(
                text(f'DELETE FROM "{table}" WHERE wiki_id = :w AND source_doc = :doc'),
                {"w": wiki_id, "doc": source_doc},
            )
            report[key] = result.rowcount or 0

        result = conn.execute(
            text("""
                DELETE FROM document_relations
                WHERE wiki_id = :w AND session_id = :sid
                  AND (from_doc = :doc OR to_doc = :doc)
            """),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )
        report["document_relations_deleted"] = result.rowcount or 0

        result = conn.execute(
            text("DELETE FROM clause_map WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )
        report["clause_map_deleted"] = result.rowcount or 0

        result = conn.execute(
            text("DELETE FROM source_positions WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )
        report["source_positions_deleted"] = result.rowcount or 0

        conn.execute(
            text("DELETE FROM document_status WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )

        # The `documents` row itself, last — every child table above is
        # keyed by source_doc directly (not documents.id via FK), so
        # nothing depends on ordering this before them; kept last only
        # for readability (row of record removed once its parts are gone).
        result = conn.execute(
            text("DELETE FROM documents WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc"),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )
        report["documents_deleted"] = result.rowcount or 0

        conn.commit()

    return report


# ---------------------------------------------------------------------------
# Embeddings (Phase 3 — pgvector search)
# ---------------------------------------------------------------------------

def upsert_question_embeddings(wiki_id: str, session_id: str, title: str,
                               questions: list[tuple[str, list[float]]],
                               doc_family: str | None = None,
                               source_doc: str | None = None) -> int:
    """Store hypothetical-question vectors for one page (§ 01 stage 06).

    Questions are written per page and replaced wholesale for that page, so a
    re-ingest can't leave a page answering questions its current content no
    longer supports — a stale question is worse than a missing one, because it
    surfaces the page confidently for a query it can't actually answer.
    """
    if not questions:
        return 0
    from sqlalchemy import text
    engine = get_engine()
    tbl = _question_table_name()
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {tbl} WHERE wiki_id = :w AND session_id = :sid AND title = :title"),
                     {"w": wiki_id, "sid": session_id, "title": title})
        n = 0
        for question, vector in questions:
            if not question or not vector:
                continue
            emb_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
            conn.execute(text(f"""
                INSERT INTO {tbl}
                    (wiki_id, session_id, title, question, embedding, doc_family, source_doc)
                VALUES (:w, :sid, :title, :q, CAST(:embedding AS vector), :fam, :doc)
                ON CONFLICT (session_id, title, question) DO UPDATE SET
                    embedding = EXCLUDED.embedding, wiki_id = :w
            """), {"w": wiki_id, "sid": session_id, "title": title, "q": question[:2000],
                   "embedding": emb_str, "fam": doc_family, "doc": source_doc})
            n += 1
        conn.commit()
    return n


def search_similar_questions(wiki_id: str, session_id: str, query_embedding: list[float],
                             limit: int = 10,
                             doc_family: str | None = None,
                             max_pages_sharing: int = 1) -> list[dict]:
    """Find pages whose hypothetical questions match the query.

    Returns page titles, not questions — the question is a retrieval handle,
    and what the caller ultimately needs is the page that can answer it. Each
    page appears once, at its best-matching question's score.

    The ranking has to happen in two steps. DISTINCT ON requires its own key to
    lead the ORDER BY, so a single-level query is sorted by TITLE, and a LIMIT
    on it returns the alphabetically-first N pages rather than the best-matching
    ones — sorting the result afterwards only reorders that alphabetical slice.
    Confirmed against the live table: querying with a page's OWN embedding, a
    perfect 1.0 match, did not return that page at all; it returned ten pages
    beginning "Absence..." and "Acceptance...", scoring around 0.5.

    So the inner query picks each page's best-matching question with no limit,
    and the outer one ranks pages by that score and takes the top N.

    max_pages_sharing drops question texts that too many pages share. Ingest
    generates a lot of boilerplate — "How does the Agreement define
    'Confidential Information'?" is the stored question for 124 different pages
    in this corpus, and every one of them scores identically against a query
    close to it. Such a question ranks documents at random, and feeding that into
    a fusion that rewards any highly-ranked candidate is how a page from an
    unrelated agreement gets into the context. 1 keeps only questions unique to
    one page (75% of this corpus's rows); a higher value trades precision for
    recall, and 0 disables the filter.
    """
    from sqlalchemy import text
    engine = get_engine()
    tbl = _question_table_name()
    emb_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"
    family_clause = " AND doc_family = :fam" if doc_family else ""
    params = {"w": wiki_id, "sid": session_id, "embedding": emb_str, "limit": limit}
    if doc_family:
        params["fam"] = doc_family
    if max_pages_sharing and max_pages_sharing > 0:
        params["maxshare"] = max_pages_sharing
        pool = """
            SELECT s.title, s.question, s.embedding
            FROM scoped s
            JOIN (SELECT question FROM scoped
                  GROUP BY question HAVING count(DISTINCT title) <= :maxshare) d
              ON d.question = s.question
        """
    else:
        pool = "SELECT title, question, embedding FROM scoped"
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            WITH scoped AS (
                SELECT title, question, embedding
                FROM {tbl}
                WHERE wiki_id = :w AND session_id = :sid{family_clause}
            ), pool AS ({pool})
            SELECT title, question, score FROM (
                SELECT DISTINCT ON (title)
                       title, question,
                       1 - (embedding <=> CAST(:embedding AS vector)) AS score
                FROM pool
                ORDER BY title, embedding <=> CAST(:embedding AS vector)
            ) best
            ORDER BY score DESC
            LIMIT :limit
        """), params).fetchall()
    return [{"title": r.title, "question": r.question, "score": float(r.score)}
            for r in rows]


def count_question_embeddings(wiki_id: str, session_id: str) -> int:
    from sqlalchemy import text
    try:
        with get_engine().connect() as conn:
            return int(conn.execute(text(
                f"SELECT COUNT(*) FROM {_question_table_name()} WHERE wiki_id = :w AND session_id = :sid"
            ), {"w": wiki_id, "sid": session_id}).scalar() or 0)
    except Exception:
        return 0


def upsert_embedding(wiki_id: str, session_id: str, title: str, embedding: list[float],
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
                INSERT INTO {tbl} (wiki_id, session_id, title, embedding, doc_family)
                VALUES (:w, :sid, :title, CAST(:embedding AS vector), :fam)
                ON CONFLICT (session_id, title) DO UPDATE SET
                    embedding  = EXCLUDED.embedding,
                    doc_family = COALESCE(EXCLUDED.doc_family, {tbl}.doc_family),
                    wiki_id    = :w
            """),
            {"w": wiki_id, "sid": session_id, "title": title, "embedding": emb_str, "fam": doc_family},
        )
        conn.commit()


def search_similar_pages(
    wiki_id: str, session_id: str, query_embedding: list[float], limit: int = 25,
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
    params = {"w": wiki_id, "sid": session_id, "embedding": emb_str, "limit": limit}
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
                WHERE wiki_id = :w AND session_id = :sid
                {family_clause}
                {cached_clause}
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """),
            params,
        )
        return [row.title for row in rows]


def _cite_key(s: str) -> str:
    """Comparison key for an authority name.

    'Arbitration and Conciliation Act 1996' and 'Arbitration and Conciliation
    Act, 1996' are the same statute recorded twice — normalized_form is
    model-written and does not converge on punctuation. Stripping everything
    but alphanumerics collapses that without pretending to understand citation
    formats.
    """
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def find_documents_citing(wiki_id: str, session_id: str, authority: str,
                          limit: int = 50) -> list[dict]:
    """Documents that cite a given statute/rule/authority — a SQL join, no LLM.

    Retrieval cannot answer "which documents cite the Arbitration Act": the
    citation is one line inside documents that are otherwise about unrelated
    subjects, so semantic similarity to the question ranks the wrong pages.
    The citations table already holds the answer as structured rows.

    Matched on a punctuation-stripped key so the two spellings of the same Act
    that the extraction produced are counted as one authority.
    """
    from sqlalchemy import text
    key = _cite_key(authority)
    if not key:
        return []
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT source_doc, citation_text, normalized_form, authority_type,
                   page_num, confidence
            FROM citations
            WHERE wiki_id = :w AND session_id = :sid
        """), {"w": wiki_id, "sid": session_id}).fetchall()

    hits: dict[str, dict] = {}
    for r in rows:
        for cand in (r.normalized_form, r.citation_text):
            ck = _cite_key(cand)
            if ck and (key in ck or ck in key):
                cur = hits.setdefault(r.source_doc, {
                    "source_doc": r.source_doc,
                    "citation_text": (r.citation_text or r.normalized_form or "").strip(),
                    "authority_type": r.authority_type,
                    "pages": set(),
                })
                if r.page_num:
                    cur["pages"].add(int(r.page_num))
                break
    out = [{**h, "pages": sorted(h["pages"])} for h in hits.values()]
    out.sort(key=lambda h: h["source_doc"])
    return out[:limit]


def get_authorities_cited(wiki_id: str, session_id: str, source_doc: str) -> list[dict]:
    """Every authority a document cites, deduplicated by normalized key."""
    from sqlalchemy import text
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT citation_text, normalized_form, authority_type, page_num
            FROM citations
            WHERE wiki_id = :w AND session_id = :sid
              AND (source_doc = :d OR source_doc ILIKE '%' || :d || '%')
            ORDER BY page_num NULLS LAST
        """), {"w": wiki_id, "sid": session_id, "d": source_doc}).fetchall()
    seen, out = set(), []
    for r in rows:
        label = (r.normalized_form or r.citation_text or "").strip()
        k = _cite_key(label)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append({"authority": label, "authority_type": r.authority_type,
                    "page_num": r.page_num})
    return out


def get_document_relations(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """Amendment / reference edges for a document, in BOTH directions.

    "What does this agreement amend?" and "which documents amend it?" are
    different questions over the same table, and only the outgoing direction is
    visible from the document's own text — the incoming one exists solely as
    rows written when the OTHER document was ingested.

    Unresolved edges are returned separately rather than dropped: "references a
    Share Purchase Agreement we do not hold" is a true and useful answer, and
    silently omitting it would read as "references nothing".
    """
    from sqlalchemy import text
    with get_engine().connect() as conn:
        out_rows = conn.execute(text("""
            SELECT to_doc, to_doc_raw, label, resolved, confidence, evidence_text
            FROM document_relations
            WHERE wiki_id = :w AND session_id = :sid
              AND (from_doc = :d OR from_doc ILIKE '%' || :d || '%')
        """), {"w": wiki_id, "sid": session_id, "d": source_doc}).fetchall()
        in_rows = conn.execute(text("""
            SELECT from_doc, label, confidence, evidence_text
            FROM document_relations
            WHERE wiki_id = :w AND session_id = :sid
              AND to_doc IS NOT NULL
              AND (to_doc = :d OR to_doc ILIKE '%' || :d || '%')
        """), {"w": wiki_id, "sid": session_id, "d": source_doc}).fetchall()

    outgoing, unresolved = [], []
    for r in out_rows:
        rec = {"doc": r.to_doc, "raw": r.to_doc_raw, "label": r.label,
               "evidence": (r.evidence_text or "")[:300]}
        (outgoing if r.resolved and r.to_doc else unresolved).append(rec)
    incoming = [{"doc": r.from_doc, "label": r.label,
                 "evidence": (r.evidence_text or "")[:300]} for r in in_rows]
    return {"outgoing": outgoing, "incoming": incoming, "unresolved": unresolved}


def find_similar_documents(
    wiki_id: str, session_id: str, source_doc: str, limit: int = 5,
    same_type_only: bool = True, probe: int = 400,
) -> list[dict]:
    """Documents most similar to `source_doc`, for "closest precedent" queries.

    Retrieval elsewhere is question-driven: it embeds the question and finds
    pages. That can never answer "which OTHER documents resemble this one",
    because the question names only one document and every page it retrieves
    belongs to it — which is exactly why those questions came back as "no other
    documents are present for comparison".

    Method: average the target document's page vectors into one centroid, take
    the nearest pages corpus-wide, then aggregate those page hits back up to
    their documents. A document that matches on several pages ranks above one
    that matches on a single stray page, so `pages_matched` is part of the
    score rather than max-similarity alone.

    same_type_only keeps a Disclosure Letter's precedents to other Disclosure
    Letters — a precedent of a different instrument is rarely what is meant.
    """
    from sqlalchemy import text
    engine = get_engine()
    tbl = _emb_table_name()
    with engine.connect() as conn:
        centroid = conn.execute(text(f"""
            SELECT AVG(e.embedding)::text
            FROM {tbl} e
            JOIN pages p ON p.wiki_id = e.wiki_id AND p.session_id = e.session_id
                        AND p.title = e.title
            WHERE e.wiki_id = :w AND e.session_id = :sid AND p.source_doc = :d
              AND e.title NOT LIKE 'Q:%'
        """), {"w": wiki_id, "sid": session_id, "d": source_doc}).scalar()
        if not centroid:
            return []

        doc_type = conn.execute(text("""
            SELECT doc_type FROM documents
            WHERE wiki_id = :w AND source_doc = :d LIMIT 1
        """), {"w": wiki_id, "d": source_doc}).scalar()

        params = {"w": wiki_id, "sid": session_id, "c": centroid,
                  "d": source_doc, "probe": probe, "limit": limit}
        type_join, type_where = "", ""
        if same_type_only and doc_type:
            type_join = ("JOIN documents dd ON dd.wiki_id = p.wiki_id "
                         "AND dd.source_doc = p.source_doc")
            type_where = "AND dd.doc_type = :dt"
            params["dt"] = doc_type

        rows = conn.execute(text(f"""
            WITH hits AS (
                SELECT p.source_doc,
                       1 - (e.embedding <=> CAST(:c AS vector)) AS score
                FROM {tbl} e
                JOIN pages p ON p.wiki_id = e.wiki_id AND p.session_id = e.session_id
                            AND p.title = e.title
                {type_join}
                WHERE e.wiki_id = :w AND e.session_id = :sid
                  AND p.source_doc <> :d
                  AND e.title NOT LIKE 'Q:%'
                  {type_where}
                ORDER BY e.embedding <=> CAST(:c AS vector)
                LIMIT :probe
            )
            SELECT source_doc, MAX(score) AS best, AVG(score) AS mean, COUNT(*) AS n
            FROM hits
            GROUP BY source_doc
            ORDER BY (MAX(score) * 0.6 + AVG(score) * 0.4) DESC, n DESC
            LIMIT :limit
        """), params).fetchall()

    return [{"source_doc": r[0], "best_score": round(float(r[1]), 4),
             "mean_score": round(float(r[2]), 4), "pages_matched": int(r[3]),
             "doc_type": doc_type} for r in rows]


def backfill_embedding_families(wiki_id: str, session_id: str) -> int:
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
                  AND pe.wiki_id = :w
                  AND pe.session_id = :sid
                  AND pm.doc_family IS NOT NULL
                  AND pe.doc_family IS DISTINCT FROM pm.doc_family
            """),
            {"w": wiki_id, "sid": session_id},
        )
        conn.commit()
        return result.rowcount or 0


def count_embeddings(wiki_id: str, session_id: str) -> int:
    """Return the number of pages with embeddings stored for a session."""
    from sqlalchemy import text
    engine = get_engine()
    tbl = _emb_table_name()
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT COUNT(*) FROM {tbl} WHERE wiki_id = :w AND session_id = :sid"),
            {"w": wiki_id, "sid": session_id},
        ).scalar()
        return result or 0


def upsert_page(
    wiki_id: str,
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
                    (wiki_id, session_id, title, content, summary, source_doc,
                     contradiction_flagged, variants, append_count, char_count, last_modified)
                VALUES
                    (:w, :sid, :title, :content, :summary, :source_doc,
                     :cf, CAST(:variants AS jsonb), 1, :char_count, now())
                ON CONFLICT (session_id, title) DO UPDATE SET
                    content               = EXCLUDED.content,
                    summary               = EXCLUDED.summary,
                    source_doc            = EXCLUDED.source_doc,
                    contradiction_flagged = EXCLUDED.contradiction_flagged,
                    variants              = EXCLUDED.variants,
                    append_count          = pages.append_count + 1,
                    char_count            = EXCLUDED.char_count,
                    last_modified         = now(),
                    wiki_id               = EXCLUDED.wiki_id
            """),
            {
                "w": wiki_id,
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

def get_relations(wiki_id: str, session_id: str) -> list[dict]:
    """Return all relations for a session as [{from, to, label}]."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT from_title, to_title, label FROM relations WHERE wiki_id = :w AND session_id = :sid"),
            {"w": wiki_id, "sid": session_id},
        )
        return [{"from": r.from_title, "to": r.to_title, "label": r.label} for r in rows]


def upsert_relation(wiki_id: str, session_id: str, from_title: str, to_title: str, label: str) -> None:
    """Insert a single relation if it doesn't exist."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO relations (wiki_id, session_id, from_title, to_title, label)
                VALUES (:w, :sid, :from_title, :to_title, :label)
                ON CONFLICT (session_id, from_title, to_title, label) DO NOTHING
            """),
            {"w": wiki_id, "sid": session_id, "from_title": from_title, "to_title": to_title, "label": label},
        )
        conn.commit()


def bulk_upsert_relations(wiki_id: str, session_id: str, rels: list[tuple[str, str, str]]) -> None:
    """Batch-insert (from_title, to_title, label) tuples — skips existing rows."""
    if not rels:
        return
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO relations (wiki_id, session_id, from_title, to_title, label)
                VALUES (:w, :sid, :from_title, :to_title, :label)
                ON CONFLICT (session_id, from_title, to_title, label) DO NOTHING
            """),
            [
                {"w": wiki_id, "sid": session_id, "from_title": f, "to_title": t, "label": l}
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

def migrate_from_json(wiki_id: str, session_id: str, json_path: str) -> None:
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
                        (wiki_id, session_id, title, content, summary, source_doc,
                         contradiction_flagged, variants, char_count)
                    VALUES
                        (:w, :sid, :title, :content, :summary, :source_doc,
                         :cf, CAST(:variants AS jsonb), :char_count)
                    ON CONFLICT (session_id, title) DO NOTHING
                """),
                {
                    "w": wiki_id,
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
                    INSERT INTO relations (wiki_id, session_id, from_title, to_title, label)
                    VALUES (:w, :sid, :from_title, :to_title, :label)
                    ON CONFLICT (session_id, from_title, to_title, label) DO NOTHING
                """),
                {
                    "w": wiki_id,
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

def find_pages_mentioning_title(wiki_id: str, session_id: str, title: str) -> list[str]:
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
                WHERE wiki_id = :w AND session_id = :sid
                  AND title      != :title
                  AND content_tsv @@ plainto_tsquery('english', :tokens)
            """),
            {"w": wiki_id, "sid": session_id, "title": title, "tokens": title},
        )
        return [row.title for row in rows]


def find_source_docs_mentioning_phrase(
    wiki_id: str, session_id: str, phrase: str, cap: int = 25
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
                WHERE wiki_id = :w AND session_id = :sid
                  AND title NOT LIKE 'Q:%'
                  AND source_doc IS NOT NULL
                  AND content_tsv @@ phraseto_tsquery('english', :phrase)
                LIMIT :cap
            """),
            {"w": wiki_id, "sid": session_id, "phrase": phrase, "cap": cap},
        )
        return [row.source_doc for row in rows]


def find_source_docs_by_title_tokens(
    wiki_id: str, session_id: str, tokens: list[str], kind_hint: str | None = None,
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
    params.update({"w": wiki_id, "sid": session_id, "cap": cap})
    if kind_hint and kind_hint.strip():
        conds += " AND title ILIKE :kind"
        params["kind"] = f"%{kind_hint.strip()}%"
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT DISTINCT source_doc FROM pages
                WHERE wiki_id = :w AND session_id = :sid
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
    wiki_id: str, session_id: str, append_threshold: int, char_threshold: int
) -> list[dict]:
    """Return pages that exceed the compaction thresholds."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT title, content, summary, variants, append_count, char_count, source_doc
                FROM pages
                WHERE wiki_id = :w AND session_id = :sid
                  AND (
                    append_count >= :at
                    OR (append_count >= 2 AND char_count >= :ct)
                  )
                ORDER BY append_count DESC, char_count DESC
            """),
            {"w": wiki_id, "sid": session_id, "at": append_threshold, "ct": char_threshold},
        )
        return [dict(row._mapping) for row in rows]


def reset_page_after_compaction(
    wiki_id: str,
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
                WHERE wiki_id = :w AND session_id = :sid AND title = :title
            """),
            {
                "w": wiki_id,
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
    wiki_id: str,
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
                    (wiki_id, session_id, page_title, claim, value_a, source_a, value_b, source_b)
                VALUES (:w, :sid, :title, :claim, :va, :sa, :vb, :sb)
            """),
            {
                "w": wiki_id, "sid": session_id, "title": page_title,
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


def upsert_metadata(wiki_id: str, session_id: str, doc_name: str, metadata: dict) -> None:
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
    set_clause += ", wiki_id = :w"
    params = {"w": wiki_id, "sid": session_id, "title": doc_name, **updates}
    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT INTO page_metadata (wiki_id, session_id, title, {', '.join(updates)})
                VALUES (:w, :sid, :title, {', '.join(':' + k for k in updates)})
                ON CONFLICT (session_id, title) DO UPDATE SET {set_clause}
            """),
            params,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Accuracy regression suite — cases, runs, results (§ Phase 3.5a)
# ---------------------------------------------------------------------------

def upsert_regression_case(wiki_id: str, session_id: str, name: str,
                           question: str, **fields) -> int:
    """Create or update one regression case, keyed by (wiki, session, name).

    Upsert rather than insert so re-running a seed loader is idempotent —
    a case set lives in source control and gets re-applied, and duplicating
    every case on each load would silently inflate every later pass rate.
    """
    from sqlalchemy import text
    import json as _json
    cols = {
        "archetype": fields.get("archetype"),
        "expect_scope_method": fields.get("expect_scope_method"),
        "expect_docs": _json.dumps(fields.get("expect_docs")) if fields.get("expect_docs") is not None else None,
        "expect_abstain": bool(fields.get("expect_abstain", False)),
        "must_contain": _json.dumps(fields.get("must_contain")) if fields.get("must_contain") is not None else None,
        "must_not_contain": _json.dumps(fields.get("must_not_contain")) if fields.get("must_not_contain") is not None else None,
        "expect_answer": fields.get("expect_answer"),
        "notes": fields.get("notes"),
        "scope_resolved_by": fields.get("scope_resolved_by"),
        "active": bool(fields.get("active", True)),
    }
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            INSERT INTO regression_cases
                (wiki_id, session_id, name, question, archetype, expect_scope_method,
                 expect_docs, expect_abstain, must_contain, must_not_contain,
                 expect_answer, notes, scope_resolved_by, active)
            VALUES (:w, :sid, :n, :q, :arch, :method, :docs, :abst, :mc, :mnc, :ans, :notes, :srb, :active)
            ON CONFLICT (wiki_id, session_id, name) DO UPDATE SET
                question = EXCLUDED.question,
                archetype = EXCLUDED.archetype,
                expect_scope_method = EXCLUDED.expect_scope_method,
                expect_docs = EXCLUDED.expect_docs,
                expect_abstain = EXCLUDED.expect_abstain,
                must_contain = EXCLUDED.must_contain,
                must_not_contain = EXCLUDED.must_not_contain,
                expect_answer = EXCLUDED.expect_answer,
                notes = EXCLUDED.notes,
                scope_resolved_by = EXCLUDED.scope_resolved_by,
                active = EXCLUDED.active
            RETURNING id
        """), {"w": wiki_id, "sid": session_id, "n": name, "q": question,
               "arch": cols["archetype"], "method": cols["expect_scope_method"],
               "docs": cols["expect_docs"], "abst": cols["expect_abstain"],
               "mc": cols["must_contain"], "mnc": cols["must_not_contain"],
               "ans": cols["expect_answer"], "notes": cols["notes"],
               "srb": cols["scope_resolved_by"],
               "active": cols["active"]}).fetchone()
        conn.commit()
        return int(row[0])


def get_regression_cases(wiki_id: str, session_id: str,
                         archetype: str | None = None,
                         active_only: bool = True) -> list[dict]:
    """All stored cases, oldest first so run output is stably ordered."""
    from sqlalchemy import text
    engine = get_engine()
    sql = """
        SELECT id, name, question, archetype, expect_scope_method, expect_docs,
               expect_abstain, must_contain, must_not_contain, expect_answer,
               notes, active, scope_resolved_by
        FROM regression_cases
        WHERE wiki_id = :w AND session_id = :sid
    """
    params: dict = {"w": wiki_id, "sid": session_id}
    if active_only:
        sql += " AND active"
    if archetype:
        sql += " AND archetype = :arch"
        params["arch"] = archetype
    sql += " ORDER BY id"
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [{"id": int(r[0]), "name": r[1], "question": r[2], "archetype": r[3],
             "expect_scope_method": r[4], "expect_docs": r[5],
             "expect_abstain": r[6], "must_contain": r[7],
             "must_not_contain": r[8], "expect_answer": r[9],
             "notes": r[10], "active": r[11],
             "scope_resolved_by": r[12]} for r in rows]


def delete_regression_case(wiki_id: str, session_id: str, name: str) -> bool:
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        n = conn.execute(text("""
            DELETE FROM regression_cases
            WHERE wiki_id = :w AND session_id = :sid AND name = :n
        """), {"w": wiki_id, "sid": session_id, "n": name}).rowcount
        conn.commit()
    return bool(n)


def start_regression_run(wiki_id: str, session_id: str, tier: str,
                         label: str | None = None, git_sha: str | None = None,
                         cases_total: int = 0) -> int:
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            INSERT INTO regression_runs (wiki_id, session_id, tier, label, git_sha, cases_total)
            VALUES (:w, :sid, :t, :l, :g, :n) RETURNING id
        """), {"w": wiki_id, "sid": session_id, "t": tier, "l": label,
               "g": git_sha, "n": cases_total}).fetchone()
        conn.commit()
        return int(row[0])


def record_regression_result(run_id: int, case: dict, passed: bool,
                             failures: list[str], **actual) -> None:
    from sqlalchemy import text
    import json as _json
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO regression_results
                (run_id, case_id, case_name, passed, failures, actual_scope_method,
                 actual_docs, answer, scores, total_ms, trace_id)
            VALUES (:r, :c, :n, :p, :f, :m, :d, :a, :s, :ms, :t)
        """), {"r": run_id, "c": case.get("id"), "n": case.get("name") or "?",
               "p": passed, "f": _json.dumps(failures or []),
               "m": actual.get("scope_method"),
               "d": _json.dumps(actual.get("docs")) if actual.get("docs") is not None else None,
               "a": actual.get("answer"),
               "s": _json.dumps(actual.get("scores")) if actual.get("scores") is not None else None,
               "ms": actual.get("total_ms"), "t": actual.get("trace_id")})
        conn.commit()


def finish_regression_run(run_id: int, passed: int, failed: int,
                          status: str = "complete", error: str | None = None) -> None:
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE regression_runs
               SET cases_passed = :p, cases_failed = :f, status = :s,
                   error = :e, finished_at = now()
             WHERE id = :id
        """), {"p": passed, "f": failed, "s": status, "e": error, "id": run_id})
        conn.commit()


def get_regression_runs(wiki_id: str, session_id: str, tier: str | None = None,
                        limit: int = 20) -> list[dict]:
    from sqlalchemy import text
    engine = get_engine()
    sql = """
        SELECT id, tier, label, git_sha, status, cases_total, cases_passed,
               cases_failed, error, started_at, finished_at
        FROM regression_runs WHERE wiki_id = :w AND session_id = :sid
    """
    params: dict = {"w": wiki_id, "sid": session_id, "lim": limit}
    if tier:
        sql += " AND tier = :t"
        params["t"] = tier
    sql += " ORDER BY started_at DESC LIMIT :lim"
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [{"id": int(r[0]), "tier": r[1], "label": r[2], "git_sha": r[3],
             "status": r[4], "cases_total": r[5], "cases_passed": r[6],
             "cases_failed": r[7], "error": r[8],
             "started_at": r[9].isoformat() if r[9] else None,
             "finished_at": r[10].isoformat() if r[10] else None} for r in rows]


def get_regression_results(run_id: int) -> list[dict]:
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT case_name, passed, failures, actual_scope_method, actual_docs,
                   answer, scores, total_ms, trace_id
            FROM regression_results WHERE run_id = :r ORDER BY id
        """), {"r": run_id}).fetchall()
    return [{"case_name": r[0], "passed": r[1], "failures": r[2],
             "actual_scope_method": r[3], "actual_docs": r[4], "answer": r[5],
             "scores": r[6], "total_ms": r[7], "trace_id": r[8]} for r in rows]


def compare_regression_runs(run_a: int, run_b: int) -> dict:
    """Case-level diff between two runs — the point of storing runs at all.

    Reports which cases newly fail in B (regressions), which newly pass
    (fixes), and which changed scope resolution without changing pass/fail
    (a silent behaviour change worth seeing before it becomes a bug).
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT COALESCE(a.case_name, b.case_name) AS name,
                   a.passed, b.passed, a.actual_scope_method, b.actual_scope_method
            FROM (SELECT * FROM regression_results WHERE run_id = :a) a
            FULL OUTER JOIN (SELECT * FROM regression_results WHERE run_id = :b) b
              ON a.case_name = b.case_name
            ORDER BY 1
        """), {"a": run_a, "b": run_b}).fetchall()
    out = {"regressions": [], "fixes": [], "scope_changed": [],
           "unchanged": 0, "only_in_a": [], "only_in_b": []}
    for name, pa, pb, ma, mb in rows:
        if pa is None:
            out["only_in_b"].append(name)
            continue
        if pb is None:
            out["only_in_a"].append(name)
            continue
        if pa and not pb:
            out["regressions"].append({"case": name, "scope_was": ma, "scope_now": mb})
        elif pb and not pa:
            out["fixes"].append({"case": name, "scope_was": ma, "scope_now": mb})
        else:
            if ma != mb:
                out["scope_changed"].append({"case": name, "was": ma, "now": mb})
            else:
                out["unchanged"] += 1
    return out


def _init_clause_value_columns(conn, text) -> None:
    """Normalised money on `clauses` (§ Phase 4).

    Same status-beside-value contract as contracts/obligations: a clause whose
    typed_value holds a figure gets an amount, and one that does not gets an
    explicit status rather than a NULL that an aggregate would silently skip
    without saying so.
    """
    conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS value_amount NUMERIC"))
    conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS value_currency TEXT"))
    conn.execute(text("ALTER TABLE clauses ADD COLUMN IF NOT EXISTS value_status TEXT"))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS clauses_value_idx
        ON clauses (wiki_id, session_id, clause_type_canon, value_amount)
        WHERE value_amount IS NOT NULL
    """))


def backfill_clause_values(wiki_id: str, session_id: str | None = None,
                           dry_run: bool = False) -> dict:
    """Parse clauses.typed_value into comparable money columns.

    The corpus stores the same concept under several invented key names —
    {'total': 'Rs. 9,284,268,071'}, {'total_value': 'Rs. 4,085,376,476'} — so
    this looks for any of them rather than one canonical key, and falls back to
    the clause's own verbatim text when typed_value carries no figure.
    """
    from sqlalchemy import text
    from services import normalize
    import json as _json
    engine = get_engine()
    where = "wiki_id = :w" + (" AND session_id = :sid" if session_id else "")
    params: dict = {"w": wiki_id}
    if session_id:
        params["sid"] = session_id

    # Keys the extraction has actually used for a monetary total, in priority
    # order. Anything else falls through to the verbatim text.
    money_keys = ("total", "total_value", "value", "amount", "consideration",
                  "contract_value", "price", "payment_value")

    with engine.connect() as conn:
        _init_clause_value_columns(conn, text)
        conn.commit()
        rows = conn.execute(text(
            f"SELECT id, typed_value, verbatim_text FROM clauses WHERE {where}"
            f" AND clause_type_canon IN ('contract_value', 'fees', 'liability_cap')"
        ), params).fetchall()

        stats: dict[str, int] = {}
        for cid, typed, verbatim in rows:
            raw = None
            if isinstance(typed, dict):
                for k in money_keys:
                    if typed.get(k) is not None:
                        raw = typed[k]
                        break
            elif isinstance(typed, str) and typed.strip().startswith("{"):
                try:
                    d = _json.loads(typed)
                    for k in money_keys:
                        if d.get(k) is not None:
                            raw = d[k]
                            break
                except Exception:
                    pass
            if raw is None:
                raw = verbatim
            parsed = normalize.parse_money(raw)
            stats[parsed["status"]] = stats.get(parsed["status"], 0) + 1
            if not dry_run:
                conn.execute(text("""
                    UPDATE clauses SET value_amount = :a, value_currency = :c,
                                       value_status = :s
                     WHERE id = :id
                """), {"a": parsed["amount"], "c": parsed["currency"],
                       "s": parsed["status"], "id": cid})
        if not dry_run:
            conn.commit()
    return {"dry_run": dry_run, "rows": len(rows), "by_status": stats}


def _init_normalized_columns(conn, text) -> None:
    """Normalised value columns (§ Phase 3.5c), added beside the raw fields.

    Every value column has a matching *_status column. That pairing is the
    whole design: a NULL amount with status 'reference' means the figure lives
    in a schedule, 'unparsed' means we could not read it, and 'absent' means
    the field was empty. Gap detection must be able to tell those apart —
    collapsing them into a bare NULL is what turns "we could not read this cap"
    into "this contract has no cap".
    """
    conn.execute(text("ALTER TABLE contracts ADD COLUMN IF NOT EXISTS liability_cap_amount NUMERIC"))
    conn.execute(text("ALTER TABLE contracts ADD COLUMN IF NOT EXISTS liability_cap_currency TEXT"))
    conn.execute(text("ALTER TABLE contracts ADD COLUMN IF NOT EXISTS liability_cap_status TEXT"))
    conn.execute(text("ALTER TABLE obligations ADD COLUMN IF NOT EXISTS deadline_days NUMERIC"))
    conn.execute(text("ALTER TABLE obligations ADD COLUMN IF NOT EXISTS deadline_business_days BOOLEAN"))
    conn.execute(text("ALTER TABLE obligations ADD COLUMN IF NOT EXISTS deadline_status TEXT"))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS contracts_cap_amount_idx
        ON contracts (wiki_id, liability_cap_amount)
        WHERE liability_cap_amount IS NOT NULL
    """))


def backfill_normalized_values(wiki_id: str, session_id: str | None = None,
                               dry_run: bool = False) -> dict:
    """Parse existing raw values into the normalised columns.

    Deterministic, no model call, re-runnable after any parser change.
    """
    from sqlalchemy import text
    from services import normalize
    engine = get_engine()
    out: dict = {"dry_run": dry_run}

    with engine.connect() as conn:
        _init_normalized_columns(conn, text)
        conn.commit()

        cap_where = "wiki_id = :w" + (" AND session_id = :sid" if session_id else "")
        params: dict = {"w": wiki_id}
        if session_id:
            params["sid"] = session_id

        caps = conn.execute(text(
            f"SELECT id, liability_cap FROM contracts WHERE {cap_where}"), params).fetchall()
        cap_stats: dict[str, int] = {}
        for cid, raw in caps:
            parsed = normalize.parse_money(raw)
            cap_stats[parsed["status"]] = cap_stats.get(parsed["status"], 0) + 1
            if not dry_run:
                conn.execute(text("""
                    UPDATE contracts
                       SET liability_cap_amount = :a, liability_cap_currency = :c,
                           liability_cap_status = :s
                     WHERE id = :id
                """), {"a": parsed["amount"], "c": parsed["currency"],
                       "s": parsed["status"], "id": cid})

        obls = conn.execute(text(
            f"SELECT id, deadline FROM obligations WHERE {cap_where}"), params).fetchall()
        obl_stats: dict[str, int] = {}
        for oid, raw in obls:
            parsed = normalize.parse_duration(raw)
            obl_stats[parsed["status"]] = obl_stats.get(parsed["status"], 0) + 1
            if not dry_run:
                conn.execute(text("""
                    UPDATE obligations
                       SET deadline_days = :d, deadline_business_days = :b,
                           deadline_status = :s
                     WHERE id = :id
                """), {"d": parsed["days"], "b": parsed["business_days"],
                       "s": parsed["status"], "id": oid})
        if not dry_run:
            conn.commit()

    out["liability_caps"] = {"rows": len(caps), "by_status": cap_stats}
    out["obligations"] = {"rows": len(obls), "by_status": obl_stats}
    return out


def backfill_clause_type_canon(wiki_id: str, session_id: str | None = None,
                               dry_run: bool = False) -> dict:
    """Populate clauses.clause_type_canon from the raw clause_type.

    Deterministic string mapping — no LLM call, no embedding, so this is free
    and safe to re-run after every vocabulary change. Updates by distinct raw
    label rather than row by row: 31,457 rows carry only ~6,000 distinct
    labels, so this is a few thousand statements instead of thirty thousand.

    Rows whose label does not map are set to NULL explicitly rather than left
    at whatever they held before, so re-running after a vocabulary change can
    take a mapping away as well as add one.
    """
    from sqlalchemy import text
    from services import clause_vocab
    engine = get_engine()
    where = "wiki_id = :w" + (" AND session_id = :sid" if session_id else "")
    params: dict = {"w": wiki_id}
    if session_id:
        params["sid"] = session_id

    with engine.connect() as conn:
        labels = [r[0] for r in conn.execute(text(
            f"SELECT DISTINCT clause_type FROM clauses WHERE {where}"), params)]
        mapping = clause_vocab.classify_all([l for l in labels if l])
        mapped = {k: v for k, v in mapping.items() if v}
        if dry_run:
            rows = conn.execute(text(
                f"SELECT count(*) FROM clauses WHERE {where} "
                f"AND clause_type = ANY(:labels)"),
                {**params, "labels": list(mapped.keys())}).scalar()
            return {"dry_run": True, "labels_total": len(labels),
                    "labels_mapped": len(mapped), "rows_would_map": int(rows or 0)}

        updated = 0
        for raw, canon in mapped.items():
            updated += conn.execute(text(
                f"UPDATE clauses SET clause_type_canon = :c "
                f"WHERE {where} AND clause_type = :raw "
                f"AND clause_type_canon IS DISTINCT FROM :c"),
                {**params, "c": canon, "raw": raw}).rowcount or 0
        unmapped_labels = [l for l in labels if l and not mapping.get(l)]
        cleared = 0
        if unmapped_labels:
            cleared = conn.execute(text(
                f"UPDATE clauses SET clause_type_canon = NULL "
                f"WHERE {where} AND clause_type = ANY(:labels) "
                f"AND clause_type_canon IS NOT NULL"),
                {**params, "labels": unmapped_labels}).rowcount or 0
        conn.commit()

        total, with_canon = conn.execute(text(
            f"SELECT count(*), count(clause_type_canon) FROM clauses WHERE {where}"),
            params).fetchone()

    return {"dry_run": False, "labels_total": len(labels), "labels_mapped": len(mapped),
            "rows_updated": updated, "rows_cleared": cleared,
            "rows_total": int(total), "rows_with_canon": int(with_canon),
            "coverage": round(int(with_canon) / max(int(total), 1), 4)}


def clause_canon_summary(wiki_id: str, session_id: str | None = None) -> dict:
    """Canonical-type distribution, plus the biggest unmapped raw labels.

    The unmapped list is the vocabulary's own to-do list — it is ordered by how
    many clauses are affected, so extending the canon can be driven by volume
    rather than by guesswork.
    """
    from sqlalchemy import text
    engine = get_engine()
    where = "wiki_id = :w" + (" AND session_id = :sid" if session_id else "")
    params: dict = {"w": wiki_id}
    if session_id:
        params["sid"] = session_id
    with engine.connect() as conn:
        by_canon = conn.execute(text(f"""
            SELECT clause_type_canon, count(*) FROM clauses
            WHERE {where} AND clause_type_canon IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """), params).fetchall()
        unmapped = conn.execute(text(f"""
            SELECT clause_type, count(*) FROM clauses
            WHERE {where} AND clause_type_canon IS NULL
            GROUP BY 1 ORDER BY 2 DESC LIMIT 30
        """), params).fetchall()
        total, with_canon = conn.execute(text(
            f"SELECT count(*), count(clause_type_canon) FROM clauses WHERE {where}"),
            params).fetchone()
    return {
        "rows_total": int(total), "rows_mapped": int(with_canon),
        "coverage": round(int(with_canon) / max(int(total), 1), 4),
        "by_canon": [{"canon": r[0], "count": int(r[1])} for r in by_canon],
        "top_unmapped": [{"clause_type": r[0], "count": int(r[1])} for r in unmapped],
    }


# documents.effective_date is TEXT, and on this corpus it holds three shapes:
# 1,125 rows in ISO form, 76 in prose the extractor copied verbatim from the
# document ("Date: 01 June 2025", "1ST August 2022", "1st day of February 2026
# (\u201cEffective Date\u201d)", "23.09.2025"), and 178 nulls. Sorted as text,
# "Date: 18 July 2025" lands above every 2026 date, because 'D' sorts after '2'
# — which is exactly how a "most recent first" list ended up leading with a
# 2025 document.
#
# The column is deliberately NOT rewritten. The raw string is what the document
# says and what an answer cites; parsing happens for ordering only, and a value
# this cannot parse sorts last rather than being guessed at.
_RX_DATE_ISO = re.compile(r"^\s*(\d{4})-(\d{1,2})(?:-(\d{1,2}))?\s*$")
_RX_DATE_YEAR = re.compile(r"^\s*(\d{4})\s*$")
_RX_DATE_DOTTED = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_RX_DATE_PROSE = re.compile(
    r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:day\s+of\s+)?"
    r"([A-Za-z]{3,9})\.?\s+(\d{4})\b", re.IGNORECASE)
_RX_DATE_PROSE_MY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b", re.IGNORECASE)

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_effective_date(raw):
    """A ``datetime.date`` for ordering, or None when the text cannot be read.

    None is a real answer here, not a failure: an unreadable date sorts to the
    end of a "most recent first" list, which is where an undated document
    belongs. Inventing a date to make the sort tidy would put a document in a
    position the corpus does not support.
    """
    import datetime as _dt
    if not raw:
        return None
    text_val = str(raw).strip()
    if not text_val:
        return None

    def _safe(y, m, d):
        try:
            return _dt.date(int(y), int(m), int(d))
        except (ValueError, TypeError):
            return None

    m = _RX_DATE_ISO.match(text_val)
    if m:
        return _safe(m.group(1), m.group(2), m.group(3) or 1)
    m = _RX_DATE_YEAR.match(text_val)
    if m:
        return _safe(m.group(1), 1, 1)
    m = _RX_DATE_DOTTED.search(text_val)
    if m:
        # Day-first: every dotted value on this corpus is Indian convention.
        return _safe(m.group(3), m.group(2), m.group(1))
    m = _RX_DATE_PROSE.search(text_val)
    if m:
        month = _MONTHS.get(m.group(2)[:3].lower())
        if month:
            return _safe(m.group(3), month, m.group(1))
    # Month and year with no day — the first of that month orders correctly
    # against everything else without claiming a day the document never gave.
    m = _RX_DATE_PROSE_MY.search(text_val)
    if m:
        month = _MONTHS.get(m.group(1)[:3].lower())
        if month:
            return _safe(m.group(2), month, 1)
    return None


# A date embedded in a filename: "..._LegaOpin_20250912", "Amdt_2021-12-31",
# "LegaOpin 17-04-2023", "MAT-2025-3539_...06-11-2024".
_FILENAME_DATE_PATTERNS = (
    (re.compile(r"(20\d{2})(\d{2})(\d{2})"), "ymd"),
    (re.compile(r"(20\d{2})-(\d{2})-(\d{2})"), "ymd"),
    (re.compile(r"(\d{2})-(\d{2})-(20\d{2})"), "dmy"),
    (re.compile(r"(\d{2})\.(\d{2})\.(20\d{2})"), "dmy"),
    (re.compile(r"(\d{2})_(\d{2})_(20\d{2})"), "dmy"),
)


def date_from_filename(source_doc: str):
    """A ``datetime.date`` read out of a document's filename, or None.

    Ingest leaves ``documents.effective_date`` empty on 174 of 1,372 documents,
    and 31 of those carry the date plainly in their own filename
    ("National Council for_LegaOpin_20250912"). A question that recites that
    date then cannot resolve to the document, because the only date the
    resolver can compare against is the empty column.

    Read rather than inferred: this parses a string the ingest pipeline already
    produced, so it costs nothing and cannot invent a date the corpus does not
    already carry. A filename with no parseable date returns None and the
    column stays empty, which is the honest state.
    """
    import datetime as _dt
    base = (source_doc or "").split("_", 1)[-1]
    for rx, order in _FILENAME_DATE_PATTERNS:
        m = rx.search(base)
        if not m:
            continue
        a, b, c = m.groups()
        y, mo, d = (a, b, c) if order == "ymd" else (c, b, a)
        try:
            got = _dt.date(int(y), int(mo), int(d))
        except (ValueError, TypeError):
            continue
        # A filename can carry a matter year ("MAT-2025-3539") that is not a
        # date; a full valid calendar date is the signal that this is one.
        if 2000 <= got.year <= 2100:
            return got
    return None


def backfill_effective_dates_from_filenames(wiki_id: str, session_id: str,
                                            dry_run: bool = True) -> dict:
    """Fill an empty effective_date from the document's own filename.

    Only ever writes where the column is empty or unparseable — a date the
    extractor read out of the document itself is better evidence than a
    filename and is never overwritten.
    """
    from sqlalchemy import text as _sql
    engine = get_engine()
    filled, examined = [], 0
    with engine.connect() as conn:
        rows = conn.execute(_sql(
            "SELECT source_doc, effective_date FROM documents "
            "WHERE wiki_id = :w AND session_id = :s"
        ), {"w": wiki_id, "s": session_id}).fetchall()
        for sd, ed in rows:
            examined += 1
            if parse_effective_date(ed):
                continue
            got = date_from_filename(sd)
            if not got:
                continue
            filled.append({"source_doc": sd, "date": got.isoformat()})
            if not dry_run:
                conn.execute(_sql(
                    "UPDATE documents SET effective_date = :d "
                    "WHERE wiki_id = :w AND session_id = :s AND source_doc = :sd"
                ), {"d": got.isoformat(), "w": wiki_id, "s": session_id, "sd": sd})
        if not dry_run:
            conn.commit()
    logger.info("effective_date backfill: %d of %d documents %s",
                len(filled), examined, "would be filled" if dry_run else "filled")
    return {"examined": examined, "filled": len(filled),
            "dry_run": dry_run, "documents": filled}


def find_documents_by_effective_date(wiki_id: str, session_id: str,
                                     iso_date: str, cap: int = 25) -> list[str]:
    """Documents whose stored effective_date is this day.

    Compared on the PARSED value, not the stored string: the column is TEXT and
    holds ISO dates, prose dates and dotted dates side by side, so a literal
    comparison would match one shape and miss the rest.
    """
    from sqlalchemy import text as _sql
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_sql(
            "SELECT source_doc, effective_date FROM documents "
            "WHERE wiki_id = :w AND session_id = :s AND effective_date IS NOT NULL"
        ), {"w": wiki_id, "s": session_id}).fetchall()
    out = []
    for sd, ed in rows:
        got = parse_effective_date(ed)
        if got and got.isoformat() == iso_date:
            out.append(sd)
            if len(out) >= cap:
                break
    return out


def find_defined_term(wiki_id: str, session_id: str, source_docs: list[str],
                      term: str) -> list[dict]:
    """The stored definition of one term in one or more documents.

    defined_terms holds 2,537 rows across 602 documents and nothing read it.
    Measured on the 200-question audit, twelve of the twenty-two "how is the
    term X defined" questions already had their answer sitting in this table,
    quoted verbatim from the document — eleven were being re-derived by the
    answer model at ten to twenty thousand tokens each, and one was answered
    "the Definitions section includes 'Applicable Law' as a defined term"
    without ever giving the definition.
    """
    from sqlalchemy import text
    if not source_docs or not term:
        return []
    engine = get_engine()
    params: dict = {"w": wiki_id, "s": session_id, "t": term.strip()}
    doc_clauses = []
    for i, sd in enumerate(source_docs[:6]):
        doc_clauses.append(f"source_doc = :d{i}")
        params[f"d{i}"] = sd
    where_docs = " OR ".join(doc_clauses)
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT source_doc, term, definition, page_num
            FROM defined_terms
            WHERE wiki_id = :w AND session_id = :s
              AND ({where_docs})
              AND (lower(btrim(term, ' "')) = lower(btrim(:t, ' "'))
                   OR term ILIKE '%' || :t || '%')
            ORDER BY length(term)
            LIMIT 6
        """), params).fetchall()
    return [{"source_doc": r[0], "term": r[1], "definition": r[2],
             "page_num": r[3]} for r in rows]


def count_documents_of_type_without_parties(wiki_id: str, session_id: str,
                                            doc_type_patterns: list) -> int:
    """How many documents of this type have NO indexed parties at all.

    A "this party has no such instrument" answer is a claim about the whole
    corpus, and it can only be made from an index that covers the corpus. A
    document whose parties row is NULL is invisible to that count, so it cannot
    be used to contradict the claim - and the claim gets made anyway. Measured
    live: a Framework Supply Agreement with 25 pages and a NULL parties row was
    reported as not existing, because the only documents the index could see
    for that party were of other types.
    """
    if not doc_type_patterns:
        return 0
    from sqlalchemy import text
    clauses = " OR ".join("doc_type ILIKE :p%d" % i for i in range(len(doc_type_patterns)))
    params = {"w": wiki_id, "s": session_id}
    for i, pat in enumerate(doc_type_patterns):
        params["p%d" % i] = pat
    with get_engine().connect() as conn:
        return conn.execute(text(
            "SELECT count(*) FROM documents WHERE wiki_id = :w "
            "AND (parties IS NULL OR parties::text IN ('', '[]', 'null')) "
            "AND (%s)" % clauses), params).scalar() or 0


def count_documents_by_party(wiki_id: str, session_id: str,
                             parties: list[str] | None = None,
                             doc_type_hint: str | None = None,
                             limit: int = 25,
                             doc_type_patterns: list[str] | None = None) -> dict:
    """Count documents matching a party (or every party pair member) and type.

    Reads documents.parties, a clean JSONB string array populated on the large
    majority of the corpus, so this is an exact count rather than an estimate
    inferred from whatever retrieval happened to return. That distinction is
    the entire point: asked "how many contracts do we have with X", a
    retrieval-based answer reports how many documents it managed to fetch,
    which on a corpus of 101 is a confidently wrong small number.

    With two or more parties, counts documents naming ALL of them — "contracts
    with X and Y" means the agreements between them, not the union.

    Returns the matched documents too (capped), because a bare number a lawyer
    cannot audit is not usable as an answer.
    """
    from sqlalchemy import text
    engine = get_engine()
    clauses = ["d.wiki_id = :w", "d.session_id = :sid"]
    # Bounded so a whole-corpus count can never pull unbounded rows into
    # memory just to order the handful that get shown.
    _ROW_FETCH_CAP = 2000
    params: dict = {"w": wiki_id, "sid": session_id, "lim": limit,
                    "cap": _ROW_FETCH_CAP}

    for i, party in enumerate(parties or []):
        # Party match is substring and case-insensitive against the array
        # elements: the corpus stores full legal names ("Tata Power Renewable
        # Energy Limited") and a lawyer asks for "Tata Power".
        clauses.append(f"""EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(COALESCE(d.parties, '[]'::jsonb)) AS p(name)
            WHERE p.name ILIKE :party{i}
        )""")
        params[f"party{i}"] = f"%{party.strip()}%"

    # A lawyer's word for an instrument almost never equals the stored
    # doc_type. "NDA" is filed on this corpus as "Mutual Confidentiality and
    # Non-Disclosure Agreement", as "Non-Disclosure Agreement", and as both
    # again in upper case - 176 distinct doc_type strings for far fewer
    # actual instrument types. Counting a single ILIKE would report 19 NDAs
    # where there are 145, which is the exact failure this whole function
    # exists to prevent. The caller resolves one concept to every spelling
    # that names it and they are OR'd here, so one concept counts once.
    _patterns = [p.strip() for p in (doc_type_patterns or []) if p and p.strip()]
    if not _patterns and doc_type_hint:
        _patterns = [doc_type_hint.strip()]
    if _patterns:
        _ors = []
        for i, pat in enumerate(_patterns):
            _ors.append(f"d.doc_type ILIKE :dt{i}")
            params[f"dt{i}"] = f"%{pat}%"
        clauses.append("(" + " OR ".join(_ors) + ")")

    where = " AND ".join(clauses)
    with engine.connect() as conn:
        total = conn.execute(text(
            f"SELECT count(*) FROM documents d WHERE {where}"), params).scalar()
        # Ordered in Python, not SQL: effective_date is TEXT holding three
        # different shapes, so ORDER BY sorts it lexically and puts prose dates
        # above every ISO one. Fetching under a hard cap and sorting the parsed
        # value is both correct and cheap — the largest realistic result here is
        # one instrument type across the corpus, a few hundred narrow rows.
        rows = conn.execute(text(f"""
            SELECT d.source_doc, d.doc_type, d.effective_date, d.parties
            FROM documents d WHERE {where}
            ORDER BY d.source_doc
            LIMIT :cap
        """), params).fetchall()
        by_type = conn.execute(text(f"""
            SELECT COALESCE(NULLIF(d.doc_type, ''), 'Unclassified') AS t, count(*)
            FROM documents d WHERE {where}
            GROUP BY 1 ORDER BY 2 DESC LIMIT 12
        """), params).fetchall()
    import datetime as _dt
    _ordered = sorted(
        rows,
        key=lambda r: (parse_effective_date(r[2]) or _dt.date.min, r[0]),
        reverse=True)
    _shown = [{"source_doc": r[0], "doc_type": r[1],
               "effective_date": str(r[2]) if r[2] else None,
               "parties": r[3]} for r in _ordered[:limit]]
    return {
        "total": int(total or 0),
        "parties": parties or [],
        "doc_type": doc_type_hint,
        "doc_type_patterns": _patterns,
        "by_type": [{"doc_type": r[0], "count": int(r[1])} for r in by_type],
        "documents": _shown,
        "truncated": int(total or 0) > len(_shown),
    }


def upsert_page_quality(wiki_id: str, session_id: str, source_doc: str,
                        page_quality: list[dict]) -> int:
    """Store per-page extraction provenance for one document.

    Replaced wholesale per document rather than appended: a re-ingest that
    produced better text must not leave the old page's "OCR failed" row behind
    to warn about a problem that no longer exists.
    """
    if not page_quality:
        return 0
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            DELETE FROM page_quality
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = :d
        """), {"w": wiki_id, "sid": session_id, "d": source_doc})
        n = 0
        for pq in page_quality:
            conn.execute(text("""
                INSERT INTO page_quality
                    (wiki_id, session_id, source_doc, page_num, extraction_method,
                     ocr_engine, char_count, needed_ocr, below_floor)
                VALUES (:w, :sid, :d, :p, :m, :e, :c, :n, :b)
                ON CONFLICT (wiki_id, session_id, source_doc, page_num) DO UPDATE SET
                    extraction_method = EXCLUDED.extraction_method,
                    ocr_engine = EXCLUDED.ocr_engine,
                    char_count = EXCLUDED.char_count,
                    needed_ocr = EXCLUDED.needed_ocr,
                    below_floor = EXCLUDED.below_floor
            """), {"w": wiki_id, "sid": session_id, "d": source_doc,
                   "p": pq.get("page_num"), "m": pq.get("extraction_method") or "unknown",
                   "e": pq.get("ocr_engine"), "c": pq.get("char_count") or 0,
                   "n": bool(pq.get("needed_ocr")), "b": bool(pq.get("below_floor"))})
            n += 1
        conn.commit()
    return n


def get_document_quality(wiki_id: str, session_id: str,
                         source_docs: list[str]) -> dict[str, dict]:
    """Extraction-quality summary per document, for the reader-facing warning.

    Returns only documents that actually have a problem — a caller iterating
    the result to build a warning should get nothing back for a clean corpus
    rather than having to filter. Documents ingested before page_quality
    existed simply have no rows and are absent, which reads correctly: the
    warning says what is known to be bad, never asserts that silence is good.
    """
    if not source_docs:
        return {}
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT source_doc,
                   count(*)                                  AS pages,
                   count(*) FILTER (WHERE below_floor)       AS unreadable,
                   count(*) FILTER (WHERE needed_ocr)        AS ocr_pages,
                   array_agg(page_num ORDER BY page_num) FILTER (WHERE below_floor) AS bad_pages
            FROM page_quality
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = ANY(:docs)
            GROUP BY source_doc
            HAVING count(*) FILTER (WHERE below_floor) > 0
        """), {"w": wiki_id, "sid": session_id, "docs": list(source_docs)}).fetchall()
    return {r[0]: {"pages": int(r[1]), "unreadable_pages": int(r[2]),
                   "ocr_pages": int(r[3]), "bad_page_numbers": list(r[4] or [])}
            for r in rows}


def corpus_quality_summary(wiki_id: str, session_id: str, limit: int = 100) -> dict:
    """Admin-side view: which documents have unreadable pages, worst first."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        totals = conn.execute(text("""
            SELECT count(DISTINCT source_doc), count(*),
                   count(*) FILTER (WHERE below_floor),
                   count(*) FILTER (WHERE needed_ocr)
            FROM page_quality WHERE wiki_id = :w AND session_id = :sid
        """), {"w": wiki_id, "sid": session_id}).fetchone()
        rows = conn.execute(text("""
            SELECT source_doc, count(*) AS pages,
                   count(*) FILTER (WHERE below_floor) AS unreadable,
                   count(*) FILTER (WHERE needed_ocr)  AS ocr_pages
            FROM page_quality
            WHERE wiki_id = :w AND session_id = :sid
            GROUP BY source_doc
            HAVING count(*) FILTER (WHERE below_floor) > 0
            ORDER BY (count(*) FILTER (WHERE below_floor))::float / GREATEST(count(*),1) DESC,
                     unreadable DESC
            LIMIT :lim
        """), {"w": wiki_id, "sid": session_id, "lim": limit}).fetchall()
    return {
        "documents_tracked": int(totals[0] or 0),
        "pages_tracked": int(totals[1] or 0),
        "pages_unreadable": int(totals[2] or 0),
        "pages_needing_ocr": int(totals[3] or 0),
        "documents": [{"source_doc": r[0], "pages": int(r[1]),
                       "unreadable_pages": int(r[2]), "ocr_pages": int(r[3]),
                       "unreadable_share": round(int(r[2]) / max(int(r[1]), 1), 3)}
                      for r in rows],
    }


def find_duplicate_documents(wiki_id: str) -> list[dict]:
    """Groups of documents sharing a file_hash — byte-identical uploads.

    wiki.ingest already refuses a duplicate at upload time via
    backbone.find_by_file_hash, and that check is correct, but it is a
    read-then-write race: the check runs before extraction and the
    `documents` row is not written until extraction finishes, tens of
    seconds later. Two identical files ingested inside that window both see
    an empty result and both insert. Every duplicate pair in this corpus
    arrived that way — same bulk run, timestamps seconds apart.

    Reports extraction richness per copy so a caller can keep the better one
    rather than an arbitrary one. The files being byte-identical, differences
    between copies are LLM extraction variance over the same bytes, not one
    copy being more complete than the other in any meaningful sense.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.file_hash, d.session_id, d.source_doc, d.created_at,
                   (SELECT count(*) FROM pages p WHERE p.source_doc = d.source_doc) AS pages,
                   (SELECT count(*) FROM clauses c WHERE c.source_doc = d.source_doc) AS clauses,
                   (SELECT count(*) FROM obligations o WHERE o.source_doc = d.source_doc) AS obligations
            FROM documents d
            WHERE d.wiki_id = :w AND d.file_hash IS NOT NULL
              AND d.file_hash IN (
                  SELECT file_hash FROM documents
                  WHERE wiki_id = :w AND file_hash IS NOT NULL
                  GROUP BY file_hash HAVING count(*) > 1)
            ORDER BY d.file_hash, d.created_at
        """), {"w": wiki_id}).fetchall()

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[0], []).append({
            "session_id": r[1], "source_doc": r[2],
            "created_at": r[3].isoformat() if r[3] else None,
            "pages": int(r[4]), "clauses": int(r[5]), "obligations": int(r[6]),
            "richness": int(r[4]) + int(r[5]) + int(r[6]),
        })
    out = []
    for file_hash, copies in groups.items():
        # Keep the richest copy; tie-break on oldest, so the choice is
        # deterministic and re-running the report never proposes a different
        # winner for the same data.
        ranked = sorted(copies, key=lambda c: (-c["richness"], c["created_at"] or ""))
        out.append({"file_hash": file_hash, "copies": copies,
                    "keep": ranked[0]["source_doc"],
                    "remove": [c["source_doc"] for c in ranked[1:]]})
    return out


def enforce_file_hash_uniqueness(wiki_id: str) -> dict:
    """Add the unique index that closes the ingest race, if the data allows.

    The application-level check cannot close this on its own — any check that
    reads before a slow write has a window. A partial unique index on
    (wiki_id, file_hash) makes the second insert fail regardless of timing,
    which is the only version of this guarantee that holds under concurrency.

    Refuses rather than forces when duplicates still exist: creating the index
    would fail anyway, and reporting which groups block it is more useful than
    a raw Postgres error.
    """
    dupes = find_duplicate_documents(wiki_id)
    if dupes:
        return {"created": False, "blocked_by": len(dupes),
                "groups": [{"keep": g["keep"], "remove": g["remove"]} for g in dupes]}
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS documents_wiki_file_hash_uniq
            ON documents (wiki_id, file_hash)
            WHERE file_hash IS NOT NULL
        """))
        conn.commit()
    return {"created": True, "blocked_by": 0, "groups": []}


def latency_percentiles(wiki_session_id: str, days: int = 30) -> list[dict]:
    """p50/p90/p95 total latency grouped by scope-resolution method.

    Reads query_traces, which already records per-stage timings, the scope
    decision and token counts for every real query — the § 08 latency
    targets have had nothing measuring them, and this is that, with no new
    instrumentation needed.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT COALESCE(trace->'scope_decision'->>'method', 'unknown') AS method,
                   count(*) AS n,
                   round(percentile_cont(0.5)  WITHIN GROUP (ORDER BY total_ms)) AS p50,
                   round(percentile_cont(0.90) WITHIN GROUP (ORDER BY total_ms)) AS p90,
                   round(percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms)) AS p95,
                   round(avg((trace->>'llm_call_count')::numeric), 2) AS avg_llm_calls
            FROM query_traces
            WHERE wiki_session_id = :sid
              AND created_at > now() - make_interval(days => :d)
              AND total_ms IS NOT NULL
            GROUP BY 1 ORDER BY n DESC
        """), {"sid": wiki_session_id, "d": days}).fetchall()
    return [{"method": r[0], "queries": int(r[1]), "p50_ms": int(r[2] or 0),
             "p90_ms": int(r[3] or 0), "p95_ms": int(r[4] or 0),
             "avg_llm_calls": float(r[5] or 0)} for r in rows]


def stage_latency_breakdown(wiki_session_id: str, days: int = 30) -> list[dict]:
    """Median wall time per pipeline stage — says which stage to attack.

    Stages come from the trace's own `stages` array, so this stays correct
    if the graph gains or loses a node.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s->>'name' AS stage,
                   count(*) AS n,
                   round(percentile_cont(0.5) WITHIN GROUP (ORDER BY (s->>'duration_ms')::numeric)) AS p50,
                   round(percentile_cont(0.95) WITHIN GROUP (ORDER BY (s->>'duration_ms')::numeric)) AS p95
            FROM query_traces q
            CROSS JOIN LATERAL jsonb_array_elements(q.trace->'stages') AS s
            WHERE q.wiki_session_id = :sid
              AND q.created_at > now() - make_interval(days => :d)
            GROUP BY 1 ORDER BY p50 DESC NULLS LAST
        """), {"sid": wiki_session_id, "d": days}).fetchall()
    return [{"stage": r[0], "samples": int(r[1]), "p50_ms": int(r[2] or 0),
             "p95_ms": int(r[3] or 0)} for r in rows]


# ---------------------------------------------------------------------------
# Chat messages (conversational UX)
# ---------------------------------------------------------------------------

def insert_message(
    session_id: str,
    role: str,
    content: str,
    msg_type: str = "text",
    metadata: dict | None = None,
    user_id: int | None = None,
) -> int:
    """Insert a chat message and return its id.

    user_id records the authenticated author. Nothing reads it yet — it's
    stored so per-user chat isolation stays buildable later; see the column
    comment in _run_schema_statements for why it can't wait.
    """
    from sqlalchemy import text
    engine = get_engine()
    meta_json = json.dumps(metadata) if metadata is not None else None
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                INSERT INTO chat_messages (session_id, role, content, msg_type, metadata, user_id)
                VALUES (:sid, :role, :content, :msg_type, CAST(:metadata AS jsonb), :user_id)
                RETURNING id
            """),
            {
                "sid": session_id, "role": role, "content": content,
                "msg_type": msg_type, "metadata": meta_json, "user_id": user_id,
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

def get_source_docs(wiki_id: str, session_id: str) -> list[str]:
    """Return distinct source_doc values for a session."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT source_doc FROM pages
                WHERE wiki_id = :w AND session_id = :sid AND source_doc != ''
            """),
            {"w": wiki_id, "sid": session_id},
        )
        return [r.source_doc for r in rows]


# ---------------------------------------------------------------------------
# Source positions (citation exact-location support)
# ---------------------------------------------------------------------------

def store_page_map(wiki_id: str, session_id: str, source_doc: str, page_map: list[dict]) -> None:
    """Store page-level character positions for a source document."""
    if not page_map:
        return
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        for entry in page_map:
            conn.execute(
                text("""
                    INSERT INTO source_positions (wiki_id, session_id, source_doc, page_num, char_start, char_end)
                    VALUES (:w, :sid, :doc, :pn, :cs, :ce)
                    ON CONFLICT (session_id, source_doc, page_num) DO UPDATE SET
                        char_start = EXCLUDED.char_start,
                        char_end   = EXCLUDED.char_end,
                        wiki_id    = EXCLUDED.wiki_id
                """),
                {
                    "w": wiki_id, "sid": session_id, "doc": source_doc,
                    "pn": entry["page_num"], "cs": entry["char_start"], "ce": entry["char_end"],
                },
            )
        conn.commit()


def get_page_map(wiki_id: str, session_id: str, source_doc: str) -> list[dict]:
    """Return page positions for a source document."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT page_num, char_start, char_end
                FROM source_positions
                WHERE wiki_id = :w AND session_id = :sid AND source_doc = :doc
                ORDER BY page_num
            """),
            {"w": wiki_id, "sid": session_id, "doc": source_doc},
        )
        return [{"page_num": r.page_num, "char_start": r.char_start, "char_end": r.char_end} for r in rows]


def find_quote_position(wiki_id: str, session_id: str, source_doc: str, quote_text: str) -> dict:
    """Find the page number and character offset of a quote in a source document.

    Loads the document text from the upload path, does a fuzzy match, then maps
    the offset back to a PDF page using the stored page_map.
    """
    import re as _re

    page_map = get_page_map(wiki_id, session_id, source_doc)
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


def get_metadata(wiki_id: str, session_id: str, doc_name: str) -> dict:
    """Return the metadata dict for a document, or {} if none stored."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(f"""
                SELECT {', '.join(_METADATA_COLUMNS)}
                FROM page_metadata
                WHERE wiki_id = :w AND session_id = :sid AND title = :title
            """),
            {"w": wiki_id, "sid": session_id, "title": doc_name},
        ).fetchone()
        if row is None:
            return {}
        return {k: v for k, v in zip(_METADATA_COLUMNS, row) if v is not None}


def get_document_types(wiki_id: str, session_id: str) -> dict[str, str]:
    """Every document's ingest-extracted instrument type, keyed by source_doc.

    `documents.doc_type` records the instrument's own name ("Rejoinder in the
    Petition", "Disclosure Letter", "Statement of Work") in the same wording a
    question uses to ask about it — a far more direct signal than the filename,
    which encodes the same thing as an abbreviation the corpus never explains
    ("RITPAN", "DiscLtr", "SOW"). Returned whole (one small query, ~570 rows
    here) so callers can do token-level matching in Python rather than pushing
    normalisation guesswork into SQL.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT source_doc, doc_type
                FROM documents
                WHERE wiki_id = :w AND session_id = :sid
                  AND doc_type IS NOT NULL AND doc_type <> ''
            """),
            {"w": wiki_id, "sid": session_id},
        )
        return {row.source_doc: row.doc_type for row in rows if row.source_doc}


def get_documents_by_family(wiki_id: str, session_id: str, doc_family: str) -> list[str]:
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
                WHERE wiki_id = :w AND session_id = :sid AND doc_family = :fam
            """),
            {"w": wiki_id, "sid": session_id, "fam": doc_family},
        )
        return [row.title for row in rows]


def find_docs_sharing_parties(wiki_id: str, session_id: str, source_doc: str,
                              type_hint: str, exclude: str | None = None,
                              cap: int = 4) -> list[str]:
    """Documents naming at least one of ``source_doc``'s own parties, filtered
    to a doc_type matching ``type_hint``.

    Built for "the original X agreement" referenced alongside an amendment
    the question names by party ("the Apex Meridian amendment") — the
    amendment resolves via the party detector, but the original it amends
    often carries no party name of its own in the question at all. Ingest's
    own cross-reference resolution frequently can't pin the specific document
    either (an amendment stating "the agreement dated as referenced in the
    recitals below" names no filename or date inline for the resolver to
    match), leaving ``document_relations`` with an unresolved edge. The
    ``documents.parties`` column is populated independently of that
    resolution, from the same extraction that reads the amendment's own
    signature block — so two documents between the same named parties are
    findable by that alone, without depending on the cross-reference having
    resolved.

    Returns [] rather than guessing when the amendment document itself has
    no recorded parties, or when the JOIN would be unbounded (no type_hint).
    """
    if not type_hint:
        return []
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT d2.source_doc
            FROM documents d1
            CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(d1.parties, '[]'::jsonb)) AS p1(party)
            JOIN documents d2
              ON d2.wiki_id = d1.wiki_id
             AND d2.source_doc <> d1.source_doc
            CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(d2.parties, '[]'::jsonb)) AS p2(party)
            WHERE d1.wiki_id = :w AND d1.session_id = :sid AND d1.source_doc = :d
              AND p1.party = p2.party
              AND d2.doc_type ILIKE :hint
              AND d2.source_doc <> :excl
            LIMIT :cap
        """), {"w": wiki_id, "sid": session_id, "d": source_doc,
               "hint": f"%{type_hint}%", "excl": exclude or source_doc, "cap": cap}).fetchall()
        return [r[0] for r in rows]


def get_families_of_documents(wiki_id: str, session_id: str,
                              documents: list[str]) -> dict[str, str]:
    """Map each named document to its doc_family (documents with none are omitted).

    The inverse of get_documents_by_family, for the case where the documents are
    already known and what's needed is whether two of them are the same KIND of
    instrument.
    """
    if not documents:
        return {}
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT title, doc_family
                FROM page_metadata
                WHERE wiki_id = :w AND session_id = :sid
                  AND title = ANY(:docs) AND doc_family IS NOT NULL
            """),
            {"w": wiki_id, "sid": session_id, "docs": list(documents)},
        )
        return {row.title: row.doc_family for row in rows}


def get_documents_by_folder_hint(wiki_id: str, session_id: str, keywords: list[str]) -> list[str]:
    """Documents whose upload-folder name matches one of a family's keywords.

    Fallback for get_documents_by_family: page_metadata.doc_family comes from
    ingest-time CONTENT classification, which can disagree with where the file
    was actually filed (documents.family_method='content_folder_mismatch' when
    it does) and leave doc_family NULL — invisible to family-scoped questions
    even though the corpus (and the user asking about "the Legal Opinions")
    still considers it part of that family. folder_hint is the raw signal
    behind that mismatch and costs nothing extra to read — it was already
    stored at ingest.
    """
    if not keywords:
        return []
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT source_doc FROM documents
                WHERE wiki_id = :w AND session_id = :sid
                  AND folder_hint ILIKE ANY(:kws)
            """),
            {"w": wiki_id, "sid": session_id,
             "kws": [f"%{kw}%" for kw in keywords]},
        )
        return [row.source_doc for row in rows]


def list_doc_families(wiki_id: str, session_id: str) -> list[str]:
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
                WHERE wiki_id = :w AND session_id = :sid AND doc_family IS NOT NULL
            """),
            {"w": wiki_id, "sid": session_id},
        )
        return [row.doc_family for row in rows]


# Structural anchors are recorded at ingest for every document whose text prints
# numbered sections — 470 of the live corpus's 568. clause_map is a separate
# table filled by a backfill script that was never run against this corpus, and
# holds zero rows for it, so every clause-number question fell straight through
# the lookup below and was answered as though its document were unnumbered.
# Anchors carry the same two facts clause_map does (a number, its heading) plus
# the character offset, so they serve as the fallback source here rather than
# needing their own path through the caller.
_ANCHOR_CLAUSE_KINDS = ("section", "clause", "schedule")


def _plausible_heading(text: str) -> bool:
    """Whether an anchor's heading_text is a section NAME rather than its body.

    The anchor parser records a numbered line whatever follows the number, so
    heading_text is sometimes a table row ("HY 10 39,402,795 70,235,481") or the
    opening of a wrapped body sentence ("NimbusForge shall maintain, test at
    least annually, ..."). Both matter: the heading is put into the retrieval
    query, where a body sentence or a row of figures drags BM25 off the section
    it was meant to find, and it is also shown to the user as the section's name.
    """
    t = (text or "").strip()
    if not (2 <= len(t) <= 70):
        return False
    words = t.split()
    if not 1 <= len(words) <= 8:
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < len(t) / 2:
        return False
    return not t.endswith((",", ";", "-"))


def _anchor_clauses(wiki_id: str, session_id: str, doc_hint: str,
                    clause_num: str | None = None) -> list[dict]:
    """Numbered sections for a document, read from structural_anchors."""
    from sqlalchemy import text
    sql = """
        SELECT DISTINCT anchor_label, heading_text, page_title, char_start
        FROM structural_anchors
        WHERE wiki_id = :w AND session_id = :sid
          AND anchor_kind = ANY(:kinds)
          AND (source_doc ILIKE '%' || :doc || '%'
               OR :doc ILIKE '%' || source_doc || '%')
    """
    params = {"w": wiki_id, "sid": session_id, "doc": doc_hint,
              "kinds": list(_ANCHOR_CLAUSE_KINDS)}
    if clause_num is not None:
        sql += " AND anchor_label = :num"
        params["num"] = clause_num
    sql += " ORDER BY char_start"
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [{"num": r.anchor_label, "heading": (r.heading_text or "").strip(),
             "page_title": r.page_title or "", "char_start": r.char_start}
            for r in rows]


def clauses_of_type(wiki_id: str, session_id: str, source_doc: str,
                    canon: str) -> list[dict]:
    """Clauses of one canonical type recorded for one document.

    Reads clause_type_canon rather than matching prose: "does this contain a
    warranty clause" is a question about the clause TYPE, and the words of a
    representation of authority ("represents and warrants that it has full
    power") satisfy a prose search while answering a different question.
    """
    from sqlalchemy import text
    if not source_doc or not canon:
        return []
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT clause_type, clause_type_canon, verbatim_text, page_num
            FROM clauses
            WHERE wiki_id = :w AND session_id = :s
              AND source_doc = :d AND clause_type_canon = :c
            ORDER BY length(verbatim_text) DESC
            LIMIT 5
        """), {"w": wiki_id, "s": session_id, "d": source_doc, "c": canon}).fetchall()
    return [{"clause_type": r[0], "clause_type_canon": r[1],
             "verbatim_text": r[2], "page_num": r[3]} for r in rows]


def clause_count_for_doc(wiki_id: str, session_id: str, source_doc: str) -> int:
    """How many typed clauses were recorded for a document.

    The denominator behind reading an absence: a document with two extracted
    clauses is silent about everything, and that silence is the extractor's,
    not the document's.
    """
    from sqlalchemy import text
    if not source_doc:
        return 0
    with get_engine().connect() as conn:
        return int(conn.execute(text("""
            SELECT count(*) FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
        """), {"w": wiki_id, "s": session_id, "d": source_doc}).scalar() or 0)


def clause_types_for_doc(wiki_id: str, session_id: str, source_doc: str,
                         limit: int = 8) -> list[str]:
    """The canonical clause types a document does carry, most common first."""
    from sqlalchemy import text
    if not source_doc:
        return []
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT clause_type_canon, count(*) FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
              AND clause_type_canon IS NOT NULL AND clause_type_canon <> 'structural'
            GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT :n
        """), {"w": wiki_id, "s": session_id, "d": source_doc, "n": limit}).fetchall()
    return [r[0] for r in rows]


def lookup_clause(wiki_id: str, session_id: str, doc_hint: str, clause_num: str) -> list[dict]:
    """Resolve a clause number to its heading and wiki page(s) for one document.

    doc_hint is whatever name the scope resolver produced — it may be the full
    source_doc or a fragment of it, so match by containment either way. Returns
    [] when the document is unnumbered or the number does not exist, and the
    caller distinguishes those two cases via doc_clause_numbers().

    Falls back to structural_anchors when clause_map has nothing, which for this
    corpus is always (see _ANCHOR_CLAUSE_KINDS above).
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT heading, page_title
                FROM clause_map
                WHERE wiki_id = :w AND session_id = :sid AND clause_num = :num
                  AND (source_doc ILIKE '%' || :doc || '%'
                       OR :doc ILIKE '%' || source_doc || '%')
            """),
            {"w": wiki_id, "sid": session_id, "num": clause_num, "doc": doc_hint},
        )
        hits = [{"heading": r.heading, "page_title": r.page_title} for r in rows]
    if hits:
        return hits
    # Only headings that read as section NAMES: this value is put into the
    # retrieval query and shown to the user, and an unusable one is worse than
    # none — the caller treats [] as "number not found", which doc_clause_numbers
    # below then qualifies correctly.
    return [{"heading": a["heading"], "page_title": a["page_title"]}
            for a in _anchor_clauses(wiki_id, session_id, doc_hint, clause_num)
            if _plausible_heading(a["heading"])]


def doc_clause_numbers(wiki_id: str, session_id: str, doc_hint: str) -> list[str]:
    """Every clause number the map knows for a document ([] = unnumbered source).

    Deliberately NOT served by the structural_anchors fallback that lookup_clause
    uses. The caller turns this list into a flat assertion to the user — "no
    clause 12 in this document, its source is numbered 1-8" — which is only sound
    if the list is COMPLETE. clause_map is a full parse of the source PDF and can
    be trusted whole. Anchors record only the numbered lines the parser
    recognised, and nothing in them proves the parse reached the end of the
    document: a run of 1..8 is equally consistent with an eight-section contract
    and with a twelve-section one whose tail the parser lost. Asserting the
    negative from that would manufacture exactly the confident-and-wrong answer
    the anchor fallback exists to remove.

    So anchors serve the positive path only (this number IS present, here is its
    heading), and this stays empty until clause_map is actually backfilled —
    which leaves the out-of-range check exactly as switched-off as it already is
    for this corpus, rather than switching it on unsoundly.
    """
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT clause_num
                FROM clause_map
                WHERE wiki_id = :w AND session_id = :sid
                  AND (source_doc ILIKE '%' || :doc || '%'
                       OR :doc ILIKE '%' || source_doc || '%')
            """),
            {"w": wiki_id, "sid": session_id, "doc": doc_hint},
        )
        nums = [r.clause_num for r in rows]
    # "1" < "10" < "2" under string sort; sort numerically by dotted parts.
    return sorted(nums, key=lambda n: [int(p) for p in n.split(".")])
