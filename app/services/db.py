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

        _init_backbone_schema(conn, text)

        conn.commit()


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

def get_pages(session_id: str, include_archived: bool = False) -> dict[str, dict]:
    """Return all pages for a session as {title: {content, summary, source_doc, ...}}.

    include_archived=False (the default, used everywhere retrieval/browsing
    happens) excludes pages whose source_doc is archived — this is THE
    enforcement point for "an archived document drops out of search/chat":
    every caller (wiki index load, hybrid retrieval, /wiki/graph, /wiki/pages)
    funnels through here, so filtering once here covers all of them rather
    than needing the same check re-added at every call site.
    """
    from sqlalchemy import text
    engine = get_engine()
    archived_clause = "" if include_archived else """
                AND NOT EXISTS (
                    SELECT 1 FROM document_status ds
                    WHERE ds.session_id = pages.session_id
                      AND ds.source_doc = pages.source_doc
                      AND ds.status = 'archived'
                )"""
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT title, content, summary, source_doc, contradiction_flagged, variants
                FROM pages
                WHERE session_id = :sid{archived_clause}
            """),
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


def get_page_list(session_id: str, include_archived: bool = False) -> list[dict]:
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
                    WHERE ds.session_id = pages.session_id
                      AND ds.source_doc = pages.source_doc
                      AND ds.status = 'archived'
                )"""
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT title, source_doc, char_count, contradiction_flagged, last_modified
                FROM pages
                WHERE session_id = :sid{archived_clause}
                ORDER BY title
            """),
            {"sid": session_id},
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


def rename_page(session_id: str, old_title: str, new_title: str) -> bool:
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
            text("SELECT 1 FROM pages WHERE session_id = :sid AND title = :t"),
            {"sid": session_id, "t": old_title},
        ).first()
        if not exists:
            return False
        clash = conn.execute(
            text("SELECT 1 FROM pages WHERE session_id = :sid AND title = :t"),
            {"sid": session_id, "t": new_title},
        ).first()
        if clash:
            return False

        params = {"sid": session_id, "new": new_title, "old": old_title}
        conn.execute(text("UPDATE pages SET title = :new WHERE session_id = :sid AND title = :old"), params)
        conn.execute(text("UPDATE relations SET from_title = :new WHERE session_id = :sid AND from_title = :old"), params)
        conn.execute(text("UPDATE relations SET to_title = :new WHERE session_id = :sid AND to_title = :old"), params)
        conn.execute(text("UPDATE clause_map SET page_title = :new WHERE session_id = :sid AND page_title = :old"), params)
        conn.execute(text("UPDATE contradictions SET page_title = :new WHERE session_id = :sid AND page_title = :old"), params)

        for emb_table in _page_embedding_tables(conn):
            conn.execute(
                text(f'UPDATE "{emb_table}" SET title = :new WHERE session_id = :sid AND title = :old'),
                params,
            )

        conn.commit()
    return True


def delete_page(session_id: str, title: str) -> bool:
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
            text("SELECT 1 FROM pages WHERE session_id = :sid AND title = :t"),
            {"sid": session_id, "t": title},
        ).first()
        if not exists:
            return False

        params = {"sid": session_id, "t": title}
        conn.execute(text("DELETE FROM pages WHERE session_id = :sid AND title = :t"), params)
        conn.execute(text("DELETE FROM relations WHERE session_id = :sid AND (from_title = :t OR to_title = :t)"), params)
        conn.execute(text("DELETE FROM clause_map WHERE session_id = :sid AND page_title = :t"), params)
        conn.execute(text("DELETE FROM contradictions WHERE session_id = :sid AND page_title = :t"), params)

        for emb_table in _page_embedding_tables(conn):
            conn.execute(
                text(f'DELETE FROM "{emb_table}" WHERE session_id = :sid AND title = :t'),
                params,
            )

        conn.commit()
    return True


def merge_pages(session_id: str, source_title: str, target_title: str) -> bool:
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
            text("SELECT content, summary FROM pages WHERE session_id = :sid AND title = :t"),
            {"sid": session_id, "t": source_title},
        ).first()
        tgt = conn.execute(
            text("SELECT content, summary FROM pages WHERE session_id = :sid AND title = :t"),
            {"sid": session_id, "t": target_title},
        ).first()
        if not src or not tgt:
            return False

        merged_content = f"{tgt.content}\n\n{src.content}"
        merged_summary = tgt.summary or src.summary

        conn.execute(
            text("""UPDATE pages SET content = :c, summary = :s, char_count = :cc, last_modified = now()
                     WHERE session_id = :sid AND title = :t"""),
            {"c": merged_content, "s": merged_summary, "cc": len(merged_content),
             "sid": session_id, "t": target_title},
        )
        conn.execute(
            text("DELETE FROM pages WHERE session_id = :sid AND title = :t"),
            {"sid": session_id, "t": source_title},
        )

        repoint = {"sid": session_id, "tgt": target_title, "src": source_title}
        conn.execute(text("UPDATE relations SET from_title = :tgt WHERE session_id = :sid AND from_title = :src"), repoint)
        conn.execute(text("UPDATE relations SET to_title = :tgt WHERE session_id = :sid AND to_title = :src"), repoint)
        # A re-point can create a self-loop (if source and target were
        # already directly connected) or an exact duplicate of an edge that
        # already existed under target_title — both are noise now, drop them.
        conn.execute(
            text("DELETE FROM relations WHERE session_id = :sid AND from_title = :tgt AND to_title = :tgt"),
            {"sid": session_id, "tgt": target_title},
        )
        conn.execute(text("""
            DELETE FROM relations a USING relations b
            WHERE a.session_id = :sid AND b.session_id = :sid
              AND a.ctid > b.ctid
              AND a.from_title = b.from_title AND a.to_title = b.to_title AND a.label = b.label
        """), {"sid": session_id})

        conn.execute(text("UPDATE clause_map SET page_title = :tgt WHERE session_id = :sid AND page_title = :src"), repoint)
        conn.execute(text("UPDATE contradictions SET page_title = :tgt WHERE session_id = :sid AND page_title = :src"), repoint)

        for emb_table in _page_embedding_tables(conn):
            conn.execute(
                text(f'DELETE FROM "{emb_table}" WHERE session_id = :sid AND title = ANY(:titles)'),
                {"sid": session_id, "titles": [source_title, target_title]},
            )

        conn.commit()
    return True


# ---------------------------------------------------------------------------
# Review Queue — clauses (target architecture § 02, first slice)
# ---------------------------------------------------------------------------

def insert_clauses(session_id: str, source_doc: str, clauses: list[dict]) -> int:
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
            conn.execute(
                text("""
                    INSERT INTO clauses
                        (session_id, source_doc, clause_type, verbatim_text,
                         typed_value, confidence, page_num, stakes)
                    VALUES
                        (:sid, :doc, :ctype, :vtext, :tval, :conf, :page, :stakes)
                """),
                {
                    "sid": session_id, "doc": source_doc, "ctype": clause_type,
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


def supersede_review_items(session_id: str, source_doc: str) -> int:
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
            WHERE session_id = :sid AND source_doc = :doc
              AND review_status = 'pending'
        """), {"sid": session_id, "doc": source_doc})
        conn.commit()
        return res.rowcount or 0


def get_review_queue(session_id: str) -> list[dict]:
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
                WHERE session_id = :sid AND review_status = 'pending'
            """),
            {"sid": session_id},
        ).fetchall()
        other_rows = conn.execute(
            text("""
                SELECT id, source_doc, item_kind, item_label, item_value,
                       typed_value, confidence, page_num, stakes, reason, created_at
                FROM review_queue
                WHERE session_id = :sid AND review_status = 'pending'
            """),
            {"sid": session_id},
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


def resolve_clause(session_id: str, clause_id: int, action: str, edited_text: str | None = None) -> bool:
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
            "sid": session_id, "id": clause_id,
            "status": review_status, "resolution": resolution_map[action],
        }
        if action == "edit":
            if not edited_text or not edited_text.strip():
                raise ValueError("edited_text is required for action=edit")
            result = conn.execute(
                text("""
                    UPDATE clauses SET review_status = :status, resolution = :resolution,
                           verbatim_text = :vtext, reviewed_at = now()
                    WHERE session_id = :sid AND id = :id AND review_status = 'pending'
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
                    WHERE session_id = :sid AND id = :id AND review_status = 'pending'
                """),
                params,
            )
        conn.commit()
        return (result.rowcount or 0) > 0


def resolve_review_item(session_id: str, item_id: int, action: str,
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
        params = {"sid": session_id, "id": item_id, "status": review_status,
                  "resolution": resolution_map[action]}
        if action == "edit":
            if not edited_text or not edited_text.strip():
                raise ValueError("edited_text is required for action=edit")
            result = conn.execute(text("""
                UPDATE review_queue SET review_status = :status, resolution = :resolution,
                       item_value = :val, reviewed_at = now()
                WHERE session_id = :sid AND id = :id AND review_status = 'pending'
            """), {**params, "val": _crypto().encrypt(edited_text.strip())})
        else:
            result = conn.execute(text("""
                UPDATE review_queue SET review_status = :status, resolution = :resolution,
                       reviewed_at = now()
                WHERE session_id = :sid AND id = :id AND review_status = 'pending'
            """), params)
        conn.commit()
        return (result.rowcount or 0) > 0


def bulk_accept_review_items(session_id: str, min_confidence: float) -> int:
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
            WHERE session_id = :sid AND review_status = 'pending'
              AND stakes = 'low' AND confidence >= :min_conf
        """), {"sid": session_id, "min_conf": min_confidence})
        conn.commit()
        return result.rowcount or 0


def bulk_accept_clauses(session_id: str, min_confidence: float) -> int:
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
                WHERE session_id = :sid AND review_status = 'pending'
                  AND stakes = 'low' AND confidence >= :min_conf
            """),
            {"sid": session_id, "min_conf": min_confidence},
        )
        conn.commit()
        return result.rowcount or 0


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
# Document lifecycle (target architecture § 01.4 — Archive / Hard-delete)
#
# See services/documents.py for the orchestration layer (file deletion,
# sessions.json bookkeeping) and the disclosed limitation around pages
# merged from multiple source documents — everything here operates on
# whatever pages.source_doc currently records, which for a merged page is
# only the most recent contributing document, not the full set.
# ---------------------------------------------------------------------------

def get_document_statuses(session_id: str) -> dict[str, dict]:
    """Return {source_doc: {status, archived_at}} for every document that has
    ever been archived. A document with no row here is active."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT source_doc, status, archived_at FROM document_status WHERE session_id = :sid"),
            {"sid": session_id},
        )
        return {
            r.source_doc: {
                "status": r.status,
                "archived_at": r.archived_at.isoformat() if r.archived_at else None,
            }
            for r in rows
        }


def archive_document(session_id: str, source_doc: str) -> None:
    """Mark a document archived. Idempotent — archiving twice just refreshes archived_at."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO document_status (session_id, source_doc, status, archived_at)
                VALUES (:sid, :doc, 'archived', now())
                ON CONFLICT (session_id, source_doc)
                DO UPDATE SET status = 'archived', archived_at = now()
            """),
            {"sid": session_id, "doc": source_doc},
        )
        conn.commit()


def unarchive_document(session_id: str, source_doc: str) -> None:
    """Restore an archived document to active by removing its status row —
    keeps "row exists" == "archived" a valid invariant everywhere else that
    reads this table, rather than also needing to check a status value."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM document_status WHERE session_id = :sid AND source_doc = :doc"),
            {"sid": session_id, "doc": source_doc},
        )
        conn.commit()


def is_document_archived(session_id: str, source_doc: str) -> bool:
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT 1 FROM document_status
                WHERE session_id = :sid AND source_doc = :doc AND status = 'archived'
            """),
            {"sid": session_id, "doc": source_doc},
        ).first()
        return row is not None


def delete_document_data(session_id: str, source_doc: str) -> dict:
    """Cascade-delete every DB row cleanly attributable to one document.

    Does NOT touch the uploaded file on disk or sessions.json — see
    services/documents.py, which owns those and calls this for the DB side.

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
        "relations_deleted": 0,
    }

    with engine.connect() as conn:
        titles = [r.title for r in conn.execute(
            text("SELECT title FROM pages WHERE session_id = :sid AND source_doc = :doc"),
            {"sid": session_id, "doc": source_doc},
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
                    text(f'DELETE FROM "{emb_table}" WHERE session_id = :sid AND title = ANY(:titles)'),
                    {"sid": session_id, "titles": titles},
                )
                report["embeddings_deleted"] += result.rowcount or 0

            result = conn.execute(
                text("DELETE FROM pages WHERE session_id = :sid AND source_doc = :doc"),
                {"sid": session_id, "doc": source_doc},
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
                    WHERE session_id = :sid
                      AND (from_title = ANY(:titles) OR to_title = ANY(:titles))
                """),
                {"sid": session_id, "titles": titles},
            )
            report["relations_deleted"] = result.rowcount or 0

        result = conn.execute(
            text("DELETE FROM clause_map WHERE session_id = :sid AND source_doc = :doc"),
            {"sid": session_id, "doc": source_doc},
        )
        report["clause_map_deleted"] = result.rowcount or 0

        result = conn.execute(
            text("DELETE FROM source_positions WHERE session_id = :sid AND source_doc = :doc"),
            {"sid": session_id, "doc": source_doc},
        )
        report["source_positions_deleted"] = result.rowcount or 0

        conn.execute(
            text("DELETE FROM document_status WHERE session_id = :sid AND source_doc = :doc"),
            {"sid": session_id, "doc": source_doc},
        )

        conn.commit()

    return report


# ---------------------------------------------------------------------------
# Embeddings (Phase 3 — pgvector search)
# ---------------------------------------------------------------------------

def upsert_question_embeddings(session_id: str, title: str,
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
        conn.execute(text(f"DELETE FROM {tbl} WHERE session_id = :sid AND title = :title"),
                     {"sid": session_id, "title": title})
        n = 0
        for question, vector in questions:
            if not question or not vector:
                continue
            emb_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
            conn.execute(text(f"""
                INSERT INTO {tbl}
                    (session_id, title, question, embedding, doc_family, source_doc)
                VALUES (:sid, :title, :q, CAST(:embedding AS vector), :fam, :doc)
                ON CONFLICT (session_id, title, question) DO UPDATE SET
                    embedding = EXCLUDED.embedding
            """), {"sid": session_id, "title": title, "q": question[:2000],
                   "embedding": emb_str, "fam": doc_family, "doc": source_doc})
            n += 1
        conn.commit()
    return n


def search_similar_questions(session_id: str, query_embedding: list[float],
                             limit: int = 10,
                             doc_family: str | None = None) -> list[dict]:
    """Find pages whose hypothetical questions match the query.

    Returns page titles, not questions — the question is a retrieval handle,
    and what the caller ultimately needs is the page that can answer it. Each
    page appears once, at its best-matching question's score.
    """
    from sqlalchemy import text
    engine = get_engine()
    tbl = _question_table_name()
    emb_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"
    family_clause = " AND doc_family = :fam" if doc_family else ""
    params = {"sid": session_id, "embedding": emb_str, "limit": limit}
    if doc_family:
        params["fam"] = doc_family
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT ON (title)
                   title, question,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM {tbl}
            WHERE session_id = :sid{family_clause}
            ORDER BY title, embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """), params).fetchall()
    out = [{"title": r.title, "question": r.question, "score": float(r.score)}
           for r in rows]
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def count_question_embeddings(session_id: str) -> int:
    from sqlalchemy import text
    try:
        with get_engine().connect() as conn:
            return int(conn.execute(text(
                f"SELECT COUNT(*) FROM {_question_table_name()} WHERE session_id = :sid"
            ), {"sid": session_id}).scalar() or 0)
    except Exception:
        return 0


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
