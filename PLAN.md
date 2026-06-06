# Legal Wiki — Production Scaling Plan

This file tracks all planned and completed engineering work to scale the system
from 50–100 documents to 17–20k documents in a legal production environment,
and from a single-session tool to a shared platform across a large organisation.

**Legend:** ✅ Done · 🔄 In Progress · ⬜ Planned

---

## Background

The system works well at 50–100 legal docs per session. At 17–20k docs in
production across a large organisation, two classes of problems emerge:

- **Hard failures** — the system crashes or stops responding (O(N²) cross-ref,
  index.json I/O deadlock, context-window overflow at query time)
- **Silent quality failures** — answers degrade without error signals (wiki pages
  grow unbounded via appends, contradiction detection breaks down, page selection
  returns wrong pages)

The plan is structured into phases: cheapest/safest changes first, structural
rewrites later.

---

## Phase 1 — Cost Reduction & Quick Fixes ✅ COMPLETE

Zero architecture changes. All changes are additive or swap-in replacements.

### C1 — Dual-Model Routing ✅
**Files:** `config.py`, `llm.py`, `wiki.py`, `advanced_modes.py`, `.env`, `.env.example`

- Added `AZURE_FAST_DEPLOYMENT` + `OPENROUTER_FAST_MODEL` config vars
- Added `fast: bool = False` param to `llm.ask()` — routes to cheap model when `True`
- `fast_ask()` now correctly routes to cheap model (was silently using full model)
- Added `active_model(fast)` helper to `llm.py` for token logging
- **Cheap model used for:** contradiction pre-flight, JSON repair, cell extraction (Review/Compare)
- **Full model kept for:** ingest synthesis, answer generation, drafting, compare narrative
- **Models configured:**
  - Azure big: `gpt-5.4` · Azure fast: `gpt-5.4-mini`
  - OpenRouter big: `openai/gpt-oss-120b:free` · OpenRouter fast: `openai/gpt-oss-20b:free`

### C2 — Merge Confidence Evaluation Into Answer Prompt ✅
**Files:** `prompts.py`, `wiki.py`

- `ANSWER_PROMPT` now instructs the model to append `CONFIDENCE_SCORE` and
  `CONFIDENCE_REASON` as structured lines at the end of the `<reasoning>` block
- `generate_answer()` extracts these via regex before stripping the reasoning block
- Eliminated the separate `_evaluate_confidence()` LLM call entirely
- **Effect:** query cost reduced from 3 LLM calls → 2 calls (>20 pages) or 2 → 1 (≤20 pages)

### C5 — Explicit `max_tokens` Budgets ✅
**Files:** `config.py`, `wiki.py`

Added `MAX_TOKENS_*` constants in `config.py` as a single source of truth:

| Constant | Value | Used For |
|---|---|---|
| `MAX_TOKENS_INGEST_SINGLE` | 4096 | Short-doc single-call synthesis |
| `MAX_TOKENS_INGEST_OVERVIEW` | 1500 | Phase-1 overview + topic list |
| `MAX_TOKENS_INGEST_DETAIL` | 3500 | Phase-2 per-segment detail extraction |
| `MAX_TOKENS_CONTRADICTION` | 300 | Pairwise contradiction pre-flight |
| `MAX_TOKENS_JSON_REPAIR` | 2048 | LLM JSON repair |
| `MAX_TOKENS_PAGE_SELECTION` | 1000 | Page title selection |
| `MAX_TOKENS_ANSWER` | 4096 | Full legal synthesis answer |

All `llm.ask()` call sites now pass an explicit cap — no more silent 4096-token defaults.

### Bug Fixes ✅
**Files:** `wiki.py`, `app.py`

- **`_ingest_overview()` return bug** — function returned 2 values `(topics, dict)` but
  caller expected 3 `(doc_type, topics, dict)`. Crashed silently on every long-document
  ingest. Fixed: now correctly returns `(doc_type, topics, dict)`.
- **Page selection "not covered" bug** — Phase 1 mistakenly set page selection to
  `fast=True`. Cheap model returned wrong/unparseable results; old fallback returned
  first-30 pages by insertion order (missed docs ingested later). Fixed:
  - Page selection reverted to **full model**
  - New `_keyword_fallback_pages()` function replaces first-30 fallback — scores all
    pages by word-overlap with the question, returns top-25

### Token Usage Tracking ✅
**Files:** `wiki.py`, `app.py`

- `_select_relevant_pages()` now returns `(titles, usage)` tuple
- `get_context()` surfaces `page_selection_usage` in its return dict
- `generate_answer()` accepts `page_selection_usage` and builds a full
  `token_breakdown` list and `token_total` aggregate
- Every query response now includes:
  ```json
  "token_breakdown": [
    {"call": "page_selection",    "model": "...", "prompt_tokens": 480, "completion_tokens": 95,  "total_tokens": 575},
    {"call": "answer_generation", "model": "...", "prompt_tokens": 3840,"completion_tokens": 720, "total_tokens": 4560}
  ],
  "token_total": {"prompt_tokens": 4320, "completion_tokens": 815, "total_tokens": 5135}
  ```
- Token usage also logged per-query to `data/logs/{session_id}_log.md`

### Token Cost Fixes ✅
**Files:** `wiki.py`, `config.py`

- **Q: page context cap** — cached answer pages (Q:) capped at `MAX_QPAGE_CONTEXT_CHARS`
  (3,000 chars) when building wiki context. Primary clause pages are never truncated.
  Prevents 84k-token answer calls caused by large prior answers crowding out source content.
- **BM25 pre-filter for page selection** — `_select_relevant_pages()` now runs
  `_keyword_fallback_pages()` first (top `PAGE_SELECTION_PREFILTER_N = 150` candidates)
  before passing to the LLM. Drops the selection call from 80k → ~10k tokens regardless
  of wiki size. Safe for legal: BM25 scores by keyword overlap so document-specific pages
  always rank into the candidate set.

### Documentation ✅
**Files:** `README.md`, `SYSTEM_OVERVIEW.md`, `SYSTEM_FLOW.md`, `.env.example`

- All docs updated to reflect dual-model routing, merged confidence eval, new env vars,
  corrected model names, and token budget system

---

## Phase 2 — Storage Foundation ✅ COMPLETE

**Why PostgreSQL, not SQLite:**

SQLite is a single-file database designed for one writer at a time. In a large
organisation with many concurrent users, multiple Gunicorn workers, and Celery
ingest workers all hitting the same session, SQLite's file-level WAL becomes a
hard bottleneck. More critically, Phase 3 requires vector similarity search over
140k+ embeddings at query time — PostgreSQL's `pgvector` extension handles this
natively with a sub-5ms HNSW index. SQLite has no production-grade equivalent.
PostgreSQL also ships with full-text search (tsvector + GIN index) which
completely eliminates the custom inverted index build in Phase 4 S2.

In short: PostgreSQL + pgvector does the work of SQLite + a vector DB + a custom
inverted index — as a single managed service, with proper concurrency, and without
additional infrastructure.

### S1 — Replace `index.json` with PostgreSQL ✅
**Files:** `wiki.py`, `config.py`, new `services/db.py`

**Why it's critical:** At 140k pages, `index.json` will be several GB. Every single ingest
segment deserialises the entire file, mutates it in RAM, and re-serialises it — under a lock
that blocks all concurrent writes. A crash mid-write corrupts the entire session's knowledge
base with no recovery path.

**What to build:**
- New `services/db.py` — SQLAlchemy abstraction layer (async-compatible, connection-pooled)
- Enable `pgvector` extension on first connect: `CREATE EXTENSION IF NOT EXISTS vector`
- Schema:

  ```sql
  CREATE TABLE pages (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL,
    title         TEXT NOT NULL,
    content       TEXT NOT NULL DEFAULT '',
    summary       TEXT NOT NULL DEFAULT '',
    source_doc    TEXT NOT NULL DEFAULT '',
    contradiction_flagged BOOLEAN NOT NULL DEFAULT FALSE,
    append_count  INT NOT NULL DEFAULT 0,
    char_count    INT NOT NULL DEFAULT 0,
    last_modified TIMESTAMPTZ NOT NULL DEFAULT now(),
    content_tsv   TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (session_id, title)
  );
  CREATE INDEX ON pages USING gin(content_tsv);   -- full-text search (replaces S2 custom index)
  CREATE INDEX ON pages (session_id, source_doc); -- fast doc-scoped lookups

  CREATE TABLE relations (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    from_title TEXT NOT NULL,
    to_title   TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    UNIQUE (session_id, from_title, to_title, label)
  );

  CREATE TABLE page_embeddings (
    session_id TEXT NOT NULL,
    title      TEXT NOT NULL,
    embedding  VECTOR(1536),             -- matches text-embedding-3-large / nemotron dimensions
    PRIMARY KEY (session_id, title)
  );
  -- HNSW index built in Phase 3 after embeddings are populated
  -- CREATE INDEX ON page_embeddings USING hnsw (embedding vector_cosine_ops)
  --   WITH (m = 16, ef_construction = 64);

  CREATE TABLE ingest_progress (
    session_id  TEXT PRIMARY KEY,
    total       INT NOT NULL DEFAULT 0,
    current     INT NOT NULL DEFAULT 0,
    message     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending',
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
  );

  CREATE TABLE page_metadata (
    session_id       TEXT NOT NULL,
    title            TEXT NOT NULL,
    governing_law    TEXT,
    jurisdiction     TEXT,
    effective_date   TEXT,
    termination_notice TEXT,
    liability_cap    TEXT,
    ip_ownership     TEXT,
    parties          TEXT,
    auto_renewal     TEXT,
    notice_period    TEXT,
    payment_terms    TEXT,
    PRIMARY KEY (session_id, title)
  );
  ```

- Replace `_load_index()` / `_save_index()` in `wiki.py` with DB read/write calls
- `_atomic_merge()` becomes `INSERT ... ON CONFLICT (session_id, title) DO UPDATE` —
  no full file cycle, row-level locking, crash-safe
- Migrate existing `index.json` files to PostgreSQL on first load (one-time migration script)
- **Connection config:** `DATABASE_URL` env var; SQLAlchemy pool (`pool_size=10,
  max_overflow=20`) handles Gunicorn/Celery worker concurrency without PgBouncer at
  current scale; add PgBouncer in Phase 6 when concurrent users > 100

**Unlocks:** S2, S3, S4, C4, C7 all depend on this.

### S5 — Replace In-Memory `PROGRESS_STORE` with Persistent State ✅
**Files:** `config.py`, `app.py`, `wiki.py` (uses `ingest_progress` table from S1)

**Why:** `PROGRESS_STORE = {}` is a plain Python dict. Dies on restart, not shared between
Gunicorn workers, grows forever in long-running processes.

**What to build:**
- `ingest_progress` table is already in the S1 schema above
- Replace all `config.PROGRESS_STORE` reads/writes in `app.py` and `wiki.py` with DB calls
- Add TTL cleanup: rows older than 7 days auto-purged via a scheduled DELETE

---

## Phase 3 — Query Scaling ✅ COMPLETE

Fixes the hard failure at query time when page count exceeds model context limits.
Depends on S1 (PostgreSQL) being complete.

### S7 + C4 — pgvector Search to Replace LLM Page Selection ✅
**Files:** `wiki.py`, `services/embedder.py`, `services/db.py`, `config.py`

**Why it's critical:** `_select_relevant_pages()` feeds ALL page titles + summaries to the LLM.
At 140k pages, the index alone exceeds any model's context window by 10–50×. Every query
would throw a context-length error. This is not a cost problem — it's a hard crash.

**What was built:**
- At ingest time: each page's summary (or first 400 chars of content) is embedded and
  stored in `page_embeddings` via `upsert_embedding()`. Embedding happens **outside** the
  session lock so HTTP calls don't block parallel ingest threads.
- Embedding model: `nvidia/llama-nemotron-embed-vl-1b-v2:free` via OpenRouter → 2048-dim vectors.
  OpenAI SDK mis-parses OpenRouter's response envelope for this model; `embedder.py` uses
  `requests.post()` directly instead.
- `EMBEDDING_DIMENSIONS=2048` set in `.env`; schema migration in `_init_schema` auto-detects
  and recreates the table if dimensions mismatch (e.g. upgrading from 1536 → 2048).
- HNSW index created with guard: `if EMBEDDING_DIMENSIONS <= 2000` — pgvector ≤ 0.6 enforces
  a 2000-dim hard limit on HNSW/IVFFlat. At 2048 dims the system falls back to exact cosine
  scan (still fast at current scale; HNSW becomes available if a ≤2000-dim model is used).
- `search_similar_pages()` in `db.py`:
  ```sql
  SELECT title FROM page_embeddings
  WHERE session_id = $1
  ORDER BY embedding <=> CAST($2 AS vector)
  LIMIT 25;
  ```
- `_select_relevant_pages()` Path 1 now uses **hybrid retrieval**: pgvector cosine top-25
  merged with BM25 keyword supplement (up to `HYBRID_BM25_SUPPLEMENT_N = 15` additional pages).
  Vector results rank first; BM25 pages are appended only if not already present. This fixes
  cases where semantic similarity is weak but keyword overlap is strong (e.g. specific party
  names, procedural terms like "committed to Sessions Court").
- Backfill script `app/backfill_embeddings.py` — idempotent, batches of 16, stores embeddings
  for all existing sessions. Run after upgrading from a pre-Phase-3 deployment.
- **Effect:** page selection 1 LLM call + 80k tokens → 0 LLM calls + ~5ms DB query + negligible BM25 in-memory scoring. No regression on queries that already worked; hybrid retrieval fixes false-negative "Not covered" answers caused by semantic search misses.

---

## Phase 4 — Ingest Quality & Scale ⬜ PLANNED

Fixes the silent quality failures that grow worse as the wiki scales.
Depends on S1 (PostgreSQL).

### S2 — Replace O(N²) Cross-Reference with PostgreSQL FTS ⬜
**Files:** `wiki.py`

**Why it's critical:** The cross-reference pass in `_merge_wiki()` runs on every single ingest:
```
for title_a in all_pages:        # N iterations
    for title_b in all_pages:    # N iterations each  → O(N²)
        if title_b in content:   # substring search
```
At 140k pages: **19.6 billion substring comparisons per document ingest**, running synchronously
under the session lock. The system stops responding.

**What to build:**
- The `content_tsv GENERATED ALWAYS` column in S1 + the GIN index already handles this.
- To find all pages whose titles appear in a new page's content:
  ```sql
  SELECT title FROM pages
  WHERE session_id = $1
    AND content_tsv @@ plainto_tsquery('english', $2)  -- $2 = new page title tokens
  ```
- No custom inverted index table, no token management — PostgreSQL FTS handles it natively,
  including stemming (so "terminating" matches "termination") which the old substring search
  missed entirely.
- Replace the nested loop in `_merge_wiki()` with a single DB query per new page insert.
- **Effect:** O(N²) Python loop → O(log N) GIN index lookup per insert.

### S3 — Page Compaction / Re-Synthesis ⬜
**Files:** `wiki.py`, `services/db.py`

**Why:** Pages grow without bound. "Governing Law (NDA)" after 500 NDAs is 500 content blocks
separated by `---`. The LLM reasoning over this produces degraded, incoherent answers.
The wiki is supposed to get smarter over time — instead it gets worse for the most-used pages.

**What to build:**
- `append_count` and `char_count` are tracked per page in the S1 schema
- Compaction trigger: when `append_count >= 5` OR `char_count >= 8000`
- Compaction prompt: send all variants to the full model:
  *"These are N versions of the same topic from different documents. Synthesise into one
  coherent wiki page, preserving all distinct facts, noting genuine contradictions explicitly,
  and tagging each claim with its source document."*
- After compaction: reset `append_count = 0`, update `char_count`, re-embed summary (C4)
- For legal: compaction is the correct place for multi-variant contradiction resolution
- Find pages due for compaction efficiently:
  ```sql
  SELECT title FROM pages
  WHERE session_id = $1
    AND (append_count >= 5 OR char_count >= 8000);
  ```

> **⚠️ Accuracy caveat:** Compaction is the one Phase 4 component with a small downside risk.
> The re-synthesis LLM call can lose specific verbatim details (exact figures, exact dates)
> if the prompt does not explicitly instruct the model to preserve them. To mitigate:
> - Always include the full `quotes` array verbatim in the compaction prompt — treat it as
>   sacrosanct, never re-summarise quotes.
> - Instruct the model: *"Preserve every exact figure, date, and amount verbatim — do not
>   paraphrase numeric values."*
> - After compaction, run a spot-check: verify that every value from the `quotes` array
>   still appears literally in the new `content`.
> - Only trigger compaction on pages where `append_count >= 5` — leave low-append pages
>   untouched. A page with 2 appends is unlikely to have grown incoherent and does not
>   need re-synthesis.
>
> All other Phase 4 and Phase 5 components improve accuracy and reduce cost with no
> meaningful downside risk.

### S4 — Fix Contradiction Detection Architecture ⬜
**Files:** `wiki.py`

**Why:** The current pairwise per-append model breaks down at scale. After 50 appends,
"existing content" is already 50 concatenated blocks — the LLM is comparing new text
against an incoherent blob. Contradiction flags accumulate but are never resolved.

**What to build:**
- Move contradiction detection to the compaction pass (S3), not per-append
- During compaction, the LLM sees all variants cleanly and can identify real contradictions
  with proper source attribution
- Per-append: only run structural NER/regex pre-filter (C3) to decide if compaction
  should be triggered sooner
- Store resolved contradictions as structured data in a `contradictions` table:
  `(session_id, page_title, claim, value_a, source_a, value_b, source_b, detected_at)`

### C3 — Structural NER Pre-Filter for Contradiction Detection ⬜
**Files:** `wiki.py`

**Why:** The contradiction pre-flight fires an LLM call for every page append with >200 chars
on both sides. At 20k docs with repeated clause types, this is hundreds of thousands of calls.
Most same-type-clause appends share the same structural values (same governing law, same
standard percentages) — the LLM call adds no value.

**What to build:**
- Extract numeric/date/party entities from both text versions using regex before the LLM call:
  ```python
  AMOUNT_RE = re.compile(r'\$[\d,]+|\b\d+(?:\.\d+)?\s*(?:million|crore|lakh|USD|INR)\b', re.I)
  DATE_RE   = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
  PCT_RE    = re.compile(r'\b\d+(?:\.\d+)?\s*%')
  ```
- Only fire the LLM contradiction check if extracted values differ
- Pages where all structural values agree skip the LLM pass entirely
- Estimated reduction: ~80% of contradiction check calls eliminated

### C7 — Structured Metadata Pre-Extraction for Review Mode ⬜
**Files:** `wiki.py`, `advanced_modes.py` (uses `page_metadata` table from S1)

**Why:** Review mode fires `N_docs × N_columns` LLM calls every time a job runs. Standard legal
fields (Governing Law, Liability Cap, Jurisdiction, Effective Date, IP Ownership, Termination
Notice, Parties) appear in virtually every document and could be extracted once at ingest time.

**What to build:**
- Expand `INGEST_PROMPT_TEMPLATE` to also return a `metadata` block with ~10 standard fields
- Write extracted fields to the `page_metadata` table (already in S1 schema)
- Review jobs: query `page_metadata` first; only fire an LLM extraction call for
  non-standard or missing columns
- **Effect:** majority of Review cells become DB lookups instead of LLM calls

---

## Phase 5 — Infrastructure ⬜ PLANNED

Operational hardening for multi-user production. Depends on Phases 2–4.

### S6 — Celery Job Queue for Ingest ⬜
**Files:** `app.py`, `wiki.py`, new `celery_app.py`

**Why:** The current global `ThreadPoolExecutor` runs all ingest work inside the Flask process.
No backpressure, no retry, no rate limiting — 20k docs submitted at once spawns thousands of
threads and immediately exhausts the LLM API rate limit.

**What to build:**
- Celery workers (separate processes) with Redis as broker
- One Celery task per document; subtasks per segment
- Configurable concurrency per worker — matches your API rate limit tier
- Automatic retry with exponential backoff on LLM timeouts and rate limit errors
- Task status stored in PostgreSQL `ingest_progress` table (from S5) — no separate
  Redis backend needed for durability, Redis is broker only

---

## Phase 6 — Organisation-Scale Hardening ⬜ PLANNED

Adds the multi-user, multi-team, and high-availability concerns that matter when
the system is shared across a large organisation. Depends on Phases 2–5.

### O1 — Multi-Tenancy: Schema-Per-Team Isolation ⬜
**Files:** `services/db.py`, `config.py`, `app.py`

**Why:** Different teams (M&A, litigation, compliance, procurement) must not see each
other's documents. Row-level security is possible but adds complexity to every query.
Schema-per-tenant gives complete data isolation at the PostgreSQL level with no
application-layer enforcement needed.

**What to build:**
- Each team/organisation gets a dedicated PostgreSQL schema: `CREATE SCHEMA {tenant_id}`
- All tables from S1 live inside the tenant schema — same structure, zero data mixing
- `db.py` takes a `tenant_id` parameter and sets `SET search_path TO {tenant_id}` on
  each connection checkout
- Tenant provisioning: one SQL migration per new team, takes < 1 second
- Cross-tenant analytics (e.g. "how many NDAs across all teams?") goes through a
  dedicated read-only aggregation role that has SELECT on all schemas — never the
  application role

### O2 — Connection Pooling: PgBouncer ⬜
**Files:** deployment config (Docker Compose / Kubernetes)

**Why:** PostgreSQL allows ~100–200 active connections before performance degrades.
A large organisation with many Gunicorn workers + Celery workers will hit this ceiling.
Each SQLAlchemy pool is per-process — 10 Gunicorn workers × pool_size=10 = 100 connections
before a single Celery worker connects.

**What to build:**
- PgBouncer in transaction-pooling mode in front of PostgreSQL
- Application connects to PgBouncer (port 6432); PgBouncer maintains a fixed pool to
  PostgreSQL (e.g. 50 server connections shared across all app connections)
- All `DATABASE_URL` env vars point to PgBouncer, not directly to PostgreSQL
- `pool_pre_ping=True` in SQLAlchemy to handle PgBouncer connection resets

### O3 — Read Replicas for Query Scaling ⬜
**Files:** `services/db.py`

**Why:** Ingest (writes) and query (reads) have very different load profiles. Ingest is
bursty — a team uploads 500 documents, writes spike for an hour, then quiet. Queries
are steady throughout the day. At org scale these two patterns compete for the same
database.

**What to build:**
- Managed PostgreSQL read replica (Azure Flexible Server supports this natively)
- `db.py` exposes two engine instances: `write_engine` (primary) and `read_engine`
  (replica)
- All SELECT-only paths (get_context, query_cache lookup, metadata lookup) use
  `read_engine`
- All INSERT/UPDATE paths (ingest, compaction, cache invalidation) use `write_engine`
- Replication lag is typically < 100ms — acceptable for legal query workloads where
  the user just ingested a document and expects to query it within a few seconds; guard
  with a per-session `after_ingest_at` timestamp and fall back to `write_engine` for
  queries within 5 seconds of an ingest

### O4 — Audit Logging ⬜
**Files:** `services/db.py`, `app.py`

**Why:** Legal work in a large organisation requires a complete audit trail — who queried
what, when, and what answer they received. This is a compliance requirement in most
enterprise legal contexts, not a nice-to-have.

**What to build:**
- `audit_log` table:
  `(id, tenant_id, user_id, session_id, action, question, answer_hash, pages_used,
   token_total, confidence_score, created_at)`
- Write one row per query and per ingest event (document name, page count, duration)
- `answer_hash` is a SHA-256 of the answer text — allows deduplication without storing
  full answers in the audit log
- Retention: configurable TTL; default 2 years for legal compliance
- Expose a read-only `/audit` endpoint (admin role only) for compliance reporting

---

## Dependency Map

```
S1 (PostgreSQL + pgvector)  ←── Required by everything below
├── S2 (FTS cross-ref via tsvector GIN)
├── S3 (page compaction)         ←── requires append_count in S1
├── S4 (contradiction rework)    ←── requires S3
├── S5 (persistent progress)     ←── ingest_progress table in S1
├── C4 (pgvector search)         ←── requires page_embeddings + HNSW index in S1
├── C7 (metadata cache)          ←── requires page_metadata table in S1
└── O1 (multi-tenancy)           ←── requires S1 schema design to support tenant scoping

C3 (NER pre-filter)              ←── no dependencies, can be done anytime
S6 (Celery)                      ←── requires S1 + S5
O2 (PgBouncer)                   ←── requires S1 deployed
O3 (read replicas)               ←── requires S1 + read/write split in db.py
O4 (audit logging)               ←── requires S1 + multi-tenancy awareness from O1
```

---

## Recommended Sequencing

| Order | Task | Effort | Impact |
|---|---|---|---|
| 1 | **S1** — PostgreSQL + pgvector | High | Unblocks everything; eliminates json deadlock |
| 2 | **S5** — Persistent progress | Low | Pairs naturally with S1; survives restarts |
| 3 | **S2** — FTS cross-reference | Low | O(N²) → O(log N); PostgreSQL does the work |
| 4 | **C4** — pgvector page selection | Medium | Fixes query hard-failure; eliminates 80k-token LLM call |
| 5 | **S3** — Page compaction | Medium | Restores wiki quality; required before org-scale ingest |
| 6 | **S4** — Contradiction rework | Medium | Requires S3 |
| 7 | **C3** — NER pre-filter | Low | ~80% reduction in contradiction LLM calls |
| 8 | **C7** — Metadata cache | Medium | Reduces Review mode from N×M calls to DB lookups |
| 9 | **S6** — Celery | High | Production-grade ingest; required before 20k doc load |
| 10 | **O1** — Multi-tenancy | Medium | Required before multiple teams go live |
| 11 | **O2** — PgBouncer | Low | Add when concurrent users > 100 |
| 12 | **O3** — Read replicas | Medium | Add when query load noticeably affects ingest speed |
| 13 | **O4** — Audit logging | Low | Required for legal compliance sign-off |
