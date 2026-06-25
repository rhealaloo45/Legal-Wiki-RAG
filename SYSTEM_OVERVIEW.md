# System Overview: Legal Wiki RAG

This document describes the full technical architecture of the system — components, data stores, model routing, and the design decisions behind each layer.

For step-by-step pipeline flows, see `SYSTEM_FLOW.md`.
For component and pipeline diagrams, see `ARCHITECTURE.md` and `FLOWCHART.md`.

---

## 1. Architecture summary

The system is a single-process Flask application backed by PostgreSQL + pgvector. There is no separate vector database, no message queue (yet), and no microservices. Intelligence lives in three services: `wiki.py` (ingest + query primitives), `intent_agent.py` (LangGraph query orchestration + intent classification), and `advanced_modes.py` (Review/Compare/Draft).

The `/query` route is a LangGraph `StateGraph`. Each node streams a real-time stage event to the browser via Server-Sent Events; the chat UI turns these into animated progress tiles.

```
Browser (Chat UI)  ◄── Server-Sent Events (stage tiles + answer)
  └── Flask (app.py)
        ├── ThreadPoolExecutor  ──► wiki.ingest()        ──► LLM (full model)
        │                                                 ──► Embedder
        │                                                 ──► PostgreSQL
        ├── /query (SSE)  ──► intent_agent LangGraph:
        │     classify_intent ─► LLM (fast) → 1 of 5 intents (regex fast-path first)
        │     disambiguation  ─► wiki.classify_query()    ──► LLM (fast) → document chips
        │     clarification   ─► wiki.check_ambiguity()   ──► LLM (fast) → clarifying question
        │     retrieve        ─► wiki.get_context()       ──► pgvector + BM25 + entity matching
        │     generate        ─► wiki.generate_answer()   ──► LLM (full: intent-specific prompt)
        │     validate        ─► format check (logged, non-blocking)
        ├── /messages           ──► chat_messages table   ──► conversation history
        ├── /document/locate    ──► source_positions      ──► quote position lookup
        ├── /review, /compare   ──► advanced_modes        ──► LLM (fast model)
        └── /draft              ──► advanced_modes         ──► LLM (full model)

PostgreSQL
  ├── pages              (title, content, summary, source_doc, append_count, char_count,
  │                       contradiction_flagged, variants, content_tsv [GIN index])
  ├── page_embeddings    (session_id, title, embedding VECTOR)
  ├── relations          (session_id, from_title, to_title, label)
  ├── page_metadata      (session_id, title, governing_law, jurisdiction, parties, ...)
  ├── contradictions     (session_id, page_title, claim, value_a, source_a, value_b, source_b)
  ├── ingest_progress    (session_id, total, current, message, status)
  ├── chat_messages      (session_id, role, content, msg_type, metadata JSONB)
  └── source_positions   (session_id, source_doc, page_num, char_start, char_end)
```

---

## 2. LLM routing

Two model tiers, one `ask()` function:

| Tier | Azure config | OpenRouter config | Used for |
|---|---|---|---|
| **Full** | `AZURE_OPENAI_DEPLOYMENT` | `OPENROUTER_MODEL` | Ingest synthesis, answer generation, compaction, drafting, compare narrative |
| **Fast** | `AZURE_FAST_DEPLOYMENT` | `OPENROUTER_FAST_MODEL` | Cell extraction (Review/Compare), JSON repair, contradiction pre-flight, document disambiguation, query clarification, intent classification |

All calls: `temperature=0.0` (deterministic). Explicit `max_tokens` cap on every call — no silent 4096-token defaults.

Fast client has a 45s timeout and 1 retry (bulk extraction tolerance). Full client has 120s timeout and 2 retries (synthesis tasks can be slow).

---

## 3. Embedding

- Provider: Azure (`text-embedding-3-large`) or OpenRouter (`nvidia/llama-nemotron-embed-vl-1b-v2:free`)
- OpenRouter embeddings use direct `requests.post()` — the OpenAI SDK mis-parses OpenRouter's response envelope for this model
- Embeddings are 2048-dimensional (Nemotron) or 1536-dimensional (Azure)
- Query embeddings use `"search_query:"` prefix; document embeddings use `"search_document:"` prefix
- Batch size: 16 texts per API call
- HNSW index created if `EMBEDDING_DIMENSIONS ≤ 2000`; otherwise exact cosine scan

---

## 4. Database schema

### `pages` table
The core knowledge store. One row per wiki page per session.

| Column | Type | Notes |
|---|---|---|
| `session_id` | TEXT | Isolates sessions |
| `title` | TEXT | Unique per session. Format: `"Topic – Case Name (Doc Type)"` for case-specific pages; `"Topic (Doc Type)"` for shared concept pages |
| `content` | TEXT | Full synthesised content. Multiple sources separated by `---\n*[From: doc_name]*` |
| `summary` | TEXT | One-line summary, used for embedding and BM25 pre-filter |
| `source_doc` | TEXT | Last document to update this page |
| `append_count` | INT | Number of times content has been merged from a new document |
| `char_count` | INT | Current content length in characters |
| `contradiction_flagged` | BOOL | Set by C3 NER pre-filter when structural values differ |
| `variants` | JSONB | Snapshot of each version before merge (used by compaction) |
| `content_tsv` | TSVECTOR GENERATED | Auto-generated from `content`, GIN-indexed for O(log N) FTS cross-reference |

### `page_embeddings` table
One row per page. Embedding stored as `VECTOR(dims)`. Used by pgvector for cosine similarity search at query time.

### `page_metadata` table
Structured extraction of standard legal fields from each document, written once at ingest. Review Mode reads from here instead of calling the LLM.

Fields: `governing_law`, `jurisdiction`, `effective_date`, `termination_notice`, `liability_cap`, `ip_ownership`, `parties`, `auto_renewal`, `notice_period`, `payment_terms`.

### `contradictions` table
Structured record of detected contradictions: `(session_id, page_title, claim, value_a, source_a, value_b, source_b)`. Populated during page compaction.

### `ingest_progress` table
Per-session ingest progress for UI polling. Survives process restarts (unlike the old in-memory `PROGRESS_STORE`).

---

## 5. Wiki ingest pipeline

### 5.1 Document ingestion strategy

The system adapts to document length:

**Short documents (≤100K chars):**
One LLM call with `INGEST_PROMPT_TEMPLATE`. Returns `{doc_type, metadata, pages, relations}` in a single JSON object.

**Long documents (>100K chars):**
- Phase 1 — Overview: reads the beginning + end of the document, identifies the document type and a list of all topics that need wiki pages. Uses `OVERVIEW_PROMPT_TEMPLATE` (1,500 token budget).
- Phase 2 — Detail: the document is split into 40K-char segments with 500-char overlap. Each segment is processed in parallel by `DETAIL_PROMPT_TEMPLATE` (3,500 token budget), which creates/updates pages for topics found in that segment. Runs in `ThreadPoolExecutor` with up to `WIKI_MAX_WORKERS` threads.

### 5.2 Page title rules

Two categories, enforced by the ingest prompt:

**Case-specific pages** — prefixed with a short case identifier derived from the parties' names. These pages must not be merged across different cases:
```
Facts – Yuvraj Kanther (Court Judgment)
Procedural History – Yogesh Kumar (Court Judgment)
Holding – Yerikala Sunkalamma (Court Judgment)
Charges – Yuvraj Kanther (Court Judgment)
```

**Shared legal concept pages** — no case prefix. These pages are intentionally merged when multiple documents discuss the same concept, accumulating knowledge across all cases:
```
Section 319 CrPC (Court Judgment)
Hardeep Singh v. State of Punjab (Court Judgment)
Section 80 CPC (Supreme Court Judgment)
```

### 5.3 Atomic merge (`_atomic_merge_db`)

Called for each ingest result under a per-session lock. For each page in the new data:

1. **Lookup**: check if the page already exists in the DB
2. **New page**: `upsert_page()` with content, summary, and source doc
3. **Existing page**:
   - **C3 NER pre-filter**: extract amounts, dates, percentages from both versions via regex. If structural values differ → set `contradiction_flagged = True`, append a variant snapshot to `variants` JSONB
   - **Merge**: `existing_content + "\n\n---\n*[From: {clean_doc_name}]*\n\n" + new_content`
   - Increment `append_count`, update `char_count`
4. **C7 metadata**: upsert document-level metadata fields to `page_metadata`
5. **S2 cross-reference** (outside lock, after all pages merged): for each new page, use `content_tsv @@ plainto_tsquery()` to find other pages that mention this page's title; add bidirectional relations

### 5.4 Embedding (outside lock)

After the merge lock is released, all updated pages are embedded in batch and stored in `page_embeddings`. This keeps HTTP embedding calls out of the critical section.

### 5.5 Page compaction (S3)

Run once at the end of each document's ingest. Finds pages where:
- `append_count >= COMPACTION_APPEND_THRESHOLD` (default 5), **OR**
- `append_count >= 2 AND char_count >= COMPACTION_CHAR_THRESHOLD` (default 8,000)

The `append_count >= 2` guard prevents freshly ingested single-version pages from being compacted regardless of size.

For each due page, `_compact_page()`:
1. Sends all content variants to the full LLM with `COMPACTION_PROMPT_TEMPLATE`
2. LLM re-synthesises into a single coherent page, noting genuine contradictions with source attribution
3. Resets: `content = new_content`, `append_count = 0`, `variants = NULL`
4. Re-embeds the compacted content
5. Extracts and stores any contradictions found to the `contradictions` table

---

## 6. Query pipeline

### 6.1 Page selection (hybrid retrieval)

With a wiki of potentially thousands of pages, the system uses a three-path cascade:

**Path 1 — Hybrid vector + BM25** (primary, DB mode):
1. Embed the question with `"search_query:"` prefix
2. pgvector cosine similarity → top-15 pages (`VECTOR_SEARCH_TOP_K`)
3. BM25 keyword scoring over all page summaries → top-8 supplement (`HYBRID_BM25_SUPPLEMENT_N`)
4. Merge: vector results first, BM25 pages appended if not already present
5. Result: up to 23 pages, 0 LLM calls, ~5ms DB query

**Path 2 — BM25 + LLM selection** (fallback if no embeddings):
1. BM25 pre-filter: top-150 candidates (`PAGE_SELECTION_PREFILTER_N`)
2. Full LLM call with page titles + summaries → returns JSON list of selected titles

**Path 3 — BM25 keyword-only** (fallback if LLM fails):
Returns top-25 pages by keyword overlap score.

### 6.2 Context building

For each selected page:
- Regular pages: capped at `MAX_PAGE_CONTEXT_CHARS` (2,000 chars)
- Cached-answer pages (titles starting with `Q:`): capped at `MAX_QPAGE_CONTEXT_CHARS` (3,000 chars)
- Pages with `contradiction_flagged = True`: prepend `[WARNING: This page contains conflicting claims...]`

### 6.3 LangGraph orchestration & intent classification

The `/query` route is a LangGraph `StateGraph` (`intent_agent.py`) with six nodes and conditional edges:

```
classify_intent → disambiguation → clarification → retrieve → generate → validate
                       │ (needs)        │ (needs)
                       ▼                ▼
                      END              END
```

1. **classify_intent** — classifies the query into one of five lawyer intents: `factual`, `risk_assessment`, `comparison`, `obligation`, `drafting`. Regex fast-path (0 tokens) handles obvious queries; ambiguous queries fall to a fast LLM call (`MAX_TOKENS_INTENT_CLASSIFY` = 150). Any failure defaults to `factual`.

2. **disambiguation** — skips if the question names a document (numbered pattern, entity+doc-type pattern, or known entity from page titles via `_question_mentions_known_entity()`). Otherwise a fast LLM call checks for vague document references.

3. **clarification** — fast LLM call determines if the question needs one clarifying question. Hard limit: 1 per turn.

4. **retrieve** (`wiki.get_context()`) — hybrid page selection, tuned by per-intent `retrieval_hints`. Entity-aware retrieval: when the question mentions a distinctive entity/party name from page titles (e.g. "ReVolt"), those pages are force-included even if the source filename doesn't match. Numbered doc-type patterns ("service agreement 3") also match via enhanced `_detect_mentioned_files()`.

5. **generate** (`wiki.generate_answer(intent=...)`) — selects the intent-specific prompt.

6. **validate** — light format check (comparison → table present; obligation → list/table; drafting → clause formulations). Non-blocking.

Each node emits a custom stage event via LangGraph's stream writer; `app.py` relays these as SSE.

### 6.4 Answer generation — intent-specific prompts

| Intent | Prompt | Output shape |
|---|---|---|
| `factual` | `ANSWER_PROMPT` | Direct answer, 20+ precision rules |
| `risk_assessment` | `ASSESSMENT_PROMPT` | Reasoned judgment, risk classification, recommendation |
| `comparison` | `COMPARISON_PROMPT` | Side-by-side table + key differences + who-it-favors |
| `obligation` | `OBLIGATION_PROMPT` | Duty/deadline table (party · duty · trigger · consequence · clause) |
| `drafting` | `DRAFTING_PROMPT` | Aggressive/balanced/conservative clause formulations with implications |

All five prompts include:
- **Conversation history** block (last 3-5 exchanges, max ~2000 chars)
- **Document metadata** block (party names, governing law, effective dates from `page_metadata`)
- Full wiki context (capped pages, contradiction warnings, source labels)
- Chain-of-thought `<reasoning>` block with confidence score — zero additional LLM calls

The reasoning-block extraction uses a tolerant regex that handles unclosed tags, whitespace variants, and unicode characters. The classified intent is returned in the answer payload and surfaced as a coloured tag on the answer card.

---

## 7. Advanced modes

### Review Mode
User specifies column names. For each (document, column) pair:
1. Check `page_metadata` table for cached standard fields (`governing_law`, `parties`, etc.)
2. If cached and matching → return cached value (confidence 0.95), skip LLM call
3. Otherwise: fetch wiki text for that document → `extract_cell(doc_text, column_name)` via fast LLM
4. Returns `{value, confidence, quote}`

Cells run concurrently via `ThreadPoolExecutor`. Result exported to confidence-colour-coded Excel.

### Compare Mode
1. Fast LLM identifies aspects to compare across the selected documents
2. Concurrent `extract_cell()` for each (aspect, document) pair
3. Outlier detection: flag documents that differ significantly from the consensus on each aspect
4. Full-model narrative synthesis across all aspects
5. Excel export (aspects × documents matrix)

### Draft Mode
1. Classify the request (new draft vs refinement)
2. Detect stance (party A / party B / neutral)
3. Optionally ground the draft in relevant wiki pages
4. Full-model draft generation
5. Iterative refinement loop
6. DOCX export via `python-docx`

---

## 8. Concurrency model

- **Ingest**: `ThreadPoolExecutor(max_workers=WIKI_MAX_WORKERS)` in Flask process. Each document is one task. Segments of a large document run as sub-tasks within the same executor.
- **Session locks**: `threading.Lock` per `session_id` protects the atomic merge step. Embedding calls happen outside the lock.
- **Review/Compare**: `ThreadPoolExecutor` for concurrent cell extraction across (document × column) pairs.
- **No Celery yet**: all work runs inside the Flask process. Phase 5 will move ingest to Celery workers for backpressure, retry, and rate-limit compliance.

---

## 9. Token budget summary

| Call | Model tier | Budget |
|---|---|---|
| Ingest: single-call synthesis | Full | 4,096 |
| Ingest: overview pass | Full | 1,500 |
| Ingest: detail segment | Full | 3,500 |
| Page compaction | Full | 4,096 |
| Answer generation | Full | 4,096 |
| JSON repair | Fast | 2,048 |
| Cell extraction (Review/Compare) | Fast | 300 |
| Document disambiguation | Fast | 200 |
| Query clarification | Fast | 300 |
| Intent classification (LLM fallback) | Fast | 150 |
| Page selection (BM25+LLM fallback) | Full | 1,000 |

---

## 10. Session isolation

All data is scoped by `session_id` (UUID). Each session has:
- Its own rows in all PostgreSQL tables (filtered by `session_id`)
- Its own upload prefix (`{session_id}_{filename}`)
- Its own log file (`data/logs/{session_id}_log.md`)
- Its own per-session threading lock

Sessions are listed in `data/sessions.json` with display names.
