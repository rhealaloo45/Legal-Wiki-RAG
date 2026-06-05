# Legal Wiki — Production Scaling Plan

This file tracks all planned and completed engineering work to scale the system
from 50–100 documents to 17–20k documents in a legal production environment.

**Legend:** ✅ Done · 🔄 In Progress · ⬜ Planned

---

## Background

The system works well at 50–100 legal docs per session. At 17–20k docs in
production, two classes of problems emerge:

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
| `MAX_TOKENS_INGEST_DETAIL` | 3500 | Phase-2 segment detail extraction |
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

### Documentation ✅
**Files:** `README.md`, `SYSTEM_OVERVIEW.md`, `SYSTEM_FLOW.md`, `.env.example`

- All docs updated to reflect dual-model routing, merged confidence eval, new env vars,
  corrected model names, and token budget system

---

## Phase 2 — Storage Foundation ⬜ PLANNED

The `index.json` file is the single biggest bottleneck. Everything in Phase 3 and beyond
depends on this being replaced first. **This phase must be completed before any other
structural work.**

### S1 — Replace `index.json` with SQLite ⬜
**Files:** `wiki.py`, `config.py`, new `services/db.py`

**Why it's critical:** At 140k pages, `index.json` will be several GB. Every single ingest
segment deserialises the entire file, mutates it in RAM, and re-serialises it — under a lock
that blocks all concurrent writes. A crash mid-write corrupts the entire session's knowledge
base with no recovery path.

**What to build:**
- New `services/db.py` — SQLite abstraction layer with WAL mode (handles concurrent reads)
- Schema:
  - `pages` table: `(title, content, summary, source_doc, contradiction_flagged, append_count, char_count, last_modified)`
  - `relations` table: `(from_title, to_title, label)` with unique constraint for dedup
  - `page_embeddings` table: `(title, embedding BLOB)` — reserved for Phase 4 vector search
- Replace `_load_index()` / `_save_index()` in `wiki.py` with DB read/write calls
- `_atomic_merge()` becomes a single `INSERT OR REPLACE` / `UPDATE` — no full file cycle
- Migrate existing `index.json` files to SQLite on first load (backward compatibility)
- WAL mode: reads (queries) no longer block writes (ingest) and vice versa
- Crash-safe: SQLite transactions either commit fully or roll back

**Unlocks:** S2, S3, S4, C4, C6, C7 all depend on this.

### S5 — Replace In-Memory `PROGRESS_STORE` with Persistent State ⬜
**Files:** `config.py`, `app.py`, new `services/db.py` (extends S1)

**Why:** `PROGRESS_STORE = {}` is a plain Python dict. Dies on restart, not shared between
Gunicorn workers, grows forever in long-running processes.

**What to build:**
- Add `ingest_progress` table to SQLite DB (from S1): `(session_id, total, current, message, status, started_at, updated_at)`
- Replace all `config.PROGRESS_STORE` reads/writes in `app.py` and `wiki.py`
- Add TTL cleanup: rows older than 7 days auto-purged

---

## Phase 3 — Query Scaling ⬜ PLANNED

Fixes the hard failure at query time when page count exceeds model context limits.
Depends on S1 (SQLite) being complete.

### S7 + C4 — Vector Search to Replace LLM Page Selection ⬜
**Files:** `wiki.py`, `services/embedder.py` (already built, currently unused), `services/db.py`

**Why it's critical:** `_select_relevant_pages()` feeds ALL page titles + summaries to the LLM.
At 140k pages, the index alone exceeds any model's context window by 10–50×. Every query
would throw a context-length error. This is not a cost problem — it's a hard crash.

`embedder.py` is already fully implemented with `embed()` and `embed_batch()`. It just needs
to be connected to the query path.

**What to build:**
- At ingest time: embed each page's summary, store vector in `page_embeddings` table (S1)
- At query time: embed the question → cosine search over stored vectors → return top-25
  candidates — no LLM call at all
- Keep the existing document-mention detection (`_detect_mentioned_files`) — force those
  pages in alongside vector-retrieved supplementary pages
- `_select_relevant_pages()` becomes a vector lookup; remove the LLM call entirely
- **Effect:** page selection goes from 1 LLM call → 0 LLM calls; handles 200k pages in milliseconds

### C6 — Semantic Query Cache ⬜
**Files:** `wiki.py`, `services/db.py`

**Why:** Legal work is highly repetitive — the same questions about liability caps, governing
law, and termination notice periods get asked constantly. The existing "file-back-to-wiki"
mechanic (Q: pages) partially addresses this, but the full page-selection + answer-generation
pipeline still runs on every query.

**What to build:**
- Before running `get_context()`, embed the question and compare against stored embeddings
  of existing `Q:` page titles
- If cosine similarity > threshold (e.g. 0.93) AND wiki hasn't been modified since the cached
  answer was filed, return the cached answer directly — 0 LLM calls
- Cache invalidation: check `last_modified` on pages tagged with the same source docs
- **Legal domain safety:** invalidation must be reliable. A new document that adds a
  contradicting clause must invalidate cached answers about that topic.

---

## Phase 4 — Ingest Quality & Scale ⬜ PLANNED

Fixes the silent quality failures that grow worse as the wiki scales.
Depends on S1 (SQLite).

### S2 — Replace O(N²) Cross-Reference with Inverted Index ⬜
**Files:** `wiki.py`, `services/db.py`

**Why it's critical:** The cross-reference pass in `_merge_wiki()` runs on every single ingest:
```
for title_a in all_pages:        # N iterations
    for title_b in all_pages:    # N iterations each  → O(N²)
        if title_b in content:   # substring search
```
At 140k pages: **19.6 billion substring comparisons per document ingest**, running synchronously
under the session lock. The system stops responding.

**What to build:**
- Maintain an `inverted_index` table in SQLite: `(token, page_title)`
- On new page insert: scan only its content against the existing title set
- On title change: update the inverted index incrementally
- O(new_pages × existing_titles) per ingest instead of O(total_pages²)

### S3 — Page Compaction / Re-Synthesis ⬜
**Files:** `wiki.py`, `services/db.py`

**Why:** Pages grow without bound. "Governing Law (NDA)" after 500 NDAs is 500 content blocks
separated by `---`. The LLM reasoning over this produces degraded, incoherent answers.
The wiki is supposed to get smarter over time — instead it gets worse for the most-used pages.

**What to build:**
- Track `append_count` and `char_count` per page in the DB (already in S1 schema)
- Compaction trigger: when `append_count >= 5` OR `char_count >= 8000`
- Compaction prompt: send all variants to the full model:
  *"These are N versions of the same topic from different documents. Synthesise into one
  coherent wiki page, preserving all distinct facts, noting genuine contradictions explicitly,
  and tagging each claim with its source document."*
- This is the "Lint" operation described in `llm_wiki.md` — currently missing entirely
- For legal: compaction is also where multi-variant contradiction resolution should happen

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
- Store resolved contradictions as structured data: `{claim, value_a, source_a, value_b, source_b}`

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
**Files:** `wiki.py`, `advanced_modes.py`, `services/db.py`

**Why:** Review mode fires `N_docs × N_columns` LLM calls every time a job runs. Standard legal
fields (Governing Law, Liability Cap, Jurisdiction, Effective Date, IP Ownership, Termination
Notice, Parties) appear in virtually every document and could be extracted once at ingest time.

**What to build:**
- Expand `INGEST_PROMPT_TEMPLATE` to also return a `metadata` block with ~10 standard fields
- Store in a `page_metadata` table in SQLite alongside content
- Review jobs: check metadata table first; only fire extraction LLM call for non-standard columns
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
- Task status stored in Redis / SQLite (connects to S5)

---

## Dependency Map

```
S1 (SQLite)  ←── Required by everything below
├── S2 (inverted index)
├── S3 (page compaction)     ←── requires append_count tracking in S1
├── S4 (contradiction rework) ←── requires S3
├── S5 (persistent progress)
├── C4 (vector search)       ←── requires page_embeddings table in S1
├── C6 (semantic cache)      ←── requires C4 embeddings
└── C7 (metadata cache)      ←── requires page_metadata table in S1

C3 (NER pre-filter)          ←── no dependencies, can be done anytime
S6 (Celery)                  ←── requires S1 + S5
```

## Recommended Sequencing

| Order | Task | Effort | Impact |
|---|---|---|---|
| 1 | **S1** — SQLite | High | Unblocks everything |
| 2 | **S5** — Persistent progress | Low | Pairs naturally with S1 |
| 3 | **S2** — Inverted index | Medium | Prevents O(N²) freeze |
| 4 | **C4** — Vector search | Medium | Fixes query hard-failure at scale |
| 5 | **S3** — Page compaction | Medium | Restores wiki quality at scale |
| 6 | **S4** — Contradiction rework | Medium | Requires S3 |
| 7 | **C3** — NER pre-filter | Low | Reduces ingest cost |
| 8 | **C6** — Semantic cache | Medium | Reduces repeat-query cost |
| 9 | **C7** — Metadata cache | Medium | Reduces Review mode cost |
| 10 | **S6** — Celery | High | Production-grade ingest robustness |
