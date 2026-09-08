# Legal Wiki RAG

A production-grade legal intelligence platform that transforms unstructured legal documents into a structured, queryable knowledge base using LLM-driven wiki synthesis — not retrieval-augmented generation.

---

## What makes this different from RAG

Standard RAG retrieves raw document chunks at query time and asks the LLM to reason over them on the fly. Legal Wiki builds a **persistent structured wiki** at ingest time. Each document enriches the same knowledge base: pages are merged, contradictions flagged, cross-references built, and shared legal concepts accumulated across all cases. Queries read from pre-compiled synthesis, not raw text.

| | Standard RAG | Legal Wiki |
|---|---|---|
| Query cost | ~2,000–5,500 tokens | 21,007 prompt / 1,091 completion, mean over 2,527 traced queries — **0 tokens** on the 14 deterministic paths |
| Cross-doc synthesis quality | Weak | Strong (pre-built) |
| At 20k docs | Recall degrades | Consistent (compaction keeps pages coherent) |
| Ingest cost | Near zero | LLM calls per doc (paid once, queried many times) |
| Hallucination surface | Raw chunks, no pre-filtering | Pre-synthesised + 20 answer-prompt rules |

The query-cost row is measured, not estimated, and it is higher than the figure
this table used to carry. The interesting number is the second half: on a
48-question suite spanning sixteen question shapes, **28 of 48 answers cost
nothing at all**, because the question was one the database answers exactly.

---

## Current state — what is on, what is off

Every flag below is read at runtime; the defaults are what ships. Nothing here
is aspirational — a feature that is off is described as off, and why.

| | State | Notes |
|---|---|---|
| `ENABLE_INTENT_CLASSIFIER` | **on** | Five-intent routing; regex fast path first, fast model for the rest |
| `ENABLE_CLARIFICATION` | **on** | Skipped for corpus-wide questions and for follow-ups on a settled document — it is a model call and it decided borderline follow-ups both ways |
| `ENABLE_ANSWER_VALIDATION` | **on** | Deterministic grounding first; escalates to a judge only when inconclusive |
| `ENABLE_TERM_CHECK` | **on** | Defined-term drafting-defect check |
| `ENABLE_GENERAL_KNOWLEDGE` / `ENABLE_GK_LLM_FALLBACK` | **on** | Answers outside the corpus, clearly separated from document answers |
| `ENABLE_META_LLM_FALLBACK` | **on** | Questions about the corpus itself |
| `USE_QUESTION_EMBEDDINGS` | **on** | Third retrieval channel — hypothetical questions per page |
| `PII_REDACTION_ENABLED` | **on** | |
| `AUTH_ENABLED` | **on** | |
| `ENABLE_QUERY_DECOMPOSITION` | **on** | Runs in one place only: *after* the whole-question route has already declined. Sub-questions route through the zero-LLM structured paths, so a rescue costs nothing beyond its SQL. Placed there deliberately — anywhere earlier it could hand a fragment to a resolver that has lost the party pair it needed |
| `ENABLE_RERANK` | **off** | The reranker orders candidates by `title: summary` — metadata, not content — so it cannot break the ties that actually matter. Enabling it adds a model call without addressing the failure. A cross-encoder scoring question and passage jointly is the thing that would |
| `ENABLE_ENQUIRY_AGENT` | **off** | Built, works, and measured twice as contributing nothing. On a 49-turn thread suite, carryover had already resolved 36 of 39 follow-ups, so the gate never opened — zero enquiry referents used, 20.4 minutes against 16.5. A 76-question run agreed independently. Kept as insurance for referent shapes the deterministic resolvers do not cover, not because it is believed to help |
| `ENABLE_ANSWER_CACHE` | **off** | |
| `STRUCTURAL_VISION_ENABLED` | **off** | Vision pass for structural parsing; not needed at this corpus's quality |
| `DISABLE_INGEST` | **off** | Ingest is available |
| HNSW vector index | **unavailable** | Not a flag. pgvector indexes at most 2,000 dimensions and these are 3,072, so every dense search is an exact cosine scan — perfect recall, no tuning, and fine at this corpus size |

---

## Features

### Wiki Pipeline
- **Adaptive ingest**: single LLM call for short docs (≤100K chars); two-phase overview + parallel segmentation for long docs
- **Case-aware page titling**: case-specific pages (facts, holding, charges, procedural history) isolated per case; shared legal concepts (statutes, precedents, doctrines) merged and enriched across all cases
- **Source attribution**: merged pages carry `[From: ...]` labels so the answer model attributes correctly without cross-case contamination
- **Hybrid retrieval**: pgvector cosine search + BM25 keyword ranking fused via Reciprocal Rank Fusion (RRF) at query time; a wider candidate pool + per-document diversification kicks in for broad/family questions, with an optional (off by default) LLM rerank pass
- **Page compaction**: pages that grow through repeated merges are re-synthesised by the LLM, keeping the wiki coherent at scale
- **D3.js knowledge graph** visualisation of pages and relations

### Deterministic paths — questions the database answers, not the model

Before the query graph runs, the question is tested against **14 fast paths**.
Each answers from SQL or an index lookup and returns immediately, at zero
tokens. This is not an optimisation bolted on for speed: routing these through
retrieval produced answers that were specific, sourced and wrong, because
retrieval answers from what it fetched and cannot report that it stopped early.

| Path | Answers | Was |
|---|---|---|
| **Counting** | "How many Shareholder Agreements do we hold?" · "How many contracts mention arbitration?" | Retrieval counted its own retrieved pages: *"There are 21 documents"* against a corpus of 1,372; *"fifteen"* contracts mention arbitration where 701 do |
| **Document set** | "Show every Shareholder Agreement where X is the lead strategic shareholder" | Named one of the five that match — and a set answered from a sample reads exactly like a complete set |
| **Compound filter** | "Which joint ventures involve X *and* list sanctions exposure as an exit trigger?" | Reported none; seven match. Retrieval ranks by one notion of relevance, so a two-condition question gets whichever condition the embedding weighted |
| **Analytics** | Aggregates, gaps ("which contracts state no governing law"), trends by year, expiry windows | A negative over a corpus cannot be established from a sample of pages |
| **Relations** | Amendment chains and incoming/outgoing edges, reachable by date as well as title | Reported a document was not in the corpus while holding both it and its `amends` edge |
| **Citations** | "Which documents cite the Trade Marks Act, 1999?" — statutes *and* procedural authorities like `Order XXXIX CPC` | Ranked documents *about* injunctions over the ones carrying the citation |
| **Calculation / date math** | Term lengths, elapsed and remaining days, business-day offsets against a per-document holiday calendar | |
| **Clause precedent** | "Have we agreed to a 30-day notice period before?" — split into clauses that state that figure and clauses that state a different one | Headed a 45-day clause as "matching" a 30-day question |
| **Playbook compliance** | "Does this comply with our house standard?" | |
| Plus | social, meta, defined terms, conflation, absent-instrument | |

Every one of these reports the denominator it could not test — documents with
no recorded expiry date, no governing law, no term length — rather than
dropping them silently. A filter that quietly discards what it could not
evaluate reports a small set as though it were the whole answer.

**Position in a thread does not change what a question is.** These paths were
originally skipped on any follow-up turn. Asked third in a thread, "which
agreements expire in the next 90 days" bypassed the expiry index, was answered
over the twelve documents the conversation happened to hold, and opened by
repeating the previous answer — 49,210 tokens to get wrong what the index
answers exactly, free, in under a second. A question carrying no anaphora now
reaches the index whatever its position; "which of *those* expire…" still takes
the conversational path, because it needs it.

### Query Mode (Ask) — Conversational Chat
- **Chat interface** with persistent message history (PostgreSQL-backed). Full conversation thread with user messages, assistant answers, disambiguation prompts, and clarification questions
- **Intent classifier agent** (LangGraph): every query is classified from a lawyer's perspective into one of five intents — **factual**, **risk_assessment**, **comparison**, **obligation**, **drafting** — which selects a tailored prompt template. Regex fast-path (0 tokens) handles obvious queries; a fast LLM call resolves ambiguous ones and falls back to `factual`. Controlled by `ENABLE_INTENT_CLASSIFIER`
- **Streaming progress stages**: the `/query` endpoint streams Server-Sent Events as the LangGraph pipeline runs (classifying → intent identified → retrieving → pages retrieved → generating → validating). The chat UI renders these as animated vertical stage tiles, then the answer card carries a coloured **intent tag** alongside the confidence badge
- **Document disambiguation**: when a query targets an unspecified document, system asks user to pick from ingested docs or upload a new one. Skips automatically when the user names a specific document ("service agreement 1"), mentions a known entity/party name ("ReVolt JV Agreement"), uses broad/collective phrasing ("across all NDAs"), or names a counterparty whose full-text content search resolves to exactly one document
- **Follow-up clarification**: for ambiguous queries, system asks one clarifying question with suggested options before answering. Controlled by `ENABLE_CLARIFICATION` env var
- **Scope resolution**: a dedicated `resolve_scope` step pins retrieval to a single document, a document family, or the whole corpus — via a priority cascade of named-file detection, party-name content search, known-entity matching, and broad-phrasing/family detection — before the hybrid retriever runs
- **Five intent-specific prompt modes**: `ANSWER_PROMPT` (factual, 20+ precision rules), `ASSESSMENT_PROMPT` (risk/go-no-go, reasoned judgment), `COMPARISON_PROMPT` (side-by-side table), `OBLIGATION_PROMPT` (duty/deadline checklist), `DRAFTING_PROMPT` (clause text with aggressive/balanced/conservative formulations) — all grounded in context, and all sharing table-citation discipline rules so tabular outputs aren't exempt from inline citations
- **Document metadata injection**: party names, governing law, effective dates from `page_metadata` are injected into the prompt so answers use actual party names instead of generic "Service Provider"
- **Entity-aware retrieval**: when the question mentions a distinctive entity/party name from page titles (e.g. "ReVolt", "Meridian"), those pages are force-included in context even if the source filename doesn't match
- Chain-of-thought reasoning with self-assessed confidence score (0–100) extracted from `<reasoning>` block — zero extra LLM calls. Tolerant regex handles unclosed tags and unicode variants
- Per-page context cap (2,000 chars) prevents any single large merged page from crowding out others
- **Improved citations**: references include PDF page numbers, verbatim quotes, and clause references. Citation clicks open the source PDF to the correct page with highlighted text
- **Deterministic citation verification**: after generation, every quoted span in the answer is checked verbatim against the retrieved context (`_verify_answer_citations`) and cross-checked against the document it's attributed to (`_verify_citation_attribution`) — regex/substring checks, not another LLM's opinion. Failures trigger one corrective-retry pass; if issues remain, `[CITATION WARNING: ...]` / `[ATTRIBUTION WARNING: ...]` banners are appended to the answer
- **Grounding check**: a separate LLM audit call (`_check_grounding`, independent of the answer LLM's self-reported confidence score) scores how well the finished answer's factual claims are supported by context, treating "not covered" findings as correct behaviour rather than a violation. Confidence (self-reported) and grounding (independently audited) are two different numbers shown to the user. Controlled by `ENABLE_ANSWER_VALIDATION`

### Review Mode
- User defines columns (e.g. "Governing Law", "Liability Cap"); system extracts values from all uploaded documents concurrently
- Metadata cache: standard legal fields extracted once at ingest time and served from DB — repeated jobs do not re-call the LLM for cached fields
- Confidence-colour-coded Excel export (green ≥ 0.8, yellow ≥ 0.5, red < 0.5)

### Compare Mode
- Automatic aspect identification across selected documents
- Outlier detection with narrative synthesis
- Excel export (aspects as rows, documents as columns)

### Draft Mode
- Context-aware legal document drafting grounded in wiki knowledge
- Stance detection (favours party A / party B / neutral)
- Iterative refinement with DOCX export

### Infrastructure
- **PostgreSQL + pgvector**: single managed store for pages, embeddings, metadata, progress, contradictions, relations, chat messages, source positions
- **Dual-model routing**: full model for synthesis/answers, fast model for contradiction checks/cell extraction/JSON repair/disambiguation/clarification/intent classification
- **LangGraph query orchestration**: the Ask pipeline is a `StateGraph` with conditional edges; nodes stream real-time stage events to the browser via Server-Sent Events
- **SQLAlchemy connection pool** (pool_size=10, max_overflow=20)
- Per-session threading locks for race-free parallel ingest
- OCR fallback via Tesseract for scanned PDFs
- **Page-level position tracking**: PDF page numbers and character offsets stored at ingest for precise citation linking

---

## Setup

### Requirements
- Python 3.9+
- PostgreSQL 14+ with `pgvector` extension
- Azure OpenAI, OpenRouter, **or** NVIDIA NIM API credentials
- Tesseract (optional, for scanned PDFs)

### Install

```bash
git clone <repo>
cd Legal-Wiki-RAG
pip install -r requirements.txt
```

### Configure

```bash
cp app/.env.example app/.env
```

Key variables in `app/.env`:

```env
# Provider: "azure", "openrouter", or "nvidia" — independently selectable for LLM vs embeddings
LLM_PROVIDER=nvidia
EMBEDDING_PROVIDER=nvidia

# Azure OpenAI — GPT-5.x reasoning deployments (both tiers currently point at gpt-5.4-mini)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://....openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5.4-mini
AZURE_FAST_DEPLOYMENT=gpt-5.4-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# OpenRouter (fallback)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o
OPENROUTER_FAST_MODEL=google/gemma-4-27b-it
OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free

# NVIDIA NIM
NVIDIA_API_KEY=...
NVIDIA_MODEL=openai/gpt-oss-120b
NVIDIA_FAST_MODEL=openai/gpt-oss-20b
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embed-v1
NVIDIA_EMBEDDING_DIMENSIONS=4096

# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/legalwiki
```

> **Azure GPT-5.x note:** GPT-5.x / o-series reasoning deployments reject the classic `max_tokens` and `temperature` chat-completion params — they require `max_completion_tokens` and no explicit `temperature`. `llm.py`'s `_is_azure()` / `_completion_kwargs()` helpers build the right kwargs per provider automatically; the nvidia/openrouter OpenAI-compatible endpoints keep `temperature=0.0` + `max_tokens` unchanged.

> **Runtime provider switch:** `/api/settings/llm` and `/api/settings/embedding` let you switch providers from the UI without restarting — but the change is in-memory only (`config.LLM_PROVIDER` / `config.EMBEDDING_PROVIDER`) and reverts on process restart; it does not persist to `.env`. Separately, the Flask server runs with `FLASK_USE_RELOADER=0` by default (so long-running ingests survive file saves), which means any edit to `.env`, `prompts.py`, `wiki.py`, `intent_agent.py`, or `llm.py` requires a **manual process restart** to take effect.

### Switching branches with multiple databases

`app/.env` is git-ignored, so it does **not** follow the checked-out branch automatically — `git checkout` never touches it. If you keep more than one branch pointed at different databases (e.g. `optimize` → `legal_wiki`, `target-architecture-v2` → `legalwiki_v2`), a per-branch `.env.<branch>` file holds that branch's `DATABASE_URL`, and you copy the right one over `app/.env` by hand after switching:

```powershell
git checkout optimize
copy app\.env.optimize app\.env

git checkout target-architecture-v2
copy app\.env.v2 app\.env
```

Or use the helper script, which does the same copy:

```powershell
.\app\switch-env.ps1 -Branch optimize
.\app\switch-env.ps1 -Branch v2
```

There is no git hook or auto-swap — forgetting this step means `app/.env` still points at whichever database you last selected, regardless of which branch is checked out.

### Database Setup

The app uses **PostgreSQL + pgvector** as its primary store. Tables are created automatically on first connect — no manual migrations needed.

#### Quick start with Docker

```bash
# Pull and run pgvector (PostgreSQL 17 with vector extension pre-installed)
docker run -d \
  --name legal-wiki-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=pw \
  -e POSTGRES_DB=legal_wiki \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

> **Port conflict?** If port 5432 is already in use (e.g. a local Postgres install), map to a different host port:
> ```bash
> docker run -d --name legal-wiki-pg \
>   -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=pw -e POSTGRES_DB=legal_wiki \
>   -p 5433:5432 pgvector/pgvector:pg17
> ```
> Then update your `.env`: `DATABASE_URL=postgresql://postgres:pw@localhost:5433/legal_wiki`

#### Set the connection string

Add this to `app/.env`:

```env
DATABASE_URL=postgresql://postgres:pw@localhost:5432/legal_wiki
```

#### What happens on first connect

When the app starts and `DATABASE_URL` is set, [`db.py`](app/services/db.py) automatically:

1. Enables the `vector` extension (`CREATE EXTENSION IF NOT EXISTS vector`)
2. Creates all tables (`IF NOT EXISTS` — safe to run repeatedly):

| Table | Purpose |
|---|---|
| `pages` | Wiki pages with content, summary, source attribution, contradiction flags, FTS tsvector column |
| `relations` | Cross-reference edges between pages (from → to, label) |
| `page_embeddings` | pgvector embeddings for cosine similarity search (HNSW-indexed for dims ≤ 2000) |
| `ingest_progress` | Real-time ingest progress tracking per session |
| `page_metadata` | Cached document-level metadata (governing law, jurisdiction, parties, etc.) |
| `contradictions` | Structured contradiction records detected during compaction |
| `chat_messages` | Persistent chat history per session (user, assistant, system messages with metadata) |
| `source_positions` | PDF page-level character offsets for precise citation linking |

3. Creates GIN index on `content_tsv` for full-text cross-referencing
4. Creates HNSW index on embeddings for sub-5ms vector search (auto-skipped if embedding dimensions > 2000)

#### File-mode fallback

If `DATABASE_URL` is **not set**, the app falls back to file-based storage (`data/index.json`). This mode works for single-user local testing but does not support:
- pgvector hybrid retrieval
- Page compaction
- FTS cross-referencing
- Metadata caching

#### Useful Docker commands

```bash
# Check container status
docker ps -a | grep legal-wiki-pg

# View logs
docker logs legal-wiki-pg

# Stop / start
docker stop legal-wiki-pg
docker start legal-wiki-pg

# Connect with psql (inside container)
docker exec -it legal-wiki-pg psql -U postgres -d legal_wiki

# Remove container and data (destructive)
docker rm -f legal-wiki-pg
```

### Run

```bash
cd app
python app.py
# Open http://localhost:5001
```

---

## Testing

Three suites, each answering a question the others cannot. None is sufficient
alone, and that has been demonstrated rather than argued.

**Deterministic checks — no model calls, seconds.** 37 assertions over the fast
paths: does each answer with the right value, and — nine of the 37 — does it
correctly *decline* to capture a question it should leave to retrieval. This
catches the largest class of regression, a fast path swallowing a question it
should not. A party-pair lookup ("what is the term of the NDA between X and Y")
was captured mid-review by the compound branch through the bare "and" joining
two party names, and answered with a list of NDAs instead of the term; it is now
pinned by test.

**Conversation suite — `tools/conversation_suite.py`, about five minutes.** Six
scripted conversations asserting, per turn, which path answered, whether it cost
anything, what the text must and must not say, and which document it must still
be about. Turns marked *free* must cost zero tokens: each is a question the
index answers exactly, and one that starts costing tokens has stopped reaching
the index and is about to start being wrong.

**Thread suite — `tools/thread_suite.py`.** Generates threads from
single-document seeds and checks one thing: does a follow-up stay on the
document the thread pinned. Older, complementary, and would not have caught any
of the three defects the conversation suite exists for.

Above these sit the graded runs — a 200-question set, a fresh 300-question set,
a 76-question run over this deployment's own documents, and a 48-question set
covering sixteen question *shapes* rather than sixteen questions.

> **What a benchmark of independent questions cannot see.** The 48-question
> suite scored 9.1/10, and three defects reached a real session immediately
> afterwards — every one requiring a second turn to exist at all, because the
> suite starts a fresh thread per question. Suite size does not help; only
> testing sequences does. That is what the conversation suite is for.

Suite output files are gitignored: they carry corpus document names and answer
text, and this repository is public.

---

## Project structure

```
app/
├── app.py                   # Flask routes + ThreadPoolExecutor ingest scheduler
├── config.py                # All constants and env-var loading
├── services/
│   ├── db.py                # PostgreSQL abstraction layer (SQLAlchemy)
│   ├── wiki.py              # Ingest pipeline, query pipeline, compaction
│   ├── intent_agent.py      # LangGraph query orchestration (7 nodes) + 5-intent classifier + grounding check
│   ├── llm.py               # Dual-model LLM routing (Azure / OpenRouter / NVIDIA NIM), provider-aware kwargs
│   ├── embedder.py          # Embedding service (batch, query/doc mode prefixes)
│   ├── prompts.py           # All prompt templates (ingest, answer, assessment, comparison, obligation, drafting, compaction)
│   ├── advanced_modes.py    # Review, Compare, Draft shared logic + cell extraction
│   └── reader.py            # PDF/DOCX text extraction with OCR fallback
├── data/
│   ├── uploads/             # Uploaded files ({session_id}_{filename})
│   ├── logs/                # Per-session markdown event logs
│   └── sessions.json        # Session name/metadata store
└── templates/               # Single-page HTML + Bootstrap 5 + D3.js
```

---

## Key configuration constants

| Constant | Default | Effect |
|---|---|---|
| `VECTOR_SEARCH_TOP_K` | 15 | pgvector nearest-neighbour candidates per query (non-broad); widened to `BROAD_QUESTION_VECTOR_TOP_K` (80) for broad/family questions |
| `HYBRID_BM25_SUPPLEMENT_N` | 8 | Legacy BM25-supplement constant; current fusion pulls a BM25 ranking sized to match the vector candidate list, then fuses via RRF (see below) |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant — merges the vector and BM25 rankings; higher = flatter weighting of rank position |
| `HYBRID_FUSION_TOP_K` | 23 | Final page budget after RRF fusion for a non-broad query |
| `BROAD_QUESTION_PER_DOC_CAP` / `BROAD_QUESTION_TOTAL_CAP` | 4 / 60 | Per-document cap and total page budget after diversifying a broad/family question's fused candidate list |
| `ENABLE_RERANK` | false | Optional fast-model LLM rerank pass on top of RRF fusion, applied only to broad/family queries when enabled |
| `MAX_PAGE_CONTEXT_CHARS` | 2,000 | Per-page character cap in answer context |
| `MAX_TOKENS_ANSWER` | 4,096 | Answer generation token budget (narrow/single-document) |
| `MAX_TOKENS_ANSWER_BROAD` | 8,192 | Answer generation token budget for comparison/risk/obligation intents spanning many sources |
| `MAX_TOKENS_COMPACTION` | 4,096 | Page compaction re-synthesis token budget |
| `MAX_TOKENS_GROUNDING_CHECK` | 900 | Token budget for the independent grounding-check LLM call (escalates on truncation, capped at 4 attempts) |
| `COMPACTION_APPEND_THRESHOLD` | 5 | Compaction trigger: number of appends |
| `COMPACTION_CHAR_THRESHOLD` | 8,000 | Compaction trigger: char count (requires append_count ≥ 2) |
| `WIKI_MAX_WORKERS` | 3 | Parallel ingest worker threads |
| `ENABLE_CLARIFICATION` | true | Enable/disable follow-up clarification questions |
| `ENABLE_INTENT_CLASSIFIER` | true | Enable LLM fallback for intent classification (regex fast-path always on) |
| `ENABLE_ANSWER_VALIDATION` | true | Enable the independent grounding-check LLM audit in the `validate` node |
| `MAX_TOKENS_DISAMBIGUATION` | 200 | Token budget for document disambiguation check |
| `MAX_TOKENS_AMBIGUITY_CHECK` | 300 | Token budget for query clarification check |
| `MAX_TOKENS_INTENT_CLASSIFY` | 150 | Token budget for lawyer-intent classification |

---

## Phases completed

| Phase | What | Status |
|---|---|---|
| 1 | Dual-model routing, merged confidence eval, explicit token budgets | ✅ |
| 2 | PostgreSQL + pgvector storage, persistent progress tracking | ✅ |
| 3 | Hybrid retrieval: pgvector cosine top-K + BM25 supplement | ✅ |
| 4 | FTS cross-reference (GIN), page compaction (S3), NER contradiction pre-filter (C3), metadata cache (C7) | ✅ |
| 4.5 | Answer quality hardening: page title disambiguation, source attribution in merges, 6 new answer-prompt precision rules | ✅ |
| 4.6 | Conversational UX: chat interface, document disambiguation, clarification questions, assessment prompt mode, citation improvements, auto-prefix page titles | ✅ |
| 4.7 | Intent classifier agent (LangGraph): 5 lawyer intents, intent-specific prompts, SSE streaming progress stages, intent tag on answers, entity-aware retrieval, reasoning-block fix | ✅ |
| 4.8 | NVIDIA NIM provider, Azure GPT-5.x reasoning-model support, RRF hybrid retrieval fusion, `resolve_scope` node (7-node graph), deterministic citation verification + corrective retry, independent grounding check, party-name document resolution, disambiguation/scoping regex fixes | ✅ |
| 5 | Celery job queue for production-scale ingest | ⬜ planned |
| 6 | Multi-tenancy, PgBouncer, read replicas, audit logging | ⬜ planned |

See `PLAN.md` for full technical specification of each phase.
See `ARCHITECTURE.md` for component diagram and `FLOWCHART.md` for pipeline flowcharts.
