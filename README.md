# Legal Wiki RAG

A production-grade legal intelligence platform that transforms unstructured legal documents into a structured, queryable knowledge base using LLM-driven wiki synthesis — not retrieval-augmented generation.

---

## What makes this different from RAG

Standard RAG retrieves raw document chunks at query time and asks the LLM to reason over them on the fly. Legal Wiki builds a **persistent structured wiki** at ingest time. Each document enriches the same knowledge base: pages are merged, contradictions flagged, cross-references built, and shared legal concepts accumulated across all cases. Queries read from pre-compiled synthesis, not raw text.

| | Standard RAG | Legal Wiki |
|---|---|---|
| Query cost | ~2,000–5,500 tokens | ~5,000–6,000 tokens |
| Cross-doc synthesis quality | Weak | Strong (pre-built) |
| At 20k docs | Recall degrades | Consistent (compaction keeps pages coherent) |
| Ingest cost | Near zero | LLM calls per doc (paid once, queried many times) |
| Hallucination surface | Raw chunks, no pre-filtering | Pre-synthesised + 20 answer-prompt rules |

---

## Features

### Wiki Pipeline
- **Adaptive ingest**: single LLM call for short docs (≤100K chars); two-phase overview + parallel segmentation for long docs
- **Case-aware page titling**: case-specific pages (facts, holding, charges, procedural history) isolated per case; shared legal concepts (statutes, precedents, doctrines) merged and enriched across all cases
- **Source attribution**: merged pages carry `[From: ...]` labels so the answer model attributes correctly without cross-case contamination
- **Hybrid retrieval**: pgvector cosine top-15 + BM25 keyword supplement at query time
- **Page compaction**: pages that grow through repeated merges are re-synthesised by the LLM, keeping the wiki coherent at scale
- **D3.js knowledge graph** visualisation of pages and relations

### Query Mode (Ask)
- Chain-of-thought reasoning with self-assessed confidence score (0–100) extracted from `<reasoning>` block — zero extra LLM calls
- 20 precision rules in the answer prompt: procedural stage precision, legal standard precision, named-document completeness, cross-document source discipline, allegations vs findings, thematic selectivity, arithmetic prohibition, statute interpretation, and more
- Per-page context cap (2,000 chars) prevents any single large merged page from crowding out others

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
- **PostgreSQL + pgvector**: single managed store for pages, embeddings, metadata, progress, contradictions, relations
- **Dual-model routing**: full model for synthesis/answers, fast model for contradiction checks/cell extraction/JSON repair
- **SQLAlchemy connection pool** (pool_size=10, max_overflow=20)
- Per-session threading locks for race-free parallel ingest
- OCR fallback via Tesseract for scanned PDFs

---

## Setup

### Requirements
- Python 3.9+
- PostgreSQL 14+ with `pgvector` extension
- Azure OpenAI **or** OpenRouter API credentials
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
# Provider: "azure" or "openrouter"
LLM_PROVIDER=openrouter
EMBEDDING_PROVIDER=openrouter

# OpenRouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o
OPENROUTER_FAST_MODEL=google/gemma-4-27b-it
OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free

# Azure OpenAI (alternative)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://....openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_FAST_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/legalwiki
```

### Run

```bash
cd app
python app.py
# Open http://localhost:5001
```

---

## Project structure

```
app/
├── app.py                   # Flask routes + ThreadPoolExecutor ingest scheduler
├── config.py                # All constants and env-var loading
├── services/
│   ├── db.py                # PostgreSQL abstraction layer (SQLAlchemy)
│   ├── wiki.py              # Ingest pipeline, query pipeline, compaction
│   ├── llm.py               # Dual-model LLM routing (Azure / OpenRouter)
│   ├── embedder.py          # Embedding service (batch, query/doc mode prefixes)
│   ├── prompts.py           # All prompt templates (ingest, answer, compaction)
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
| `VECTOR_SEARCH_TOP_K` | 15 | pgvector nearest-neighbour results per query |
| `HYBRID_BM25_SUPPLEMENT_N` | 8 | BM25 keyword pages added on top of vector results |
| `MAX_PAGE_CONTEXT_CHARS` | 2,000 | Per-page character cap in answer context |
| `MAX_TOKENS_ANSWER` | 4,096 | Answer generation token budget |
| `MAX_TOKENS_COMPACTION` | 4,096 | Page compaction re-synthesis token budget |
| `COMPACTION_APPEND_THRESHOLD` | 5 | Compaction trigger: number of appends |
| `COMPACTION_CHAR_THRESHOLD` | 8,000 | Compaction trigger: char count (requires append_count ≥ 2) |
| `WIKI_MAX_WORKERS` | 3 | Parallel ingest worker threads |

---

## Phases completed

| Phase | What | Status |
|---|---|---|
| 1 | Dual-model routing, merged confidence eval, explicit token budgets | ✅ |
| 2 | PostgreSQL + pgvector storage, persistent progress tracking | ✅ |
| 3 | Hybrid retrieval: pgvector cosine top-K + BM25 supplement | ✅ |
| 4 | FTS cross-reference (GIN), page compaction (S3), NER contradiction pre-filter (C3), metadata cache (C7) | ✅ |
| 4.5 | Answer quality hardening: page title disambiguation, source attribution in merges, 6 new answer-prompt precision rules | ✅ |
| 5 | Celery job queue for production-scale ingest | ⬜ planned |
| 6 | Multi-tenancy, PgBouncer, read replicas, audit logging | ⬜ planned |

See `PLAN.md` for full technical specification of each phase.
See `ARCHITECTURE.md` for component diagram and `FLOWCHART.md` for pipeline flowcharts.
