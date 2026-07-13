import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Database — set DATABASE_URL to enable PostgreSQL storage (Phase 2+).
# When unset, the system falls back to file-based index.json storage.
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_DATABASE = bool(DATABASE_URL)

# Global Providers (azure / openrouter / nvidia)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "azure")

# Azure Config
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# OpenRouter embedding models may produce different vector dimensions from Azure.
# nvidia/llama-nemotron-embed-vl-1b-v2 outputs 2048 dims; override via env if needed.
OPENROUTER_EMBEDDING_DIMENSIONS = int(os.getenv("OPENROUTER_EMBEDDING_DIMENSIONS", "2048"))


def get_embedding_dimensions() -> int:
    """Return the vector dimension for the currently active embedding provider."""
    if EMBEDDING_PROVIDER == "openrouter":
        return OPENROUTER_EMBEDDING_DIMENSIONS
    if EMBEDDING_PROVIDER == "nvidia":
        return NVIDIA_EMBEDDING_DIMENSIONS
    return EMBEDDING_DIMENSIONS

# OpenRouter Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY_WIKI", ""))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")

# Fast/cheap model for non-synthesis tasks:
#   contradiction pre-flight, page selection, JSON repair, cell extraction.
#   Set these to a smaller/cheaper deployment — full synthesis calls ignore these.
AZURE_FAST_DEPLOYMENT = os.getenv("AZURE_FAST_DEPLOYMENT", "gpt-5.4-mini")
OPENROUTER_FAST_MODEL = os.getenv("OPENROUTER_FAST_MODEL", "google/gemma-4-27b-it")

# NVIDIA NIM Config
NVIDIA_API_KEY              = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_ENDPOINT             = os.getenv("NVIDIA_ENDPOINT", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL                = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-120b")
NVIDIA_FAST_MODEL           = os.getenv("NVIDIA_FAST_MODEL", "openai/gpt-oss-20b")
NVIDIA_EMBEDDING_MODEL      = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embed-v1")
NVIDIA_EMBEDDING_DIMENSIONS = int(os.getenv("NVIDIA_EMBEDDING_DIMENSIONS", "4096"))

# ---------------------------------------------------------------------------
# Token budget constants — one source of truth for every llm.ask() call.
# Keeping these explicit prevents silently burning 4096-token defaults on
# calls whose outputs are small JSON objects.
# ---------------------------------------------------------------------------

# Ingest pipeline
MAX_TOKENS_INGEST_SINGLE   = 16000  # Single-call short-doc synthesis (10-30 pages)
MAX_TOKENS_INGEST_OVERVIEW = 2000   # Phase-1 overview + topic list
MAX_TOKENS_INGEST_DETAIL   = 8000   # Phase-2 per-segment detail extraction

# Merge / maintenance  (cheap model)
MAX_TOKENS_CONTRADICTION   = 300    # Pairwise contradiction pre-flight
MAX_TOKENS_JSON_REPAIR     = 2048   # LLM JSON repair (bounded by input)

# Query pipeline
MAX_TOKENS_PAGE_SELECTION    = 1000  # JSON list of up to 25 page titles (full model)
MAX_TOKENS_ANSWER            = 4096  # Full legal synthesis with CoT + citations (single-document / narrow factual)
MAX_TOKENS_ANSWER_BROAD      = 8192  # Comparison/risk/obligation intents span many sources — reasoning + table + refs need more room
MAX_TOKENS_DISAMBIGUATION    = 200   # Classify if query targets an unspecified document
MAX_TOKENS_AMBIGUITY_CHECK   = 300   # Determine if query needs clarification
MAX_TOKENS_INTENT_CLASSIFY   = 150   # Classify lawyer intent (factual/risk/comparison/obligation/drafting)
MAX_TOKENS_MATTER_REFERENCE  = 60    # One-off backfill: extract matter/case/docket reference string, or "null"
MAX_TOKENS_COMPACTION        = 4096  # Re-synthesis of bloated pages (S3, Phase 4)
MAX_QPAGE_CONTEXT_CHARS      = 3_000 # Cap on cached-answer (Q:) pages in context
MAX_PAGE_CONTEXT_CHARS       = 2_000 # Cap on any single wiki page in context (prevents merged pages dominating)
MAX_TOTAL_CONTEXT_CHARS      = 60_000 # Cap on the combined wiki_content sent per LLM call (prevents context-window overflow on broad queries that match many similar documents)
ENTITY_MATCH_MAX_PAGES       = 50    # Above this, an "entity" match is too common (reused across many docs) to force-scope the query to it
AMBIGUITY_DOC_SAMPLE_CAP     = 40    # Doc sample size shown to the check_ambiguity() LLM prompt — type-diverse, not a raw head-slice
PAGE_SELECTION_PREFILTER_N   = 150   # BM25 candidates sent to LLM for final selection (from potentially 1000s of pages)
VECTOR_SEARCH_TOP_K          = 15    # Nearest-neighbour results from pgvector (Phase 3)
HYBRID_BM25_SUPPLEMENT_N     = 8     # BM25 keyword pages added on top of vector results (hybrid retrieval)
BROAD_QUESTION_VECTOR_TOP_K  = 80    # Wider candidate pool for "across all X" questions, before per-document diversification
BROAD_QUESTION_PER_DOC_CAP   = 4     # Max pages any single document can contribute to a broad-question candidate list
BROAD_QUESTION_TOTAL_CAP     = 60    # Final page budget for a broad question after diversification (vs. 15 for a normal question) — raised to fit a Parties page + clause page per document without starving document breadth

# Reranking (Phase 3)
RRF_K                        = 60    # Reciprocal Rank Fusion constant — standard default; larger = flatter weighting of rank position
HYBRID_FUSION_TOP_K          = 23    # Final page budget after RRF fusion for a NON-broad hybrid query (≈ old VECTOR_SEARCH_TOP_K + HYBRID_BM25_SUPPLEMENT_N, preserves prior context size)
RERANK_CANDIDATE_N           = 25    # Candidates sent to the optional LLM reranker (titles+summaries only)
MAX_TOKENS_RERANK            = 2048  # Fast-model rerank: the gpt-oss reasoning model spends most of the budget on hidden reasoning, so a small cap (e.g. 400) returns EMPTY output — 2048 is the smallest that reliably emits the JSON ranking for ~25 candidates

# Compaction thresholds (S3, Phase 4)
COMPACTION_APPEND_THRESHOLD  = int(os.getenv("COMPACTION_APPEND_THRESHOLD", "5"))
COMPACTION_CHAR_THRESHOLD    = int(os.getenv("COMPACTION_CHAR_THRESHOLD", "8000"))

# Concurrency settings for wiki pipeline
WIKI_MAX_WORKERS = int(os.getenv("WIKI_MAX_WORKERS", "3"))

# Resolve all data paths to absolute so they stay stable regardless of CWD
_APP_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_PATH = os.path.join(_APP_DIR, "data", "chroma")
WIKI_PATH = os.path.join(_APP_DIR, "data", "wiki")
UPLOAD_PATH = os.path.join(_APP_DIR, "data", "uploads")
SESSIONS_PATH = os.path.join(_APP_DIR, "data", "sessions.json")
LOGS_PATH = os.path.join(_APP_DIR, "data", "logs")

# Pre-create data directories at import time
for _d in [CHROMA_PATH, WIKI_PATH, UPLOAD_PATH, LOGS_PATH]:
    os.makedirs(_d, exist_ok=True)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = int(os.getenv("TOP_K", "40"))

# Global progress store for UI feedback
PROGRESS_STORE = {}

# Conversational UX
ENABLE_CLARIFICATION = os.getenv("ENABLE_CLARIFICATION", "true").lower() == "true"
ENABLE_INTENT_CLASSIFIER = os.getenv("ENABLE_INTENT_CLASSIFIER", "true").lower() == "true"
ENABLE_ANSWER_VALIDATION = os.getenv("ENABLE_ANSWER_VALIDATION", "true").lower() == "true"
MAX_TOKENS_GROUNDING_CHECK = 900  # bumped from 500: full (untruncated) answers can surface more ungrounded_claims entries

# Optional LLM reranking pass (Phase 3). RRF fusion is always on (free); this adds
# a fast-model relevance rerank ON TOP, applied only to broad/family queries where
# precision matters most. Off by default — enable to A/B its quality/latency cost.
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"

# OCR — path to Tesseract executable (set in .env if not on PATH)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
