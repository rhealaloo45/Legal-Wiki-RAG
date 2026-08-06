import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Database — set DATABASE_URL to enable PostgreSQL storage (Phase 2+).
# When unset, the system falls back to file-based index.json storage.
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_DATABASE = bool(DATABASE_URL)

# ---------------------------------------------------------------------------
# Production wiki mode — every wiki-scoped call (retrieval, files, graph) is
# redirected to one fixed "main" session regardless of the caller's own
# session_id, so every chat thread queries the same wiki. PRODUCTION_WIKI_SESSION_ID
# below is only the STARTING value for that pointer; app.py tracks the live
# value in data/main_session.json, which a completed local ingest updates
# automatically (see app.py:_set_main_session_id). Two independent concerns:
#   - PRODUCTION_WIKI_SESSION_ID: which session is "main" right now (mutable
#     locally, effectively fixed on Azure since ingest never runs there).
#   - DISABLE_INGEST: whether ingest-capable / session-destructive routes are
#     locked at all. True only on the deployed Azure app — ingestion happens
#     locally and the finished wiki is shipped to Azure Postgres out of band,
#     never through the deployed app itself. False locally, where ingesting
#     new content and having it become the new main session is the point.
# ---------------------------------------------------------------------------
PRODUCTION_WIKI_SESSION_ID = os.getenv("PRODUCTION_WIKI_SESSION_ID", "")
DISABLE_INGEST = os.getenv("DISABLE_INGEST", "false").lower() == "true"

# Global Providers (azure / openrouter / nvidia)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "azure")

# Azure Config — chat/completions resource
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
# GPT-5.x/o-series burn an uncapped share of max_completion_tokens on hidden
# reasoning before writing any visible content — confirmed live: a 37-page
# single-doc summary spent its entire 8192-token retry budget on reasoning,
# leaving <20 visible chars (default effort is "medium"). Capping effort to
# "low" leaves far more of the budget for the actual answer.
AZURE_REASONING_EFFORT = os.getenv("AZURE_REASONING_EFFORT", "low")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Azure embeddings can live on a separate AI Foundry resource from chat —
# falls back to the chat resource's key/endpoint if not set separately, so a
# single-resource setup still works with zero extra config.
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY", AZURE_OPENAI_API_KEY)
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT", AZURE_OPENAI_ENDPOINT)

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
MAX_TOKENS_META_CLASSIFY     = 150   # Meta-question LLM fallback — single-word answer, but Azure
                                      # reasoning models can spend the whole budget on hidden
                                      # reasoning before emitting it (same pitfall as intent-classify
                                      # above); matched to that budget rather than a tighter one.
MAX_TOKENS_GENERAL_KNOWLEDGE = 2048  # General legal-knowledge answers (no retrieval). The prompt caps the
                                      # answer at 3-6 short paragraphs, so this is sized for that plus the
                                      # hidden-reasoning overhead the Azure nano models spend before any
                                      # visible text — deliberately well under MAX_TOKENS_ANSWER, since a
                                      # long answer on this path is a symptom, not a feature.
MAX_TOKENS_MATTER_REFERENCE  = 60    # One-off backfill: extract matter/case/docket reference string, or "null"
MAX_TOKENS_COMPACTION        = 4096  # Re-synthesis of bloated pages (S3, Phase 4)
MAX_QPAGE_CONTEXT_CHARS      = 3_000 # Cap on cached-answer (Q:) pages in context
MAX_PAGE_CONTEXT_CHARS       = 2_000 # Cap on any single wiki page in context (prevents merged pages dominating)
MAX_TOTAL_CONTEXT_CHARS      = 60_000 # Cap on the combined wiki_content sent per LLM call (prevents context-window overflow on broad queries that match many similar documents)
ENTITY_MATCH_MAX_PAGES       = 50    # Above this, an "entity" match is too common (reused across many docs) to force-scope the query to it

# Draft mode — budgeted so a full generate-or-refine call stays under 12k
# tokens total (classify + retrieval + this call's input + output combined),
# not just the completion cap in isolation. Raised from an initial 10k
# target: measured full_document refine (the most expensive case — it both
# sends and regenerates a whole document) at 7.4k-8.6k tokens across most
# runs, one outlier at 10.36k on reasoning-token variance alone. 12k gives
# that case real headroom instead of sitting right on the edge.
MAX_DRAFT_WIKI_CONTEXT_CHARS = 4_000  # ~1000 tokens of grounding context — was 8000, halved to leave room for output
# On the active reasoning model (gpt-5-nano), hidden reasoning tokens count
# against max_completion_tokens before any visible text — a cap set below
# that reasoning floor returns EMPTY output with finish_reason="length",
# not a shorter draft (confirmed live: 1800 and 4500, the original values
# here, both went blank on real prompts; measured thresholds directly —
# clause-type failed at 2500, succeeded at 3200; full_document failed at
# 4096, succeeded at 5500). Same failure mode already documented elsewhere
# in this file for MAX_TOKENS_ANSWER / MAX_TOKENS_RERANK. These floors carry
# real margin above the measured minimums, not the minimums themselves,
# since reasoning-token spend isn't deterministic run to run.
MAX_TOKENS_DRAFT_SHORT       = 4_096  # clause / communication / letter — matches MAX_TOKENS_ANSWER's proven-safe floor for this model
MAX_TOKENS_DRAFT_LONG        = 7_000  # full_document / tracker — measured safe at 5500, this adds ~1500 tokens of margin against run-to-run variance
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

# Resolve all data paths to absolute so they stay stable regardless of CWD.
# DATA_DIR overrides the base — on App Service, code under wwwroot gets wiped
# on every redeploy, so set DATA_DIR=/home/data (persisted) in production.
_APP_DIR = os.path.abspath(os.path.dirname(__file__))
_DATA_DIR = os.getenv("DATA_DIR", _APP_DIR)
WIKI_PATH = os.path.join(_DATA_DIR, "data", "wiki")
UPLOAD_PATH = os.path.join(_DATA_DIR, "data", "uploads")
SESSIONS_PATH = os.path.join(_DATA_DIR, "data", "sessions.json")
MAIN_SESSION_PATH = os.path.join(_DATA_DIR, "data", "main_session.json")
LOGS_PATH = os.path.join(_DATA_DIR, "data", "logs")

# Pre-create data directories at import time
for _d in [WIKI_PATH, UPLOAD_PATH, LOGS_PATH]:
    os.makedirs(_d, exist_ok=True)

# Global progress store for UI feedback
PROGRESS_STORE = {}

# Conversational UX
ENABLE_CLARIFICATION = os.getenv("ENABLE_CLARIFICATION", "true").lower() == "true"
ENABLE_INTENT_CLASSIFIER = os.getenv("ENABLE_INTENT_CLASSIFIER", "true").lower() == "true"
ENABLE_ANSWER_VALIDATION = os.getenv("ENABLE_ANSWER_VALIDATION", "true").lower() == "true"
# Deterministic string-count check of the question's legal topics against the
# retrieved pages. Independent of ENABLE_ANSWER_VALIDATION on purpose — the LLM
# grounding check scored a confirmed fabrication at 90% and endorsed the invented
# clause in its summary, so this is the only signal that caught it. Costs no LLM
# call and never rewrites an answer, only appends a warning. Kept as a flag so it
# can be turned off from App Service settings without a redeploy.
ENABLE_TERM_CHECK = os.getenv("ENABLE_TERM_CHECK", "true").lower() == "true"
# Cheap LLM fallback for meta/capability questions ("what should I be asking
# you?") that don't match the regex fast-path's known phrasings. Hard-gated
# in intent_agent._is_meta_query_extended to messages with ZERO legal
# vocabulary anywhere in them, so a real document question can never reach
# this call — see that function's docstring. Flag exists so it can be turned
# off from App Service settings without a redeploy if it ever misfires.
ENABLE_META_LLM_FALLBACK = os.getenv("ENABLE_META_LLM_FALLBACK", "true").lower() == "true"
# General legal-knowledge carve-out: "what is arbitration", "what does force
# majeure mean" — settled concepts the corpus does not define and previously
# got a bare "not covered". Answers from the model's own knowledge on a
# separate path that never touches retrieval, carryover or the grounded
# prompts, behind the deterministic gates in
# intent_agent._general_knowledge_kind. Its answers are the only ones in this
# system with no document to check them against, so it is a flag: turning it
# off restores the previous refusal behaviour with no other effect.
ENABLE_GENERAL_KNOWLEDGE = os.getenv("ENABLE_GENERAL_KNOWLEDGE", "true").lower() == "true"
# Tiebreak LLM call for definitional questions whose subject is a legal term the
# vocabulary regex doesn't list ("what is laches"). Hard-gated the same way as
# the meta fallback — short, definitional phrasing, no document reference, no
# advice framing — so it can only ever ADD a general answer for a question that
# would otherwise have fallen through; it is never consulted for a question
# heading to the document pipeline.
ENABLE_GK_LLM_FALLBACK = os.getenv("ENABLE_GK_LLM_FALLBACK", "true").lower() == "true"
MAX_TOKENS_GROUNDING_CHECK = 1500  # bumped 900→1500: on Azure reasoning models (nano) the 900-token first pass routinely spent the whole budget on hidden reasoning and truncated (finish_reason=length), forcing a wasted retry at 1800 before any visible JSON — starting at 1500 skips that discarded first call on big answers. The escalating-budget ladder in _check_grounding still doubles from here (1500→3000→6000→12000) for the largest contexts.

# Optional LLM reranking pass (Phase 3). RRF fusion is always on (free); this adds
# a fast-model relevance rerank ON TOP, applied only to broad/family queries where
# precision matters most. Off by default — enable to A/B its quality/latency cost.
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"

# OCR — path to Tesseract executable (set in .env if not on PATH)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
