import os
from dotenv import load_dotenv

load_dotenv()

# Global Providers (azure or openrouter)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "azure")

# Azure Config
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# OpenRouter Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY_WIKI", ""))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")

# Fast/cheap model for non-synthesis tasks:
#   contradiction pre-flight, page selection, JSON repair, cell extraction.
#   Set these to a smaller/cheaper deployment — full synthesis calls ignore these.
AZURE_FAST_DEPLOYMENT = os.getenv("AZURE_FAST_DEPLOYMENT", "gpt-5.4-mini")
OPENROUTER_FAST_MODEL = os.getenv("OPENROUTER_FAST_MODEL", "google/gemma-4-27b-it")

# ---------------------------------------------------------------------------
# Token budget constants — one source of truth for every llm.ask() call.
# Keeping these explicit prevents silently burning 4096-token defaults on
# calls whose outputs are small JSON objects.
# ---------------------------------------------------------------------------

# Ingest pipeline
MAX_TOKENS_INGEST_SINGLE   = 4096   # Single-call short-doc synthesis (10-30 pages)
MAX_TOKENS_INGEST_OVERVIEW = 1500   # Phase-1 overview + topic list
MAX_TOKENS_INGEST_DETAIL   = 3500   # Phase-2 per-segment detail extraction

# Merge / maintenance  (cheap model)
MAX_TOKENS_CONTRADICTION   = 300    # Pairwise contradiction pre-flight
MAX_TOKENS_JSON_REPAIR     = 2048   # LLM JSON repair (bounded by input)

# Query pipeline
MAX_TOKENS_PAGE_SELECTION  = 1000   # JSON list of up to 25 page titles (full model)
MAX_TOKENS_ANSWER          = 4096   # Full legal synthesis with CoT + citations

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

# OCR — path to Tesseract executable (set in .env if not on PATH)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
