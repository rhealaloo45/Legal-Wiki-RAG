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
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")

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
