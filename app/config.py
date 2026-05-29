import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Concurrency settings for wiki pipeline
WIKI_MAX_WORKERS = int(os.getenv("WIKI_MAX_WORKERS", "3"))

# Resolve all data paths to absolute so they stay stable regardless of CWD
_APP_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_PATH = os.path.join(_APP_DIR, "data", "chroma")
WIKI_PATH = os.path.join(_APP_DIR, "data", "wiki")
UPLOAD_PATH = os.path.join(_APP_DIR, "data", "uploads")
SESSIONS_PATH = os.path.join(_APP_DIR, "data", "sessions.json")

# Pre-create data directories at import time
for _d in [CHROMA_PATH, WIKI_PATH, UPLOAD_PATH]:
    os.makedirs(_d, exist_ok=True)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = int(os.getenv("TOP_K", "40"))

# Global progress store for UI feedback
PROGRESS_STORE = {}

# OCR — path to Tesseract executable (set in .env if not on PATH)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
