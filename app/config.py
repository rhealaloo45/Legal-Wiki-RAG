import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")      # "ollama" | "openrouter"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Separate OpenRouter keys per pipeline
OPENROUTER_API_KEY_WIKI = os.getenv("OPENROUTER_API_KEY_WIKI", "")
OPENROUTER_API_KEY_RAG = os.getenv("OPENROUTER_API_KEY_RAG", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
EMBED_MODEL = "llama3"

# Resolve all data paths to absolute so they stay stable regardless of CWD
_APP_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_PATH = os.path.join(_APP_DIR, "data", "chroma")
WIKI_PATH = os.path.join(_APP_DIR, "data", "wiki")
UPLOAD_PATH = os.path.join(_APP_DIR, "data", "uploads")

# Pre-create data directories at import time
for _d in [CHROMA_PATH, WIKI_PATH, UPLOAD_PATH]:
    os.makedirs(_d, exist_ok=True)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = 3
