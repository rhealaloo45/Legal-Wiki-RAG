"""
Embedding service — uses Ollama's local embeddings.

Provides both single-text and batch embedding to minimize HTTP overhead
when processing many chunks during multi-document ingestion.
"""

import ollama
import config


def embed(text: str) -> list[float]:
    """Generate an embedding vector for a single text."""
    r = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
    return r["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in one call.

    Falls back to sequential embedding if the batch API is unavailable
    (older Ollama versions).
    """
    if not texts:
        return []
    try:
        r = ollama.embed(model=config.EMBED_MODEL, input=texts)
        return r["embeddings"]
    except (AttributeError, TypeError, KeyError):
        # Fallback: older ollama client without batch support
        return [embed(t) for t in texts]
