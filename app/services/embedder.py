"""
Embedding service — uses Ollama's local embeddings.

Provides both single-text and batch embedding to minimize HTTP overhead
when processing many chunks during multi-document ingestion.
"""

import ollama
import config


def embed(text: str, is_query: bool = True) -> list[float]:
    """Generate an embedding vector for a single text."""
    prefix = "search_query: " if is_query else "search_document: "
    prefixed_text = text if text.startswith(prefix) else f"{prefix}{text}"
    r = ollama.embeddings(model=config.EMBED_MODEL, prompt=prefixed_text)
    return r["embedding"]


def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in batches.

    Processes in batches of 100 to prevent crashing the embedding model
    with excessively large payloads on huge documents.
    """
    if not texts:
        return []
    prefix = "search_query: " if is_query else "search_document: "
    prefixed_texts = [t if t.startswith(prefix) else f"{prefix}{t}" for t in texts]
    
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[i:i + batch_size]
        try:
            r = ollama.embed(model=config.EMBED_MODEL, input=batch)
            all_embeddings.extend(r["embeddings"])
        except (AttributeError, TypeError, KeyError):
            # Fallback: older ollama client without batch support
            all_embeddings.extend([embed(t, is_query) for t in batch])
            
    return all_embeddings
