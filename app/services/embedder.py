"""
Embedding service — uses Ollama's local Llama3 embeddings.
"""

import ollama
import config


def embed(text: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    r = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
    return r["embedding"]
