"""
Embedding service — uses Azure OpenAI embeddings.

Provides both single-text and batch embedding.
"""

from openai import AzureOpenAI
import config

# Lazy load the client
_client = None

def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )
    return _client

def embed(text: str, is_query: bool = True) -> list[float]:
    """Generate an embedding vector for a single text."""
    client = get_client()
    prefix = "search_query: " if is_query else "search_document: "
    prefixed_text = text if text.startswith(prefix) else f"{prefix}{text}"
    
    response = client.embeddings.create(
        input=[prefixed_text],
        model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding


def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in batches."""
    if not texts:
        return []
        
    client = get_client()
    prefix = "search_query: " if is_query else "search_document: "
    prefixed_texts = [t if t.startswith(prefix) else f"{prefix}{t}" for t in texts]
    
    all_embeddings = []
    # Azure OpenAI embeddings limits depend on the tier, but 16/100 is usually safe
    batch_size = 16 
    for i in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        all_embeddings.extend([data.embedding for data in response.data])
            
    return all_embeddings
