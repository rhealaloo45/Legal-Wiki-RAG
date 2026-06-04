"""
Embedding service — supports Azure OpenAI and OpenRouter embeddings.

Provides both single-text and batch embedding.
"""

from openai import AzureOpenAI, OpenAI
import config

# Lazy load the clients
_client = None
_or_client = None

def get_openrouter_client() -> OpenAI:
    global _or_client
    if _or_client is None:
        _or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY
        )
    return _or_client

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
    prefix = "search_query: " if is_query else "search_document: "
    prefixed_text = text if text.startswith(prefix) else f"{prefix}{text}"
    
    if config.EMBEDDING_PROVIDER == "openrouter":
        client = get_openrouter_client()
        model_name = config.OPENROUTER_EMBEDDING_MODEL
    else:
        client = get_client()
        model_name = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

    response = client.embeddings.create(
        input=[prefixed_text],
        model=model_name
    )
    return response.data[0].embedding


def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in batches."""
    if not texts:
        return []
        
    prefix = "search_query: " if is_query else "search_document: "
    prefixed_texts = [t if t.startswith(prefix) else f"{prefix}{t}" for t in texts]
    
    if config.EMBEDDING_PROVIDER == "openrouter":
        client = get_openrouter_client()
        model_name = config.OPENROUTER_EMBEDDING_MODEL
    else:
        client = get_client()
        model_name = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        
    all_embeddings = []
    # Azure OpenAI embeddings limits depend on the tier, but 16/100 is usually safe
    batch_size = 16 
    for i in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model_name
        )
        all_embeddings.extend([data.embedding for data in response.data])
            
    return all_embeddings
