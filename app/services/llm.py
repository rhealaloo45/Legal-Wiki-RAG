"""
LLM abstraction — supports Azure OpenAI and OpenRouter.
Entry point: ask(prompt, pipeline) -> tuple[str, dict]
"""

import logging
from openai import AzureOpenAI, OpenAI
import config

logger = logging.getLogger(__name__)

# Lazy load clients — default and fast (shorter timeout for bulk extraction)
_client = None
_fast_client = None
_or_client = None
_or_fast_client = None

def get_openrouter_client() -> OpenAI:
    global _or_client
    if _or_client is None:
        _or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
            timeout=120.0,
            max_retries=2
        )
    return _or_client

def _get_fast_openrouter_client() -> OpenAI:
    global _or_fast_client
    if _or_fast_client is None:
        _or_fast_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
            timeout=45.0,
            max_retries=1
        )
    return _or_fast_client

def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            timeout=120.0,       # 2-minute timeout for normal calls
            max_retries=2
        )
    return _client

def _get_fast_client() -> AzureOpenAI:
    """Client with aggressive timeout for bulk cell extraction tasks."""
    global _fast_client
    if _fast_client is None:
        _fast_client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            timeout=45.0,        # 45-second timeout for fast extraction
            max_retries=1
        )
    return _fast_client

def ask(prompt: str, pipeline: str = "wiki", max_tokens: int = None) -> tuple[str, dict]:
    """Send a prompt to the selected LLM and return the response text and usage dict.

    Args:
        prompt:   The full prompt string.
        pipeline: ignored in this version, but kept for compatibility.
        max_tokens: Limit the maximum tokens in the completion response.
    """
    if config.LLM_PROVIDER == "openrouter":
        client = get_openrouter_client()
        model_name = config.OPENROUTER_MODEL
    else:
        client = get_client()
        model_name = config.AZURE_OPENAI_DEPLOYMENT

    try:
        kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
            
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0,
        }
        return content, usage
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise RuntimeError(f"LLM unavailable: {e}") from e

def fast_ask(prompt: str, max_tokens: int = 150) -> tuple[str, dict]:
    """Lightweight LLM call optimised for bulk cell extraction.
    
    Uses a shorter timeout and single retry to avoid blocking the 
    ThreadPoolExecutor when the API is slow or rate-limited.
    """
    if config.LLM_PROVIDER == "openrouter":
        client = _get_fast_openrouter_client()
        model_name = config.OPENROUTER_MODEL
    else:
        client = _get_fast_client()
        model_name = config.AZURE_OPENAI_DEPLOYMENT

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=max_tokens
        )
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0,
        }
        return content, usage
    except Exception as e:
        logger.error(f"Fast LLM call failed: {e}")
        raise RuntimeError(f"LLM unavailable (fast): {e}") from e
