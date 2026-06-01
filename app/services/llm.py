"""
LLM abstraction — supports Azure OpenAI.
Entry point: ask(prompt, pipeline) -> tuple[str, dict]
"""

import logging
from openai import AzureOpenAI
import config

logger = logging.getLogger(__name__)

# Lazy load clients — default and fast (shorter timeout for bulk extraction)
_client = None
_fast_client = None

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
    """Send a prompt to Azure OpenAI and return the response text and usage dict.

    Args:
        prompt:   The full prompt string.
        pipeline: ignored in this Azure OpenAI version, but kept for compatibility.
        max_tokens: Limit the maximum tokens in the completion response.
    """
    client = get_client()
    try:
        kwargs = {
            "model": config.AZURE_OPENAI_DEPLOYMENT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
            
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return content, usage
    except Exception as e:
        logger.error(f"Azure OpenAI call failed: {e}")
        raise RuntimeError(f"Azure OpenAI unavailable: {e}") from e

def fast_ask(prompt: str, max_tokens: int = 150) -> tuple[str, dict]:
    """Lightweight LLM call optimised for bulk cell extraction.
    
    Uses a shorter timeout and single retry to avoid blocking the 
    ThreadPoolExecutor when the API is slow or rate-limited.
    """
    client = _get_fast_client()
    try:
        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=max_tokens
        )
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return content, usage
    except Exception as e:
        logger.error(f"Fast LLM call failed: {e}")
        raise RuntimeError(f"Azure OpenAI unavailable (fast): {e}") from e
