"""
LLM abstraction — supports Azure OpenAI.
Entry point: ask(prompt, pipeline) -> tuple[str, dict]
"""

import logging
from openai import AzureOpenAI
import config

logger = logging.getLogger(__name__)

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
            kwargs["max_tokens"] = max_tokens
            
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
