"""
LLM abstraction — supports Ollama (local) and OpenRouter (cloud).
Entry point: ask(prompt, pipeline) -> str

The `pipeline` parameter ("rag" or "wiki") selects which OpenRouter API key
to use, allowing each pipeline to have its own key/quota.
"""

import requests
import logging
import config

logger = logging.getLogger(__name__)

def ask(prompt: str, pipeline: str = "wiki") -> tuple[str, dict]:
    """Send a prompt to the configured LLM provider and return the response text and usage dict.

    Args:
        prompt:   The full prompt string.
        pipeline: "rag" or "wiki" — determines which OpenRouter API key is used.
    """
    if config.LLM_PROVIDER == "openrouter":
        try:
            return _ask_openrouter(prompt, pipeline)
        except RuntimeError as e:
            logger.warning(f"OpenRouter failed: {e}. Falling back to Ollama.")
            return _ask_ollama(prompt)
    return _ask_ollama(prompt)


def _ask_ollama(prompt: str) -> tuple[str, dict]:
    """Call Ollama's local generate endpoint."""
    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0)
        }
        return data.get("response", ""), usage
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama unavailable: {e}") from e


def _ask_openrouter(prompt: str, pipeline: str) -> tuple[str, dict]:
    """Call OpenRouter chat completions endpoint with the pipeline-specific key."""
    api_key = (
        config.OPENROUTER_API_KEY_RAG if pipeline == "rag"
        else config.OPENROUTER_API_KEY_WIKI
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return data["choices"][0]["message"]["content"], usage
    except (requests.RequestException, KeyError, IndexError) as e:
        raise RuntimeError(f"OpenRouter unavailable ({pipeline}): {e}") from e
