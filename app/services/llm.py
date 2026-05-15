"""
LLM abstraction — supports Ollama (local) and OpenRouter (cloud).
Entry point: ask(prompt, pipeline) -> str

The `pipeline` parameter ("rag" or "wiki") selects which OpenRouter API key
to use, allowing each pipeline to have its own key/quota.
"""

import requests
import config


def ask(prompt: str, pipeline: str = "wiki") -> str:
    """Send a prompt to the configured LLM provider and return the response text.

    Args:
        prompt:   The full prompt string.
        pipeline: "rag" or "wiki" — determines which OpenRouter API key is used.
    """
    if config.LLM_PROVIDER == "openrouter":
        return _ask_openrouter(prompt, pipeline)
    return _ask_ollama(prompt)


def _ask_ollama(prompt: str) -> str:
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
        return resp.json().get("response", "")
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama unavailable: {e}") from e


def _ask_openrouter(prompt: str, pipeline: str) -> str:
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
        return data["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError) as e:
        raise RuntimeError(f"OpenRouter unavailable ({pipeline}): {e}") from e
