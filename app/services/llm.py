"""
LLM abstraction — supports Ollama (local) and OpenRouter (cloud).
Entry point: ask(prompt, pipeline) -> str

The `pipeline` parameter ("rag" or "wiki") selects which OpenRouter API key
to use, allowing each pipeline to have its own key/quota.
"""

import requests
import logging
import time
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
        # Increased timeout to 600s. Since wiki ingestion uses max_workers=5, 
        # multiple requests might queue up in Ollama and exceed the standard 120s timeout.
        resp = requests.post(url, json=payload, timeout=600)
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
    """Call OpenRouter chat completions with retry-with-backoff and model fallback.

    Strategy:
      1. Try the primary model up to MAX_RETRIES times, respecting Retry-After.
      2. If still rate-limited, try each fallback model in order.
      3. Only raise RuntimeError if every model is exhausted.
    """
    api_key = (
        config.OPENROUTER_API_KEY_RAG if pipeline == "rag"
        else config.OPENROUTER_API_KEY_WIKI
    )

    models_to_try = [config.OPENROUTER_MODEL] + config.OPENROUTER_FALLBACK_MODELS
    last_error = None

    for model in models_to_try:
        result = _try_model(model, prompt, api_key)
        if result is not None:
            return result
        logger.warning("Model %s exhausted retries, trying next fallback...", model)

    raise RuntimeError(
        f"All OpenRouter models rate-limited or unavailable ({pipeline}). "
        f"Tried: {', '.join(models_to_try)}"
    )


_MAX_RETRIES = 4
_BASE_BACKOFF = 2  # seconds


def _try_model(model: str, prompt: str, api_key: str) -> tuple[str, dict] | None:
    """Attempt a single model with retries. Returns (text, usage) or None if exhausted."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage") or {}
                return data["choices"][0]["message"]["content"], usage

            if resp.status_code == 429:
                wait = _parse_retry_after(resp)
                logger.info(
                    "Rate-limited on %s (attempt %d/%d), waiting %.1fs...",
                    model, attempt + 1, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            # Non-retryable HTTP error — skip this model
            resp.raise_for_status()

        except requests.Timeout:
            logger.warning("Timeout on %s (attempt %d/%d)", model, attempt + 1, _MAX_RETRIES)
            time.sleep(_BASE_BACKOFF)
            continue
        except (requests.RequestException, KeyError, IndexError) as e:
            logger.warning("Error on %s (attempt %d/%d): %s", model, attempt + 1, _MAX_RETRIES, e)
            return None  # non-retryable error, skip model

    return None  # all retries exhausted for this model


def _parse_retry_after(resp: requests.Response) -> float:
    """Extract wait time from a 429 response. Checks Retry-After header and JSON body."""
    # Check standard header first
    retry_header = resp.headers.get("Retry-After")
    if retry_header:
        try:
            return max(float(retry_header), 1.0)
        except ValueError:
            pass

    # Check OpenRouter's JSON error body
    try:
        body = resp.json()
        metadata = body.get("error", {}).get("metadata", {})
        seconds = metadata.get("retry_after_seconds")
        if seconds is not None:
            return max(float(seconds), 1.0)
    except Exception:
        pass

    # Default exponential backoff
    return _BASE_BACKOFF
