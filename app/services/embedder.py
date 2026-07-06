"""
Embedding service — supports Azure OpenAI and OpenRouter embeddings.

Provides both single-text and batch embedding.

OpenRouter note: the OpenAI SDK mis-parses some OpenRouter embedding responses,
so OpenRouter calls go through a direct requests.post() call instead.
"""

from __future__ import annotations

import logging
import time
import requests as _requests

import config

logger = logging.getLogger(__name__)

# Transient HTTP status codes worth retrying (gateway errors, rate limits, etc.)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds — doubles each attempt: 2s, 4s, 8s

# Lazy-loaded clients
_azure_client = None
_nvidia_client = None


def _get_azure_client():
    global _azure_client
    if _azure_client is None:
        from openai import OpenAI
        _azure_client = OpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            base_url=f"{config.AZURE_OPENAI_ENDPOINT}/openai/v1",
        )
    return _azure_client


def _get_nvidia_client():
    global _nvidia_client
    if _nvidia_client is None:
        from openai import OpenAI
        _nvidia_client = OpenAI(
            api_key=config.NVIDIA_API_KEY,
            base_url=config.NVIDIA_ENDPOINT,
        )
    return _nvidia_client


# ---------------------------------------------------------------------------
# OpenRouter — direct HTTP (bypasses OpenAI SDK response-parsing issues)
# ---------------------------------------------------------------------------

def _embed_openrouter(texts: list[str]) -> list[list[float]]:
    """Call the OpenRouter embeddings endpoint directly via requests.

    The OpenAI SDK raises 'No embedding data received' for some OpenRouter
    models (e.g. nvidia/llama-nemotron-embed-vl-1b-v2) because it expects a
    slightly different response envelope.  A plain HTTP call avoids this.

    Retries up to _MAX_RETRIES times with exponential backoff on transient
    5xx / 429 errors (e.g. 502 Bad Gateway from OpenRouter's upstream).
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = _requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": config.OPENROUTER_EMBEDDING_MODEL, "input": texts},
                timeout=60,
            )

            # Retry on transient server-side errors without raising immediately
            if resp.status_code in _RETRYABLE_STATUSES:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "OpenRouter embeddings returned HTTP %d (attempt %d/%d) — "
                    "retrying in %.0fs. Body: %s",
                    resp.status_code, attempt + 1, _MAX_RETRIES,
                    wait, resp.text[:200],
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            body = resp.json()

            # OpenRouter sometimes returns 200 with an error payload instead of
            # a proper 5xx — treat it as retryable too.
            if "error" in body and "data" not in body:
                err_code = body["error"].get("code", 0)
                if err_code in _RETRYABLE_STATUSES or err_code == 0:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "OpenRouter embeddings returned error payload (attempt %d/%d) — "
                        "retrying in %.0fs. Body: %s",
                        attempt + 1, _MAX_RETRIES, wait, str(body)[:200],
                    )
                    time.sleep(wait)
                    continue
                raise ValueError(
                    f"OpenRouter embeddings API error: {body['error']}"
                )

            data = body.get("data", [])
            if not data:
                raise ValueError(
                    f"OpenRouter embeddings API returned no data. "
                    f"Status {resp.status_code}. Body: {str(body)[:300]}"
                )

            # Sort by index so output order matches input order
            data.sort(key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in data]

        except (_requests.exceptions.ConnectionError,
                _requests.exceptions.Timeout) as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "OpenRouter embeddings network error (attempt %d/%d) — "
                "retrying in %.0fs: %s",
                attempt + 1, _MAX_RETRIES, wait, exc,
            )
            last_exc = exc
            time.sleep(wait)

    raise RuntimeError(
        f"OpenRouter embeddings failed after {_MAX_RETRIES} attempts"
        + (f": {last_exc}" if last_exc else "")
    )


# ---------------------------------------------------------------------------
# NVIDIA NIM
# ---------------------------------------------------------------------------

def _embed_nvidia(texts: list[str], is_query: bool = False) -> list[list[float]]:
    client = _get_nvidia_client()
    input_type = "query" if is_query else "passage"
    all_embeddings: list[list[float]] = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=config.NVIDIA_EMBEDDING_MODEL,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"},
        )
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------

def _embed_azure(texts: list[str]) -> list[list[float]]:
    client = _get_azure_client()
    all_embeddings: list[list[float]] = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(text: str, is_query: bool = True) -> list[float]:
    """Generate an embedding vector for a single text."""
    if config.EMBEDDING_PROVIDER == "nvidia":
        return _embed_nvidia([text], is_query=is_query)[0]

    prefix = "search_query: " if is_query else "search_document: "
    prefixed = text if text.startswith(prefix) else f"{prefix}{text}"

    if config.EMBEDDING_PROVIDER == "openrouter":
        return _embed_openrouter([prefixed])[0]
    return _embed_azure([prefixed])[0]


def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Generate embedding vectors for multiple texts."""
    if not texts:
        return []

    if config.EMBEDDING_PROVIDER == "nvidia":
        return _embed_nvidia(texts, is_query=is_query)

    prefix = "search_query: " if is_query else "search_document: "
    prefixed = [t if t.startswith(prefix) else f"{prefix}{t}" for t in texts]

    if config.EMBEDDING_PROVIDER == "openrouter":
        all_embeddings: list[list[float]] = []
        batch_size = 16
        for i in range(0, len(prefixed), batch_size):
            all_embeddings.extend(_embed_openrouter(prefixed[i : i + batch_size]))
        return all_embeddings
    return _embed_azure(prefixed)
