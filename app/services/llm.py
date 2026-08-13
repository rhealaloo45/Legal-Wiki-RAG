"""
LLM abstraction — supports Azure OpenAI, OpenRouter, and NVIDIA NIM.
Entry point: ask(prompt, pipeline) -> tuple[str, dict]
"""

import logging
import re
from openai import OpenAI, RateLimitError
import config

logger = logging.getLogger(__name__)

# Lazy load clients — default and fast (shorter timeout for bulk extraction)
_client = None
_fast_client = None
_or_client = None
_or_fast_client = None
_nv_client = None
_nv_fast_client = None

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

def _get_nvidia_client() -> OpenAI:
    global _nv_client
    if _nv_client is None:
        _nv_client = OpenAI(
            api_key=config.NVIDIA_API_KEY,
            base_url=config.NVIDIA_ENDPOINT,
            timeout=300.0,
            max_retries=1
        )
    return _nv_client

def _get_nvidia_fast_client() -> OpenAI:
    global _nv_fast_client
    if _nv_fast_client is None:
        _nv_fast_client = OpenAI(
            api_key=config.NVIDIA_API_KEY,
            base_url=config.NVIDIA_ENDPOINT,
            timeout=120.0,
            max_retries=1
        )
    return _nv_fast_client

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            base_url=f"{config.AZURE_OPENAI_ENDPOINT}/openai/v1",
            timeout=120.0,
            max_retries=2
        )
    return _client

def _get_fast_client() -> OpenAI:
    """Client with aggressive timeout for bulk cell extraction tasks."""
    global _fast_client
    if _fast_client is None:
        _fast_client = OpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            base_url=f"{config.AZURE_OPENAI_ENDPOINT}/openai/v1",
            timeout=45.0,
            max_retries=1
        )
    return _fast_client

def active_model(fast: bool = False) -> str:
    """Return the model name that will be used for a given fast/full call.
    Useful for labelling token-usage log entries."""
    if config.LLM_PROVIDER == "openrouter":
        return config.OPENROUTER_FAST_MODEL if fast else config.OPENROUTER_MODEL
    if config.LLM_PROVIDER == "nvidia":
        return config.NVIDIA_FAST_MODEL if fast else config.NVIDIA_MODEL
    return config.AZURE_FAST_DEPLOYMENT if fast else config.AZURE_OPENAI_DEPLOYMENT


def _is_azure() -> bool:
    """The active LLM provider is Azure (anything that isn't the two
    OpenAI-compatible community endpoints)."""
    return config.LLM_PROVIDER not in ("openrouter", "nvidia")


def _is_reasoning_model(model_name: str) -> bool:
    """True for GPT-5.x / o-series deployments, which take `reasoning_effort`
    and reject a non-default `temperature`. Classic Azure chat deployments
    (gpt-4o, gpt-4o-mini, gpt-4.1, ...) reject `reasoning_effort` outright —
    keyed off the deployment name since Azure has no model-family field."""
    name = model_name.lower()
    return bool(re.match(r"^(gpt-5|o1|o3|o4)\b", name)) or "gpt-5" in name


def _completion_kwargs(model_name: str, prompt: str, max_tokens: int | None,
                       reasoning_effort: str | None = None) -> dict:
    """Build chat.completions kwargs, adapted to the active provider's API.

    Azure GPT-5.x / o-series reasoning deployments differ from the classic
    Chat Completions contract in two ways that otherwise 400 the request:
      - they reject `max_tokens` and require `max_completion_tokens`;
      - they reject any non-default `temperature` (only the default is allowed),
    so we omit temperature entirely on that path.
    The nvidia/openrouter OpenAI-compatible endpoints keep the classic params
    (deterministic `temperature=0.0` + `max_tokens`).

    reasoning_effort overrides the global config default for this one call —
    used to escalate effort on a retry when a low-effort reasoning-model call
    came back with hidden reasoning but zero visible content.
    """
    kwargs = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
    if _is_azure():
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if _is_reasoning_model(model_name):
            kwargs["reasoning_effort"] = reasoning_effort or config.AZURE_REASONING_EFFORT
        else:
            kwargs["temperature"] = 0.0
    else:
        kwargs["temperature"] = 0.0
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
    return kwargs


def ask(prompt: str, pipeline: str = "wiki", max_tokens: int = None, fast: bool = False,
        reasoning_effort: str = None) -> tuple[str, dict]:
    """Send a prompt to the selected LLM and return the response text and usage dict.

    Args:
        prompt:     The full prompt string.
        pipeline:   Kept for compatibility; not used for routing.
        max_tokens: Cap the completion token count. Always pass an explicit value —
                    relying on the model default wastes budget on small-output calls.
        fast:       Route to the cheap/fast model (AZURE_FAST_DEPLOYMENT /
                    OPENROUTER_FAST_MODEL) instead of the full synthesis model.
                    Use for: contradiction checks, page selection, JSON repair.
                    Do NOT use for: ingest synthesis, answer generation, drafting.
        reasoning_effort: Override the global AZURE_REASONING_EFFORT for this one
                    call (Azure reasoning models only; ignored elsewhere). Used to
                    escalate effort on an empty-answer retry.
    """
    if config.LLM_PROVIDER == "openrouter":
        client = get_openrouter_client()
        model_name = config.OPENROUTER_FAST_MODEL if fast else config.OPENROUTER_MODEL
    elif config.LLM_PROVIDER == "nvidia":
        client = _get_nvidia_client()
        model_name = config.NVIDIA_FAST_MODEL if fast else config.NVIDIA_MODEL
    else:
        client = get_client()
        model_name = config.AZURE_FAST_DEPLOYMENT if fast else config.AZURE_OPENAI_DEPLOYMENT

    try:
        kwargs = _completion_kwargs(model_name, prompt, max_tokens, reasoning_effort)
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0,
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
        }
        return content, usage
    except RateLimitError:
        raise  # bubble up — callers must stop on 429, not silently skip
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise RuntimeError(f"LLM unavailable: {e}") from e

def ask_vision(image_b64: str, prompt: str, max_tokens: int = 4096, fast: bool = True) -> tuple[str, dict]:
    """Send a single page image to the Azure OpenAI deployment for OCR/transcription.

    Used as an alternative to Tesseract when local OCR fails to read a scanned
    page (skew, low resolution, redaction artefacts) — the vision-capable chat
    deployment reads the rendered page image directly. Azure-only: routes
    through the same client/deployment as text calls, just with an image
    content block attached.

    Args:
        image_b64: Base64-encoded PNG bytes of the rendered page.
        prompt:    Instruction describing what to transcribe.
        max_tokens: Completion budget — a dense legal page can run long.
        fast:      Use AZURE_FAST_DEPLOYMENT (cheaper) vs the full deployment.
    """
    if not _is_azure():
        raise RuntimeError("ask_vision requires LLM_PROVIDER=azure (OCR_ENGINE=azure_vision)")

    client = _get_fast_client() if fast else get_client()
    model_name = config.AZURE_FAST_DEPLOYMENT if fast else config.AZURE_OPENAI_DEPLOYMENT

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
    }]
    kwargs = {"model": model_name, "messages": messages}
    if _is_reasoning_model(model_name):
        kwargs["max_completion_tokens"] = max_tokens
        kwargs["reasoning_effort"] = config.AZURE_REASONING_EFFORT
    else:
        kwargs["max_completion_tokens"] = max_tokens
        kwargs["temperature"] = 0.0

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0,
        }
        return content, usage
    except RateLimitError:
        raise
    except Exception as e:
        logger.error(f"Vision LLM call failed: {e}")
        raise RuntimeError(f"LLM unavailable (vision): {e}") from e


def fast_ask(prompt: str, max_tokens: int = 150) -> tuple[str, dict]:
    """Lightweight LLM call for bulk extraction tasks (cell extraction, column inference,
    aspect identification, outlier detection).

    Routes to the cheap model (AZURE_FAST_DEPLOYMENT / OPENROUTER_FAST_MODEL) with a
    short timeout and single retry to avoid blocking the ThreadPoolExecutor.
    Never use for synthesis tasks that require legal reasoning depth.
    """
    if config.LLM_PROVIDER == "openrouter":
        client = _get_fast_openrouter_client()
        model_name = config.OPENROUTER_FAST_MODEL
    elif config.LLM_PROVIDER == "nvidia":
        client = _get_nvidia_fast_client()
        model_name = config.NVIDIA_FAST_MODEL
    else:
        client = _get_fast_client()
        model_name = config.AZURE_FAST_DEPLOYMENT

    try:
        kwargs = _completion_kwargs(model_name, prompt, max_tokens)
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0,
        }
        return content, usage
    except RateLimitError:
        raise
    except Exception as e:
        logger.error(f"Fast LLM call failed: {e}")
        raise RuntimeError(f"LLM unavailable (fast): {e}") from e
