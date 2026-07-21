"""
Opik (Comet ML) tracing and LLM-as-judge evaluation helper.

When OPIK_URL_OVERRIDE is set (pointing to the self-hosted backend), this
module:
  - Provides a conditional `track` decorator that wraps opik.track — it is a
    transparent no-op when Opik is disabled, so all call-sites are safe.
  - run_evals(question, answer, context, intent) — runs Hallucination,
    AnswerRelevance, and ContextPrecision judges and logs the scores back to
    the active Opik span via opik_context.update_current_span().

All errors are swallowed so that eval failures never propagate into the
main application pipeline.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ── Enabled flag ──────────────────────────────────────────────────────────────
# Checked at import time so `track` can be a static decorator.
# OPIK_URL_OVERRIDE must point at the Opik backend (e.g. http://host.docker.internal:8080).
from dotenv import load_dotenv
load_dotenv()

_OPIK_URL = os.getenv("OPIK_URL_OVERRIDE", "").strip()
if _OPIK_URL and not _OPIK_URL.endswith("/api"):
    _OPIK_URL = f"{_OPIK_URL.rstrip('/')}/api"

_ENABLED = bool(_OPIK_URL)

if _ENABLED:
    try:
        os.environ["OPIK_URL_OVERRIDE"] = _OPIK_URL
        os.environ.setdefault("OPIK_CONSOLE_LOGGING_LEVEL", "WARNING")
        os.environ.setdefault("OPIK_LOG_START_TRACE_SPAN", "false")
        import opik as _opik_sdk
        logger.info("[Opik] Tracing enabled → %s", _OPIK_URL)
    except Exception as _init_err:
        logger.warning("[Opik] SDK init failed (%s) — tracing disabled.", _init_err)
        _ENABLED = False


# ── Conditional @track decorator ─────────────────────────────────────────────

import functools

def track(**kwargs):
    """
    Wrap a function with opik.track when Opik is enabled; return it unchanged
    otherwise. Usage is identical to @opik.track():

        @opik_tracing.track(type="llm", name="llm.ask")
        def ask(...):
            ...
    """
    def decorator(func):
        if not _ENABLED:
            return func
        try:
            import opik as _opik_sdk
            tracked_func = _opik_sdk.track(**kwargs)(func)

            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                try:
                    return tracked_func(*args, **func_kwargs)
                finally:
                    try:
                        _opik_sdk.flush_tracker()
                    except Exception:
                        pass
            return wrapper
        except Exception as exc:
            logger.debug("[Opik] Could not decorate %s: %s", getattr(func, "__name__", func), exc)
            return func
    return decorator


# ── LLM judge model factory ───────────────────────────────────────────────────

def _judge_model():
    """
    Return an opik LiteLLMChatModel configured to use the same LLM provider
    that the app itself uses (via config.py).  Falls back gracefully if any
    import fails.
    """
    try:
        import config
        from opik.evaluation.models import LiteLLMChatModel

        if config.LLM_PROVIDER == "openrouter":
            return LiteLLMChatModel(
                model_name=f"openrouter/{config.OPENROUTER_MODEL}",
                api_key=config.OPENROUTER_API_KEY,
            )
        if config.LLM_PROVIDER == "nvidia":
            return LiteLLMChatModel(
                model_name=f"nvidia_nim/{config.NVIDIA_MODEL}",
                api_key=config.NVIDIA_API_KEY,
                api_base=config.NVIDIA_ENDPOINT,
            )
        # Azure
        return LiteLLMChatModel(
            model_name=f"azure/{config.AZURE_OPENAI_DEPLOYMENT}",
            api_key=config.AZURE_OPENAI_API_KEY,
            api_base=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    except Exception as exc:
        logger.warning("[Opik] Could not build judge model: %s", exc)
        return None


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_evals(question: str, answer: str, context: str, intent: str = "factual") -> dict:
    """
    Run three LLM-as-judge evaluation metrics for a generated answer and log
    the scores to the active Opik span.

    Metrics:
      • Hallucination     — are any answer claims absent from the retrieved context?
      • AnswerRelevance   — does the answer address the question?
      • ContextPrecision  — are the retrieved context chunks relevant to the question?

    Returns a dict of {metric_name: score_value} (0.0–1.0).
    Returns {} if Opik is disabled or any error occurs.
    """
    if not _ENABLED:
        return {}

    scores: dict = {}
    try:
        from opik.evaluation.metrics import Hallucination, AnswerRelevance, ContextPrecision
        from opik import opik_context

        judge = _judge_model()
        context_list = [context] if isinstance(context, str) and context else []

        kwargs_base = {"model": judge} if judge is not None else {}

        hal_metric = Hallucination(**kwargs_base)
        rel_metric = AnswerRelevance(require_context=False, **kwargs_base)
        prec_metric = ContextPrecision(**kwargs_base)

        feedback_list = []

        # 1. Hallucination
        if context_list:
            try:
                hal_res = hal_metric.score(input=question, output=answer, context=context_list)
                val = round(float(hal_res.value), 4) if hal_res.value is not None else 0.0
                scores["hallucination"] = val
                feedback_list.append({"name": "hallucination", "value": val, "reason": getattr(hal_res, "reason", "")})
            except Exception as e:
                logger.warning("[Opik Evals] Hallucination metric failed: %s", e)

        # 2. AnswerRelevance
        try:
            rel_res = rel_metric.score(input=question, output=answer, context=context_list) if context_list else rel_metric.score(input=question, output=answer)
            val = round(float(rel_res.value), 4) if rel_res.value is not None else 0.0
            scores["answer_relevance"] = val
            feedback_list.append({"name": "answer_relevance", "value": val, "reason": getattr(rel_res, "reason", "")})
        except Exception as e:
            logger.warning("[Opik Evals] AnswerRelevance metric failed: %s", e)

        # 3. ContextPrecision
        if context_list:
            try:
                prec_res = prec_metric.score(input=question, output=answer, context=context_list, expected_output=answer)
                val = round(float(prec_res.value), 4) if prec_res.value is not None else 0.0
                scores["context_precision"] = val
                feedback_list.append({"name": "context_precision", "value": val, "reason": getattr(prec_res, "reason", "")})
            except Exception as e:
                logger.warning("[Opik Evals] ContextPrecision metric failed: %s", e)

        logger.info("[Opik Evals] intent=%s | scores=%s", intent, scores)

        # Attach inputs/outputs & feedback scores to both trace & span so they register in all UI views
        if feedback_list:
            try:
                opik_context.update_current_trace(
                    input={"question": question, "context": context_list},
                    output={"answer": answer},
                    feedback_scores=feedback_list
                )
            except Exception as e:
                logger.debug("[Opik] update_current_trace exception: %s", e)

            try:
                opik_context.update_current_span(
                    input={"question": question, "context": context_list},
                    output={"answer": answer},
                    metadata={**scores, "eval_intent": intent},
                    feedback_scores=feedback_list,
                )
            except Exception as e:
                logger.debug("[Opik] update_current_span exception: %s", e)

        try:
            import opik as _opik_sdk
            _opik_sdk.flush_tracker()
        except Exception:
            pass

    except Exception as exc:
        logger.warning("[Opik Evals] Evaluation failed: %s", exc)

    return scores
