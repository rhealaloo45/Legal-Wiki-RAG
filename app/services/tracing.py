"""
Per-query tracing — records what a single /query request actually did:
stage timings, the scope decision, which pages were retrieved (and why),
every LLM call (prompt/response previews + token usage), and the final
validation result. Written once to the query_traces table when the request
finishes, keyed to the assistant chat message it produced.

Zero-cost when nothing is tracing: every call site fetches the current trace
via get_trace() and no-ops if it's None (no active request, DB mode off,
called from a script/test with no trace started).

Not thread-unsafe-by-accident: the active trace lives in a contextvar, so
concurrent requests on different threads never see each other's trace, and
finish_and_persist() always resets the var so a reused worker thread can't
leak a stale trace into the next request.
"""

import functools
import logging
import time
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

# Keep stored prompt/response text bounded — a page-selection or answer
# prompt can run to tens of thousands of chars (see MAX_TOTAL_CONTEXT_CHARS),
# and a trace exists to show WHAT was sent, not to be a second copy of the
# full wiki. Full char counts are still recorded alongside the preview.
_PREVIEW_CHARS = 2000

# Cap how many candidate titles get stored per retrieval channel — enough to
# see the shape of the ranking without a broad-question RRF fusion (which can
# rank 100+ candidates) bloating the trace row.
_MAX_CANDIDATES = 20

_current: "ContextVar[Optional[Trace]]" = ContextVar("current_trace", default=None)


class Trace:
    def __init__(self, question: str, chat_session_id: str, wiki_session_id: str):
        self.question = question
        self.chat_session_id = chat_session_id
        self.wiki_session_id = wiki_session_id
        self.started_at = time.time()
        self.stages: list[dict] = []
        self.llm_calls: list[dict] = []
        self.scope_decision: Optional[dict] = None
        self.retrieval: Optional[dict] = None
        self.pages: list[dict] = []
        self.pages_meta: dict = {}
        self.validation: Optional[dict] = None

    def record_stage(self, name: str, duration_ms: float, error: str = None) -> None:
        self.stages.append({"name": name, "duration_ms": round(duration_ms, 1), "error": error})

    def log_scope_decision(self, decision: dict) -> None:
        self.scope_decision = decision

    def log_page_selection(self, method: str, **fields) -> None:
        """Record which retrieval channel(s) produced the final page list.

        method: "vector+bm25" | "llm_select" | "keyword_fallback" | etc.
        fields: channel-specific detail, e.g. vector=[...], bm25=[...],
        selected=[...], emb_count=N, is_broad=bool. Lists are truncated to
        _MAX_CANDIDATES titles.
        """
        trimmed = {
            k: (v[:_MAX_CANDIDATES] if isinstance(v, list) else v)
            for k, v in fields.items()
        }
        self.retrieval = {"method": method, **trimmed}

    def log_pages(self, pages: list[dict], omitted: int, total_chars: int, cap: int) -> None:
        """Record the pages actually placed in the answer-generation context —
        the direct answer to "what chunks did it take"."""
        self.pages = pages
        self.pages_meta = {"omitted_for_budget": omitted, "total_chars": total_chars, "cap": cap}

    def log_llm_call(self, pipeline: str, model: str, prompt: str, response: str,
                      usage: dict, latency_ms: float, fast: bool = False) -> None:
        self.llm_calls.append({
            "pipeline": pipeline,
            "model": model,
            "fast": fast,
            "latency_ms": round(latency_ms, 1),
            "prompt_chars": len(prompt or ""),
            "prompt_preview": (prompt or "")[:_PREVIEW_CHARS],
            "response_chars": len(response or ""),
            "response_preview": (response or "")[:_PREVIEW_CHARS],
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
            "completion_tokens": (usage or {}).get("completion_tokens", 0),
            "finish_reason": (usage or {}).get("finish_reason"),
        })

    def log_validation(self, validation: dict) -> None:
        self.validation = validation

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "wiki_session_id": self.wiki_session_id,
            "stages": self.stages,
            "scope_decision": self.scope_decision,
            "retrieval": self.retrieval,
            "pages": self.pages,
            "pages_meta": self.pages_meta,
            "llm_calls": self.llm_calls,
            "llm_call_count": len(self.llm_calls),
            "total_prompt_tokens": sum(c["prompt_tokens"] for c in self.llm_calls),
            "total_completion_tokens": sum(c["completion_tokens"] for c in self.llm_calls),
            "validation": self.validation,
        }


def start_trace(question: str, chat_session_id: str, wiki_session_id: str):
    """Begin a new trace and make it the current one for this thread/request.

    Returns (trace, token) — pass both to finish_and_persist() so the
    contextvar is always reset, even if persistence itself fails.
    """
    trace = Trace(question, chat_session_id, wiki_session_id)
    token = _current.set(trace)
    return trace, token


def get_trace() -> Optional[Trace]:
    return _current.get()


def finish_and_persist(trace: Optional[Trace], token, message_id: int = None) -> Optional[int]:
    """Finalize a trace and write it to query_traces. Always resets the
    contextvar first so a reused thread never leaks a trace into the next,
    unrelated request."""
    try:
        _current.reset(token)
    except Exception:
        pass
    if trace is None:
        return None
    total_ms = round((time.time() - trace.started_at) * 1000)
    try:
        import config
        if not config.USE_DATABASE:
            return None
        from services import db
        return db.insert_trace(
            session_id=trace.chat_session_id,
            wiki_session_id=trace.wiki_session_id,
            message_id=message_id,
            question=trace.question,
            total_ms=total_ms,
            trace=trace.to_dict(),
        )
    except Exception as e:
        logger.error("Failed to persist query trace: %s", e)
        return None


def traced_node(name: str):
    """Decorator for LangGraph node functions — records wall time for the
    node under `name` on whatever trace is current, with zero overhead when
    no trace is active. Leaves the node's own logic/signature untouched."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            trace = get_trace()
            if trace is None:
                return fn(state)
            t0 = time.time()
            error = None
            try:
                return fn(state)
            except Exception as e:
                error = str(e)
                raise
            finally:
                trace.record_stage(name, (time.time() - t0) * 1000, error=error)
        return wrapper
    return decorator
