"""
Cost pre-flight gate — target architecture § Phase 0-parallel, "Cost
pre-flight gate on bulk operations".

Estimates LLM call count and approximate USD cost for a batch of documents
before ingest is queued, so a bulk upload or resume_ingest can't fire an
unbounded number of paid calls with no number shown first. Every calculation
here is arithmetic over locally-extracted text length — this module makes
zero LLM/embedding API calls itself.

Call-count accuracy: _plan_calls reuses wiki._SINGLE_CALL_THRESHOLD and
wiki._split_segments directly rather than reimplementing that branching, so
the estimate can't silently drift from what wiki.ingest() actually does.

Cost accuracy: intentionally rough. Input/output token counts are chars/4
and each call's configured max_tokens cap (a worst-case ceiling, not the
model's typical output) — see config.MAX_TOKENS_INGEST_*. Pricing is public
list pricing for the configured deployments, not billing-API-verified.
Embedding cost is a coarse pages-per-call assumption, since how many wiki
pages a given call actually produces is a semantic decision only the LLM
makes at ingest time. This is a pre-flight warning, not an invoice — errs
toward overestimating, not under.
"""

import logging
import os

import config
from services import wiki

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # standard rough heuristic for English legal text

# A batch at or above either threshold is "bulk" enough to require an
# explicit confirm=true before /upload or /resume_ingest actually queues it.
COST_PREFLIGHT_MIN_FILES = int(os.getenv("COST_PREFLIGHT_MIN_FILES", "3"))
COST_PREFLIGHT_THRESHOLD_USD = float(os.getenv("COST_PREFLIGHT_THRESHOLD_USD", "0.25"))

# Public list USD price per 1M tokens, by deployment name. Falls back to the
# pricier gpt-4o tier for anything unrecognized — a spend gate should
# overestimate an unknown model's cost, not underestimate it.
_CHAT_PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5.4-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.4": {"input": 2.50, "output": 10.00},
}
_CHAT_PRICE_FALLBACK = {"input": 2.50, "output": 10.00}

_EMBED_PRICE_PER_1M = {
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}
_EMBED_PRICE_FALLBACK = {"input": 0.13, "output": 0.0}

# Overview call's input is the fixed text[:6000] + text[-3000:] excerpt built
# in wiki.ingest() — not the full document.
_OVERVIEW_INPUT_CHARS = 9000

# Rough average wiki pages created/touched per extraction call — embeddings
# are billed per page, not per input segment, and how many pages a call
# produces is a semantic LLM decision, not something computable in advance.
_EST_PAGES_PER_CALL = 3
_EMBED_CHARS_PER_PAGE = 400  # matches wiki.py's embed_text truncation


def _extract_text_cheap(file_path: str) -> str:
    """Best-effort text length probe — no OCR fallback, since this is only
    used to size an estimate, not to actually ingest. Scanned/image PDFs
    will under-count here; that's an accepted, documented approximation
    (their real cost is only known once OCR runs during actual ingest)."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "".join((p.extract_text() or "") for p in reader.pages)
        elif ext == ".docx":
            from services.reader import _read_docx
            return _read_docx(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    except Exception as e:
        logger.warning("Cost pre-flight: couldn't read %s (%s) — treating as empty", file_path, e)
        return ""


def _plan_calls(text: str) -> list[dict]:
    """The exact sequence of ingest LLM calls wiki.ingest() would make for
    this text — mirrors its single-call vs. overview+detail branching."""
    if len(text) <= wiki._SINGLE_CALL_THRESHOLD:
        return [{"type": "single", "input_chars": len(text), "max_output_tokens": config.MAX_TOKENS_INGEST_SINGLE}]

    calls = [{"type": "overview", "input_chars": _OVERVIEW_INPUT_CHARS, "max_output_tokens": config.MAX_TOKENS_INGEST_OVERVIEW}]
    for seg in wiki._split_segments(text):
        calls.append({"type": "detail", "input_chars": len(seg), "max_output_tokens": config.MAX_TOKENS_INGEST_DETAIL})
    return calls


def estimate_ingest_cost(file_paths: list[str]) -> dict:
    """Estimate LLM call count and USD cost for ingesting these files.

    Pure arithmetic over locally-extracted text — makes no LLM or embedding
    API calls. Meant to be shown to the user for confirmation before a bulk
    upload or resume_ingest queues real ingest work.
    """
    chat_deployment = config.AZURE_OPENAI_DEPLOYMENT
    chat_price = _CHAT_PRICE_PER_1M.get(chat_deployment, _CHAT_PRICE_FALLBACK)
    embed_deployment = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    embed_price = _EMBED_PRICE_PER_1M.get(embed_deployment, _EMBED_PRICE_FALLBACK)

    docs = []
    total_calls = 0
    total_input_tokens = 0.0
    total_output_tokens = 0.0
    total_pages_est = 0

    for path in file_paths:
        text = _extract_text_cheap(path)
        calls = _plan_calls(text) if text else [{"type": "single", "input_chars": 0, "max_output_tokens": config.MAX_TOKENS_INGEST_SINGLE}]

        doc_input_tokens = sum(c["input_chars"] for c in calls) / CHARS_PER_TOKEN
        doc_output_tokens = sum(c["max_output_tokens"] for c in calls)  # tokens already, not chars
        doc_pages_est = max(1, len(calls) * _EST_PAGES_PER_CALL)

        docs.append({
            "filename": os.path.basename(path),
            "chars": len(text),
            "llm_calls": len(calls),
            "estimated_pages": doc_pages_est,
        })
        total_calls += len(calls)
        total_input_tokens += doc_input_tokens
        total_output_tokens += doc_output_tokens
        total_pages_est += doc_pages_est

    chat_cost = (
        total_input_tokens / 1_000_000 * chat_price["input"]
        + total_output_tokens / 1_000_000 * chat_price["output"]
    )
    embed_tokens = total_pages_est * _EMBED_CHARS_PER_PAGE / CHARS_PER_TOKEN
    embed_cost = embed_tokens / 1_000_000 * embed_price["input"]

    return {
        "documents": docs,
        "document_count": len(file_paths),
        "estimated_llm_calls": total_calls,
        "estimated_pages": total_pages_est,
        "estimated_cost_usd": round(chat_cost + embed_cost, 4),
        "chat_deployment": chat_deployment,
        "embed_deployment": embed_deployment,
        "note": (
            "Rough order-of-magnitude estimate from locally-extracted text length "
            "and each call's configured max-token ceiling, not actual model usage. "
            "Scanned/image PDFs are undercounted (OCR text isn't available until "
            "ingest actually runs). Not a substitute for real billing."
        ),
    }


def needs_confirmation(estimate: dict, file_count: int) -> bool:
    """Whether this batch is bulk enough to hold for explicit confirm=true
    rather than queuing ingest immediately."""
    return (
        file_count >= COST_PREFLIGHT_MIN_FILES
        or estimate["estimated_cost_usd"] >= COST_PREFLIGHT_THRESHOLD_USD
    )
