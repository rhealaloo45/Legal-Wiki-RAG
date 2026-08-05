"""
Flask application — RAG vs LLM Wiki comparison app.

Routes:
  GET  /              → Serve single-page UI
  POST /upload        → Ingest file into both RAG and Wiki pipelines (parallel)
  POST /query         → Query both pipelines in parallel, return side-by-side answers
  GET  /wiki/graph    → Return wiki graph data for D3 rendering
  GET  /wiki/pages    → Return list of all wiki page titles for the browser panel
  GET  /wiki/page     → Return full content of a single wiki page
  GET  /progress      → Poll ingest progress
  DELETE /session     → Clear all data for a session (reset)
  GET  /health        → Health check — confirms app + data dirs are reachable
"""

import os
import re
import sys
import uuid
import time
import json
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

# Ensure project root is on the path so `import config` works
sys.path.insert(0, os.path.dirname(__file__))

import config
from services import wiki, hybrid, advanced_modes, draft
import threading

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256 MB total upload limit (folder uploads)

# Stores for Review/Compare modes
REVIEW_STORE = {}
COMPARE_STORE = {}
_review_locks = {}
_compare_locks = {}

def _get_review_lock(job_id):
    if job_id not in _review_locks:
        _review_locks[job_id] = threading.Lock()
    return _review_locks[job_id]

def _get_compare_lock(job_id):
    if job_id not in _compare_locks:
        _compare_locks[job_id] = threading.Lock()
    return _compare_locks[job_id]

# Ensure data directories exist
for d in [config.WIKI_PATH, config.UPLOAD_PATH]:
    os.makedirs(d, exist_ok=True)

# Configure Tesseract OCR path if set in .env (Windows users)
if config.TESSERACT_CMD:
    from services.reader import configure_tesseract
    configure_tesseract(config.TESSERACT_CMD)

executor = ThreadPoolExecutor(max_workers=10)

# Per-session locks so concurrent ingest threads can atomically increment the
# wiki_done counter without racing each other and under-counting completions.
_progress_locks: dict[str, threading.Lock] = {}
_progress_locks_lock = threading.Lock()

def _get_progress_lock(session_id: str) -> threading.Lock:
    with _progress_locks_lock:
        if session_id not in _progress_locks:
            _progress_locks[session_id] = threading.Lock()
        return _progress_locks[session_id]

# ---------------------------------------------------------------------------
# Progress store helpers — abstract over in-memory dict vs PostgreSQL (S5)
# ---------------------------------------------------------------------------
def _update_doc_status(session_id: str, doc_name: str, status: str,
                        step: str = "", pages: int = 0) -> None:
    """Set a specific document's status in the docs_list of the progress dict."""
    with _get_progress_lock(session_id):
        progress = _get_progress(session_id)
        for doc in progress.get("docs_list", []):
            if doc.get("name") == doc_name:
                doc["status"] = status
                doc["step"] = step
                if pages:
                    doc["pages"] = pages
                break
        _set_progress(session_id, progress)


def _refresh_wiki_stats(session_id: str) -> None:
    """Update pages_total and relations_total in the wiki progress block."""
    try:
        docs_in_db = 0
        if config.USE_DATABASE:
            from services import db as _db
            pages = _db.count_pages(session_id)
            rels = _db.count_relations(session_id)
            # Distinct documents actually persisted — used to reconcile the
            # wiki_done counter below.
            try:
                docs_in_db = len(_db.get_source_docs(session_id))
            except Exception:
                docs_in_db = 0
        else:
            idx = wiki._load_index(session_id)
            pages = len(idx.get("pages", {}))
            rels = len(idx.get("relations", []))
        with _get_progress_lock(session_id):
            progress = _get_progress(session_id)
            wiki_prog = progress.get("wiki", {})
            wiki_prog["pages_total"] = pages
            wiki_prog["relations_total"] = rels
            progress["wiki"] = wiki_prog
            # Reconcile the done-counter with reality: wiki_done is bumped in a
            # worker's finally-block AFTER the doc's pages are written to the DB,
            # so a worker killed mid-flight (e.g. by the Flask auto-reloader
            # restarting the server) writes the doc but never increments the
            # counter — leaving the UI showing fewer done than are actually in
            # the DB. Floor the counter at the true distinct-doc count (never
            # decrease it, never exceed total) so the display self-heals.
            if docs_in_db:
                docs = progress.get("docs", {})
                total = docs.get("total", 0)
                floor = min(docs_in_db, total) if total else docs_in_db
                docs["wiki_done"] = max(docs.get("wiki_done", 0), floor)
                progress["docs"] = docs
            _set_progress(session_id, progress)
    except Exception as e:
        logger.error("Failed to refresh wiki stats: %s", e)


def _get_progress(session_id: str) -> dict:
    if config.USE_DATABASE:
        from services import db as _db
        return _db.get_progress(session_id) or {}
    return config.PROGRESS_STORE.get(session_id, {})


def _set_progress(session_id: str, data: dict) -> None:
    if config.USE_DATABASE:
        from services import db as _db
        _db.set_progress(session_id, data)
    else:
        config.PROGRESS_STORE[session_id] = data


def _delete_progress(session_id: str) -> None:
    if config.USE_DATABASE:
        from services import db as _db
        _db.delete_progress(session_id)
    else:
        config.PROGRESS_STORE.pop(session_id, None)


ALLOWED_EXTENSIONS = {".txt", ".pdf"}

# ---------------------------------------------------------------------------
# RAG Query Logging — append every query/context/response to a JSON file
# ---------------------------------------------------------------------------
RAG_QUERY_LOG_PATH = os.path.join(config.LOGS_PATH, "rag_query_log.json")
_rag_log_lock = threading.Lock()


def _log_rag_query(question: str, wiki_context: str, answer: str) -> None:
    """Append a query record to the RAG query log JSON file.

    The wiki_context string is split on '## ' page headings so each
    retrieved wiki section becomes a separate entry in the contexts list.
    """
    # Split the context into individual page chunks
    contexts: list[str] = []
    if wiki_context:
        # wiki_context is formatted as "## Title\ncontent\n\n## Title2\ncontent2..."
        import re as _re
        parts = _re.split(r'(?=^## )', wiki_context, flags=_re.MULTILINE)
        for part in parts:
            chunk = part.strip()
            if chunk:
                contexts.append(chunk)

    record = {
        "query": question,
        "contexts": contexts,
        "response": answer,
    }

    with _rag_log_lock:
        # Read existing log (or start fresh)
        existing: list[dict] = []
        if os.path.exists(RAG_QUERY_LOG_PATH):
            try:
                with open(RAG_QUERY_LOG_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        existing.append(record)

        with open(RAG_QUERY_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info("Logged RAG query to %s (total records: %d)", RAG_QUERY_LOG_PATH, len(existing))


def load_sessions():
    if not os.path.exists(config.SESSIONS_PATH):
        return {}
    try:
        with open(config.SESSIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load sessions: %s", e)
        return {}

def save_sessions(sessions):
    try:
        with open(config.SESSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2)
    except Exception as e:
        logger.error("Failed to save sessions: %s", e)


def _get_main_session_id() -> str:
    """The session every chat currently answers from — a mutable pointer.

    Starts at config.PRODUCTION_WIKI_SESSION_ID (the .env default) and is
    updated by _set_main_session_id() whenever a local ingest completes, so
    "New chat" always targets whatever was most recently ingested. On the
    deployed Azure app DISABLE_INGEST=true means this file is never written,
    so it always falls back to the fixed .env value.
    """
    try:
        with open(config.MAIN_SESSION_PATH, "r", encoding="utf-8") as f:
            sid = json.load(f).get("session_id", "")
            if sid:
                return sid
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return config.PRODUCTION_WIKI_SESSION_ID


def _set_main_session_id(session_id: str) -> None:
    try:
        with open(config.MAIN_SESSION_PATH, "w", encoding="utf-8") as f:
            json.dump({"session_id": session_id}, f)
        logger.info("Main session pointer updated to %s", session_id)
    except OSError as e:
        logger.error("Failed to update main session pointer: %s", e)


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the single-page UI."""
    main_session_id = _get_main_session_id()
    production_wiki_name = ""
    if main_session_id:
        sessions = load_sessions()
        production_wiki_name = sessions.get(main_session_id, {}).get("name", main_session_id)
    return render_template("index.html", llm_provider="AZURE OPENAI",
                            production_mode=bool(main_session_id),
                            production_wiki_name=production_wiki_name,
                            ingest_disabled=config.DISABLE_INGEST)


@app.route("/health")
def health():
    """Health check — confirms the app is running and data dirs are accessible."""
    checks = {
        "status": "ok",
        "llm_provider": "azure",
        "model": config.AZURE_OPENAI_DEPLOYMENT,
        "data_dirs": {
            "wiki": os.path.isdir(config.WIKI_PATH),
            "uploads": os.path.isdir(config.UPLOAD_PATH),
        },
    }
    return jsonify(checks)


def _locked_in_production():
    """Guard for ingest-capable / session-destructive routes.

    Gated on DISABLE_INGEST specifically, NOT on whether a main-session
    pointer exists — locally we want a main session (so chats are consistent)
    while still allowing ingest (so a fresh local ingest can become the new
    main session, see _set_main_session_id). DISABLE_INGEST is true only on
    the deployed Azure app, where ingestion never runs — new content is
    ingested locally and shipped to Azure Postgres out of band.
    Returns a 403 response to short-circuit the route, or None to let it
    proceed.
    """
    if config.DISABLE_INGEST:
        return jsonify({"error": "Ingestion is disabled on this deployment. "
                                  "Content is updated by the wiki administrator out of band."}), 403
    return None


@app.route("/api/admin/reembed/<session_id>", methods=["POST"])
def reembed_session(session_id: str):
    """Backfill pgvector embeddings for all wiki pages in a session.

    Safe to call at any time — it upserts, so re-running is idempotent.
    Use this when the embedding provider was changed after ingestion (e.g. from
    azure to openrouter) or when embeddings failed silently during ingest.

    Returns JSON: {"embedded": N, "skipped": M, "provider": "openrouter"}
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured — no embeddings to backfill"}), 400

    try:
        from services import db as _db, embedder as _embedder

        pages = _db.get_pages(session_id)
        if not pages:
            return jsonify({"error": f"No pages found for session {session_id}"}), 404

        # Build (title, text_to_embed) pairs — prefer summary, fall back to content[:400]
        pairs: list[tuple[str, str]] = []
        for title, page in pages.items():
            summary = page.get("summary", "") if isinstance(page, dict) else ""
            content = page.get("content", "") if isinstance(page, dict) else str(page)
            embed_text = (summary or content[:400]).strip()
            if embed_text:
                pairs.append((title, embed_text))

        if not pairs:
            return jsonify({"embedded": 0, "skipped": 0, "provider": config.EMBEDDING_PROVIDER}), 200

        # Embed in batches of 16 (matches embedder.py's internal batch size)
        BATCH = 16
        embedded = 0
        skipped = 0
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i : i + BATCH]
            texts = [t for _, t in batch]
            try:
                embeddings = _embedder.embed_batch(texts, is_query=False)
                for (title, _), embedding in zip(batch, embeddings):
                    _db.upsert_embedding(session_id, title, embedding)
                    embedded += 1
                logger.info("reembed %s: batch %d/%d done (%d pages)", session_id, i // BATCH + 1, (len(pairs) + BATCH - 1) // BATCH, embedded)
            except Exception as exc:
                logger.error("reembed batch %d failed: %s", i // BATCH, exc)
                skipped += len(batch)

        logger.info("reembed complete for session %s: %d embedded, %d skipped", session_id, embedded, skipped)
        return jsonify({"embedded": embedded, "skipped": skipped, "provider": config.EMBEDDING_PROVIDER})

    except Exception as exc:
        logger.error("reembed endpoint error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/settings/llm", methods=["GET"])
def get_llm_settings():
    return jsonify({"provider": getattr(config, "LLM_PROVIDER", "azure")})

@app.route("/api/settings/llm", methods=["POST"])
def set_llm_settings():
    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.json or {}
    provider = data.get("provider", "azure")
    config.LLM_PROVIDER = provider
    return jsonify({"status": "ok", "provider": provider})

@app.route("/api/rules", methods=["GET"])
def get_rules():
    from services import rules as _rules
    return jsonify({"rules": _rules.load_rules()})

@app.route("/api/rules", methods=["POST"])
def create_rule():
    from services import rules as _rules
    data = request.json or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    rule = _rules.add_rule(text)
    return jsonify({"rule": rule})

@app.route("/api/rules/<rule_id>", methods=["PUT"])
def edit_rule(rule_id):
    from services import rules as _rules
    data = request.json or {}
    rule = _rules.update_rule(rule_id, text=data.get("text"), enabled=data.get("enabled"))
    if rule is None:
        return jsonify({"error": "rule not found"}), 404
    return jsonify({"rule": rule})

@app.route("/api/rules/<rule_id>", methods=["DELETE"])
def remove_rule(rule_id):
    from services import rules as _rules
    if not _rules.delete_rule(rule_id):
        return jsonify({"error": "rule not found"}), 404
    return jsonify({"status": "ok"})

@app.route("/api/rules/reorder", methods=["PUT"])
def reorder_rules_route():
    from services import rules as _rules
    data = request.json or {}
    ordered_ids = data.get("order") or []
    return jsonify({"rules": _rules.reorder_rules(ordered_ids)})

@app.route("/api/rules/reset", methods=["POST"])
def reset_rules_route():
    from services import rules as _rules
    return jsonify({"rules": _rules.reset_rules()})


@app.route("/api/settings/embedding", methods=["GET"])
def get_embedding_settings():
    return jsonify({"provider": getattr(config, "EMBEDDING_PROVIDER", "azure")})

@app.route("/api/settings/embedding", methods=["POST"])
def set_embedding_settings():
    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.json or {}
    provider = data.get("provider", "azure")
    old_provider = config.EMBEDDING_PROVIDER
    config.EMBEDDING_PROVIDER = provider
    # If the provider changed and we're using PostgreSQL, reset the DB engine so
    # the dimension migration check re-runs against the new vector size on next use.
    if provider != old_provider and config.USE_DATABASE:
        try:
            from services import db as _db
            _db.reset_engine()
            logger.info("DB engine reset after embedding provider change: %s → %s", old_provider, provider)
        except Exception as _e:
            logger.warning("Could not reset DB engine after provider change: %s", _e)
    return jsonify({"status": "ok", "provider": provider})


@app.route("/upload", methods=["POST"])
def upload():
    """Upload files or folders → immediately accept, then ingest in background via executor.

    Supports nested folder uploads: the frontend sends a `relative_paths` JSON
    array containing the original folder-relative path for each file (e.g.
    "cases/2024/contract.pdf").  These are used to generate descriptive saved
    filenames while keeping them flat on disk.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    files = request.files.getlist("file")
    session_id = request.form.get("session_id", str(uuid.uuid4()))

    if not files or not files[0].filename:
        return jsonify({"error": "No file provided"}), 400

    # Parse optional relative paths sent by the folder-upload frontend
    relative_paths_raw = request.form.get("relative_paths", "")
    try:
        relative_paths = json.loads(relative_paths_raw) if relative_paths_raw else []
    except (json.JSONDecodeError, TypeError):
        relative_paths = []

    # Filter and save files to disk first (fast, synchronous)
    saved_paths = []
    metadata_list = []
    for i, file in enumerate(files):
        if not _allowed_file(file.filename):
            continue

        rel_path = ""
        # Use the relative path (if available) to build a descriptive flat name
        if i < len(relative_paths) and relative_paths[i]:
            rel_path = relative_paths[i]
            # e.g. "cases/2024/contract.pdf" → "cases_2024_contract.pdf"
            safe_name = rel_path.replace("/", "_").replace("\\", "_").replace(os.sep, "_")
        else:
            safe_name = file.filename.replace(os.sep, "_")

        save_path = os.path.join(config.UPLOAD_PATH, f"{session_id}_{safe_name}")
        file.save(save_path)
        saved_paths.append(save_path)
        metadata_list.append({
            "relative_path": rel_path if rel_path else file.filename,
            "filename": file.filename
        })
        logger.info("Saved upload: %s with relative path: %s", save_path, rel_path)

    if not saved_paths:
        return jsonify({"error": "No valid files (.txt, .pdf) found"}), 400

    # Initialize progress with per-document tracking
    progress = {
        "phase": "processing",
        "docs": {"total": len(saved_paths), "wiki_done": 0},
        "wiki": {"step": "queued", "message": "", "pages_total": 0, "relations_total": 0},
        "docs_list": [
            {"name": os.path.basename(p), "status": "queued", "pages": 0, "step": ""}
            for p in saved_paths
        ],
    }
    _set_progress(session_id, progress)

    # Submit all tasks to executor (non-blocking)
    for save_path, meta in zip(saved_paths, metadata_list):
        executor.submit(_ingest_single_doc_wiki, save_path, session_id)

    # Save session metadata
    sessions = load_sessions()
    sessions[session_id] = {
        "id": session_id,
        "name": f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": time.time(),
        "updated_at": time.time(),
        "files": len(saved_paths),
        "file_paths": [meta["relative_path"] for meta in metadata_list],
        "history": []
    }
    save_sessions(sessions)

    # Return immediately — frontend will poll /progress
    return jsonify({
        "status": "accepted",
        "files_queued": len(saved_paths),
        "session_id": session_id,
    })


@app.route("/resume_ingest")
def resume_ingest():
    """Re-queue only the docs from a session's uploads that never got any pages.

    For recovering an interrupted large batch (container restart mid-ingest,
    deploy mid-ingest, etc.) without re-processing — and re-billing LLM/embed
    calls for — documents that already completed. Reads straight from disk
    (config.UPLOAD_PATH) and Postgres; doesn't need the files re-uploaded,
    since they're still sitting wherever the original /upload saved them.

    Guards against being called again while a previous resume_ingest call's
    docs are still in flight — otherwise a second call could re-queue (and
    re-bill) whatever hasn't finished writing to `pages` yet.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not config.USE_DATABASE:
        return jsonify({"error": "resume_ingest requires DATABASE_URL (Postgres) mode"}), 400

    from services import db as _db
    from sqlalchemy import text
    engine = _db.get_engine()

    # Two gunicorn worker processes means an in-process Python lock alone
    # can't stop a near-simultaneous double-call landing on different
    # workers. Use a Postgres advisory lock (cross-process) around the whole
    # check-then-set critical section below instead — if another
    # resume_ingest call for this session is mid-flight, this one backs off
    # immediately rather than racing it.
    lock_key = hash(("resume_ingest", session_id)) & 0x7FFFFFFF
    with engine.connect() as lock_conn:
        got_lock = lock_conn.execute(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": lock_key}
        ).scalar()
        if not got_lock:
            return jsonify({
                "error": "Another resume_ingest call for this session is being processed right now — try again in a moment.",
            }), 409
        try:
            progress = _get_progress(session_id)
            pending_lock = progress.get("resume_lock") or []
            if pending_lock:
                status_by_name = {d.get("name"): d.get("status") for d in progress.get("docs_list", [])}
                still_running = [n for n in pending_lock if status_by_name.get(n) not in ("done", "error")]
                if still_running:
                    return jsonify({
                        "error": "A previous resume_ingest call is still in flight for this session.",
                        "still_running": len(still_running),
                        "hint": "Poll /progress until these finish before calling resume_ingest again.",
                    }), 409

            prefix = f"{session_id}_"
            uploaded = [f for f in os.listdir(config.UPLOAD_PATH) if f.startswith(prefix)]
            if not uploaded:
                return jsonify({"error": f"No uploaded files found for session {session_id}"}), 404

            with engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT DISTINCT source_doc FROM pages WHERE session_id = :sid"),
                    {"sid": session_id},
                )
                indexed = {r.source_doc for r in rows}

            missing = [f for f in uploaded if f not in indexed]
            if not missing:
                return jsonify({
                    "status": "nothing_to_resume",
                    "uploaded": len(uploaded),
                    "already_indexed": len(indexed),
                })

            new_progress = {
                "phase": "processing",
                "docs": {"total": len(missing), "wiki_done": 0},
                "wiki": {"step": "queued", "message": "", "pages_total": 0, "relations_total": 0},
                "docs_list": [
                    {"name": f, "status": "queued", "pages": 0, "step": ""}
                    for f in missing
                ],
                "resume_lock": missing,
            }
            _set_progress(session_id, new_progress)

            for fname in missing:
                save_path = os.path.join(config.UPLOAD_PATH, fname)
                executor.submit(_ingest_single_doc_wiki, save_path, session_id)

            return jsonify({
                "status": "accepted",
                "uploaded": len(uploaded),
                "already_indexed": len(indexed),
                "resuming": len(missing),
                "session_id": session_id,
            })
        finally:
            lock_conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock_key})


@app.route("/backfill_embeddings")
def backfill_embeddings_route():
    """Embed any pages that have synthesized content but no vector yet.

    Cheap fixup for pages that got interrupted between the synthesis step
    (writes to `pages`) and the embed step (writes to the embedding table) —
    e.g. a container restart landing in that gap. Runs synchronously since
    it's a handful of pages at most, not a full re-ingest.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not config.USE_DATABASE:
        return jsonify({"error": "backfill_embeddings requires DATABASE_URL (Postgres) mode"}), 400

    import backfill_embeddings
    result = backfill_embeddings.backfill(target_session=session_id)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Background Pipelines
# ---------------------------------------------------------------------------
def _ingest_single_doc_wiki(file_path: str, session_id: str):
    """Worker for Wiki building of a single doc."""
    doc_name = os.path.basename(file_path)
    try:
        result = wiki.ingest(file_path, session_id)
        pages = (result or {}).get("pages_updated", 0)
        _refresh_wiki_stats(session_id)
        _update_doc_status(session_id, doc_name, "done", "", pages)
    except Exception as e:
        logger.error("Wiki ingest failed for %s: %s", file_path, e)
        _update_doc_status(session_id, doc_name, "error", str(e)[:60])
    finally:
        with _get_progress_lock(session_id):
            progress = _get_progress(session_id)
            docs = progress.get("docs", {})
            docs["wiki_done"] = docs.get("wiki_done", 0) + 1
            progress["docs"] = docs
            _set_progress(session_id, progress)
        _check_completion(session_id)


def _check_completion(session_id: str):
    """Mark phase as complete when all documents finish both pipelines.

    A completed local ingest becomes the new main session — every subsequent
    chat (any session_id) answers from it — unless ingest is disabled
    (deployed Azure app), where this code path never runs since /upload
    already 403s before any doc reaches _ingest_single_doc_wiki.
    """
    with _get_progress_lock(session_id):
        progress = _get_progress(session_id)
        docs = progress.get("docs", {})
        total = docs.get("total", 0)
        already_complete = progress.get("phase") == "complete"
        # if total > 0 and docs.get("rag_done", 0) >= total and docs.get("wiki_done", 0) >= total:
        if total > 0 and docs.get("wiki_done", 0) >= total:
            progress["phase"] = "complete"
            _set_progress(session_id, progress)
            if not already_complete and not config.DISABLE_INGEST:
                _set_main_session_id(session_id)


@app.route("/messages")
def get_messages():
    """Return chat message history for a session."""
    session_id = request.args.get("session_id", "")
    limit = int(request.args.get("limit", "100"))
    if not session_id:
        return jsonify({"messages": []})
    if config.USE_DATABASE:
        from services import db as _db
        messages = _db.get_messages(session_id, limit=limit)
    else:
        messages = []
    return jsonify({"messages": messages})


@app.route("/document/locate")
def locate_in_document():
    """Find the page number and character offset of a quote in a source document."""
    session_id = request.args.get("session_id", "")
    doc_name = request.args.get("doc_name", "").strip()
    quote = request.args.get("quote", "").strip()
    if not session_id or not doc_name or not quote:
        return jsonify({"found": False, "page_num": 0, "char_offset": 0})
    session_id = _get_main_session_id() or session_id
    if config.USE_DATABASE:
        from services import db as _db
        result = _db.find_quote_position(session_id, doc_name, quote)
        return jsonify(result)
    return jsonify({"found": False, "page_num": 0, "char_offset": 0})


def _store_chat_msg(session_id, role, content, msg_type="text", metadata=None):
    """Insert a chat message if DB is enabled, otherwise no-op."""
    if config.USE_DATABASE:
        from services import db as _db
        try:
            return _db.insert_message(session_id, role, content, msg_type, metadata)
        except Exception as e:
            logger.error("Failed to store chat message: %s", e)
    return None


def _update_session_history(session_id: str, question: str) -> None:
    """Push a question to the session's history and auto-name new sessions.

    Also CREATES the sessions.json entry if it doesn't exist yet — previously
    this only updated an existing entry, which every ingest-driven session had
    (created in /upload) but a chat-only "New Chat" session never does. Without
    this, chat-only sessions had messages persisted in Postgres but were
    invisible in the sidebar session list — indistinguishable from being lost.
    """
    sessions = load_sessions()
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "name": f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "created_at": time.time(),
            "updated_at": time.time(),
            "files": 0,
            "file_paths": [],
            "history": [],
        }
    if session_id in sessions:
        if not sessions[session_id].get("history") or sessions[session_id]["history"][0] != question:
            sessions[session_id].setdefault("history", []).insert(0, question)
        sessions[session_id]["updated_at"] = time.time()
        if len(sessions[session_id]["history"]) == 1 and sessions[session_id]["name"].startswith("Session "):
            new_name = question[:30] + ("..." if len(question) > 30 else "")
            sessions[session_id]["name"] = new_name
        save_sessions(sessions)


@app.route("/query", methods=["POST"])
def query_route():
    """Query the wiki pipeline via the LangGraph intent agent.

    Streams Server-Sent Events: progress stages (classifying, intent_identified,
    retrieving, pages_retrieved, generating) followed by a terminal event of type
    'answer', 'disambiguation', or 'clarification'.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "")
    target_doc = data.get("target_doc", "").strip()
    is_followup = data.get("is_followup", False)
    exclude_cached_answers = bool(data.get("exclude_cached_answers", False))

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400

    # session_id scopes chat history (per-thread); wiki_session_id scopes which
    # wiki content is searched. In production these diverge on purpose — every
    # chat thread queries the one fixed wiki. In dev, with PRODUCTION_WIKI_SESSION_ID
    # unset, they're the same value and behavior is unchanged.
    wiki_session_id = _get_main_session_id() or session_id

    # Store user message in chat history (once, before streaming)
    _store_chat_msg(session_id, "user", question, "text")

    t0 = time.time()

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    @stream_with_context
    def generate():
        from services import intent_agent
        final_emitted = False
        try:
            for ev in intent_agent.run_query_stream(question, wiki_session_id, target_doc, is_followup,
                                                     exclude_cached_answers,
                                                     chat_session_id=session_id):
                etype = ev.get("type")
                logger.info("SSE stage: %s | %s", ev.get("stage", etype), ev.get("message", ""))

                if etype == "disambiguation":
                    # Deliberately never forwards a document list — see
                    # intent_agent.check_disambiguation_node. Only the message
                    # and the original question (so a page reload can still
                    # resolve a typed reply) leave the server.
                    payload = ev.get("payload", {})
                    _store_chat_msg(session_id, "assistant", payload.get("message", ""),
                                    "disambiguation",
                                    {"original_question": payload.get("original_question", question)})
                    final_emitted = True
                    yield _sse({
                        "type": "disambiguation",
                        "message": payload.get("message", ""),
                        "total_elapsed_ms": round((time.time() - t0) * 1000),
                    })

                elif etype == "clarification":
                    payload = ev.get("payload", {})
                    _store_chat_msg(session_id, "assistant", payload.get("message", ""),
                                    "clarification",
                                    {"options": payload.get("options", []),
                                     "original_question": question})
                    final_emitted = True
                    yield _sse({
                        "type": "clarification",
                        "message": payload.get("message", ""),
                        "options": payload.get("options", []),
                        "total_elapsed_ms": round((time.time() - t0) * 1000),
                    })

                elif etype == "answer":
                    wiki_result = ev.get("payload", {})
                    # Pop before anything touches wiki_result further — never send the
                    # raw retrieved context over SSE to the frontend, only use it locally
                    # for the RAG query log.
                    debug_context = wiki_result.pop("_debug_context", "")
                    wiki_result["elapsed_ms"] = round((time.time() - t0) * 1000)
                    _store_chat_msg(session_id, "assistant", wiki_result.get("answer", ""),
                                    "answer", {
                                        "confidence_score": wiki_result.get("confidence_score", 0),
                                        "files_used": wiki_result.get("files_used", []),
                                        "token_total": wiki_result.get("token_total", {}),
                                        "validation": wiki_result.get("validation", {}),
                                        "intent": wiki_result.get("intent", "factual"),
                                        "intent_label": wiki_result.get("intent_label", ""),
                                        "intent_confidence": wiki_result.get("intent_confidence", 0),
                                        # Consumed by the NEXT turn's scope
                                        # carryover (wiki._carryover_scope).
                                        "scope_method": wiki_result.get("scope_method", ""),
                                        "scope_docs": wiki_result.get("scope_docs", []),
                                        # Render flags — without these a reloaded
                                        # thread shows a help/greeting reply as a
                                        # normal answer card, and drops the
                                        # not-legal-advice notice entirely.
                                        "meta_answer": wiki_result.get("meta_answer", False),
                                        "advice_notice": wiki_result.get("advice_notice", ""),
                                        # Deterministic term-presence warning. Must
                                        # persist for the same reason as the notice
                                        # above: the caution has to survive a reload,
                                        # or a reopened thread shows the unverified
                                        # answer with nothing marking it.
                                        "context_warning": wiki_result.get("context_warning", ""),
                                        "context_note": wiki_result.get("context_note", ""),
                                        # Deterministic counts shown beside the
                                        # confidence percentage; must survive a
                                        # reload like the banners do.
                                        "answer_facts": wiki_result.get("answer_facts", {}),
                                        # Was set on the live payload but never stored,
                                        # so every reloaded answer rendered "0.0s".
                                        "elapsed_ms": wiki_result.get("elapsed_ms", 0),
                                    })
                    _update_session_history(session_id, question)
                    try:
                        _log_rag_query(question, debug_context, wiki_result.get("answer", ""))
                    except Exception as log_err:
                        logger.error("Failed to log RAG query: %s", log_err)
                    final_emitted = True
                    logger.info("SSE answer: intent=%s conf=%s%%",
                                wiki_result.get("intent"), wiki_result.get("confidence_score"))
                    yield _sse({
                        "type": "answer",
                        "wiki": wiki_result,
                        "total_elapsed_ms": round((time.time() - t0) * 1000),
                    })

                else:
                    yield _sse(ev)

        except Exception as e:
            logger.error("Query stream failed (%s): %s", type(e).__name__, e)
            yield _sse({"type": "error", "error": f"{type(e).__name__}: {e}"})

        if not final_emitted:
            yield _sse({"type": "error", "error": "No answer was produced."})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Matches the document-type folder segment inside an underscore-flattened
# source_doc name (e.g. "...Court Case Documents (1)_Court Case Document 1
# (1).pdf"), regardless of what the TOP-level upload folder is called — used
# by /files's reconstruction fallback below, which previously hardcoded one
# specific top-level name and un-suffixed subfolder names that matched
# neither this corpus's real prefix ("Legal AI - Test"/"Legal AI - Raja") nor
# its "(1)"-suffixed subfolder names, so every file fell through to a single
# flat level with no folder grouping at all (confirmed live on the deployed
# instance, whose wiki arrived via DB import rather than the app's own
# /upload flow — so it never has a `sessions.json` entry with file_paths,
# and always hits this fallback).
_DOC_TYPE_FOLDER_RE = re.compile(
    r'(Court\s+Case\s+Documents?|Joint\s+Venture\s+Agreements?|Judgments?|'
    r'Legal\s+Opinions?|NDA|Service\s+Agreements?|Shareholders?\s+Agreements?)'
    r'(?:\s*\(\d+\))?',
    re.IGNORECASE,
)


@app.route("/files")
def file_structure():
    """Return nested file tree of successfully uploaded files."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({})
    session_id = _get_main_session_id() or session_id
        
    try:
        sessions = load_sessions()
        paths = set()

        # Docs with zero indexed pages (OCR/ingest failures) never made it
        # into the corpus — exclude them here so the Files tab doesn't imply
        # they're searchable when generate_answer() will never see them.
        indexed_docs: set[str] = set()
        if config.USE_DATABASE:
            try:
                from services import db as _db
                from sqlalchemy import text
                engine = _db.get_engine()
                with engine.connect() as conn:
                    rows = conn.execute(
                        text("SELECT DISTINCT source_doc FROM pages WHERE session_id = :sid"),
                        {"sid": session_id},
                    )
                    indexed_docs = {r.source_doc for r in rows}
            except Exception as _idx_err:
                logger.warning("Could not fetch indexed docs for /files filtering: %s", _idx_err)

        # 1. Try to read from session metadata first
        if session_id in sessions and "file_paths" in sessions[session_id]:
            for p in sessions[session_id]["file_paths"]:
                # source_doc is stored as session_id + "_" + path with "/" flattened to "_"
                flat_key = f"{session_id}_" + p.replace("\\", "/").replace("/", "_")
                if indexed_docs and flat_key not in indexed_docs:
                    continue
                paths.add(p.replace("\\", "/"))
        else:
            # 2. Fallback: Scan config.UPLOAD_PATH and reconstruct original paths
            prefix = f"{session_id}_"
            for fname in os.listdir(config.UPLOAD_PATH):
                if fname.startswith(prefix):
                    if indexed_docs and fname not in indexed_docs:
                        continue
                    rel_name = fname[len(prefix):]

                    # Reconstruct "<top folder>/<doc-type folder>/<filename>"
                    # by locating the doc-type segment wherever it falls —
                    # works for any top-level folder name, not just one
                    # hardcoded string. Underscores are the flattening
                    # separator only at the boundary right around that match;
                    # trimming just there (not a blind split on every "_")
                    # keeps underscores that are part of the filename itself
                    # intact (e.g. "Test_CCD_01.txt").
                    reconstructed = False
                    m = _DOC_TYPE_FOLDER_RE.search(rel_name)
                    if m:
                        top = rel_name[:m.start()].rstrip('_')
                        type_folder = m.group(0)
                        file_part = rel_name[m.end():].lstrip('_')
                        if top and file_part:
                            paths.add(f"{top}/{type_folder}/{file_part}")
                            reconstructed = True

                    if not reconstructed:
                        paths.add(rel_name.replace("\\", "/"))
                        
        tree = {}
        for p in paths:
            parts = [x for x in p.split("/") if x]
            curr = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    curr[part] = "file"
                else:
                    curr = curr.setdefault(part, {})
                    
        return jsonify(tree)
    except Exception as e:
        logger.error("Failed to fetch file structure: %s", e)
        return jsonify({})


@app.route("/wiki/graph")
def wiki_graph():
    """Return wiki graph data (pages + relations) for D3 rendering."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"pages": {}, "relations": []})
    session_id = _get_main_session_id() or session_id
    return jsonify(wiki.get_graph(session_id))


@app.route("/wiki/backfill_embeddings", methods=["POST"])
def wiki_backfill_embeddings():
    """Generate embeddings for pages that lack them — enables pgvector hybrid
    retrieval for sessions ingested before embeddings existed (or when the
    embedding API was rate-limited). Safe to run repeatedly."""
    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "") or request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    try:
        result = wiki.backfill_embeddings(session_id)
        return jsonify(result)
    except Exception as e:
        logger.error("Backfill embeddings failed: %s", e)
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.route("/wiki/backfill_source_docs", methods=["POST"])
def wiki_backfill_source_docs():
    """Populate empty source_doc fields from page title parentheses."""
    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "") or request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    try:
        result = wiki.backfill_source_docs(session_id)
        return jsonify(result)
    except Exception as e:
        logger.error("Backfill source_docs failed: %s", e)
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.route("/wiki/pages")
def wiki_pages_list():
    """Return a sorted list of all wiki page titles for the browser panel."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"pages": []})
    session_id = _get_main_session_id() or session_id
    index = wiki.get_graph(session_id)
    titles = sorted(index.get("pages", {}).keys())
    return jsonify({"pages": titles})


@app.route("/wiki/page")
def wiki_page_detail():
    """Return the full content of a single wiki page by title."""
    session_id = request.args.get("session_id", "")
    title = request.args.get("title", "").strip()
    if not session_id or not title:
        return jsonify({"error": "session_id and title are required"}), 400
    session_id = _get_main_session_id() or session_id
    index = wiki.get_graph(session_id)
    content = index.get("pages", {}).get(title)
    if content is None:
        return jsonify({"error": "Page not found"}), 404
    return jsonify({"title": title, "content": content})


def _find_upload(session_id, doc_name):
    """Find an uploaded file by session ID and document name."""
    # Strip the compare mode upload marker if present
    doc_name = doc_name.replace(" ⚡", "").strip()
    basename = doc_name.replace("/", "_").replace("\\", "_")
    
    # 1. Try with the current session prefix
    prefix = f"{session_id}_"
    for fname in os.listdir(config.UPLOAD_PATH):
        if fname.startswith(prefix) and fname.endswith(basename):
            return os.path.join(config.UPLOAD_PATH, fname)
            
    # 2. Resilient fallback: search across all uploads for any session matching basename
    for fname in os.listdir(config.UPLOAD_PATH):
        if fname.endswith(basename):
            return os.path.join(config.UPLOAD_PATH, fname)
            
    return None


@app.route("/document")
def get_document():
    """Return the raw text content of an uploaded document."""
    session_id = request.args.get("session_id", "")
    doc_name = request.args.get("name", "").strip()
    if not session_id or not doc_name:
        return jsonify({"error": "session_id and name are required"}), 400
    session_id = _get_main_session_id() or session_id

    from services.reader import read_file
    target = _find_upload(session_id, doc_name)

    if not target or not os.path.exists(target):
        return jsonify({"error": "Document not found"}), 404

    try:
        text = read_file(target)
        return jsonify({"name": doc_name, "text": text})
    except Exception as e:
        logger.error("Failed to read document %s: %s", doc_name, e)
        return jsonify({"error": f"Failed to read document: {e}"}), 500


@app.route("/document/raw")
def get_document_raw():
    """Serve the original uploaded file (PDF/txt) for browser-native rendering."""
    from flask import send_file
    session_id = request.args.get("session_id", "")
    doc_name = request.args.get("name", "").strip()
    if not session_id or not doc_name:
        return "session_id and name are required", 400
    session_id = _get_main_session_id() or session_id

    target = _find_upload(session_id, doc_name)
    if not target or not os.path.exists(target):
        return "Document not found", 404

    ext = os.path.splitext(target)[1].lower()
    mime = "application/pdf" if ext == ".pdf" else "text/plain"
    return send_file(target, mimetype=mime)


@app.route("/log", methods=["GET"])
def get_log():
    """Return the plaintext contents of the session log."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return "session_id required", 400
    
    log_path = os.path.join(config.LOGS_PATH, f"{session_id}_log.md")
    if not os.path.exists(log_path):
        return "No log found for this session.", 404
        
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        logger.error("Failed to read log file: %s", e)
        return "Failed to read log.", 500


@app.route("/progress")
def progress():
    """Return the current progress of an ongoing ingest."""
    session_id = request.args.get("session_id", "")
    return jsonify(_get_progress(session_id) or {"rag": {}, "wiki": {}})


@app.route("/sessions", methods=["GET"])
def get_sessions():
    """Return all saved sessions sorted by recently updated."""
    sessions = load_sessions()
    session_list = sorted(sessions.values(), key=lambda x: x.get("updated_at", 0), reverse=True)
    return jsonify({"sessions": session_list})


@app.route("/session/<session_id>", methods=["GET"])
def get_session(session_id):
    """Return details for a specific session."""
    sessions = load_sessions()
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(sessions[session_id])


@app.route("/session/<session_id>", methods=["PUT"])
def rename_session(session_id):
    """Rename an existing session."""
    data = request.get_json(silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "Name is required"}), 400
        
    sessions = load_sessions()
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
        
    sessions[session_id]["name"] = new_name
    sessions[session_id]["updated_at"] = time.time()
    save_sessions(sessions)
    return jsonify({"status": "ok", "name": new_name})


@app.route("/session", methods=["DELETE"])
def clear_session():
    """Delete all data associated with a session (wiki index, uploads)."""
    _lock = _locked_in_production()
    if _lock:
        return _lock
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    errors = []

    # Remove wiki index directory
    wiki_dir = os.path.join(config.WIKI_PATH, session_id)
    if os.path.isdir(wiki_dir):
        try:
            shutil.rmtree(wiki_dir)
            logger.info("Deleted wiki dir: %s", wiki_dir)
        except Exception as e:
            errors.append(f"wiki: {e}")

    # Remove uploaded files for this session
    try:
        for fname in os.listdir(config.UPLOAD_PATH):
            if fname.startswith(session_id):
                os.remove(os.path.join(config.UPLOAD_PATH, fname))
                logger.info("Deleted upload: %s", fname)
    except Exception as e:
        errors.append(f"uploads: {e}")

    # Clear progress store entry
    _delete_progress(session_id)

    # Clear chat messages
    if config.USE_DATABASE:
        try:
            from services import db as _db
            _db.delete_messages(session_id)
        except Exception as e:
            errors.append(f"chat: {e}")

    # Remove from sessions metadata
    sessions = load_sessions()
    if session_id in sessions:
        del sessions[session_id]
        save_sessions(sessions)

    if errors:
        return jsonify({"status": "partial", "errors": errors})
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Review Mode Routes
# ---------------------------------------------------------------------------
@app.route("/review/start", methods=["POST"])
def review_start():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    doc_names = data.get("doc_names", [])
    question = data.get("question", "")
    
    if not session_id or not question:
        return jsonify({"error": "session_id and question are required"}), 400
        
    job_id = str(uuid.uuid4())
    REVIEW_STORE[job_id] = {
        "status": "running",
        "total": 0,  # Will be updated by the background job once columns are generated
        "completed": 0,
        "rows": {d: {} for d in doc_names},
        "flagged": [],
        "error": None,
        "columns": [] # Will be updated by the background job
    }
    
    _get_review_lock(job_id)  # Initialize the lock for this job
    logger.info(f"Starting review job {job_id} for {len(doc_names)} docs with prompt: {question[:50]}...")
    executor.submit(advanced_modes._run_review_job, job_id, session_id, doc_names, question, REVIEW_STORE, _review_locks)
    return jsonify({"job_id": job_id})

@app.route("/review/progress")
def review_progress():
    job_id = request.args.get("job_id", "")
    if job_id not in REVIEW_STORE:
        return jsonify({"error": "job not found"}), 404
    
    store = REVIEW_STORE[job_id]
    completed = store["completed"]
    total = store["total"]
    percent = round((completed / total) * 100, 1) if total > 0 else 0
    
    return jsonify({
        "status": store["status"],
        "total": total,
        "completed": completed,
        "percent": percent,
        "flagged_count": len(store["flagged"])
      })

@app.route("/review/result")
def review_result():
    job_id = request.args.get("job_id", "")
    if job_id not in REVIEW_STORE:
        return jsonify({"error": "job not found"}), 404
    return jsonify(REVIEW_STORE[job_id])

@app.route("/review/export")
def review_export():
    from flask import send_file
    import io
    job_id = request.args.get("job_id", "")
    if job_id not in REVIEW_STORE:
        return "job not found", 404
        
    excel_bytes = advanced_modes.export_matrix_to_xlsx(REVIEW_STORE[job_id], "review")
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"Review_Export_{job_id[:8]}.xlsx"
    )

# ---------------------------------------------------------------------------
# Compare Mode Routes
# ---------------------------------------------------------------------------
@app.route("/compare/start", methods=["POST"])
def compare_start():
    session_id = request.form.get("session_id", "")
    doc_names_json = request.form.get("doc_names", "[]")
    question = request.form.get("question", "")
    uploaded_file = request.files.get("uploaded_file")
    
    try:
        doc_names = json.loads(doc_names_json)
    except:
        doc_names = []
        
    if not session_id or not question:
        return jsonify({"error": "session_id and question are required"}), 400
        
    job_id = str(uuid.uuid4())
    COMPARE_STORE[job_id] = {
        "status": "running",
        "stage": "starting",
        "question": question,
        "sources": [],
        "aspects": [],
        "table": {},
        "outliers": [],
        "narrative": None,
        "error": None
    }
    
    temp_path = None
    uploaded_text = None
    uploaded_name = None
    
    if uploaded_file and uploaded_file.filename:
        uploaded_name = uploaded_file.filename
        temp_path = os.path.join(config.UPLOAD_PATH, f"temp_{job_id}_{uploaded_name}")
        uploaded_file.save(temp_path)
        from services.reader import read_file
        try:
            uploaded_text = read_file(temp_path)
        except Exception as e:
            logger.error(f"Failed to read uploaded temp file: {e}")
            uploaded_text = ""
            
    _get_compare_lock(job_id)  # Initialize the lock for this job
    logger.info(f"Starting compare job {job_id} for {len(doc_names)} docs (upload={uploaded_name is not None})")
    executor.submit(advanced_modes._run_compare_job, job_id, session_id, doc_names, question, 
                    uploaded_text, uploaded_name, temp_path, COMPARE_STORE, _compare_locks)
    return jsonify({"job_id": job_id})

@app.route("/compare/progress")
def compare_progress():
    job_id = request.args.get("job_id", "")
    if job_id not in COMPARE_STORE:
        return jsonify({"error": "job not found"}), 404
        
    store = COMPARE_STORE[job_id]
    return jsonify({
        "status": store["status"],
        "stage": store.get("stage", "")
    })

@app.route("/compare/result")
def compare_result():
    job_id = request.args.get("job_id", "")
    if job_id not in COMPARE_STORE:
        return jsonify({"error": "job not found"}), 404
    return jsonify(COMPARE_STORE[job_id])

@app.route("/compare/export")
def compare_export():
    from flask import send_file
    import io
    job_id = request.args.get("job_id", "")
    if job_id not in COMPARE_STORE:
        return "job not found", 404
        
    excel_bytes = advanced_modes.export_matrix_to_xlsx(COMPARE_STORE[job_id], "compare")
    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"Compare_Export_{job_id[:8]}.xlsx"
    )

# ---------------------------------------------------------------------------
# Draft Mode Routes
# ---------------------------------------------------------------------------
@app.route("/draft/start", methods=["POST"])
def draft_start():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    prompt = data.get("prompt", "")
    use_wiki = data.get("use_wiki", True)
    
    if not session_id or not prompt:
        return jsonify({"error": "session_id and prompt are required"}), 400
        
    job_id = str(uuid.uuid4())
    draft.DRAFT_STORE[job_id] = {
        "status": "starting",
        "current_version": 0,
        "versions": {},
        "error": None
    }
    
    draft._get_draft_lock(job_id)
    executor.submit(draft._run_draft_job, job_id, session_id, prompt, use_wiki)
    return jsonify({"job_id": job_id})

@app.route("/draft/refine", methods=["POST"])
def draft_refine():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    job_id = data.get("job_id", "")
    instruction = data.get("instruction", "")
    
    if not job_id or not instruction:
        return jsonify({"error": "job_id and instruction are required"}), 400
        
    if job_id not in draft.DRAFT_STORE:
        return jsonify({"error": "draft job not found"}), 404
        
    executor.submit(draft._run_refine_job, job_id, session_id, instruction)
    return jsonify({"status": "refining"})

@app.route("/draft/version", methods=["GET"])
def draft_version():
    job_id = request.args.get("job_id", "")
    if job_id not in draft.DRAFT_STORE:
        return jsonify({"error": "job not found"}), 404
        
    return jsonify(draft.DRAFT_STORE[job_id])

@app.route("/draft/export", methods=["GET"])
def draft_export():
    import io
    from flask import send_file
    
    job_id = request.args.get("job_id", "")
    if job_id not in draft.DRAFT_STORE:
        return "job not found", 404
        
    store = draft.DRAFT_STORE[job_id]
    current_v = store.get("current_version", 0)
    
    if not current_v or current_v not in store.get("versions", {}):
        return "No complete draft available", 404
        
    draft_text = store["versions"][current_v]["text"]
    docx_bytes = draft.export_draft_to_docx(draft_text)
    
    return send_file(
        io.BytesIO(docx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        as_attachment=True,
        download_name=f"Draft_Export_{job_id[:8]}.docx"
    )

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    # The Werkzeug auto-reloader restarts the whole process on any .py change,
    # which KILLS in-flight ingest workers (docs already written to the DB but
    # the wiki_done counter never bumped) and drops still-queued docs — a primary
    # cause of ingestion "getting stuck" while editing code. Default the reloader
    # OFF so long-running ingests survive file saves; set FLASK_USE_RELOADER=1
    # to re-enable hot-reload for pure UI/code iteration when not ingesting.
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "0") == "1"
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=use_reloader)
