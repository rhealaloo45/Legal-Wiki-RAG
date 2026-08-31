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
from datetime import datetime, timedelta
import threading

from flask import (
    Flask, render_template, request, jsonify, Response, stream_with_context,
    session, redirect, url_for, has_request_context, g,
)

# Ensure project root is on the path so `import config` works
sys.path.insert(0, os.path.dirname(__file__))

import config
from services import wiki, hybrid, advanced_modes, draft, redaction, tracing, auth, upload_validation, cost_estimate, wiki_pages
import threading

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256 MB total upload limit (folder uploads)

# ---------------------------------------------------------------------------
# Session / auth config — see services/auth.py and config.py for the rationale
# behind each setting. Applied whether or not AUTH_ENABLED, so that turning
# auth on later doesn't also silently change cookie behaviour.
# ---------------------------------------------------------------------------
if config.FLASK_SECRET_KEY:
    app.secret_key = config.FLASK_SECRET_KEY
else:
    # Ephemeral key: sessions die on restart, and multiple gunicorn workers
    # would each sign with a different key (so logins would appear random).
    # Fine for a local dev run, never acceptable deployed — hence the warning.
    app.secret_key = os.urandom(32)
    if config.AUTH_ENABLED:
        logger.warning(
            "FLASK_SECRET_KEY is not set — using a random per-process key. "
            "Sessions will not survive a restart and will break across gunicorn "
            "workers. Set FLASK_SECRET_KEY in .env for anything but local dev."
        )

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,          # JS can't read the cookie
    SESSION_COOKIE_SAMESITE="Lax",         # blocks cross-site POST rides
    SESSION_COOKIE_SECURE=config.SESSION_COOKIE_SECURE,  # HTTPS-only; see config.py
    PERMANENT_SESSION_LIFETIME=timedelta(days=config.SESSION_LIFETIME_DAYS),
)

if not config.AUTH_ENABLED:
    logger.warning(
        "AUTH_ENABLED=false — every route is publicly reachable with no login. "
        "This is only appropriate for a trusted local instance."
    )

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

executor = ThreadPoolExecutor(max_workers=config.WIKI_MAX_WORKERS)

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
            _wid = current_wiki_id()
            pages = _db.count_pages(_wid, session_id)
            rels = _db.count_relations(_wid, session_id)
            # Distinct documents actually persisted — used to reconcile the
            # wiki_done counter below.
            try:
                docs_in_db = len(_db.get_source_docs(_wid, session_id))
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


ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

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


def _default_wiki_id() -> str:
    from services.wikis import DEFAULT_WIKI_ID
    return DEFAULT_WIKI_ID


def _get_main_session_id(wiki_id: str | None = None) -> str:
    """The session THIS WIKI's chats currently answer from — a mutable,
    per-wiki pointer.

    One pointer used to serve the whole app, back when there was only ever
    one wiki. Now that switching wikis is real, a global pointer would mean
    activating a brand-new wiki still answered chat from whatever wiki was
    ingested into last — exactly the bug this per-wiki keying closes.
    Updated by _set_main_session_id() whenever a local ingest completes, so
    "New chat" under a given wiki always targets whatever was most recently
    ingested into THAT wiki. The config.PRODUCTION_WIKI_SESSION_ID .env
    fallback only applies to the default wiki — it predates multi-wiki and
    named the one wiki that existed; a fresh wiki has no chat pointer until
    something is actually ingested into it, which is correct, not a bug.
    """
    wiki_id = wiki_id or current_wiki_id()
    try:
        with open(config.MAIN_SESSION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        by_wiki = data.get("by_wiki")
        if by_wiki is None and data.get("session_id"):
            # Pre-multi-wiki file — that single pointer belonged to the one
            # wiki that existed at the time, i.e. the default wiki.
            by_wiki = {_default_wiki_id(): data["session_id"]}
        sid = (by_wiki or {}).get(wiki_id, "")
        if sid:
            return sid
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    if wiki_id == _default_wiki_id():
        return config.PRODUCTION_WIKI_SESSION_ID
    return ""


def _set_main_session_id(session_id: str, wiki_id: str | None = None) -> None:
    wiki_id = wiki_id or current_wiki_id()
    try:
        try:
            with open(config.MAIN_SESSION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            data = {}
        by_wiki = data.get("by_wiki")
        if by_wiki is None:
            by_wiki = {}
            if data.get("session_id"):
                by_wiki[_default_wiki_id()] = data["session_id"]
        by_wiki[wiki_id] = session_id
        with open(config.MAIN_SESSION_PATH, "w", encoding="utf-8") as f:
            json.dump({"by_wiki": by_wiki}, f)
        logger.info("Main session pointer for wiki %s updated to %s", wiki_id, session_id)
    except OSError as e:
        logger.error("Failed to update main session pointer: %s", e)


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Auth gate
#
# Deliberately a global before_request hook rather than a decorator on each
# route. There are ~47 routes and more coming with the target-architecture
# work; an allowlist fails CLOSED (a new route is protected unless someone
# opts it out on purpose), while per-route decorators fail OPEN the first
# time someone forgets one. That failure is silent and the route is the
# whole vulnerability.
# ---------------------------------------------------------------------------

# Endpoint names (not URL paths) reachable without a session.
#   login/logout — the gate itself, obviously
#   health       — Azure App Service probes it unauthenticated
#   static       — CSS/JS/favicon; the login page needs them to render
_PUBLIC_ENDPOINTS = {"login", "logout", "health", "static"}


def _wants_html() -> bool:
    """True for a browser navigating, false for the SPA's fetch() calls.

    Browser address-bar navigation sends `Accept: text/html,...`; fetch()
    defaults to `*/*`. Decides redirect-to-login vs. 401-JSON so the SPA gets
    a status it can act on instead of a login page rendered into a JSON parse.
    """
    return request.method == "GET" and "text/html" in request.headers.get("Accept", "")


def _safe_next(target: str) -> str:
    """Only allow same-site relative redirects.

    Without this, `/login?next=https://evil.example` turns the login form into
    an open redirect — a credential-phishing primitive, since the URL genuinely
    starts on this trusted origin. A leading `//` is also rejected: browsers
    read `//evil.example` as protocol-relative and leave the site.
    """
    if not target or not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


@app.before_request
def _bind_wiki_id():
    """Bind the active wiki once, at request entry (§ 01.6 Concurrency).

    The active-wiki pointer is a mutable global. If an admin switches wikis
    mid-request, a request that started resolving against wiki A could finish
    reading from wiki B — a cross-wiki read that no isolation check would
    catch, because each individual query was correctly scoped to whatever the
    pointer said at the moment it ran.

    Resolving once into `g` and reading it from there for the rest of the
    request makes that impossible: the value cannot change underneath a
    request that already started.
    """
    if not config.USE_DATABASE:
        return None
    try:
        from services import wikis as _wikis
        g.wiki_id = _wikis.active_wiki_id()
    except Exception as err:
        logger.warning("Could not bind active wiki for this request: %s", err)
    return None


def current_wiki_id() -> str:
    """The wiki this request is bound to. Always prefer this over calling
    wikis.active_wiki_id() inside a request — that re-reads the live pointer
    and reintroduces the mid-request switch this binding exists to prevent.

    Ingest runs on background threads with no request context, so this must
    not raise there: `g` is unavailable outside a request, and falling back to
    reading the pointer is correct in that case — a background job has no
    request whose start it could be bound to.
    """
    from services import wikis as _wikis
    if has_request_context():
        bound = getattr(g, "wiki_id", None)
        if bound:
            return bound
    return _wikis.active_wiki_id()


@app.before_request
def _require_login():
    if not config.AUTH_ENABLED:
        return None
    if request.endpoint in _PUBLIC_ENDPOINTS:
        return None
    if session.get("user_id"):
        return None

    if _wants_html():
        return redirect(url_for("login", next=request.full_path.rstrip("?")))
    return jsonify({"error": "Authentication required", "login_required": True}), 401


@app.route("/login", methods=["GET", "POST"])
def login():
    if not config.AUTH_ENABLED:
        return redirect("/")

    # Already signed in — no reason to show the form again.
    if request.method == "GET" and session.get("user_id"):
        return redirect("/")

    next_url = _safe_next(request.args.get("next", "/"))

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        ip = auth.client_ip(request)

        try:
            user, error, retry_after = auth.verify_login(username, password, ip)
        except Exception as e:
            logger.error("Login failed with an unexpected error: %s", e)
            return render_template(
                "login.html", error="Login is unavailable — check the server logs.",
                next_url=next_url, username=username,
            ), 500

        if user is None:
            if retry_after:
                minutes = max(1, round(retry_after / 60))
                error = f"{error} Locked for about {minutes} minute{'s' if minutes != 1 else ''}."
            # 401 for a bad password, 429 when the limiter refused it — the
            # form renders identically, but the status is honest to anything
            # reading it programmatically.
            return render_template(
                "login.html", error=error, next_url=next_url, username=username,
            ), (429 if retry_after else 401)

        # Rotate the session id on privilege change so a session fixation
        # attempt (attacker plants a known cookie pre-login) can't survive
        # into the authenticated session.
        session.clear()
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session.permanent = True
        logger.info("Login succeeded for user=%r from ip=%r", user["username"], ip)
        return redirect(next_url)

    return render_template("login.html", error="", next_url=next_url, username="")


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


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
                            ingest_disabled=config.DISABLE_INGEST,
                            auth_enabled=config.AUTH_ENABLED,
                            username=session.get("username", ""))


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

        _wid = current_wiki_id()
        pages = _db.get_pages(_wid, session_id)
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
                    _db.upsert_embedding(_wid, session_id, title, embedding)
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
    rejected = []
    for i, file in enumerate(files):
        if not _allowed_file(file.filename):
            rejected.append({"filename": file.filename, "reason": "unsupported file type (.txt, .pdf, .docx only)"})
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

        # Validation gate — before parsing starts (target architecture §
        # "File-upload validation"). Rejects malformed/malicious files
        # (decompression bombs, corrupt structures, mismatched content) so
        # they never reach the ingest pipeline. See services/upload_validation.py.
        ok, reason = upload_validation.validate_upload(save_path, file.filename)
        if not ok:
            logger.warning("Rejected upload %r: %s", file.filename, reason)
            rejected.append({"filename": file.filename, "reason": reason})
            try:
                os.remove(save_path)
            except OSError:
                pass
            continue

        # Encrypt at rest — after validation, so a malformed file is rejected
        # while still readable rather than encrypted first and inspected later.
        # No-op unless ENCRYPTION_KEY is set; readers decrypt transparently.
        try:
            from services import crypto as _crypto
            if _crypto.encrypt_file(save_path):
                logger.info("Encrypted upload at rest: %s", os.path.basename(save_path))
        except Exception as _enc_err:
            logger.error("Could not encrypt %s: %s", save_path, _enc_err)

        saved_paths.append(save_path)
        metadata_list.append({
            "relative_path": rel_path if rel_path else file.filename,
            "filename": file.filename
        })
        logger.info("Saved upload: %s with relative path: %s", save_path, rel_path)

    if not saved_paths:
        return jsonify({
            "error": "No valid files (.txt, .pdf, .docx) found",
            "rejected": rejected,
        }), 400

    # Cost pre-flight gate — target architecture § Phase 0-parallel, "Cost
    # pre-flight gate on bulk operations". Zero LLM calls: estimate_ingest_cost
    # only reads the just-saved files locally. A bulk-sized batch (see
    # cost_estimate.needs_confirmation) is held here — saved to disk and
    # registered in sessions.json so it's visible in the UI, but NOT queued to
    # the executor — until the caller re-POSTs with confirm=true. Only
    # meaningful in DB mode, since /resume_ingest (the confirm step below) is.
    estimate = None
    hold_for_confirmation = False
    if config.USE_DATABASE:
        estimate = cost_estimate.estimate_ingest_cost(saved_paths)
        confirmed = request.form.get("confirm", "").lower() == "true"
        hold_for_confirmation = cost_estimate.needs_confirmation(estimate, len(saved_paths)) and not confirmed

    # Initialize progress with per-document tracking
    doc_status = "awaiting_confirmation" if hold_for_confirmation else "queued"
    progress = {
        "phase": "awaiting_confirmation" if hold_for_confirmation else "processing",
        "docs": {"total": len(saved_paths), "wiki_done": 0},
        "wiki": {"step": doc_status, "message": "", "pages_total": 0, "relations_total": 0},
        "docs_list": [
            {"name": os.path.basename(p), "status": doc_status, "pages": 0, "step": ""}
            for p in saved_paths
        ],
    }
    _set_progress(session_id, progress)

    if not hold_for_confirmation:
        # Submit all tasks to executor (non-blocking)
        for save_path, meta in zip(saved_paths, metadata_list):
            executor.submit(_ingest_single_doc_wiki, save_path, session_id)

    # Save session metadata
    sessions = load_sessions()
    # Preserve the wiki a session was originally stamped with — a second
    # upload into the same session_id must not silently re-file it under
    # whatever wiki happens to be active now.
    _existing_wiki_id = sessions.get(session_id, {}).get("wiki_id")
    sessions[session_id] = {
        "id": session_id,
        "name": f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": time.time(),
        "updated_at": time.time(),
        "files": len(saved_paths),
        "file_paths": [meta["relative_path"] for meta in metadata_list],
        "history": [],
        "wiki_id": _existing_wiki_id or current_wiki_id(),
    }
    save_sessions(sessions)

    if hold_for_confirmation:
        return jsonify({
            "status": "confirm_required",
            "estimate": estimate,
            "files_queued": 0,
            "session_id": session_id,
            "rejected": rejected,
        })

    # Return immediately — frontend will poll /progress
    return jsonify({
        "status": "accepted",
        "files_queued": len(saved_paths),
        "session_id": session_id,
        "rejected": rejected,
        "estimate": estimate,
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
                    text("SELECT DISTINCT source_doc FROM pages WHERE wiki_id = :w AND session_id = :sid"),
                    {"w": current_wiki_id(), "sid": session_id},
                )
                indexed = {r.source_doc for r in rows}

            missing = [f for f in uploaded if f not in indexed]
            if not missing:
                return jsonify({
                    "status": "nothing_to_resume",
                    "uploaded": len(uploaded),
                    "already_indexed": len(indexed),
                })

            # Cost pre-flight gate — same rule as /upload (see there for
            # rationale): a bulk-sized "missing" batch is held for explicit
            # confirm=true before it's queued, whether this call originated
            # from /upload's own hold or a direct crash-recovery resume.
            missing_paths = [os.path.join(config.UPLOAD_PATH, f) for f in missing]
            estimate = cost_estimate.estimate_ingest_cost(missing_paths)
            confirmed = request.args.get("confirm", "").lower() == "true"
            if cost_estimate.needs_confirmation(estimate, len(missing)) and not confirmed:
                return jsonify({
                    "status": "confirm_required",
                    "estimate": estimate,
                    "uploaded": len(uploaded),
                    "already_indexed": len(indexed),
                    "pending": len(missing),
                    "session_id": session_id,
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
                "estimate": estimate,
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
        dup_of = (result or {}).get("duplicate_of")
        if dup_of:
            # No pages were written and no LLM call was made — surfaced as
            # its own status rather than "done" so the ingest dashboard
            # doesn't imply this document was actually processed.
            _update_doc_status(session_id, doc_name, "duplicate",
                              f"matches {dup_of}", 0)
            return
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
                # Use the wiki this SESSION was actually stamped with at
                # upload time (see /upload), not whatever wiki happens to be
                # active right now on this background thread — an admin
                # switching wikis mid-ingest must not misfile the pointer
                # under the wiki they switched TO instead of the one this
                # batch was uploaded into.
                _session_wiki = load_sessions().get(session_id, {}).get("wiki_id")
                _set_main_session_id(session_id, wiki_id=_session_wiki)


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


@app.route("/trace/<int:message_id>")
def get_trace_route(message_id):
    """Return the recorded pipeline trace for one assistant chat message —
    stage timings, retrieval detail, LLM calls. See services/tracing.py."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Tracing requires database mode"}), 400
    from services import db as _db
    trace = _db.get_trace_by_message_id(message_id)
    if not trace:
        return jsonify({"error": "No trace found for this message"}), 404
    return jsonify(trace)


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
        result = _db.find_quote_position(current_wiki_id(), session_id, doc_name, quote)
        return jsonify(result)
    return jsonify({"found": False, "page_num": 0, "char_offset": 0})


def _current_user_id():
    """Authenticated user id, or None when auth is off or there's no request.

    Guarded rather than reading session directly: the /query writer runs
    inside a stream_with_context generator (request context alive, fine), but
    a future caller on a plain background thread would otherwise raise instead
    of just recording an unattributed message.
    """
    try:
        if has_request_context():
            return session.get("user_id")
    except Exception:
        pass
    return None


def _store_chat_msg(session_id, role, content, msg_type="text", metadata=None):
    """Insert a chat message if DB is enabled, otherwise no-op."""
    if config.USE_DATABASE:
        from services import db as _db
        try:
            return _db.insert_message(session_id, role, content, msg_type, metadata,
                                      user_id=_current_user_id())
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
            "wiki_id": current_wiki_id(),
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
        last_msg_id = None
        trace, trace_token = tracing.start_trace(question, session_id, wiki_session_id)
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
                    last_msg_id = _store_chat_msg(session_id, "assistant", payload.get("message", ""),
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
                    last_msg_id = _store_chat_msg(session_id, "assistant", payload.get("message", ""),
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
                    wiki_result["answer"] = redaction.redact_pii(wiki_result.get("answer", ""))
                    wiki_result["elapsed_ms"] = round((time.time() - t0) * 1000)
                    last_msg_id = _store_chat_msg(session_id, "assistant", wiki_result.get("answer", ""),
                                    "answer", {
                                        "confidence_score": wiki_result.get("confidence_score", 0),
                                        # Counts, not a self-reported score —
                                        # persisted for the same reason as the
                                        # render flags below: a reloaded thread
                                        # that drops them shows an answer whose
                                        # claims look unchecked.
                                        "citation_check": wiki_result.get("citation_check", {}),
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
                                        "general_knowledge": wiki_result.get("general_knowledge", False),
                                        "not_covered": wiki_result.get("not_covered", False),
                                        "general_knowledge_note": wiki_result.get("general_knowledge_note", ""),
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
                    wiki_result["message_id"] = last_msg_id
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
        finally:
            tracing.finish_and_persist(trace, trace_token, message_id=last_msg_id)

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


def _session_document_paths(session_id: str) -> dict[str, str]:
    """Return {relative_path: source_doc_key} for every uploaded document in
    a session — the reconstruction logic shared by /files (tree view) and
    /document/list (admin lifecycle view), so the two can't drift apart on
    what counts as "this session's documents."

    Docs with zero indexed pages (OCR/ingest failures) never made it into
    the corpus — excluded here so neither view implies they're searchable
    when generate_answer() will never see them.
    """
    from services.documents import flatten_doc_key

    sessions = load_sessions()
    indexed_docs: set[str] = set()
    if config.USE_DATABASE:
        try:
            from services import db as _db
            from sqlalchemy import text
            engine = _db.get_engine()
            with engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT DISTINCT source_doc FROM pages WHERE wiki_id = :w AND session_id = :sid"),
                    {"w": current_wiki_id(), "sid": session_id},
                )
                indexed_docs = {r.source_doc for r in rows}
        except Exception as _idx_err:
            logger.warning("Could not fetch indexed docs for document listing: %s", _idx_err)

    paths: dict[str, str] = {}  # relative_path -> source_doc key

    # 1. Try to read from session metadata first
    if session_id in sessions and "file_paths" in sessions[session_id]:
        for p in sessions[session_id]["file_paths"]:
            flat_key = flatten_doc_key(session_id, p)
            if indexed_docs and flat_key not in indexed_docs:
                continue
            paths[p.replace("\\", "/")] = flat_key
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
                        paths[f"{top}/{type_folder}/{file_part}"] = fname
                        reconstructed = True

                if not reconstructed:
                    paths[rel_name.replace("\\", "/")] = fname

    return paths


@app.route("/files")
def file_structure():
    """Return nested file tree of active (non-archived) uploaded files.

    Archived documents are excluded by default — this is the file-browser /
    Review-Compare-picker view, and an archived document should not be
    selectable there any more than it should surface in chat retrieval (see
    db.get_pages). Use /document/list for the admin view that shows
    everything, archived included.
    """
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({})
    session_id = _get_main_session_id() or session_id

    try:
        paths = _session_document_paths(session_id)

        archived: set[str] = set()
        if config.USE_DATABASE:
            try:
                from services import db as _db
                archived = {
                    doc for doc, info in _db.get_document_statuses(current_wiki_id(), session_id).items()
                    if info["status"] == "archived"
                }
            except Exception as _arch_err:
                logger.warning("Could not fetch archived docs for /files filtering: %s", _arch_err)

        tree = {}
        for p, source_doc in paths.items():
            if source_doc in archived:
                continue
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


@app.route("/document/list")
def document_list():
    """Return a flat list of every document in a session, active and
    archived, for the admin document-lifecycle view (Files panel's manage
    mode) — unlike /files this includes archived docs so they can be
    restored or hard-deleted.
    """
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"documents": []})
    session_id = _get_main_session_id() or session_id

    try:
        paths = _session_document_paths(session_id)
        statuses = {}
        if config.USE_DATABASE:
            from services import db as _db
            statuses = _db.get_document_statuses(current_wiki_id(), session_id)

        documents = [
            {
                "path": p,
                "source_doc": source_doc,
                "status": statuses.get(source_doc, {}).get("status", "active"),
                "archived_at": statuses.get(source_doc, {}).get("archived_at"),
            }
            for p, source_doc in sorted(paths.items())
        ]
        return jsonify({"documents": documents})
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        return jsonify({"documents": []})


@app.route("/document/archive", methods=["POST"])
def document_archive():
    """Archive a document — reversible, hides it from search/chat/pickers
    without deleting anything. See services/documents.py."""
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured — document lifecycle needs Postgres"}), 400

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    source_doc = data.get("source_doc", "")
    if not session_id or not source_doc:
        return jsonify({"error": "session_id and source_doc are required"}), 400
    # Same redirect /files and /document/list apply — a caller listing
    # documents under this session_id and one archiving/deleting from it must
    # resolve to the SAME underlying session, or a source_doc handed back
    # from the list call can point at a different session's real document.
    # services/documents.py._assert_ownership is the hard backstop if this
    # ever drifts; this redirect is what makes the common case just work.
    session_id = _get_main_session_id() or session_id

    from services import documents as _documents
    try:
        result = _documents.archive_document(current_wiki_id(), session_id, source_doc)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Archive failed for %r: %s", source_doc, e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/document/unarchive", methods=["POST"])
def document_unarchive():
    """Restore an archived document to active."""
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured — document lifecycle needs Postgres"}), 400

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    source_doc = data.get("source_doc", "")
    if not session_id or not source_doc:
        return jsonify({"error": "session_id and source_doc are required"}), 400
    session_id = _get_main_session_id() or session_id

    from services import documents as _documents
    try:
        result = _documents.unarchive_document(current_wiki_id(), session_id, source_doc)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Unarchive failed for %r: %s", source_doc, e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/document/delete", methods=["POST"])
def document_delete():
    """Permanently delete a document — separate, deliberate action from
    Archive, requires an explicit confirm flag. See services/documents.py
    and db.delete_document_data for exactly what is and isn't covered
    (shared/merged pages are a disclosed, known gap — not silently claimed
    as fully removed).
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured — document lifecycle needs Postgres"}), 400

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    source_doc = data.get("source_doc", "")
    confirm = data.get("confirm", False)
    if not session_id or not source_doc:
        return jsonify({"error": "session_id and source_doc are required"}), 400
    if not confirm:
        return jsonify({"error": "Hard delete requires confirm: true"}), 400
    session_id = _get_main_session_id() or session_id

    from services import documents as _documents
    try:
        sessions = load_sessions()
        report = _documents.hard_delete_document(current_wiki_id(), session_id, source_doc, sessions)
        save_sessions(sessions)
        return jsonify(report)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Hard delete failed for %r: %s", source_doc, e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


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


@app.route("/admin/wikis")
def admin_wikis_list():
    """List every wiki with its page/document counts and which is active."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import wikis as _wikis
    try:
        return jsonify({"wikis": _wikis.list_wikis(), "active": _wikis.active_wiki_id()})
    except Exception as e:
        logger.error("Wiki list failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/admin/wikis/create", methods=["POST"])
def admin_wikis_create():
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    from services import wikis as _wikis
    try:
        wiki_id = _wikis.create_wiki(data.get("name", ""),
                                     created_by=session.get("user_id"))
        return jsonify({"status": "created", "wiki_id": wiki_id})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Wiki create failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/admin/wikis/activate", methods=["POST"])
def admin_wikis_activate():
    """Switch the system-level active wiki.

    Every read and write resolves against whichever wiki is active, so this
    is the one control that changes what the whole application is looking at.
    Requests already in flight are unaffected — they bound their wiki_id at
    entry (see _bind_wiki_id).
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    wiki_id = (data.get("wiki_id") or "").strip()
    if not wiki_id:
        return jsonify({"error": "wiki_id is required"}), 400
    from services import wikis as _wikis
    try:
        _wikis.set_active_wiki(wiki_id)
        return jsonify({"status": "activated", "wiki_id": wiki_id})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Wiki activate failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/admin/wikis/archive", methods=["POST"])
def admin_wikis_archive():
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    if not data.get("confirm"):
        return jsonify({"error": "confirm:true is required"}), 400
    from services import wikis as _wikis
    try:
        _wikis.archive_wiki((data.get("wiki_id") or "").strip())
        return jsonify({"status": "archived"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Wiki archive failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# ---------------------------------------------------------------------------
# Prompt library (Phase 3) — reusable, wiki-scoped {{placeholder}} templates
# ---------------------------------------------------------------------------

@app.route("/admin/prompts", methods=["GET", "POST"])
def admin_prompts():
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import prompt_library as _lib
    wiki_id = current_wiki_id()
    if request.method == "GET":
        return jsonify({"templates": _lib.list_all(wiki_id, request.args.get("category")),
                        "categories": _lib.categories(wiki_id)})
    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.get_json(silent=True) or {}
    session_id = _get_main_session_id() or data.get("session_id", "")
    try:
        return jsonify(_lib.create(wiki_id, session_id, data.get("name", ""),
                                   data.get("body", ""), data.get("category"))), 201
    except _lib.PromptLibraryError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin/prompts/<int:template_id>", methods=["GET", "PATCH", "DELETE"])
def admin_prompt(template_id: int):
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import prompt_library as _lib
    wiki_id = current_wiki_id()
    if request.method == "GET":
        found = _lib.get(wiki_id, template_id)
        return jsonify(found) if found else (jsonify({"error": "Not found"}), 404)
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if request.method == "DELETE":
        return (jsonify({"status": "deleted"}) if _lib.delete(wiki_id, template_id)
                else (jsonify({"error": "Not found"}), 404))
    data = request.get_json(silent=True) or {}
    try:
        changed = _lib.update(wiki_id, template_id, data.get("name"),
                              data.get("body"), data.get("category"))
    except _lib.PromptLibraryError as e:
        return jsonify({"error": str(e)}), 400
    if not changed:
        return jsonify({"error": "Nothing to update, or template not found"}), 400
    return jsonify(_lib.get(wiki_id, template_id))


@app.route("/admin/prompts/<int:template_id>/render", methods=["POST"])
def admin_prompt_render(template_id: int):
    """Fill a template's {{placeholders}}. Zero LLM calls — string
    substitution only. A variable with no supplied value is left literal in
    the output and listed in `missing`, so a half-filled template is still
    usable rather than silently blank."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import prompt_library as _lib
    found = _lib.get(current_wiki_id(), template_id)
    if not found:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(silent=True) or {}
    return jsonify(_lib.render(found["body"], data.get("values") or {}))


# ---------------------------------------------------------------------------
# Deviation Dashboard (Phase 3) — a SQL aggregation over playbook_findings
# ---------------------------------------------------------------------------

@app.route("/admin/playbooks/dashboard")
def admin_deviation_overview():
    """One line per playbook with a completed run: its latest run and that
    run's verdict counts, worst-first."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import deviation as _dev
    return jsonify({"playbooks": _dev.overview(current_wiki_id())})


@app.route("/admin/playbooks/<int:playbook_id>/deviation")
def admin_deviation_detail(playbook_id: int):
    """Full breakdown for one playbook: by clause type, by document
    (worst-first), a priority list, and documents added since the run that
    have not been assessed yet. Defaults to the latest complete run; pass
    ?run_id= to inspect a past one."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import deviation as _dev
    run_id = request.args.get("run_id")
    found = _dev.dashboard(current_wiki_id(), playbook_id,
                           int(run_id) if run_id else None)
    return jsonify(found) if found else (
        jsonify({"error": "No completed run for this playbook"}), 404)


# ---------------------------------------------------------------------------
# Precedent layer (Phase 2) — document roles + clause embeddings
# ---------------------------------------------------------------------------

@app.route("/admin/precedent/roles", methods=["GET", "POST"])
def admin_precedent_roles():
    """Show the role split, derive it from families, or tag one document.

    POST {"derive": true}                     fill untagged documents
    POST {"derive": true, "overwrite": true}  re-derive, discarding manual tags
    POST {"source_doc": "...", "role": "..."} tag one document explicitly
    """
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import precedent as _prec
    wiki_id = current_wiki_id()
    if request.method == "GET":
        return jsonify({"roles": _prec.role_summary(wiki_id),
                        "valid_roles": list(_prec.ROLES)})
    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.get_json(silent=True) or {}
    if data.get("source_doc"):
        try:
            done = _prec.set_role(wiki_id, data["source_doc"], data.get("role", ""))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return (jsonify({"status": "tagged", "roles": _prec.role_summary(wiki_id)})
                if done else (jsonify({"error": "Document not found"}), 404))
    if data.get("derive"):
        res = _prec.derive_roles(wiki_id, overwrite=bool(data.get("overwrite")))
        res["roles"] = _prec.role_summary(wiki_id)
        return jsonify(res)
    return jsonify({"error": "Provide derive:true, or source_doc + role"}), 400


@app.route("/admin/precedent/embeddings", methods=["GET", "POST"])
def admin_precedent_embeddings():
    """Coverage, or queue embedding of clauses that have no vector yet.

    Embeddings only — no chat model is involved, and only precedent-role
    clauses are eligible, so reference material never enters the drafting pool.
    """
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import precedent as _prec
    wiki_id = current_wiki_id()
    session_id = _get_main_session_id() or request.args.get("session_id", "")
    if request.method == "GET":
        return jsonify(_prec.coverage(wiki_id, session_id))
    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.get_json(silent=True) or {}
    session_id = _get_main_session_id() or data.get("session_id", session_id)
    cov = _prec.coverage(wiki_id, session_id)
    if not cov["pending"]:
        return jsonify({"status": "up_to_date", **cov})
    executor.submit(_prec.embed_pending, wiki_id, session_id, 128,
                    data.get("max_clauses"))
    return jsonify({"status": "queued", **cov})


@app.route("/admin/precedent/search")
def admin_precedent_search():
    """Rank precedent clauses against a drafting request — what Draft Mode reads."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import precedent as _prec
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Provide q"}), 400
    session_id = _get_main_session_id() or request.args.get("session_id", "")
    return jsonify({"query": q, "results": _prec.search_clauses(
        current_wiki_id(), session_id, q,
        limit=int(request.args.get("limit", 12)),
        clause_type=request.args.get("clause_type"))})


# ---------------------------------------------------------------------------
# Playbooks (Phase 2) — house positions per clause type, run over a Collection
# ---------------------------------------------------------------------------

@app.route("/admin/playbooks", methods=["GET", "POST"])
def admin_playbooks():
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import playbooks as _pb
    wiki_id = current_wiki_id()
    if request.method == "GET":
        return jsonify({"playbooks": _pb.list_all(wiki_id)})
    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.get_json(silent=True) or {}
    session_id = _get_main_session_id() or data.get("session_id", "")
    try:
        return jsonify(_pb.create(wiki_id, session_id, data.get("name", ""),
                                  data.get("description"))), 201
    except _pb.PlaybookError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin/playbooks/<int:playbook_id>", methods=["GET", "DELETE"])
def admin_playbook(playbook_id: int):
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import playbooks as _pb
    wiki_id = current_wiki_id()
    if request.method == "GET":
        found = _pb.get(wiki_id, playbook_id)
        return jsonify(found) if found else (jsonify({"error": "Not found"}), 404)
    _lock = _locked_in_production()
    if _lock:
        return _lock
    try:
        deleted = _pb.delete(wiki_id, playbook_id)
    except _pb.PlaybookError as e:
        # A run is still in flight; deleting now would cascade the run row out
        # from under the worker writing findings against it.
        return jsonify({"error": str(e)}), 409
    return (jsonify({"status": "deleted"}) if deleted
            else (jsonify({"error": "Not found"}), 404))


@app.route("/admin/playbooks/<int:playbook_id>/rules", methods=["POST", "DELETE"])
def admin_playbook_rules(playbook_id: int):
    """Add/replace a rule, or remove one. One rule per clause type."""
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import playbooks as _pb
    wiki_id = current_wiki_id()
    if not _pb.get(wiki_id, playbook_id):
        return jsonify({"error": "Playbook not found"}), 404
    data = request.get_json(silent=True) or {}
    if request.method == "DELETE":
        return jsonify({"removed": _pb.remove_rule(
            wiki_id, playbook_id, data.get("clause_type", ""))})
    try:
        return jsonify(_pb.add_rule(
            wiki_id, playbook_id, data.get("clause_type", ""),
            data.get("standard", ""), data.get("fallback"),
            data.get("unacceptable"), data.get("guidance"),
            data.get("severity", "medium"), int(data.get("ordinal", 0))))
    except _pb.PlaybookError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin/playbooks/<int:playbook_id>/run", methods=["POST"])
def admin_playbook_run(playbook_id: int):
    """Run a playbook over a collection or an explicit document list.

    Cost-gated and queued like every other bulk operation: one LLM call per
    matched clause, so a large collection is real spend and is held for
    confirm=true before anything is queued.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import playbooks as _pb, collections as _col
    wiki_id = current_wiki_id()
    data = request.get_json(silent=True) or {}
    session_id = _get_main_session_id() or data.get("session_id", "")

    book = _pb.get(wiki_id, playbook_id)
    if not book:
        return jsonify({"error": "Playbook not found"}), 404
    if not book["rules"]:
        return jsonify({"error": "Playbook has no rules to run"}), 400

    collection_id = collection_name = None
    if data.get("collection"):
        collection_id = _col.resolve(wiki_id, data["collection"])
        if not collection_id:
            return jsonify({"error": "Collection not found"}), 404
        meta = _col.get(wiki_id, collection_id, with_documents=False)
        collection_name = meta["name"] if meta else None
        documents = _col.documents_in(wiki_id, collection_id)
    else:
        documents = data.get("source_docs") or []
    if not documents:
        return jsonify({"error": "Nothing to run over — give a collection or source_docs"}), 400

    # One classify call per matched clause. Counting them first is a DB read,
    # so the estimate costs nothing and is exact rather than extrapolated.
    clause_calls = 0
    for d in documents:
        for rule in book["rules"]:
            clause_calls += len(_pb.clauses_for_rule(wiki_id, d, rule["clause_type"]))
    if clause_calls > 50 and str(data.get("confirm", "")).lower() != "true":
        return jsonify({
            "status": "confirm_required",
            "documents": len(documents), "rules": len(book["rules"]),
            "estimated_llm_calls": clause_calls,
        })

    def _work():
        try:
            _pb.run(wiki_id, session_id, playbook_id, documents,
                    collection_id, collection_name)
        except Exception as e:
            logger.error("Playbook run failed: %s", e)

    executor.submit(_work)
    return jsonify({"status": "queued", "documents": len(documents),
                    "rules": len(book["rules"]),
                    "estimated_llm_calls": clause_calls})


@app.route("/admin/playbooks/runs")
def admin_playbook_runs():
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import playbooks as _pb
    pid = request.args.get("playbook_id")
    return jsonify({"runs": _pb.list_runs(current_wiki_id(),
                                          int(pid) if pid else None)})


@app.route("/admin/playbooks/runs/<int:run_id>")
def admin_playbook_run_detail(run_id: int):
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import playbooks as _pb
    found = _pb.get_run(current_wiki_id(), run_id,
                        with_findings=request.args.get("findings", "1") != "0")
    return jsonify(found) if found else (jsonify({"error": "Run not found"}), 404)


# ---------------------------------------------------------------------------
# Collections (Phase 2) — named, wiki-scoped document sets
# ---------------------------------------------------------------------------

@app.route("/admin/collections", methods=["GET", "POST"])
def admin_collections():
    """List collections in the active wiki, or create one."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import collections as _col
    wiki_id = current_wiki_id()
    if request.method == "GET":
        return jsonify({"collections": _col.list_all(wiki_id)})

    _lock = _locked_in_production()
    if _lock:
        return _lock
    data = request.get_json(silent=True) or {}
    session_id = _get_main_session_id() or data.get("session_id", "")
    try:
        created = _col.create(wiki_id, session_id, data.get("name", ""),
                              data.get("description"))
    except _col.CollectionError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(created), 201


@app.route("/admin/collections/<int:collection_id>",
           methods=["GET", "PATCH", "DELETE"])
def admin_collection(collection_id: int):
    """Read, rename, or delete one collection.

    DELETE removes the collection and its membership rows only — a collection
    is a label over the corpus, never a container of it, so the documents stay.
    """
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import collections as _col
    wiki_id = current_wiki_id()

    if request.method == "GET":
        found = _col.get(wiki_id, collection_id)
        if not found:
            return jsonify({"error": "Collection not found"}), 404
        return jsonify(found)

    _lock = _locked_in_production()
    if _lock:
        return _lock

    if request.method == "DELETE":
        return (jsonify({"status": "deleted"}) if _col.delete(wiki_id, collection_id)
                else (jsonify({"error": "Collection not found"}), 404))

    data = request.get_json(silent=True) or {}
    try:
        changed = _col.rename(wiki_id, collection_id, data.get("name"),
                              data.get("description"))
    except _col.CollectionError as e:
        return jsonify({"error": str(e)}), 400
    if not changed:
        return jsonify({"error": "Nothing to update, or collection not found"}), 400
    return jsonify(_col.get(wiki_id, collection_id, with_documents=False))


@app.route("/admin/collections/<int:collection_id>/documents",
           methods=["POST", "DELETE"])
def admin_collection_documents(collection_id: int):
    """Add or remove member documents.

    POST accepts either an explicit {"source_docs": [...]} or a filter
    ({"doc_family"|"doc_type"|"name_contains"}). A filter is evaluated ONCE
    here and its result stored as a fixed list: a collection that re-evaluated
    on every read could change under a playbook run, leaving the run record
    describing documents it never processed.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import collections as _col
    wiki_id = current_wiki_id()
    if not _col.get(wiki_id, collection_id, with_documents=False):
        return jsonify({"error": "Collection not found"}), 404

    data = request.get_json(silent=True) or {}
    session_id = _get_main_session_id() or data.get("session_id", "")
    docs = data.get("source_docs") or []

    if request.method == "DELETE":
        return jsonify({"removed": _col.remove_documents(wiki_id, collection_id, docs)})

    if docs:
        result = _col.add_documents(wiki_id, session_id, collection_id, docs)
    elif any(data.get(k) for k in ("doc_family", "doc_type", "name_contains")):
        result = _col.add_by_filter(wiki_id, session_id, collection_id,
                                    data.get("doc_family"), data.get("doc_type"),
                                    data.get("name_contains"))
    else:
        return jsonify({"error": "Provide source_docs, or a doc_family / "
                                 "doc_type / name_contains filter"}), 400
    return jsonify(result)


@app.route("/admin/documents/reingest", methods=["POST"])
def admin_reingest():
    """Re-ingest documents: one, an explicit set, or a whole family/version band.

    Selection (first match wins):
        {"source_doc": "..."}                     one document
        {"source_docs": ["...", "..."]}           an explicit set (a Collection)
        {"family": "contract",                    everything in a family, optionally
         "max_schema_version": 1}                 only what predates a schema change

    `max_schema_version` is the point of stamping it: after changing what
    extraction asks for, the documents that need redoing are exactly the ones
    still carrying the old version, and re-running the rest costs money to
    produce identical rows.

    Swap, not blend. wiki.ingest() merges into existing pages by design — that
    is right for a NEW document contributing to shared pages, and wrong for a
    re-ingest, where the previous extraction of this same document is still
    sitting there and would be appended to rather than replaced. So each
    document's own page data is deleted first, inside the worker, immediately
    before it is re-read. The typed tables already swap themselves
    (backbone.replace_document_rows) and the Review Queue already supersedes
    rather than deletes (db.supersede_review_items), so prior reviewer
    judgements survive as history.

    Cost-gated like /upload: a bulk-sized batch is held until the caller
    re-POSTs with confirm=true, and anything past a handful is queued to the
    executor rather than run inside the request.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400

    data = request.get_json(silent=True) or {}
    wiki_id = current_wiki_id()
    session_id = _get_main_session_id() or data.get("session_id", "")
    if not session_id:
        return jsonify({"error": "No session to re-ingest"}), 400

    from sqlalchemy import text as _sql
    from services import db as _db

    targets: list[str] = []
    if data.get("source_doc"):
        targets = [data["source_doc"]]
    elif data.get("source_docs"):
        targets = [d for d in data["source_docs"] if d]
    elif data.get("family"):
        params = {"w": wiki_id, "s": session_id, "f": data["family"]}
        clause = ""
        if data.get("max_schema_version") is not None:
            clause = "AND schema_version <= :v"
            params["v"] = int(data["max_schema_version"])
        with _db.get_engine().connect() as c:
            targets = [r[0] for r in c.execute(_sql(f"""
                SELECT source_doc FROM documents
                WHERE wiki_id = :w AND session_id = :s AND doc_family = :f {clause}
                ORDER BY source_doc
            """), params)]
    else:
        return jsonify({"error": "Specify source_doc, source_docs, or family"}), 400

    if not targets:
        return jsonify({"error": "No documents matched", "queued": 0}), 400

    # Only documents whose file is still on disk can be re-read at all.
    runnable, missing = [], []
    for d in targets:
        (runnable if os.path.exists(os.path.join(config.UPLOAD_PATH, d))
         else missing).append(d)
    if not runnable:
        return jsonify({"error": "No source files on disk for the matched documents",
                        "no_file_on_disk": len(missing),
                        "no_file_examples": missing[:10]}), 400

    estimate = cost_estimate.estimate_ingest_cost(
        [os.path.join(config.UPLOAD_PATH, d) for d in runnable])
    if (cost_estimate.needs_confirmation(estimate, len(runnable))
            and str(data.get("confirm", "")).lower() != "true"):
        return jsonify({
            "status": "confirm_required",
            "estimate": estimate,
            "matched": len(targets),
            "runnable": len(runnable),
            "no_file_on_disk": len(missing),
        })

    _set_progress(session_id, {
        "phase": "processing",
        "docs": {"total": len(runnable), "wiki_done": 0},
        "wiki": {"step": "re-ingesting", "message": "", "pages_total": 0,
                 "relations_total": 0},
        "docs_list": [{"name": d, "status": "queued", "pages": 0, "step": ""}
                      for d in runnable],
    })
    for d in runnable:
        executor.submit(_reingest_one, wiki_id, session_id, d)

    return jsonify({
        "status": "queued",
        "matched": len(targets),
        "queued": len(runnable),
        "no_file_on_disk": len(missing),
        "no_file_examples": missing[:10],
        "estimate": estimate,
    })


def _reingest_one(wiki_id: str, session_id: str, source_doc: str):
    """Delete this document's EXCLUSIVE page data, then ingest it again.

    Runs in the worker rather than up-front, so a queue of 400 documents does
    not strip the wiki bare before the first one has been re-read.

    Uses delete_document_pages_exclusive, not delete_document_data: the latter
    clears every page whose source_doc names this document, and that column
    records only the LAST writer, so merged concept pages built mostly by other
    documents are caught by it. Re-ingesting one document cannot rebuild those
    — it only re-contributes its own share — so a whole-document delete here
    destroys other documents' work. Pages with more than one contributor are
    left for wiki.ingest to merge into instead.
    """
    path = os.path.join(config.UPLOAD_PATH, source_doc)
    try:
        from services import db as _db
        report = _db.delete_document_pages_exclusive(wiki_id, session_id, source_doc)
        logger.info("Re-ingest: cleared %s (%s)", source_doc, report)
    except Exception as e:
        # Do not ingest on top of data we failed to clear — that is the blend
        # this route exists to avoid.
        logger.error("Re-ingest: clearing %s failed, skipping: %s", source_doc, e)
        _update_doc_status(session_id, source_doc, "error", f"clear failed: {e}"[:60])
        with _get_progress_lock(session_id):
            p = _get_progress(session_id)
            p.setdefault("docs", {})["wiki_done"] = p.get("docs", {}).get("wiki_done", 0) + 1
            _set_progress(session_id, p)
        return
    _ingest_single_doc_wiki(path, session_id)


@app.route("/admin/documents/reingest/status")
def admin_reingest_status():
    """Schema-version spread per family — what a version-scoped re-ingest would hit."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from sqlalchemy import text as _sql
    from services import db as _db
    wiki_id = current_wiki_id()
    session_id = _get_main_session_id() or request.args.get("session_id", "")
    with _db.get_engine().connect() as c:
        rows = c.execute(_sql("""
            SELECT doc_family, schema_version, count(*)
            FROM documents WHERE wiki_id = :w AND session_id = :s
            GROUP BY 1, 2 ORDER BY 1, 2
        """), {"w": wiki_id, "s": session_id}).fetchall()
    families: dict = {}
    for fam, ver, n in rows:
        families.setdefault(fam or "unknown", {})[str(ver)] = int(n)
    return jsonify({"session_id": session_id, "by_family_and_version": families})


@app.route("/admin/documents/backfill_file_hashes", methods=["POST"])
def admin_backfill_file_hashes():
    """Compute file_hash (raw bytes, no extraction) for every document that
    predates upload-time duplicate detection.

    This is the cheap, zero-LLM backfill — it only ever opens the file and
    hashes its bytes, so it never touches the reader or the OCR fallback.
    Run this before a big re-upload; it's what makes wiki.ingest()'s file_hash
    check actually recognize the overlap instead of every existing row coming
    up NULL. Idempotent and safe to call repeatedly, same as the content_hash
    backfill below.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import backbone as _backbone
    wiki_id = current_wiki_id()
    pending = _backbone.documents_missing_file_hash(wiki_id)
    queued, no_file = [], []
    for row in pending:
        path = os.path.join(config.UPLOAD_PATH, row["source_doc"])
        if os.path.exists(path):
            queued.append(row["source_doc"])
            executor.submit(_backfill_one_file_hash, wiki_id, row["session_id"],
                           row["source_doc"], path)
        else:
            no_file.append(row["source_doc"])
    return jsonify({
        "status": "queued",
        "total_missing": len(pending),
        "queued": len(queued),
        "no_file_on_disk": len(no_file),
        "no_file_examples": no_file[:10],
    })


def _backfill_one_file_hash(wiki_id: str, session_id: str, source_doc: str, path: str):
    try:
        from services import backbone as _backbone
        h = wiki._file_hash(path)
        if h:
            _backbone.backfill_file_hash(wiki_id, session_id, source_doc, h)
    except Exception as e:
        logger.warning("File-hash backfill failed for %s: %s", source_doc, e)


@app.route("/admin/documents/backfill_file_hashes/status")
def admin_backfill_file_hashes_status():
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import backbone as _backbone
    pending = _backbone.documents_missing_file_hash(current_wiki_id())
    return jsonify({"still_missing": len(pending)})


@app.route("/admin/documents/backfill_hashes", methods=["POST"])
def admin_backfill_content_hashes():
    """Compute content_hash (extracted text) for every document that predates
    duplicate detection.

    WARNING — this is NOT free in this deployment: it calls the reader on
    every document to extract text, and any scanned/low-text page routes
    through Azure vision OCR (a real, billed LLM call per page), since
    config.OCR_ENGINE=azure_vision here. Prefer /admin/documents/backfill_file_hashes
    first — it catches exact re-uploads at zero cost with no extraction at
    all. Only run this one with explicit go-ahead, for the secondary
    same-text-different-bytes case.

    Queued per-document on the same executor ingest uses: a few hundred
    documents needing OCR can take a long time, and this must not block the
    request or hold a gunicorn worker hostage. Idempotent by construction —
    the UPDATE only ever touches a row that still has no hash — so it is
    always safe to call again: after an interrupted run, after new files
    appear on disk for previously file-less rows, on a schedule, whatever.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import backbone as _backbone
    wiki_id = current_wiki_id()
    pending = _backbone.documents_missing_hash(wiki_id)
    queued, no_file = [], []
    for row in pending:
        path = os.path.join(config.UPLOAD_PATH, row["source_doc"])
        if os.path.exists(path):
            queued.append(row["source_doc"])
            executor.submit(_backfill_one_hash, wiki_id, row["session_id"],
                           row["source_doc"], path)
        else:
            no_file.append(row["source_doc"])
    return jsonify({
        "status": "queued",
        "total_missing": len(pending),
        "queued": len(queued),
        # These can never be hash-backfilled — the file that would need
        # re-reading is gone. Named explicitly rather than silently dropped,
        # since it means duplicate detection has a permanent blind spot for
        # exactly these documents until they're re-uploaded.
        "no_file_on_disk": len(no_file),
        "no_file_examples": no_file[:10],
    })


def _backfill_one_hash(wiki_id: str, session_id: str, source_doc: str, path: str):
    try:
        from services.reader import read_file_with_positions as _read
        from services import backbone as _backbone
        text = _read(path)["text"]
        h = wiki._content_hash(text)
        if h:
            _backbone.backfill_content_hash(wiki_id, session_id, source_doc, h)
    except Exception as e:
        logger.warning("Hash backfill failed for %s: %s", source_doc, e)


@app.route("/admin/documents/backfill_hashes/status")
def admin_backfill_hashes_status():
    """How much of the backfill above is left — poll this after queuing it."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import backbone as _backbone
    pending = _backbone.documents_missing_hash(current_wiki_id())
    return jsonify({"still_missing": len(pending)})


@app.route("/admin/documents")
def admin_documents_registry():
    """The typed `documents` registry for the active wiki — family, doc type,
    jurisdiction, classification confidence and typed-row counts per document.
    This is what the backbone actually knows about each document, as distinct
    from /document/list which is the file-level view."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session_id = _get_main_session_id() or session_id
    from services import db as _db
    from sqlalchemy import text as _sql
    wiki_id = current_wiki_id()
    try:
        with _db.get_engine().connect() as conn:
            rows = conn.execute(_sql("""
                SELECT source_doc, doc_family, doc_type, jurisdiction, lifecycle,
                       family_confidence, family_method, schema_version, created_at
                FROM documents
                WHERE wiki_id = :w AND session_id = :s
                ORDER BY created_at DESC
            """), {"w": wiki_id, "s": session_id}).fetchall()
        return jsonify({"documents": [
            {"source_doc": r[0], "doc_family": r[1], "doc_type": r[2],
             "jurisdiction": r[3], "lifecycle": r[4], "family_confidence": r[5],
             "family_method": r[6], "schema_version": r[7],
             "created_at": r[8].isoformat() if r[8] else None}
            for r in rows
        ], "wiki_id": wiki_id})
    except Exception as e:
        logger.error("Document registry read failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/admin/wiki/pages")
def admin_wiki_pages():
    """Admin page browser listing — title, source_doc, char_count,
    contradiction_flagged, last_modified. Richer than /wiki/pages (bare
    titles, used by the D3 graph), so kept as its own route rather than
    changing that one's response shape under existing callers."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session_id = _get_main_session_id() or session_id
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import db as _db
    return jsonify({"pages": _db.get_page_list(current_wiki_id(), session_id)})


@app.route("/admin/wiki/page/rename", methods=["POST"])
def admin_wiki_page_rename():
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    old_title = data.get("old_title", "")
    new_title = data.get("new_title", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session_id = _get_main_session_id() or session_id
    try:
        return jsonify(wiki_pages.rename_page(current_wiki_id(), session_id, old_title, new_title))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Page rename failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/admin/wiki/page/merge", methods=["POST"])
def admin_wiki_page_merge():
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    source_title = data.get("source_title", "")
    target_title = data.get("target_title", "")
    confirm = data.get("confirm", False)
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not confirm:
        return jsonify({"error": "Merge requires confirm: true"}), 400
    session_id = _get_main_session_id() or session_id
    try:
        return jsonify(wiki_pages.merge_pages(current_wiki_id(), session_id, source_title, target_title))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Page merge failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/admin/wiki/page/delete", methods=["POST"])
def admin_wiki_page_delete():
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    title = data.get("title", "")
    confirm = data.get("confirm", False)
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not confirm:
        return jsonify({"error": "Delete requires confirm: true"}), 400
    session_id = _get_main_session_id() or session_id
    try:
        return jsonify(wiki_pages.delete_page(current_wiki_id(), session_id, title))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Page delete failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/review_queue")
def review_queue_list():
    """Pending clauses for the Review Queue — target architecture § 02,
    first slice. See services/db.py's get_review_queue for sort order."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session_id = _get_main_session_id() or session_id
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    from services import db as _db
    return jsonify({"clauses": _db.get_review_queue(current_wiki_id(), session_id)})


@app.route("/review_queue/documents")
def review_queue_documents():
    """Review Queue grouped by document — classification plus every extracted
    field with its own confidence. See db.get_review_documents."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session_id = _get_main_session_id() or session_id
    from services import db as _db
    try:
        docs = _db.get_review_documents(current_wiki_id(), session_id)
        return jsonify({"documents": docs})
    except Exception as e:
        logger.error("Review document listing failed: %s", e, exc_info=True)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/review_queue/document")
def review_queue_document_detail():
    """One document's flagged items plus its extracted text, for the
    side-by-side verification pane. The text comes from the stored page map
    rather than re-reading the file, so opening a document in review costs no
    parsing and no OCR."""
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    session_id = request.args.get("session_id", "")
    source_doc = request.args.get("source_doc", "")
    if not session_id or not source_doc:
        return jsonify({"error": "session_id and source_doc are required"}), 400
    session_id = _get_main_session_id() or session_id
    from services import db as _db
    from sqlalchemy import text as _sql
    try:
        _wid = current_wiki_id()
        items = _db.get_document_review_items(_wid, session_id, source_doc)
        pages = []
        with _db.get_engine().connect() as conn:
            rows = conn.execute(_sql("""
                SELECT title, summary, content FROM pages
                WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
                ORDER BY title
            """), {"w": _wid, "s": session_id, "d": source_doc}).fetchall()
            for r in rows:
                pages.append({"title": r[0], "summary": r[1],
                              "content": (r[2] or "")[:6000]})
            anchors = [
                {"label": a[0], "kind": a[1], "heading": a[2]}
                for a in conn.execute(_sql("""
                    SELECT anchor_label, anchor_kind, heading_text
                    FROM structural_anchors
                    WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
                    ORDER BY ordinal LIMIT 200
                """), {"w": _wid, "s": session_id, "d": source_doc}).fetchall()
            ]
        return jsonify({"items": items, "pages": pages, "anchors": anchors})
    except Exception as e:
        logger.error("Review document detail failed: %s", e, exc_info=True)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/review_queue/field", methods=["POST"])
def review_queue_edit_field():
    """Correct one extracted metadata field.

    The extraction is preserved as previous_value and the field is stamped as
    human-edited — a corrected value and a model-extracted one that happen to
    match are not the same fact.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    source_doc = data.get("source_doc", "")
    field_name = (data.get("field") or "").strip()
    family = (data.get("family") or "").strip() or None
    if not session_id or not source_doc or not field_name:
        return jsonify({"error": "session_id, source_doc and field are required"}), 400
    session_id = _get_main_session_id() or session_id

    value = data.get("value")
    if isinstance(value, str) and not value.strip():
        value = None  # cleared field means "not stated", not an empty string

    from services import backbone as _backbone
    try:
        result = _backbone.update_metadata_field(
            current_wiki_id(), session_id, source_doc, field_name, value, family)
        return jsonify({"status": "updated", **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Metadata field edit failed: %s", e, exc_info=True)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/review_queue/resolve_document", methods=["POST"])
def review_queue_resolve_document():
    """Approve or reject every pending item for one document.

    High-stakes clauses below the threshold are still excluded from an
    approve, server-side — reviewing document-by-document is a workflow
    convenience, not a way around the stakes rule. The response reports what
    stayed pending so the caller can tell a finished document from a
    partly-finished one.
    """
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    source_doc = data.get("source_doc", "")
    action = data.get("action", "")
    min_confidence = float(data.get("min_confidence", 0.0) or 0.0)
    if not session_id or not source_doc:
        return jsonify({"error": "session_id and source_doc are required"}), 400
    session_id = _get_main_session_id() or session_id
    from services import db as _db
    try:
        result = _db.resolve_document(current_wiki_id(), session_id, source_doc, action, min_confidence)
        return jsonify({"status": "resolved", **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Review document resolve failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/review_queue/resolve", methods=["POST"])
def review_queue_resolve():
    """Resolve one clause: accept, reject, or edit. See
    services/db.py's resolve_clause for the provenance-marking rule."""
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    clause_id = data.get("clause_id")
    action = data.get("action", "")
    edited_text = data.get("edited_text")
    # Clauses live in `clauses`, every other flagged kind in `review_queue`.
    # The client echoes back the item_kind it was given rather than the server
    # guessing — an id collision across the two tables is otherwise silently
    # resolvable against the wrong row.
    item_kind = (data.get("item_kind") or "clause").strip()
    if not session_id or clause_id is None:
        return jsonify({"error": "session_id and clause_id are required"}), 400
    session_id = _get_main_session_id() or session_id
    from services import db as _db
    try:
        _wid = current_wiki_id()
        if item_kind == "clause":
            ok = _db.resolve_clause(_wid, session_id, int(clause_id), action, edited_text)
        else:
            ok = _db.resolve_review_item(_wid, session_id, int(clause_id), action, edited_text)
        if not ok:
            return jsonify({"error": "Item not found, or already resolved"}), 404
        return jsonify({"status": "resolved", "clause_id": clause_id,
                        "item_kind": item_kind, "action": action})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Review Queue resolve failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/review_queue/bulk_accept", methods=["POST"])
def review_queue_bulk_accept():
    """Accept every pending LOW-stakes clause at or above min_confidence in
    one call — high-stakes clauses are excluded server-side regardless of
    what's sent, see services/db.py's bulk_accept_clauses."""
    _lock = _locked_in_production()
    if _lock:
        return _lock
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    min_confidence = data.get("min_confidence", 0.6)
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session_id = _get_main_session_id() or session_id
    from services import db as _db
    try:
        _wid = current_wiki_id()
        n = _db.bulk_accept_clauses(_wid, session_id, float(min_confidence))
        n += _db.bulk_accept_review_items(_wid, session_id, float(min_confidence))
        return jsonify({"status": "accepted", "accepted": n})
    except Exception as e:
        logger.error("Review Queue bulk accept failed: %s", e)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/register")
def contract_register():
    """The Contract Register — target architecture § 06, Phase 1.

    A pure read: every ingested document is already a row, and this returns
    it with whatever standard fields its own family defines. No LLM call, no
    background job, no cost.
    """
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session_id = _get_main_session_id() or session_id
    from services import register as _register
    try:
        return jsonify(_register.register_rows(
            current_wiki_id(), session_id,
            family=(request.args.get("family") or "").strip() or None,
            search=request.args.get("search"),
            limit=min(int(request.args.get("limit", 500)), 2000),
            offset=max(int(request.args.get("offset", 0)), 0),
        ))
    except Exception as e:
        logger.error("Contract Register read failed: %s", e, exc_info=True)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/obligations")
def obligation_tracker():
    """The Obligation tracker — target architecture § 06, Phase 1.

    Also a pure read. The response carries a coverage block alongside the
    rows because an empty `obligations` table means the corpus predates
    obligation extraction, not that the documents impose no duties, and the
    UI has to be able to tell those apart.
    """
    if not config.USE_DATABASE:
        return jsonify({"error": "Database not configured"}), 400
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    session_id = _get_main_session_id() or session_id
    from services import register as _register
    try:
        return jsonify(_register.obligation_rows(
            current_wiki_id(), session_id,
            party=(request.args.get("party") or "").strip() or None,
            source_doc=(request.args.get("source_doc") or "").strip() or None,
            search=request.args.get("search"),
            with_deadline=request.args.get("with_deadline") == "true",
            limit=min(int(request.args.get("limit", 1000)), 5000),
            offset=max(int(request.args.get("offset", 0)), 0),
        ))
    except Exception as e:
        logger.error("Obligation tracker read failed: %s", e, exc_info=True)
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


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
    if ext == ".pdf":
        mime = "application/pdf"
    elif ext == ".docx":
        # Browsers can't render this inline — it downloads/prompts rather than
        # displaying, unlike the text/plain garbling a wrong mimetype would
        # cause. The doc-reader/citation panels use extracted text for
        # preview instead; this route is the "view original" fallback.
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        mime = "text/plain"
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
    """Return saved sessions for the active wiki, sorted by recently updated.

    Scoped to current_wiki_id() — activating a different wiki must change
    what shows in the sidebar, not just what chat answers from. A session
    with no wiki_id (created before this field existed) is treated as
    belonging to the default wiki, matching how the DB rows themselves were
    backfilled — not silently dropped, not shown under every wiki.
    """
    sessions = load_sessions()
    wiki_id = current_wiki_id()
    default_wiki_id = _default_wiki_id()
    session_list = [
        s for s in sessions.values()
        if s.get("wiki_id", default_wiki_id) == wiki_id
    ]
    session_list.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
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
    """Delete all data associated with a session (wiki index, uploads).

    NOT gated on _locked_in_production() — that guard exists for routes that
    mutate the shared WIKI CONTENT (ingest-capable), but in production mode
    every "chat" in the sidebar is its own throwaway session_id used only for
    chat_messages/history, entirely separate from the fixed main wiki session
    (see _get_main_session_id) — deleting one never touches ingested content.
    Applying the ingest lock here meant users could never delete a chat on
    Azure at all: every attempt silently 403'd (the frontend didn't check the
    response status either — see deleteSession() in index.html), leaving the
    chat in the sidebar forever and, if it was the active chat, reloading
    into the upload/ingest panel instead of starting a fresh chat.
    Only refuse when session_id is the actual shared main session — deleting
    THAT would delete the production wiki content itself.
    """
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if session_id == _get_main_session_id():
        return jsonify({"error": "Cannot delete the shared wiki session."}), 403

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

    session_id = _get_main_session_id() or session_id
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

    session_id = _get_main_session_id() or session_id
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
