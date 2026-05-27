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
import sys
import uuid
import time
import json
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from flask import Flask, render_template, request, jsonify

# Ensure project root is on the path so `import config` works
sys.path.insert(0, os.path.dirname(__file__))

import config
from services import rag, wiki, hybrid

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256 MB total upload limit (folder uploads)

# Ensure data directories exist
for d in [config.CHROMA_PATH, config.WIKI_PATH, config.UPLOAD_PATH]:
    os.makedirs(d, exist_ok=True)

# Configure Tesseract OCR path if set in .env (Windows users)
if config.TESSERACT_CMD:
    from services.reader import configure_tesseract
    configure_tesseract(config.TESSERACT_CMD)

executor = ThreadPoolExecutor(max_workers=10)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}

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


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the single-page UI."""
    return render_template("index.html", llm_provider="AZURE OPENAI")


@app.route("/health")
def health():
    """Health check — confirms the app is running and data dirs are accessible."""
    checks = {
        "status": "ok",
        "llm_provider": "azure",
        "model": config.AZURE_OPENAI_DEPLOYMENT,
        "data_dirs": {
            "chroma": os.path.isdir(config.CHROMA_PATH),
            "wiki": os.path.isdir(config.WIKI_PATH),
            "uploads": os.path.isdir(config.UPLOAD_PATH),
        },
    }
    return jsonify(checks)


@app.route("/upload", methods=["POST"])
def upload():
    """Upload files or folders → immediately accept, then ingest in background via executor.

    Supports nested folder uploads: the frontend sends a `relative_paths` JSON
    array containing the original folder-relative path for each file (e.g.
    "cases/2024/contract.pdf").  These are used to generate descriptive saved
    filenames while keeping them flat on disk.
    """
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

    # Initialize progress with document-level counters
    progress = {
        "phase": "processing",
        "docs": {"total": len(saved_paths), "rag_done": 0, "wiki_done": 0},
        "rag": {},
        "wiki": {},
    }
    config.PROGRESS_STORE[session_id] = progress

    # Submit all tasks to executor (non-blocking)
    for save_path, meta in zip(saved_paths, metadata_list):
        # executor.submit(_ingest_single_doc_rag, save_path, session_id, meta)
        executor.submit(_ingest_single_doc_wiki, save_path, session_id)

    # Save session metadata
    sessions = load_sessions()
    sessions[session_id] = {
        "id": session_id,
        "name": f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": time.time(),
        "updated_at": time.time(),
        "files": len(saved_paths),
        "history": []
    }
    save_sessions(sessions)

    # Return immediately — frontend will poll /progress
    return jsonify({
        "status": "accepted",
        "files_queued": len(saved_paths),
        "session_id": session_id,
    })


# ---------------------------------------------------------------------------
# Background Pipelines
# ---------------------------------------------------------------------------
def _ingest_single_doc_rag(file_path: str, session_id: str, meta: dict = None):
    """Worker for RAG embedding of a single doc."""
    try:
        rag.ingest(file_path, session_id, meta)
    except Exception as e:
        logger.error("RAG ingest failed for %s: %s", file_path, e)
    finally:
        progress = config.PROGRESS_STORE.get(session_id, {})
        docs = progress.get("docs", {})
        docs["rag_done"] = docs.get("rag_done", 0) + 1
        _check_completion(session_id)


def _ingest_single_doc_wiki(file_path: str, session_id: str):
    """Worker for Wiki building of a single doc."""
    try:
        wiki.ingest(file_path, session_id)
    except Exception as e:
        logger.error("Wiki ingest failed for %s: %s", file_path, e)
    finally:
        progress = config.PROGRESS_STORE.get(session_id, {})
        docs = progress.get("docs", {})
        docs["wiki_done"] = docs.get("wiki_done", 0) + 1
        _check_completion(session_id)


def _check_completion(session_id: str):
    """Mark phase as complete when all documents finish both pipelines."""
    progress = config.PROGRESS_STORE.get(session_id, {})
    docs = progress.get("docs", {})
    total = docs.get("total", 0)
    # if total > 0 and docs.get("rag_done", 0) >= total and docs.get("wiki_done", 0) >= total:
    if total > 0 and docs.get("wiki_done", 0) >= total:
        progress["phase"] = "complete"


@app.route("/query", methods=["POST"])
def query_route():
    """Query all pipelines (RAG, Wiki, Hybrid) in parallel and return side-by-side answers."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400

    # Step 1: Fetch contexts concurrently
    t0 = time.time()
    # rag_ctx_future = executor.submit(rag.get_context, question, session_id)
    wiki_ctx_future = executor.submit(wiki.get_context, question, session_id)

    # rag_context, chunk_details = rag_ctx_future.result(timeout=60)
    rag_context = ""
    chunk_details = []
    wiki_context, selected_titles = wiki_ctx_future.result(timeout=300) # Wiki context fetch does an LLM call if pages > 20

    # Step 2: Generate answers concurrently
    # rag_ans_future = executor.submit(rag.generate_answer, question, rag_context, chunk_details)
    wiki_ans_future = executor.submit(wiki.generate_answer, question, wiki_context, selected_titles, session_id)
    # hybrid_ans_future = executor.submit(hybrid.generate_answer, question, rag_context, chunk_details, wiki_context, selected_titles)

    rag_t0 = time.time()
    rag_result = {"answer": "RAG is temporarily disabled.", "chunks": []}
    rag_result["elapsed_ms"] = round((time.time() - rag_t0) * 1000)

    wiki_t0 = time.time()
    try:
        wiki_result = wiki_ans_future.result(timeout=300)
    except Exception as e:
        logger.error("Wiki generation error (%s): %s", type(e).__name__, e)
        wiki_result = {"answer": f"⚠️ Wiki error: {type(e).__name__}: {e}", "pages_used": []}
    wiki_result["elapsed_ms"] = round((time.time() - wiki_t0) * 1000)

    hybrid_t0 = time.time()
    hybrid_result = {"answer": "Hybrid is temporarily disabled.", "usage": {}}
    hybrid_result["elapsed_ms"] = round((time.time() - hybrid_t0) * 1000)

    # Update session history
    sessions = load_sessions()
    if session_id in sessions:
        # Avoid duplicate consecutive questions
        if not sessions[session_id].get("history") or sessions[session_id]["history"][0] != question:
            sessions[session_id].setdefault("history", []).insert(0, question)
        sessions[session_id]["updated_at"] = time.time()
        
        # Rename session if it's the first question and still using the default name
        if len(sessions[session_id]["history"]) == 1 and sessions[session_id]["name"].startswith("Session "):
            new_name = question[:30] + ("..." if len(question) > 30 else "")
            sessions[session_id]["name"] = new_name
            
        save_sessions(sessions)

    return jsonify({
        "rag": rag_result,
        "wiki": wiki_result,
        "hybrid": hybrid_result,
        "total_elapsed_ms": round((time.time() - t0) * 1000)
    })


@app.route("/files")
def file_structure():
    """Return nested file tree of successfully embedded files."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({})
        
    try:
        from services.rag import _get_collection
        col = _get_collection(session_id)
        docs = col.get(include=["metadatas"])
        
        paths = set()
        for m in docs.get("metadatas", []):
            if m:
                path = m.get("relative_path", m.get("filename", "Unknown"))
                path = path.replace("\\", "/")
                paths.add(path)
                
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
    return jsonify(wiki.get_graph(session_id))


@app.route("/wiki/pages")
def wiki_pages_list():
    """Return a sorted list of all wiki page titles for the browser panel."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"pages": []})
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
    index = wiki.get_graph(session_id)
    content = index.get("pages", {}).get(title)
    if content is None:
        return jsonify({"error": "Page not found"}), 404
    return jsonify({"title": title, "content": content})


@app.route("/progress")
def progress():
    """Return the current progress of an ongoing ingest."""
    session_id = request.args.get("session_id", "")
    return jsonify(config.PROGRESS_STORE.get(session_id, {"rag": {}, "wiki": {}}))


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
    """Delete all data associated with a session (ChromaDB collection, wiki index, uploads)."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    errors = []

    # Remove ChromaDB collection
    try:
        client = rag._get_client()
        col_name = f"rag_{session_id}"
        try:
            client.delete_collection(col_name)
            logger.info("Deleted ChromaDB collection: %s", col_name)
        except Exception:
            pass  # collection may not exist yet
    except Exception as e:
        errors.append(f"chroma: {e}")

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
    config.PROGRESS_STORE.pop(session_id, None)

    # Remove from sessions metadata
    sessions = load_sessions()
    if session_id in sessions:
        del sessions[session_id]
        save_sessions(sessions)

    if errors:
        return jsonify({"status": "partial", "errors": errors})
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
