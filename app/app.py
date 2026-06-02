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
from services import rag, wiki, hybrid, advanced_modes
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
    
    wiki_ctx_res = wiki_ctx_future.result(timeout=300) # Wiki context fetch does an LLM call if pages > 20
    wiki_context = wiki_ctx_res.get("context", "")
    selected_titles = wiki_ctx_res.get("selected_titles", [])
    bm25_count = wiki_ctx_res.get("bm25_count", 0)

    # Step 2: Generate answers concurrently
    # rag_ans_future = executor.submit(rag.generate_answer, question, rag_context, chunk_details)
    wiki_ans_future = executor.submit(wiki.generate_answer, question, wiki_context, selected_titles, session_id, bm25_count)
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
    """Return nested file tree of successfully uploaded files."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({})
        
    try:
        sessions = load_sessions()
        paths = set()
        
        # 1. Try to read from session metadata first
        if session_id in sessions and "file_paths" in sessions[session_id]:
            for p in sessions[session_id]["file_paths"]:
                paths.add(p.replace("\\", "/"))
        else:
            # 2. Fallback: Scan config.UPLOAD_PATH and reconstruct original paths
            prefix = f"{session_id}_"
            for fname in os.listdir(config.UPLOAD_PATH):
                if fname.startswith(prefix):
                    rel_name = fname[len(prefix):]
                    
                    # Reconstruction logic for existing folders
                    reconstructed = False
                    top_dir = "Legal AI Tool - Tata Group"
                    subdirs = [
                        "Court Case Documents",
                        "Joint Venture Agreements",
                        "Judgments",
                        "Legal Opinions",
                        "NDA",
                        "Service Agreement",
                        "Shareholder Agreements"
                    ]
                    
                    if rel_name.startswith(f"{top_dir}_"):
                        rest = rel_name[len(top_dir)+1:]
                        for subdir in subdirs:
                            if rest.startswith(f"{subdir}_"):
                                file_part = rest[len(subdir)+1:]
                                paths.add(f"{top_dir}/{subdir}/{file_part}")
                                reconstructed = True
                                break
                    
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
# Review Mode Routes
# ---------------------------------------------------------------------------
@app.route("/review/start", methods=["POST"])
def review_start():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    doc_names = data.get("doc_names", [])
    columns = data.get("columns", [])
    
    if not session_id or not doc_names or not columns:
        return jsonify({"error": "session_id, doc_names, and columns are required"}), 400
        
    job_id = str(uuid.uuid4())
    REVIEW_STORE[job_id] = {
        "status": "running",
        "total": len(doc_names) * len(columns),
        "completed": 0,
        "rows": {d: {} for d in doc_names},
        "flagged": [],
        "error": None,
        "columns": columns
    }
    
    _get_review_lock(job_id)  # Initialize the lock for this job
    logger.info(f"Starting review job {job_id} for {len(doc_names)} docs and {len(columns)} columns")
    executor.submit(advanced_modes._run_review_job, job_id, session_id, doc_names, columns, REVIEW_STORE, _review_locks)
    return jsonify({"job_id": job_id, "total_cells": len(doc_names) * len(columns)})

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
        
    if not doc_names and not uploaded_file:
        return jsonify({"error": "Must select at least one document or upload a file"}), 400
        
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
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
