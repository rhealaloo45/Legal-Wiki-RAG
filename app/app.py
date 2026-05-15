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
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify

# Ensure project root is on the path so `import config` works
sys.path.insert(0, os.path.dirname(__file__))

import config
from services import rag, wiki

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure data directories exist
for d in [config.CHROMA_PATH, config.WIKI_PATH, config.UPLOAD_PATH]:
    os.makedirs(d, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=4)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the single-page UI."""
    return render_template("index.html", llm_provider=config.LLM_PROVIDER.upper())


@app.route("/health")
def health():
    """Health check — confirms the app is running and data dirs are accessible."""
    checks = {
        "status": "ok",
        "llm_provider": config.LLM_PROVIDER,
        "model": config.OPENROUTER_MODEL if config.LLM_PROVIDER == "openrouter" else config.OLLAMA_MODEL,
        "data_dirs": {
            "chroma": os.path.isdir(config.CHROMA_PATH),
            "wiki": os.path.isdir(config.WIKI_PATH),
            "uploads": os.path.isdir(config.UPLOAD_PATH),
        },
    }
    return jsonify(checks)


@app.route("/upload", methods=["POST"])
def upload():
    """Upload a file → ingest into both RAG and Wiki pipelines in parallel."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    files = request.files.getlist("file")
    session_id = request.form.get("session_id", str(uuid.uuid4()))

    if not files or not files[0].filename:
        return jsonify({"error": "No file provided"}), 400

    total_rag = {"chunks_stored": 0}
    total_wiki = {"pages_updated": 0, "relations": 0}

    for file in files:
        if not _allowed_file(file.filename):
            continue

        # Save uploaded file
        safe_name = file.filename.replace(os.sep, "_")
        save_path = os.path.join(config.UPLOAD_PATH, f"{session_id}_{safe_name}")
        file.save(save_path)
        logger.info("Saved upload: %s", save_path)

        # Run both pipelines in parallel for this file
        rag_future = executor.submit(rag.ingest, save_path, session_id)
        wiki_future = executor.submit(wiki.ingest, save_path, session_id)

        try:
            rag_result = rag_future.result(timeout=1200)
            total_rag["chunks_stored"] += rag_result.get("chunks_stored", 0)
        except Exception as e:
            logger.error("RAG ingest error (%s): %s", type(e).__name__, e)

        try:
            wiki_result = wiki_future.result(timeout=1200)
            total_wiki["pages_updated"] += wiki_result.get("pages_updated", 0)
            total_wiki["relations"] += wiki_result.get("relations", 0)
        except Exception as e:
            logger.error("Wiki ingest error (%s): %s", type(e).__name__, e)

    return jsonify({
        "status": "ok",
        "files_processed": len(files),
        "rag": total_rag,
        "wiki": total_wiki,
    })


@app.route("/query", methods=["POST"])
def query_route():
    """Query both pipelines in parallel and return side-by-side answers with timing."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400

    t0 = time.time()
    rag_future = executor.submit(rag.query, question, session_id)
    wiki_future = executor.submit(wiki.query, question, session_id)

    rag_t0 = time.time()
    try:
        rag_result = rag_future.result(timeout=120)
    except Exception as e:
        logger.error("RAG query error: %s", e)
        rag_result = {"answer": f"⚠️ RAG error: {e}", "chunks": []}
    rag_result["elapsed_ms"] = round((time.time() - rag_t0) * 1000)

    wiki_t0 = time.time()
    try:
        wiki_result = wiki_future.result(timeout=120)
    except Exception as e:
        logger.error("Wiki query error: %s", e)
        wiki_result = {"answer": f"⚠️ Wiki error: {e}", "pages_used": [], "relations": []}
    wiki_result["elapsed_ms"] = round((time.time() - wiki_t0) * 1000)

    logger.info("Query answered in %.2fs total", time.time() - t0)
    return jsonify({"rag": rag_result, "wiki": wiki_result})


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

    if errors:
        return jsonify({"status": "partial", "errors": errors})
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
