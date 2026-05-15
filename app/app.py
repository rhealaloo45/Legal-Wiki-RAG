"""
Flask application — RAG vs LLM Wiki comparison app.

Routes:
  GET  /            → Serve single-page UI
  POST /upload      → Ingest file into both RAG and Wiki pipelines (parallel)
  POST /query       → Query both pipelines in parallel, return side-by-side answers
  GET  /wiki/graph  → Return wiki graph data for D3 rendering
"""

import os
import sys
import uuid
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


@app.route("/upload", methods=["POST"])
def upload():
    """Upload a file → ingest into both RAG and Wiki pipelines in parallel."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    session_id = request.form.get("session_id", str(uuid.uuid4()))

    if not file.filename or not _allowed_file(file.filename):
        return jsonify({"error": "Only .txt and .pdf files are accepted"}), 400

    # Save uploaded file
    safe_name = file.filename.replace(os.sep, "_")
    save_path = os.path.join(config.UPLOAD_PATH, f"{session_id}_{safe_name}")
    file.save(save_path)
    logger.info("Saved upload: %s", save_path)

    # Run both pipelines in parallel
    rag_future = executor.submit(rag.ingest, save_path, session_id)
    wiki_future = executor.submit(wiki.ingest, save_path, session_id)

    try:
        rag_result = rag_future.result(timeout=300)
    except Exception as e:
        logger.error("RAG ingest error: %s", e)
        rag_result = {"error": str(e), "chunks_stored": 0}

    try:
        wiki_result = wiki_future.result(timeout=300)
    except Exception as e:
        logger.error("Wiki ingest error: %s", e)
        wiki_result = {"error": str(e), "pages_updated": 0, "relations": 0}

    return jsonify({
        "status": "ok",
        "filename": safe_name,
        "rag": rag_result,
        "wiki": wiki_result,
    })


@app.route("/query", methods=["POST"])
def query_route():
    """Query both pipelines in parallel and return side-by-side answers."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400

    rag_future = executor.submit(rag.query, question, session_id)
    wiki_future = executor.submit(wiki.query, question, session_id)

    try:
        rag_result = rag_future.result(timeout=120)
    except Exception as e:
        logger.error("RAG query error: %s", e)
        rag_result = {"answer": f"⚠️ RAG error: {e}", "chunks": []}

    try:
        wiki_result = wiki_future.result(timeout=120)
    except Exception as e:
        logger.error("Wiki query error: %s", e)
        wiki_result = {"answer": f"⚠️ Wiki error: {e}", "pages_used": [], "relations": []}

    return jsonify({"rag": rag_result, "wiki": wiki_result})


@app.route("/wiki/graph")
def wiki_graph():
    """Return wiki graph data (pages + relations) for D3 rendering."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return jsonify({"pages": {}, "relations": []})
    return jsonify(wiki.get_graph(session_id))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
