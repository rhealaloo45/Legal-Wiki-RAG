"""
RAG pipeline — chunk → embed → store in ChromaDB → retrieve at query time.
No LangChain. Simple sliding-window chunker.
"""

import os
import re
import logging
import chromadb
from pypdf import PdfReader

import config
from services import embedder, llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent ChromaDB client — lazy singleton
# ---------------------------------------------------------------------------
_chroma_client = None


def _get_client():
    """Return a persistent ChromaDB client, creating it once on first call."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _chroma_client


def _get_collection(session_id: str):
    """Get or create a ChromaDB collection scoped to the session.

    Uses a no-op embedding function so ChromaDB never tries to re-embed
    stored documents with its own default embedder on restart — we always
    supply pre-computed embeddings from Ollama.
    """
    client = _get_client()
    return client.get_or_create_collection(
        name=f"rag_{session_id}",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------
def _read_file(file_path: str) -> str:
    """Read text from a .txt or .pdf file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking — simple sliding window
# ---------------------------------------------------------------------------
def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + config.CHUNK_SIZE
        chunk_text = text[start:end]
        chunks.append({
            "source": source,
            "chunk_index": idx,
            "text": chunk_text,
        })
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
        idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def ingest(file_path: str, session_id: str) -> dict:
    """Read, chunk, embed, and store a file in ChromaDB."""
    source_name = os.path.basename(file_path)
    text = _read_file(file_path)
    chunks = _chunk_text(text, source_name)

    logger.info("RAG ingest: %d chunks from %s", len(chunks), source_name)

    collection = _get_collection(session_id)

    ids = []
    documents = []
    metadatas = []

    for ch in chunks:
        ids.append(f"{source_name}__chunk_{ch['chunk_index']}")
        documents.append(ch["text"])
        metadatas.append({
            "source": ch["source"],
            "chunk_index": ch["chunk_index"],
        })

    # Embed chunks in parallel (4 workers) for speed
    from concurrent.futures import ThreadPoolExecutor, as_completed

    embeddings = [None] * len(documents)

    def _embed_one(idx):
        vec = embedder.embed(documents[idx])
        logger.info("  Embedded chunk %d/%d", idx + 1, len(documents))
        return idx, vec

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_embed_one, i): i for i in range(len(documents))}
        for future in as_completed(futures):
            idx, vec = future.result()
            embeddings[idx] = vec

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    logger.info("RAG ingest complete: %d chunks stored", len(chunks))
    return {"chunks_stored": len(chunks)}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def query(question: str, session_id: str) -> dict:
    """Retrieve relevant chunks and answer via LLM."""
    collection = _get_collection(session_id)

    q_embedding = embedder.embed(question)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=config.TOP_K,
    )

    # Build context from retrieved chunks
    chunk_details = []
    context_parts = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else None
            source = meta.get("source", "unknown")
            cidx = meta.get("chunk_index", i)
            chunk_details.append({
                "source": source,
                "chunk_index": cidx,
                "text": doc,
                "distance": round(dist, 4) if dist is not None else None,
            })
            context_parts.append(f"[{source}, chunk {cidx}]:\n{doc}")

    context = "\n---\n".join(context_parts)

    prompt = (
        "Answer using only these excerpts. Cite [Source, chunk N] inline.\n"
        "---\n"
        f"{context}\n"
        "---\n"
        f"Question: {question}"
    )

    try:
        answer = llm.ask(prompt, pipeline="rag")
    except RuntimeError as e:
        answer = f"⚠️ LLM error: {e}"

    return {"answer": answer, "chunks": chunk_details}
