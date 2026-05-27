"""
RAG pipeline — chunk → embed → store in ChromaDB → retrieve at query time.
No LangChain. Simple sliding-window chunker.
"""
import os
import re
import json
import logging
import threading
import chromadb

import config
from services import embedder, llm
from services.reader import read_file as _read_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent ChromaDB client — lazy singleton (Thread-Safe)
# ---------------------------------------------------------------------------
_chroma_client = None
_chroma_lock = threading.RLock()


def _get_client():
    """Return a persistent ChromaDB client, creating it once on first call."""
    global _chroma_client
    with _chroma_lock:
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
    with _chroma_lock:
        return client.get_or_create_collection(
            name=f"rag_{session_id}",
            metadata={"hnsw:space": "cosine"},
        )




# ---------------------------------------------------------------------------
# Chunking — simple sliding window
# ---------------------------------------------------------------------------
def _parse_json_safe(raw: str) -> dict | None:
    """Try to parse JSON from LLM output by extracting the outermost brackets."""
    start = raw.find('{')
    end = raw.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        cleaned = raw[start:end+1]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
            
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _extract_document_metadata(excerpt: str) -> dict:
    """Extract document type, date, parties, and summary from the document excerpt using LLM."""
    prompt = (
        "You are a legal metadata extractor. Read the first page/excerpt of the document and extract key metadata.\n\n"
        "Extract:\n"
        "1. Document Type (e.g. Court Case Judgment, NDA, Service Agreement, Shareholder Agreement, Act, Ordinance, etc.)\n"
        "2. Key Parties involved (e.g. Appellant, Respondent, Plaintiff, Defendant, Contracting Parties)\n"
        "3. Important Date(s) (e.g. Judgment Date, Agreement Date, Effective Date)\n"
        "4. A brief 1-2 sentence summary of the document's main subject.\n\n"
        "DOCUMENT EXCERPT:\n"
        f"{excerpt}\n\n"
        "OUTPUT FORMAT — Respond with valid JSON only, no explanation, no markdown fences:\n"
        "{\n"
        "  \"document_type\": \"...\",\n"
        "  \"parties\": \"...\",\n"
        "  \"date\": \"...\",\n"
        "  \"brief_summary\": \"...\"\n"
        "}"
    )
    
    try:
        raw, _ = llm.ask(prompt, pipeline="rag")
        parsed = _parse_json_safe(raw)
        
        # Simple fallback/repair logic using standard wiki repair utility if available
        if parsed is None:
            try:
                import services.wiki as wiki_service
                parsed = wiki_service._repair_json(raw)
            except Exception:
                pass
        
        if isinstance(parsed, dict):
            return {
                "document_type": parsed.get("document_type", "Unknown"),
                "parties": parsed.get("parties", "Unknown"),
                "date": parsed.get("date", "Unknown"),
                "brief_summary": parsed.get("brief_summary", "Unknown")
            }
    except Exception as e:
        logger.error("Failed to extract document metadata: %s", e)
    
    return {
        "document_type": "Unknown",
        "parties": "Unknown",
        "date": "Unknown",
        "brief_summary": "Unknown"
    }


def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with basic metadata."""
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


def _chunk_text_with_metadata(text: str, source: str, relative_path: str, filename: str, category: str, doc_metadata: dict) -> list[dict]:
    """Split text into overlapping chunks and prepend metadata header to the text of each chunk."""
    header = (
        f"[Document Metadata]\n"
        f"Filename: {filename}\n"
        f"Relative Path: {relative_path}\n"
        f"Category: {category}\n"
        f"Document Type: {doc_metadata.get('document_type', 'Unknown')}\n"
        f"Parties: {doc_metadata.get('parties', 'Unknown')}\n"
        f"Date: {doc_metadata.get('date', 'Unknown')}\n"
        f"Summary: {doc_metadata.get('brief_summary', 'Unknown')}\n"
        f"---\n"
    )
    
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + config.CHUNK_SIZE
        chunk_body = text[start:end]
        # Prepend the header to the chunk text so it is embedded and visible to the LLM
        chunk_text = f"{header}{chunk_body}"
        
        chunks.append({
            "source": source,
            "chunk_index": idx,
            "text": chunk_text,
            "relative_path": relative_path,
            "filename": filename,
            "category": category,
            "document_type": doc_metadata.get("document_type", "Unknown"),
            "parties": doc_metadata.get("parties", "Unknown"),
            "date": doc_metadata.get("date", "Unknown"),
            "brief_summary": doc_metadata.get("brief_summary", "Unknown")
        })
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
        idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def ingest(file_path: str, session_id: str, meta: dict = None) -> dict:
    """Read, chunk, embed, and store a file in ChromaDB."""
    if meta is None:
        meta = {}

    source_name = os.path.basename(file_path)
    text = _read_file(file_path)

    # Extract path metadata
    relative_path = meta.get("relative_path", source_name)
    filename = meta.get("filename", source_name)
    parts = [p for p in relative_path.replace("\\", "/").split("/") if p]
    category = parts[-2] if len(parts) >= 2 else "General"

    # Extract legal metadata via LLM using first 4000 characters
    logger.info("Extracting document metadata for %s...", filename)
    doc_metadata = _extract_document_metadata(text[:4000])
    logger.info("Metadata extracted: %s", doc_metadata)

    chunks = _chunk_text_with_metadata(
        text=text, 
        source=source_name,
        relative_path=relative_path,
        filename=filename,
        category=category,
        doc_metadata=doc_metadata
    )

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
            "relative_path": ch["relative_path"],
            "filename": ch["filename"],
            "category": ch["category"],
            "document_type": ch["document_type"],
            "parties": ch["parties"],
            "date": ch["date"],
            "brief_summary": ch["brief_summary"],
        })

    progress = config.PROGRESS_STORE.setdefault(session_id, {"rag": {}, "wiki": {}})
    progress["rag"] = {"current": 0, "total": len(documents), "message": f"Embedding {len(documents)} chunks (batch)..."}

    # Batch embed all chunks in one call instead of one-at-a-time (not a query)
    embeddings = embedder.embed_batch(documents, is_query=False)

    progress["rag"]["current"] = len(documents)
    progress["rag"]["message"] = f"Complete: {len(chunks)} chunks stored."

    if ids:
        with _chroma_lock:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    logger.info("RAG ingest complete: %d chunks stored", len(chunks))
    return {"chunks_stored": len(chunks)}


def _detect_mentioned_sources(question: str, sources: set[str]) -> set[str]:
    """Detect which source documents the user is asking about."""
    if not sources:
        return set()

    question_lower = question.lower()
    matched: set[str] = set()

    for source in sources:
        # 1. Exact match
        if source.lower() in question_lower:
            matched.add(source)
            continue

        # 2. Strip a leading session-id prefix
        stripped = re.sub(r'^[a-f0-9_-]+?_', '', source, count=1)
        if stripped and stripped.lower() in question_lower:
            matched.add(source)
            continue

        # 3. Replace underscores with spaces
        spacified = stripped.replace('_', ' ')
        if spacified and spacified.lower() in question_lower:
            matched.add(source)
            continue

        # 4. Also try the spacified version without the extension
        no_ext = os.path.splitext(spacified)[0]
        if no_ext and len(no_ext) >= 4 and no_ext.lower() in question_lower:
            matched.add(source)
            continue

    if matched:
        logger.info("RAG: Detected file mentions in query: %s", matched)

    return matched


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def get_context(question: str, session_id: str) -> tuple[str, list]:
    """Retrieve relevant chunks and format as a context string and chunk list."""
    collection = _get_collection(session_id)


    q_embedding = embedder.embed(question, is_query=True)

    with _chroma_lock:
        all_docs = collection.get(include=["metadatas"])
        
    unique_sources = set()
    if all_docs and all_docs.get("metadatas"):
        for meta in all_docs["metadatas"]:
            if meta and "source" in meta:
                unique_sources.add(meta["source"])
                
    matched_sources = _detect_mentioned_sources(question, unique_sources)
    where_filter = None
    if matched_sources:
        if len(matched_sources) == 1:
            where_filter = {"source": list(matched_sources)[0]}
        else:
            where_filter = {"source": {"$in": list(matched_sources)}}

    with _chroma_lock:
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=config.TOP_K,
            where=where_filter
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
                "relative_path": meta.get("relative_path", ""),
                "filename": meta.get("filename", ""),
                "category": meta.get("category", ""),
                "document_type": meta.get("document_type", ""),
                "parties": meta.get("parties", ""),
                "date": meta.get("date", ""),
                "brief_summary": meta.get("brief_summary", ""),
            })
            context_parts.append(f"[{source}, chunk {cidx}]:\n{doc}")

    if not chunk_details:
        return "", []

    context = "\n---\n".join(context_parts)
    return context, chunk_details


def generate_answer(question: str, context: str, chunk_details: list) -> dict:
    """Generate an answer using the provided RAG context."""
    if not chunk_details:
        return {
            "answer": "I'm sorry, but I can't provide an answer based on the excerpts you requested.",
            "chunks": [],
            "usage": {}
        }

    from services.prompts import ANSWER_PROMPT
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    usage = {}
    try:
        answer, usage = llm.ask(prompt, pipeline="rag")
    except RuntimeError as e:
        answer = f"⚠️ LLM error: {e}"

    return {"answer": answer, "chunks": chunk_details, "usage": usage}


def query(question: str, session_id: str) -> dict:
    """Convenience method combining retrieval and generation."""
    context, chunk_details = get_context(question, session_id)
    return generate_answer(question, context, chunk_details)
