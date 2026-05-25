"""
LLM Wiki pipeline — incremental knowledge compilation.

Unlike RAG (which re-derives from raw chunks at query time), the Wiki pipeline
builds a persistent, structured knowledge base at *ingest* time. Each new source
enriches the same wiki — pages are merged, contradictions flagged, cross-refs
added. Queries read from pre-compiled synthesis, not raw documents.

Architecture:
  - Pages store both full content and a one-line summary for fast index lookup.
  - Per-session threading locks prevent race conditions during parallel ingestion.
  - Query uses a two-step index-based retrieval: pick relevant pages by summary,
    then answer from only those pages' full content.
"""

import json
import os
import re
import logging
import threading
import concurrent.futures

import config
from services import llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread safety — per-session locks for wiki index access
# ---------------------------------------------------------------------------
_session_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()  # protects _session_locks itself


def _get_session_lock(session_id: str) -> threading.Lock:
    """Get or create a lock for the given session."""
    with _locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


# ---------------------------------------------------------------------------
# Wiki I/O
# ---------------------------------------------------------------------------
def _wiki_dir(session_id: str) -> str:
    """Return the wiki directory for a session, creating it if needed."""
    d = os.path.join(config.WIKI_PATH, session_id)
    os.makedirs(d, exist_ok=True)
    return d


def _index_path(session_id: str) -> str:
    return os.path.join(_wiki_dir(session_id), "index.json")


def _load_index(session_id: str) -> dict:
    """Load existing wiki index or return empty scaffold.

    Pages use the new structure: {"title": {"content": "...", "summary": "..."}}
    Gracefully handles the old flat format {"title": "content string"}.
    """
    path = _index_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Migrate old flat format if needed
        pages = data.get("pages", {})
        migrated = {}
        for title, value in pages.items():
            if isinstance(value, str):
                migrated[title] = {"content": value, "summary": ""}
            else:
                migrated[title] = value
        data["pages"] = migrated
        return data
    return {"pages": {}, "relations": []}


def _save_index(session_id: str, index: dict):
    """Persist wiki index to disk."""
    path = _index_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# File reading (mirrors rag.py but we need the full text, not chunks)
# ---------------------------------------------------------------------------
def _read_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# JSON parsing with repair fallback
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
            
    # Try one more time with simple markdown stripping just in case it's a top-level array or something
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _repair_json(raw: str) -> dict:
    """Ask the LLM to fix malformed JSON. Returns parsed dict or empty fallback."""
    repair_prompt = (
        "The following is malformed JSON. Fix it and return only valid JSON, "
        "no explanation:\n"
        f"{raw}"
    )
    try:
        fixed, _ = llm.ask(repair_prompt, pipeline="wiki")
        result = _parse_json_safe(fixed)
        if result is not None:
            return result
    except RuntimeError:
        pass
    logger.error("JSON repair failed — returning empty wiki payload")
    return {"pages": {}, "relations": []}


# ---------------------------------------------------------------------------
# Merge logic — the heart of compounding wiki behaviour
# ---------------------------------------------------------------------------
def _merge_wiki(existing: dict, new_data: dict) -> tuple[dict, int, int]:
    """
    Merge new_data into existing wiki index.

    Returns (merged_index, pages_updated_count, new_relations_count).

    Pages use the structure: {"content": "...", "summary": "..."}
    """
    pages = dict(existing.get("pages", {}))
    relations = list(existing.get("relations", []))

    new_pages = new_data.get("pages", {})
    new_relations = new_data.get("relations", [])

    pages_updated = 0

    # -- Merge pages --
    for title, new_value in new_pages.items():
        # Normalize new_value to {content, summary} format
        if isinstance(new_value, str):
            new_content = new_value
            new_summary = ""
        else:
            new_content = new_value.get("content", "")
            new_summary = new_value.get("summary", "")

        if title in pages:
            # Existing page — append content, keep better summary
            existing_page = pages[title]
            existing_content = existing_page.get("content", "")
            existing_summary = existing_page.get("summary", "")
            pages[title] = {
                "content": existing_content + "\n\n---\n" + new_content,
                "summary": new_summary if new_summary else existing_summary,
            }
            pages_updated += 1
        else:
            pages[title] = {"content": new_content, "summary": new_summary}
            pages_updated += 1

    # -- Merge relations (deduplicate by exact (from, to, label) tuple) --
    existing_tuples = {
        (r["from"], r["to"], r["label"]) for r in relations
    }
    new_rel_count = 0
    for rel in new_relations:
        key = (rel.get("from", ""), rel.get("to", ""), rel.get("label", ""))
        if key not in existing_tuples:
            relations.append(rel)
            existing_tuples.add(key)
            new_rel_count += 1

    # -- Cross-reference pass: detect implicit "mentions" relations --
    all_titles = set(pages.keys())
    for title_a, page_data in pages.items():
        content = page_data.get("content", "") if isinstance(page_data, dict) else page_data
        for title_b in all_titles:
            if title_a == title_b:
                continue
            if title_b in content:
                key = (title_a, title_b, "mentions")
                if key not in existing_tuples:
                    relations.append({
                        "from": title_a,
                        "to": title_b,
                        "label": "mentions",
                    })
                    existing_tuples.add(key)
                    new_rel_count += 1

    merged = {"pages": pages, "relations": relations}
    return merged, pages_updated, new_rel_count


# ---------------------------------------------------------------------------
# Ingest — synthesis-oriented prompts, smarter segmentation
# ---------------------------------------------------------------------------

# Single-call prompt for documents that fit in one LLM call (≤ 100K chars)
INGEST_PROMPT_TEMPLATE = """\
You are a legal wiki knowledge synthesizer. Read this document and create wiki pages \
that capture its legal meaning, statutory basis, precedents, and judicial reasoning.

PRINCIPLES:
- SOURCE INTEGRITY: DO NOT hallucinate or invent citations. Only cite cases, statutes, \
  or document names explicitly present in the text. This information comes from the \
  document '{doc_name}'. Explicitly mention the document name in your synthesis.
- FACTUAL PRECISION: DO NOT hallucinate dates or facts. If a date is not explicitly \
  stated, do not include it. Extract EXACT verbatim quotes for critical dates, \
  figures, and holdings.
- LEGAL DEPTH: Create pages for key precedents, statutory provisions, and the \
  judicial reasoning (ratio decidendi). Explain HOW the law was applied to the \
  facts, not just what the law is. Explain the Holding/Conclusion.
- Each page should read like a well-written wiki article.
- Include exact numbers, amounts, dates, rates, and timeframes verbatim.
- Flag contradictions or ambiguities you notice.

PAGE TITLES: Use specific, descriptive titles (e.g., "Ratio Decidendi: Late Payment Penalties", \
"Application of Section 3") — not generic ones like "Overview".

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{{
  "pages": {{
    "Descriptive Page Title": {{
      "content": "4-10 sentence detailed synthesis with specific provisions, numbers, and conditions. Explain what it means and how it connects to other parts of the document.",
      "summary": "One-line summary of what this page covers."
    }}
  }},
  "relations": [
    {{"from": "Page Title A", "to": "Page Title B", "label": "short verb phrase"}}
  ]
}}

Extract 10-30 pages and 10-40 relations. Cover the document thoroughly.

DOCUMENT:
{text}"""


# Phase 1 prompt for large documents: extract overview + entity list
OVERVIEW_PROMPT_TEMPLATE = """\
You are a legal wiki knowledge synthesizer. Read this document excerpt (beginning and \
end of a larger document) and produce:

1. A "Document Overview" page summarizing the document's purpose, parties, \
   scope, and key themes.
2. A list of ALL specific topics, provisions, precedents, and legal concepts that \
   should each get their own wiki page.

SOURCE INTEGRITY: This excerpt is from '{doc_name}'. DO NOT hallucinate citations.

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{{
  "overview_page": {{
    "content": "Detailed 6-12 sentence summary of the document's purpose, parties involved, scope, and key themes.",
    "summary": "One-line summary of the document."
  }},
  "topics": [
    "Topic or Provision Name 1",
    "Topic or Provision Name 2"
  ]
}}

DOCUMENT EXCERPT:
{text}"""


# Phase 2 prompt for large documents: detailed extraction with known topics
DETAIL_PROMPT_TEMPLATE = """\
You are a legal wiki knowledge synthesizer. You are processing a SEGMENT of a larger \
document named '{doc_name}'. An overview pass has already identified these topics that need wiki pages:

KNOWN TOPICS: {topics}

Your job: read this segment and create/update wiki pages for any of the known \
topics that appear here. Also create pages for any NEW important legal topics you \
discover.

RULES:
- SOURCE INTEGRITY: DO NOT hallucinate cases or citations. Only use what is in the text.
- FACTUAL PRECISION: Extract exact dates, amounts, and figures verbatim. Do not invent dates.
- LEGAL DEPTH: Focus on statutory interpretation, judicial reasoning, and precedents.
- Reuse the KNOWN TOPIC names exactly as page titles when applicable.
- Each page should be 4-10 sentences of detailed synthesis.

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{{
  "pages": {{
    "Page Title": {{
      "content": "Detailed synthesis...",
      "summary": "One-line summary."
    }}
  }},
  "relations": [
    {{"from": "Page Title A", "to": "Page Title B", "label": "short verb phrase"}}
  ]
}}

DOCUMENT SEGMENT:
{text}"""


# Threshold: documents under this size are processed in one LLM call
_SINGLE_CALL_THRESHOLD = 100000
# Segment size for large documents
_INGEST_CHUNK_SIZE = 40000


def ingest(file_path: str, session_id: str) -> dict:
    """Read a source document, extract wiki pages via LLM, and merge into the session wiki.

    Short documents (≤ 100K chars): processed in a single LLM call.
    Long documents: two-phase approach — overview first, then detailed segments
    with the overview's topic list as context to reduce redundancy.
    Segments are processed concurrently to improve speed.
    """
    text = _read_file(file_path)
    doc_name = os.path.basename(file_path)

    logger.info("Wiki ingest: %s (%d chars)", doc_name, len(text))

    progress = config.PROGRESS_STORE.setdefault(session_id, {"rag": {}, "wiki": {}})

    if len(text) <= _SINGLE_CALL_THRESHOLD:
        # --- Short document: single LLM call ---
        progress["wiki"] = {"current": 0, "total": 1, "message": f"Processing {doc_name}..."}

        parsed = _ingest_single_call(text, doc_name)
        progress["wiki"]["current"] = 1

        total_pages, total_rels = _atomic_merge(session_id, parsed)
    else:
        # --- Long document: two-phase approach ---
        segments = _split_segments(text)
        total_steps = 1 + len(segments)  # 1 overview + N segments
        progress["wiki"] = {"current": 0, "total": total_steps, "message": f"Overview pass for {doc_name}..."}

        # Phase 1: Overview
        overview_text = text[:6000] + "\n\n[...]\n\n" + text[-3000:]
        topics, overview_parsed = _ingest_overview(overview_text, doc_name)
        progress["wiki"]["current"] = 1

        # Merge overview page immediately
        total_pages, total_rels = _atomic_merge(session_id, overview_parsed)

        # Phase 2: Detailed segments concurrently
        completed_segments = 0
        max_workers = (
            config.WIKI_MAX_WORKERS_OPENROUTER
            if config.LLM_PROVIDER == "openrouter"
            else config.WIKI_MAX_WORKERS_OLLAMA
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_ingest_detail_segment, seg, topics, doc_name): i
                for i, seg in enumerate(segments)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                completed_segments += 1
                msg = f"Detail pass completed {completed_segments}/{len(segments)} for {doc_name}"
                logger.info("  %s", msg)
                progress["wiki"]["current"] = 1 + completed_segments
                progress["wiki"]["message"] = msg
                
                try:
                    parsed = future.result()
                    p, r = _atomic_merge(session_id, parsed)
                    total_pages += p
                    total_rels += r
                except Exception as exc:
                    logger.error("Segment %d for %s generated an exception: %s", i, doc_name, exc)

    logger.info("Wiki ingest complete: %d pages, %d relations", total_pages, total_rels)
    progress["wiki"]["message"] = f"Complete: {total_pages} pages extracted."
    return {"pages_updated": total_pages, "relations": total_rels}


def _ingest_single_call(text: str, doc_name: str) -> dict:
    """Process a short document in one LLM call."""
    prompt = INGEST_PROMPT_TEMPLATE.format(text=text, doc_name=doc_name)
    try:
        raw, _ = llm.ask(prompt, pipeline="wiki")
    except RuntimeError as e:
        logger.error("LLM call failed during wiki ingest: %s", e)
        return {"pages": {}, "relations": []}

    parsed = _parse_json_safe(raw)
    if parsed is None:
        parsed = _repair_json(raw)

    if "pages" not in parsed:
        parsed["pages"] = {}
    if "relations" not in parsed:
        parsed["relations"] = []

    return parsed


def _ingest_overview(text: str, doc_name: str) -> tuple[list[str], dict]:
    """Phase 1: extract overview + topic list from document excerpt."""
    prompt = OVERVIEW_PROMPT_TEMPLATE.format(text=text, doc_name=doc_name)
    try:
        raw, _ = llm.ask(prompt, pipeline="wiki")
    except RuntimeError as e:
        logger.error("LLM overview call failed: %s", e)
        return [], {"pages": {}, "relations": []}

    parsed = _parse_json_safe(raw)
    if parsed is None:
        parsed = _repair_json(raw)

    topics = parsed.get("topics", [])
    overview_page = parsed.get("overview_page", {})

    # Convert to standard merge format
    doc_pages = {}
    if overview_page:
        doc_pages["Document Overview"] = overview_page

    return topics, {"pages": doc_pages, "relations": []}


def _ingest_detail_segment(text: str, topics: list[str], doc_name: str) -> dict:
    """Phase 2: extract detailed pages from a segment with known topic context."""
    topics_str = ", ".join(topics) if topics else "None identified yet"
    prompt = DETAIL_PROMPT_TEMPLATE.format(text=text, topics=topics_str, doc_name=doc_name)
    try:
        raw, _ = llm.ask(prompt, pipeline="wiki")
    except RuntimeError as e:
        logger.error("LLM detail call failed: %s", e)
        return {"pages": {}, "relations": []}

    parsed = _parse_json_safe(raw)
    if parsed is None:
        parsed = _repair_json(raw)

    if "pages" not in parsed:
        parsed["pages"] = {}
    if "relations" not in parsed:
        parsed["relations"] = []

    return parsed


def _split_segments(text: str) -> list[str]:
    """Split text into overlapping segments for large document processing."""
    segments = []
    start = 0
    while start < len(text):
        end = start + _INGEST_CHUNK_SIZE
        segments.append(text[start:end])
        start += _INGEST_CHUNK_SIZE - 500  # 500 char overlap
    return segments if segments else [text]


def _atomic_merge(session_id: str, new_data: dict) -> tuple[int, int]:
    """Thread-safe: load index, merge new data, save — all under lock."""
    lock = _get_session_lock(session_id)
    with lock:
        existing = _load_index(session_id)
        merged, pages_updated, new_rels = _merge_wiki(existing, new_data)
        _save_index(session_id, merged)
    return pages_updated, new_rels


# ---------------------------------------------------------------------------
# Query — index-based retrieval for accuracy at scale
# ---------------------------------------------------------------------------

# Step 1: select relevant pages by title + summary
PAGE_SELECT_PROMPT = """\
You are selecting relevant wiki pages to answer a question. Below is an index \
of all available pages with their one-line summaries.

Pick the 10-15 MOST RELEVANT pages for answering this question. Return ONLY \
a JSON array of page titles, no explanation:

["Page Title 1", "Page Title 2", ...]

WIKI INDEX:
{page_index}

QUESTION: {question}"""


# Step 2: answer from selected pages
QUERY_PROMPT_TEMPLATE = """\
You are answering a question using selected wiki pages. These pages were \
compiled from source documents.

RULES:
- Answer primarily from the wiki content below.
- Reference specific wiki pages inline using [Page Title] notation whenever you draw from them.
- If the wiki does not contain enough information, say so — do NOT make up facts.
- IMPORTANT: If your answer introduces new concepts, synthesis, or insights not already explicit in the wiki pages, \
extract them into new pages and relations so the wiki grows smarter.

OUTPUT FORMAT — respond with valid JSON only, no markdown fences:
{{
  "answer": "Your detailed answer here...",
  "new_pages": {{
    "New Concept/Insight": {{
      "content": "Detailed explanation derived from your answer...",
      "summary": "One-line summary."
    }}
  }},
  "new_relations": [
    {{"from": "Existing Page", "to": "New Concept", "label": "explains"}}
  ]
}}

WIKI PAGES:
{wiki_content}

QUESTION: {question}"""


def get_context(question: str, session_id: str) -> tuple[str, list]:
    """Select relevant pages for a query and return them as a formatted string + list of titles."""
    index = _load_index(session_id)
    pages = index.get("pages", {})

    if not pages:
        return "", []

    # --- Step 1: Select relevant pages via index ---
    # For small wikis (≤ 20 pages), skip selection and use all pages
    if len(pages) <= 20:
        selected_titles = list(pages.keys())
    else:
        selected_titles = _select_relevant_pages(pages, question)

    # --- Step 2: Answer from selected pages ---
    wiki_parts = []
    for title in selected_titles:
        if title in pages:
            page = pages[title]
            content = page.get("content", "") if isinstance(page, dict) else page
            wiki_parts.append(f"## {title}\n{content}\n")
    wiki_content = "\n".join(wiki_parts)

    return wiki_content, selected_titles


def generate_answer(question: str, wiki_content: str, selected_titles: list, session_id: str) -> dict:
    """Generate an answer using the provided wiki content."""
    index = _load_index(session_id)
    pages = index.get("pages", {})
    relations = index.get("relations", [])

    if not wiki_content:
        return {
            "answer": "The wiki is empty — no documents have been ingested yet.",
            "pages_used": [],
            "relations": relations,
            "usage": {}
        }

    from services.prompts import ANSWER_PROMPT
    prompt = ANSWER_PROMPT.format(context=wiki_content, question=question)

    usage = {}
    try:
        answer, usage = llm.ask(prompt, pipeline="wiki")
    except RuntimeError as e:
        answer = f"⚠️ LLM error: {e}"

    # Extract [Page Title] references from the answer
    referenced = re.findall(r"\[([^\]]+)\]", answer)
    valid_titles = set(pages.keys())
    pages_used = [t for t in referenced if t in valid_titles]
    # Deduplicate while preserving order
    seen = set()
    pages_used_dedup = []
    for t in pages_used:
        if t not in seen:
            pages_used_dedup.append(t)
            seen.add(t)

    return {
        "answer": answer,
        "pages_used": pages_used_dedup,
        "relations": relations,
        "usage": usage,
    }


def _select_relevant_pages(pages: dict, question: str) -> list[str]:
    """Use LLM to pick the most relevant pages for a question based on title + summary."""
    # Build compact index: "Title: summary"
    index_lines = []
    for title, page in pages.items():
        summary = page.get("summary", "") if isinstance(page, dict) else ""
        line = f"- {title}: {summary}" if summary else f"- {title}"
        index_lines.append(line)
    page_index = "\n".join(index_lines)

    prompt = PAGE_SELECT_PROMPT.format(page_index=page_index, question=question)

    try:
        raw, _ = llm.ask(prompt, pipeline="wiki")
        parsed = _parse_json_safe(raw)
        if isinstance(parsed, list):
            # Filter to titles that actually exist
            valid = [t for t in parsed if t in pages]
            if valid:
                return valid
    except (RuntimeError, Exception) as e:
        logger.error("Page selection failed: %s — falling back to all pages", e)

    # Fallback: return all pages (capped at 30 to avoid prompt overflow)
    return list(pages.keys())[:30]


def get_graph(session_id: str) -> dict:
    """Return the full wiki index for graph rendering.

    Converts the internal {content, summary} format to flat {title: content}
    for backward compatibility with the D3 frontend.
    """
    index = _load_index(session_id)
    # Flatten pages for the frontend
    flat_pages = {}
    for title, page in index.get("pages", {}).items():
        if isinstance(page, dict):
            flat_pages[title] = page.get("content", "")
        else:
            flat_pages[title] = page
    return {"pages": flat_pages, "relations": index.get("relations", [])}
