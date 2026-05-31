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
from services.reader import read_file as _read_file

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
# Thread safety — per-session locks for log writes
# ---------------------------------------------------------------------------
_log_locks: dict[str, threading.Lock] = {}

def _get_log_lock(session_id: str) -> threading.Lock:
    """Get or create a lock for logging in the given session."""
    with _locks_lock:
        if session_id not in _log_locks:
            _log_locks[session_id] = threading.Lock()
        return _log_locks[session_id]

def _log_event(session_id: str, event_type: str, detail: str):
    """Append a timestamped event to the session log."""
    from datetime import datetime
    lock = _get_log_lock(session_id)
    with lock:
        log_path = os.path.join(config.LOGS_PATH, f"{session_id}_log.md")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"## [{timestamp}] {event_type} | {detail}\n")
        except Exception as e:
            logger.error("Failed to write to session log: %s", e)


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
def _merge_wiki(existing: dict, new_data: dict, doc_name: str = "Unknown") -> tuple[dict, int, int, list]:
    """
    Merge new_data into existing wiki index.

    Returns (merged_index, pages_updated_count, new_relations_count, contradictions_found).

    Pages use the structure: {"content": "...", "summary": "..."}
    """
    pages = dict(existing.get("pages", {}))
    relations = list(existing.get("relations", []))

    new_pages = new_data.get("pages", {})
    new_relations = new_data.get("relations", [])

    pages_updated = 0
    contradictions_found = []

    # -- Merge pages --
    for title, new_value in new_pages.items():
        # Normalize new_value to {content, summary, quotes} format
        if isinstance(new_value, str):
            new_content = new_value
            new_summary = ""
            new_quotes = []
        else:
            new_content = new_value.get("content", "")
            new_summary = new_value.get("summary", "")
            new_quotes = new_value.get("quotes", [])

        # Append quotes to the content
        if new_quotes:
            quote_text = "\n\n**Supporting Quotes:**\n" + "\n".join(f"> {q}" for q in new_quotes)
            new_content += quote_text

        if title in pages:
            # Existing page — append content, keep better summary
            existing_page = pages[title]
            existing_content = existing_page.get("content", "")
            existing_summary = existing_page.get("summary", "")
            
            # Contradiction check pre-flight
            if len(new_content) > 200 and len(existing_content) > 200:
                prompt = (
                    "Do these two texts contradict each other on any specific factual claim (dates, values, obligations, parties)?\n"
                    "Reply JSON only:\n"
                    "{\"contradicts\": bool, \"claim\": str|null, \"value_a\": str|null, \"value_b\": str|null}\n\n"
                    f"Text A:\n{existing_content}\n\n"
                    f"Text B:\n{new_content}"
                )
                try:
                    raw, _ = llm.ask(prompt, max_tokens=100)
                    parsed = _parse_json_safe(raw)
                    if parsed and parsed.get("contradicts"):
                        existing_page["contradiction_flagged"] = True
                        from datetime import datetime
                        if "variants" not in existing_page:
                            existing_page["variants"] = [{"source": "Previous", "value": existing_content, "date_ingested": datetime.now().isoformat()}]
                        existing_page["variants"].append({
                            "source": doc_name,
                            "value": new_content,
                            "date_ingested": datetime.now().isoformat()
                        })
                        contradictions_found.append({
                            "title": title,
                            "claim": parsed.get("claim"),
                            "val_a": parsed.get("value_a"),
                            "val_b": parsed.get("value_b"),
                            "doc": doc_name
                        })
                except Exception as e:
                    logger.error("Contradiction check failed: %s", e)

            pages[title] = {
                "content": existing_content + "\n\n---\n" + new_content,
                "summary": new_summary if new_summary else existing_summary,
                "source_doc": doc_name
            }
            if "contradiction_flagged" in existing_page:
                pages[title]["contradiction_flagged"] = existing_page["contradiction_flagged"]
            if "variants" in existing_page:
                pages[title]["variants"] = existing_page["variants"]

            pages_updated += 1
        else:
            pages[title] = {"content": new_content, "summary": new_summary, "source_doc": doc_name}
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
    return merged, pages_updated, new_rel_count, contradictions_found


# ---------------------------------------------------------------------------
# Ingest — synthesis-oriented prompts, smarter segmentation
# ---------------------------------------------------------------------------

INGEST_PROMPT_TEMPLATE = """\
You are a legal wiki knowledge synthesizer. Read this document and create wiki pages \
that capture its legal meaning, statutory basis, precedents, and judicial reasoning.

PRINCIPLES:
- SOURCE INTEGRITY: DO NOT hallucinate or invent citations. Only cite cases, statutes, \
  or document names explicitly present in the text. This information comes from the \
  file '{doc_name}'.
- DOCUMENT TYPE INFERENCE: You must determine the actual nature of the document from its contents \
  (e.g., "Non-Disclosure Agreement", "Master Service Agreement", "Court Judgment", "Legal Opinion"). \
  Do NOT just use the filename '{doc_name}'.
- FACTUAL PRECISION: DO NOT hallucinate dates or facts. If a date is not explicitly \
  stated, do not include it. Extract EXACT verbatim quotes for critical dates, \
  figures, and holdings.
- ANTI-HALLUCINATION: You MUST extract exact verbatim quotes to support your synthesis for EVERY page.
- LEGAL DEPTH: Create pages for key precedents, statutory provisions, and the \
  judicial reasoning (ratio decidendi). Explain HOW the law was applied to the \
  facts, not just what the law is. Explain the Holding/Conclusion.
- ROLE & OBLIGATION ACCURACY: Be extremely precise about WHO bears an obligation and WHO receives a benefit (e.g., who pays a fee, who provides a certificate). Do not reverse roles.
- LIMITATIONS & EXCEPTIONS: Accurately capture if a clause (like indemnity or confidentiality) is subject to a limitation of liability. Do not state obligations remain 'full' if they are capped or limited.
- TERMINATION: Clearly distinguish between unilateral (one-party) and mutual termination rights.
- Each page should read like a well-written wiki article.
- Include exact numbers, amounts, dates, rates, and timeframes verbatim.
- Flag contradictions or ambiguities you notice.

PAGE TITLES: You MUST append the inferred Document Type in parentheses to EVERY page title. For example: "Payment Terms (Master Service Agreement)", "Confidentiality (NDA)".

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{{
  "doc_type": "The inferred document type (e.g. 'NDA')",
  "pages": {{
    "Descriptive Page Title (Inferred Doc Type)": {{
      "content": "4-10 sentence detailed synthesis with specific provisions, numbers, and conditions. Explain what it means and how it connects to other parts of the document.",
      "summary": "One-line summary of what this page covers.",
      "quotes": ["Exact verbatim quote 1 from the text", "Exact verbatim quote 2 from the text"]
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
DOCUMENT TYPE INFERENCE: You must determine the actual nature of the document from its contents \
(e.g., "Non-Disclosure Agreement", "Master Service Agreement"). Do NOT just use the filename.

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{{
  "doc_type": "The inferred document type (e.g. 'NDA')",
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
document named '{doc_name}'. The document type has been inferred as '{doc_type}'.
An overview pass has already identified these topics that need wiki pages:

KNOWN TOPICS: {topics}

Your job: read this segment and create/update wiki pages for any of the known \
topics that appear here. Also create pages for any NEW important legal topics you \
discover.

RULES:
- SOURCE INTEGRITY: DO NOT hallucinate cases or citations. Only use what is in the text.
- FACTUAL PRECISION: Extract exact dates, amounts, and figures verbatim. Do not invent dates.
- ANTI-HALLUCINATION: You MUST extract exact verbatim quotes to support your synthesis for EVERY page.
- LEGAL DEPTH: Focus on statutory interpretation, judicial reasoning, and precedents.
- ROLE & OBLIGATION ACCURACY: Be extremely precise about WHO bears an obligation and WHO receives a benefit (e.g., who pays a fee, who provides a certificate). Do not reverse roles.
- LIMITATIONS & EXCEPTIONS: Accurately capture if a clause (like indemnity or confidentiality) is subject to a limitation of liability. Do not state obligations remain 'full' if they are capped or limited.
- TERMINATION: Clearly distinguish between unilateral (one-party) and mutual termination rights.
- PAGE TITLES: You MUST append the inferred Document Type in parentheses to EVERY page title. For example: "Topic Name ({doc_type})".
- Each page should be 4-10 sentences of detailed synthesis.

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{{
  "pages": {{
    "Page Title (Inferred Doc Type)": {{
      "content": "Detailed synthesis...",
      "summary": "One-line summary.",
      "quotes": ["Exact verbatim quote 1", "Exact verbatim quote 2"]
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

    total_contradictions = 0

    if len(text) <= _SINGLE_CALL_THRESHOLD:
        # --- Short document: single LLM call ---
        progress["wiki"] = {"current": 0, "total": 1, "message": f"Processing {doc_name}..."}

        parsed = _ingest_single_call(text, doc_name)
        progress["wiki"]["current"] = 1

        total_pages, total_rels, total_contradictions = _atomic_merge(session_id, parsed, doc_name)
    else:
        # --- Long document: two-phase approach ---
        segments = _split_segments(text)
        total_steps = 1 + len(segments)  # 1 overview + N segments
        progress["wiki"] = {"current": 0, "total": total_steps, "message": f"Overview pass for {doc_name}..."}

        # Phase 1: Overview
        overview_text = text[:6000] + "\n\n[...]\n\n" + text[-3000:]
        doc_type, topics, overview_parsed = _ingest_overview(overview_text, doc_name)
        progress["wiki"]["current"] = 1

        # Merge overview page immediately
        total_pages, total_rels, tc = _atomic_merge(session_id, overview_parsed, doc_name)
        total_contradictions += tc

        # Phase 2: Detailed segments concurrently
        completed_segments = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.WIKI_MAX_WORKERS
        ) as executor:
            future_to_index = {
                executor.submit(_ingest_detail_segment, seg, topics, doc_name, doc_type): i
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
                    p, r, c = _atomic_merge(session_id, parsed, doc_name)
                    total_pages += p
                    total_rels += r
                    total_contradictions += c
                except Exception as exc:
                    logger.error("Segment %d for %s generated an exception: %s", i, doc_name, exc)
                    _log_event(session_id, "ERROR", f"Doc: {doc_name} | Segment {i} failed: {exc}")

    logger.info("Wiki ingest complete: %d pages, %d relations", total_pages, total_rels)
    _log_event(session_id, "INGEST", f"Doc: {doc_name} | Pages updated: {total_pages} | Contradictions found: {total_contradictions}")
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


def _ingest_overview(text: str, doc_name: str) -> tuple[str, list[str], dict]:
    """Phase 1: extract overview + topic list from document excerpt."""
    prompt = OVERVIEW_PROMPT_TEMPLATE.format(text=text, doc_name=doc_name)
    try:
        raw, _ = llm.ask(prompt, pipeline="wiki")
    except RuntimeError as e:
        logger.error("LLM overview call failed: %s", e)
        return "Unknown Document", [], {"pages": {}, "relations": []}

    parsed = _parse_json_safe(raw)
    if parsed is None:
        parsed = _repair_json(raw)

    doc_type = parsed.get("doc_type", doc_name)
    topics = parsed.get("topics", [])
    overview_page = parsed.get("overview_page", {})

    # Convert to standard merge format
    doc_pages = {}
    if overview_page:
        # Append inferred doc type to the overview page title
        doc_pages[f"Document Overview ({doc_type})"] = overview_page

    return topics, {"pages": doc_pages, "relations": []}


def _ingest_detail_segment(text: str, topics: list[str], doc_name: str, doc_type: str) -> dict:
    """Phase 2: extract detailed pages from a segment with known topic context."""
    topics_str = ", ".join(topics) if topics else "None identified yet"
    prompt = DETAIL_PROMPT_TEMPLATE.format(text=text, topics=topics_str, doc_name=doc_name, doc_type=doc_type)
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


def _atomic_merge(session_id: str, new_data: dict, doc_name: str = "Unknown") -> tuple[int, int, int]:
    """Thread-safe: load index, merge new data, save — all under lock."""
    lock = _get_session_lock(session_id)
    with lock:
        existing = _load_index(session_id)
        merged, pages_updated, new_rels, contradictions = _merge_wiki(existing, new_data, doc_name)
        _save_index(session_id, merged)
        
        for c in contradictions:
            _log_event(
                session_id, 
                "CONTRADICTION", 
                f"Page: {c['title']} | Claim: {c['claim']} | Source A: {c.get('val_a')} | Source B: {c.get('val_b')}"
            )
            
    return pages_updated, new_rels, len(contradictions)


# ---------------------------------------------------------------------------
# File-mention detection — prioritize pages from a specific source document
# ---------------------------------------------------------------------------
def _extract_doc_names(pages: dict) -> dict[str, str]:
    """Extract unique document names from page titles.

    Page titles follow the convention ``Topic Name (doc_name)``.  Returns a
    mapping  {doc_name_from_title: doc_name_from_title}  so callers can look up
    the canonical form used in titles.
    """
    doc_names: dict[str, str] = {}
    for title in pages:
        match = re.search(r'\(([^)]+)\)\s*$', title.strip())
        if match:
            doc_names[match.group(1)] = match.group(1)
    return doc_names


def _detect_mentioned_files(question: str, pages: dict) -> set[str]:
    """Detect which source documents the user is asking about.

    Builds several normalised variants of each doc name (with/without session
    prefix, underscores replaced by spaces, etc.) and checks whether any of
    them appear in the question text.  Returns the set of *canonical* doc names
    (as they appear in page titles) that matched.
    """
    doc_names = _extract_doc_names(pages)
    if not doc_names:
        return set()

    question_lower = question.lower()
    matched: set[str] = set()

    for canonical in doc_names:
        # 1. Exact match on the full doc name
        if canonical.lower() in question_lower:
            matched.add(canonical)
            continue

        # 2. Strip a leading session-id prefix (UUID or hex followed by _)
        stripped = re.sub(r'^[a-f0-9_-]+?_', '', canonical, count=1)
        if stripped and stripped.lower() in question_lower:
            matched.add(canonical)
            continue

        # 3. Replace underscores with spaces (user likely types spaces)
        spacified = stripped.replace('_', ' ')
        if spacified and spacified.lower() in question_lower:
            matched.add(canonical)
            continue

        # 4. Also try the spacified version without the extension
        no_ext = os.path.splitext(spacified)[0]
        if no_ext and len(no_ext) >= 4 and no_ext.lower() in question_lower:
            matched.add(canonical)
            continue

    if matched:
        logger.info("Detected file mentions in query: %s", matched)

    return matched


def _pages_from_files(pages: dict, file_names: set[str]) -> list[str]:
    """Return page titles that belong to any of the given source files."""
    result = []
    for title in pages:
        for fname in file_names:
            if fname in title:
                result.append(title)
                break
    return result


# ---------------------------------------------------------------------------
# Query — index-based retrieval for accuracy at scale
# ---------------------------------------------------------------------------

# Step 1: select relevant pages by title + summary
PAGE_SELECT_PROMPT = """\
You are selecting relevant wiki pages to answer a question. Below is an index \
of all available pages with their one-line summaries.

Pick the 10-15 MOST RELEVANT pages for answering this question. Pay close attention \
to any document types, names, or categories mentioned in the question (e.g., "NDAs", "Joint Venture Agreements") \
and ONLY select pages that match those constraints. Return ONLY a JSON array of page titles, no explanation:

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
    """Select relevant pages for a query and return them as a formatted string + list of titles.

    If the question mentions a specific source file (e.g. "Legal Opinion 2.pdf"),
    all pages originating from that file are force-included so the answer stays
    grounded in the correct document.
    """
    index = _load_index(session_id)
    pages = index.get("pages", {})

    if not pages:
        return {"context": "", "selected_titles": [], "bm25_count": 0}

    # --- Step 0: Detect file mentions in the question ---
    mentioned_files = _detect_mentioned_files(question, pages)
    file_pages = _pages_from_files(pages, mentioned_files) if mentioned_files else []

    bm25_count = 0
    pages_for_llm = pages

    if len(pages) > 20:
        try:
            from rank_bm25 import BM25Okapi
            corpus = []
            keys = list(pages.keys())
            for k in keys:
                summary = pages[k].get("summary", "") if isinstance(pages[k], dict) else ""
                corpus.append(f"{k} {summary}".lower().split())
            bm25 = BM25Okapi(corpus)
            tokenized_query = question.lower().split()
            top_40_keys = bm25.get_top_n(tokenized_query, keys, n=40)
            pages_for_llm = {k: pages[k] for k in top_40_keys}
            bm25_count = len(pages_for_llm)
        except ImportError:
            pass

    # --- Step 1: Select relevant pages ---
    if file_pages:
        # File explicitly mentioned — force those pages in
        if len(pages) <= 20:
            # Small wiki: file pages first, then everything else
            other = [t for t in pages if t not in file_pages]
            selected_titles = file_pages + other
        else:
            # Large wiki: file pages + LLM-selected supplementary pages
            llm_selected = _select_relevant_pages(pages_for_llm, question)
            seen = set(file_pages)
            supplementary = [t for t in llm_selected if t not in seen]
            selected_titles = file_pages + supplementary
        logger.info("File-focused query: %d pages from mentioned file(s), %d total selected",
                     len(file_pages), len(selected_titles))
    else:
        # No file mentioned — original behaviour
        if len(pages) <= 20:
            selected_titles = list(pages.keys())
        else:
            selected_titles = _select_relevant_pages(pages_for_llm, question)

    # --- Step 2: Build context string from selected pages ---
    wiki_parts = []
    for title in selected_titles:
        if title in pages:
            page = pages[title]
            content = page.get("content", "") if isinstance(page, dict) else page
            
            if isinstance(page, dict) and page.get("contradiction_flagged"):
                content = "[WARNING: This page contains conflicting claims. Surface the conflict explicitly in your answer. Do not resolve it.]\n" + content
                
            wiki_parts.append(f"## {title}\n{content}\n")
    wiki_content = "\n".join(wiki_parts)

    return {
        "context": wiki_content,
        "selected_titles": selected_titles,
        "bm25_count": bm25_count
    }


def _evaluate_confidence(question: str, context: str, answer: str) -> dict:
    """Evaluate LLM confidence score for the generated answer based on the context."""
    prompt = f"""\
You are an expert legal editor. Evaluate the confidence score of a generated answer to a user's question, based on the provided context.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate the alignment, completeness, and supportiveness of the context for the generated answer.
Provide a confidence score between 0 and 100 representing how well the answer is supported by the context:
- 90-100: Context fully and directly answers the question with specific details.
- 70-80: Context mostly answers the question but some non-critical details might be missing or require minor inference.
- 40-60: Context only partially answers the question; major gaps exist.
- 0-30: Context does not support the answer or does not contain relevant information.

Respond with ONLY a JSON object containing:
- "confidence_score": an integer between 0 and 100
- "reason": a short 1-sentence explanation of the score

JSON:"""
    try:
        raw, _ = llm.ask(prompt, pipeline="wiki")
        parsed = _parse_json_safe(raw)
        if parsed and "confidence_score" in parsed:
            score = int(parsed["confidence_score"])
            reason = parsed.get("reason", "")
            return {"score": score, "reason": reason}
    except Exception as e:
        logger.error("Failed to evaluate confidence: %s", e)
    
    # Fallback score based on context presence
    score = 85 if context else 0
    return {"score": score, "reason": "Confidence evaluated from context availability."}


def generate_answer(question: str, wiki_content: str, selected_titles: list, session_id: str, bm25_count: int = 0) -> dict:
    """Generate an answer using the provided wiki content."""
    index = _load_index(session_id)
    pages = index.get("pages", {})
    relations = index.get("relations", [])

    if not wiki_content:
        return {
            "answer": "The wiki is empty — no documents have been ingested yet.",
            "pages_used": [],
            "files_used": [],
            "relations": relations,
            "usage": {},
            "confidence_score": 0,
            "confidence_reason": "No context available."
        }

    from services.prompts import ANSWER_PROMPT
    prompt = ANSWER_PROMPT.format(context=wiki_content, question=question)

    usage = {}
    try:
        answer, usage = llm.ask(prompt, pipeline="wiki")
        import re
        answer = re.sub(r'<reasoning>.*?</reasoning>', '', answer, flags=re.DOTALL).strip()
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

    # Extract source files from page titles
    files_used = []
    for title in pages_used_dedup:
        match = re.search(r'\(([^)]+)\)\s*$', title.strip())
        if match:
            doc_name = match.group(1)
            if doc_name not in files_used:
                files_used.append(doc_name)

    # Evaluate confidence score
    confidence = _evaluate_confidence(question, wiki_content, answer)

    # Log query
    _log_event(session_id, "QUERY", f"Q: {question[:60]}... | BM25 Shortlist: {bm25_count} | Pages selected: {len(selected_titles)} | Confidence: {confidence['score']}")

    # Answer filing back to wiki
    if confidence["score"] >= 80:
        q_title_prefix = f"Q: {question[:50]}"
        existing_titles = [t for t in pages if t.startswith("Q: ")]
        
        target_title = None
        for t in existing_titles:
            if q_title_prefix in t:
                target_title = t
                break
                
        if not target_title:
            target_title = f"{q_title_prefix}..."
            
        new_page_payload = {
            "pages": {
                target_title: {
                    "content": answer,
                    "summary": answer[:100].replace("\n", " ") + "...",
                    "quotes": []
                }
            },
            "relations": []
        }
        # We must re-fetch lock via _atomic_merge
        _atomic_merge(session_id, new_page_payload, doc_name="Query Answer")
        _log_event(session_id, "QUERY_FILED", f"Page: {target_title}")

    return {
        "answer": answer,
        "pages_used": pages_used_dedup,
        "files_used": files_used,
        "relations": relations,
        "usage": usage,
        "confidence_score": confidence["score"],
        "confidence_reason": confidence["reason"]
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
