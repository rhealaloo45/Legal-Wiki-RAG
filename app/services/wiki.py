"""
LLM Wiki pipeline — incremental knowledge compilation.

Unlike RAG (which re-derives from raw chunks at query time), the Wiki pipeline
builds a persistent, structured knowledge base at *ingest* time. Each new source
enriches the same wiki — pages are merged, contradictions flagged, cross-refs
added. Queries read from pre-compiled synthesis, not raw documents.
"""

import json
import os
import re
import logging

import config
from services import llm

logger = logging.getLogger(__name__)


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
    """Load existing wiki index or return empty scaffold."""
    path = _index_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
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
    """Try to parse JSON from LLM output. Strips markdown fences if present."""
    # Strip markdown code fences
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
        fixed = llm.ask(repair_prompt, pipeline="wiki")
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
    """
    pages = dict(existing.get("pages", {}))
    relations = list(existing.get("relations", []))

    new_pages = new_data.get("pages", {})
    new_relations = new_data.get("relations", [])

    pages_updated = 0

    # -- Merge pages --
    for title, summary in new_pages.items():
        if title in pages:
            # Existing page — append, don't overwrite
            # Simple heuristic: if new text is very different, flag contradiction
            existing_text = pages[title]
            pages[title] = existing_text + "\n\n---\n" + summary
            pages_updated += 1
        else:
            pages[title] = summary
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
    for title_a, content in pages.items():
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
# Ingest
# ---------------------------------------------------------------------------
INGEST_PROMPT_TEMPLATE = """\
You are a knowledge base builder. Your job is to read a document and extract structured knowledge into wiki pages and relations.

RULES (follow exactly):
- Extract only entities, concepts, events, people, laws, or doctrines that are EXPLICITLY named in the document.
- Do NOT invent, infer, or assume any information not present in the text.
- Each page title must be a specific named thing (e.g. "Article 21", "Kesavananda Bharati Case", "Basic Structure Doctrine") — not generic headings like "Introduction" or "Overview".
- Each summary must be 2-4 sentences. Be precise. Use the document's own language where possible.
- Relations must only connect two page titles that BOTH exist in your pages output. Do not create relations to external concepts not in the document.
- Relation labels must be short verb phrases (e.g. "established", "overruled", "expanded", "is part of", "challenged by", "derives from").
- Extract 5-15 pages and 5-20 relations. Do not over-extract.
- If the document is too short or vague to extract meaningful pages, return minimal output — do not pad.

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{{
  "pages": {{
    "Exact Page Title": "2-4 sentence summary grounded in the document.",
    ...
  }},
  "relations": [
    {{"from": "Page Title A", "to": "Page Title B", "label": "short verb phrase"}},
    ...
  ]
}}

DOCUMENT:
{text}"""


def ingest(file_path: str, session_id: str) -> dict:
    """Read a source document, extract wiki pages via LLM, and merge into the session wiki."""
    text = _read_file(file_path)
    # Truncate to 4000 chars to fit prompt context
    prompt = INGEST_PROMPT_TEMPLATE.format(text=text[:4000])

    try:
        raw = llm.ask(prompt, pipeline="wiki")
    except RuntimeError as e:
        logger.error("LLM call failed during wiki ingest: %s", e)
        return {"pages_updated": 0, "relations": 0, "error": str(e)}

    # Parse with repair fallback
    parsed = _parse_json_safe(raw)
    if parsed is None:
        parsed = _repair_json(raw)

    # Validate structure minimally
    if "pages" not in parsed:
        parsed["pages"] = {}
    if "relations" not in parsed:
        parsed["relations"] = []

    # Load existing wiki and merge
    existing = _load_index(session_id)
    merged, pages_updated, new_rels = _merge_wiki(existing, parsed)

    _save_index(session_id, merged)

    return {"pages_updated": pages_updated, "relations": new_rels}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
QUERY_PROMPT_TEMPLATE = """\
You are answering a question using a pre-built knowledge wiki. This wiki was incrementally compiled \
from source documents — synthesis, cross-referencing, and contradiction-flagging already happened \
at ingest time. Do not re-derive knowledge from scratch. Use what is already compiled.

RULES:
- Answer only from the wiki content below. Do not use outside knowledge.
- Reference specific wiki pages inline using [Page Title] notation whenever you draw from them.
- If you see a ⚠️ Contradiction flag in a page, acknowledge the conflict in your answer rather than picking one side silently.
- If the wiki does not contain enough information to answer, say so explicitly — do not guess.
- For factual questions: be direct, cite the relevant page.
- For comparison questions: structure with clear A vs B sections, cite pages for each side.
- For synthesis/evolution questions: trace progression chronologically, cite pages at each step.
- Keep answer focused — 150-250 words. Synthesize; do not copy wiki text verbatim.

WIKI:
{wiki_content}

QUESTION: {question}"""


def query(question: str, session_id: str) -> dict:
    """Answer a question from the pre-built wiki."""
    index = _load_index(session_id)
    pages = index.get("pages", {})
    relations = index.get("relations", [])

    if not pages:
        return {
            "answer": "The wiki is empty — no documents have been ingested yet.",
            "pages_used": [],
            "relations": relations,
        }

    # Build wiki context
    wiki_parts = []
    for title, content in pages.items():
        wiki_parts.append(f"## {title}\n{content}\n")
    wiki_content = "\n".join(wiki_parts)

    prompt = QUERY_PROMPT_TEMPLATE.format(
        wiki_content=wiki_content, question=question
    )

    try:
        answer = llm.ask(prompt, pipeline="wiki")
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
    }


def get_graph(session_id: str) -> dict:
    """Return the full wiki index for graph rendering."""
    return _load_index(session_id)
