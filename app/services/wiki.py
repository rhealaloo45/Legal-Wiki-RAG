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

if config.USE_DATABASE:
    from services import db as _db

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
# Wiki I/O — file-based (fallback when DATABASE_URL is not set)
# ---------------------------------------------------------------------------
def _wiki_dir(session_id: str) -> str:
    """Return the wiki directory for a session, creating it if needed."""
    d = os.path.join(config.WIKI_PATH, session_id)
    os.makedirs(d, exist_ok=True)
    return d


def _index_path(session_id: str) -> str:
    return os.path.join(_wiki_dir(session_id), "index.json")


def _load_index_file(session_id: str) -> dict:
    """Load wiki index from index.json or return empty scaffold."""
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


def _save_index_file(session_id: str, index: dict) -> None:
    """Persist wiki index to disk."""
    path = _index_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Wiki I/O — DB-backed (when DATABASE_URL is set)
# ---------------------------------------------------------------------------
def _load_index_db(session_id: str) -> dict:
    """Load wiki index from PostgreSQL. Auto-migrates from index.json on first access."""
    json_path = _index_path(session_id)
    if os.path.exists(json_path) and _db.count_pages(session_id) == 0:
        logger.info("Auto-migrating session %s from index.json to PostgreSQL", session_id)
        _db.migrate_from_json(session_id, json_path)
        os.rename(json_path, json_path + ".migrated")

    pages = _db.get_pages(session_id)
    relations = _db.get_relations(session_id)
    return {"pages": pages, "relations": relations}


# ---------------------------------------------------------------------------
# Unified Wiki I/O — dispatches to DB or file based on config
# ---------------------------------------------------------------------------
def _load_index(session_id: str) -> dict:
    if config.USE_DATABASE:
        return _load_index_db(session_id)
    return _load_index_file(session_id)


def _save_index(session_id: str, index: dict) -> None:
    """Persist wiki index. No-op in DB mode — writes happen directly in _atomic_merge."""
    if not config.USE_DATABASE:
        _save_index_file(session_id, index)




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
    """Ask the LLM to fix malformed JSON. Returns parsed dict or empty fallback.

    Uses the fast/cheap model — JSON repair is purely syntactic formatting work,
    not legal synthesis. Token cap at MAX_TOKENS_JSON_REPAIR since output is
    bounded by the input size.
    """
    repair_prompt = (
        "The following is malformed JSON. Fix it and return only valid JSON, "
        "no explanation:\n"
        f"{raw}"
    )
    try:
        fixed, _ = llm.ask(
            repair_prompt,
            pipeline="wiki",
            fast=True,
            max_tokens=config.MAX_TOKENS_JSON_REPAIR,
        )
        result = _parse_json_safe(fixed)
        if result is not None:
            return result
    except RuntimeError:
        pass
    logger.error("JSON repair failed — returning empty wiki payload")
    return {"pages": {}, "relations": []}


# ---------------------------------------------------------------------------
# C3 — Structural NER pre-filter for contradiction detection (Phase 4)
# ---------------------------------------------------------------------------
_AMOUNT_RE = re.compile(
    r'\$[\d,]+(?:\.\d+)?'
    r'|\b\d[\d,]*(?:\.\d+)?\s*(?:million|crore|lakh|thousand|USD|INR|GBP|EUR)\b',
    re.I,
)
_DATE_RE = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
_PCT_RE  = re.compile(r'\b\d+(?:\.\d+)?\s*%')


def _extract_structural_values(text: str) -> set[str]:
    """Return lowercased set of monetary amounts, dates, and percentages found in text."""
    vals: set[str] = set()
    vals.update(m.strip().lower() for m in _AMOUNT_RE.findall(text))
    vals.update(m.strip().lower() for m in _DATE_RE.findall(text))
    vals.update(m.strip().lower() for m in _PCT_RE.findall(text))
    return vals


def _has_structural_conflict(text_a: str, text_b: str) -> bool:
    """Return True if the two texts should be checked for contradictions by an LLM.

    Returns False (skip LLM) only when both texts contain structured values
    (amounts/dates/percentages) and all of those values match exactly.
    If either text has no structured values we cannot rule out a contradiction,
    so we return True to let the LLM decide.
    """
    vals_a = _extract_structural_values(text_a)
    vals_b = _extract_structural_values(text_b)
    if not vals_a or not vals_b:
        return True
    return bool(vals_a.symmetric_difference(vals_b))


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
            # Uses fast/cheap model — this is a structured boolean check, not synthesis.
            if len(new_content) > 200 and len(existing_content) > 200:
                prompt = (
                    "Do these two texts contradict each other on any specific factual claim (dates, values, obligations, parties)?\n"
                    "Reply JSON only:\n"
                    "{\"contradicts\": bool, \"claim\": str|null, \"value_a\": str|null, \"value_b\": str|null}\n\n"
                    f"Text A:\n{existing_content}\n\n"
                    f"Text B:\n{new_content}"
                )
                try:
                    raw, _ = llm.ask(
                        prompt,
                        fast=True,
                        max_tokens=config.MAX_TOKENS_CONTRADICTION,
                    )
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
- ALLEGATIONS VS. FINDINGS: In court judgments and criminal matters, rigorously \
  distinguish between (1) prosecution/plaintiff allegations, (2) party contentions, \
  (3) court observations or prima facie findings, and (4) final orders/convictions. \
  Use language like "the prosecution alleged", "the court held", "the petitioner contended". \
  Never write a charge or prosecution theory as if it were a concluded judicial finding.
- ROLE & OBLIGATION ACCURACY: Be extremely precise about WHO bears an obligation and WHO receives a benefit (e.g., who pays a fee, who provides a certificate). Do not reverse roles.
- LIMITATIONS & EXCEPTIONS: Accurately capture if a clause (like indemnity or confidentiality) is subject to a limitation of liability. Do not state obligations remain 'full' if they are capped or limited.
- TERMINATION: Clearly distinguish between unilateral (one-party) and mutual termination rights.
- ARITHMETIC PROHIBITION: Never compute, derive, multiply, or extrapolate numeric values. Only state figures that appear VERBATIM in the source text. If the text says "₹1 lakh per acre" and separately mentions a land area, do NOT multiply them to produce a total — quote each figure as it appears. A derived number is a hallucination even if the arithmetic is correct.
- STATUTE INTERPRETATION: When a statute section number appears (e.g., "Section 182 IPC"), only describe what the text explicitly says about it. Do NOT apply external legal knowledge to explain what that section means or implies.
- RELIEF SEQUENCING: When capturing suit prayers or reliefs, preserve the exact order and primacy as stated in the plaint/petition. Do not reorder reliefs by perceived importance.
- PROCEDURAL ORDER RATIONALE (CRITICAL): For every procedural order in a court judgment \
  (committal to Sessions Court, framing/discharge of charges, remand, stay, transfer), you MUST \
  explicitly record (a) which court made the order, (b) under which section/provision, and \
  (c) the court's stated reason or finding that triggered it. Example: "The Magistrate committed \
  the case to the Sessions Court under Section 209 CrPC because it found that Section 304 Part II \
  IPC — an offence exclusively triable by the Sessions Court — was prima facie attracted." Never \
  record a procedural step without its stated reason.
- MULTI-STAGE LITIGATION (CRITICAL): When a case has multiple stages (e.g., Trial Court → High \
  Court → Supreme Court, or First SLP → Remand → Second Appeal), clearly label and separate each \
  stage. Record what each court decided and why, using stage labels such as "First SLP", "High \
  Court (Writ Petition)", "Supreme Court (Final Appeal)". Never blend outcomes from different \
  stages into a single undifferentiated account.
- Each page should read like a well-written wiki article.
- Include exact numbers, amounts, dates, rates, and timeframes verbatim.
- Flag contradictions or ambiguities you notice.

PAGE TITLES: You MUST append the inferred Document Type in parentheses to EVERY page title. \
CASE-SPECIFIC PAGES (CRITICAL): For court judgments, some pages describe facts unique to THIS \
case and must NOT merge with pages from other cases. For these pages, prefix the title with the \
SHORT CASE IDENTIFIER (first party's last name from the document, e.g. "Yuvraj Kanther"): \
  - Facts, Background, Procedural History, Parties → "Facts – Yuvraj Kanther (Court Judgment)" \
  - Charges, FIR, Offences → "Charges – Yuvraj Kanther (Court Judgment)" \
  - Holding, Disposition, Final Order → "Holding – Yuvraj Kanther (Court Judgment)" \
  - Appellants'/Respondents' Contentions, Arguments → "Appellants' Contentions – Yuvraj Kanther (Court Judgment)" \
  - Relief, Costs, Sentence → "Relief – Yuvraj Kanther (Court Judgment)" \
SHARED LEGAL CONCEPT PAGES: Pages about statutes, precedents, legal doctrines, and general \
principles should NOT include the case name — they are intentionally merged when multiple cases \
discuss the same concept: "Section 319 CrPC (Court Judgment)", "Hardeep Singh v. Punjab (Court Judgment)". \
For contracts: "Payment Terms (Master Service Agreement)", "Confidentiality (NDA)".

OUTPUT FORMAT — respond with valid JSON only, no explanation, no markdown fences:
{{
  "doc_type": "The inferred document type (e.g. 'NDA')",
  "metadata": {{
    "governing_law": "Jurisdiction whose law governs (e.g. 'English law') or null",
    "jurisdiction": "Court/tribunal with exclusive jurisdiction or null",
    "effective_date": "Contract start date verbatim or null",
    "termination_notice": "Notice period required to terminate (e.g. '30 days') or null",
    "liability_cap": "Maximum liability figure or formula verbatim or null",
    "ip_ownership": "Who owns IP created under the agreement or null",
    "parties": "Comma-separated list of party names or null",
    "auto_renewal": "Auto-renewal clause verbatim or null",
    "notice_period": "General notice period for communications or null",
    "payment_terms": "Payment due date / terms (e.g. 'Net 30') or null"
  }},
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
CASE-SPECIFIC TOPICS (CRITICAL): For court judgments, separate topics into two categories: \
1. CASE-SPECIFIC (facts, procedural history, charges, holding, parties, contentions, relief, costs): \
   prefix each topic with the SHORT CASE IDENTIFIER (first party's last name from the document): \
   e.g. "Procedural History – Yuvraj Kanther", "Facts – Yuvraj Kanther", "Holding – Yuvraj Kanther". \
2. SHARED LEGAL CONCEPTS (statutes, precedents, doctrines, principles): NO case-name prefix — \
   these pages are intentionally merged when multiple cases discuss the same concept: \
   e.g. "Section 319 CrPC", "Hardeep Singh v. Punjab", "Non-obstante Clause". \
For contracts, plain topic names are fine (e.g. "Payment Terms", "Confidentiality").

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
- ALLEGATIONS VS. FINDINGS: In court judgments and criminal matters, rigorously \
  distinguish between (1) prosecution/plaintiff allegations, (2) party contentions, \
  (3) court observations or prima facie findings, and (4) final orders/convictions. \
  Use language like "the prosecution alleged", "the court held", "the petitioner contended". \
  Never write a charge or prosecution theory as if it were a concluded judicial finding.
- ROLE & OBLIGATION ACCURACY: Be extremely precise about WHO bears an obligation and WHO receives a benefit (e.g., who pays a fee, who provides a certificate). Do not reverse roles.
- LIMITATIONS & EXCEPTIONS: Accurately capture if a clause (like indemnity or confidentiality) is subject to a limitation of liability. Do not state obligations remain 'full' if they are capped or limited.
- TERMINATION: Clearly distinguish between unilateral (one-party) and mutual termination rights.
- ARITHMETIC PROHIBITION: Never compute, derive, multiply, or extrapolate numeric values. Only state figures that appear VERBATIM in the source text. If the text says "₹1 lakh per acre" and separately mentions a land area, do NOT multiply them — quote each figure as it appears. A derived number is a hallucination even if the arithmetic is correct.
- STATUTE INTERPRETATION: When a statute section number appears, only describe what the text explicitly says about it. Do NOT apply external legal knowledge to explain what that section means or implies.
- RELIEF SEQUENCING: When capturing suit prayers or reliefs, preserve the exact order and primacy as stated in the plaint/petition. Do not reorder reliefs by perceived importance.
- PROCEDURAL ORDER RATIONALE (CRITICAL): For every procedural order (committal to Sessions Court, \
  framing/discharge of charges, remand, stay, transfer), explicitly record (a) which court made \
  the order, (b) under which section, and (c) the court's stated reason. Never record a \
  procedural step without its reason.
- MULTI-STAGE LITIGATION (CRITICAL): If the case has multiple stages (Trial Court → High Court → \
  Supreme Court, or First SLP → Remand → Second Appeal), label and separate each stage. Record \
  what each court decided and why. Never blend outcomes from different stages.
- PAGE TITLES: You MUST append the inferred Document Type in parentheses to EVERY page title. \
  CASE-SPECIFIC TITLES (CRITICAL): For court judgments, the KNOWN TOPICS list will contain \
  some topics prefixed with a case identifier (e.g. "Facts – Yuvraj Kanther") and some without \
  (e.g. "Section 319 CrPC"). Preserve these prefixes exactly when generating page titles. \
  If a case-specific topic (facts, holding, charges, procedural history, contentions, relief) \
  in the KNOWN TOPICS list lacks a prefix, add the case identifier from the document text. \
  Shared legal concept pages (statutes, precedents, doctrines) must NOT have a case prefix. \
  Example: "Facts – Yuvraj Kanther ({doc_type})", "Section 319 CrPC ({doc_type})". \
  For contracts: "Topic Name ({doc_type})".
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


# ---------------------------------------------------------------------------
# Progress helpers — abstract over PROGRESS_STORE vs PostgreSQL (S5)
# ---------------------------------------------------------------------------
def _get_session_progress(session_id: str) -> dict:
    """Return the current progress dict for this session."""
    if config.USE_DATABASE:
        return _db.get_progress(session_id) or {"rag": {}, "wiki": {}}
    return config.PROGRESS_STORE.setdefault(session_id, {"rag": {}, "wiki": {}})


def _save_session_progress(session_id: str, progress: dict) -> None:
    """Persist the progress dict (no-op for in-memory store since it mutates by reference)."""
    if config.USE_DATABASE:
        _db.set_progress(session_id, progress)
    else:
        config.PROGRESS_STORE[session_id] = progress


def _update_wiki_progress(session_id: str, wiki_update: dict) -> None:
    """Merge wiki_update into the 'wiki' sub-dict and persist."""
    progress = _get_session_progress(session_id)
    w = progress.setdefault("wiki", {})
    w.update(wiki_update)
    _save_session_progress(session_id, progress)


def _update_doc_step(session_id: str, doc_name: str, status: str, step: str = "") -> None:
    """Update a single document's status in the progress docs_list."""
    progress = _get_session_progress(session_id)
    bare = os.path.basename(doc_name)
    for doc in progress.get("docs_list", []):
        if doc.get("name") == bare:
            doc["status"] = status
            doc["step"] = step
            break
    _save_session_progress(session_id, progress)


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

    # Signal: file has been read, starting synthesis
    _update_doc_step(session_id, doc_name, "synthesizing")

    total_contradictions = 0

    if len(text) <= _SINGLE_CALL_THRESHOLD:
        # --- Short document: single LLM call ---
        _update_wiki_progress(session_id, {"current": 0, "total": 1,
                                            "message": f"Processing {doc_name}..."})
        _update_doc_step(session_id, doc_name, "synthesizing", "1/1")
        parsed = _ingest_single_call(text, doc_name)
        _update_doc_step(session_id, doc_name, "merging")
        _update_wiki_progress(session_id, {"current": 1, "total": 1,
                                            "message": f"Processing {doc_name}..."})
        total_pages, total_rels, total_contradictions = _atomic_merge(session_id, parsed, doc_name)
    else:
        # --- Long document: two-phase approach ---
        segments = _split_segments(text)
        total_steps = 1 + len(segments)
        _update_wiki_progress(session_id, {"current": 0, "total": total_steps,
                                            "message": f"Overview pass for {doc_name}..."})
        _update_doc_step(session_id, doc_name, "synthesizing", f"0/{len(segments)}")

        overview_text = text[:6000] + "\n\n[...]\n\n" + text[-3000:]
        doc_type, topics, overview_parsed = _ingest_overview(overview_text, doc_name)
        _update_wiki_progress(session_id, {"current": 1, "total": total_steps,
                                            "message": f"Overview pass for {doc_name}..."})

        total_pages, total_rels, tc = _atomic_merge(session_id, overview_parsed, doc_name)
        total_contradictions += tc

        completed_segments = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.WIKI_MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(_ingest_detail_segment, seg, topics, doc_name, doc_type): i
                for i, seg in enumerate(segments)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                completed_segments += 1
                msg = f"Detail pass completed {completed_segments}/{len(segments)} for {doc_name}"
                logger.info("  %s", msg)
                _update_doc_step(session_id, doc_name, "synthesizing",
                                 f"{completed_segments}/{len(segments)}")
                _update_wiki_progress(session_id, {"current": 1 + completed_segments,
                                                    "total": total_steps, "message": msg})
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
    _log_event(session_id, "INGEST",
               f"Doc: {doc_name} | Pages updated: {total_pages} | Contradictions found: {total_contradictions}")

    # S3: compact any pages that have grown beyond the quality thresholds
    try:
        compacted = run_compaction(session_id)
        if compacted > 0:
            _update_wiki_progress(session_id, {"message": f"Complete: {total_pages} pages extracted, {compacted} compacted."})
        else:
            _update_wiki_progress(session_id, {"message": f"Complete: {total_pages} pages extracted."})
    except Exception as _ce:
        logger.error("Post-ingest compaction failed: %s", _ce)
        _update_wiki_progress(session_id, {"message": f"Complete: {total_pages} pages extracted."})

    return {"pages_updated": total_pages, "relations": total_rels}


def _ingest_single_call(text: str, doc_name: str) -> dict:
    """Process a short document in one LLM call."""
    prompt = INGEST_PROMPT_TEMPLATE.format(text=text, doc_name=doc_name)
    try:
        raw, _ = llm.ask(prompt, pipeline="wiki", max_tokens=config.MAX_TOKENS_INGEST_SINGLE)
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
        raw, _ = llm.ask(
            prompt,
            pipeline="wiki",
            max_tokens=config.MAX_TOKENS_INGEST_OVERVIEW,
        )
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

    # NOTE: Must return all three values — (doc_type, topics, parsed_pages).
    # The caller unpacks: doc_type, topics, overview_parsed = _ingest_overview(...)
    return doc_type, topics, {"pages": doc_pages, "relations": []}


def _ingest_detail_segment(text: str, topics: list[str], doc_name: str, doc_type: str) -> dict:
    """Phase 2: extract detailed pages from a segment with known topic context."""
    topics_str = ", ".join(topics) if topics else "None identified yet"
    prompt = DETAIL_PROMPT_TEMPLATE.format(text=text, topics=topics_str, doc_name=doc_name, doc_type=doc_type)
    try:
        raw, _ = llm.ask(
            prompt,
            pipeline="wiki",
            max_tokens=config.MAX_TOKENS_INGEST_DETAIL,
        )
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
    """Thread-safe merge: dispatches to DB or file path based on config.USE_DATABASE."""
    if config.USE_DATABASE:
        return _atomic_merge_db(session_id, new_data, doc_name)
    return _atomic_merge_file(session_id, new_data, doc_name)


def _atomic_merge_file(session_id: str, new_data: dict, doc_name: str = "Unknown") -> tuple[int, int, int]:
    """File-based merge: load index.json → merge in Python → save index.json, under lock."""
    lock = _get_session_lock(session_id)
    with lock:
        existing = _load_index_file(session_id)
        merged, pages_updated, new_rels, contradictions = _merge_wiki(existing, new_data, doc_name)
        _save_index_file(session_id, merged)
        for c in contradictions:
            _log_event(
                session_id,
                "CONTRADICTION",
                f"Page: {c['title']} | Claim: {c['claim']} | Source A: {c.get('val_a')} | Source B: {c.get('val_b')}",
            )
    return pages_updated, new_rels, len(contradictions)


def _atomic_merge_db(session_id: str, new_data: dict, doc_name: str = "Unknown") -> tuple[int, int, int]:
    """DB-backed merge: per-row upserts under a session lock.

    Keeps the Python lock to serialize the cross-reference pass (Phase 4/S2 will
    replace the O(N²) loop with a single PostgreSQL FTS query and remove it).
    """
    lock = _get_session_lock(session_id)
    # Collect (title, embed_text) pairs here; embed OUTSIDE the lock so HTTP
    # calls don't block other ingest threads waiting on the session lock.
    pages_to_embed: list[tuple[str, str]] = []

    with lock:
        pages_updated = 0
        new_rels_count = 0
        contradictions_found: list[dict] = []

        new_pages = new_data.get("pages", {})
        new_relations = new_data.get("relations", [])

        # -- Merge pages --
        for title, new_value in new_pages.items():
            if isinstance(new_value, str):
                new_content, new_summary, new_quotes = new_value, "", []
            else:
                new_content = new_value.get("content", "")
                new_summary = new_value.get("summary", "")
                new_quotes = new_value.get("quotes", [])

            if new_quotes:
                quote_text = "\n\n**Supporting Quotes:**\n" + "\n".join(f"> {q}" for q in new_quotes)
                new_content += quote_text

            existing = _db.get_page(session_id, title)

            if existing:
                existing_content = existing["content"]
                existing_summary = existing["summary"]
                contradiction_flagged = existing.get("contradiction_flagged", False)
                variants = existing.get("variants")

                # S4 + C3: Replace per-append LLM contradiction check with a fast NER
                # pre-filter.  If structural values (amounts, dates, percentages) differ,
                # flag the page and record a variant snapshot so the compaction pass
                # (S3) can detect and surface the contradiction during re-synthesis.
                # This eliminates hundreds of thousands of LLM calls at scale while
                # preserving all the raw material the compaction LLM needs.
                if (len(new_content) > 200 and len(existing_content) > 200
                        and _has_structural_conflict(existing_content, new_content)):
                    contradiction_flagged = True
                    from datetime import datetime
                    if not variants:
                        variants = [{"source": "Previous", "value": existing_content,
                                     "date_ingested": datetime.now().isoformat()}]
                    variants.append({"source": doc_name, "value": new_content,
                                     "date_ingested": datetime.now().isoformat()})
                    contradictions_found.append({
                        "title": title, "claim": None,
                        "val_a": None, "val_b": None, "doc": doc_name,
                    })

                # Strip session-UUID prefix and extension for a readable label.
                _raw_label = doc_name
                import re as _re
                _raw_label = _re.sub(r'^[0-9a-f]{8}-[0-9a-f-]{27}_', '', _raw_label)
                _raw_label = _re.sub(r'\.(pdf|PDF)$', '', _raw_label)
                merged_content = (
                    existing_content
                    + f"\n\n---\n*[From: {_raw_label}]*\n\n"
                    + new_content
                )
                merged_summary = new_summary if new_summary else existing_summary
                _db.upsert_page(session_id, title, merged_content, merged_summary, doc_name,
                                contradiction_flagged, variants)
                # Use the freshest summary for the embedding
                embed_text = (new_summary or existing_summary or new_content)[:400]
            else:
                _db.upsert_page(session_id, title, new_content, new_summary, doc_name, False, None)
                embed_text = (new_summary or new_content)[:400]

            pages_to_embed.append((title, embed_text))
            pages_updated += 1

        # -- C7: Persist document-level metadata extracted at ingest time --
        metadata = new_data.get("metadata")
        if metadata and isinstance(metadata, dict):
            try:
                _db.upsert_metadata(session_id, doc_name, metadata)
            except Exception as _me:
                logger.error("Metadata upsert failed for '%s': %s", doc_name, _me)

        # -- Merge explicit relations --
        for rel in new_relations:
            _db.upsert_relation(
                session_id, rel.get("from", ""), rel.get("to", ""), rel.get("label", "")
            )
            new_rels_count += 1

        # -- S2: FTS cross-reference — O(log N) per new page instead of O(N²) --
        # Only process pages in the current batch.  Pairs that were already in
        # the wiki were processed in a prior merge; skipping them is safe because
        # _db.bulk_upsert_relations uses ON CONFLICT DO NOTHING.
        #
        # Direction A — existing page content mentions new title:
        #   Find pages whose content_tsv matches the new title's tokens (GIN index).
        # Direction B — new page content mentions existing titles:
        #   Python substring check against the title list only (no content fetch).
        existing_titles = _db.get_page_titles(session_id)
        existing_title_set = set(existing_titles)
        mention_rels: list[tuple[str, str, str]] = []
        for new_title, new_val in new_pages.items():
            new_content_for_xref = (
                new_val.get("content", "") if isinstance(new_val, dict) else str(new_val)
            )
            # Direction A: who already mentions this new title?
            try:
                mentioning = _db.find_pages_mentioning_title(session_id, new_title)
                for existing_title in mentioning:
                    mention_rels.append((existing_title, new_title, "mentions"))
            except Exception as _xref_err:
                logger.warning("FTS cross-ref failed for '%s': %s", new_title, _xref_err)
            # Direction B: which existing titles does the new page mention?
            for existing_title in existing_title_set:
                if existing_title != new_title and existing_title in new_content_for_xref:
                    mention_rels.append((new_title, existing_title, "mentions"))
        if mention_rels:
            _db.bulk_upsert_relations(session_id, mention_rels)

        for c in contradictions_found:
            _log_event(
                session_id,
                "CONTRADICTION",
                f"Page: {c['title']} | Claim: {c['claim']} | Source A: {c.get('val_a')} | Source B: {c.get('val_b')}",
            )

    # -- Embed pages OUTSIDE the lock (HTTP calls should not hold the session lock) --
    _embed_pages_batch(session_id, pages_to_embed)

    return pages_updated, new_rels_count, len(contradictions_found)


# ---------------------------------------------------------------------------
# Embedding helper (Phase 3) — called OUTSIDE the session lock
# ---------------------------------------------------------------------------
def _embed_pages_batch(session_id: str, pages_to_embed: list[tuple[str, str]]) -> None:
    """Embed page summaries and store in page_embeddings table.

    Called AFTER the session lock is released so embedding HTTP calls don't
    block other ingest threads.  Failures are logged and swallowed — vector
    search falls back to BM25 gracefully for un-embedded pages.

    Args:
        pages_to_embed: list of (title, text_to_embed) where text_to_embed is
                        the page summary, or the first 400 chars of content
                        when no summary is available.
    """
    if not config.USE_DATABASE or not pages_to_embed:
        return
    try:
        from services import embedder as _embedder
        texts = [text for _, text in pages_to_embed]
        embeddings = _embedder.embed_batch(texts, is_query=False)
        for (title, _), embedding in zip(pages_to_embed, embeddings):
            _db.upsert_embedding(session_id, title, embedding)
        logger.info(
            "Embedded %d pages for session %s", len(pages_to_embed), session_id
        )
    except Exception as e:
        logger.error(
            "Embedding batch failed for session %s (%d pages skipped): %s",
            session_id, len(pages_to_embed), e,
        )


# ---------------------------------------------------------------------------
# S3 — Page compaction / re-synthesis (Phase 4)
# ---------------------------------------------------------------------------

COMPACTION_PROMPT_TEMPLATE = """\
You are a legal wiki editor. The following are {n} versions of the same topic \
from different legal documents. Re-synthesise them into one coherent wiki page, \
preserving ALL distinct facts, noting genuine contradictions explicitly with source \
attribution, and tagging claims with their source document.

RULES:
- Preserve every exact figure, date, and amount VERBATIM — do not paraphrase numeric values.
- Keep the best supporting quotes from across all versions.
- Note contradictions explicitly: "Document A states X; Document B states Y."
- Do NOT resolve contradictions — surface them clearly for human review.
- The resulting page should read like a well-structured wiki article.
- Apply all the usual hallucination-prevention rules: do not invent citations or dates.

TOPIC: {title}

VERSIONS:
{variants_text}

OUTPUT FORMAT — respond with valid JSON only, no markdown fences:
{{
  "content": "Re-synthesised wiki page content...",
  "summary": "One-line summary.",
  "quotes": ["Exact verbatim quote 1", "Exact verbatim quote 2"],
  "contradictions": [
    {{"claim": "...", "value_a": "...", "source_a": "...", "value_b": "...", "source_b": "..."}}
  ]
}}"""


def _compact_page(session_id: str, title: str, page_data: dict) -> None:
    """Re-synthesise a bloated page.  Called outside the session lock (S3, Phase 4).

    Uses the variants list when available; otherwise splits content on the
    section separator.  After compaction:
    - append_count resets to 0
    - variants column is cleared (they're now merged into content)
    - The page is re-embedded with the fresh summary
    - Detected contradictions are written to the contradictions table (S4)
    """
    content = page_data.get("content", "")
    variants = page_data.get("variants") or []

    if variants:
        variants_text = "\n\n---\n".join(
            f"**Source: {v.get('source', 'Unknown')}**\n{v.get('value', '')}"
            for v in variants
        )
        n = len(variants)
    else:
        parts = content.split("\n\n---\n")
        variants_text = "\n\n---\n".join(
            f"**Version {i + 1}**\n{p}" for i, p in enumerate(parts)
        )
        n = len(parts)

    prompt = COMPACTION_PROMPT_TEMPLATE.format(
        title=title, variants_text=variants_text, n=n
    )
    try:
        raw, _ = llm.ask(prompt, pipeline="wiki", max_tokens=config.MAX_TOKENS_COMPACTION)
    except RuntimeError as e:
        logger.error("Compaction LLM call failed for '%s': %s", title, e)
        return

    parsed = _parse_json_safe(raw)
    if parsed is None:
        parsed = _repair_json(raw)

    new_content = parsed.get("content", content)
    new_summary = parsed.get("summary", "")
    new_quotes  = parsed.get("quotes", [])
    detected_contradictions = parsed.get("contradictions", [])

    if new_quotes:
        quote_text = "\n\n**Supporting Quotes:**\n" + "\n".join(f"> {q}" for q in new_quotes)
        new_content += quote_text

    contradiction_flagged = bool(detected_contradictions)

    _db.reset_page_after_compaction(session_id, title, new_content, new_summary, contradiction_flagged)

    # Store structured contradictions (S4)
    for c in detected_contradictions:
        try:
            _db.upsert_contradiction(
                session_id, title,
                c.get("claim"), c.get("value_a"), c.get("source_a"),
                c.get("value_b"), c.get("source_b"),
            )
            _log_event(
                session_id, "CONTRADICTION",
                f"Page: {title} | Claim: {c.get('claim')} | "
                f"{c.get('source_a')} vs {c.get('source_b')}",
            )
        except Exception as _ce:
            logger.error("Failed to store contradiction for '%s': %s", title, _ce)

    # Re-embed with fresh summary
    embed_text = (new_summary or new_content[:400])
    _embed_pages_batch(session_id, [(title, embed_text)])

    logger.info("Compacted page '%s' (%d → 1 version, contradictions=%d)",
                title, n, len(detected_contradictions))


def run_compaction(session_id: str) -> int:
    """Find pages due for compaction and re-synthesise them.

    Called at the end of every ingest so the wiki quality stays high as
    documents accumulate.  No-op in file mode (compaction is DB-only).
    Returns the number of pages successfully compacted.
    """
    if not config.USE_DATABASE:
        return 0

    due = _db.find_pages_due_for_compaction(
        session_id,
        config.COMPACTION_APPEND_THRESHOLD,
        config.COMPACTION_CHAR_THRESHOLD,
    )
    if not due:
        return 0

    logger.info("Compaction: %d pages due for session %s", len(due), session_id)
    compacted = 0
    for page_data in due:
        try:
            _compact_page(session_id, page_data["title"], dict(page_data))
            compacted += 1
        except Exception as e:
            logger.error("Compaction failed for page '%s': %s", page_data["title"], e)

    _log_event(session_id, "COMPACTION", f"Compacted {compacted}/{len(due)} pages")
    return compacted


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

Pick the 15-25 MOST RELEVANT pages for answering this question. 
CRITICAL RULES:
1. If the question mentions specific document types, names, or categories (e.g., "NDAs", "Joint Venture Agreements"), ONLY select pages that match those constraints.
2. If the question is GENERAL (e.g., "List all obligations", "Compare the terms"), you MUST select relevant pages from ACROSS MULTIPLE DIFFERENT DOCUMENTS to ensure a comprehensive, cross-document answer. Do not restrict your selection to just one document.
3. Return ONLY a JSON array of page titles, no explanation.

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
- Do NOT write the source file name as normal text. Instead, whenever you state a claim, append the EXACT Source File Name(s) provided in the text block header wrapped in double brackets like this: [[Source File Name]].
- If the wiki does not contain enough information, say so — do NOT make up facts.
- IMPORTANT: If your answer introduces new concepts, synthesis, or insights not already explicit in the wiki pages, \
extract them into new pages and relations so the wiki grows smarter.
- DO NOT append a "References", "Sources", or "Citations" list at the end of your answer. All citations MUST be inline ONLY.

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
    page_selection_usage: dict = {}

    # --- Step 1: Select relevant pages ---
    if file_pages:
        # File explicitly mentioned — force those pages in
        if len(pages) <= 20:
            # Small wiki: file pages first, then everything else
            other = [t for t in pages if t not in file_pages]
            selected_titles = file_pages + other
        else:
            # Large wiki: file pages + vector/LLM-selected supplementary pages
            llm_selected, page_selection_usage = _select_relevant_pages(pages_for_llm, question, session_id)
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
            selected_titles, page_selection_usage = _select_relevant_pages(pages_for_llm, question, session_id)

    # --- Step 2: Build context string from selected pages ---
    # Q: pages are cached prior answers — cap so they don't crowd out source content.
    # Regular pages: cap at MAX_PAGE_CONTEXT_CHARS to bound total prompt size.
    # Shared concept pages (no case prefix, multi-source) can grow large through merges;
    # capping ensures one bloated page doesn't consume half the context window.
    _QPAGE_CAP = config.MAX_QPAGE_CONTEXT_CHARS
    _PAGE_CAP  = config.MAX_PAGE_CONTEXT_CHARS

    wiki_parts = []
    for title in selected_titles:
        if title in pages:
            page = pages[title]
            content = page.get("content", "") if isinstance(page, dict) else page

            if isinstance(page, dict) and page.get("contradiction_flagged"):
                content = "[WARNING: This page contains conflicting claims. Surface the conflict explicitly in your answer. Do not resolve it.]\n" + content

            if title.startswith("Q:") and len(content) > _QPAGE_CAP:
                content = content[:_QPAGE_CAP] + "\n[...truncated — cached answer summary only]"
            elif len(content) > _PAGE_CAP:
                content = content[:_PAGE_CAP] + "\n[...truncated]"

            wiki_parts.append(f"## {title}\n{content}\n")
    wiki_content = "\n".join(wiki_parts)

    return {
        "context": wiki_content,
        "selected_titles": selected_titles,
        "bm25_count": bm25_count,
        "page_selection_usage": page_selection_usage,
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


def generate_answer(question: str, wiki_content: str, selected_titles: list, session_id: str, bm25_count: int = 0, page_selection_usage: dict = None) -> dict:
    """Generate an answer using the provided wiki content."""
    index = _load_index(session_id)
    pages = index.get("pages", {})
    relations = index.get("relations", [])

    # Accumulate per-call token usage for this query
    token_breakdown: list[dict] = []

    # Attach page-selection usage if it was an LLM call (not fallback)
    if page_selection_usage and page_selection_usage.get("prompt_tokens"):
        token_breakdown.append({
            "call": "page_selection",
            "model": llm.active_model(fast=False),
            "prompt_tokens": page_selection_usage.get("prompt_tokens", 0),
            "completion_tokens": page_selection_usage.get("completion_tokens", 0),
            "total_tokens": page_selection_usage.get("prompt_tokens", 0) + page_selection_usage.get("completion_tokens", 0),
        })

    if not wiki_content:
        return {
            "answer": "The wiki is empty — no documents have been ingested yet.",
            "pages_used": [],
            "files_used": [],
            "relations": relations,
            "usage": {},
            "confidence_score": 0,
            "confidence_reason": "No context available.",
            "token_breakdown": token_breakdown,
            "token_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    from services.prompts import ANSWER_PROMPT
    prompt = ANSWER_PROMPT.format(context=wiki_content, question=question)

    usage = {}
    confidence_score = 75   # conservative default if extraction fails
    confidence_reason = "Default — could not parse confidence from reasoning block."

    try:
        raw_answer, usage = llm.ask(
            prompt,
            pipeline="wiki",
            max_tokens=config.MAX_TOKENS_ANSWER,
        )

        # --- Extract confidence from reasoning BEFORE stripping the block ---
        # The ANSWER_PROMPT instructs the model to append two structured lines
        # at the end of <reasoning>:
        #   CONFIDENCE_SCORE: [int]
        #   CONFIDENCE_REASON: [sentence]
        # Extracting here avoids a second LLM call for _evaluate_confidence().
        reasoning_match = re.search(
            r'(?i)<reasoning>(.*?)</reasoning>', raw_answer, flags=re.DOTALL
        )
        if reasoning_match:
            reasoning_text = reasoning_match.group(1)
            score_match = re.search(r'(?i)CONFIDENCE[_\s]*SCORE[^0-9]*(\d+)', reasoning_text)
            reason_match = re.search(
                r'(?i)CONFIDENCE[_\s]*REASON[^\w]*(.+?)(?:\n|$)', reasoning_text
            )
            if score_match:
                try:
                    confidence_score = min(100, max(0, int(score_match.group(1))))
                except ValueError:
                    pass
            if reason_match:
                confidence_reason = reason_match.group(1).strip()

        # Strip reasoning tags for the user-facing answer
        answer = re.sub(
            r'(?i)<reasoning>.*?</reasoning>', '', raw_answer, flags=re.DOTALL
        ).strip()

        # --- Fallback confidence when model skipped the reasoning block ---
        # A short "Not covered" answer means the context had nothing relevant —
        # confidence should be 0, not the generic 75 default.
        if confidence_score == 75 and confidence_reason.startswith("Default"):
            _not_covered = re.search(
                r'not covered|no information|not contain|no relevant|'
                r'does not contain|not found|not available',
                answer, flags=re.IGNORECASE
            )
            if _not_covered or len(answer) < 150:
                confidence_score = 0
                confidence_reason = "Model found no relevant context for this question."
            else:
                # Substantial answer but no reasoning block — model answered directly
                confidence_score = 72
                confidence_reason = "Model answered without reasoning block; score estimated."

    except RuntimeError as e:
        answer = f"⚠️ LLM error: {e}"
        confidence_score = 0
        confidence_reason = "LLM call failed."

    # Extract [Reference] from the answer
    referenced = re.findall(r"\[([^\]]+)\]", answer)
    
    # Get all canonical doc names from pages
    canonical_files = set()
    for title in pages.keys():
        match = re.search(r'\(([^)]+)\)\s*$', title.strip())
        if match:
            canonical_files.add(match.group(1))
            
    pages_used_dedup = []
    files_used = []
    seen = set()
    
    for t in referenced:
        if t not in seen:
            pages_used_dedup.append(t)
            seen.add(t)
            # Try to figure out which file this refers to
            for cfile in canonical_files:
                # Strip UUID for comparison
                stripped_cfile = re.sub(r'^[a-f0-9-]{36}_', '', cfile)
                if stripped_cfile.lower() in t.lower() or cfile.lower() in t.lower():
                    if cfile not in files_used:
                        files_used.append(cfile)

    # Confidence is already extracted from the reasoning block above —
    # no second LLM call needed.
    confidence = {"score": confidence_score, "reason": confidence_reason}

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

    # Record answer-generation call in the breakdown
    token_breakdown.append({
        "call": "answer_generation",
        "model": llm.active_model(fast=False),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
    })

    # Aggregate totals across all calls in this query
    token_total = {
        "prompt_tokens":     sum(e["prompt_tokens"]     for e in token_breakdown),
        "completion_tokens": sum(e["completion_tokens"] for e in token_breakdown),
        "total_tokens":      sum(e["total_tokens"]      for e in token_breakdown),
    }

    # Log per-call breakdown to session log
    breakdown_str = " | ".join(
        f"{e['call']} ({e['model']}): {e['total_tokens']} tokens"
        for e in token_breakdown
    )
    _log_event(session_id, "TOKEN_USAGE", f"Total: {token_total['total_tokens']} | {breakdown_str}")

    return {
        "answer": answer,
        "pages_used": pages_used_dedup,
        "files_used": files_used,
        "relations": relations,
        "usage": usage,
        "confidence_score": confidence["score"],
        "confidence_reason": confidence["reason"],
        "token_breakdown": token_breakdown,
        "token_total": token_total,
    }


def _keyword_fallback_pages(pages: dict, question: str, n: int = 25) -> list[str]:
    """Score pages by keyword overlap with the question — used when LLM selection fails.

    Much better than first-N: ensures pages matching the question's subject matter
    are included regardless of insertion order in the wiki.
    """
    # Tokenise question; drop words shorter than 4 chars (common stop words)
    q_words = {w for w in re.sub(r'[^\w\s]', '', question.lower()).split() if len(w) >= 4}
    scored = []
    for title, page in pages.items():
        summary = page.get("summary", "") if isinstance(page, dict) else ""
        combined = (title + " " + summary).lower()
        score = sum(1 for w in q_words if w in combined)
        scored.append((score, title))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:n]]


def _select_relevant_pages(
    pages: dict, question: str, session_id: str | None = None
) -> tuple[list[str], dict]:
    """Select the most relevant pages for a question.

    Priority order:
      1. pgvector cosine similarity search (DB mode only, Phase 3) — 0 LLM calls, ~5ms
      2. BM25 pre-filter + LLM selection (fallback or file mode)
      3. BM25 keyword-only (if LLM selection fails)

    Returns (selected_titles, usage_dict).
    usage_dict is empty when vector search is used (no LLM call made).
    """

    # ------------------------------------------------------------------ #
    # Path 1 — hybrid: pgvector cosine + BM25 keyword supplement         #
    #                                                                     #
    # Vector search alone can miss pages where the query terms have low  #
    # semantic overlap with the embedding but strong keyword overlap      #
    # (e.g. "committed to Sessions Court", "Yogesh Kumar preserved").    #
    # Merging BM25 results fills that gap without an LLM call.           #
    # ------------------------------------------------------------------ #
    if config.USE_DATABASE and session_id:
        try:
            emb_count = _db.count_embeddings(session_id)
            if emb_count > 0:
                from services import embedder as _embedder
                q_embedding = _embedder.embed(question, is_query=True)
                vector_titles = _db.search_similar_pages(
                    session_id, q_embedding, limit=config.VECTOR_SEARCH_TOP_K
                )
                # Validate titles against the in-memory pages dict (guards against
                # stale embeddings pointing at deleted pages)
                valid_vector = [t for t in vector_titles if t in pages]

                # BM25 supplement: add keyword-matched pages that vector missed.
                # Vector results come first (better semantic rank); BM25 pages
                # are appended only if not already present.
                bm25_supplement = _keyword_fallback_pages(
                    pages, question, n=config.HYBRID_BM25_SUPPLEMENT_N
                )
                seen: set[str] = set(valid_vector)
                hybrid: list[str] = list(valid_vector)
                bm25_added = 0
                for t in bm25_supplement:
                    if t not in seen:
                        hybrid.append(t)
                        seen.add(t)
                        bm25_added += 1

                if hybrid:
                    logger.info(
                        "Page selection: %d pages via hybrid "
                        "(vector=%d, bm25_added=%d, embeddings_in_db=%d)",
                        len(hybrid), len(valid_vector), bm25_added, emb_count,
                    )
                    return hybrid, {}

                logger.warning(
                    "pgvector returned %d titles but none matched current pages — "
                    "falling back to BM25+LLM", len(vector_titles)
                )
        except Exception as e:
            logger.error("pgvector page selection failed: %s — falling back to BM25+LLM", e)

    # ------------------------------------------------------------------ #
    # Path 2 — BM25 pre-filter + LLM selection                           #
    # ------------------------------------------------------------------ #
    # BM25 pre-filter: narrow to top candidates before LLM selection.
    # At 1000+ pages the full index exceeds model context; pre-filtering keeps the
    # selection prompt to ~10k tokens regardless of wiki size.
    candidate_titles = _keyword_fallback_pages(pages, question, n=config.PAGE_SELECTION_PREFILTER_N)
    candidate_pages = {t: pages[t] for t in candidate_titles if t in pages}

    # Build compact index: "Title: summary"
    index_lines = []
    for title, page in candidate_pages.items():
        summary = page.get("summary", "") if isinstance(page, dict) else ""
        line = f"- {title}: {summary}" if summary else f"- {title}"
        index_lines.append(line)
    page_index = "\n".join(index_lines)

    prompt = PAGE_SELECT_PROMPT.format(page_index=page_index, question=question)

    try:
        raw, usage = llm.ask(
            prompt,
            pipeline="wiki",
            max_tokens=config.MAX_TOKENS_PAGE_SELECTION,
        )
        parsed = _parse_json_safe(raw)
        if isinstance(parsed, list):
            valid = [t for t in parsed if t in candidate_pages]
            if valid:
                logger.info(
                    "Page selection: %d pages selected by LLM (from %d BM25 candidates)",
                    len(valid), len(candidate_pages),
                )
                return valid, usage
        logger.warning("Page selection: LLM returned unparseable result — using keyword fallback")
    except (RuntimeError, Exception) as e:
        logger.error("Page selection LLM call failed: %s — using keyword fallback", e)

    # ------------------------------------------------------------------ #
    # Path 3 — BM25 keyword-only fallback                                 #
    # ------------------------------------------------------------------ #
    fallback = _keyword_fallback_pages(pages, question)
    logger.info("Page selection fallback: %d pages via keyword scoring", len(fallback))
    return fallback, {}


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
