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

import hashlib
import json
import os
import re
import logging
import threading
import concurrent.futures
from functools import lru_cache

import config
from services import llm
from services import tracing
from services.reader import read_file as _read_file

if config.USE_DATABASE:
    from services import db as _db

logger = logging.getLogger(__name__)


def _active_wiki_id() -> str:
    """The wiki_id every legacy-table db.py call in this module scopes to.

    wiki.py runs mostly in background ingest threads (executor.submit), not
    inside a Flask request, so it reads the live active-wiki pointer directly
    rather than through app.py's request-bound current_wiki_id() — same
    reasoning _persist_structured already uses for the backbone tables.
    """
    from services import wikis
    return wikis.active_wiki_id()

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
    wiki_id = _active_wiki_id()
    json_path = _index_path(session_id)
    if os.path.exists(json_path) and _db.count_pages(wiki_id, session_id) == 0:
        logger.info("Auto-migrating session %s from index.json to PostgreSQL", session_id)
        _db.migrate_from_json(wiki_id, session_id, json_path)
        os.rename(json_path, json_path + ".migrated")

    pages = _db.get_pages(wiki_id, session_id)
    relations = _db.get_relations(wiki_id, session_id)
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
# ---------------------------------------------------------------------------
# Auto-prefix: ensure every document-specific page has a unique identifier
# ---------------------------------------------------------------------------
# Doc types whose pages are document-specific and MUST be prefixed
_CONTRACT_DOC_TYPES = re.compile(
    r'\b(?:Master Services?\s+Agreement|Services?\s+Agreement|Service Level Agreement|'
    r'Professional Services Agreement|NDA|Non.?Disclosure|Shareholder.?s? Agreement|'
    r'Joint Venture|Share Purchase|Subscription Agreement|'
    r'Master Services?\s+Agreement Amendment|Employment Agreement|'
    r'Consulting Agreement|License Agreement|Supply Agreement)\b',
    re.IGNORECASE,
)
# Shared concept pages that should NOT be prefixed (statutes, legal doctrines)
_SHARED_PAGE_PATTERNS = re.compile(
    r'^(?:Section\s+\d|Article\s+\d|Indian\s+Arbitration|'
    r'Arbitration\s+Act|Companies\s+Act|SEBI|RBI|GDPR|'
    r'IT\s+Act|Contract\s+Act|Transfer\s+of\s+Property|'
    r'Code\s+of\s+Civil\s+Procedure|CPC|CrPC|IPC)',
    re.IGNORECASE,
)
# A bare doc-type abbreviation ("Equity Split – JVA", "Notices – NDA") is NOT a
# real per-document prefix — it's identical for every document of that type and
# will silently collide across documents in the DB (each ingest overwrites the
# previous one's source_doc + loses that document's own facts). Only a prefix
# like "– JVA-HeliosAether" or "– SA-Meridian" actually disambiguates.
_BARE_TYPE_SUFFIX = re.compile(r'^(?:JVA|NDA|SA|SHA|LO|CCD|J|TBJ)\.?$', re.IGNORECASE)


def _make_doc_identifier(doc_name: str) -> str:
    """Derive a short identifier from a filename for page-title prefixing.

    "2109224e_Service Agreement 1_redacted.pdf" → "SA1"
    "Legal_AI_Tool_NDA_3_Redacted.pdf" → "NDA3"
    """
    # Strip UUID prefix and extension
    clean = re.sub(r'^[a-f0-9-]{36}_', '', doc_name)
    clean = re.sub(r'_redacted\.pdf$', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\.(pdf|txt|docx)$', '', clean, flags=re.IGNORECASE)
    # Replace separators
    clean = clean.replace('_', ' ').replace('-', ' ').strip()

    # Try to extract type + number: "Service Agreement 1" → "SA1"
    m = re.search(r'(Service\s+Agreement|Shareholder\s+Agreement|Joint\s+Venture(?:\s+Agreement)?|'
                  r'NDA|Legal\s+Opinion|Court\s+Case(?:\s+Document)?|Judgment|'
                  r'Tata\s+Brand\s+Judgment)\s*(\d+)',
                  clean, re.IGNORECASE)
    if m:
        type_str = m.group(1).strip()
        num = m.group(2)
        abbrevs = {
            'service agreement': 'SA', 'shareholder agreement': 'SHA',
            'joint venture agreement': 'JVA', 'joint venture': 'JVA',
            'nda': 'NDA', 'legal opinion': 'LO',
            'court case document': 'CCD', 'court case': 'CCD',
            'judgment': 'J', 'tata brand judgment': 'TBJ',
        }
        abbr = abbrevs.get(type_str.lower(), type_str[:3].upper())
        return f"{abbr}{num}"

    # Try "Test_SA_01" pattern
    m2 = re.search(r'Test\s+(SA|SHA|JVA|NDA|CCD|LO|Judgment)\s*(\d+)', clean, re.IGNORECASE)
    if m2:
        return f"Test-{m2.group(1).upper()}{m2.group(2)}"

    # Fallback: use first 15 chars
    fallback = clean[:15].strip()
    return fallback if fallback else "Doc"


# Ordered keyword → canonical-family rules. The ingest LLM's `doc_type` is free
# text ("Non-Disclosure Agreement", "NDA", "Master Services Agreement", "Court
# Judgment", ...), so we normalize to a small controlled vocabulary that scope
# resolution (Phase 2) and metadata-filtered vector search (Phase 1) can group
# and filter on. Rules are checked in order; the FIRST whose keyword appears in
# the lowercased doc_type wins — so more specific families (e.g. "master services
# agreement" → "Service Agreement" via the "service" keyword) must not be
# shadowed by a broader rule listed earlier. Kept deliberately coarse: family is
# for grouping ("all NDAs"), not fine-grained typing.
_DOC_FAMILY_RULES = (
    ("non-disclosure", "NDA"),
    ("nondisclosure", "NDA"),
    ("nda", "NDA"),
    ("shareholder", "Shareholder Agreement"),
    ("joint venture", "Joint Venture Agreement"),
    ("jva", "Joint Venture Agreement"),
    ("share purchase", "Share Purchase Agreement"),
    ("subscription agreement", "Subscription Agreement"),
    ("employment", "Employment Agreement"),
    ("consulting", "Consulting Agreement"),
    ("license", "License Agreement"),
    ("licence", "License Agreement"),
    ("supply", "Supply Agreement"),
    ("service level", "Service Level Agreement"),
    ("service", "Service Agreement"),   # covers "Master Services Agreement" too
    ("legal opinion", "Legal Opinion"),
    ("opinion", "Legal Opinion"),
    ("judgment", "Court Judgment"),
    ("judgement", "Court Judgment"),
    ("court case", "Court Judgment"),
    ("court", "Court Judgment"),
    ("pleading", "Pleading"),
    ("petition", "Pleading"),
    ("plaint", "Pleading"),
    ("suit", "Court Judgment"),
    ("lawsuit", "Court Judgment"),
)


def _normalize_doc_family(doc_type: str | None) -> str | None:
    """Collapse the LLM's free-text doc_type to a canonical family label.

    Returns None when doc_type is empty or matches no known family rule (callers
    treat a None family as "ungrouped" — still retrievable, just not part of a
    family-scoped query).
    """
    if not doc_type:
        return None
    dt = doc_type.lower()
    for keyword, family in _DOC_FAMILY_RULES:
        if keyword in dt:
            return family
    return None


def _auto_prefix_title(title: str, doc_id: str) -> str:
    """Add document identifier prefix to unprefixed contract/agreement pages.

    Pages that already have a genuine per-document ' – ' prefix (e.g. "SA-Meridian")
    or that are shared legal concepts are left unchanged. Only pages whose doc-type
    parenthetical matches a contract type get prefixed. A bare doc-type abbreviation
    used as a would-be prefix ("Equity Split – JVA") does NOT count as prefixed —
    it's substituted with the real doc_id so it stops colliding across documents.
    """
    if not doc_id or doc_id == "Doc":
        return title

    pre_paren = title.split('(')[0]
    dash_idx = pre_paren.rfind(' – ')
    if dash_idx >= 0:
        base_title = pre_paren[:dash_idx].strip()
        suffix = pre_paren[dash_idx + 3:].strip()
        if _BARE_TYPE_SUFFIX.match(suffix) and not _SHARED_PAGE_PATTERNS.match(base_title):
            rest = title[len(pre_paren):]  # preserve any trailing "(...)" untouched
            return f"{base_title} – {doc_id}{rest}"
        # Genuine per-document prefix already present (e.g. "– JVA-HeliosAether")
        return title

    # Check if this is a contract-type page
    paren_match = re.search(r'\(([^)]+)\)\s*$', title)
    if not paren_match:
        return title
    doc_type = paren_match.group(1)
    if not _CONTRACT_DOC_TYPES.search(doc_type):
        return title
    # Don't prefix shared legal concept pages
    base_title = title[:paren_match.start()].strip()
    if _SHARED_PAGE_PATTERNS.match(base_title):
        return title
    # Don't prefix "Document Overview" or "Q:" pages
    if base_title.startswith(('Document Overview', 'Q:')):
        return title
    # Add prefix
    return f"{base_title} – {doc_id} ({doc_type})"


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
    _doc_id = _make_doc_identifier(doc_name)

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

        # Auto-prefix unprefixed contract/agreement pages
        title = _auto_prefix_title(title, _doc_id)

        # Append quotes to the content
        if new_quotes:
            quote_text = "\n\n**Supporting Quotes:**\n" + "\n".join(f"> {q}" for q in new_quotes)
            new_content += quote_text

        # Guard against title collisions between DIFFERENT source documents
        # (see matching comment in _atomic_merge_db). Only doc-specific pages
        # (contract-type parenthetical) need this — shared concept/statute
        # pages are meant to merge across documents.
        if title in pages:
            _existing_doc = pages[title].get("source_doc") if isinstance(pages[title], dict) else None
            if _existing_doc not in (None, "", doc_name):
                _paren = re.search(r'\(([^)]+)\)\s*$', title)
                if _paren and _CONTRACT_DOC_TYPES.search(_paren.group(1)):
                    title = f"{title} #{_doc_id}"

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
- STAMP/CHALLAN CAPTURE (CRITICAL): If the document includes a stamp certificate, e-stamp, \
  or stamp duty challan anywhere in it — including as a cover-page watermark or header, not \
  just a separate attached page — you MUST create a dedicated wiki page for it capturing the \
  certificate/GRN number, date, purchaser/payer name, and amount verbatim. Also populate the \
  metadata field 'matter_reference' with this certificate/GRN number. Never omit this because \
  it looks like decorative letterhead rather than agreement text — it is frequently the only \
  reliable date and party-identification anchor in the document.
- SERVICE-LEVEL / CADENCE PRECISION (CRITICAL): For any recurring obligation — reporting \
  frequency, review or hygiene cadence, response times, SLA turnaround — quote the literal \
  cadence and its clause context verbatim (e.g. "Weekly reports every Monday", "changed and \
  managed once a month"). Do NOT compress these into vague paraphrase like "periodic reporting" \
  or "regular reviews" — the exact frequency is often the entire legal content of the clause.
- METADATA COMPLETENESS: Populate governing_law, jurisdiction, and parties whenever the \
  document states them anywhere in the text, even if some party names are redacted elsewhere — \
  describe what is identifiable (e.g. "Tata Sons Private Limited; counterparty name redacted") \
  rather than leaving the field null just because one side is unnamed.

PAGE TITLES: You MUST append the inferred Document Type in parentheses to EVERY page title. \
DOCUMENT-SPECIFIC PAGES (CRITICAL): Most pages describe provisions unique to THIS specific \
document and must NOT merge with pages from other documents of the same type. You MUST attach \
a SHORT DOCUMENT IDENTIFIER derived from the document. \
TITLE ORDER (STRICT — never deviate): "{{Topic}} – {{Document Identifier}} ({{Document Type}})". \
The topic ALWAYS comes first, then a dash, then the document identifier, then the type in \
parentheses — the identifier NEVER comes before the topic. \
Wrong: "SA1-Crayons – Term and Termination (Service Agreement)". \
Right: "Term and Termination – SA1-Crayons (Service Agreement)". \
  For court judgments: use first party's last name (e.g. "Yuvraj Kanther") \
  For contracts/agreements: use a short identifier from filename or parties that distinguishes \
  this document from others of the same type. Derive it from the counterparty name, the \
  filename, or a unique label (e.g. "SA1-Crayons" for Service Agreement 1 with Crayons, \
  "NDA-Acme" for an NDA with Acme Corp, "SHA3-Meridian" for Shareholder Agreement 3 with Meridian). \
  Keep identifier SHORT (2-4 words max). \
Examples of DOCUMENT-SPECIFIC pages that MUST have the identifier in this order: \
  - Court judgments: Facts, Procedural History, Charges, Holding, Contentions, Relief, Costs \
    → "Facts – Yuvraj Kanther (Court Judgment)" \
  - Contracts: Term, Termination, Payment, Fees, Indemnity, Liability, Scope of Services, \
    Obligations, Representations, IP Ownership, Confidentiality, Governing Law, Dispute Resolution, \
    Parties, Assignment → "Term and Termination – SA1-Crayons (Service Agreement)" \
SHARED LEGAL CONCEPT PAGES: Pages about statutes, precedents, legal doctrines, and general \
legal principles that are the SAME regardless of which document discusses them should NOT \
include the document identifier — they merge intentionally: \
  "Section 319 CrPC (Court Judgment)", "Indian Arbitration Act (Service Agreement)". \
Only use shared pages for genuinely universal legal concepts, NOT for document-specific clauses.

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
    "payment_terms": "Payment due date / terms (e.g. 'Net 30') or null",
    "matter_reference": "Matter/case/docket/reference number printed in the document header or caption, verbatim, or null"
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
  ],
  "clauses": [
    {{
      "type": "Short clause category, e.g. 'Liability Cap', 'Termination for Convenience', 'Indemnity', 'Payment Terms'",
      "text": "Exact verbatim clause text from the document",
      "typed_value": {{"...": "..."}} or null,
      "confidence": 1.0,
      "page": 3
    }}
  ]
}}

CLAUSE EXTRACTION: In addition to the wiki pages above, extract each individually identifiable \
clause as a separate entry in "clauses". "text" must be an exact verbatim quote — never paraphrase. \
"typed_value" is an optional small object holding the clause's structured value when it has one \
(e.g. {{"multiplier": 2, "basis": "prior 12 months' fees"}} for a liability cap) — use null when the \
clause doesn't reduce to a simple structured value. Rate "confidence" using this rubric, the same \
one used elsewhere in this system: 1.0 = exact verbatim match with no ambiguity, 0.8 = clearly \
stated but the exact wording required light interpretation, 0.5 = the clause is implied rather \
than explicitly stated, 0.0 = you are not actually confident this is a real clause in the text. \
Extract every clause you can identify — do not filter by confidence, low-confidence entries are \
exactly what the Review Queue is for. CONTRACT VALUE: if the document states a total, aggregate or \
annual contract value, a total consideration, or a total price, ALWAYS emit it as its own clause \
with type "Total Contract Value" and typed_value {{"total": "<amount exactly as written, including \
currency>"}}. This field was previously extracted only when the model happened to volunteer it — \
present on roughly 17 of 31,000 clauses across this corpus — so it is called out explicitly here \
rather than left to judgement. Never compute or infer it: emit it only when the document states it. \
COVER EVERY NUMBERED SECTION: if the document numbers its \
sections, each numbered heading must appear as a "clauses" entry, including the back-half \
boilerplate (Relationship Of Parties, Compliance With Laws, No Third Party Rights, Waiver, \
Severability, Counterparts, Further Assurance). Those carry no negotiated value, which is exactly \
why they get skipped — and exactly what a question naming "Section 12" asks for.

Extract 10-40 relations. Cover the document thoroughly.
{family_block}

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
DOCUMENT-SPECIFIC TOPICS (CRITICAL): Separate topics into two categories: \
1. DOCUMENT-SPECIFIC (provisions, clauses, obligations, terms unique to THIS document): \
   prefix each topic with a SHORT DOCUMENT IDENTIFIER: \
   For court judgments: first party's last name (e.g. "Facts – Yuvraj Kanther", "Holding – Yuvraj Kanther"). \
   For contracts/agreements: a short identifier from the counterparty name or filename that \
   distinguishes this document from others of the same type (e.g. "Term – SA1-Crayons", \
   "Indemnity – SA1-Crayons", "Payment – NDA-Acme", "Scope – SHA3-Meridian"). \
   Keep identifier SHORT (2-4 words max). EVERY clause-level topic MUST have this prefix. \
2. SHARED LEGAL CONCEPTS (statutes, precedents, doctrines, principles): NO prefix — \
   these merge intentionally: e.g. "Section 319 CrPC", "Indian Arbitration Act".

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
- STAMP/CHALLAN CAPTURE (CRITICAL): If this segment includes a stamp certificate, e-stamp, or \
  stamp duty challan — including as a cover-page watermark or header — create a dedicated wiki \
  page for it with the certificate/GRN number, date, purchaser/payer name, and amount verbatim.
- SERVICE-LEVEL / CADENCE PRECISION (CRITICAL): For recurring obligations (reporting frequency, \
  review/hygiene cadence, response times, SLA turnaround), quote the literal cadence verbatim \
  (e.g. "Weekly reports every Monday"). Do not compress into vague paraphrase like "periodic".
- PAGE TITLES: You MUST append the inferred Document Type in parentheses to EVERY page title. \
  DOCUMENT-SPECIFIC TITLES (CRITICAL): The KNOWN TOPICS list will contain some topics with \
  a document identifier attached (e.g. "Facts – Yuvraj Kanther", "Term – SA1-Crayons") and some \
  without (e.g. "Section 319 CrPC", "Indian Arbitration Act"). Preserve these exactly, IN THE \
  SAME ORDER, when generating page titles. \
  TITLE ORDER (STRICT — never deviate): the topic ALWAYS comes first, then a dash, then the \
  document identifier — never the reverse. Wrong: "SA1-Crayons – Term". \
  Right: "Term – SA1-Crayons". \
  If a document-specific topic (any clause, provision, obligation, term, or fact specific to \
  THIS document) in the KNOWN TOPICS list has no identifier yet, add one AFTER the topic, in \
  that same "{{Topic}} – {{Identifier}}" order. \
  Shared legal concept pages (statutes, precedents, doctrines) must NOT have an identifier. \
  Examples: "Facts – Yuvraj Kanther ({doc_type})", "Term – SA1-Crayons ({doc_type})", \
  "Section 319 CrPC ({doc_type})", "Indian Arbitration Act ({doc_type})".
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
  ],
  "clauses": [
    {{
      "type": "Short clause category, e.g. 'Liability Cap', 'Termination for Convenience', 'Indemnity', 'Payment Terms'",
      "text": "Exact verbatim clause text from this segment",
      "typed_value": {{"...": "..."}} or null,
      "confidence": 1.0,
      "page": 3
    }}
  ]
}}

CLAUSE EXTRACTION: In addition to the wiki pages above, extract each individually identifiable \
clause in this segment as a separate entry in "clauses". "text" must be an exact verbatim quote — \
never paraphrase. "typed_value" is an optional small object holding the clause's structured value \
when it has one, else null. Rate "confidence" using this rubric: 1.0 = exact verbatim match with \
no ambiguity, 0.8 = clearly stated but the exact wording required light interpretation, 0.5 = the \
clause is implied rather than explicitly stated, 0.0 = you are not actually confident this is a \
real clause. Extract every clause you can identify — do not filter by confidence. CONTRACT VALUE: \
if this segment states a total, aggregate or annual contract value, a total consideration, or a \
total price, ALWAYS emit it as its own clause with type "Total Contract Value" and typed_value \
{{"total": "<amount exactly as written, including currency>"}}. Never compute or infer it: emit it \
only when the segment states it. COVER EVERY \
NUMBERED SECTION in this segment: each numbered heading must appear as a "clauses" entry, including \
back-half boilerplate (Relationship Of Parties, Compliance With Laws, No Third Party Rights, Waiver, \
Severability, Counterparts, Further Assurance) — those get skipped precisely because they carry no \
negotiated value, and they are exactly what a question naming "Section 12" asks for.
{family_block}

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


def _persist_clauses(session_id: str, doc_name: str, parsed: dict) -> None:
    """Write out any "clauses" the ingest LLM call returned alongside its
    pages/relations — Review Queue § 02 first slice. Independent of
    _atomic_merge/_merge_wiki on purpose: clauses are append-only per
    ingest call, no merge-by-title logic like pages have, so this never
    touches that (complex, already-tested) machinery. Only called for the
    single-call and per-segment detail passes, not the overview pass —
    the overview reads a coarse excerpt, not full page content, so it has
    nothing reliable to extract clauses from.
    """
    if not config.USE_DATABASE:
        return
    clauses = parsed.get("clauses") or []
    if not clauses:
        return
    try:
        n = _db.insert_clauses(_active_wiki_id(), session_id, doc_name, clauses)
        if n:
            logger.info("Persisted %d clause(s) for %s into the Review Queue", n, doc_name)
    except Exception as e:
        logger.error("Failed to persist clauses for %s: %s", doc_name, e)


def _collect_structured(parsed: dict, bucket: dict) -> None:
    """Accumulate stage 03's structured output across passes.

    Each segment contributes its own rows; they're reconciled once at the end
    rather than per segment, because a duplicate is only detectable against
    the rows the *other* segments produced.
    """
    if not isinstance(parsed, dict):
        return
    for key in ("citations", "structural_anchors", "tables", "figures",
                "document_references", "obligations"):
        rows = parsed.get(key)
        if isinstance(rows, list):
            bucket.setdefault(key, []).extend(r for r in rows if isinstance(r, dict))
    meta = parsed.get("family_metadata")
    if isinstance(meta, dict):
        # First non-null wins per field. Later segments see less of the
        # document, so a value from an earlier pass is the better-sourced one;
        # this also means a segment that "helpfully" restates a field it can't
        # actually see can't overwrite the pass that genuinely read it.
        merged = bucket.setdefault("family_metadata", {})
        for k, v in meta.items():
            if k not in merged or merged[k] is None:
                merged[k] = v
    hq = parsed.get("hypothetical_questions")
    if isinstance(hq, dict):
        store = bucket.setdefault("hypothetical_questions", {})
        for title, qs in hq.items():
            if isinstance(qs, list):
                store.setdefault(title, []).extend(str(q) for q in qs if q)


# Cap per page. The synthesis call is asked for 2-4; this bounds what a
# runaway response can cost, since every question is a vector to embed and
# store. Truncating is safe — the questions are alternative handles on the
# same page, so losing the fifth costs a little recall, not correctness.
_MAX_QUESTIONS_PER_PAGE = 6


def _embed_hypothetical_questions(session_id: str, doc_name: str,
                                  questions_by_page: dict,
                                  doc_family: str | None) -> None:
    """Stage 06 — embed the questions each page can answer.

    The third embedding type. A lawyer asks "can they terminate for
    convenience?"; the page is titled "Term and Termination" and says
    "either party may terminate on 30 days' notice without cause". Page-level
    similarity has to bridge that gap on vocabulary alone. A stored question
    phrased the way a lawyer would ask it closes it directly.

    Only pages that actually exist get questions stored. A question keyed to
    a title the merge didn't produce (a hallucinated or renamed page) would
    be an orphan vector that surfaces a page which isn't there.
    """
    if not config.USE_DATABASE or not questions_by_page:
        return
    try:
        from services import embedder

        wiki_id = _active_wiki_id()
        existing = set(_db.get_page_titles(wiki_id, session_id))
        pairs: list[tuple[str, str]] = []
        skipped_unknown = 0
        for title, questions in questions_by_page.items():
            if title not in existing:
                skipped_unknown += 1
                continue
            seen: set[str] = set()
            for q in questions[:_MAX_QUESTIONS_PER_PAGE]:
                q = str(q).strip()
                key = q.lower()
                if q and key not in seen:
                    seen.add(key)
                    pairs.append((title, q))

        if skipped_unknown:
            logger.info("Stage 06: skipped questions for %d page title(s) that "
                        "no page exists under (%s)", skipped_unknown, doc_name)
        if not pairs:
            return

        vectors = embedder.embed_batch([q for _, q in pairs], is_query=False)
        by_page: dict[str, list[tuple[str, list[float]]]] = {}
        for (title, q), vec in zip(pairs, vectors):
            if vec:
                by_page.setdefault(title, []).append((q, vec))

        total = 0
        for title, items in by_page.items():
            total += _db.upsert_question_embeddings(
                wiki_id, session_id, title, items, doc_family, doc_name)
        logger.info("Stage 06: embedded %d hypothetical question(s) across %d page(s) for %s",
                    total, len(by_page), doc_name)
    except Exception as err:
        # Same containment rule as the typed tables: this is additive
        # retrieval signal, and losing it must not fail an ingest whose pages
        # and structured rows are already correct.
        logger.error("Stage 06 question embedding failed for %s: %s", doc_name, err)


# Amendment lookups happen once per merged page, which is thousands of times
# per ingest. The edge set for a document changes only when its references are
# written (once, before the merge), so it is cached for the duration and
# invalidated explicitly there rather than re-queried per page.
_AMENDMENT_CACHE: dict[tuple[str, str], set[str]] = {}
_AMENDMENT_CACHE_LOCK = threading.Lock()


def _amendment_pair_cached(session_id: str, doc_a: str, doc_b: str) -> bool:
    if not doc_a or not doc_b or doc_a == doc_b:
        return False
    key = (session_id, doc_a)
    with _AMENDMENT_CACHE_LOCK:
        partners = _AMENDMENT_CACHE.get(key)
    if partners is None:
        try:
            from services import doc_references as _refs
            partners = _refs.amendment_partners(session_id, doc_a)
        except Exception as err:
            logger.debug("Amendment lookup failed for %s: %s", doc_a, err)
            partners = set()
        with _AMENDMENT_CACHE_LOCK:
            _AMENDMENT_CACHE[key] = partners
    return doc_b in partners


def _invalidate_amendment_cache(session_id: str, doc_name: str) -> None:
    with _AMENDMENT_CACHE_LOCK:
        _AMENDMENT_CACHE.pop((session_id, doc_name), None)
        # The reverse direction matters too: writing "A amends B" changes what
        # a later merge of B should conclude about A.
        for key in [k for k in _AMENDMENT_CACHE if k[0] == session_id]:
            _AMENDMENT_CACHE.pop(key, None)


def _sr_is_high_stakes(family: str | None, field_name: str) -> bool:
    """Registry lookup, wrapped so a missing family can't break persistence."""
    try:
        from services import schema_registry as _sr
        return _sr.is_high_stakes_metadata(family, field_name)
    except Exception:
        return False


def _resolve_doc_references(session_id: str, doc_name: str, parsed: dict) -> None:
    """Stage 03/04 — write this document's outgoing references as edges.

    Called BEFORE the page merge, not after, and that ordering is the whole
    point (§ 01 stage 04). Contradiction detection compares a new page against
    the existing one; if this document amends the one that wrote the existing
    content, their disagreement is a resolved version chain, not a conflict.
    The merge can only know that if the amendment edge already exists when it
    runs.
    """
    if not config.USE_DATABASE:
        return
    refs = parsed.get("document_references") if isinstance(parsed, dict) else None
    if not refs:
        return
    try:
        from services import doc_references as _refs, wikis as _wikis
        counts = _refs.persist_references(
            _wikis.active_wiki_id(), session_id, doc_name, refs)
        _invalidate_amendment_cache(session_id, doc_name)
        if counts:
            logger.info("Document references for %s: %s", doc_name, counts)
    except Exception as err:
        logger.error("Document-reference resolution failed for %s: %s", doc_name, err)


def _queue_review_items(wiki_id: str, session_id: str, doc_name: str,
                        classification: dict, meta_report, family: str | None,
                        tables: list, figures: list) -> None:
    """Stage 07 — the evaluation gate, extended past clause confidence.

    Everything flagged upstream converges here: a doubtful doc-type call, a
    metadata field that failed validation or came back low-confidence, and a
    table or figure the model wasn't sure it read correctly. Below threshold
    an extraction lands in the queue instead of entering the index silently,
    which is the whole point — silence is the failure mode, not low
    confidence itself.
    """
    from services import schema_registry as _sr

    # Prior pending items for this document are archived, never deleted — a
    # reviewer's earlier judgement is evidence about how this document reads.
    try:
        n = _db.supersede_review_items(wiki_id, session_id, doc_name)
        if n:
            logger.info("Superseded %d prior review item(s) for %s", n, doc_name)
    except Exception as err:
        logger.warning("Could not supersede prior review items for %s: %s", doc_name, err)

    items: list[dict] = []

    if classification.get("flagged"):
        items.append({
            "item_kind": "doc_type",
            "item_label": f"Document type: {classification.get('doc_family')}",
            "item_value": (classification.get("doc_type")
                           or classification.get("doc_family")),
            "confidence": classification.get("family_confidence") or 0.0,
            "reason": classification.get("flag_reason"),
            "typed_value": {
                "family": classification.get("doc_family"),
                "folder_family": classification.get("folder_family"),
                "folder_hint": classification.get("folder_hint"),
                "method": classification.get("family_method"),
                "reasoning": classification.get("reasoning"),
            },
        })

    fam_def = _sr.get(family)
    for field_name, result in (meta_report.fields or {}).items():
        if field_name == "__payload__":
            continue
        if not result.flagged:
            continue
        items.append({
            "item_kind": "metadata",
            "item_label": f"{field_name} ({fam_def.key})",
            "item_value": result.raw if result.raw is not None else result.value,
            "confidence": result.confidence,
            # High-stakes metadata is per-family, from the registry — a
            # contract's governing law and a judgment's holding both need
            # individual sign-off, but for different reasons and in
            # different families.
            "high_stakes": _sr.is_high_stakes_metadata(family, field_name),
            "reason": result.reason,
            "typed_value": {"coerced": result.coerced, "value": result.value},
        })

    for kind, rows in (("table", tables), ("figure", figures)):
        for row in rows:
            conf = row.get("confidence") or row.get("_confidence") or 0.0
            if conf > _TABLE_FIGURE_REVIEW_THRESHOLD and not row.get("_flagged"):
                continue
            label = (row.get("caption") or row.get("description")
                     or f"unlabelled {kind}")
            items.append({
                "item_kind": kind,
                "item_label": f"{kind.title()}: {str(label)[:120]}",
                "item_value": json.dumps(
                    {k: v for k, v in row.items() if not k.startswith("_")},
                    ensure_ascii=False, default=str,
                )[:4000],
                "confidence": conf,
                "page_num": row.get("page_num"),
                "reason": "; ".join(row.get("_validation_notes") or [])
                          or f"low {kind} extraction confidence",
            })

    if not items:
        return
    try:
        n = _db.insert_review_items(wiki_id, session_id, doc_name, items)
        logger.info("Review Queue: %d item(s) flagged for %s", n, doc_name)
    except Exception as err:
        logger.error("Could not queue review items for %s: %s", doc_name, err)


# A table or figure below this lands in the queue. Structure extraction is
# the least reliable thing in the pipeline — a table reconstructed from a
# guess at its layout looks exactly as authoritative as one read correctly,
# so the bar for letting one through unreviewed is higher than for prose.
_TABLE_FIGURE_REVIEW_THRESHOLD = 0.75


# A right recorded as a duty reverses what the clause does — the tracker
# would report that a party must do something the agreement merely lets it
# do. The prompt says so; a live ingest showed the model returning "may set
# off any amount owed to it" as an obligation anyway, so the rule is enforced
# here as well rather than only asked for.
#
# The negation is what decides, not the modal: "may not disclose" is a
# prohibition and a real obligation, so only a permissive opener with no
# negative attached is dropped.
_PERMISSIVE_DUTY_RE = re.compile(
    r"^\s*(?:may|can|could|is\s+(?:entitled|permitted|free)\s+to|"
    r"shall\s+be\s+entitled\s+to|has\s+the\s+(?:right|option)\s+to|"
    r"at\s+its\s+(?:option|discretion)|in\s+its\s+discretion)\b",
    re.IGNORECASE,
)
_NEGATED_PERMISSIVE_RE = re.compile(
    r"^\s*(?:may|can|could|shall\s+be\s+entitled)\s+(?:not|never|no)\b",
    re.IGNORECASE,
)


def _is_permissive_duty(duty: str | None) -> bool:
    if not duty:
        return False
    if _NEGATED_PERMISSIVE_RE.match(duty):
        return False
    return bool(_PERMISSIVE_DUTY_RE.match(duty))


def _persist_structured(session_id: str, doc_name: str, bucket: dict,
                        classification: dict, anchors_from_text: list) -> None:
    """Stage 04 — reconcile the structured rows and swap them in.

    Deliberately wrapped: a failure to persist typed rows must not fail the
    ingest of a document whose wiki pages merged fine. The typed tables are
    additive to the existing pipeline, and taking the whole ingest down with
    them would make the backbone a liability rather than an addition.
    """
    if not config.USE_DATABASE:
        return
    try:
        from services import backbone, extraction_validation as _ev
        from services import family_prompt, wikis

        wiki_id = wikis.active_wiki_id()
        family = classification.get("doc_family")

        document_id = backbone.upsert_document(
            wiki_id, session_id, doc_name,
            doc_family=family,
            doc_type=classification.get("doc_type"),
            jurisdiction=classification.get("jurisdiction"),
            family_confidence=classification.get("family_confidence"),
            family_method=classification.get("family_method"),
            folder_hint=classification.get("folder_hint"),
            # Stamped here so future ingests can find this document before
            # spending anything on it — see _file_hash/_content_hash and the
            # two duplicate checks near the top of ingest().
            content_hash=classification.get("content_hash"),
            file_hash=classification.get("file_hash"),
        )

        raw_meta = bucket.get("family_metadata") or {}
        meta_report = _ev.validate_payload(
            raw_meta, family_prompt.metadata_spec(family),
            base_confidence=classification.get("family_confidence") or 0.8,
        )
        family_row = dict(meta_report.values)
        family_row["confidence"] = meta_report.confidence
        # Per-FIELD provenance, not just a row-level score. A reviewer looking
        # at a document needs to know which individual value is shaky — a
        # single number for the whole row tells them the document is doubtful
        # without telling them where to look, which is most of the work.
        family_row["typed_value"] = {
            "fields": {
                name: {
                    "value": res.value,
                    "raw": res.raw if res.raw != res.value else None,
                    "confidence": round(res.confidence, 3),
                    "flagged": res.flagged,
                    "coerced": res.coerced,
                    "reason": res.reason,
                    "high_stakes": _sr_is_high_stakes(family, name),
                }
                for name, res in (meta_report.fields or {}).items()
                if name != "__payload__"
            },
            "validated": meta_report.values,
            "flagged_fields": meta_report.flagged,
            "notes": meta_report.notes(),
        }

        # Promote the document-level facts the family extraction just produced
        # onto the `documents` row. Without this they are extracted on every
        # ingest, written into the typed table's typed_value blob, and then
        # never surfaced: documents.effective_date / parties / expiry_date sat
        # at 0% populated across the whole corpus, so the Contract Register and
        # Obligation tracker (which read those columns) had nothing to show.
        # Field names differ per family — a judgment has decided_date, an
        # opinion has opinion_date — so each is mapped to the shared column.
        _vals = meta_report.values or {}

        def _first(*names):
            for n in names:
                v = _vals.get(n)
                if v not in (None, "", [], {}):
                    return v
            return None

        _parties = _first("parties")
        if not _parties:
            _sides = [v for v in (_vals.get("plaintiffs"), _vals.get("defendants"),
                                  _vals.get("grantor"), _vals.get("grantee"))
                      if v not in (None, "", [], {})]
            flat = []
            for s in _sides:
                flat.extend(s if isinstance(s, list) else [s])
            _parties = flat or None

        _doc_meta = {
            "effective_date": _first("effective_date", "opinion_date", "decided_date"),
            "expiry_date": _first("expiry_date"),
            "parties": _parties,
            "status": _first("status", "disposition", "binding_status"),
        }
        _doc_meta = {k: v for k, v in _doc_meta.items() if v not in (None, "", [], {})}
        if _doc_meta:
            try:
                backbone.upsert_document(wiki_id, session_id, doc_name, **_doc_meta)
            except Exception as _dm_err:
                logger.warning("Could not promote document metadata for %s: %s",
                               doc_name, _dm_err)

        citations, _ = _ev.sanitize_rows(
            bucket.get("citations"),
            {"citation_text": "text", "authority_type": "text",
             "normalized_form": "text", "page": "number", "confidence": "number"},
            required=("citation_text",),
        )
        citations = backbone.reconcile_rows(citations, ("normalized_form",)) \
            if all(c.get("normalized_form") for c in citations) \
            else backbone.reconcile_rows(citations, ("citation_text",))
        for c in citations:
            c["page_num"] = c.pop("page", None)
            c["confidence"] = c.get("confidence") or c.get("_confidence")

        # Obligations are deduped on the sentence that imposes the duty, not
        # on the duty text: two segments describing the same clause paraphrase
        # the duty differently but quote the same sentence, so the paraphrase
        # is the field that fails to match when it matters most.
        obligations, _ = _ev.sanitize_rows(
            bucket.get("obligations"),
            {"obligated_party": "text", "duty": "text", "trigger": "text",
             "deadline": "text", "notice_period": "duration",
             "consequence": "text", "verbatim_text": "text",
             "page": "number", "confidence": "number"},
            required=("obligated_party", "duty"),
        )
        obligations = backbone.reconcile_rows(obligations, ("verbatim_text",)) \
            if all(o.get("verbatim_text") for o in obligations) \
            else backbone.reconcile_rows(obligations, ("obligated_party", "duty"))
        # Logged rather than dropped silently: whether the model stopped
        # emitting rights as duties, or is still emitting them and this guard
        # is what keeps them out, is the difference between the prompt working
        # and the guard carrying it — and only a log line tells them apart.
        _rights = [o for o in obligations if _is_permissive_duty(o.get("duty"))]
        if _rights:
            logger.info("Dropped %d permissive clause(s) miscast as obligations in %s: %s",
                        len(_rights), doc_name, [o.get("duty") for o in _rights][:3])
        obligations = [o for o in obligations if not _is_permissive_duty(o.get("duty"))]
        for o in obligations:
            o["page_num"] = o.pop("page", None)
            o["confidence"] = o.get("confidence") or o.get("_confidence")

        tables, _ = _ev.sanitize_rows(
            bucket.get("tables"),
            {"caption": "text", "columns": "list", "rows": "list",
             "page": "number", "confidence": "number"},
        )
        for t in tables:
            t["page_num"] = t.pop("page", None)
            t["extraction_method"] = "synthesis"
            t["confidence"] = t.get("confidence") or t.get("_confidence")

        figures, _ = _ev.sanitize_rows(
            bucket.get("figures"),
            {"figure_kind": "text", "description": "text", "page": "number",
             "confidence": "number"},
            required=("description",),
        )
        for f in figures:
            f["page_num"] = f.pop("page", None)
            f["extraction_method"] = "synthesis"
            f["confidence"] = f.get("confidence") or f.get("_confidence")

        # Anchors come from the deterministic regex parse, not the model —
        # the model's "structural_anchors" output is used only to confirm what
        # the parse already found. A regex over the real text cannot invent a
        # paragraph number; a language model can, and an invented anchor is a
        # confident-looking pointer to text that isn't there.
        anchor_rows = [a.as_row() for a in anchors_from_text]

        written = backbone.replace_document_rows(
            wiki_id, session_id, doc_name, document_id,
            family_row=family_row, family_key=family,
            obligations=obligations,
            citations=citations, anchors=anchor_rows,
            tables=tables, figures=figures,
        )
        logger.info("Backbone rows for %s: %s (family=%s)", doc_name, written, family)

        for party in (family_row.get("parties") or []):
            backbone.upsert_entity(wiki_id, str(party), "party", doc_name)

        # --- Stage 07: evaluation gate --------------------------------------
        _queue_review_items(wiki_id, session_id, doc_name, classification,
                            meta_report, family, tables, figures)

    except Exception as err:
        logger.error("Backbone persistence failed for %s (wiki pages unaffected): %s",
                     doc_name, err, exc_info=True)


# Below this, extracted text is not a meaningful dedup signal — a near-empty
# OCR failure and an unrelated near-empty OCR failure on a different document
# would otherwise hash identically and collide as a false "duplicate".
_MIN_HASH_CHARS = 200


def _file_hash(path: str) -> str | None:
    """SHA-256 of the raw uploaded file's bytes. Computed straight off disk,
    before any text extraction or OCR runs — this is the actual upload-time
    dedup signal. Unlike _content_hash, this costs nothing: no LLM, no OCR,
    not even the CPU work of parsing the file format.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _content_hash(text: str) -> str | None:
    """SHA-256 of the extracted text, whitespace-normalized. None if the
    text is too short to be a trustworthy signal (see _MIN_HASH_CHARS).

    Exact-content matching only, by design — this never attempts fuzzy or
    near-duplicate detection (simhash, embedding similarity). Two genuinely
    different contracts that happen to read similarly must never be merged
    as "the same document" in a legal corpus; that false positive costs far
    more than missing a near-duplicate does. Whitespace normalization is as
    far as this goes, to survive incidental re-extraction/line-ending
    differences without drifting into similarity matching.
    """
    normalized = " ".join((text or "").split())
    if len(normalized) < _MIN_HASH_CHARS:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def ingest(file_path: str, session_id: str) -> dict:
    """Read a source document, extract wiki pages via LLM, and merge into the session wiki.

    Short documents (≤ 100K chars): processed in a single LLM call.
    Long documents: two-phase approach — overview first, then detailed segments
    with the overview's topic list as context to reduce redundancy.
    Segments are processed concurrently to improve speed.
    """
    doc_name = os.path.basename(file_path)

    # --- Duplicate check #1: identical raw file, checked at upload time -----
    # Runs BEFORE text extraction/OCR — the whole point. A re-uploaded file
    # is caught from its raw bytes alone, so it never touches the reader, let
    # alone the vision-OCR fallback (a real, billed LLM call per scanned page
    # in this deployment). Scoped to the active wiki, not the session: the
    # goal is catching a document re-uploaded under a fresh session_id, which
    # is what a repeated folder upload actually does.
    file_hash = _file_hash(file_path)
    if config.USE_DATABASE and file_hash:
        try:
            from services import backbone as _backbone
            file_dup = _backbone.find_by_file_hash(_active_wiki_id(), file_hash)
        except Exception as _dup_err:
            logger.warning("File-hash duplicate check failed for %s, ingesting normally: %s",
                           doc_name, _dup_err)
            file_dup = None
        if file_dup and file_dup["source_doc"] != doc_name:
            logger.info("Skipping %s — identical file already ingested as %s (no extraction run)",
                       doc_name, file_dup["source_doc"])
            _log_event(session_id, "DUPLICATE",
                      f"{doc_name} skipped — identical file already ingested "
                      f"as {file_dup['source_doc']}")
            return {
                "pages_updated": 0,
                "relations": 0,
                "duplicate_of": file_dup["source_doc"],
                "duplicate_family": file_dup.get("doc_family"),
            }

    from services.reader import read_file_with_positions as _read_with_pos
    result = _read_with_pos(file_path)
    text = result["text"]
    page_map = result["page_map"]

    # Per-page extraction provenance (§ Phase 3.5b). Written before any of the
    # expensive extraction below, and never allowed to fail the ingest: this
    # table exists to disclose a quality problem, so a bug in the disclosure
    # must not become a reason the document does not get ingested at all.
    page_quality = result.get("page_quality") or []
    if config.USE_DATABASE and page_quality:
        try:
            _db.upsert_page_quality(_active_wiki_id(), session_id, doc_name, page_quality)
            _unreadable = sum(1 for p in page_quality if p.get("below_floor"))
            if _unreadable:
                logger.warning("%s: %d of %d page(s) unreadable after extraction "
                               "— Document QA warning will fire",
                               doc_name, _unreadable, len(page_quality))
                _log_event(session_id, "QUALITY",
                           f"{doc_name}: {_unreadable} of {len(page_quality)} "
                           f"page(s) could not be read")
        except Exception as e:
            logger.error("Failed to record page quality for %s: %s", doc_name, e)

    # --- Duplicate check #2: identical extracted text, different bytes -----
    # Secondary safety net for the case file_hash can't catch — same content
    # re-saved/re-scanned into a byte-different file. Only reachable once
    # extraction already ran for a document that passed check #1, so it adds
    # no extra OCR/LLM cost of its own.
    content_hash = _content_hash(text)
    if config.USE_DATABASE and content_hash:
        try:
            from services import backbone as _backbone
            dup = _backbone.find_by_content_hash(_active_wiki_id(), content_hash)
        except Exception as _dup_err:
            logger.warning("Duplicate check failed for %s, ingesting normally: %s",
                           doc_name, _dup_err)
            dup = None
        if dup and dup["source_doc"] != doc_name:
            logger.info("Skipping %s — identical content already ingested as %s",
                       doc_name, dup["source_doc"])
            _log_event(session_id, "DUPLICATE",
                      f"{doc_name} skipped — identical content already ingested "
                      f"as {dup['source_doc']}")
            return {
                "pages_updated": 0,
                "relations": 0,
                "duplicate_of": dup["source_doc"],
                "duplicate_family": dup.get("doc_family"),
            }

    # Store page-level positions for citation location support
    if config.USE_DATABASE and page_map:
        try:
            _db.store_page_map(_active_wiki_id(), session_id, doc_name, page_map)
        except Exception as _pm_err:
            logger.warning("Failed to store page map for %s: %s", doc_name, _pm_err)

    logger.info("Wiki ingest: %s (%d chars, %d pages)", doc_name, len(text), len(page_map))

    # --- Stage 02: doc-type + jurisdiction classification -------------------
    # Runs before the length fork, because which family applies decides which
    # schema stage 03 asks for — deciding it after would mean extracting the
    # contract schema from a judgment and then relabelling the result.
    _update_doc_step(session_id, doc_name, "classifying")
    try:
        from services import classifier as _classifier
        classification = _classifier.classify_document(text, file_path)
    except Exception as _cls_err:
        logger.error("Classification failed for %s, using generic: %s", doc_name, _cls_err)
        classification = {"doc_family": "generic", "family_confidence": 0.0,
                          "flagged": True, "flag_reason": str(_cls_err)}
    # Stamped here so both duplicate checks above have something to find on
    # the *next* ingest of this document.
    classification["content_hash"] = content_hash
    classification["file_hash"] = file_hash
    family_key = classification.get("doc_family")

    # --- Structural anchors: one deterministic parse, two consumers ---------
    from services import structure as _structure
    anchors = _structure.parse_anchors(text)
    logger.info("Structure: %d anchor(s) in %s (%.1f per 10k chars)",
                len(anchors), doc_name, _structure.structure_ratio(text, anchors))

    # Signal: file has been read, starting synthesis
    _update_doc_step(session_id, doc_name, "synthesizing")

    total_contradictions = 0
    structured: dict = {}

    if len(text) <= _SINGLE_CALL_THRESHOLD:
        # --- Short document: single LLM call ---
        _update_wiki_progress(session_id, {"current": 0, "total": 1,
                                            "message": f"Processing {doc_name}..."})
        _update_doc_step(session_id, doc_name, "synthesizing", "1/1")
        parsed = _ingest_single_call(text, doc_name, family_key)
        _persist_clauses(session_id, doc_name, parsed)
        _collect_structured(parsed, structured)
        # Before the merge — amendment edges must exist for contradiction
        # detection to tell a version chain from a real conflict.
        _resolve_doc_references(session_id, doc_name, parsed)
        _update_doc_step(session_id, doc_name, "merging")
        _update_wiki_progress(session_id, {"current": 1, "total": 1,
                                            "message": f"Processing {doc_name}..."})
        total_pages, total_rels, total_contradictions = _atomic_merge(session_id, parsed, doc_name)
    else:
        # --- Long document: two-phase approach ---
        segments = [s.text for s in
                    _structure.split_segments(text, _INGEST_CHUNK_SIZE, anchors)]
        total_steps = 1 + len(segments)
        _update_wiki_progress(session_id, {"current": 0, "total": total_steps,
                                            "message": f"Overview pass for {doc_name}..."})
        _update_doc_step(session_id, doc_name, "synthesizing", f"0/{len(segments)}")

        overview_text = text[:6000] + "\n\n[...]\n\n" + text[-3000:]
        doc_type, topics, overview_parsed = _ingest_overview(overview_text, doc_name)
        # Carry the inferred doc_type into the merge so it gets persisted to
        # page_metadata (Phase 0). The overview/detail prompts don't emit a
        # top-level "doc_type" in their pages dict the way the single-call prompt
        # does, so inject it here — this is the one place the long-doc path knows it.
        overview_parsed["doc_type"] = doc_type
        _update_wiki_progress(session_id, {"current": 1, "total": total_steps,
                                            "message": f"Overview pass for {doc_name}..."})

        total_pages, total_rels, tc = _atomic_merge(session_id, overview_parsed, doc_name)
        total_contradictions += tc

        completed_segments = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.WIKI_MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(_ingest_detail_segment, seg, topics, doc_name,
                                doc_type, family_key): i
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
                    _persist_clauses(session_id, doc_name, parsed)
                    _collect_structured(parsed, structured)
                    _resolve_doc_references(session_id, doc_name, parsed)
                    p, r, c = _atomic_merge(session_id, parsed, doc_name)
                    total_pages += p
                    total_rels += r
                    total_contradictions += c
                except Exception as exc:
                    logger.error("Segment %d for %s generated an exception: %s", i, doc_name, exc)
                    _log_event(session_id, "ERROR", f"Doc: {doc_name} | Segment {i} failed: {exc}")

    # --- Stage 04: reconcile + swap in the typed rows ----------------------
    _update_doc_step(session_id, doc_name, "persisting")
    _persist_structured(session_id, doc_name, structured, classification, anchors)

    # --- Stage 06: hypothetical-question embeddings ------------------------
    _embed_hypothetical_questions(
        session_id, doc_name, structured.get("hypothetical_questions") or {},
        classification.get("doc_family"),
    )

    logger.info("Wiki ingest complete: %d pages, %d relations", total_pages, total_rels)
    _log_event(session_id, "INGEST",
               f"Doc: {doc_name} | Pages updated: {total_pages} | Contradictions found: {total_contradictions}")
    if classification.get("flagged"):
        _log_event(session_id, "REVIEW",
                   f"Doc: {doc_name} | Classification flagged: {classification.get('flag_reason')}")

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


# Unicode hyphen/dash variants the answer LLM sometimes emits for compound
# terms (e.g. "field‑of‑use", "NDA‑Tata") that are NOT the plain ASCII
# hyphen used in stored page titles / source text. A visually-identical string
# fails a naive lower+whitespace-only normalization if one side uses U+2011
# (non-breaking hyphen) and the other uses U+002D (ASCII "-") — found live
# testing a citation-label false positive where the LLM's restated label
# ("Obligations of Receiving Party – NDA‑Tata – NDA") didn't match the
# real title ("...NDA-Tata...") only because of this character difference.
_HYPHEN_VARIANTS_RE = re.compile('[‐‑‒–—−]')


def _norm_for_match(s: str) -> str:
    """Shared normalization for all quote/title verification comparisons:
    collapse whitespace, lowercase, fold unicode hyphen/dash variants to a
    plain ASCII "-", and strip trailing/leading sentence punctuation (a
    citation label quoted mid-sentence often picks up a trailing comma or
    period from the surrounding prose, e.g. '"...Service Agreement,"' — that
    punctuation isn't part of the real page title, so leaving it in broke the
    exact-match lookup in _known_page_titles() and caused the label itself to
    be flagged as an unverifiable content quote instead of recognized as a
    citation label). Safe for quote-content matching too: stripping trailing
    punctuation from a short extracted span before a substring-containment
    check against the full corpus can only make a match MORE likely to be
    found, never mask a genuine mismatch.
    """
    s = _HYPHEN_VARIANTS_RE.sub('-', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s.strip('.,;:')


def _alnum_only(s: str) -> str:
    """Strip everything but letters/digits, for comparing a citation label

    against a page title when the model has reformatted punctuation (e.g.
    "Purpose and Permitted Use (NDA)" vs "Purpose and Permitted Use – NDA") —
    both keys collapse to the same alnum string. Only ever used for the
    title-label exclusion, never for content-quote verification, so
    loosening punctuation here can't let a fabricated content quote slip
    through — it can only better-recognize a real title being cited as a
    label.
    """
    return re.sub(r'[^a-z0-9]', '', s)


def _filter_verified_quotes(parsed: dict, source_text: str) -> dict:
    """Drop any 'quotes' entries that aren't actually verbatim substrings of the
    raw source text the ingest LLM was given.

    The ingest schema's "quotes" field is documented as "Exact verbatim quote
    from the text" — but the ingest LLM sometimes fills it with its own
    descriptive narration instead (e.g. "The petition cites this Act to
    demonstrate that the commercial court has jurisdiction over the matter").
    That narration then gets appended into the stored page content under a
    "**Supporting Quotes:**" heading, and the answer-generation LLM has no way
    to tell it apart from genuine document language — it ends up presented in
    quotation marks as if verbatim. This is the only point in the pipeline
    where the true original text is still available to check against, so
    filtering happens here, before the quote ever reaches storage.
    """
    def _norm(s: str) -> str:
        return _norm_for_match(s)

    src_norm = _norm(source_text)
    pages = parsed.get("pages", {})
    for title, page in pages.items():
        if not isinstance(page, dict):
            continue
        quotes = page.get("quotes", [])
        if not quotes:
            continue
        verified = [q for q in quotes if _norm(q) in src_norm]
        dropped = len(quotes) - len(verified)
        if dropped:
            logger.warning(
                "Ingest quote verification: dropped %d/%d unverifiable quote(s) for page '%s'",
                dropped, len(quotes), title,
            )
        page["quotes"] = verified
    return parsed


def _ingest_single_call(text: str, doc_name: str, family_key: str | None = None) -> dict:
    """Process a short document in one LLM call."""
    from services import family_prompt
    prompt = INGEST_PROMPT_TEMPLATE.format(
        text=text, doc_name=doc_name,
        family_block=family_prompt.build_supplement(family_key),
    )
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

    return _filter_verified_quotes(parsed, text)


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


def _ingest_detail_segment(text: str, topics: list[str], doc_name: str, doc_type: str,
                           family_key: str | None = None) -> dict:
    """Phase 2: extract detailed pages from a segment with known topic context."""
    from services import family_prompt
    topics_str = ", ".join(topics) if topics else "None identified yet"
    prompt = DETAIL_PROMPT_TEMPLATE.format(
        text=text, topics=topics_str, doc_name=doc_name, doc_type=doc_type,
        family_block=family_prompt.build_supplement(family_key, segment_mode=True),
    )
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

    return _filter_verified_quotes(parsed, text)


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
    wiki_id = _active_wiki_id()
    lock = _get_session_lock(session_id)
    # Collect (title, embed_text) pairs here; embed OUTSIDE the lock so HTTP
    # calls don't block other ingest threads waiting on the session lock.
    pages_to_embed: list[tuple[str, str]] = []
    # All pages in one merge call come from `doc_name`, so they share one family
    # (Phase 1) — resolved from doc_type in the metadata block below, then stamped
    # onto every embedding row so vector search can pre-filter by family at scale.
    doc_family_for_batch: str | None = None

    with lock:
        pages_updated = 0
        new_rels_count = 0
        contradictions_found: list[dict] = []

        new_pages = new_data.get("pages", {})
        new_relations = new_data.get("relations", [])

        # Build a short doc identifier for auto-prefixing unprefixed pages
        _doc_id = _make_doc_identifier(doc_name)

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

            # Auto-prefix unprefixed contract/agreement pages
            title = _auto_prefix_title(title, _doc_id)

            existing = _db.get_page(wiki_id, session_id, title)

            # Guard against title collisions between DIFFERENT source documents.
            # The ingest LLM sometimes invents the same entity-derived identifier
            # for two unrelated documents (e.g. two JVAs both involving parties
            # named "Aether"/"Helios" both get titled "... – JVA-HeliosAether").
            # _auto_prefix_title can't catch this — it looks like a real per-doc
            # prefix. Only a doc-specific page (has a contract-type parenthetical)
            # needs this guard; shared concept/statute pages are meant to merge
            # across documents. Without this, the second document's ingest
            # silently overwrites source_doc and clobbers the first document's
            # own numbers (dollar figures, equity splits, venue clauses, etc.)
            # under a shared row.
            if existing and existing.get("source_doc") not in (None, "", doc_name):
                _paren = re.search(r'\(([^)]+)\)\s*$', title)
                if _paren and _CONTRACT_DOC_TYPES.search(_paren.group(1)):
                    title = f"{title} #{_doc_id}"
                    existing = _db.get_page(wiki_id, session_id, title)

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
                # Amendment edges are consulted FIRST (§ 01 stage 04 — "contradiction
                # detection re-sequenced to run after amendment-chain edges"). If this
                # document amends the one that wrote the existing content, the two
                # disagreeing is what a version chain looks like, not a conflict.
                # Flagging it would put a resolved amendment in front of a reviewer as
                # an unresolved dispute — worse than noise, because it is wrong about
                # which text governs.
                _prior_doc = existing.get("source_doc") or ""
                _amended = False
                if _prior_doc and _prior_doc != doc_name:
                    _amended = _amendment_pair_cached(session_id, doc_name, _prior_doc)

                if (not _amended and len(new_content) > 200 and len(existing_content) > 200
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
                elif _amended and _has_structural_conflict(existing_content, new_content):
                    # Still recorded as a variant so the compaction pass sees both
                    # versions — the amendment supersedes the earlier text, and
                    # losing the earlier text would lose the chain itself.
                    from datetime import datetime
                    if not variants:
                        variants = [{"source": "Previous", "value": existing_content,
                                     "date_ingested": datetime.now().isoformat()}]
                    variants.append({"source": f"{doc_name} (amendment)",
                                     "value": new_content,
                                     "date_ingested": datetime.now().isoformat()})
                    logger.info("Page '%s': %s amends %s — recorded as a version "
                                "chain, not a contradiction", title, doc_name, _prior_doc)

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
                _db.upsert_page(wiki_id, session_id, title, merged_content, merged_summary, doc_name,
                                contradiction_flagged, variants)
                # Use the freshest summary for the embedding
                embed_text = (new_summary or existing_summary or new_content)[:400]
            else:
                _db.upsert_page(wiki_id, session_id, title, new_content, new_summary, doc_name, False, None)
                embed_text = (new_summary or new_content)[:400]

            pages_to_embed.append((title, embed_text))
            pages_updated += 1

        # -- C7: Persist document-level metadata extracted at ingest time --
        # Fold the inferred doc_type + normalized doc_family in alongside the
        # LLM-extracted metadata (Phase 0). doc_type is present on new_data for
        # both the single-call path (top-level prompt field) and the long-doc
        # path (injected in ingest() after the overview pass). Note the long-doc
        # path emits NO "metadata" object at all, so start from {} and still
        # persist doc_type/doc_family when that's all we have.
        metadata = new_data.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        _doc_type = new_data.get("doc_type")
        if _doc_type and isinstance(_doc_type, str):
            metadata["doc_type"] = _doc_type
            _fam = _normalize_doc_family(_doc_type)
            if _fam:
                metadata["doc_family"] = _fam
                doc_family_for_batch = _fam
        if metadata:
            try:
                _db.upsert_metadata(wiki_id, session_id, doc_name, metadata)
            except Exception as _me:
                logger.error("Metadata upsert failed for '%s': %s", doc_name, _me)

        # -- Merge explicit relations --
        for rel in new_relations:
            _db.upsert_relation(
                wiki_id, session_id, rel.get("from", ""), rel.get("to", ""), rel.get("label", "")
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
        existing_titles = _db.get_page_titles(wiki_id, session_id)
        existing_title_set = set(existing_titles)
        mention_rels: list[tuple[str, str, str]] = []
        for new_title, new_val in new_pages.items():
            new_content_for_xref = (
                new_val.get("content", "") if isinstance(new_val, dict) else str(new_val)
            )
            # Direction A: who already mentions this new title?
            try:
                mentioning = _db.find_pages_mentioning_title(wiki_id, session_id, new_title)
                for existing_title in mentioning:
                    mention_rels.append((existing_title, new_title, "mentions"))
            except Exception as _xref_err:
                logger.warning("FTS cross-ref failed for '%s': %s", new_title, _xref_err)
            # Direction B: which existing titles does the new page mention?
            for existing_title in existing_title_set:
                if existing_title != new_title and existing_title in new_content_for_xref:
                    mention_rels.append((new_title, existing_title, "mentions"))
        if mention_rels:
            _db.bulk_upsert_relations(wiki_id, session_id, mention_rels)

        for c in contradictions_found:
            _log_event(
                session_id,
                "CONTRADICTION",
                f"Page: {c['title']} | Claim: {c['claim']} | Source A: {c.get('val_a')} | Source B: {c.get('val_b')}",
            )

    # -- Embed pages OUTSIDE the lock (HTTP calls should not hold the session lock) --
    _embed_pages_batch(wiki_id, session_id, pages_to_embed, doc_family_for_batch)

    return pages_updated, new_rels_count, len(contradictions_found)


# ---------------------------------------------------------------------------
# Embedding helper (Phase 3) — called OUTSIDE the session lock
# ---------------------------------------------------------------------------
def _embed_pages_batch(wiki_id: str, session_id: str, pages_to_embed: list[tuple[str, str]],
                       doc_family: str | None = None) -> None:
    """Embed page summaries and store in page_embeddings table.

    Called AFTER the session lock is released so embedding HTTP calls don't
    block other ingest threads.  Failures are logged and swallowed — vector
    search falls back to BM25 gracefully for un-embedded pages.

    Args:
        pages_to_embed: list of (title, text_to_embed) where text_to_embed is
                        the page summary, or the first 400 chars of content
                        when no summary is available.
        doc_family: normalized family for every page in this batch (all pages in
                    a single merge come from one document), stamped onto each
                    embedding row so vector search can pre-filter by family
                    (Phase 1). None for documents whose type maps to no family.
    """
    if not config.USE_DATABASE or not pages_to_embed:
        return
    try:
        from services import embedder as _embedder
        texts = [text for _, text in pages_to_embed]
        embeddings = _embedder.embed_batch(texts, is_query=False)
        for (title, _), embedding in zip(pages_to_embed, embeddings):
            _db.upsert_embedding(wiki_id, session_id, title, embedding, doc_family)
        logger.info(
            "Embedded %d pages for session %s", len(pages_to_embed), session_id
        )
    except Exception as e:
        logger.error(
            "Embedding batch failed for session %s (%d pages skipped): %s",
            session_id, len(pages_to_embed), e,
        )


def backfill_source_docs(session_id: str) -> dict:
    """Populate empty source_doc fields by extracting the filename from page titles.

    Page titles follow the pattern:
        Topic Name (sessionid_path_to_filename.pdf)
    This extracts the filename portion and updates DB rows where source_doc is empty.
    """
    if not config.USE_DATABASE:
        return {"ok": False, "reason": "file mode — source_doc backfill is DB-only"}

    from sqlalchemy import text
    engine = _db.get_engine()
    updated = 0
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT title FROM pages WHERE session_id = :sid AND (source_doc IS NULL OR source_doc = '')"),
            {"sid": session_id},
        ).fetchall()
        for (title,) in rows:
            paren_start = title.rfind("(")
            paren_end = title.rfind(")")
            if paren_start < 0 or paren_end <= paren_start:
                continue
            raw_path = title[paren_start + 1:paren_end]
            # Strip session_id prefix (uuid_)
            parts = raw_path.split("_", 1)
            if len(parts) == 2 and len(parts[0]) == 36:
                raw_path = parts[1]
            # Extract just the filename (last path segment)
            filename = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
            if not filename:
                continue
            conn.execute(
                text("UPDATE pages SET source_doc = :sd WHERE session_id = :sid AND title = :title"),
                {"sd": filename, "sid": session_id, "title": title},
            )
            updated += 1
        conn.commit()
    logger.info("Backfilled source_doc for %d pages in session %s", updated, session_id)
    return {"ok": True, "updated": updated, "total_empty": len(rows)}


def backfill_embeddings(session_id: str, batch_size: int = 16) -> dict:
    """Generate embeddings for any pages in a session that lack them.

    Lets existing sessions (ingested before embeddings existed, or when the
    embedding API was rate-limited) gain pgvector hybrid retrieval without a full
    re-ingest. Embeds the page summary (or first 400 chars of content) — the same
    text used during normal ingest. Returns a summary dict.
    """
    if not config.USE_DATABASE:
        return {"ok": False, "reason": "file mode — embeddings are DB-only", "embedded": 0}

    wiki_id = _active_wiki_id()
    pages = _db.get_pages(wiki_id, session_id)
    if not pages:
        return {"ok": False, "reason": "no pages in session", "embedded": 0}

    existing = _db.count_embeddings(wiki_id, session_id)

    # Build the list of (title, text) for pages that need embedding.
    pending: list[tuple[str, str]] = []
    for title, page in pages.items():
        text = (page.get("summary") or page.get("content", "")[:400]).strip()
        if text:
            pending.append((title, text))

    if not pending:
        return {"ok": True, "reason": "nothing to embed", "embedded": 0,
                "total_pages": len(pages), "existing_embeddings": existing}

    embedded = 0
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        _embed_pages_batch(wiki_id, session_id, chunk)  # logs + swallows failures per batch
        embedded += len(chunk)

    final = _db.count_embeddings(wiki_id, session_id)
    logger.info("Backfill complete for session %s: %d embeddings now present (was %d)",
                session_id, final, existing)
    return {
        "ok": True,
        "total_pages": len(pages),
        "attempted": embedded,
        "existing_embeddings": existing,
        "embeddings_now": final,
    }


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

    wiki_id = _active_wiki_id()
    _db.reset_page_after_compaction(wiki_id, session_id, title, new_content, new_summary, contradiction_flagged)

    # Store structured contradictions (S4)
    for c in detected_contradictions:
        try:
            _db.upsert_contradiction(
                wiki_id, session_id, title,
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
    _embed_pages_batch(wiki_id, session_id, [(title, embed_text)])

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
        _active_wiki_id(), session_id,
        config.COMPACTION_APPEND_THRESHOLD,
        config.COMPACTION_CHAR_THRESHOLD,
    )
    if not due:
        return 0

    logger.info("Compaction: %d pages due for session %s", len(due), session_id)
    compacted = 0
    for page_data in due:
        title = page_data["title"]
        # Per-page lock (§ 01.6 Concurrency). Two concurrent ingests can both
        # push the same page past the threshold and both start re-synthesising
        # it — two LLM calls producing two competing rewrites, one of which
        # silently overwrites the other. The lock is per page rather than per
        # session so unrelated pages still compact in parallel.
        try:
            with _db.page_compaction_lock(session_id, title) as acquired:
                if not acquired:
                    logger.info("Compaction: '%s' already being compacted "
                                "elsewhere — skipping", title)
                    continue
                # Re-read under the lock: the holder we just waited behind may
                # have already compacted this page, in which case the row we
                # were handed is stale and recompacting would burn a call to
                # rewrite something already rewritten.
                fresh = _db.get_page(_active_wiki_id(), session_id, title)
                if fresh and fresh.get("append_count", 0) < config.COMPACTION_APPEND_THRESHOLD \
                        and len(fresh.get("content", "")) < config.COMPACTION_CHAR_THRESHOLD:
                    logger.info("Compaction: '%s' no longer due after lock wait", title)
                    continue
                _compact_page(session_id, title, dict(fresh or page_data))
                compacted += 1
        except Exception as e:
            logger.error("Compaction failed for page '%s': %s", title, e)

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


def _distinct_source_docs(pages: dict) -> set[str]:
    """Return the set of distinct raw source_doc values across all pages."""
    docs: set[str] = set()
    for page in pages.values():
        if isinstance(page, dict):
            sd = page.get("source_doc", "")
            if sd:
                docs.add(sd)
    return docs


# Broad, cross-document question detector — used only to widen and diversify the
# hybrid vector-search candidate pool. A flat top-K nearest-neighbour search has
# no document-diversity awareness: for "across all Service Agreements" style
# questions on a corpus with 7+ matching documents, the top-K can fill up with
# pages from just 2-3 documents whose clauses happen to be lexically closest,
# silently starving the rest even though they're all relevant. Confirmed live
# (500-doc session): cross-SA liability-caps and cross-court-case digital-
# evidence questions both synthesized only 2-3 of 7+ relevant documents.
# "across the court case documents" is as broad as "across all court case
# documents", but only the bare "across the corpus|documents" form matched — so
# a typed plural ("across the court case documents, what digital evidence…")
# fell through to disambiguation and got answered from ONE document instead of
# synthesised across all of them. Allow up to 3 intervening type words before
# the plural noun.
_BROAD_SCOPE_RE = re.compile(
    r'\bacross all\b|\ball of the\b|\beach of the\b|'
    r'\bacross the (?:\w+\s+){0,3}'
    r'(?:corpus|documents?|agreements?|ndas?|judge?ments?|opinions?|cases?)\b|'
    r'\bevery (service agreement|shareholders? agreement|joint venture agreement|judge?ment|court case)\b',
    re.IGNORECASE,
)


_PARTIES_TITLE_RE = re.compile(r'^parties\b', re.IGNORECASE)


def _diversify_by_document(titles: list[str], pages: dict, per_doc_cap: int, total_cap: int) -> list[str]:
    """Cap how many pages from any single source_doc can occupy the candidate
    list, preserving similarity order, so a broad question's page budget gets
    spread across documents instead of concentrating on the closest few.

    Also force-includes each document's "Parties" identity page alongside its
    semantically-closest clause page. Without this, a per-document cap of 1
    page picks only whichever clause page matches the query topic (e.g.
    "Limitation of Liability"), which never contains party names — the model
    still produces correct party names (evidently carried over from earlier
    turns in the same long session), but the grounding checker then flags
    them as unsupported by *this* turn's context, since they genuinely aren't
    in it. Confirmed live: spot-checked 3 "ungrounded" party pairings the
    grounding check flagged, all 3 were factually exact — a retrieval gap,
    not a fabrication.
    """
    parties_by_doc: dict[str, str] = {}
    for t, p in pages.items():
        if isinstance(p, dict) and _PARTIES_TITLE_RE.match(t):
            sd = p.get("source_doc", "")
            if sd and sd not in parties_by_doc:
                parties_by_doc[sd] = t

    per_doc_count: dict[str, int] = {}
    result: list[str] = []
    seen_docs: set[str] = set()
    for t in titles:
        page = pages.get(t)
        sd = page.get("source_doc", "") if isinstance(page, dict) else ""

        if sd not in seen_docs:
            seen_docs.add(sd)
            parties_title = parties_by_doc.get(sd)
            if parties_title and parties_title != t and len(result) < total_cap:
                result.append(parties_title)
                per_doc_count[sd] = per_doc_count.get(sd, 0) + 1

        if per_doc_count.get(sd, 0) >= per_doc_cap or t in result:
            continue
        result.append(t)
        per_doc_count[sd] = per_doc_count.get(sd, 0) + 1
        if len(result) >= total_cap:
            break
    return result


def _norm_doc_name(name: str) -> str:
    """Normalise a source-doc path/filename to a comparable lowercase string.

    "<uuid>_Legal AI Tool .../Service Agreement 2_redacted.pdf" → "service agreement 2"
    """
    s = name.replace("\\", "/").rsplit("/", 1)[-1]      # basename
    s = re.sub(r'^[a-f0-9-]{36}_', '', s)               # strip session-id prefix
    s = os.path.splitext(s)[0]                          # drop extension
    s = s.replace('_', ' ').lower()
    s = re.sub(r'\b(redacted|test|final|draft|copy|v\d+)\b', ' ', s)
    # Browser/OS duplicate-download folder names get "(1)", "(2)" etc. appended
    # (e.g. a folder re-downloaded as "Service Agreement (1)") — this parenthesized
    # number is a filesystem artifact, not a document identifier, but every file
    # in that folder inherits it into source_doc. Left in, "\b1\b" in the numbered-
    # pattern matcher below spuriously matches this artifact on EVERY file in the
    # folder (since "(" / ")" are non-word chars, "\b1\b" matches inside "(1)"),
    # flooding "service agreement 1" to match all 67 docs in the folder instead of
    # zero (confirmed live: a session with no real "Service Agreement 1" doc still
    # matched every "Service Agreement (1)_..." file and pulled in ~995 pages).
    s = re.sub(r'\(\d+\)', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _numbered_docs_in(question: str, doc_names) -> set[str]:
    """Which of ``doc_names`` the question names by document type + number.

    Factored out of ``_detect_mentioned_files`` so the same matching can run
    against an arbitrary name set — notably the UPLOAD list, which (unlike the
    page index) still contains documents that ingested with zero pages.

    A question can name MORE THAN ONE document this way (e.g. "compare Tata
    Brand Judgment 6 and Tata Brand Judgment 8") — iterate every match rather
    than just the first, and pair each number with its OWN adjacent type word
    rather than reusing whichever type happened to appear earliest in the
    question. Confirmed live: a question naming "Judgment 6" and "Judgment 8"
    only force-included Judgment 6 (the second document was silently dropped to
    generic retrieval, which pulled an unrelated document instead).
    """
    matched: set[str] = set()
    # Underscore-normalise before matching: the corpus's own synthetic filenames
    # are underscore-joined ("Test_SHA_01.txt"), and users echo that shorthand
    # back ("sha_01") — but "_" is a \w char, so it doesn't satisfy the \s*
    # gap between type and number in _DOC_NAME_PATTERN and the match silently
    # fails, same length so match.end() offsets used below stay valid.
    question = question.replace('_', ' ')
    for num_match in _DOC_NAME_PATTERN.finditer(question):
        t = num_match.group(1).lower()
        t = re.sub(r'\s+', ' ', t).strip()
        # Distinctive core token the filename must contain (see _DOC_TYPE_CORE) —
        # NOT the first word, which for "legal opinion" is the non-distinctive
        # "legal" that prefixes every source_doc.
        type_core = _DOC_TYPE_CORE.get(t, t.split()[0])
        # A shorthand numbered list shares ONE type word across several numbers:
        # "Service Agreement 1, 2 & 3", "compare NDA 1, 2, and 3". The base
        # pattern only binds the type to the FIRST number, so 2 and 3 were
        # silently dropped and the "comparison" ran on a single document
        # (confirmed live: "Compare the Service Agreement 1, 2 & 3" retrieved
        # only SA 1). Consume the trailing ", N"/"& N"/"and N" run and apply the
        # same type to each number.
        nums = [num_match.group(2)]
        tail = question[num_match.end():]
        while True:
            m = re.match(r'\s*(?:,|&|and)\s*(\d+)', tail)
            if not m:
                break
            nums.append(m.group(1))
            tail = tail[m.end():]
        for doc_num in nums:
            # Match the number allowing zero-padding: the user types "service
            # agreement 1" but redacted test files are saved zero-padded as
            # "Test_SA_01" (norm → "... sa 01"), so a bare \b1\b never matched and
            # the document was treated as non-existent. (?<!\d)0*N(?!\d) matches
            # "01" and "1" but NOT "10"/"11"/"21" — the surrounding digit guards
            # keep it from bleeding into a different document number.
            num_re = rf'(?<!\d)0*{re.escape(doc_num)}(?!\d)'
            for sd in doc_names:
                norm = _norm_doc_name(sd)
                if re.search(num_re, norm) and (not type_core or type_core in norm):
                    matched.add(sd)
    return matched


# A synthetic corpus document, named "Test_<TYPE>_<NN>" (e.g. "Test_SHA_01",
# "Test_JVA_04", "Test_Opinion_35"). These sit ALONGSIDE the real documents in
# the same upload folder ("Legal AI - Test_Shareholder Agreements (1)"), and the
# real ones are NOT so-named (they are "Shareholder Agreement 1_redacted", "NDA 7
# _Redacted", "Court Case Document 5", "Tata Brand Judgment 6", etc.). The user's
# convention is that the real documents are the ones that matter; the Test_* files
# are fictional stand-ins. NOTE the leading separator class instead of \b — the
# uploader underscore-joins the folder path into source_doc, so the marker is
# preceded by "_" (a word char), and \b never fires between "_" and "test".
_SYNTHETIC_DOC_RE = re.compile(r'(?:^|[_ /\\-])test_[a-z]+_\d+(?![a-z])', re.I)


def _is_synthetic_test_doc(source_doc: str) -> bool:
    """True when source_doc is a synthetic "Test_<TYPE>_<NN>" stand-in, not a
    real corpus document."""
    base = re.sub(r'^[a-f0-9-]{36}_', '', source_doc.replace("\\", "/").rsplit("/", 1)[-1])
    return bool(_SYNTHETIC_DOC_RE.search(base))


def _numbered_doc_collisions(question: str, doc_names) -> list[str]:
    """Human-readable labels for each numbered reference in the question that
    matches BOTH a real document and a synthetic Test_* sibling.

    "SHA 1" matches the real "Shareholder Agreement 1_redacted" AND the synthetic
    "Test_SHA_01" (same type + number, different naming scheme) — both get pinned
    into context, and the answer LLM then silently answers from whichever has the
    richer content, usually the synthetic one, with no indication it switched
    documents (confirmed live: a GridEdge SHA question was answered entirely from
    Test_SHA_01's fictional Aether/Helios/Apex parties and $1,000,000 veto
    threshold, none of which belong to the real GridEdge agreement). Mirrors the
    per-number matching loop in _numbered_docs_in so the collision is detected
    against the exact same references the user named — no behaviour change to the
    matcher itself, this is a read-only overlay used to warn the user.
    """
    collisions: list[str] = []
    for num_match in _DOC_NAME_PATTERN.finditer(question):
        t = re.sub(r'\s+', ' ', num_match.group(1).lower()).strip()
        type_core = _DOC_TYPE_CORE.get(t, t.split()[0])
        nums = [num_match.group(2)]
        tail = question[num_match.end():]
        while True:
            m = re.match(r'\s*(?:,|&|and)\s*(\d+)', tail)
            if not m:
                break
            nums.append(m.group(1))
            tail = tail[m.end():]
        for doc_num in nums:
            num_re = rf'(?<!\d)0*{re.escape(doc_num)}(?!\d)'
            hits = [
                sd for sd in doc_names
                if re.search(num_re, _norm_doc_name(sd))
                and (not type_core or type_core in _norm_doc_name(sd))
            ]
            has_synthetic = any(_is_synthetic_test_doc(sd) for sd in hits)
            has_real = any(not _is_synthetic_test_doc(sd) for sd in hits)
            if has_synthetic and has_real:
                label = f"{t} {doc_num}"
                if label not in collisions:
                    collisions.append(label)
    return collisions


def _uploaded_doc_names(session_id: str) -> set[str]:
    """Every document UPLOADED to this session, read from the upload directory.

    The page index only knows documents that produced at least one page. A
    document whose ingest yielded nothing — e.g. a scanned PDF whose OCR failed
    — is absent from it entirely, so no page-index query can ever reveal that
    the user's named document exists but is empty. The upload directory is the
    only record that they supplied it at all.

    Returns names with the ``{session_id}_`` prefix stripped; ``_norm_doc_name``
    normalises these to the same string as the corresponding indexed
    ``source_doc``, so the two sets are directly comparable.
    """
    try:
        prefix = f"{session_id}_"
        return {
            fn[len(prefix):] for fn in os.listdir(config.UPLOAD_PATH)
            if fn.startswith(prefix)
        }
    except Exception as e:
        logger.error("_uploaded_doc_names failed: %s", e)
        return set()


def _detect_mentioned_files(question: str, pages: dict) -> set[str]:
    """Detect which SPECIFIC source documents the user is asking about.

    Matches the question against each distinct ``source_doc`` (filename), not
    against the document *type* — so "service agreement 2" scopes to exactly that
    document instead of every service agreement. Returns a set of raw source_doc
    strings (use ``_pages_from_files`` to expand to page titles).

    Priority:
      1. Numbered type pattern ("service agreement 2", "NDA 3") → the doc whose
         normalised name contains that type word AND that number.
      2. Distinctive full-name match (the doc's normalised name appears verbatim).
    """
    src_docs = _distinct_source_docs(pages)
    if not src_docs:
        return set()

    q = " " + re.sub(r'\s+', ' ', question.lower()).strip() + " "
    matched: set[str] = set()

    # 0. Raw filename mention (e.g. "Test_JVA_01.txt") — basename match.
    # Bypasses _norm_doc_name's underscore-to-space and "test"-word stripping,
    # which otherwise turns "Test_JVA_01.txt" into "jva 01" and never matches
    # the literal underscore-joined filename text the user pasted.
    #
    # Uses endswith rather than exact equality: the website uploader flattens
    # the original folder path into the saved filename with underscores (e.g.
    # source_doc = "<session-uuid>_Joint Venture Agreements_Test_JVA_01.txt"),
    # so stripping only the leading UUID still leaves a folder-name prefix
    # ("Joint Venture Agreements_") in front of the real filename.
    raw_mentions = re.findall(r'\b([\w\-]+\.(?:txt|pdf|docx))\b', question, re.IGNORECASE)
    if raw_mentions:
        mentions_lower = [m.lower() for m in raw_mentions]
        for sd in src_docs:
            basename = re.sub(r'^[a-f0-9-]{36}_', '', sd.replace("\\", "/").rsplit("/", 1)[-1]).lower()
            if any(basename.endswith(m) for m in mentions_lower):
                matched.add(sd)
        if matched:
            logger.info("Detected file mention (raw filename): %s", matched)
            return matched

    # 1. Numbered type pattern — precise per-document scoping.
    matched |= _numbered_docs_in(question, src_docs)
    if matched:
        logger.info("Detected file mention (numbered): %s", {_norm_doc_name(d) for d in matched})
        return matched

    # 2. Distinctive full-name match (e.g. user pastes the exact doc name)
    for sd in src_docs:
        norm = _norm_doc_name(sd)
        if len(norm) >= 6 and norm in q:
            matched.add(sd)
    if matched:
        logger.info("Detected file mention (name): %s", {_norm_doc_name(d) for d in matched})

    return matched


def _pages_from_files(pages: dict, source_docs: set[str]) -> list[str]:
    """Return page titles whose ``source_doc`` is in the given set."""
    result = []
    for title, page in pages.items():
        if isinstance(page, dict) and page.get("source_doc", "") in source_docs:
            result.append(title)
    return result


# Ingest writes every page prose-first, verbatim-evidence-last, under this
# heading. That layout and a plain content[:cap] tail-slice are in direct
# conflict: the slice always eats the quotes and always keeps the paraphrase —
# the exact inversion of what an answer needs. Measured on the Hyden-Lexus MSA
# at the 2,000-char cap: "Limitation of Liability and Carve-outs" (2,474 chars)
# lost the tail of its own Clause 10.4 quote mid-sentence, so the INR
# 15,00,00,000 figure never reached the model and the answer reported the cap
# as "not fully reproduced in the provided context" while the database held it
# verbatim. "Intellectual Property Allocation" (2,232) lost Clauses 5.4 and 5.5
# the same way, and "AI Governance" (2,137) lost Clause 6.7.
#
# Two downstream effects, not one. The obvious one is the missing fact. The
# quieter one is that the model, left with prose only, paraphrases it and
# _verify_answer_citations then cannot match that paraphrase to any retrieved
# quote — which is where the recurring "[CITATION NOTE: excerpt could not be
# matched to the retrieved source text]" banners come from. Both are fixed by
# spending the budget on evidence instead of summary.
_SUPPORTING_QUOTES_RE = re.compile(r'\n\*\*Supporting Quotes:\*\*[ \t]*\n', re.IGNORECASE)

# Always keep at least this much summary. The quotes alone are clause text with
# no framing; the opening prose is what tells the model which document and
# which topic they belong to.
_MIN_PROSE_CHARS = 400


def _truncate_page_content(content: str, cap: int) -> str:
    """Trim an over-long page to ``cap`` chars without destroying its quotes.

    Trims the prose summary and keeps the Supporting Quotes block whole. When
    the quotes alone cannot fit, drops WHOLE quote lines from the end rather
    than cutting one mid-sentence — a half-quote is worse than an absent one,
    because it reads as complete and is what the model then cites.

    Falls back to the old tail-slice for pages with no quotes block (cached
    "Q:" answers, older pages ingested before the format settled).
    """
    if len(content) <= cap:
        return content

    m = _SUPPORTING_QUOTES_RE.search(content)
    if not m:
        return content[:cap] + "\n[...truncated]"

    prose = content[:m.start()]
    header = content[m.start():m.end()]
    body = content[m.end():]

    note = "\n[...summary trimmed — supporting quotes kept in full]"
    prose_budget = cap - (len(header) + len(body)) - len(note)
    if prose_budget >= _MIN_PROSE_CHARS:
        return prose[:prose_budget].rstrip() + note + header + body

    # Quotes alone overflow the cap: keep the minimum prose, then as many whole
    # quote lines as fit. Quotes appear in document order, so truncating from
    # the end keeps the earliest clauses — the ones the page is titled for.
    note = "\n[...truncated — later supporting quotes omitted]"
    avail = cap - _MIN_PROSE_CHARS - len(header) - len(note)
    kept: list[str] = []
    used = 0
    for line in body.split("\n"):
        if used + len(line) + 1 > avail:
            break
        kept.append(line)
        used += len(line) + 1
    if not kept:
        return content[:cap] + "\n[...truncated]"
    return prose[:_MIN_PROSE_CHARS].rstrip() + header + "\n".join(kept) + note


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


# "... of the SOW ... AS STATED IN the Power of Attorney between ...". The
# marker splits a question into the provision being asked for and the document
# the question claims states it.
_CROSS_REF_RE = re.compile(
    r'\bas\s+(?:stated|set\s+out|recorded|described|provided|specified)\s+in\s+the\s+',
    re.IGNORECASE,
)


def _cross_reference_identity(question: str, pages: dict,
                              selected_titles: list) -> tuple[str, str] | None:
    """(citing_label, parties) for a cross-reference the retrieval does not bear
    out, or None when the question makes no such claim or the claim is
    satisfiable.

    Deliberately conservative: fires only when the CITING document — the one
    after "as stated in" — is absent from the retrieved pages entirely. A
    document that WAS retrieved may genuinely quote the other's clause, and
    deciding that is the answer model's job, not a regex's. Shared by
    _failed_cross_reference (the context-injected warning) and generate_answer's
    deterministic override (see there for why a warning alone was not enough).
    """
    m = _CROSS_REF_RE.search(question or "")
    if not m:
        return None
    citing = question[m.end():]
    cited_names = [n.group(1).strip() for n in _PARTY_NAME_RE.finditer(citing)]
    if not cited_names:
        return None
    # The instrument the question says does the citing, for a readable message.
    citing_type = re.match(r'([A-Za-z][A-Za-z /\-]{2,60}?)\s+between\b', citing)
    citing_label = citing_type.group(1).strip() if citing_type else "second document"

    retrieved_docs = {
        (pages.get(t) or {}).get("source_doc", "")
        for t in (selected_titles or []) if isinstance(pages.get(t), dict)
    }
    haystack = " ".join(_norm_doc_name(d).lower() for d in retrieved_docs if d)
    # Present if the retrieved filenames carry a distinctive word of either
    # party named as the citing document's parties.
    for name in cited_names:
        token = _distinctive_party_token(name)
        if token and token.lower() in haystack:
            return None
    return citing_label, " and ".join(cited_names[:2])


def _failed_cross_reference(question: str, pages: dict,
                            selected_titles: list) -> str | None:
    """A context-injected warning naming a cross-reference the retrieval does
    not bear out, or None. See generate_answer for the deterministic backstop
    this warning alone turned out not to be sufficient on its own."""
    identity = _cross_reference_identity(question, pages, selected_titles)
    if not identity:
        return None
    citing_label, parties = identity
    return (f"the question asks for a provision \"as stated in\" the {citing_label} "
            f"between {parties}, and that document is NOT among the retrieved pages. "
            f"Its cross-reference therefore cannot be confirmed. Say plainly that the "
            f"two documents are unrelated and that it does not contain the provision "
            f"asked about.")


def _cross_reference_failure_answer(question: str, pages: dict,
                                    selected_titles: list) -> str | None:
    """The complete, deterministic answer to a question whose cross-reference
    is proven unsatisfiable, or None when the question makes none.

    Confirmed live (Q01212) that a context-injected warning is not sufficient
    on its own: the model wrote a compliant opening sentence saying the
    cross-reference could not be confirmed, then added a SECOND section quoting
    the first document's own governing-law clause anyway, under its own
    heading, offered "for completeness." That reads to a user as though the
    cross-reference held. The warning is a real signal (competing against many
    other prompt rules, as the earlier fix for this same class of question
    found), but whether the citing document was retrieved is a fact this
    function already knows with certainty — nothing is gained by asking the
    model to also arrive at it and then trusting it not to hedge past that
    fact. So this bypasses generation entirely for this one question shape:
    no LLM call, and no model output for a compliance check to fail.
    """
    identity = _cross_reference_identity(question, pages, selected_titles)
    if not identity:
        return None
    citing_label, parties = identity
    return (f"These two documents are unrelated. The {citing_label} between "
            f"{parties} is not among the documents used to answer this "
            f"question, and it does not contain the provision asked about. "
            f"The question's premise — that this second document states or "
            f"records that provision — does not hold.")


# The "current value under the agreement family" shape asks for the value AFTER
# amendment; its mirror ("in the original agreement, before it was amended by
# ...") asks for the one that was replaced. Only the first is redirected here.
_AS_AMENDED_RE = re.compile(
    r'\b(?:current|currently|after\s+giving\s+effect|as\s+amended|'
    r'now\s+in\s+force|presently)\b',
    re.IGNORECASE,
)

# Both documents recite a date, and the date is the only thing that tells them
# apart — same instrument family, same two parties.
_DATED_RE = re.compile(
    r'\bdated\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|'
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    re.IGNORECASE,
)


def _amendment_family_directive(question: str) -> str | None:
    """Say which of an amendment family's two documents states the value asked for.

    Scope resolution now retrieves both (see _expand_amendment_family), which
    leaves the model holding two documents that answer the same question with
    two different figures and no stated rule for choosing. The question itself
    carries the rule — "after giving effect to this amendment" — but it is one
    clause at the end of a long sentence, and the original is the document named
    first and quoted at greater length.

    Returns None for the mirror shape, which wants the superseded value and must
    not be pointed at the amendment.
    """
    m = _AMENDMENT_TAIL_RE.search(question or "")
    if not m or not _AS_AMENDED_RE.search(question or ""):
        return None
    before, after = question[:m.start()], question[m.end():]
    original_date = (_DATED_RE.findall(before) or [""])[-1]
    amend_date = (_DATED_RE.findall(after) or [""])[0]
    if not amend_date or amend_date == original_date:
        return None
    orig_label = (f"the original dated {original_date}" if original_date
                  else "the original agreement")
    return (f"this question names TWO documents and asks for the value that governs "
            f"AFTER amendment. The answer is the value stated in the AMENDMENT dated "
            f"{amend_date}, not the one in {orig_label} — the original states the "
            f"figure the amendment replaced. Note that the amendment's date may be "
            f"EARLIER than the original's; the question says which document amends "
            f"which, and that is what decides it, not which date is later. Quote the "
            f"amendment as the source. If the amendment is not among the retrieved "
            f"pages, say that plainly rather than answering from the original.")


def get_context(question: str, session_id: str, target_doc: str = "", retrieval_hints: dict = None,
                 exclude_cached_answers: bool = False,
                 doc_family: "str | list[str] | None" = None, force_broad: bool = False,
                 force_docs: "list[str] | None" = None,
                 family_docs: "list[str] | None" = None) -> tuple[str, list]:
    """Select relevant pages for a query and return them as a formatted string + list of titles.

    If the question mentions a specific source file (e.g. "Legal Opinion 2.pdf"),
    all pages originating from that file are force-included so the answer stays
    grounded in the correct document.

    doc_family / force_broad (Phase 2): forwarded from the resolved scope to the
    hybrid page-selection path — doc_family pre-filters the pgvector search to a
    document family, force_broad widens+diversifies the candidate pool. Both are
    inert (None / False) for single-document and default corpus scopes, keeping
    behaviour identical to before Phase 2 for those cases.

    exclude_cached_answers: when True, drops cached "Q:" answer pages (see
    generate_answer()'s answer-filing step) from consideration entirely, so a
    question isn't answered by echoing a previous answer to the same/similar
    question. Intended for QA/testing repeat-asks — a real user benefits from
    the cache, but retesting the same question after a fix needs a guaranteed
    fresh generation to actually observe whether behavior changed. Forced on
    whenever config.ENABLE_ANSWER_CACHE is off, so disabling the feature also
    hides the pages earlier runs already filed.
    """
    index = _load_index(session_id)
    pages = index.get("pages", {})
    # Turning the cache OFF has to hide the pages already filed, not just stop
    # writing new ones: a session that has answered questions before still holds
    # "Q:" pages, and leaving them retrievable means the feature is still running
    # on everything it wrote earlier. One effective flag so both channels — the
    # in-memory pool here and the pgvector ranking below — agree.
    exclude_cached_answers = exclude_cached_answers or not config.ENABLE_ANSWER_CACHE
    if exclude_cached_answers:
        pages = {t: p for t, p in pages.items() if not t.startswith("Q:")}

    if not pages:
        return {"context": "", "selected_titles": [], "bm25_count": 0}

    # --- Step 0: Detect file mentions in the question or use target_doc ---
    # Scope resolution (resolve_scope) may have already pinned specific documents
    # by party-name content match — documents the in-question detectors below
    # cannot find (party masked in metadata, filed under a bare type+number).
    # Honour that pin first and scope STRICTLY to those documents' own pages
    # (same strict, supplementary-free treatment as an explicit target_doc): the
    # user named one specific agreement, so cross-document pages are contamination.
    forced_set = {d for d in (force_docs or []) if d}
    forced_pages = [
        title for title, page in pages.items()
        if isinstance(page, dict) and page.get("source_doc", "") in forced_set
    ] if forced_set else []
    strict_scope = False
    if forced_pages:
        mentioned_files = forced_set
        file_pages = forced_pages
        strict_scope = True
        logger.info("Scope-pinned to %d document(s) by party match: %d page(s)",
                    len(forced_set), len(file_pages))
    elif target_doc:
        # Match by source_doc field (DB) or by title substring (file-based)
        file_pages = [
            title for title, page in pages.items()
            if isinstance(page, dict) and (
                page.get("source_doc", "") == target_doc
                or target_doc in page.get("source_doc", "")
                or target_doc in title
            )
        ]
        mentioned_files = {target_doc} if file_pages else set()
    else:
        mentioned_files = _detect_mentioned_files(question, pages)
        file_pages = _pages_from_files(pages, mentioned_files) if mentioned_files else []

        # Fallback: if no file-level mention, check if the question names a known
        # entity from a document identifier (e.g. "ReVolt", "Meridian", "Yuvraj
        # Kanther") and force-include those pages so the answer is correctly scoped.
        if not file_pages:
            matched_titles = _pages_matching_question_entity(question, pages)
            # Scope resolution already established which document FAMILY the
            # question is about; the entity fallback must stay inside it. It
            # matches on a party name alone, so an umbrella party pulls in pages
            # from every instrument it appears in — and because these pages are
            # FORCE-included below, they bypass the family pre-filter applied to
            # the vector search. Measured on Q21 ("the Service Agreement of Tata
            # Steel Limited"): the scope resolved to the Service Agreement family,
            # the entity fallback force-included NDA 7 pages anyway, and the answer
            # reported a "thirty (30) days" notice period that appears nowhere in
            # Service Agreement 7 — whose own page says the term is fifteen months
            # and convenience termination is "upon notice" with no day count.
            if matched_titles and family_docs:
                _fam_set = {d for d in family_docs if d}
                _kept = [
                    t for t in matched_titles
                    if isinstance(pages.get(t), dict)
                    and pages[t].get("source_doc", "") in _fam_set
                ]
                if len(_kept) != len(matched_titles):
                    logger.info("Entity match: dropped %d of %d page(s) outside the "
                                "resolved document family",
                                len(matched_titles) - len(_kept), len(matched_titles))
                matched_titles = _kept
            # A distinctive entity ("ReVolt", "Yuvraj Kanther") should only match a
            # handful of pages. If it matches a huge slice of the wiki, the "entity"
            # is actually a common party name reused across many unrelated documents
            # (e.g. "Aether"/"Helios" appearing in most of a synthetic test corpus) —
            # forcing all of them in would blow the context budget. Fall through to
            # normal hybrid vector/BM25 selection instead.
            if matched_titles and len(matched_titles) <= config.ENTITY_MATCH_MAX_PAGES:
                file_pages = matched_titles
                logger.info("Entity-matched %d pages from question", len(file_pages))
            elif matched_titles:
                logger.warning(
                    "Entity match found %d pages (> %d cap) — treating as too broad, "
                    "falling back to normal page selection",
                    len(matched_titles), config.ENTITY_MATCH_MAX_PAGES,
                )

    bm25_count = 0
    pages_for_llm = pages
    page_selection_usage: dict = {}

    # --- Step 1: Select relevant pages ---
    if file_pages and (target_doc or strict_scope):
        # Explicit single-document scope: either a UI pin (target_doc — the
        # "summarise this document" flow and the disambiguation folder-picker) or
        # a party-name content match resolved upstream (strict_scope). The user
        # named exactly ONE document, so supplementary cross-document pages are pure
        # contamination here: an isolation test on the Test_JVA_05 summary showed the
        # supplementary pass dragged in ~30 boilerplate clause pages from an unrelated
        # Source Code Escrow Agreement, which the answer LLM then variably cited AS
        # JVA5 — both a correctness bug (misattribution) and the primary driver of the
        # run-to-run citation-warning non-determinism. Scope strictly to the pinned
        # document's own pages; skip supplementary retrieval entirely.
        selected_titles = file_pages
        # Multi-document strict scope ("the agreement between X and Y" resolving
        # to several instruments) force-includes EVERY page of EVERY pinned
        # document with no relevance ranking, and the char-budget loop below
        # truncates selected_titles in whatever order they arrive in — which is
        # document-enumeration order, not relevance order. Confirmed live: a
        # 3-document, 55-page scope (Consulting Agreement + Amendment + the
        # actual IT Outsourcing Agreement) exceeded the 60k budget by 13 pages;
        # those 13 were the IOA's LAST pages because it happened to be enumerated
        # last, and one of them — "Retention, Escrow and Indemnity" — was the
        # page carrying the liability cap the question asked about. The answer
        # reported no cap while the correct document was fully in scope and
        # simply never reached. The structured-extraction block below already
        # solves this exact failure for clause-table content via `_q_overlap`
        # (see the Term Sheet / Relationship Of Parties incident in that
        # comment) — this applies the identical fix to the raw page text, which
        # was the one channel it was never extended to. Single-document scope is
        # deliberately left untouched: reordering one document's own pages could
        # break a coherent read-through, and there is no cross-document budget
        # race to fix when there is only one document.
        if len(forced_set) > 1:
            _q_tokens = {w for w in re.findall(r'[a-z0-9]{3,}', (question or "").lower())
                        if w not in _NARROW_TOKEN_STOPWORDS}
            if _q_tokens:
                def _page_overlap(_t: str) -> int:
                    _p = pages.get(_t)
                    _c = _p.get("content", "") if isinstance(_p, dict) else (_p or "")
                    return len(_q_tokens & set(re.findall(r'[a-z0-9]{3,}', (_t + " " + _c).lower())))
                _before = list(selected_titles)
                selected_titles = sorted(selected_titles, key=lambda t: -_page_overlap(t))
                if selected_titles != _before:
                    logger.info("Multi-document strict scope (%d docs, %d pages): reordered by "
                                "question-term overlap so a budget cut drops the least relevant "
                                "pages first, not whichever document was enumerated last",
                                len(forced_set), len(selected_titles))
        logger.info("Single-document scope (%s): scoped to %d page(s), supplementary retrieval skipped",
                     target_doc or f"party:{sorted(forced_set)}", len(file_pages))
        _trace = tracing.get_trace()
        if _trace:
            _trace.log_page_selection(
                "pinned to document(s)" if strict_scope else "pinned to target_doc",
                selected=file_pages, pinned_docs=sorted(forced_set) if strict_scope else [target_doc],
            )
    elif file_pages:
        # File mention detected from the question text (not an explicit UI pin) — force
        # those pages in, then add topic-collision-filtered supplementary pages.
        #
        # Guard against same-named-entity/topic collisions across UNRELATED documents
        # (e.g. two different test JVAs both naming their JV "SolarNexus ... LLC" with
        # different parties/numbers, or a small test session where every ingested doc
        # shares generic clause topics like "Intellectual Property"). If a candidate
        # supplementary page covers the same clause topic as one already force-included
        # from the named document, but belongs to a DIFFERENT source document, drop it —
        # otherwise its numbers get blended into the requested document's answer and
        # misattributed to it. Applied regardless of wiki size — small test sessions are
        # exactly where this contamination is most visible (few pages, generic topics).
        def _topic_of(title: str) -> str:
            dash = title.find(" – ")
            return title[:dash].strip() if dash > 0 else title

        file_doc_set = {
            pages[t].get("source_doc", "")
            for t in file_pages
            if isinstance(pages.get(t), dict) and pages[t].get("source_doc")
        }
        file_topics = {_topic_of(t) for t in file_pages}
        seen = set(file_pages)

        def _drop_colliding(candidates: list[str]) -> list[str]:
            kept = []
            for t in candidates:
                if t in seen:
                    continue
                p = pages.get(t)
                sd = p.get("source_doc", "") if isinstance(p, dict) else ""
                if _topic_of(t) in file_topics and sd and sd not in file_doc_set:
                    logger.info("Dropping supplementary page %r — same topic as forced "
                                "document but from a different source_doc (%s)", t, sd)
                    continue
                kept.append(t)
            return kept

        if len(pages) <= 20:
            # Small wiki: file pages first, then everything else minus collisions
            other = _drop_colliding(list(pages.keys()))
            selected_titles = file_pages + other
        else:
            # Large wiki: file pages + vector/LLM-selected supplementary pages
            llm_selected, page_selection_usage = _select_relevant_pages(
                pages_for_llm, question, session_id,
                exclude_cached_answers=exclude_cached_answers,
            )
            supplementary = _drop_colliding(llm_selected)
            selected_titles = file_pages + supplementary
        logger.info("File-focused query: %d pages from mentioned file(s), %d total selected",
                     len(file_pages), len(selected_titles))
    else:
        # No file mentioned — original behaviour, now with optional family
        # pre-filter + broad-widen forwarded from the resolved scope (Phase 2).
        _candidates = pages_for_llm
        if family_docs:
            _fam_set = {d for d in family_docs if d}
            _in_family = {
                t: p for t, p in pages_for_llm.items()
                if isinstance(p, dict) and p.get("source_doc", "") in _fam_set
            }
            # doc_family pre-filters the VECTOR search only. BM25 ranks over whatever
            # pool it is handed, and the RRF fusion + per-document diversification
            # that follow actively spread results ACROSS documents — so a page from
            # outside the family can enter through the keyword channel and then be
            # promoted for variety. Measured on Q21: a Service-Agreement family scope
            # still returned NDA 7 pages this way, and the answer reported a "thirty
            # (30) days" convenience-notice period that appears nowhere in Service
            # Agreement 7 (whose own page says fifteen-month term, notice unspecified).
            # Narrowing the candidate pool makes the resolved family bind every
            # channel. Falls back to the full pool when the family matches no page
            # here, so a scope decision can never starve retrieval outright.
            if _in_family:
                logger.info("Family scope: candidate pool narrowed from %d to %d page(s)",
                            len(pages_for_llm), len(_in_family))
                _candidates = _in_family
        if len(_candidates) <= 20:
            selected_titles = list(_candidates.keys())
        else:
            selected_titles, page_selection_usage = _select_relevant_pages(
                _candidates, question, session_id,
                doc_family=doc_family, force_broad=force_broad,
                exclude_cached_answers=exclude_cached_answers,
            )

    # --- Step 2: Build context string from selected pages ---
    # Q: pages are cached prior answers — cap so they don't crowd out source content.
    # Regular pages: cap at MAX_PAGE_CONTEXT_CHARS to bound total prompt size.
    # Shared concept pages (no case prefix, multi-source) can grow large through merges;
    # capping ensures one bloated page doesn't consume half the context window.
    _QPAGE_CAP = config.MAX_QPAGE_CONTEXT_CHARS
    _PAGE_CAP  = config.MAX_PAGE_CONTEXT_CHARS

    wiki_parts = []
    # A question can assert that one document states another's provision ("the
    # governing law of the SOW ... AS STATED IN the Power of Attorney"). That is
    # a claim, and when it is false the answer has to say so. A prompt rule
    # alone did not hold: competing against thirty other rules, the model kept
    # noting the second document's absence and then quoting the FIRST
    # document's own clause as the answer — which reads to the user as though
    # the cross-reference checked out. Stating the finding as retrieved
    # evidence, at the top of the context, is what the model actually acts on.
    _xref = _failed_cross_reference(question, pages, selected_titles)
    if _xref:
        logger.info("Cross-reference check: %s", _xref)
        wiki_parts.append(f"[CROSS-REFERENCE CHECK — read before answering: {_xref}]\n")

    # Same mechanism, for the same reason: an amendment family puts two
    # documents in front of the model that answer the question with two
    # different figures, and the rule for choosing is one clause at the end of a
    # long question ("after giving effect to this amendment").
    _amend = _amendment_family_directive(question)
    if _amend:
        logger.info("Amendment-family directive: %s", _amend)
        wiki_parts.append(f"[AMENDMENT FAMILY — read before answering: {_amend}]\n")

    # When retrieval is file-focused, prepend a header so the LLM knows which
    # document the pages come from (handles "Services Agreement" vs "Service Agreement").
    if file_pages and mentioned_files:
        doc_names = [re.sub(r'\b(redacted|Redacted|_)\b', ' ', os.path.splitext(
            d.replace("\\", "/").rsplit("/", 1)[-1])[0]).strip()
            for d in mentioned_files]
        doc_names = [re.sub(r'\s+', ' ', d) for d in doc_names]
        wiki_parts.append(f"[The following pages are from: {', '.join(doc_names)}]\n")

    # Document-level metadata (execution/effective date, parties, governing law)
    # is stored in page_metadata, which retrieval never searches — ranking runs
    # over page CONTENT. A field extracted correctly at ingest but kept only as
    # metadata is therefore invisible to the answer, and the pipeline reports
    # "not covered" about a fact already in its own database. Measured: Court
    # Case Document 2 carries effective_date "06 July 2025" and Joint Venture
    # Agreement 3 "18 November 2025" — the exact ground truth of two questions
    # that scored 2/10 and were classified ingest-capped. Surfacing a compact
    # header for the documents actually selected costs a few indexed lookups and
    # a few hundred characters, and never displaces page content (it is added
    # before the budget loop and counted against the same cap).
    if config.USE_DATABASE and selected_titles:
        _meta_docs = []
        for _t in selected_titles:
            _p = pages.get(_t)
            _sd = _p.get("source_doc", "") if isinstance(_p, dict) else ""
            if _sd and _sd not in _meta_docs:
                _meta_docs.append(_sd)
            if len(_meta_docs) >= 6:
                break
        _meta_lines = []
        for _sd in _meta_docs:
            try:
                _md = _db.get_metadata(_active_wiki_id(), session_id, _sd)
            except Exception as _md_err:
                logger.warning("metadata lookup failed for %s: %s", _sd, _md_err)
                continue
            _labels = {"effective_date": "date of this document (execution / effective / filing date)",
                       "parties": "parties", "governing_law": "governing law",
                       "jurisdiction": "jurisdiction"}
            _bits = [f"{_labels[_k]}: {_md[_k]}"
                     for _k in ("effective_date", "parties", "governing_law", "jurisdiction")
                     if _md.get(_k)]
            if _bits:
                _meta_lines.append(f"- {_norm_doc_name(_sd)} — " + "; ".join(_bits))
        if _meta_lines:
            # Framing matters as much as inclusion. A first version headed these
            # "metadata … use only if the question asks for one of these fields"
            # and the answer LLM ignored it: asked for the date of a filing whose
            # date sat in the block, it still replied "not covered in the provided
            # documents". Two causes, both addressed here — the field name
            # "effective date" did not obviously answer "date of the application",
            # and an uncitable block loses to the prompt's rule that every fact
            # carry a citation. So the field is named for what it is, and the
            # block is declared citable against the document it describes.
            wiki_parts.append(
                "[Document facts recorded at ingest. These are drawn from the documents "
                "themselves and may be cited as such, naming the document. Where the "
                "question asks for one of these fields, answer from it rather than "
                "reporting the fact as unavailable:]\n" + "\n".join(_meta_lines) + "\n"
            )

        # Clauses and tables extracted at ingest live in their own typed
        # stores (clauses, tables) — real structured data, but retrieval only
        # ever searches page CONTENT, so a clause or table row that never made
        # it into a page's prose summary is invisible to the answer, the same
        # blind spot the metadata block above closes for document-level
        # facts. Measured: a Framework Supply Agreement's full 8-vendor
        # pricing table sits correctly in `tables`, but the wiki page's prose
        # only mentioned one vendor's row — the answer LLM confidently said
        # it couldn't determine the highest-value vendor. A Term Sheet's real
        # Survival clause sits correctly in `clauses`, but no page for that
        # document mentions "survival" at all.
        from sqlalchemy import text as _struct_sql
        # Which of the scoped documents the question is actually ABOUT, and which
        # clause of it. A multi-document scope (party-multi, comparison) is mostly
        # siblings pulled in for context, and the structured block is emitted in
        # page-selection order and truncated from the end — so the one document
        # the question names could be cut away entirely while its siblings'
        # boilerplate survived. Confirmed live on the 500-question evaluation:
        # "Section 4 (Relationship Of Parties) of the Term Sheet between Tata
        # AutoComp Systems Limited and Castellane EPC Pte. Ltd." resolved a
        # 5-document scope whose clause text totals ~32k characters against a 20k
        # cap; the Term Sheet sorted last, its clauses were truncated away, and
        # the answer reported the section as absent while the exact clause sat in
        # the `clauses` table. Ordering by overlap with the question's own words
        # costs nothing and puts the named document — and its named clause —
        # first, where no cap can reach them.
        _q_tokens = {w for w in re.findall(r'[a-z0-9]{3,}', (question or "").lower())
                     if w not in _NARROW_TOKEN_STOPWORDS}

        def _q_overlap(text: str) -> int:
            return len(_q_tokens & set(re.findall(r'[a-z0-9]{3,}', (text or "").lower())))

        _meta_docs = sorted(_meta_docs, key=lambda d: -_q_overlap(_norm_doc_name(d)))
        _struct_lines = []
        for _sd in _meta_docs:
            try:
                with _db.get_engine().connect() as _sconn:
                    _clause_rows = _sconn.execute(_struct_sql("""
                        SELECT clause_type, verbatim_text FROM clauses
                        WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
                        ORDER BY id
                    """), {"w": _active_wiki_id(), "s": session_id, "d": _sd}).fetchall()
                    _table_rows = _sconn.execute(_struct_sql("""
                        SELECT caption, columns, rows FROM tables
                        WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
                        ORDER BY id
                    """), {"w": _active_wiki_id(), "s": session_id, "d": _sd}).fetchall()
                    # A figure's stored description is the ONLY record of what a
                    # diagram shows — unlike a clause, no prose page paraphrases
                    # it, because a floor plan or an org chart has no text for
                    # the page-writing pass to summarise. Confirmed live: both
                    # "what does the floor plan diagram ... show" and "what does
                    # the org chart diagram ... show" answered "not covered"
                    # while the figure row sat in the typed store, correctly
                    # extracted at ingest, simply never read back.
                    _figure_rows = _sconn.execute(_struct_sql("""
                        SELECT page_num, figure_kind, description FROM figures
                        WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
                        ORDER BY page_num, id
                    """), {"w": _active_wiki_id(), "s": session_id, "d": _sd}).fetchall()
            except Exception as _struct_err:
                logger.warning("structured-data lookup failed for %s: %s", _sd, _struct_err)
                continue
            _doc_lines = []
            # Tables first: a full table is much harder to reconstruct from
            # prose than a clause (which the prose page for that topic
            # usually paraphrases reasonably well anyway), and a table's rows
            # only ever get MORE valuable to preserve intact as they get
            # bigger — so if the length cap below has to cut something, it
            # should cut clause text, not a table's later rows. Confirmed
            # live: with clauses first, an 8-vendor Table 1 was fully
            # preserved but a second table (Schedule 3, the one containing
            # the actual highest-value vendor) got truncated away entirely.
            for _cap_title, _cols, _rows in _table_rows:
                _doc_lines.append(f"  - Table: {_cap_title or '(untitled)'}")
                if _cols:
                    _doc_lines.append(f"    Columns: {', '.join(str(c) for c in _cols)}")
                for _r in (_rows or [])[:30]:
                    _parsed_row = _r
                    # A row is stored as a stringified list rather than a
                    # nested JSON array — decode it for a readable line
                    # rather than dumping the raw repr into the prompt.
                    if isinstance(_r, str):
                        try:
                            import ast as _ast
                            _parsed_row = _ast.literal_eval(_r)
                        except Exception:
                            _parsed_row = _r
                    _doc_lines.append(f"    Row: {_parsed_row}")
            # Figures before clauses for the same reason tables come first: if
            # the cap has to cut something, it should cut clause text, which the
            # prose pages largely restate anyway, not the one description of a
            # diagram that exists nowhere else.
            for _pnum, _fkind, _fdesc in _figure_rows:
                if not _fdesc:
                    continue
                _where = f"page {_pnum}" if _pnum else "page unknown"
                # Longer than a clause's 500 because a described diagram has no
                # prose page to fall back on: its node labels, room names and
                # legend entries exist ONLY here, and cutting them off mid-list
                # leaves the answer reporting the diagram as unreadable.
                _doc_lines.append(
                    f"  - Figure ({_fkind or 'figure'}, {_where}): {_fdesc.strip()[:2000]}")
            # Same reasoning as the document ordering above, one level down: the
            # clause the question NAMES goes first, so a per-document cut can
            # never be what removes it. A question that names no clause leaves
            # every overlap at zero and the original ingest order intact.
            #
            # Ranked on the clause TEXT as well as its label, because not every
            # clause has a label that describes it. A board resolution's rows are
            # headed by their own opening words ("Resolved That — Nothing in this
            # Agreement shall be construed"), since what a board resolves is
            # ordinary clause substance that naming would have to guess at — so
            # for those the words a question shares are in the body, not the head.
            for _ct, _vt in sorted(
                    _clause_rows,
                    key=lambda r: -(_q_overlap(r[0]) * 2 + _q_overlap((r[1] or "")[:400]))):
                if _vt:
                    _doc_lines.append(f'  - {_ct}: "{_vt.strip()[:500]}"')
            if _doc_lines:
                _struct_lines.append((_sd, f"{_norm_doc_name(_sd)}:\n" + "\n".join(_doc_lines)))
        if _struct_lines:
            # Budget the cap ACROSS documents rather than truncating one joined
            # string from the end. A single joined block spends itself on
            # whichever documents happen to sort first and leaves later ones with
            # nothing — the failure described above. Each document takes an even
            # share of what is left when its turn comes, and a document that
            # needs less than its share hands the remainder to the next one, so
            # the common case (one big document, several small siblings) still
            # gets the big one through intact.
            _struct_cap_chars = 20000
            _remaining = _struct_cap_chars
            _kept_blocks = []
            for _i, (_sd, _block) in enumerate(_struct_lines):
                _docs_left = len(_struct_lines) - _i
                _share = max(_remaining // _docs_left, 1200)
                if len(_block) > _share:
                    _block = _block[:_share] + "\n  [...this document's remaining clauses truncated]"
                    logger.info("Structured block for %s truncated to its %d-char share "
                                "(%d document(s) in scope)",
                                _norm_doc_name(_sd), _share, len(_struct_lines))
                _kept_blocks.append(_block)
                _remaining = max(_remaining - len(_block), 0)
                if _remaining <= 0 and _i + 1 < len(_struct_lines):
                    logger.info("Structured-extraction budget exhausted after %d of %d document(s)",
                                _i + 1, len(_struct_lines))
                    break
            _struct_block = "\n\n".join(_kept_blocks)
            wiki_parts.append(
                "[Structured extraction recorded at ingest — full clause text, complete "
                "table data, and descriptions of figures/diagrams, independent of the "
                "prose page summaries below. These are drawn from the documents "
                "themselves and may be cited as such, naming the document. A table's "
                "full rows live here even when a page's prose summary only mentions a "
                "sample of them. A Figure line is a DESCRIPTION of a diagram, not text "
                "printed in the document — report what it says the diagram shows, but "
                "never present it as a verbatim quote:]\n" + _struct_block + "\n"
            )

    _TOTAL_CAP = config.MAX_TOTAL_CONTEXT_CHARS
    total_chars = sum(len(p) for p in wiki_parts)
    pages_omitted = 0
    _trace_pages = []

    for title in selected_titles:
        if title in pages:
            if total_chars >= _TOTAL_CAP:
                pages_omitted += 1
                continue

            page = pages[title]
            content = page.get("content", "") if isinstance(page, dict) else page
            _orig_chars = len(content)

            if isinstance(page, dict) and page.get("contradiction_flagged"):
                content = "[WARNING: This page contains conflicting claims. Surface the conflict explicitly in your answer. Do not resolve it.]\n" + content

            if title.startswith("Q:") and len(content) > _QPAGE_CAP:
                content = content[:_QPAGE_CAP] + "\n[...truncated — cached answer summary only]"
            elif len(content) > _PAGE_CAP:
                content = _truncate_page_content(content, _PAGE_CAP)

            # Clean the title for LLM context: strip UUID prefix and path noise
            # "Topic (uuid_Legal AI Tool - Group_Type_Name_redacted.pdf)"
            #   → "Topic – Service Agreement 4"
            display_title = title
            source_label = title
            paren = title.rfind("(")
            if paren > 0:
                topic = title[:paren].strip()
                raw_path = title[paren + 1:title.rfind(")")].strip() if ")" in title else ""
                clean_path = re.sub(r'^[a-f0-9-]{36}_', '', raw_path)
                clean_file = clean_path.replace("\\", "/").rsplit("/", 1)[-1]
                clean_file = os.path.splitext(clean_file)[0]
                clean_file = clean_file.replace("_", " ").strip()
                clean_file = re.sub(r'\b(redacted|Redacted)\b', '', clean_file).strip()
                # Extract just the doc name: last meaningful segment
                # "Legal AI Tool - Tata Group Service Agreement Service Agreement 4"
                #   → "Service Agreement 4"
                for prefix in ["Legal AI Tool - Tata Group ", "Legal AI Tool - "]:
                    if clean_file.startswith(prefix):
                        clean_file = clean_file[len(prefix):]
                # Remove repeated type prefix: "Service Agreement Service Agreement 4" → "Service Agreement 4"
                parts = clean_file.split()
                mid = len(parts) // 2
                if mid >= 2 and parts[:mid] == parts[mid:2*mid]:
                    clean_file = " ".join(parts[mid:])
                display_title = f"{topic} – {clean_file}" if clean_file else topic
                source_label = clean_file or topic
            # Prefer the real source_doc FILENAME for the [From:] label. A page's
            # display title carries its instrument-type label (e.g. "Source Code
            # Escrow Agreement"), which can differ from the file's own identifier
            # (e.g. Test_JVA_05.txt whose content is actually an escrow agreement).
            # The citation-attribution check compares a cited filename identifier
            # (e.g. "JVA5") against this label — matching it against the type-label
            # instead of the filename produced false "misattribution" warnings for
            # quotes that were correctly attributed to their own file. Emitting the
            # filename here gives the checker the right token to match against.
            src_doc = page.get("source_doc", "") if isinstance(page, dict) else ""
            src_file = ""
            if src_doc:
                src_file = re.sub(r'^[a-f0-9-]{36}_', '', src_doc.replace("\\", "/").rsplit("/", 1)[-1])
                src_file = os.path.splitext(src_file)[0]
            from_label = src_file or source_label
            # Real "---" separator + "[From: ...]" label per source block so the
            # answer prompts' SCOPE / CROSS-DOCUMENT / DISTINCT-SOURCE rules bind
            # to markers that actually exist in the context (previously they
            # referenced separators/labels that were never emitted). The label sits
            # under the "##" header so it falls inside this block for _PAGE_BLOCK_RE.
            part = f"\n---\n## {display_title}\n[From: {from_label}]\n{content}\n"
            wiki_parts.append(part)
            total_chars += len(part)
            _trace_pages.append({
                "title": display_title, "source_doc": from_label,
                "chars_included": len(content), "chars_original": _orig_chars,
                "truncated": len(content) < _orig_chars,
            })

    _trace = tracing.get_trace()
    if _trace:
        _trace.log_pages(_trace_pages, pages_omitted, total_chars, _TOTAL_CAP)

    if pages_omitted:
        wiki_parts.append(
            f"\n[NOTE: {pages_omitted} additional matching page(s) were omitted from this "
            f"context to stay within the model's token limit. The answer below is based only "
            f"on the pages shown above.]"
        )
        logger.warning(
            "generate_answer: omitted %d/%d selected pages — total context exceeded %d chars",
            pages_omitted, len(selected_titles), _TOTAL_CAP,
        )
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


_ASSESSMENT_PATTERNS = re.compile(
    r'(?:go\s*/\s*no[- ]?go|recommend|recommendation|should\s+(?:we|i|tata)\s+sign|'
    r'risk\s+assessment|risk\s+review|advise|advisory|red\s+flag|deal[- ]?breaker|'
    r'approve|approval|sign\s+off|signoff|would\s+you\s+(?:recommend|advise|sign)|'
    r'safe\s+to\s+sign|ready\s+to\s+(?:sign|execute)|negotiation\s+strategy|'
    r'accept\s+or\s+reject|proceed\s+or\s+not)',
    re.IGNORECASE,
)


def _is_assessment_query(question: str) -> bool:
    """Return True if the question asks for a legal recommendation or risk assessment."""
    return bool(_ASSESSMENT_PATTERNS.search(question))


def _build_metadata_block(session_id: str, selected_titles: list, pages: dict) -> str:
    """Build a metadata context block with party names and document info.

    Fetches from the page_metadata table (C7) when available, falls back to
    extracting party info from page titles and content.
    """
    metadata_lines = []

    # Collect source docs from selected pages
    source_docs = set()
    for title in selected_titles:
        if title in pages:
            page = pages[title]
            if isinstance(page, dict):
                sd = page.get("source_doc", "")
                if sd:
                    source_docs.add(sd)

    if not source_docs:
        return ""

    # Try DB metadata first
    if config.USE_DATABASE:
        for doc in source_docs:
            try:
                meta = _db.get_metadata(_active_wiki_id(), session_id, doc)
                if meta:
                    clean_name = re.sub(r'^[a-f0-9-]{36}_', '', doc)
                    parts = [f"Document: {clean_name}"]
                    if meta.get("parties"):
                        parts.append(f"Parties: {meta['parties']}")
                    if meta.get("governing_law"):
                        parts.append(f"Governing Law: {meta['governing_law']}")
                    if meta.get("effective_date"):
                        parts.append(f"Effective Date: {meta['effective_date']}")
                    if meta.get("jurisdiction"):
                        parts.append(f"Jurisdiction: {meta['jurisdiction']}")
                    if meta.get("liability_cap"):
                        parts.append(f"Liability Cap: {meta['liability_cap']}")
                    if meta.get("termination_notice"):
                        parts.append(f"Termination Notice: {meta['termination_notice']}")
                    if meta.get("payment_terms"):
                        parts.append(f"Payment Terms: {meta['payment_terms']}")
                    if meta.get("matter_reference"):
                        parts.append(f"Matter Reference: {meta['matter_reference']}")
                    if len(parts) > 1:
                        metadata_lines.append(" | ".join(parts))
            except Exception:
                pass

    # Fallback: extract party info from page content if no DB metadata
    if not metadata_lines:
        for title in selected_titles:
            if title in pages and any(kw in title.lower() for kw in ["parties", "overview", "recital"]):
                page = pages[title]
                content = page.get("content", "") if isinstance(page, dict) else str(page)
                if content:
                    metadata_lines.append(f"From '{title}': {content[:300]}")
                    break

    if not metadata_lines:
        return ""

    return "\nDOCUMENT METADATA:\n" + "\n".join(metadata_lines) + "\n"


# Matches quoted spans of at least 15 chars, straight or curly quote marks.
# 15 chars filters out trivial quoted words/labels that aren't meant as verbatim
# source citations.
#
# Content uses a GREEDY "." (not an exclusion class like [^"”]) so a quote is
# matched to the LAST quote character in the window, not the very next one.
# Legal citations legitimately contain nested quotes (e.g. `by and among: 1.
# AETHER TECHNOLOGIES INC. ("Depositor"); 2. ...`) — with an exclusion class,
# the regex naively pairs up quote characters two at a time: a short nested
# quote below the 15-char floor gets skipped, but its closing mark then gets
# reused as the OPENING delimiter for the next scan, producing garbled
# fragments like `") is dated ... ("` that fail verbatim verification even
# though the real, full quote is genuine. Greedy matching consumes through
# nested quotes and backtracks from the end, capturing the true outer span.
# "." doesn't cross newlines, so this can't merge two unrelated quotes in
# different paragraphs — but a markdown table renders each row on its own
# line with multiple short quotes across cells (e.g. a risk-assessment
# table's "Verbatim Text" column), and greedy "." WILL span across pipe-
# delimited cells on the SAME line, merging one row's quote with another
# row's quote plus the non-verbatim table markup between them — which then
# fails even the inner-segment fallback, since row labels/pipes aren't
# source text. "|" is excluded from content for exactly this reason: a
# real legal quote essentially never contains a literal pipe character, so
# this bounds greedy matching to one table cell without narrowing anything
# that matters. Quote characters themselves are excluded from content for the
# same reason "|" is: a line like `"Title" – Supporting Quote: "actual text"`
# would otherwise match as ONE span running from the first quote's open
# through the second quote's close — merging a citation label with its own
# quoted text and producing a span that matches neither. Excluding quote
# chars forces each `"..."` span to stop at its own closing quote instead.
# Pairs quote characters at ANY length; callers decide what is long enough to
# be worth verifying, via _is_checkable_quote.
#
# The length floor used to live in the pattern, and that produced a false
# CITATION WARNING on the model's own prose. In `The question "elaborate on 2"
# reasonably refers to item 2 in the Closing Conditions Table...`, the span
# "elaborate on 2" is fourteen characters - below the old floor - so the match
# failed there and restarted from its CLOSING quote, which then paired with the
# next opening quote and captured the narrative between them as a quotation.
# The warning that followed told the reader that a sentence the model wrote
# about itself could not be found in the source: true, useless, and the fastest
# way to teach someone to ignore the warnings that do matter.
_QUOTE_SPAN_RE = re.compile(r'["“]([^"“”|\n]{0,500})["”]')

_QUOTE_MIN_CHARS = 15


def _is_checkable_quote(text: str) -> bool:
    """Long enough to be a quotation rather than a quoted word or label."""
    return len(text or "") >= _QUOTE_MIN_CHARS

# Matches the start of a reference-list line: an optional bullet/dash marker
# followed by a "[N]"-style citation number, e.g. "- [1] FileName, ..." or
# "[2] FileName ...". Used to scope the bare-quote stripping in
# _drop_unverifiable_reference_quotes to actual reference lines, so a quote
# appearing in ordinary answer prose is never mistaken for a citation excerpt.
_RX_REFERENCE_LINE_START = re.compile(r'^\s*(?:[-*]\s*)?\[\d+\]')

# Splits a quote on ellipsis markers ("..." or "…"). Legal citations legitimately
# splice together non-adjacent sentences this way (e.g. "BETWEEN: ... Each of
# the aforesaid shall be referred to as a 'Party'"), so requiring the WHOLE
# spliced string to be one continuous substring of the source produces false
# positives on genuinely verbatim quotes — each segment must independently be
# verbatim instead.
_ELLIPSIS_SPLIT_RE = re.compile(r'\s*(?:\.\.\.|…)\s*')


def _quote_segments(q: str) -> list[str]:
    return [s.strip() for s in _ELLIPSIS_SPLIT_RE.split(q) if len(s.strip()) >= 8]


# Splits a quote on internal straight-quote characters — the counterpart to
# _quote_segments' ellipsis handling. The greedy _QUOTE_SPAN_RE above fixes
# nested-quote citations (one long quote containing "sub-quotes") by matching
# the full outer span, but that same greediness can merge two SEPARATE short
# genuine quotes that happen to sit on the same line (e.g. `"X" from "Y"`)
# into one span that fails whole-string verification. Splitting on internal
# quote marks and checking each piece independently — same as the ellipsis
# fallback — recovers both cases without reopening the fabrication-detection
# hole the greedy regex was built to close in the first place.
_INTERNAL_QUOTE_SPLIT_RE = re.compile(r'["“”]')

# Common short words that appear as narration BETWEEN two separately-quoted
# phrases in the same sentence (e.g. `**"quote one"** and that **"quote two"**`).
# When two adjacent quotes are each individually bolded, the greedy quote-span
# match merges them into one span with this connector prose caught in the
# middle; _quote_inner_segments correctly splits it back into 3 pieces, but the
# middle piece is markdown formatting + filler words, never meant to be a
# verbatim quote itself — requiring it to independently match context (like
# the two real quotes either side of it) dragged genuinely correct answers
# down. A segment made ENTIRELY of these words (after stripping markdown
# emphasis markers) is connector prose, not a quote fragment, and shouldn't be
# required to verify. Confirmed live: "** and that **" between two genuine,
# independently-verifiable quotes was the sole reason a fully-grounded answer
# got flagged.
_CONNECTOR_ONLY_WORDS = {
    "and", "that", "but", "or", "which", "who", "the", "a", "an", "is", "are",
    "was", "were", "states", "stating", "further", "also", "it", "its", "this",
    "these", "those", "as", "to", "of", "in",
}


def _is_connector_only_segment(s: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", s.lower())
    if not words:
        return True  # pure punctuation/markdown noise, not a quote
    return all(w in _CONNECTOR_ONLY_WORDS for w in words)


def _quote_inner_segments(q: str) -> list[str]:
    segments = []
    for s in _INTERNAL_QUOTE_SPLIT_RE.split(q):
        s = s.strip()
        if len(s) < 8:
            continue
        stripped_md = re.sub(r'[*_]+', '', s).strip()
        if _is_connector_only_segment(stripped_md):
            continue
        segments.append(stripped_md or s)
    return segments


# Follow-up to the connector-word filter above: found on retest of the same
# CCD6 question, two genuine quotes are sometimes separated by real narrative
# words ("This distinction underpins the request for an...") rather than pure
# filler ("and that") — _is_connector_only_segment correctly refuses to drop a
# segment with real content, so that segment still has to independently
# "verify" against context even though it was never a quote itself, just the
# model's own connecting prose between two already-confirmed genuine quotes.
# A word-list can't tell "model narration between two quotes" apart from
# "genuine document text that itself contains internal punctuation" (the
# nested-quote case, e.g. `("Depositor")`, where EVERY split piece — including
# the short ones — really is part of one continuous verbatim excerpt and must
# still independently verify). The distinguishing signal isn't the segment's
# own words, it's its position: if a segment fails to verify but sits directly
# between two segments that DO independently verify, it's excused as narration
# bridging two confirmed real quotes rather than a free-standing unverifiable
# claim. A segment with no verified neighbor on both sides (leading/trailing,
# or next to another unverified segment) still must verify on its own — this
# only excuses the specific "quote, narration, quote" shape.
def _segments_effectively_verified(q: str, ctx_norm: str, question_norm: str) -> bool:
    def _seg_norm_ok(seg: str) -> bool:
        sn = _norm_for_match(seg)
        return sn in ctx_norm or (bool(question_norm) and sn in question_norm)

    parts = []
    for s in _INTERNAL_QUOTE_SPLIT_RE.split(q):
        stripped_md = re.sub(r'[*_]+', '', s.strip()).strip()
        if len(stripped_md) < 8:
            continue
        parts.append((stripped_md, _seg_norm_ok(stripped_md)))

    if not parts:
        return True

    for i, (seg, verified) in enumerate(parts):
        if verified:
            continue
        if _is_connector_only_segment(seg):
            continue
        prev_ok = parts[i - 1][1] if i > 0 else False
        next_ok = parts[i + 1][1] if i < len(parts) - 1 else False
        if prev_ok and next_ok:
            continue  # real narrative prose bridging two confirmed genuine quotes
        return False
    return True


def _known_page_titles(context: str) -> set[str]:
    """Normalized page titles present in the context (from '## Title' headers).

    Used to exclude citation labels like `[1] "Commercial Courts Act, 2015 –
    Statute" – the Act establishes...` from quote verification — the model is
    quoting the PAGE TITLE as a citation label (a normal, correct citation
    format), not claiming that string is a verbatim excerpt from the source
    text. Without this, the quote-verification regex treats any quoted span as
    a content-verbatim claim and flags the title itself as unverifiable.

    Keyed on alphanumerics only (see _alnum_only) so a model-reformatted label
    like "Purpose and Permitted Use – NDA-Greensteel – NDA" still matches the
    real title "Purpose and Permitted Use – NDA-Greensteel (NDA)" — exact
    punctuation match was rejecting these and flagging the title itself as a
    fabricated quote.
    """
    return {
        _alnum_only(_norm_for_match(title))
        for title, _ in _PAGE_BLOCK_RE.findall(context)
    }


# Ingest appends "\n\n**Supporting Quotes:**\n> quote 1\n> quote 2" to a page's
# content (see _atomic_merge_db etc.) — those quotes already passed
# _filter_verified_quotes at ingest time, i.e. they're confirmed verbatim
# against the ORIGINAL source document, not just this session's context. The
# rest of a page's body is LLM-synthesized descriptive prose (a paraphrase),
# not a verbatim source excerpt, even when it happens to read naturally.
_SUPPORTING_QUOTES_RE = re.compile(r'\*\*Supporting Quotes:\*\*\s*\n((?:^>.*(?:\n|$))+)', re.MULTILINE)


def _block_verification_text(title: str, body: str) -> str:
    """Text used to verify quotes against a single page block — restricted to
    its Supporting Quotes section when present (the ingest-time-verified
    portion).

    Falls back to the full body ONLY for cached 'Q:' answer pages (which
    never go through the quotes pipeline at all — no heading is expected).
    For a regular document-derived page with NO Supporting Quotes heading,
    that means ingest-time verification (_filter_verified_quotes) proposed
    quotes for this page and found NONE of them verbatim-verifiable — falling
    back to its descriptive prose would let the answer LLM lift a sentence
    from that unverified prose, wrap it in quotation marks, and have it pass
    as if it were a genuine excerpt (confirmed live: NDA_08's "Confidential
    Information" page has no Supporting Quotes section, and the answer LLM
    quoted its prose sentence verbatim as if it were sourced from the
    original document). Such a page has no verifiable quotable content, so
    it contributes nothing here — any quote attributed to it correctly fails.
    """
    if title.startswith("Q:"):
        # A cached prior answer is NOT a source document. Returning its body
        # here made the ENTIRE cached Q&A — including the user's own question
        # text — verifiable "quotable" material, so the answer LLM could quote a
        # previous question back and have it pass as a contract excerpt
        # (confirmed live: an SA 1 answer cited "…ummarize this document in 10
        # bullet points for a G…" as a source quote).
        #
        # Same reasoning as the no-quotes page below: a page carrying no
        # verifiable ORIGINAL content contributes nothing to the verification
        # corpus, so any quote attributed to it correctly fails. A quote the
        # cached answer merely REPEATS from a real document still verifies
        # against that document's own page when it is in context — which is
        # exactly the condition under which it is safe to cite.
        return ""
    sq_matches = _SUPPORTING_QUOTES_RE.findall(body)
    return '\n'.join(sq_matches) if sq_matches else ""


# The structured-extraction block get_context prepends (clauses and tables from
# their typed stores). Its contents carry the SAME provenance guarantee as a
# Supporting Quotes block — a clause row's text is stored verbatim, either
# because the ingest prompt requires an exact quote or because
# backfill_sections.py lifted it out of the PDF by regex — so it belongs in the
# strict corpus. Left out, every answer sourced from a clause row picked up a
# spurious "read as paraphrase rather than exact wording" note over text that is
# verbatim source. Confirmed live on the Section 12 (Relationship Of Parties)
# answer, whose quote is character-for-character the document's own.
_STRUCTURED_BLOCK_RE = re.compile(
    r'\[Structured extraction recorded at ingest[^\]]*\]\n(.*?)(?=\n## |\Z)', re.S)


def _strict_verification_corpus(context: str) -> str:
    """Build the citation-verification text corpus, restricted to Supporting
    Quotes blocks per page (see _block_verification_text) plus the verbatim
    structured-extraction block. Falls back to the raw context unchanged if no
    '## Title' page blocks are found at all.
    """
    blocks = _PAGE_BLOCK_RE.findall(context)
    if not blocks:
        return context
    parts = [_block_verification_text(title, body) for title, body in blocks]
    for struct in _STRUCTURED_BLOCK_RE.findall(context):
        # Figure lines are excluded on purpose: a figure's description is
        # GENERATED (a vision pass describing a diagram), not text printed in
        # the document, so letting it verify quotes would do exactly what
        # _block_verification_text refuses to do for synthesized page prose.
        parts.append('\n'.join(
            ln for ln in struct.splitlines() if not ln.lstrip().startswith("- Figure (")))
    return '\n'.join(parts)


# A reference/citation line ending in a placeholder standing in for a real
# excerpt the model didn't have — e.g. `... Liability and Indemnity — "Not
# provided in excerpt"`, `... Section 2.3. Quote: (not provided here)`, a
# BARE `| Quote: not provided in excerpt` with no wrapping quotes/parens at all,
# or a double-wrapped `— "(none)"` (parens nested inside the outer quote marks)
# (confirmed live: nano varies the format across every batch — wrapped, bare,
# double-wrapped). The ANSWER/ASSESSMENT/COMPARISON prompts already ban this,
# but gpt-5-nano at low reasoning effort re-emits it intermittently. The set of
# stand-in phrases is finite and none is ever a legitimate quote, so strip it
# deterministically rather than depend on model compliance. Two branches: after
# an explicit "Quote:" label the wrapper is OPTIONAL (the label alone
# disambiguates intent from ordinary prose); without that label the wrapper is
# REQUIRED (otherwise a genuine unwrapped sentence like "the clause is not
# available" would be eaten). An optional inner "(...)" is tolerated either way.
_PLACEHOLDER_QUOTE_RE = re.compile(
    r'(?im)'
    r'[ \t]*(?:[|—–-][ \t]*)?'                 # optional leading separator (pipe / dash)
    r'(?:'
    r'Quote[ \t]*:[ \t]*[("“\']?'               # "Quote:" label — wrapper optional after it
    r'|[("“\']'                                   # no label — wrapper REQUIRED
    r')'
    r'[ \t]*\(?[ \t]*'                          # optional inner paren ("(none)" inside outer quotes)
    r'(?:not[ \t]+provided(?:[ \t]+in[ \t]+excerpt|[ \t]+here)?'
    r'|not[ \t]+available|not[ \t]+applicable|n/?a|none(?:[ \t]+provided)?|nil)'
    r'[ \t]*\)?'
    r'[ \t]*\.?'
    r'[)"”\']?'                                   # closing wrapper (optional to match either branch)
    r'[ \t]*$'
)


# Pseudo-XML the model invents to organise its own output. Same family as the
# <reasoning> and <final> handling inside _run_generation_pass, but for tags no
# prompt ever mentions, so there is no fixed vocabulary to match. Confirmed live:
# an answer rendered to the user beginning with a literal "</reasoning>", another
# with a stray "</confidence>", and one wrapping its whole body in
# "<Service Agreement 2 termination rights>…</Service Agreement 2 termination rights>".
#
# Content is always PRESERVED — every rule here deletes tag characters only, never
# the text between them. That is what makes it safe to run on any answer: the
# worst case is a tag left behind, never an amputated answer.

# 1. Matched invented pair — <Foo Bar>…</Foo Bar> becomes the inner text. The
#    backreference means only a genuine open/close pair is unwrapped.
_PAIRED_PSEUDO_TAG_RE = re.compile(
    r'<\s*([A-Za-z][^<>\n]{0,80}?)\s*>(.*?)<\s*/\s*\1\s*>',
    re.DOTALL | re.IGNORECASE,
)

# 2. Control words the model wraps around its own scaffolding, orphaned or not.
#    Deliberately excludes <br>, which is legitimate and appears 384 times in
#    logged markdown tables.
_STRAY_CONTROL_TAG_RE = re.compile(
    r'<\s*/?\s*(?:reasoning|confidence(?:_score|_reason)?|thinking|thought|'
    r'scratchpad|plan|draft|analysis|answer|output|response|'
    r'final(?:_answer|_output|_table)?|references?|sources?|citations?)\s*>',
    re.IGNORECASE,
)

# 3. Any leftover tag whose NAME contains a space. No real HTML or markdown tag
#    does; "<Service Agreement 2 termination rights>" is always a model invention.
#    Catches the orphaned half of an invented pair that rule 1 could not match.
_SPACED_PSEUDO_TAG_RE = re.compile(r'<\s*/?\s*[A-Za-z][^<>\n]*?\s+[^<>\n]*?>')


def _strip_pseudo_tags(answer: str) -> str:
    """Remove invented pseudo-XML tags, always keeping the text inside them."""
    if not answer or "<" not in answer:
        return answer
    # Unwrap matched pairs repeatedly so nested inventions collapse fully.
    for _ in range(3):
        unwrapped = _PAIRED_PSEUDO_TAG_RE.sub(lambda m: m.group(2), answer)
        if unwrapped == answer:
            break
        answer = unwrapped
    answer = _STRAY_CONTROL_TAG_RE.sub('', answer)
    answer = _SPACED_PSEUDO_TAG_RE.sub('', answer)
    return answer.strip()


# A refusal renders through the same badge row as a real answer, so
# "Not addressed in the provided documents" arrived on screen wearing
# "Confidence: 92% · Grounding: 100%". Both numbers are literally correct — the
# model is confident IN ITS REFUSAL, and the refusal accurately describes what
# the context holds — but a reader skimming sees a green high-confidence badge
# attached to a non-answer and reads it as "here is an answer, and I'm sure of
# it". Detected here rather than in the frontend so the judgement lives with the
# text it describes and survives a reload.
#
# Anchored to the OPENING of the answer. A substantive answer routinely says
# "the context does not address X" about one sub-point midway through; only an
# answer that LEADS with it is a refusal.
_RX_NOT_COVERED_OPENER = re.compile(
    r'^\s*(?:[#*_>\-\s]*)?(?:'
    r'not\s+(?:addressed|covered|found|available|specified|mentioned|provided)'
    r'|no\s+(?:relevant\s+)?(?:information|provision|clause|content|text|mention)'
    r'|(?:this|the)\s+(?:retrieved\s+)?context\s+(?:does\s+not|doesn\'t)\s+'
    r'(?:contain|address|cover|include|provide)'
    r'|the\s+provided\s+documents?\s+(?:do|does)\s+not\s+'
    r'(?:contain|address|cover|include)'
    r'|(?:i\s+)?(?:could|can)\s+not\s+find'
    r')',
    re.IGNORECASE,
)

# How far in to look. The opener can sit behind a short heading the model
# emitted first ("Final answer:", "PART I —"), but not behind real content.
_NOT_COVERED_SCAN_CHARS = 220


def _is_not_covered_answer(answer: str) -> bool:
    """True when the answer LEADS with 'the documents don't cover this'.

    Drives a render flag only — it never changes the answer, the confidence
    score, or what was retrieved. A false positive costs one suppressed badge.
    """
    if not answer:
        return False
    head = answer.lstrip()[:_NOT_COVERED_SCAN_CHARS]
    if _RX_NOT_COVERED_OPENER.match(head):
        return True
    # Allow one short lead-in line ("Final answer:", "Answer:") before the
    # refusal — the reasoning-model outputs frequently open with one.
    first_break = head.find("\n")
    if 0 < first_break <= 60:
        return bool(_RX_NOT_COVERED_OPENER.match(head[first_break:].lstrip()))
    return False


def _strip_placeholder_quotes(answer: str) -> str:
    """Remove quote-wrapped placeholder stand-ins (e.g. "Not provided in excerpt")
    from the ends of reference/citation lines. Leaves the rest of the line — the
    real FileName + Clause/Section citation — intact, so a reference with no
    verbatim quote simply ends after its clause reference, as the prompts require.
    """
    if not answer or '"' not in answer and "'" not in answer and '(' not in answer:
        return answer
    return _PLACEHOLDER_QUOTE_RE.sub('', answer)


# An identifier-style LABEL, checked WITHOUT requiring immediate proximity to
# a code — real phrasing routinely restates the whole question between label
# and value ("The matter reference number for the NDA between Tata Steel
# Limited and NordForge Metallurgy GmbH is TSL/GREENSTEEL/2025/219"), which a
# proximity-anchored regex cannot bridge. Presence of this label ANYWHERE in
# the answer is the trigger; _CODE_SHAPED_TOKEN_RE below then independently
# finds candidate values.
_IDENTIFIER_LABEL_RE = re.compile(
    r'\b(?:matter\s*ref(?:erence)?(?:\s*(?:no\.?|number))?|docket\s*(?:no\.?|number)|'
    r'case\s*(?:no\.?|number)|filing\s*(?:id|reference|number)|'
    r'reference\s*(?:no\.?|number))\b',
    re.IGNORECASE,
)

# A code-shaped token: 3+ segments joined by "/" or "-", each alphanumeric.
# Requires at least one letter somewhere (so it can't match a bare number like
# a page range "10-20") and at least one segment of 2+ chars (so it can't
# match a bare clause locator like "8-1" or "5-3"). Real observed shapes:
# "TSPL/LEGALOPS/2025/058", "CS/331/2025", "2025-CV-0041".
_CODE_SHAPED_TOKEN_RE = re.compile(
    r'\b[A-Za-z0-9]{2,15}(?:[/\-][A-Za-z0-9]{2,15}){2,5}\b'
)


# A real formatted identifier is typed, not written: its alphabetic segments are
# upper-case ("TSPL/LEGALOPS/2025/058", "MAT-2018-3636", "2025-CV-0041"), and it
# carries at least one letter. _CODE_SHAPED_TOKEN_RE alone cannot tell one from
# an ordinary hyphenated English phrase, which has exactly the same shape - and
# the docstring above claims a letter is required where the pattern never
# enforced it, so a bare "30/45/60" qualified too.
#
# Confirmed live and visible to the reader: an answer about a Consultancy
# Agreement had "Take-or-pay" replaced with "[not stated in this document]",
# leaving the sentence "[not stated in this document] obligation locked with
# only limited carve-outs" - while the very document it was answering from has
# a page titled "Take-or-Pay Obligation". Alongside it went "30/45/60",
# "payment/gross-up" and "data-availability/security". The label gate does not
# help here: any long answer that mentions a Matter Reference at all opens the
# whole text to this check, and legal prose is full of hyphenated terms.
#
# The trade is deliberate: a fabricated code written in lower case is now
# missed. That costs a warning the grounding and citation checks still make
# their own way, where the false positive silently rewrites an answer's words.
def _looks_like_formatted_identifier(code: str) -> bool:
    """True when a code-shaped token is typed like a reference, not written."""
    if not any(ch.isalpha() for ch in code):
        return False          # "30/45/60", "10-20" - a range, not a reference
    for seg in re.split(r'[/\-]', code):
        if seg.isdigit():
            continue          # a year or a serial
        if any(ch.islower() for ch in seg):
            return False      # "or", "pay", "security" - English, not a code
    return True


def _verify_identifier_claims(answer: str, context: str) -> list[str]:
    """Deterministically catch a fabricated matter-reference/docket/case-number
    value — a different failure shape from a fabricated QUOTE, and confirmed to
    slip past _verify_answer_citations: the model states a plausible-looking
    code as fact (often with no surrounding quotation marks at all, e.g. "The
    matter reference is TCPL/PORTFOLIO/2025/211"), so the quote-span check
    never sees it as a quote to verify in the first place. A prompt-level rule
    telling the model not to do this was added first and did NOT stop it on
    live retest (same two codes reproduced deterministically on fresh, non-
    cached generations) — this is the same "prompt guidance alone has a
    residual failure rate, back it with a deterministic check" pattern already
    used throughout this file for quotes/attribution/party-name fabrication.

    Only scans the answer at all when an identifier LABEL is present somewhere
    in it — a bare code-shaped token with no such label nearby is far more
    likely to be a real clause/case citation the answer is legitimately
    quoting (e.g. "CS(COMM) 331/2025" in a court-case answer) than a matter
    reference, so this stays narrow to the one confirmed fabrication shape
    rather than becoming a general code-shaped-token checker.

    Case-sensitive exact-substring match against the raw context (these are
    formatted codes, not prose — case matters and normalisation would risk
    false-clearing a wrong code that only differs by case). Returns the list
    of CODE values that do not appear anywhere in the retrieved context; a
    genuine code that IS in context (confirmed live: SA4/Redwood's real
    "TSPL/LEGALOPS/2025/058") is correctly left alone.
    """
    if not answer or not context or not _IDENTIFIER_LABEL_RE.search(answer):
        return []
    # Collapse incidental whitespace around "/" and "-" separators for the
    # comparison only (never for anything that would change verification
    # elsewhere) — a genuine code in context formatted with stray spacing
    # ("TSPL/ LEGALOPS /2025/ 058") must not be treated as fabricated just
    # because the model reproduced it without the spacing.
    context_tight = re.sub(r'\s*([/\-])\s*', r'\1', context)
    unverified = []
    for m in _CODE_SHAPED_TOKEN_RE.finditer(answer):
        code = m.group(0)
        if code in unverified:
            continue
        if not _looks_like_formatted_identifier(code):
            continue
        if code not in context and code not in context_tight:
            unverified.append(code)
    return unverified


def _strip_fabricated_identifiers(answer: str, codes: list[str]) -> str:
    """Replace each fabricated identifier CODE wherever it appears in the
    answer with an honest placeholder, so the invented value cannot survive
    into the visible answer even if a corrective retry doesn't fix it. Unlike
    the quote-warning path (which only appends a banner and leaves the
    original text visible), an invented ID-shaped code reads as authoritative
    on sight — leaving it in place defeats the point even with a warning
    attached below it.
    """
    for code in codes:
        answer = answer.replace(code, "[not stated in this document]")
    return answer


# "N% / Rs. 17,118,112" — a milestone line carrying both its share of the
# contract and its cash amount.
_MILESTONE_ROW_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*%\s*/?\s*(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_MILESTONE_Q_RE = re.compile(
    r"\b(milestones?|payment obligations?|payment schedules?|contract value|"
    r"total value|payment terms|instal?lments?)\b", re.IGNORECASE,
)


def _append_milestone_total(answer: str, context: str, question: str) -> str:
    """State the total contract value when a milestone schedule is being asked about.

    Milestone schedules give each stage as "11% / Rs. 17,118,112" and never
    state the total, so a question about payment obligations gets a faithful
    table back and no headline number — the one figure the reader actually
    wants. Summing is left to Python rather than the model, which is the wrong
    tool for arithmetic over a dozen comma-formatted amounts.

    Guarded deliberately: the total is only asserted when the percentages
    account for ~100% of the contract, which is what proves the retrieved
    context held the COMPLETE schedule rather than a fragment. A partial
    schedule would otherwise produce a confident, badly wrong total.
    """
    if not answer or not _MILESTONE_Q_RE.search(question or ""):
        return answer
    if re.search(r"total contract value", answer, re.IGNORECASE):
        return answer

    rows = _MILESTONE_ROW_RE.findall(context or "")
    if len(rows) < 2:
        return answer

    # Walk the rows in document order and stop at the first point where the
    # percentages account for a whole contract. Deduplicating identical rows
    # would be wrong — a schedule legitimately contains eight stages of
    # "11% / Rs. 17,118,112" — while summing everything double-counts the
    # schedule each time it reappears in another retrieved page. Taking the
    # first run that reaches 100% handles both.
    pct_sum = amount_sum = 0.0
    used = 0
    for pct, amt in rows:
        try:
            p = float(pct)
            a = float(amt.replace(",", ""))
        except ValueError:
            return answer
        if pct_sum + p > 103.0:
            break
        pct_sum += p
        amount_sum += a
        used += 1
        if 97.0 <= pct_sum <= 103.0:
            break

    if not (97.0 <= pct_sum <= 103.0) or amount_sum <= 0 or used < 2:
        return answer

    return (f"{answer}\n\n**Total contract value: Rs. {amount_sum:,.0f}** "
            f"— the sum of all {used} milestone payments in the schedule "
            f"({pct_sum:.0f}% of contract value).")


# Only long enumerations are touched. A five-item list repeating a word is
# almost certainly saying something; a fifty-item list repeating one is padding.
_MIN_LIST_FOR_DEDUPE = 8

# Trailing qualifiers the answer layer adds when it re-lists a provision it has
# already listed: "Set-Off (repeat / Schedule 1 operational rule)", "Records
# Retention (repeat emphasis)", "Audit and Cost Allocation specifics".
_RX_LIST_LABEL_NOISE = re.compile(
    r"\b(?:repeat(?:ed|s)?|repetition|again|duplicate|details?|specifics?|"
    r"mechanics|emphasis|examples?|continued|cont\.?|reference|refs?)\b",
    re.IGNORECASE)

_RX_LIST_ITEM = re.compile(r"^(\s*)(\d{1,3})([.)])\s+(\S.*?)\s*$")


def _list_item_label(text: str) -> str:
    """The identifying head of a list item, normalised for comparison.

    Everything from the first bracket, dash or colon onward is elaboration, not
    identity: "Insurance (Seller to maintain ... Rs. 42,255,094)" and "Insurance
    amount and scope (comprehensive general liability ...)" are one clause
    described twice.
    """
    head = re.split(r"\s*[\(\[\u2014\u2013:]|\s+-\s+", text or "", maxsplit=1)[0]
    head = _RX_LIST_LABEL_NOISE.sub(" ", head)
    head = re.sub(r"[^a-z0-9 ]+", " ", head.lower())
    head = re.sub(r"\s+", " ", head).strip()
    # Drop a leading article and a trailing generic noun so "The Audit Right"
    # and "Audit Rights" collapse together.
    head = re.sub(r"^(?:the|a|an)\s+", "", head)
    # Strip a trailing generic noun only while enough remains to identify the
    # clause: "Audit Right" must not collapse to "audit", which would match far
    # too much.
    trimmed = re.sub(r"\s+(?:clause|clauses|provision|provisions|terms?|rights?|"
                     r"obligations?)$", "", head)
    if len(trimmed) >= 8:
        head = trimmed
    return head


def _dedupe_numbered_list(answer: str) -> tuple[str, int]:
    """Collapse entries a long numbered list states more than once.

    A fifty-item clause listing came back with roughly a third of it repeated —
    "Further Assurance" twice, "Set-Off" twice, "Records Retention" twice,
    "Liquidated Damages" three times — several of them labelled "(repeat)" by
    the answer layer itself. It knew, and listed them anyway. A list padded to
    fifty when it holds thirty-three distinct clauses misrepresents how much is
    in the document, which is the thing the reader was asking.

    Deterministic and conservative: only lists of at least eight items are
    touched, only an exact match on the normalised identifying head counts, and
    a head shorter than six characters is never matched on. The surviving entry
    keeps the longer of the two descriptions, so deduplication never costs
    detail, and the items are renumbered so the list a reader points back at is
    the list they were shown.
    """
    if not answer:
        return answer, 0
    lines = answer.splitlines()
    idxs = [i for i, ln in enumerate(lines) if _RX_LIST_ITEM.match(ln)]
    if len(idxs) < _MIN_LIST_FOR_DEDUPE:
        return answer, 0

    seen: dict = {}
    drop: set = set()
    for i in idxs:
        m = _RX_LIST_ITEM.match(lines[i])
        body = m.group(4)
        label = _list_item_label(body)
        if len(label) < 6:
            continue
        # A later item whose label BEGINS with an earlier one is the same
        # clause described at more length: "Insurance" then "Insurance amount
        # and scope", "Liquidated Damages" then "Liquidated Damages
        # per-contract cap". Nine characters minimum, because a short prefix
        # would swallow genuinely different clauses that share a first word.
        match_label = label if label in seen else next(
            (k for k in seen if len(k) >= 9 and label.startswith(k + " ")), "")
        if match_label:
            label = match_label
            first = seen[label]
            # Keep whichever description says more.
            if len(body) > len(_RX_LIST_ITEM.match(lines[first]).group(4)):
                fm = _RX_LIST_ITEM.match(lines[first])
                lines[first] = f"{fm.group(1)}{fm.group(2)}{fm.group(3)} {body}"
            drop.add(i)
        else:
            seen[label] = i
    if not drop:
        return answer, 0

    kept = [ln for i, ln in enumerate(lines) if i not in drop]
    # Renumber the surviving items so the numbering a reader refers back to is
    # the numbering they can see.
    n = 0
    for i, ln in enumerate(kept):
        m = _RX_LIST_ITEM.match(ln)
        if m:
            n += 1
            kept[i] = f"{m.group(1)}{n}{m.group(3)} {m.group(4)}"
    logger.info("Answer list: %d repeated entr(ies) merged from a %d-item list",
                len(drop), len(idxs))
    return "\n".join(kept), len(drop)


# The trailing MISSING_ITEMS block of the answer contract. Tolerant of the shapes
# the model actually produces: bold, a leading list marker, "None"/"n/a", and
# either a same-line value or a bulleted list beneath it.
_MISSING_ITEMS_RE = re.compile(
    r'(?im)^[ \t]*(?:[-*]\s*)?\**MISSING[_\s]*ITEMS\**[ \t]*:?[ \t]*'
    r'(?P<inline>[^\n]*)\n?(?P<body>(?:[ \t]*(?:[-*•]|\d+[.)])[^\n]*\n?)*)')

_MISSING_NONE_RE = re.compile(r'^(none|n/?a|nothing|no missing items?|-)\.?$', re.I)


# A candidate item can arrive still carrying the contract's own label - the
# model sometimes writes "MISSING_ITEMS: MISSING_ITEMS: none" on one line, and
# the inner copy then reads as the item's text. Strip any number of leading
# labels and list markers before deciding whether what is left is a real item.
_MISSING_LABEL_PREFIX_RE = re.compile(
    r'^[\s*>]*(?:[-*•]|\d+[.)])?[\s*]*MISSING[_\s]*ITEMS[\s*]*:?[\s*]*',
    re.IGNORECASE)


def _clean_missing_item(text: str) -> str:
    """One item's text, or "" when it is really a no-items marker."""
    t = (text or "").strip()
    for _ in range(3):
        stripped = _MISSING_LABEL_PREFIX_RE.sub("", t).strip()
        if stripped == t:
            break
        t = stripped
    t = t.strip().strip("*").strip()
    if not t or _MISSING_NONE_RE.match(t):
        return ""
    return t


def _extract_missing_items(answer: str) -> "tuple[str, list[str]]":
    """Split the answer from its MISSING_ITEMS block.

    Gives what the documents could not answer a slot of its own instead of a
    sentence buried somewhere in the prose. Two things follow. The reader sees
    it in one place rather than hunting for a hedge; and it becomes auditable
    per question - a run can count unanswered items instead of grepping answers
    for phrases like "not addressed", which is how they had to be counted
    before, and which cannot tell a real gap from the words appearing in a
    quote.

    Returns the answer with the block removed, and the items. An answer with no
    block - an older cached answer, or one of the zero-token fast paths that
    never went through the template - comes back untouched with an empty list,
    so nothing here depends on the model having complied.
    """
    if not answer:
        return answer, []
    items = []
    cleaned = answer
    # Every occurrence, not just the first. The model echoes the label once in
    # passing before emitting the real block at the end, and consuming only the
    # first left the trailing one rendered to the reader as raw contract syntax.
    while True:
        m = _MISSING_ITEMS_RE.search(cleaned)
        if not m:
            break
        inline = _clean_missing_item(m.group("inline") or "")
        if inline:
            items.append(inline)
        for line in (m.group("body") or "").splitlines():
            t = _clean_missing_item(re.sub(r'^[ 	]*(?:[-*•]|\d+[.)])[ 	]*', "", line))
            if t:
                items.append(t)
        cleaned = cleaned[:m.start()] + cleaned[m.end():]
    # De-duplicate while keeping order: the echo and the real block can name
    # the same item.
    seen, uniq = set(), []
    for i in items:
        k = re.sub(r'\s+', ' ', i).strip().lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(i)
    return cleaned.rstrip(), uniq


def _render_missing_items(items: "list[str]") -> str:
    """The reader-facing form of the block.

    Fixed wording rather than the model's own heading, so this one section
    cannot drift into talking about retrieval the way free prose does.
    """
    if not items:
        return ""
    if len(items) == 1:
        return "\n\nNot answered by these documents: " + items[0].rstrip(".") + "."
    return ("\n\nNot answered by these documents:\n"
            + "\n".join("- " + i.rstrip(".") + "." for i in items))


# Phrases that name the SEARCH rather than the document. Ordered: the longer,
# verb-carrying forms first, so "the context does not" becomes "these documents
# do not" rather than the ungrammatical "these documents does not".
#
# Done deterministically because prompting alone did not hold. The voice rule
# sits at the top of every template and the model still reaches for "the
# context" whenever the answer is a refusal - which is exactly when a reader is
# most likely to be told about machinery instead of about their documents.
_VOICE_SUBS = (
    (re.compile(r"\bthe (?:provided |retrieved |supplied |available )?context "
                r"(does not|doesn't)\b", re.IGNORECASE), "these documents do not"),
    (re.compile(r"\bthe (?:provided |retrieved |supplied |available )?context "
                r"(?:contains|includes|holds|provides)\b", re.IGNORECASE),
     "these documents contain"),
    (re.compile(r"\bthe (?:provided|retrieved|supplied|excerpted|available)\s+"
                r"(?:context|excerpts?|documents?|pages?|material|opinion|"
                r"sources?)\b", re.IGNORECASE), "these documents"),
    # Any modifier in front of "excerpts" still names the search, not the
    # document. The fixed list above missed "the agreement excerpts do not
    # state the auditor's name", and a one-word version of this rule then
    # missed "the Consultancy Agreement excerpts" - the modifier is often
    # the document's own name, which runs to several words.
    (re.compile(r"\bthe\s+[A-Za-z0-9'\u2019\s-]{0,70}?\s*excerpts?\b", re.IGNORECASE),
     "these documents"),
    (re.compile(r"\bin the excerpts?\b", re.IGNORECASE), "in these documents"),
    (re.compile(r"\bthe excerpted pages?\b", re.IGNORECASE), "these documents"),
    (re.compile(r"\bthe context\b", re.IGNORECASE), "these documents"),
    # Catch-all for the rest of the class: "the retrieved judgment", "the
    # provided agreement", "the supplied opinion". Dropping the adjective is
    # always grammatical and always right - the noun IS the document, and
    # whether it was retrieved is a fact about the search, not about it.
    (re.compile(r"\bthe (?:retrieved|provided|supplied|excerpted)\s+(?=[a-z])",
                re.IGNORECASE), "the "),
)

# Cleanups for what the substitutions above can leave behind.
_VOICE_FIXUPS = (
    (re.compile(r"\bthese documents does\b", re.IGNORECASE), "these documents do"),
    (re.compile(r"\bthese documents is\b", re.IGNORECASE), "these documents are"),
    (re.compile(r"\bthese documents was\b", re.IGNORECASE), "these documents were"),
    (re.compile(r"\bthese documents contains\b", re.IGNORECASE), "these documents contain"),
    (re.compile(r"\bthese documents (?:only )?(?:contains|holds)\b", re.IGNORECASE),
     "these documents contain"),
    (re.compile(r"\bThese documents\b(?=[^.]*\bhere\b)"), "These documents"),
)


def _rewrite_answer_voice(answer: str) -> tuple[str, int]:
    """Say "these documents", never "the retrieved context".

    Applied only OUTSIDE quotation marks and quote blocks: a document that
    genuinely uses one of these phrases must keep its own wording, or a
    verbatim quote stops being verbatim and the citation checks that just ran
    over it would be checking different text than the reader sees.

    Runs after those checks for the same reason - verification reads what the
    model wrote, the reader reads this.
    """
    if not answer:
        return answer, 0

    # Spans to leave alone: anything in double quotes, and any "> " quote line.
    protected = []
    for m in re.finditer(r'["“][^"“”\n]{0,500}["”]', answer):
        protected.append((m.start(), m.end()))
    for m in re.finditer(r'^\s*>.*$', answer, re.M):
        protected.append((m.start(), m.end()))

    def _inside(pos):
        return any(a <= pos < b for a, b in protected)

    out, changed = answer, 0
    for rx, repl in _VOICE_SUBS:
        pieces, last = [], 0
        for m in rx.finditer(out):
            if _inside(m.start()):
                continue
            pieces.append(out[last:m.start()])
            # Preserve sentence-initial capitalisation.
            text = repl
            if m.group(0)[:1].isupper():
                text = text[:1].upper() + text[1:]
            pieces.append(text)
            last = m.end()
            changed += 1
        if pieces:
            pieces.append(out[last:])
            out = "".join(pieces)
            # Positions moved; recompute the protected spans for the next rule.
            protected = []
            for m in re.finditer(r'["“][^"“”\n]{0,500}["”]', out):
                protected.append((m.start(), m.end()))
            for m in re.finditer(r'^\s*>.*$', out, re.M):
                protected.append((m.start(), m.end()))
    for rx, repl in _VOICE_FIXUPS:
        out = rx.sub(repl, out)
    return out, changed


def _verify_answer_citations(answer: str, context: str, question: str = "") -> list[str]:
    """Deterministically verify every quoted span in the answer is actually
    present (whitespace/case-insensitive) in the retrieved context.

    Catches a real, confirmed failure mode: the answer LLM presenting a
    paraphrase as if it were a verbatim quote (e.g. "The definition is broad,
    capturing oral, written, electronic, and physical disclosures" when the
    source actually reads "...whether orally, in writing, or in electronic or
    physical form..."). This is a substring check against the exact context the
    model was given — not another LLM's opinion — so it can't be fooled the same
    way the holistic grounding check can.

    Does NOT catch quotes that are verbatim-accurate but mislabeled with a wrong
    section number, or facts that were already wrong in the stored wiki page at
    ingest time — both are real but require comparing against the ORIGINAL
    source document, not the retrieved context, which is a separate check.

    question: when a question invents a term/name that doesn't exist in the
    corpus (a "trap" — e.g. asking about a fabricated project name), a correct
    answer quotes that invented term back to say it ISN'T real ("the excerpt
    does not mention 'Shadow-WaveSync'"). That's the model correctly refusing
    to hallucinate, not a fabricated source citation — exclude quotes that
    substantially echo the question's own text from this check.
    """
    if not answer or not context:
        return []

    def _norm(s: str) -> str:
        return _norm_for_match(s)

    ctx_norm = _norm(_strict_verification_corpus(context))
    known_titles = _known_page_titles(context)
    question_norm = _norm(question) if question else ""
    unverified = []
    for q in _QUOTE_SPAN_RE.findall(answer):
        if not _is_checkable_quote(q):
            continue
        qn = _norm(q)
        if _alnum_only(qn) in known_titles:
            continue  # citation label (page title in quotes), not a content quote
        if qn in ctx_norm:
            continue
        if question_norm and qn in question_norm:
            continue  # quoting the question's own (possibly invented) term back

        def _seg_ok(seg: str) -> bool:
            sn = _norm(seg)
            return sn in ctx_norm or (bool(question_norm) and sn in question_norm)

        segments = _quote_segments(q)
        if segments and all(_seg_ok(seg) for seg in segments):
            continue
        if _INTERNAL_QUOTE_SPLIT_RE.search(q) and _segments_effectively_verified(q, ctx_norm, question_norm):
            continue
        unverified.append(q.strip())
    return unverified


# Markdown emphasis the model writes INSIDE a quoted span, plus unicode
# space/apostrophe variants that differ from the source only visually. Folded
# ONLY for the severity classification below — never for verification itself.
# The distinction matters: relaxing what counts as verified would let altered
# text pass as an exact quote, whereas relaxing severity classification only
# decides how an already-flagged quote is described to the reader.
_SEVERITY_FOLD_RE = re.compile(r'[*_`]+')
_UNICODE_SPACE_RE = re.compile(r'[          ]')
_APOSTROPHE_VARIANTS_RE = re.compile(r'[‘’‛ʼ]')


def _norm_for_severity(s: str) -> str:
    """_norm_for_match plus markdown-emphasis and unicode space/apostrophe folding.

    Confirmed live: a genuine passage was reported as absent from context purely
    because the model bolded a word inside its own quotation marks
    ("...ordinary business **under-performance** from...").
    """
    s = _UNICODE_SPACE_RE.sub(' ', s)
    s = _APOSTROPHE_VARIANTS_RE.sub("'", s)
    s = _SEVERITY_FOLD_RE.sub('', s)
    return _norm_for_match(s)


def _drop_unverifiable_reference_quotes(answer: str, absent: list[str]) -> tuple[str, int]:
    """Delete fabricated "| Quote: …" excerpts from References lines.

    A quote the check found NOWHERE in the retrieved context is not evidence, so
    printing it and appending a warning underneath leaves the invented sentence
    on screen — the reader still sees an authoritative-looking excerpt, and the
    warning is the only thing standing between them and treating it as real.
    Measured on Q60/Q64, whose ANSWERS were both exactly right (the correct Tata
    entities, matching ground truth): each supported its correct answer with a
    quotation that paraphrased the page instead of copying it, and the resulting
    banner made a correct answer read as untrustworthy.

    Removing the excerpt keeps everything that was true — the document, the
    clause, the answer itself — and drops only the part that could not be
    substantiated. Reference lines only: a quote embedded in the answer's prose
    cannot be excised without rewriting the sentence around it, so those still
    take the banner. Returns (answer, number_removed).

    Two reference-line shapes, handled separately: the documented "FileName,
    Clause | Quote: ..." form, and a bare-quote form the model also produces —
    "[1] FileName, "quoted text."" with no "| Quote:" delimiter at all. Only
    the first shape was originally handled; confirmed missed live on Q105,
    whose citation read `- [1] ... Judgment 5 (1), "Under the Letter of
    Intent..."` — the quote was correctly flagged unverified, but the absent
    delimiter meant this function's line-scan never found it to remove.
    """
    if not absent:
        return answer, 0
    targets = [_norm_for_match(q) for q in absent if q and q.strip()]
    if not targets:
        return answer, 0

    def _matches_target(quoted_norm: str) -> bool:
        # Substring either way: the flagged span may be the whole excerpt or,
        # when the model spliced fragments, only the piece that failed
        # verification.
        return bool(quoted_norm) and any(
            t and (t in quoted_norm or quoted_norm in t) for t in targets
        )

    removed = 0
    out = []
    for line in answer.split("\n"):
        idx = line.find("| Quote:")
        if idx != -1:
            quoted = _norm_for_match(line[idx + len("| Quote:"):])
            if _matches_target(quoted):
                out.append(line[:idx].rstrip())
                removed += 1
                continue
            out.append(line)
            continue

        # Bare-quote form: a reference line (starts, after an optional bullet,
        # with a "[N]"-style citation marker) whose quote sits directly in the
        # line with no delimiter. Strip only the quoted span itself plus one
        # adjoining separator, leaving the citation/document/clause intact.
        if _RX_REFERENCE_LINE_START.match(line):
            new_line = line
            stripped_any = False
            for m in list(_QUOTE_SPAN_RE.finditer(line))[::-1]:
                if not _is_checkable_quote(m.group(1)):
                    continue
                if not _matches_target(_norm_for_match(m.group(1))):
                    continue
                start, end = m.start(), m.end()
                # Absorb one leading separator (", " / " - " / " | ") so removal
                # doesn't leave a dangling comma or dash before the cut.
                sep = re.match(r'\s*[,\-|]\s*$', new_line[max(0, start - 3):start])
                if sep:
                    start -= len(sep.group(0))
                new_line = new_line[:start].rstrip() + new_line[end:]
                stripped_any = True
            if stripped_any:
                out.append(new_line)
                removed += 1
                continue

        out.append(line)
    return "\n".join(out), removed


def _split_unverified_by_severity(unverified: list[str], context: str,
                                  question: str = "") -> tuple[list[str], list[str]]:
    """Split flagged quotes into (absent, prose_sourced) by how serious they are.

    _verify_answer_citations deliberately verifies only against each page's
    ingest-verified **Supporting Quotes** block (see _block_verification_text) —
    the rest of a page is LLM-synthesized descriptive prose, so letting a quote
    match it would let synthesized text pass as a genuine document excerpt.

    That strictness is right, but it collapses two very different failures into
    one alarming banner. Measured over 400 logged answers: only 27% of retrieved
    context sits inside a Supporting Quotes block at all (32.7% of pages have
    none), so the answer LLM routinely quotes real retrieved material that has
    no verifiable counterpart — and 55.8% of answers carried a CITATION WARNING,
    which trains the reader to ignore it and buries the rare genuine fabrication.

    Two outcomes, kept separate so each can be reported at its true severity:
      absent        — the text appears NOWHERE in the retrieved context. This is
                      the fabrication signal the check exists for; unchanged.
      prose_sourced — the text IS in the retrieved context, just in synthesized
                      page prose rather than a verified quote block. Real content,
                      wrong provenance claim: it's a paraphrase presented as an
                      exact quote, not an invention.

    Verification itself is NOT loosened — nothing that failed before passes now.
    This only decides how a failure is described.
    """
    if not unverified:
        return [], []
    ctx_norm = _norm_for_severity(context)
    question_norm = _norm_for_severity(question) if question else ""

    def _in_full_context(text: str) -> bool:
        tn = _norm_for_severity(text)
        if tn in ctx_norm or (question_norm and tn in question_norm):
            return True
        # Same ellipsis-splice allowance the strict check makes: a citation may
        # legitimately join non-adjacent spans, so each piece counts separately.
        segments = _quote_segments(text)
        return bool(segments) and all(
            _norm_for_severity(s) in ctx_norm or
            (question_norm and _norm_for_severity(s) in question_norm)
            for s in segments
        )

    absent, prose_sourced = [], []
    for q in unverified:
        (prose_sourced if _in_full_context(q) else absent).append(q)
    return absent, prose_sourced


def _nearest_verbatim_span(quote: str, context: str) -> str | None:
    """The context span a flagged (paraphrased) quote most closely paraphrases.

    The citation retry previously just re-listed the bad quotes and re-asked —
    so the model (esp. gpt-5-nano) reworded them again the same way, and the
    retry "did not improve" on nearly every answer (confirmed live). The verbatim
    text is right there in context under a "**Supporting Quotes:** > …" block;
    the model just isn't copying it. Return the exact span so the retry can show
    the model precisely what to paste. Content-word overlap ≥ 0.6 required, so a
    weak coincidental match never suggests the wrong clause.
    """
    qwords = {w for w in re.findall(r'[a-z0-9]+', quote.lower()) if len(w) > 3}
    if len(qwords) < 3:
        return None
    corpus = _strict_verification_corpus(context)
    # Candidate spans: supporting-quote lines (after "> ") first, then sentences.
    cands = re.findall(r'>\s*([^\n]{20,400})', corpus)
    cands += re.split(r'(?<=[.!?])\s+', corpus)
    best, best_score = None, 0.0
    for cand in cands:
        cwords = {w for w in re.findall(r'[a-z0-9]+', cand.lower()) if len(w) > 3}
        if not cwords:
            continue
        overlap = len(qwords & cwords) / len(qwords)
        if overlap > best_score:
            best, best_score = cand.strip(), overlap
    return best if (best and best_score >= 0.6) else None


# Page blocks in wiki_content look like "## Some Topic – DocID (DocType)\n<content>".
_PAGE_BLOCK_RE = re.compile(r'^##\s+(.+?)\s*\n(.*?)(?=^##\s+|\Z)', re.MULTILINE | re.DOTALL)

# Document-number tokens like "CCD08", "CCD-21", "JVA 01" — used to compare what
# a citation CLAIMS against what document the quote is ACTUALLY found in.
# Uses a negative lookbehind for a preceding letter (not \b) for the leading
# boundary: "_" counts as a word character in regex, so \b never fires before
# "CCD" in underscore-joined filenames like "Test_CCD_08.txt".
_DOC_NUM_RE = re.compile(r'(?<![A-Za-z])(jva|nda|sha|sa|ccd|judgment|opinion|msa)[-_\s]?0*(\d{1,3})\b', re.IGNORECASE)


def _extract_doc_number_tokens(text: str) -> set[tuple[str, str]]:
    return {(t.lower(), n) for t, n in _DOC_NUM_RE.findall(text)}


def _verify_citation_attribution(answer: str, context: str) -> list[str]:
    """Deterministically check whether a cited quote is attributed to the right
    document — catches the confirmed CCD_08 bug where a genuinely real quote
    (correctly found in context) was attributed to the wrong source file (e.g.
    a quote that actually lives under a "...Test-CCD35..." page block was cited
    in the answer as "[1] Test_CCD_08.txt, ...").

    Only fires when both the citation's claimed doc-number token and the quote's
    actual source-block doc-number token share the same TYPE (e.g. both "ccd")
    but a DIFFERENT number — this keeps false positives low since it never
    flags citations it can't confidently compare.
    """
    if not answer or not context:
        return []

    def _norm(s: str) -> str:
        return _norm_for_match(s)

    blocks = [(title.strip(), body) for title, body in _PAGE_BLOCK_RE.findall(context)]
    if not blocks:
        return []
    # Candidate discovery uses each block's FULL body, not the Supporting-Quotes-
    # restricted text _verify_answer_citations uses — that stricter scope exists to
    # stop fabricated quotes from passing as genuine (a separate, still-strict check
    # below in this same function's caller). Here the question is different: "is
    # there ANY block that could confirm this citation is attributed correctly,"
    # and using only the verified-quotes subset undercounts real candidates. Confirmed
    # live: a shared risk-assessment sentence genuinely appears in both Opinion_37's
    # Supporting Quotes section AND Opinion_41's ordinary prose (ingest-time filtering
    # only kept it under Opinion_37's heading) — citing Opinion_41 is factually
    # correct, but restricting candidate search to Supporting-Quotes-only meant
    # Opinion_41 never appeared as a candidate, so the only candidate found (Opinion_37)
    # didn't match the citation and it was wrongly flagged as misattributed.
    norm_blocks = [(title, _norm(body)) for title, body in blocks]
    known_titles = _known_page_titles(context)

    # Map each block title to its "[From: <filename>]" label (emitted by
    # get_context). The filename carries the file's own identifier (e.g.
    # "Test_JVA_05"), which the display title may not — fold it into the
    # "actual source" identifier so a quote cited by its correct filename isn't
    # falsely flagged just because the page's type-label differs from the file.
    def _from_label(body: str) -> str:
        m = re.search(r'^\[From:\s*(.+?)\]\s*$', body, re.MULTILINE)
        return m.group(1) if m else ""
    from_by_title = {title.strip(): _from_label(body) for title, body in blocks}

    mismatches = []
    for m in _QUOTE_SPAN_RE.finditer(answer):
        quote = m.group(1)
        if not _is_checkable_quote(quote):
            continue
        qn = _norm(quote)
        if _alnum_only(qn) in known_titles:
            continue  # citation label (page title in quotes), not a content quote
        candidate_titles = [title for title, body in norm_blocks if qn in body]
        if not candidate_titles:
            # Ellipsis-spliced quote ("BETWEEN: ... Each of the aforesaid...") —
            # find blocks containing every segment instead of the whole string.
            segments = _quote_segments(quote)
            if segments:
                candidate_titles = [title for title, body in norm_blocks
                                     if all(_norm(seg) in body for seg in segments)]
        if not candidate_titles:
            # Nested/adjacent-quote span (see _QUOTE_SPAN_RE) — find blocks
            # containing every quote-split inner piece instead of the whole string.
            inner_segments = _quote_inner_segments(quote)
            if inner_segments:
                candidate_titles = [title for title, body in norm_blocks
                                     if all(_norm(seg) in body for seg in inner_segments)]
        if not candidate_titles:
            continue  # not found at all — already caught by _verify_answer_citations

        label_start = max(0, m.start() - 150)
        label = answer[label_start:m.start()]
        claimed = _extract_doc_number_tokens(label)
        if not claimed:
            continue  # nothing to compare against

        # A citation is only flagged if it mismatches EVERY block that contains
        # this quote — not just the first one found. Shared boilerplate (e.g. an
        # identical Section 15.8 "Court of Chancery" forum clause repeated
        # verbatim across dozens of JVAs/SHAs/NDAs/MSAs in the same context)
        # legitimately lives under many documents at once; checking only the
        # first match produced false positives whenever the model cited a
        # DIFFERENT document that also genuinely contains the same shared text.
        # Confirmed live: a dispute-resolution question citing identical
        # Chancery-clause text as JVA1/SHA10/JVA26 was flagged as misattributed
        # to "JVA14" purely because that was the first of 7 blocks in context
        # sharing the exact same sentence — JVA1 and SHA10 also genuinely had it.
        # any_match: a candidate confirms the citation is correct.
        # any_confident_mismatch: at least one candidate was confidently
        # COMPARABLE (shared type, or a numbered claim vs. an unnumbered title)
        # and did NOT match — a type-incomparable candidate (e.g. an SHA block
        # when the claim is a JVA number) is skipped entirely rather than
        # treated as either a match or a mismatch, so one ambiguous candidate
        # can't mask a confident mismatch found against a different candidate.
        any_match = False
        any_confident_mismatch = False
        rep_title, rep_actual_str, rep_kind = candidate_titles[0], None, None
        for block_title in candidate_titles:
            actual_text = f"{block_title} {from_by_title.get(block_title, '')}"
            actual = _extract_doc_number_tokens(actual_text)
            if actual:
                shared_types = {t for t, _ in claimed} & {t for t, _ in actual}
                if not shared_types:
                    continue  # can't confidently compare this candidate — skip it
                if claimed & actual:
                    any_match = True
                    break
                any_confident_mismatch = True
                if rep_actual_str is None:
                    rep_title = block_title
                    rep_actual_str = ", ".join(f"{t.upper()}{n}" for t, n in actual)
                    rep_kind = "type"
            else:
                # The real source page's title has no numbered identifier at all
                # (e.g. "NDA-Tata" — a party-name identifier, not "NDA37"). If the
                # citation's claimed identifier appears anywhere in this
                # candidate's title/filename, treat it as a match.
                block_norm = actual_text.lower().replace("-", "").replace(" ", "").replace("_", "")
                if any(f"{t}{n}" in block_norm for t, n in claimed):
                    any_match = True
                    break
                any_confident_mismatch = True
                if rep_actual_str is None:
                    rep_title = block_title
                    rep_kind = "noid"

        if not any_match and any_confident_mismatch:
            claimed_str = ", ".join(f"{t.upper()}{n}" for t, n in claimed)
            if rep_kind == "type":
                mismatches.append(
                    f'Quote "{quote[:80]}..." cited as {claimed_str} but actually found under '
                    f'"{rep_title}" ({rep_actual_str})'
                )
            else:
                mismatches.append(
                    f'Quote "{quote[:80]}..." cited as {claimed_str} but actually found under '
                    f'"{rep_title}" (no matching identifier in the real source title)'
                )
    return mismatches


def _autocorrect_citation_attribution(answer: str, context: str) -> tuple[str, int]:
    """Deterministically fix a wrong-document citation label in place, instead
    of only warning about it.

    _verify_citation_attribution already identifies both the WRONG token a
    citation claims (e.g. "JVA18") and the CORRECT one the quote actually lives
    under (e.g. "JVA45") when they share a doc-type — confirmed live: an
    "across all JVAs" answer cited JVA45's clause as JVA18, purely because
    JVA18 happened to be the first of several JVAs sharing a repeated
    boilerplate forum clause. Only that "shared type, different number" case
    is corrected — the replacement text is unambiguous. A candidate whose real
    title has no numbered identifier at all (e.g. "NDA-Tata", a party-name
    identifier) has no safe substitution text and is left to the existing
    warning.

    Never worsens the answer: after substitution, citation-attribution is
    re-checked, and the fix is discarded (original answer returned) unless it
    strictly reduces the number of misattributed quotes.

    Returns (possibly-corrected answer, n_corrections_applied).
    """
    if not answer or not context:
        return answer, 0

    def _norm(s: str) -> str:
        return _norm_for_match(s)

    blocks = [(title.strip(), body) for title, body in _PAGE_BLOCK_RE.findall(context)]
    if not blocks:
        return answer, 0
    norm_blocks = [(title, _norm(body)) for title, body in blocks]
    known_titles = _known_page_titles(context)

    def _from_label(body: str) -> str:
        m = re.search(r'^\[From:\s*(.+?)\]\s*$', body, re.MULTILINE)
        return m.group(1) if m else ""
    from_by_title = {title.strip(): _from_label(body) for title, body in blocks}

    # (span_start, span_end, replacement_text) in reverse-position order so
    # earlier substitutions don't shift the spans of later ones.
    fixes: list[tuple[int, int, str]] = []

    for m in _QUOTE_SPAN_RE.finditer(answer):
        quote = m.group(1)
        if not _is_checkable_quote(quote):
            continue
        qn = _norm(quote)
        if _alnum_only(qn) in known_titles:
            continue
        candidate_titles = [title for title, body in norm_blocks if qn in body]
        if not candidate_titles:
            segments = _quote_segments(quote)
            if segments:
                candidate_titles = [title for title, body in norm_blocks
                                     if all(_norm(seg) in body for seg in segments)]
        if not candidate_titles:
            inner_segments = _quote_inner_segments(quote)
            if inner_segments:
                candidate_titles = [title for title, body in norm_blocks
                                     if all(_norm(seg) in body for seg in inner_segments)]
        if not candidate_titles:
            continue

        label_start = max(0, m.start() - 150)
        label = answer[label_start:m.start()]
        claim_matches = list(_DOC_NUM_RE.finditer(label))
        if not claim_matches:
            continue
        claimed = {(mm.group(1).lower(), mm.group(2)) for mm in claim_matches}

        any_match = False
        any_confident_mismatch = False
        rep_actual: "tuple[str, str] | None" = None
        for block_title in candidate_titles:
            actual_text = f"{block_title} {from_by_title.get(block_title, '')}"
            actual = _extract_doc_number_tokens(actual_text)
            if not actual:
                continue  # no numbered id in the real title — no safe substitution
            shared_types = {t for t, _ in claimed} & {t for t, _ in actual}
            if not shared_types:
                continue
            if claimed & actual:
                any_match = True
                break
            any_confident_mismatch = True
            if rep_actual is None:
                rep_actual = next((a for a in actual if a[0] in shared_types), next(iter(actual)))

        if any_match or not any_confident_mismatch or rep_actual is None:
            continue

        correct_type, correct_num = rep_actual
        for mm in claim_matches:
            if mm.group(1).lower() != correct_type:
                continue
            new_type = correct_type.upper() if mm.group(1)[0].isupper() else correct_type
            fixes.append((label_start + mm.start(), label_start + mm.end(), f"{new_type}{correct_num}"))

    if not fixes:
        return answer, 0

    fixes.sort(key=lambda f: f[0], reverse=True)
    corrected = answer
    for start, end, repl in fixes:
        corrected = corrected[:start] + repl + corrected[end:]

    try:
        before_n = len(_verify_citation_attribution(answer, context))
        after_n = len(_verify_citation_attribution(corrected, context))
    except Exception as e:
        logger.error("Citation autocorrect safety check failed: %s", e)
        return answer, 0
    if after_n < before_n:
        return corrected, before_n - after_n
    return answer, 0


# Appended to the original prompt for a one-shot corrective retry when
# citation verification flags quotes as unverifiable or misattributed —
# rather than just warning the user after the fact, give the model one
# chance to fix it before falling back to a warning.
_CITATION_RETRY_ADDENDUM = """

---
IMPORTANT CORRECTION NEEDED: Your previous answer to this question included quoted \
passages that could not be verified against the CONTEXT above (they were paraphrased, \
not exact, or attributed to the wrong document). Each flagged passage is shown below, \
and where the CONTEXT contains the matching verbatim text, the exact text to copy is \
given right after it:
{flagged}

Write the answer again. For each flagged passage: if an "Exact text in CONTEXT" line is \
shown, replace your quotation with THAT text copied character-for-character (do not \
reword "shall not exceed" into "caps at", do not add a clause/section number that isn't \
in the quote). If no exact text is shown, describe the provision in your own words \
WITHOUT quotation marks. Do not repeat the same error."""


# Words carrying no retrieval signal — excluded before measuring how much of a
# question actually appears in the retrieved context (see _refusal_looks_wrong).
_REFUSAL_CHECK_STOPWORDS = frozenset("""
what which when where does do did is are was were the this that these those and
or but for with from into onto about under over between per any some each all
how why who whom whose can could shall should will would may might must have has
had been being its it their there here you your our we they them then than such
said also more most other another only just very much many both same
please tell explain describe state provide give list name mention say
question answer document documents agreement agreements clause clauses section
sections provided context text page pages according per terms term
""".split())

# Fraction of a question's distinctive words that must appear in the retrieved
# context before a refusal is treated as suspect. Set high on purpose: the point
# is to catch answers refusing over material demonstrably sitting in front of
# them, not to argue with refusals about genuinely absent topics.
_REFUSAL_RECHECK_MIN_OVERLAP = 0.7

# Below this many distinctive words the overlap ratio is noise — a three-word
# question clears any threshold by accident.
_REFUSAL_RECHECK_MIN_TERMS = 4

# Prefix length for the stem match below. Five characters keeps "emission"/
# "emissions" and "achieve"/"achieving" together without collapsing genuinely
# different words ("terminate"/"termination" share a stem and a meaning;
# "confidential"/"confirmation" diverge well before the fifth character).
_REFUSAL_STEM_LEN = 5


def _refusal_looks_wrong(question: str, context: str) -> bool:
    """True when an answer refused but the context still holds most of what the
    question asked about.

    Exists because the same document, in the same session, is sometimes cited
    correctly for one question and then declared absent for the next — the
    retrieval was fine and the refusal was the model's own miss. Deliberately
    keyword-level and deliberately strict: it only decides whether a second
    generation pass is worth one call, and the retry that follows is still free
    to refuse again.
    """
    if not question or not context:
        return False
    ctx = _norm_for_match(context)
    # Prefix-keyed so a question's "achieve" still matches the context's
    # "achieving" — an inflected form is the same retrieval signal, and exact
    # matching alone put ordinary morphology on the wrong side of the threshold.
    ctx_stems = {w[:_REFUSAL_STEM_LEN] for w in re.findall(r'[a-z][a-z\-]+', ctx)}
    terms = {
        w for w in re.findall(r'[a-z][a-z\-]{3,}', _norm_for_match(question))
        if w not in _REFUSAL_CHECK_STOPWORDS
    }
    if len(terms) < _REFUSAL_RECHECK_MIN_TERMS:
        return False
    present = sum(1 for w in terms if w in ctx or w[:_REFUSAL_STEM_LEN] in ctx_stems)
    return present / len(terms) >= _REFUSAL_RECHECK_MIN_OVERLAP


# Appended for a one-shot retry when an answer refused over context that still
# contains the question's subject matter. Re-states the refusal as a legitimate
# outcome on purpose — the failure being corrected is a missed reading, and an
# addendum that only rewarded finding something would trade it for invention.
_REFUSAL_RETRY_ADDENDUM = """

---
IMPORTANT: Your previous answer stated that the CONTEXT above does not cover this \
question. Before settling on that, read the CONTEXT again in full — the terms this \
question asks about do appear in it, and the relevant provision may be worded \
differently from the question, sit under an unexpected heading, or be split across \
more than one passage.

Answer the question again from the CONTEXT. If, having re-read it, the CONTEXT \
genuinely does not answer the question, say so plainly again — that is a correct \
answer and is preferred over guessing. Never invent, infer, or fill in a fact that \
is not in the CONTEXT."""


# A line is "complete" if it ends with a terminal mark a truncated model
# stream wouldn't leave dangling: sentence punctuation, a closing quote, a
# closed markdown table row ("...|"), a closing code fence, or is blank.
_COMPLETE_LINE_RE = re.compile(r'(^\s*$)|([.!?"\'`)\]|:])\s*$')


def _truncate_to_last_complete_unit(text: str) -> str:
    """Drop a dangling, truncated tail (e.g. a table row cut off mid-cell) so a
    still-truncated retry ships a clean partial answer instead of a broken
    fragment. Only trims from the end and only if most of the answer survives —
    never used on an answer that already ends cleanly."""
    stripped = text.rstrip()
    if not stripped or _COMPLETE_LINE_RE.search(stripped):
        return text
    lines = stripped.split("\n")
    for i in range(len(lines) - 2, -1, -1):
        if _COMPLETE_LINE_RE.search(lines[i]):
            kept = "\n".join(lines[: i + 1]).rstrip()
            if len(kept) >= 0.4 * len(stripped):
                return kept + "\n\n_[Response was cut off before completion — some entries may be missing.]_"
            break
    return text


def generate_answer(question: str, wiki_content: str, selected_titles: list, session_id: str, bm25_count: int = 0, page_selection_usage: dict = None, conversation_context: str = "", intent: str = "factual", unconfirmed_doc_reference: bool = False, scope_note: str = "", scope_warning: str = "", clause_directive: str = "", ambiguity_directive: str = "") -> dict:
    """Generate an answer using the provided wiki content.

    scope_note: a plain-English disclosure of HOW the scope was decided, when it
        was decided by inference rather than by the question itself (currently:
        conversational carryover). Appended verbatim as a deterministic banner —
        the model must not be able to omit the one line explaining why it
        answered about a document the question never named.

    scope_warning: a plain-English WARNING appended when the question named a
        specific counterparty that could not be pinned to one document, so the
        answer was drawn from a broad corpus search that may have surfaced the
        wrong same-family document. Stronger than scope_note (emitted as
        [SCOPE WARNING], which the frontend hoists into the visible body).

    ambiguity_directive: a PROMPT instruction (not display text) used when the
        question identifies its document by a description that fits several
        documents equally. Tells the model to answer for each candidate with
        every value attributed to its own document, instead of reporting the
        top-ranked one as though it were the only match. Prepended like
        clause_directive — see that note for why mid-prompt placement fails.
    """
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
            "cached_prompt_tokens": page_selection_usage.get("cached_prompt_tokens", 0),
        })

    # A retrieval failure (e.g. a transient embedding/DB hiccup) can leave
    # wiki_content as a non-empty-but-pure-whitespace string — `if not
    # wiki_content` alone doesn't catch that, so it silently reaches the LLM
    # with a blank {context} slot. The model then correctly (from its own
    # view) says "not covered", but the answer looks like a normal confident
    # response instead of surfacing that retrieval actually returned nothing.
    if not wiki_content or not wiki_content.strip():
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

    # A failed cross-reference (see _cross_reference_failure_answer) is answered
    # without calling the LLM at all — whether the citing document was
    # retrieved is a fact already known with certainty, and a context-injected
    # warning alone was confirmed live not to hold: the model can still write a
    # compliant refusal sentence and then add a second section quoting the
    # wrong document's clause anyway. Bypassing generation removes that failure
    # mode by construction rather than asking the model not to take it.
    _xref_answer = _cross_reference_failure_answer(question, pages, selected_titles)
    if _xref_answer:
        logger.info("Cross-reference proven unsatisfiable — answering deterministically, no LLM call")
        return {
            "answer": _xref_answer,
            "pages_used": [],
            "files_used": [],
            "selected_titles": selected_titles,
            "relations": relations,
            "usage": {},
            "confidence_score": 95,
            "confidence_reason": "Deterministic: the cited document is confirmed absent from retrieval.",
            "not_covered": True,
            "citation_check": {"total": 0, "unverified": 0, "misattributed": 0,
                               "verified": 0},
            "token_breakdown": token_breakdown,
            "token_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    from services.prompts import (
        ANSWER_PROMPT, ASSESSMENT_PROMPT, COMPARISON_PROMPT,
        OBLIGATION_PROMPT, DRAFTING_PROMPT,
    )

    conv_block = ""
    if conversation_context:
        conv_block = f"\nPREVIOUS CONVERSATION:\n{conversation_context}\n"

    # Build metadata block with party names from ingested documents
    metadata_block = _build_metadata_block(session_id, selected_titles, pages)

    # User-editable global house-style rules ("use a formal tone", "say clause
    # not section", ...) — appended on top of the fixed per-intent rules above.
    from services import rules as _rules
    house_rules_block = _rules.enabled_rules_block()

    # The question named a document by pattern ("service agreement 1") that
    # check_disambiguation_node could not confirm exists in this corpus. Two
    # placements were tried and verified live before this one: appended to
    # `question` (embeds mid-prompt, right before a rigid REQUIRED OUTPUT
    # FORMAT directive the model follows very literally — got partial
    # compliance, stopped the false "Service Agreement 1 (Test_SA_44)" title
    # but dropped the disclosure-first-sentence requirement); prepended to
    # house_rules_block (substituted right after metadata_block, but
    # metadata_block itself can be large — party names and matter references
    # across every selected document — so the note still ends up hundreds of
    # characters deep, diluting it the same way). Prepending directly onto the
    # fully-composed prompt, before even the model's persona-setting opening
    # sentence, is the most salient position available.
    _unconfirmed_doc_note = (
        "CRITICAL — UNCONFIRMED DOCUMENT REFERENCE: No document in this corpus "
        "matches the specific document number/name referenced in the question "
        "below. This does not override the required output structure below "
        "(reasoning block first, if one is specified) — it constrains what "
        "goes inside it. The FIRST LINE of your reasoning (or, if no reasoning "
        "block is specified, the FIRST SENTENCE of your answer) MUST state "
        "this plainly, e.g. \"No document matching '<name>' exists in this "
        "corpus.\" The final answer itself must ALSO open with that same "
        "disclosure as its first sentence, before any table or analysis. Do "
        "NOT use the referenced name as a title or heading anywhere in the "
        "answer (e.g. never write \"... in Service Agreement 1\" as a "
        "heading). If related documents exist that the user may have meant, "
        "name them by their REAL identifiers and offer them as likely "
        "alternatives — but never answer as if the referenced document were "
        "one of them.\n\n"
    ) if unconfirmed_doc_reference else ""

    # Clause-number mapping from clause_map (intent_agent). Same placement
    # rationale as the note above: scope_note is display-only (appended to the
    # finished answer), and anything embedded mid-prompt near the question gets
    # diluted — verified live when this mapping was passed via scope_note and
    # the model answered "not covered" with the mapped section sitting in its
    # own context. Prepending is the one placement that reliably lands.
    _clause_directive_note = (
        f"CLAUSE NUMBER MAPPING: {clause_directive}\n\n"
    ) if clause_directive else ""

    # The question's identifying description matched several documents equally
    # (resolve_scope → an "ambiguous_match" on the scope decision). Same placement
    # and the same reason: this must beat the rigid REQUIRED OUTPUT FORMAT
    # directive each template ends with, which otherwise pulls the model back
    # into producing one tidy single-document answer. Placed FIRST of the three
    # so it frames the whole answer — how many answers there are is a more
    # fundamental constraint than what any one of them says.
    _ambiguity_directive_note = (
        f"AMBIGUOUS DOCUMENT DESCRIPTION: {ambiguity_directive}\n\n"
    ) if ambiguity_directive else ""

    # Pick prompt based on the classified lawyer intent (intent_agent upstream)
    _intent_prompt_map = {
        "factual": ANSWER_PROMPT,
        "risk_assessment": ASSESSMENT_PROMPT,
        "comparison": COMPARISON_PROMPT,
        "obligation": OBLIGATION_PROMPT,
        "drafting": DRAFTING_PROMPT,
    }
    prompt_template = _intent_prompt_map.get(intent, ANSWER_PROMPT)

    # Drafting intent draws on the Precedent layer as well as the pages
    # (§ Phase 2: "Draft Mode and the Ask tab's drafting intent both switch to
    # clause-level embeddings, scoped to role-tagged precedent documents").
    #
    # Appended rather than substituted: a drafting question in the Ask tab is
    # still answered from the retrieved pages, and the precedent clauses are
    # the model material to draft FROM. Replacing the page context would drop
    # the document the lawyer is actually asking about.
    if intent == "drafting":
        try:
            from services import precedent as _prec
            _pc = _prec.search_clauses(
                _active_wiki_id(), session_id, question,
                limit=getattr(config, "DRAFT_PRECEDENT_CLAUSES", 12))
            if _pc:
                _block = "\n".join(
                    f"[PRECEDENT CLAUSE] {c['clause_type']} — {c['source_doc']}"
                    f"\n{c['text']}\n" for c in _pc)
                wiki_content = (
                    f"{wiki_content}\n\n"
                    f"--- PRECEDENT CLAUSES (drafting material from other "
                    f"documents in this corpus; cite them as precedent, never "
                    f"as terms of the document under discussion) ---\n{_block}")
                logger.info("Drafting intent: added %d precedent clause(s)", len(_pc))
        except Exception as _p_err:
            logger.warning("Precedent clauses unavailable for drafting intent: %s",
                           _p_err)
    prompt = (_ambiguity_directive_note + _unconfirmed_doc_note
              + _clause_directive_note) + prompt_template.format(
        context=wiki_content,
        question=question,
        conversation_block=conv_block,
        metadata_block=metadata_block,
        house_rules_block=house_rules_block,
    )

    usage = {}
    confidence_score = 75
    confidence_reason = "Default — could not parse confidence from reasoning block."

    # Comparison/risk/obligation answers synthesize across many more sources
    # (tables + key-differences + full reference lists) and were getting cut
    # off mid-sentence under the narrow-factual budget.
    _answer_token_budget = (
        config.MAX_TOKENS_ANSWER_BROAD
        if intent in ("comparison", "risk_assessment", "obligation")
        else config.MAX_TOKENS_ANSWER
    )

    # Reasoning-block tags. The OPEN and CLOSE patterns are kept SEPARATE and the
    # close pattern requires a real slash — a previous single pattern with an
    # OPTIONAL slash (<\s*/?\s*reasoning\s*>) also matched the opening tag, which
    # combined with regex-substituting the block out intermittently deleted the
    # entire answer (confirmed live: the model correctly wrote <reasoning>plan
    # </reasoning> + a 12k-char answer, but the strip ate the answer). Extraction
    # below is positional (split on the close tag) rather than sub-based, so it is
    # deterministic and model-agnostic — works whether the model reasons briefly,
    # at length, or (like Azure GPT-5.x) keeps its native reasoning out of the
    # content entirely.
    _REASON_OPEN_RE = re.compile(r'<\s*reasoning\s*>', re.I)
    _REASON_CLOSE_RE = re.compile(r'<\s*/\s*reasoning\s*>', re.I)
    # Optional leading list marker ("- CONFIDENCE_SCORE", "* CONFIDENCE_REASON")
    # and optional bold — the model formats these lines inconsistently. Without
    # matching the bulleted form, an un-stripped "- CONFIDENCE_SCORE" line leaked
    # into the answer AND the plain form mid-text got mistaken for a reasoning
    # boundary (confirmed live: a comparison table was stripped as "reasoning").
    _CONFIDENCE_LINE_RE = r'(?im)^[ \t]*(?:[-*]\s*)?\**CONFIDENCE[_\s]*(?:SCORE|REASON)[^\n]*\n?'
    # How much text must follow the confidence lines before we treat them as the
    # END of a reasoning preamble rather than a trailer on a finished answer.
    # Guards against amputating a short answer that happens to close with its
    # confidence lines.
    _MD_REASONING_MIN_TAIL = 200
    # The model sometimes self-organizes into a working draft followed by its
    # own "Final Answer" heading with a polished restatement — confirmed live
    # on a risk_assessment answer that shipped BOTH: a full "One-Sided /
    # Ambiguity / Missing / Inconsistencies" analysis, then a "Final Answer"
    # section covering the same ground again in different words (15.8k chars,
    # 18.8k tokens for one question). Neither the <reasoning> tag split nor
    # the CONFIDENCE-line split catches this shape, since it has no
    # confidence block at all — it's a second, unrequested self-revision.
    # Not part of the prompt contract (nothing asks the model to emit this),
    # so this is entirely a symptom to strip, not a format to support.
    _FINAL_ANSWER_HEADING_RE = re.compile(
        r'(?im)^[ \t]*(?:#{1,3}\s*)?\*{0,2}Final\s+Answer\*{0,2}\s*:?\s*$'
    )
    # The same self-organizing habit also shows up as an ad-hoc TAG — the model
    # wraps its user-facing answer in <final> or <final answer> (and sometimes
    # closes it). Nothing in any prompt asks for this, so it is purely a symptom
    # to strip; confirmed live: answers rendered to the user beginning with a
    # literal "<final>" / "<final answer>" line.
    #
    # OPEN and CLOSE are kept separate for the positional split, exactly like the
    # <reasoning> tags above. _FINAL_TAG_ANY_RE deliberately DOES allow the
    # optional slash that proved catastrophic there — but only because it is used
    # to delete the TAG TEXT ITSELF (a zero-content edit), never to substitute out
    # the region between two tags. That distinction is what makes it safe: it
    # cannot remove answer content no matter which tags are present or missing.
    _FINAL_TAG_OPEN_RE = re.compile(r'<\s*final(?:[ \t]+answer)?\s*>', re.I)
    _FINAL_TAG_CLOSE_RE = re.compile(r'<\s*/\s*final(?:[ \t]+answer)?\s*>', re.I)
    _FINAL_TAG_ANY_RE = re.compile(r'<\s*/?\s*final(?:[ \t]+answer)?\s*>', re.I)
    # Floor for accepting the inside of a <final> block as the whole answer.
    # Below this the block is more likely a stray/mid-sentence tag than a real
    # wrapper, so the text is left alone and only the tag characters come off.
    _FINAL_TAG_MIN_INNER = 50

    def _run_generation_pass(gen_prompt: str, token_budget: int = None,
                             reasoning_effort: str = None) -> tuple[str, dict, int, str]:
        """Run one LLM answer-generation call and parse out answer text + confidence.

        Factored out of the main body so the corrective citation retry below can
        reuse the exact same parsing/fallback logic as the primary pass.

        reasoning_effort escalates the model's effort for this call (Azure
        reasoning models only) — used when a low-effort pass returned hidden
        reasoning but no visible answer.
        """
        pass_usage: dict = {}
        pass_score = 75
        pass_reason = "Default — could not parse confidence from reasoning block."
        try:
            raw_answer, pass_usage = llm.ask(
                gen_prompt,
                pipeline="wiki",
                max_tokens=token_budget or _answer_token_budget,
                reasoning_effort=reasoning_effort,
            )

            # --- Positional split: reasoning block vs. user-facing answer ---
            # Contract: an optional <reasoning>…CONFIDENCE_SCORE…CONFIDENCE_REASON…
            # </reasoning> block, then the answer. We locate the opening tag and
            # its matching (real-slash) closing tag by POSITION:
            #   reasoning_text = between the tags   (carries confidence)
            #   pass_answer    = everything AFTER the close tag
            # This is deterministic across models: brief reasoning, long reasoning,
            # no block at all, or an unclosed block are each handled explicitly —
            # and it never substitutes the block out, so it cannot eat the answer.
            open_m = _REASON_OPEN_RE.search(raw_answer)
            close_m = _REASON_CLOSE_RE.search(raw_answer, open_m.end()) if open_m else None
            if open_m and close_m:
                reasoning_text = raw_answer[open_m.end():close_m.start()]
                pass_answer = raw_answer[close_m.end():].strip()
            elif open_m:
                # Opened but never closed — the whole tail is reasoning; the model
                # left no separate answer section (recovered below).
                reasoning_text = raw_answer[open_m.end():]
                pass_answer = ""
            else:
                # No reasoning TAG. Two very different cases hide here:
                #
                #  (a) The model answered directly, native reasoning never
                #      entering the content (e.g. Azure GPT-5.x) — nothing to strip.
                #  (b) The model honoured the block's SHAPE but emitted it as a
                #      markdown preamble instead of a tag ("## Reasoning … ##
                #      Confidence … CONFIDENCE_SCORE: 92" then the answer). The tag
                #      search misses this entirely, so the planning prose AND the raw
                #      CONFIDENCE_SCORE line rendered straight to the user (confirmed
                #      live on an obligations answer).
                #
                # Split on the confidence lines themselves — the one part of the
                # contract these models still emit verbatim.
                reasoning_text = ""
                pass_answer = raw_answer.strip()
                _conf_lines = list(re.finditer(_CONFIDENCE_LINE_RE, raw_answer))
                if _conf_lines:
                    _tail = raw_answer[_conf_lines[-1].end():].strip()
                    _head = raw_answer[:_conf_lines[-1].start()].strip()
                    # A genuine reasoning PREAMBLE is a short plan that precedes a
                    # longer answer — so only treat the pre-confidence content as
                    # strippable reasoning when the tail (candidate answer) is at
                    # least as long as the head. The restructured comparison/
                    # obligation/assessment prompts now put the ANSWER first and
                    # the confidence lines at the end, so the model routinely emits
                    # a big body, then CONFIDENCE, then a short refs/notes tail;
                    # the old "any substantial tail → head is reasoning" rule then
                    # amputated the entire body (confirmed live: a 6509-char
                    # comparison table was discarded, leaving only the confidence
                    # lines + references). Requiring tail >= head keeps the body.
                    if len(_tail) >= _MD_REASONING_MIN_TAIL and len(_tail) >= len(_head):
                        # Substantial content AFTER the confidence lines and the
                        # head is shorter → they close a reasoning preamble; the
                        # answer is the tail.
                        reasoning_text = raw_answer[:_conf_lines[-1].end()]
                        pass_answer = _tail
                        logger.info(
                            "Reasoning preamble emitted without <reasoning> tag — "
                            "split on confidence lines (%d chars of reasoning stripped)",
                            len(reasoning_text),
                        )
                    else:
                        # Confidence lines trail a finished answer (or nothing
                        # substantial follows). Keep the answer, drop the lines
                        # in place — never treat the answer itself as reasoning.
                        reasoning_text = raw_answer[_conf_lines[0].start():_conf_lines[-1].end()]
                        pass_answer = re.sub(_CONFIDENCE_LINE_RE, '', raw_answer).strip()

            if reasoning_text:
                score_match = re.search(r'(?i)CONFIDENCE[_\s]*SCORE[^0-9]*(\d+)', reasoning_text)
                reason_match = re.search(
                    r'(?i)CONFIDENCE[_\s]*REASON[^\w]*(.+?)(?:\n|$)', reasoning_text
                )
                if score_match:
                    try:
                        pass_score = min(100, max(0, int(score_match.group(1))))
                    except ValueError:
                        pass
                if reason_match:
                    pass_reason = reason_match.group(1).strip()

            # Recovery: the model put the whole answer INSIDE the reasoning block
            # (nothing after the close tag). Recover the full reasoning body minus
            # the confidence lines. Only drop a SHORT leading plan preamble if a
            # clear content heading sits near the very start — never trim into the
            # bulk of the content (the previous heuristic could discard ~90%).
            if len(pass_answer) <= 10 and reasoning_text.strip():
                recovered = re.sub(_CONFIDENCE_LINE_RE, '', reasoning_text).strip()
                cs = re.search(r'(?m)^\s*(#{1,3}\s|\*\*[A-Z][^\n]*\*\*\s*$|\|)', recovered)
                if cs and 0 < cs.start() < len(recovered) * 0.3:
                    recovered = recovered[cs.start():].strip()
                if len(recovered) > len(pass_answer):
                    logger.warning(
                        "Answer empty after reasoning split — recovered %d chars from reasoning block",
                        len(recovered),
                    )
                    pass_answer = recovered

            # Always strip any stray CONFIDENCE lines that leaked into the answer
            pass_answer = re.sub(_CONFIDENCE_LINE_RE, '', pass_answer).strip()

            # Unwrap an ad-hoc <final>…</final> answer tag. Positional, mirroring
            # the <reasoning> split: content after the open tag (bounded by the
            # close tag when one exists) becomes the answer, so any draft the
            # model left BEFORE the tag is dropped along with the tag itself.
            # Guarded by a length floor — if the inside is too short to be a real
            # answer the text is kept as-is and only the tag characters come off
            # in the sweep below, so this can never amputate content.
            _ft_open = _FINAL_TAG_OPEN_RE.search(pass_answer)
            if _ft_open:
                _ft_close = _FINAL_TAG_CLOSE_RE.search(pass_answer, _ft_open.end())
                _inner = (
                    pass_answer[_ft_open.end():_ft_close.start()] if _ft_close
                    else pass_answer[_ft_open.end():]
                ).strip()
                if len(_inner) >= _FINAL_TAG_MIN_INNER:
                    if _ft_open.start() > 0:
                        logger.warning(
                            "Answer wrapped in a <final> tag after %d chars of draft — "
                            "kept %d chars from inside the tag",
                            _ft_open.start(), len(_inner),
                        )
                    pass_answer = _inner
            # Sweep any remaining <final>/</final> tags. Removes only the tag
            # characters, never content between them (see _FINAL_TAG_ANY_RE note).
            pass_answer = _FINAL_TAG_ANY_RE.sub('', pass_answer).strip()

            # Same family, but for tags no prompt names — </confidence>,
            # </reasoning>, "<Service Agreement 2 termination rights>". Runs
            # after the fixed-vocabulary handling above so those keep their
            # positional-split semantics; this only cleans up what is left.
            _before_pseudo = pass_answer
            pass_answer = _strip_pseudo_tags(pass_answer)
            if pass_answer != _before_pseudo:
                logger.info("Stripped invented pseudo-tag(s) from answer (%d -> %d chars)",
                            len(_before_pseudo), len(pass_answer))

            # Drop a duplicated self-revision: if a "Final Answer" heading
            # appears after a substantial chunk of prior content, the model
            # restated its own draft — keep only the polished tail, not both
            # copies. A heading in the first 20% of the text is more likely
            # an intentional section label on a short answer, not a redo.
            _fa_matches = list(_FINAL_ANSWER_HEADING_RE.finditer(pass_answer))
            if _fa_matches:
                _fa = _fa_matches[-1]
                if _fa.start() > len(pass_answer) * 0.2:
                    _tail = pass_answer[_fa.end():].strip()
                    if len(_tail) >= _MD_REASONING_MIN_TAIL:
                        logger.warning(
                            "Answer contained a duplicated 'Final Answer' self-revision — "
                            "dropped %d chars of draft, kept %d chars of final",
                            _fa.start(), len(_tail),
                        )
                        pass_answer = _tail

            # --- Fallback confidence when model skipped the reasoning block ---
            # A short "Not covered" answer means the context had nothing relevant —
            # confidence should be 0, not the generic 75 default.
            if pass_score == 75 and pass_reason.startswith("Default"):
                if len(pass_answer) < 150:
                    _not_covered = re.search(
                        r'not covered|no information|not contain|no relevant|'
                        r'does not contain|not found|not available',
                        pass_answer, flags=re.IGNORECASE
                    )
                    if _not_covered:
                        pass_score = 0
                        pass_reason = "Model found no relevant context for this question."
                    else:
                        pass_score = 50
                        pass_reason = "Very short answer; limited context."
                else:
                    # Substantial answer but no reasoning block — model answered
                    # directly (this is the NORMAL path for a model whose native
                    # reasoning stays out of the content, e.g. Azure GPT-5.x).
                    # Previously hardcoded to a flat 72 regardless of actual answer
                    # quality — that floor was silently capping otherwise-correct
                    # comparison/factual answers (confirmed live at 72% on answers
                    # that were fully grounded). Get a REAL score from the existing
                    # LLM confidence evaluator instead, so the score reflects this
                    # answer's actual support in context rather than a constant.
                    try:
                        _eval = _evaluate_confidence(question, wiki_content, pass_answer)
                        pass_score = _eval["score"]
                        pass_reason = _eval["reason"] or "Confidence evaluated (no reasoning block in generation output)."
                    except Exception as e:
                        logger.error("Fallback confidence evaluation failed: %s", e)
                        pass_score = 72
                        pass_reason = "Model answered without reasoning block; score estimated."

        except RuntimeError as e:
            pass_answer = f"⚠️ LLM error: {e}"
            pass_score = 0
            pass_reason = "LLM call failed."

        return pass_answer, pass_usage, pass_score, pass_reason

    answer, usage, confidence_score, confidence_reason = _run_generation_pass(prompt)

    # Minimum length below which an answer is not "short but real" — it's
    # effectively empty and unusable, regardless of why.
    _MIN_VIABLE_ANSWER_CHARS = 50

    # A much lower bar than _MIN_VIABLE_ANSWER_CHARS, and deliberately so — this
    # one tests for the presence of a stated answer, not its adequacy, and a
    # correct answer is often genuinely short ("18 November 2025.", "Tata Steel
    # Limited."). Measured across all 105 answers in the Q1-105 audit: every
    # legitimate short answer had a body of 17+ characters; the three broken
    # ones measured here (see _answer_lacks_body) had exactly 0. Set well below
    # the former and well above the latter — this is a "did it say ANYTHING"
    # check, not a quality bar.
    _MIN_BODY_CHARS = 8

    # Matches a "References" section header, so its own content is excluded
    # from the body-presence check below — a citation list is not an answer.
    _RX_REFERENCES_HEADER = re.compile(r'^\s*References?\s*:?\s*$', re.IGNORECASE | re.MULTILINE)

    def _answer_lacks_body(text: str) -> bool:
        """True when nothing but citations and disclosures precede References.

        A response of "References\\n[1] Document, Section" is not "short but
        real" — it never states the fact at all, it just cites where the fact
        would be. This passes every existing check: it clears
        _MIN_VIABLE_ANSWER_CHARS on citation text alone, and (for the
        corrective-retry path below) can clear the _retained_length ratio the
        same way, since a citation list can be long. Confirmed live on three
        single-fact lookups (a company name, a case number): each answer's
        entire content was its own References block, despite the correct
        document and clause being cited and the fact sitting in the retrieved
        context. None of the three ever tripped a warning, a retry, or a low
        confidence score gated on anything else, and one of them shipped
        through a citation retry the pipeline itself logged as "improved" —
        fewer unverified quotes was true only because the retry deleted the
        sentence carrying them, and length alone doesn't distinguish "a
        shorter true answer" from "the answer is gone, only citations remain."
        Bracket disclosures ([SCOPE NOTE...], [CITATION WARNING...]) are
        stripped first since they are appended regardless of outcome, same
        reasoning as intent_agent.py's _RX_BRACKET_NOTE.
        """
        body = re.sub(r'\[[A-Z][A-Z \-]{2,30}:.*?\]', '', text, flags=re.DOTALL)
        body = _RX_REFERENCES_HEADER.split(body, maxsplit=1)[0]
        return len(body.strip()) < _MIN_BODY_CHARS

    # Truncation/empty-answer retry. Two distinct failure shapes land here:
    #   (a) finish_reason == "length": a "factual"-intent question that's
    #       actually broad/cross-document (e.g. "across all JVAs in the
    #       corpus...") can exhaust the narrower MAX_TOKENS_ANSWER budget
    #       before the model ever writes real content — confirmed live: a
    #       25-page cross-document question got cut off at
    #       completion_tokens=4034 (against a 4096 cap), leaving only the
    #       model's own numbered reasoning/plan trace with no actual table
    #       ever emitted.
    #   (b) finish_reason == "stop" but the answer is near-empty: confirmed
    #       live on open-ended risk_assessment prompts ("identify one-sided
    #       clauses", "go/no-go recommendation") — the model spent its
    #       reasoning budget and then just... stopped, with a clean
    #       finish_reason that gave the original code no signal anything
    #       was wrong. This shape is invisible to the finish_reason=="length"
    #       check alone, so it shipped silently as a blank answer to the
    #       user. Both shapes get the same treatment: unlike the citation
    #       retry below, a truncated/empty response is unambiguously worse
    #       than a complete one, so this always keeps the retry — no
    #       comparison needed.
    _was_truncated = usage.get("finish_reason") == "length"
    # A references-only answer is the same failure as a char-count-empty one —
    # nothing was actually said — so it takes the identical retry path (stop-empty:
    # escalate reasoning_effort one step, see _EFFORT_ESCALATION below).
    _was_empty = len(answer.strip()) < _MIN_VIABLE_ANSWER_CHARS or _answer_lacks_body(answer)
    _at_broad_budget = _answer_token_budget >= config.MAX_TOKENS_ANSWER_BROAD

    # A same-budget retry is only worth attempting for the EMPTY case, not
    # truncation: truncation is a real content-length problem that will
    # almost certainly hit the exact same wall again at the same budget, so
    # retrying without more room just burns a call. An empty answer at
    # "stop" is a different failure — confirmed live on open-ended
    # risk_assessment prompts ("go/no-go recommendation"): gpt-5-nano spent
    # its whole reasoning allotment internally (2366 hidden reasoning tokens,
    # finish_reason=stop) and emitted ZERO visible content, twice in a row.
    # A plain same-effort re-ask hits the same wall (proven live). The lever
    # that actually addresses "reasoned then produced nothing" is
    # reasoning_effort: the primary pass runs at the global low setting to
    # save tokens, so on an empty result we escalate effort ONE step to give
    # the model room to commit to an answer. Only the empty case escalates —
    # a genuine length-truncation wants MORE output budget, not more
    # reasoning (which would make it worse).
    _EFFORT_ESCALATION = {"minimal": "low", "low": "medium", "medium": "high", "high": "high"}
    # Escalate reasoning_effort ONLY for a stop-empty (finish_reason=stop, no
    # visible output → the model gave up and needs a nudge to commit). For a
    # truncation-empty (finish_reason=length → the model spent the ENTIRE budget
    # on hidden reasoning and never reached visible output, e.g. a 61-page broad
    # question), MORE effort is exactly wrong — it reasons even harder and still
    # emits nothing (confirmed live: a "from all service agreements" answer went
    # blank this way after a medium-effort retry). Truncation wants LOW effort +
    # a bigger budget so the room goes to output, not reasoning.
    _stop_empty = _was_empty and not _was_truncated
    if (_was_truncated or _was_empty) and not (_was_truncated and _at_broad_budget):
        _retry_budget = config.MAX_TOKENS_ANSWER_BROAD if not _at_broad_budget else _answer_token_budget
        _retry_effort = _EFFORT_ESCALATION.get(config.AZURE_REASONING_EFFORT, "medium") if _stop_empty else None
        logger.warning(
            "Answer generation %s (%d completion tokens, %d chars) — retrying%s%s",
            "truncated" if _was_truncated else "came back empty",
            usage.get("completion_tokens", 0), len(answer),
            " with broader budget" if not _at_broad_budget else " at the same (already broad) budget",
            f", reasoning_effort escalated to '{_retry_effort}'" if _retry_effort else "",
        )
        retry_answer, retry_usage, retry_score, retry_reason = _run_generation_pass(
            prompt, token_budget=_retry_budget, reasoning_effort=_retry_effort
        )
        token_breakdown.append({
            "call": "truncation_retry",
            "model": llm.active_model(fast=False),
            "prompt_tokens": retry_usage.get("prompt_tokens", 0),
            "completion_tokens": retry_usage.get("completion_tokens", 0),
            "total_tokens": retry_usage.get("prompt_tokens", 0) + retry_usage.get("completion_tokens", 0),
            "cached_prompt_tokens": retry_usage.get("cached_prompt_tokens", 0),
        })
        _retry_ok = (retry_usage.get("finish_reason") != "length"
                     and len(retry_answer.strip()) >= _MIN_VIABLE_ANSWER_CHARS
                     and not _answer_lacks_body(retry_answer))
        if _retry_ok:
            answer, usage, confidence_score, confidence_reason = retry_answer, retry_usage, retry_score, retry_reason
        elif len(retry_answer.strip()) > len(answer.strip()):
            # Retry didn't fully resolve it, but produced more content than
            # the original empty/truncated pass — keep the better of the two
            # rather than shipping whichever came first.
            logger.warning("Retry still incomplete but longer than the original — keeping retry, trimmed to last complete unit")
            retry_answer = _truncate_to_last_complete_unit(retry_answer)
            answer, usage, confidence_score, confidence_reason = retry_answer, retry_usage, retry_score, retry_reason
        else:
            logger.warning("Retry did not improve on the original — keeping original, trimmed to last complete unit")
            answer = _truncate_to_last_complete_unit(answer)
    elif _was_truncated or _was_empty:
        # Truncated with no higher budget available — nothing left to try.
        logger.warning("Answer generation truncated at the broad token budget with no further retry — trimming to last complete unit")
        answer = _truncate_to_last_complete_unit(answer)

    # Refusal recheck: an answer that declines over context still holding the
    # question's subject matter gets one more pass. Placed before the citation
    # checks below so anything it produces is verified on exactly the same terms
    # as a first-pass answer, and gated on the replacement not being a refusal
    # itself — a second refusal is evidence the first one was right, and the
    # original is kept.
    if _is_not_covered_answer(answer) and _refusal_looks_wrong(question, wiki_content):
        logger.info("Answer refused but context covers %s — one recheck pass",
                    question[:60])
        recheck_answer, recheck_usage, recheck_score, recheck_reason = _run_generation_pass(
            prompt + _REFUSAL_RETRY_ADDENDUM
        )
        token_breakdown.append({
            "call": "refusal_recheck",
            "model": config.AZURE_OPENAI_DEPLOYMENT,
            "prompt_tokens": recheck_usage.get("prompt_tokens", 0),
            "completion_tokens": recheck_usage.get("completion_tokens", 0),
            "total_tokens": recheck_usage.get("prompt_tokens", 0) + recheck_usage.get("completion_tokens", 0),
            "cached_prompt_tokens": recheck_usage.get("cached_prompt_tokens", 0),
        })
        if (recheck_answer.strip()
                and not _is_not_covered_answer(recheck_answer)
                and recheck_usage.get("finish_reason") != "length"):
            logger.info("Refusal recheck produced a substantive answer — using it")
            answer, usage, confidence_score, confidence_reason = (
                recheck_answer, recheck_usage, recheck_score, recheck_reason)
        else:
            logger.info("Refusal recheck refused again — keeping the original answer")

    # Strip quote-wrapped placeholder stand-ins ("Not provided in excerpt", "(not
    # provided here)", etc.) from reference lines BEFORE the integrity checks — they
    # are a known nano non-compliance the prompts already forbid, and stripping them
    # here both cleans the output and stops them from tripping the citation check
    # (which correctly flags "Not provided in excerpt" as a non-verbatim quote).
    answer = _strip_placeholder_quotes(answer)
    answer = _append_milestone_total(answer, wiki_content, question)

    # Deterministic citation-integrity checks: flag any quoted span the model
    # presented as verbatim that doesn't actually appear in the retrieved
    # context (paraphrase dressed up as an exact quote), and any quote
    # attributed to the wrong document.
    # Collapse a long list that states the same clause more than once, before
    # the citation checks run over it — see _dedupe_numbered_list.
    answer, _list_dupes = _dedupe_numbered_list(answer)
    if _list_dupes:
        answer += (
            f"\n\n[LIST NOTE: {_list_dupes} entr(ies) above repeated a clause already "
            f"listed and were merged into it, so the numbering reflects distinct "
            f"provisions rather than the number of times each was mentioned.]"
        )

    _unverified_quotes = _verify_answer_citations(answer, wiki_content, question)
    _misattributed = _verify_citation_attribution(answer, wiki_content)
    _unverified_ids = _verify_identifier_claims(answer, wiki_content)

    # Corrective retry: give the model one chance to fix flagged quotes — either
    # match them verbatim to context or drop the quotation marks — instead of
    # just warning the user after the fact. Only retried once; if the retry
    # doesn't measurably improve things, the original answer is kept and a
    # warning is appended as before.
    if _unverified_quotes or _misattributed or _unverified_ids:
        _flagged = list(_unverified_quotes) + list(_misattributed)
        _flag_lines = []
        for f in _flagged[:6]:
            _span = _nearest_verbatim_span(f, wiki_content)
            if _span:
                _flag_lines.append(f'- Flagged (not verbatim): "{f[:180]}"\n'
                                   f'  Exact text in CONTEXT to copy: "{_span[:240]}"')
            else:
                _flag_lines.append(f'- Flagged (no verbatim match in CONTEXT — remove the '
                                   f'quotation marks and state it plainly): "{f[:180]}"')
        for code in _unverified_ids[:4]:
            _flag_lines.append(
                f'- Flagged identifier (not found anywhere in CONTEXT — this is a fabricated '
                f'value, do NOT restate it in any form): "{code}". State plainly that this '
                f'document does not state this field, instead of giving any value for it.'
            )
        _retry_prompt = prompt + _CITATION_RETRY_ADDENDUM.format(flagged="\n".join(_flag_lines))
        retry_answer, retry_usage, retry_score, retry_reason = _run_generation_pass(_retry_prompt)
        retry_unverified = _verify_answer_citations(retry_answer, wiki_content, question)
        retry_misattributed = _verify_citation_attribution(retry_answer, wiki_content)
        retry_unverified_ids = _verify_identifier_claims(retry_answer, wiki_content)

        token_breakdown.append({
            "call": "citation_retry",
            "model": llm.active_model(fast=False),
            "prompt_tokens": retry_usage.get("prompt_tokens", 0),
            "completion_tokens": retry_usage.get("completion_tokens", 0),
            "total_tokens": retry_usage.get("prompt_tokens", 0) + retry_usage.get("completion_tokens", 0),
            "cached_prompt_tokens": retry_usage.get("cached_prompt_tokens", 0),
        })

        _fewer_issues = (len(retry_unverified) + len(retry_misattributed) + len(retry_unverified_ids)
                          < len(_unverified_quotes) + len(_misattributed) + len(_unverified_ids))
        # A citation fix must not gut the answer. The retry sometimes comes back
        # far shorter — e.g. only the reasoning plan, or a truncated table — which
        # trivially has "fewer" unverified quotes simply because it has fewer
        # quotes (or none). Confirmed live: a 9,974-char comparison answer was
        # replaced by an 818-char plan-only answer that "passed" this check.
        # Require the retry to retain most of the original's length to count.
        _retained_length = len(retry_answer) >= 0.6 * len(answer)
        # Length alone isn't enough: a References block is itself long, so a
        # retry that deletes the answer sentence and keeps only citations can
        # clear 60% of the original's length while removing the fact the
        # question asked for — the one thing that legitimately has "fewer
        # unverified quotes" (there are no quotes left to flag). Confirmed
        # live: this exact shape logged as "Citation retry improved answer:
        # 1->0 unverified quotes" while shipping a bare citation list with no
        # answer. A retry this hollow is worse than the flagged original, which
        # at least stated the fact even if one quote in it couldn't be verified.
        _retry_has_body = not _answer_lacks_body(retry_answer)
        if _fewer_issues and _retained_length and _retry_has_body:
            logger.info(
                "Citation retry improved answer: %d->%d unverified quotes, %d->%d misattributed, "
                "%d->%d unverified identifiers",
                len(_unverified_quotes), len(retry_unverified),
                len(_misattributed), len(retry_misattributed),
                len(_unverified_ids), len(retry_unverified_ids),
            )
            answer, usage, confidence_score, confidence_reason = retry_answer, retry_usage, retry_score, retry_reason
            _unverified_quotes, _misattributed, _unverified_ids = retry_unverified, retry_misattributed, retry_unverified_ids
        elif _fewer_issues and not _retained_length:
            logger.info(
                "Citation retry had fewer issues but was drastically shorter "
                "(%d vs %d chars) — keeping fuller original answer",
                len(retry_answer), len(answer),
            )
        elif _fewer_issues and not _retry_has_body:
            logger.info(
                "Citation retry had fewer issues but deleted the stated answer, "
                "leaving only citations — keeping flagged original instead"
            )
        else:
            logger.info("Citation retry did not improve verification — keeping original answer")

        # Unlike unverified quotes/misattribution (banner-only, original text
        # kept visible), a fabricated identifier that survives the retry is
        # hard-stripped from the answer text itself — see
        # _strip_fabricated_identifiers's docstring for why a banner alone
        # isn't enough for this specific failure shape. This runs regardless
        # of which branch above fired, since the retry may not have touched
        # every flagged code even when it measurably helped elsewhere.
        if _unverified_ids:
            answer = _strip_fabricated_identifiers(answer, _unverified_ids)

    if _misattributed:
        # The corrective retry above already had its shot at fixing this via a
        # full regeneration; for whatever survives, try a deterministic,
        # in-place label fix before falling back to a warning — the correct
        # document is already known (it's how the mismatch was detected), so a
        # regeneration isn't needed to fix a same-type wrong-number citation.
        # Runs BEFORE any [WARNING] banner is appended below — it's a genuine
        # in-place fix to the answer's own citation text, not a banner, so it
        # must land before the reference-extraction snapshot two blocks down.
        answer, _n_fixed = _autocorrect_citation_attribution(answer, wiki_content)
        if _n_fixed:
            _misattributed = _verify_citation_attribution(answer, wiki_content)
            logger.info("Citation autocorrect: fixed %d misattributed citation(s) in place, %d remaining",
                       _n_fixed, len(_misattributed))

    # Snapshot the answer's own [N]/[From: ...] citation markers BEFORE any
    # [CITATION WARNING]/[ATTRIBUTION WARNING]/[SCOPE WARNING]/[SCOPE NOTE]
    # banner gets appended below. Those banners are themselves bracket-shaped
    # text (and routinely mention document names/quotes while explaining what
    # went wrong), so extracting references AFTER they land let the banner's
    # own prose get scanned as if it were a citation and substring-matched
    # against document identifiers — confirmed live: a comparison's References
    # section rendered "Tata Brand Judgment 2" though the answer never once
    # discussed that document, because the warning text below happened to
    # contain a fragment that matched it. This is a pure reorder — every
    # banner block after this point is unchanged, only moved past this line.
    referenced = re.findall(r"\[([^\]]+)\]", answer)

    if _unverified_quotes:
        # Report at true severity. Text found nowhere in the retrieved context is
        # a possible fabrication and keeps the hard warning; text that IS in the
        # context but outside a verified quote block is a real passage with an
        # overstated provenance claim, which gets a note instead. Splitting these
        # is what makes the hard warning meaningful again — it previously fired
        # on 55.8% of answers, the overwhelming majority for the benign reason.
        _absent_quotes, _prose_quotes = _split_unverified_by_severity(
            _unverified_quotes, wiki_content, question
        )
        logger.warning(
            "Citation-integrity check: %d quoted span(s) unverified (%d absent from "
            "context, %d sourced from page prose outside a verified quote block)",
            len(_unverified_quotes), len(_absent_quotes), len(_prose_quotes),
        )

        def _preview_of(quotes: list[str]) -> str:
            return "; ".join(
                f'"{q[:80]}..."' if len(q) > 80 else f'"{q}"' for q in quotes[:3]
            )

        if _absent_quotes:
            # Excise the unsubstantiated excerpts that sit in References lines, then
            # warn only about any still present in the answer's prose — those cannot
            # be removed without rewriting the sentence that carries them.
            answer, _dropped = _drop_unverifiable_reference_quotes(answer, _absent_quotes)
            _answer_norm = _norm_for_match(answer)
            _still_present = [q for q in _absent_quotes
                              if _norm_for_match(q) and _norm_for_match(q) in _answer_norm]
            if _dropped:
                logger.warning("Removed %d unverifiable quote(s) from References line(s)", _dropped)
                answer += (
                    f"\n\n[CITATION NOTE: {_dropped} reference(s) above quoted wording that is not in the "
                    f"document. The quote was removed; the document and clause it points to are "
                    f"unchanged.]"
                )
            if _still_present:
                answer += (
                    f"\n\n[CITATION WARNING: {len(_still_present)} quoted passage(s) above are not in the "
                    f"document — check them before relying on the wording: {_preview_of(_still_present)}]"
                )
        if _prose_quotes:
            answer += (
                f"\n\n[CITATION NOTE: {len(_prose_quotes)} passage(s) above say what the document says "
                f"but not in its exact words — read them as paraphrase: {_preview_of(_prose_quotes)}]"
            )

    if _misattributed:
        logger.warning("Citation-attribution check: %d quote(s) attributed to the wrong document: %s",
                        len(_misattributed), _misattributed)
        answer += (
            f"\n\n[ATTRIBUTION WARNING: {len(_misattributed)} quote(s) above look attributed "
            f"to the wrong document — {'; '.join(_misattributed[:2])}]"
        )

    if _unverified_ids:
        # The fabricated code itself was already hard-stripped from the answer
        # body above (see _strip_fabricated_identifiers) — this banner explains
        # WHY a field now reads "[not stated in this document]" instead of
        # silently leaving that placeholder unexplained.
        logger.warning("Identifier-fabrication check: %d identifier value(s) not found in context, stripped: %s",
                        len(_unverified_ids), _unverified_ids)
        answer += (
            f"\n\n[IDENTIFIER WARNING: {len(_unverified_ids)} field value(s) the answer initially "
            f"stated (e.g. a matter reference or case/docket number) could not be found anywhere "
            f"in this document's retrieved content and have been removed — this document does not "
            f"appear to state that field.]"
        )

    # --- Named-document substitution check ---
    # A document whose ingest produced NO pages (e.g. a scanned PDF whose OCR
    # failed) is absent from the page index entirely — so it cannot be detected
    # there, and retrieval never sees it. But a same-numbered sibling
    # ("Test_SA_01" ← "service agreement 1": both normalise to a name carrying
    # "service" + a 1) matches the same type+number, IS populated, and silently
    # supplies every word of the answer. The result reads as authoritative about
    # a document the user never asked about (confirmed live: "service agreement
    # 1" answered entirely from Test_SA_01's unrelated Helios/Zephyr Delaware
    # MSA at 92% confidence, with nothing indicating the real Service Agreement 1
    # was empty).
    #
    # Detect by matching the question against the UPLOAD list — the only record
    # that an empty document exists — and flagging any named upload the page
    # index doesn't know. Deterministic banner rather than a prompt rule: the
    # model cannot see which of its sources was a substitution.
    _empty_named: list[str] = []
    try:
        _named_uploads = _numbered_docs_in(question, _uploaded_doc_names(session_id))
        # Only a MIX matters. All-empty → no content either way, and the model
        # says "not covered" on its own; all-populated → nothing was substituted.
        if len(_named_uploads) > 1:
            _indexed_norms = {_norm_doc_name(d) for d in _distinct_source_docs(pages)}
            _populated = [u for u in _named_uploads if _norm_doc_name(u) in _indexed_norms]
            _empty = [u for u in _named_uploads if _norm_doc_name(u) not in _indexed_norms]
            if _populated and _empty:
                _empty_named = _empty
    except Exception as e:
        logger.error("Named-document substitution check failed: %s", e)

    if _empty_named:
        logger.warning(
            "Named-document substitution: %d matched document(s) have no ingested "
            "pages; answer sourced from same-numbered sibling(s): %s",
            len(_empty_named), [_norm_doc_name(d) for d in _empty_named],
        )
        _names = "; ".join(_norm_doc_name(d) for d in _empty_named[:3])
        answer += (
            f"\n\n[SCOPE WARNING: {len(_empty_named)} document(s) matching your question have "
            f"no readable text and contributed nothing to this answer — {_names}. What you "
            f"see above was drawn from other document(s) that matched the same name/number. "
            f"Check the References section names the document you actually meant.]"
        )

    # The question named a counterparty that couldn't be pinned to one document —
    # the answer came from a broad search that may have surfaced a sibling of the
    # same type. Warn deterministically (the model can't see that its source was a
    # best-guess rather than the document the user meant).
    if scope_warning:
        answer += f"\n\n[SCOPE WARNING: {scope_warning}]"

    # Scope was inferred rather than stated by the question — say so, always.
    if scope_note:
        answer += f"\n\n[SCOPE NOTE: {scope_note}]"

    # Map each selected page to its real document identifier — the SOURCE_DOC
    # filename (e.g. "...Legal Opinions (1)_Legal Opinion 6 (1).pdf"), not the
    # page title's trailing parenthetical. That parenthetical is, for
    # essentially every page in this corpus, just the bare instrument TYPE
    # ("NDA", "Legal Opinion", "Shareholders' Agreement") — the title format is
    # "{Topic} – {DocID} ({DocType})", so the type label sits in the SAME
    # position for every document regardless of which one it is. Matching
    # citation text against these type-only strings meant a fully-qualified
    # citation like "Legal Opinion 6 (1)" would substring-match the generic
    # "Legal Opinion" from some UNRELATED page's title too, and — since the old
    # loop appended every substring hit rather than the most specific one —
    # that generic, unnumbered entry could ride along into files_used and
    # render as its own bogus reference card ("1. Legal Opinion") alongside or
    # instead of the real, numbered document. A hand-maintained stoplist of
    # generic type-words can never fully close this (there are dozens of
    # instrument types in a legal corpus); source_doc is the actual per-file
    # identifier and is immune to the collision because two distinct documents
    # never share the same filename.
    canonical_files: dict[str, str] = {}  # cleaned display name -> raw source_doc
    # Pages are also titled with a short SYNTHESIZED identifier — "Dynamic
    # Injunction Framework – LO-Tata – Legal Opinion" carries "LO-Tata" — and the
    # model frequently cites bracket-style using THAT identifier ("[LO-Tata –
    # Dynamic Injunction Framework; ...]") instead of the real filename shown in
    # the context's own "[From: ...]" label right below the title (confirmed
    # live: this left files_used empty for two answers whose bracket citations
    # never matched canonical_files at all, silently falling through to the
    # arbitrary "first 3 selected pages" last resort below — which rendered
    # NDA 7 as the References list for an answer that never once cited NDA 7).
    # Index those identifiers too, using the same extractor _doc_identifier_part
    # already uses for entity-matching, so a "LO-Tata"-style citation resolves to
    # its real document exactly like a filename-style citation does.
    # Some pages never got a distinctive synthesized ID at ingest time and just
    # repeat the bare document TYPE as their "identifier" instead — e.g. a title
    # like "Assignment Clause Recommendation – Legal Opinion (Legal Opinion)"
    # makes _doc_identifier_part return "Legal Opinion", not a real per-document
    # code. That generic word appears on MANY unrelated documents, so indexing
    # it would make ANY citation mentioning "Legal Opinion" (or "Court
    # Judgment", etc.) false-match whichever one of those documents happened to
    # be written into the dict last (confirmed live: a compound citation
    # bracket containing the bare words "Legal Opinion" and "Court Judgment"
    # pulled in two topically unrelated documents — a Sentinel IP work-for-hire
    # opinion and a Delaware trade-secrets judgment — neither actually cited).
    # A generic type word can never distinguish one document from another, so
    # exclude it from the identifier index entirely rather than let it resolve
    # to an arbitrary single document.
    _GENERIC_TYPE_IDENTIFIERS = {
        "nda", "service agreement", "master services agreement", "msa",
        "shareholders agreement", "shareholder agreement", "sha",
        "joint venture agreement", "jva", "court judgment", "legal opinion",
        "legal opinions", "judgment", "petition", "affidavit", "arbitration notice",
        "complaint", "lease", "licence", "license", "deed", "memorandum",
        "settlement", "agreement", "contract",
    }
    # Collect identifier -> ALL documents claiming it, then keep only the ones a
    # single document owns. Assigning straight into a dict silently kept whichever
    # document was written LAST, so a non-distinctive identifier (a bare party name
    # like "Tata", or a case name cited by every judgment that discusses it) mapped
    # to an arbitrary document and dragged it into files_used for any answer whose
    # citation happened to contain that word. Measured on the live corpus: 63 of 466
    # identifiers were claimed by more than one document, "aether-helios" by ten.
    _ident_claims: dict[str, set[str]] = {}
    for title, page in pages.items():
        if not isinstance(page, dict):
            continue
        sd = page.get("source_doc", "")
        if not sd:
            continue
        clean = re.sub(r'^[a-f0-9-]{36}_', '', sd.replace("\\", "/").rsplit("/", 1)[-1])
        clean = os.path.splitext(clean)[0]
        canonical_files[clean] = sd
        ident = _doc_identifier_part(title)
        if len(ident) >= 4 and ident.lower() not in _GENERIC_TYPE_IDENTIFIERS:
            _ident_claims.setdefault(ident.lower(), set()).add(sd)
    # short doc identifier -> raw source_doc, distinctive identifiers only
    identifier_files: dict[str, str] = {
        ident: next(iter(docs)) for ident, docs in _ident_claims.items()
        if len(docs) == 1
    }

    pages_used_dedup = []
    files_used = []
    seen = set()

    for t in referenced:
        if t not in seen:
            pages_used_dedup.append(t)
            seen.add(t)
            # Try to figure out which file this refers to. Checked in both
            # directions since citation text conventions vary — the model may
            # write the bare identifier ("Legal Opinion 6", checked via t-in-
            # clean_name since clean_name carries extra folder-path noise) or
            # something closer to the full cleaned name (checked the other
            # way). The reverse direction (t-in-clean_name) is gated to
            # citation text of at least 6 chars: clean_name is a long,
            # noise-bearing string (folder segments, the "(1)" copy suffix),
            # so an unguarded reverse check let a short/generic bracket marker
            # (e.g. a bare "[1]") trivially match almost every document in the
            # corpus — confirmed live: this produced a 494-file "files_used"
            # list (essentially the entire corpus) instead of the 1-3 real
            # citations. The forward direction has no such risk: clean_name
            # itself is never short (source_doc filenames always carry a doc
            # type + number), so it can't spuriously match unrelated text.
            t_norm = t.lower().strip()
            for clean_name, sd in canonical_files.items():
                cn_norm = clean_name.lower()
                if cn_norm in t_norm or (len(t_norm) >= 6 and t_norm in cn_norm):
                    if sd not in files_used:
                        files_used.append(sd)
            # Forward-only, same as canonical_files above: the identifier
            # ("LO-Tata") is short and distinctive by construction (>=4 chars,
            # ingest-synthesized per document), so checking it appears IN the
            # citation text is safe — the reverse direction isn't needed since
            # citation text is never shorter than a 4-char identifier in a way
            # that would matter.
            for ident, sd in identifier_files.items():
                if sd not in files_used and _identifier_in_citation(ident, t_norm):
                    files_used.append(sd)

    # The loop above resolves PAGE citations. An answer's References block names
    # source FILES, so a document cited only there never reached files_used —
    # and the fallback below cannot rescue it, because the fallback only runs
    # when files_used is empty.
    #
    # Confirmed live: a cross-document comparison quoted the Consultancy
    # Agreement verbatim and listed it under References with the quote, while
    # files_used carried only the three NDAs scope had resolved. The document
    # was read and cited; the provenance simply failed to record it. That
    # understates the sources beneath an answer a lawyer is being asked to
    # rely on, which is worse than it sounds — the document list is how they
    # check the answer.
    #
    # Forward direction only (clean_name in answer), the same direction the
    # loop above documents as safe: clean_name is long and specific by
    # construction, so it cannot spuriously match.
    _NAMED_IN_ANSWER_CAP = 12
    if answer:
        _answer_norm = answer.lower()
        for clean_name, sd in canonical_files.items():
            if len(files_used) >= _NAMED_IN_ANSWER_CAP:
                break
            if sd in files_used:
                continue
            cn = (clean_name or "").lower()
            if len(cn) >= 12 and cn in _answer_norm:
                files_used.append(sd)
                logger.info("files_used: %r added — named in the answer text "
                            "but not resolvable from a page citation", clean_name)

    # Fallback: if no inline citations were found, populate files_used.
    # Prefer the file(s) explicitly mentioned in the question; only fall back
    # to selected-page source docs when no file was detected — and cap that
    # last-resort list, since deduping every distinct source_doc across all
    # selected_titles produces one entry per RETRIEVED document (often
    # 15+, mostly supplementary/irrelevant context, not what the answer is
    # actually about) rather than anything the answer specifically relied on.
    # Confirmed live: a short "not covered" answer with zero [N] citations and
    # no filename match rendered 15 unrelated SHA/Judgment/Service-Agreement
    # references under a single-document question.
    _FILES_USED_FALLBACK_CAP = 3
    if not files_used and selected_titles:
        mentioned = _detect_mentioned_files(question, pages)
        if not mentioned:
            # The question may not name the document explicitly (e.g. a rephrased,
            # more generic version of a question that previously named it), but the
            # ANSWER itself frequently does — especially when the model's own
            # citation didn't use a bracketed [N] marker at all, so the referenced-
            # bracket scan above never had a chance to resolve it via inline
            # citations. Confirmed live: a rephrased SA2 question (no "(SA 2)")
            # with a bolded-but-unbracketed citation fell all the way to the
            # arbitrary "first 3 selected pages" last resort below and rendered
            # unrelated documents, even though the answer text plainly named
            # "Service Agreement 2" and its real filename throughout.
            mentioned = _detect_mentioned_files(answer, pages)
        if mentioned:
            files_used = sorted(mentioned)
        else:
            seen_files = set()
            for t in selected_titles:
                if len(files_used) >= _FILES_USED_FALLBACK_CAP:
                    break
                if t.startswith("Q:"):
                    continue
                page = pages.get(t)
                if isinstance(page, dict):
                    sd = page.get("source_doc", "")
                    if sd and sd not in seen_files:
                        seen_files.add(sd)
                        files_used.append(sd)

    # Confidence is already extracted from the reasoning block above —
    # no second LLM call needed.
    if unconfirmed_doc_reference and confidence_score > 45:
        # Confidence is the model's own self-assessment of how well it answered
        # from what it was given — it has no visibility into whether the
        # document the question actually named was ever confirmed to exist.
        # Confirmed live: answers built on an unconfirmed "service agreement 1"
        # reference carried 92-96% self-reported confidence even while
        # grounding (a separate, independent check) read as low as 20-55% —
        # the two signals were never reconciled. Cap it deterministically here
        # rather than trust the model to discount its own score for a fact it
        # can't see; 45 sits below the >=80 threshold that gates auto-caching
        # an answer as a trusted "Q:" page below, so an unconfirmed-reference
        # answer can never entrench itself as future ground truth either.
        confidence_reason = (
            f"Capped at 45 (was {confidence_score}) — the question referenced a "
            f"document this corpus could not confirm exists; the model's own "
            f"confidence in its retrieval-independent reasoning cannot offset that. "
            f"{confidence_reason}"
        )
        confidence_score = 45
    confidence = {"score": confidence_score, "reason": confidence_reason}

    not_covered = _is_not_covered_answer(answer)

    # Log query
    _log_event(session_id, "QUERY", f"Q: {question[:60]}... | BM25 Shortlist: {bm25_count} | Pages selected: {len(selected_titles)} | Confidence: {confidence['score']}")

    # Answer filing back to wiki — skip if citation checks flagged anything, even
    # after the corrective retry above. Cached "Q:" pages are treated as fully
    # trusted context for future questions (see _block_verification_text), so
    # caching a still-flagged answer would let a fabricated/misattributed quote
    # entrench itself: the next time the same question is asked, the answer LLM
    # would re-quote its own cached past self, and that quote would "verify"
    # against the cache even though the underlying content was never trustworthy.
    # Confirmed live: this exact loop happened with a paraphrased NDA_08 quote.
    if (config.ENABLE_ANSWER_CACHE and confidence["score"] >= 80
            and not _unverified_quotes and not _misattributed and not _unverified_ids):
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
        "cached_prompt_tokens": usage.get("cached_prompt_tokens", 0),
    })

    # Aggregate totals across all calls in this query
    token_total = {
        "prompt_tokens":     sum(e["prompt_tokens"]     for e in token_breakdown),
        "completion_tokens": sum(e["completion_tokens"] for e in token_breakdown),
        "total_tokens":      sum(e["total_tokens"]      for e in token_breakdown),
        # Prompt tokens the provider served from its own cache. The static rule
        # body of every answer template is byte-identical on every query, so a
        # warm cache should cover most of it; a zero here means it is not being
        # hit and the reordering that made the prefix stable is not paying off.
        "cached_prompt_tokens": sum(e.get("cached_prompt_tokens", 0) for e in token_breakdown),
    }

    # Log per-call breakdown to session log
    breakdown_str = " | ".join(
        f"{e['call']} ({e['model']}): {e['total_tokens']} tokens"
        for e in token_breakdown
    )
    _log_event(session_id, "TOKEN_USAGE", f"Total: {token_total['total_tokens']} | {breakdown_str}")

    # Last step before the answer is handed back: say "these documents", never
    # "the retrieved context". Placed after every citation and grounding check
    # so those read exactly what the model wrote, while the reader gets prose
    # that talks about their documents instead of about the search.
    # Take the contract's MISSING_ITEMS block off the answer before the voice
    # rewrite, and put it back in fixed wording after — so the section the
    # reader sees is this file's sentence, not whatever the model reached for,
    # and the items themselves survive as structured data on the payload.
    answer, _missing_items = _extract_missing_items(answer)

    answer, _voice_edits = _rewrite_answer_voice(answer)
    if _voice_edits:
        logger.info("Answer voice: %d machine-register phrase(s) rewritten", _voice_edits)

    # The items were lifted out before the rewrite above, so they never passed
    # through it. They need it most: an item exists to say something is not
    # there, which is exactly the sentence the model writes about excerpts and
    # retrieval rather than about the document. Confirmed live on both items in
    # the first run of this feature ("the provided excerpts do not include...",
    # "the agreement excerpts do not state...").
    _missing_items = [_rewrite_answer_voice(i)[0] for i in _missing_items]

    if _missing_items:
        logger.info("Missing items: %d not answered by these documents: %s",
                    len(_missing_items), _missing_items[:3])
        answer += _render_missing_items(_missing_items)

    return {
        "answer": answer,
        # What the question asked for that these documents do not answer, as a
        # list rather than a sentence somewhere in the prose — see
        # _extract_missing_items. Empty for every fast path and every older
        # answer, which is the correct reading: nothing was reported missing.
        "missing_items": _missing_items,
        "pages_used": pages_used_dedup,
        "files_used": files_used,
        "selected_titles": selected_titles,
        "relations": relations,
        "usage": usage,
        "confidence_score": confidence["score"],
        "confidence_reason": confidence["reason"],
        # Render flag: the answer leads with "the documents don't cover this",
        # so the confidence/grounding badges describe a refusal, not an answer,
        # and must not be shown as though they rated one.
        "not_covered": not_covered,
        # Counts, not a predicted score. _verify_answer_citations is a substring
        # check against the exact context the model was given, so these numbers
        # are true by construction — unlike confidence_score, which is the
        # generating model's own self-report and correlates +0.13 with an
        # independent judge (i.e. not at all). Surfaced so a reader has something
        # actionable to look at instead of a percentage that means nothing.
        # `total` counts every quoted span; quotes that merely echo a page title
        # or the question are skipped by the verifier rather than failed, so they
        # count as verified here, which matches what they are.
        "citation_check": {
            "total": len([q for q in _QUOTE_SPAN_RE.findall(answer)
                          if _is_checkable_quote(q)]),
            "unverified": len(_unverified_quotes),
            "misattributed": len(_misattributed),
            # A quote can be both unverified and misattributed, so the two
            # counts are unioned rather than added — summing them would let a
            # single bad quote fail twice and report fewer verified claims
            # than the answer actually contains.
            "verified": max(0, len([q for q in _QUOTE_SPAN_RE.findall(answer)
                                    if _is_checkable_quote(q)])
                            - len(set(_unverified_quotes) | set(_misattributed))),
        },
        "token_breakdown": token_breakdown,
        "token_total": token_total,
        # Document QA warning (§ Phase 3.5b). Names the documents this answer
        # rests on whose text could not be fully extracted, so a reader can see
        # that a confident-looking answer was written over a partially
        # unreadable source. Additive metadata — the answer text is untouched.
        "document_quality_warning": _document_quality_warning(session_id, files_used),
    }


def _document_quality_warning(session_id: str, files_used: list[str]) -> dict | None:
    """Reader-facing flag for documents whose pages failed to extract.

    Reports only what is known to be bad. A document with no page_quality rows
    — ingested before the table existed — is absent from the result rather than
    reported as clean, because "we have no quality record for this" and "this
    document is fine" are different statements and only one of them is true.

    Never raises: this is disclosure, and a failure to disclose must not also
    take down the answer it was attached to.
    """
    if not config.USE_DATABASE or not files_used:
        return None
    try:
        quality = _db.get_document_quality(_active_wiki_id(), session_id, list(files_used))
    except Exception as e:
        logger.error("Document quality lookup failed: %s", e)
        return None
    if not quality:
        return None

    docs = []
    for source_doc, q in quality.items():
        pages = q["bad_page_numbers"]
        shown = ", ".join(str(p) for p in pages[:8]) + ("…" if len(pages) > 8 else "")
        docs.append({"source_doc": source_doc, "name": _norm_doc_name(source_doc),
                     "unreadable_pages": q["unreadable_pages"],
                     "total_pages": q["pages"], "page_numbers": pages,
                     "page_list": shown})
    docs.sort(key=lambda d: -d["unreadable_pages"])

    total_bad = sum(d["unreadable_pages"] for d in docs)
    if len(docs) == 1:
        d = docs[0]
        # Fully unreadable reads very differently from partially unreadable and
        # deserves its own sentence — the answer above it was written with
        # effectively nothing from this document.
        if d["unreadable_pages"] >= d["total_pages"]:
            message = (f"None of the {d['total_pages']} page(s) of "
                       f"“{d['name']}” could be read as text. Any answer "
                       f"drawn from that document may be incomplete or wrong.")
        else:
            message = (f"{d['unreadable_pages']} of {d['total_pages']} page(s) of "
                       f"“{d['name']}” could not be reliably interpreted "
                       f"(page {d['page_list']}). Analysis of that document may be "
                       f"incomplete.")
    else:
        message = (f"{total_bad} page(s) across {len(docs)} of the documents used "
                   f"could not be reliably interpreted. Analysis may be incomplete.")
    return {"message": message, "documents": docs, "unreadable_pages": total_bad}


# ---------------------------------------------------------------------------
# Conversational UX — disambiguation & clarification
# ---------------------------------------------------------------------------

_DOC_NAME_PATTERN = re.compile(
    # Type phrase, then the number. The number may be separated from the type by
    # a filler word the corpus actually uses in filenames — "Court Case DOCUMENT
    # 4", "Joint Venture AGREEMENT 4" — or by "#", "no.", "number". Without the
    # optional filler, only "court case 4"/"joint venture 4" matched, so every
    # multi-document question naming "Court Case Document N" / "Joint Venture
    # Agreement N" silently force-included NONE of those docs (confirmed live:
    # Q52 named 3 docs, only the one written "Judgment 6" matched).
    r'(services?\s+agreement|shareholders?\s+agreement|nda|'
    r'joint\s+venture(?:\s+agreement)?|legal\s+opinion|'
    r'court\s+case(?:\s+document)?|judge?ment|jva|sha|sa)'
    r'\s*(?:#|no\.?|number)?\s*(\d+)',
    re.IGNORECASE,
)

# Distinctive core token each matched type must ALSO appear as in a filename, to
# stop a number from matching documents of the wrong type. Must NOT be the type's
# first word when that word is non-distinctive: every source_doc here is prefixed
# "Legal AI - Raja …", so "legal" (from "legal opinion") appears in EVERY filename
# and would let "Legal Opinion 7" match every number-7 document of any type
# (confirmed live: Q56 matched 9 docs). "opinion" is the distinctive token.
_DOC_TYPE_CORE = {
    "service agreement": "service", "services agreement": "service",
    "shareholder agreement": "shareholder", "shareholders agreement": "shareholder",
    "nda": "nda",
    "joint venture": "venture", "joint venture agreement": "venture",
    "legal opinion": "opinion",
    "court case": "court", "court case document": "court",
    # "judgement" (British) must map to the American "judgment" the filenames
    # actually use — otherwise type_core is the user's own spelling, which is
    # never in the filename, and the doc reads as non-existent.
    "judgment": "judgment", "judgement": "judgment",
    "jva": "venture", "sha": "shareholder", "sa": "service",
}

# Matches when a question names a document type together with a distinctive entity
# or party name (e.g. "ReVolt JV Agreement", "Meridian service agreement"). The
# entity capture is capped at 3 words — real entity names are short ("Yuvraj
# Kanther", "SolarNexus"), never a full clause. Uncapped, this pattern used to
# swallow an entire preceding sentence whenever a doc-type word appeared
# incidentally in ordinary prose (e.g. "...source code from joint venture
# servers..." has nothing to do with naming a Joint Venture Agreement document),
# which made _question_names_a_document wrongly report "yes" and skip
# disambiguation entirely.
_DOC_WITH_ENTITY_PATTERN = re.compile(
    r'(?:the\s+)?([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+'
    r'(?:jv\s+agreement|jva|joint\s+venture|services?\s+agreement|nda|'
    r'shareholders?\s+agreement|sha|court\s+case|judge?ment|legal\s+opinion)',
    re.IGNORECASE,
)

# Words that are NOT distinctive entity names — when one of these immediately
# precedes a doc type ("this NDA", "the agreement"), the reference is VAGUE and
# should trigger disambiguation, not be treated as naming a specific document.
_NON_ENTITY_WORDS = {
    "this", "that", "the", "a", "an", "any", "each", "every", "our", "their",
    "your", "its", "his", "her", "some", "no", "which", "what", "does", "do",
    "is", "are", "was", "were", "review", "summarize", "summarise", "analyze",
    "analyse", "explain", "describe", "identify", "assess", "evaluate", "draft",
    "compare", "in", "of", "for", "on", "about", "regarding", "tata", "given",
    "from", "with", "and", "or", "to", "by", "under", "between", "during", "at",
    "into", "onto", "over", "after", "before", "against", "across", "within",
    # Quantifiers over the whole corpus ("across all Service Agreements") name
    # NO specific document — confirmed live to false-trigger unconfirmed_doc_reference
    # (and the resulting 45%-confidence cap) on a genuinely broad, correctly
    # cross-document-synthesized answer.
    "all", "both", "such", "these", "those", "various", "multiple", "several", "many",
}

# Matches a VAGUE singular reference: "this NDA", "the agreement", "this document"
# (a determiner + a doc type/noun) NOT followed by a number. Used to disambiguate
# among multiple documents of the same type.
_VAGUE_DOC_PATTERN = re.compile(
    r'\b(?:this|that|the|a|an)\s+'
    r'(services?\s+agreement|shareholders?\s+agreement|nda|'
    r'non[-\s]?disclosure(?:\s+agreement)?|joint\s+venture(?:\s+agreement)?|'
    r'jva|sha|legal\s+opinion|court\s+case|judgment|'
    r'motion(?:\s+to\s+\w+)?|complaint|stipulation|affidavit|opposition|pleading|'
    r'settlement\s+agreement|'
    r'agreement|document|contract)'
    r'\b(?!\s*#?\s*\d)',
    re.IGNORECASE,
)

# Maps a vague type keyword → substring that must appear in the source-doc name.
_VAGUE_TYPE_FILTER = {
    "service agreement": "service agreement", "services agreement": "service agreement",
    "shareholder agreement": "shareholder", "shareholders agreement": "shareholder",
    "sha": "shareholder",
    "nda": "nda", "non disclosure": "nda", "non-disclosure": "nda",
    "non disclosure agreement": "nda", "non-disclosure agreement": "nda",
    "joint venture": "joint venture", "joint venture agreement": "joint venture",
    "jva": "joint venture",
    "legal opinion": "legal opinion",
    "court case": "court", "judgment": "judgment",
}
# Generic determiner+noun ("this agreement", "the document") — can't narrow by type.
_GENERIC_VAGUE_WORDS = {"agreement", "document", "contract"}

# Litigation-filing types whose raw filenames (Test_CCD_08.txt, Test_Judgment_26.txt)
# don't encode the filing type the way JVA/NDA/SHA/SA filenames do — a "motion" or
# "complaint" can only be identified by looking at each page's stored doc-type
# parenthetical ("Facts – Vanguard (Motion to Dismiss)", "Parties – Aether
# (Verified Complaint)"). These must be resolved via _source_docs_by_title_type,
# not _VAGUE_TYPE_FILTER (which matches against the raw filename).
_VAGUE_TITLE_TYPE_WORDS = {
    "motion", "complaint", "stipulation", "affidavit", "opposition", "pleading",
    "settlement agreement",
}


def _title_prefix(title: str) -> str:
    """Return the doc-type label from a page title's trailing parenthetical.

    Stored titles look like "Facts – HASG (Motion to Dismiss)" or "Notice
    Provision – Test-CCD35(Motion)" — the doc-type label is the text inside the
    LAST parenthetical group. (The comma-prefixed "Motion, Facts – ..." form is
    a display-layer relabeling get_context does at query time — it is never the
    stored title, so matching on a leading comma here always returns nothing.)
    """
    m = re.search(r'\(([^)]+)\)\s*$', title)
    return m.group(1).strip().lower() if m else ""


def _source_docs_by_title_type(pages: dict, vtype: str) -> list[str]:
    """Return distinct source_docs whose pages carry a title-prefix matching vtype.

    Matches "motion" against prefixes like "motion", "motion to dismiss", "motion
    for summary judgment" (startswith), and "complaint" against "complaint",
    "verified complaint", "amended complaint" (substring), etc.
    """
    base = vtype.split()[0]  # "motion to compel" → "motion"; "settlement agreement" stays multi-word below
    if vtype == "settlement agreement":
        base = vtype
    found: set[str] = set()
    for title, page in pages.items():
        if not isinstance(page, dict):
            continue
        prefix = _title_prefix(title)
        if base in prefix:
            sd = page.get("source_doc", "")
            if sd:
                found.add(sd)
    return list(found)


def _question_names_a_document(question: str, docs: list[str]) -> bool:
    """Return True if the question names a SPECIFIC document.

    Checks:
    1. Numbered pattern ("service agreement 1", "NDA 3", "SA1")
    2. Entity name + doc type ("ReVolt JV Agreement", "Meridian service agreement"),
       but NOT a determiner + type ("this NDA", "the agreement") — those are vague.
    """
    if _DOC_NAME_PATTERN.search(question):
        return True
    m = _DOC_WITH_ENTITY_PATTERN.search(question)
    if m:
        entity = m.group(1).strip().lower()
        last_word = entity.split()[-1] if entity else ""
        # "this NDA" / "the service agreement" → vague, not a specific document
        if last_word and last_word not in _NON_ENTITY_WORDS:
            return True
    return False


def _names_numbered_document(question: str) -> bool:
    """True only for a NUMBERED document reference ("Service Agreement 1", "NDA 3").

    Distinguishes the two ways _question_names_a_document() can fire. The numbered
    branch names a document by an identifier the corpus actually uses as a
    filename, so if no such document exists it genuinely does not exist, and the
    caller is right to say so (this is the original "Service Agreement 1 doesn't
    exist" protection). The entity branch, by contrast, matches a DESCRIPTIVE
    paraphrase of subject matter ("wastewater-dosing NDA", "subsea diagnostic
    deliverables") — a phrase that is never a literal corpus name even when the
    document is real and merely filed under a bare type+number. A confirmation
    miss there means "couldn't resolve the paraphrase", NOT "document absent", so
    the caller must NOT assert non-existence on it.
    """
    return bool(_DOC_NAME_PATTERN.search(question))


# Tokens that are doc types / generic vocabulary, NOT distinctive entity names.
_ENTITY_EXCLUDE = {
    "nda", "sha", "jva", "jv", "sa", "tata", "agreement", "agreements", "service",
    "shareholder", "shareholders", "joint", "venture", "court", "judgment",
    "judgments", "legal", "opinion", "opinions", "case", "document", "documents",
    "redacted", "test", "amendment", "summary", "final", "draft", "the", "and",
    "for", "from", "with", "this", "that", "limited", "private", "company",
    # Generic legal/procedural vocabulary that occasionally ends up standing in
    # for a proper short document identifier at ingest time (e.g. "Source Code
    # Analysis", "MSA Ownership of Work Product") — these are descriptive
    # phrases, not distinctive party/entity names, and treating them as entities
    # made ordinary questions containing these common words falsely match a
    # "known entity" and skip disambiguation.
    "source", "code", "product", "products", "ownership", "analysis", "technical",
    "evidence", "civil", "procedure", "procedural", "questionnaire", "motion",
    # Same failure mode via a different cause: a handful of ingested titles have
    # their "{Topic} – {DocID}" order swapped (e.g. "TLA-Aether – No Third-Party
    # Beneficiaries" instead of "No Third-Party Beneficiaries – TLA-Aether"),
    # so _doc_identifier_part() — which correctly trusts the title convention —
    # pulls a descriptive clause-topic phrase out as if it were the identifier.
    # Blocklisting the individual words is the same mitigation already used
    # above for other leaked generic vocabulary, not a fix to the title order
    # itself (that's an ingest-time data-quality issue, not a matching bug).
    "party", "parties", "confidential", "confidentiality", "beneficiary",
    "beneficiaries", "definition", "definitions", "information", "third",
    # Found live-testing a SolarNexus-JVA equity-split question: these leaked
    # in as if they were distinctive entity names, inflating the compound-match
    # bucket with unrelated documents (a Shareholder Agreement, an NDA, a court
    # order) that happen to share generic legal/business vocabulary with the
    # question, pushing the match count past ENTITY_MATCH_MAX_PAGES and
    # abandoning force-include entirely.
    "capital", "equity", "obligations", "power", "structure", "technologies",
    "solar", "nexus", "joint venture agreement",
    # Found live-testing "identify the top 10 legal and commercial risks in this
    # document" — a vague, no-document-named question that should always
    # disambiguate: "risk" and "commercial" leaked in as if they were distinctive
    # document identifiers (from a malformed title like "Risk Assessment –
    # Commercial (...)"), so _question_mentions_known_entity() falsely matched
    # ordinary risk-assessment vocabulary and skipped disambiguation entirely.
    "risk", "risks", "commercial",
    # Found live-testing VoltMetric/SteelCircle SHA questions: "SHA-OmniRetail –
    # Transfer Restrictions" has swapped Topic/DocID order, so these generic
    # clause-topic words leaked in as if "Transfer Restrictions" were a
    # distinctive identifier, out-competing the real single-token "voltmetric"
    # entity match. Also needed as the trigger condition for the swapped-title
    # detection in _doc_identifier_part() below (a last-segment made entirely
    # of these words is treated as a topic phrase, not a real identifier).
    "transfer", "transfers", "restriction", "restrictions", "specific",
    "applicable", "analytics", "rofo", "rofr", "offer", "refusal",
    # Found live-testing "Summarize this document in 10 bullet points..." — a
    # fully generic question with zero identifying information still skipped
    # disambiguation. Root cause: malformed titles like "Definitions – General
    # (Legal Opinion)" and "Right of First Offer – General (Shared)" use the
    # bare word "General" (or a clause-topic phrase) as their identifier segment
    # when ingest synthesis had no distinctive party name to put there, so these
    # generic words leaked into the entity set the same way as the cases above.
    # A pure frequency cap (_ENTITY_DOC_FREQ_CAP) can't catch this class — each
    # of these words happened to appear in only 1-4 documents in this corpus,
    # same as a genuine rare entity name, so raw occurrence count can't tell
    # them apart from a real distinctive name. Only a vocabulary-level exclusion
    # works here.
    "general", "liability", "provision", "provisions", "obligation",
    "statutory", "principle", "principles", "reasoning", "standard", "standards",
    "types", "relief", "injunctive", "notice", "notices", "breach", "threshold",
    "approval", "rules", "protection", "harm",
    # Found live-testing a cross-document question naming an NDA, an Arbitration
    # Notice, and a Section 9 Petition by role (no numbers given): "arbitration",
    # "contractual", "alleged", "preservation", "contract", "work", "under"
    # leaked in as entities from clause-topic identifiers. _pages_matching_
    # question_entity()'s winner-take-all scoring (keep only the highest
    # combined-hit-count group) means a document that happens to ALSO share one
    # of these generic words outscores — and completely excludes — a document
    # that only matches the genuinely distinctive party names. Confirmed live:
    # the real NDA (1 hit: "nordforge") and the real Arbitration Notice (1 hit)
    # were both dropped in favour of an unrelated Section 9 Petition that
    # scored 2 by also matching leaked "arbitration" vocabulary in its own
    # identifier — even though the question named all three documents by role.
    "arbitration", "contractual", "alleged", "preservation", "contract", "work",
    "under",
    # Found live-testing a DriveConnect/VoltMetric cross-document question:
    # "Definitions – Intellectual Property (Legal Opinion)" and "Miscellaneous
    # – Governing Law and Forum (Legal Opinion)" use a generic clause-topic
    # label as their identifier segment (same fallback-label failure mode as
    # "General" above), inflating unrelated Legal Opinions to a 4-way compound
    # match that buried the real "voltmetric" (1 hit) and "joint venture
    # agreement 5" (1 hit) matches entirely. "data" is a different cause: it's
    # a legitimate word inside a real company name ("Pinnacle Data Analytics
    # LLC"), but extracting individual constituent words from a multi-word
    # identifier leaks that word as if it were its own distinctive entity —
    # excluding it loses only the ability to match on "data" alone, not the
    # full "Pinnacle Data Analytics" name.
    "intellectual", "property", "governing", "forum", "proper", "data",
}


@lru_cache(maxsize=2048)
def _identifier_boundary_re(ident: str) -> re.Pattern:
    """Match a document identifier only as a whole token in citation text.

    A plain substring test let a SHORTER identifier match inside a longer one:
    "tata" is a substring of "sa-tata", so a citation reading
    "[From: SA-Tata - Service Agreement, Clause 8.3]" attributed the answer to
    the "Tata"-identified document as well — confirmed live, this is why an
    unrelated Judgment appeared alongside the Service Agreement in files_used.

    \\b is not sufficient here: identifiers legitimately contain hyphens, and
    "-" is a non-word character, so \\btata\\b still fires inside "sa-tata".
    The lookarounds below treat hyphens and digits as part of the token, so an
    identifier only matches when nothing identifier-like abuts it on either side.
    """
    return re.compile(r'(?<![a-z0-9-])' + re.escape(ident) + r'(?![a-z0-9-])')


def _identifier_in_citation(ident: str, citation_text_lower: str) -> bool:
    return bool(_identifier_boundary_re(ident).search(citation_text_lower))


def _doc_identifier_part(title: str) -> str:
    """Return the document-identifier portion of a page title.

    Titles look like "Topic – SA-Meridian (Service Agreement)". The identifier is
    the text AFTER ' – ' and BEFORE the trailing '(...)' — e.g. "SA-Meridian",
    "JVReVolt", "Yuvraj Kanther". Topic words (before the dash) are NOT included,
    so generic legal vocabulary like "Confidential Information" is never treated
    as an entity.

    Strips the parenthetical with a regex rather than a literal " (" search:
    some ingested titles omit the space before the paren (e.g.
    "Test-CCD35(Motion)"), and a literal-space search silently leaves the
    doc-type word ("Motion", "Source Code Escrow Agreement", ...) attached to
    the identifier — which then leaks generic words like "motion"/"source"/
    "code"/"product" into the entity set as if they were distinctive party
    names, causing ordinary questions to be misdetected as "mentions a known
    entity" and skip disambiguation/clarification.
    """
    # Use the LAST " – " (closest to the trailing parenthetical), not the first —
    # some titles have a dash-separated topic too ("Holding – Motion to Dismiss –
    # HASG (Court Judgment)"), where only "HASG" is the actual document identifier.
    dash = title.rfind(" – ")
    if dash < 0:
        return ""
    rest = title[dash + 3:]
    rest = re.sub(r'\s*\([^)]*\)\s*$', '', rest).strip()

    # Root-cause fix for the recurring "swapped title order" failure mode
    # (previously only patched one leaked word at a time via _ENTITY_EXCLUDE,
    # e.g. "party"/"confidential"/"risk"/"commercial"/"transfer"/"restrictions"):
    # some ingested titles have DocID-then-Topic order ("SHA-OmniRetail –
    # Transfer Restrictions") instead of the expected Topic-then-DocID order
    # ("Transfer Restrictions – SHA-OmniRetail"), so taking the last segment
    # grabs the generic topic phrase as if it were the identifier. Confirmed
    # live: this happened with BOTH "Transfer Restrictions" (VoltMetric case)
    # and "Information Rights" (SteelCircle case) — two different generic
    # phrases, same swap pattern — so a word-list can never fully close this;
    # it needs a structural check instead. If the last segment doesn't look
    # like a real document ID at all (no doc-type prefix, no camelCase — just
    # ordinary multi-word prose) AND the alternate segment DOES look like one,
    # use the alternate segment. A genuine multi-word party name with no
    # prefix (e.g. "Meridian Portfolio Labs LLP") also fails the ID-shape check,
    # but its alternate segment ("Recitals", a plain topic word) fails it too,
    # so the original `rest` is correctly kept in that case — this only kicks
    # in when exactly one of the two candidates looks ID-shaped.
    if not _looks_like_doc_id(rest):
        # Deliberately does NOT require first_dash != dash — the common case is
        # a title with exactly ONE " – " separator, where both point to the same
        # position; the multi-dash case ("Holding – Motion to Dismiss – HASG")
        # is still handled safely since its naive first segment ("Holding")
        # won't look like a real doc ID either, so it falls through unchanged.
        first_dash = title.find(" – ")
        if first_dash >= 0:
            first_seg = title[:first_dash].strip()
            if _looks_like_doc_id(first_seg):
                return first_seg
    return rest


_DOC_ID_SHAPE_RE = re.compile(r'^(?:sha|jva?|nda|sa|msa|ccd)(?:\d*[-]|\d+$)', re.IGNORECASE)


_COMPANY_SUFFIX_RE = re.compile(
    r'\b(?:private\s+limited|pvt\.?\s*ltd\.?|limited|ltd\.?|llp|llc|inc\.?|'
    r'corp(?:oration)?|plc|gmbh|s\.?a\.?|pte\.?\s*ltd\.?)\s*$',
    re.IGNORECASE,
)


def _looks_like_doc_id(s: str) -> bool:
    """True if a title segment looks like a real document identifier rather
    than a descriptive topic phrase — a doc-type-prefixed token ("SHA-Meridian"),
    a single camelCase word ("JVReVolt"), or a company-style name ending in a
    corporate suffix ("Acme Holdings Private Limited") — not generic multi-word
    legal/clause prose, which essentially never ends in a corporate suffix.
    """
    if not s:
        return False
    if _DOC_ID_SHAPE_RE.match(s):
        return True
    if _COMPANY_SUFFIX_RE.search(s):
        return True
    return " " not in s and bool(re.search(r'[a-z][A-Z]', s))


# A genuine party/entity name (Meridian, ReVolt) is specific to one deal, so it
# should appear in identifiers from only a handful of source documents. A token
# appearing across many distinct documents is generic vocabulary that leaked in
# via a descriptive or swapped-order title, not a distinctive entity name — cap
# it here since the hand-maintained _ENTITY_EXCLUDE stoplist can never keep up
# with every corpus. Confirmed live: on a 499-doc corpus, "Summarize this
# document" (zero real identifying information) matched "liability", "general",
# "provisions", and "obligation" as if they were known entities, silently
# skipping disambiguation on a fully ambiguous query.
_ENTITY_DOC_FREQ_CAP = 4


def _extract_doc_entities(pages: dict) -> set[str]:
    """Return the set of distinctive entity tokens drawn from document identifiers.

    "SA-Meridian" → {"meridian"}, "JVReVolt" → {"revolt"},
    "Yuvraj Kanther" → {"yuvraj kanther", "yuvraj", "kanther"}. Doc-type
    abbreviations, generic words, and tokens too common across distinct
    documents to be a real entity name are excluded.
    """
    token_docs: dict[str, set[str]] = {}
    for title, page in pages.items():
        ident = _doc_identifier_part(title)
        if not ident:
            continue
        # Strip a leading doc-type token / number prefix. Handles three forms:
        #   "SA-Meridian"  → "Meridian"   (separator)
        #   "JV3-SteelLoop"→ "SteelLoop"  (type + number + separator)
        #   "JVReVolt"     → "ReVolt"     (camelCase, no separator)
        core = re.sub(r'^(?:NDA|SHA|JVA?|SA)(?=[A-Z])', '', ident)               # camelCase
        core = re.sub(r'^(?:nda|sha|jva?|sa)\d*[-\s]+', '', core, flags=re.IGNORECASE)
        core = re.sub(r'^[\d\s-]+', '', core).strip(" -")
        # A well-formed identifier is SHORT (the ingest prompt asks for 2-4 words
        # max). Longer, sentence-like "identifiers" (e.g. a stray Questionnaire
        # page titled "... - Tata Power Solar Imposter Domains") are descriptive
        # phrases, not distinctive party names — mining them for words leaks
        # ordinary vocabulary ("solar", "domains") into the global entity set,
        # which then false-matches unrelated documents via substring containment.
        if len(core.split()) > 4:
            continue
        sd = (page.get("source_doc") or title) if isinstance(page, dict) else title
        candidates = set()
        cl = core.lower()
        if len(cl) >= 4 and cl not in _ENTITY_EXCLUDE:
            candidates.add(cl)
        for w in re.findall(r"[A-Za-z]{4,}", core):
            wl = w.lower()
            if wl not in _ENTITY_EXCLUDE:
                candidates.add(wl)
        for c in candidates:
            token_docs.setdefault(c, set()).add(sd)
    return {tok for tok, docs in token_docs.items() if len(docs) <= _ENTITY_DOC_FREQ_CAP}


def _contains_token(token: str, text: str) -> bool:
    """Word-boundary-aware substring check: True if `token` appears in `text`
    as a whole word/phrase, not merely as a run of characters inside a longer
    word. Confirmed live: a plain `token in text` check let the entity "vice"
    (extracted from a "Vice Chancellor" judicial-title identifier) match inside
    "ser‑VICE‑agreement", so ANY question mentioning "service agreement" was
    falsely treated as naming a known entity — this collision class (a short
    entity string happening to be a substring of an unrelated common word) is
    distinct from, and not fixable by, the vocabulary-stoplist approach used
    elsewhere in this file, since the token itself ("vice") is a legitimate
    entity fragment, just not present here as a standalone word.
    """
    return re.search(rf'\b{re.escape(token)}\b', text) is not None


def _question_mentions_known_entity(question: str, pages: dict) -> bool:
    """True if the question mentions a distinctive entity/party name from a
    document identifier (e.g. "ReVolt", "Meridian", "Yuvraj Kanther") OR from a
    source_doc filename ("Hyden", "Brackenpyre").

    The two sources cover different corpora. Page titles are what ingest
    SYNTHESISES and often abbreviate the party into an initialism ("HYD-LEX",
    "BRP-SOL") that no user ever types. Filenames keep the name the user
    actually asked about. Checking titles alone means a question naming the
    party by its real name matches nothing here even though the document is
    right there — confirmed live for "Hyden" (matches zero title-derived
    entities) and "Brackenpyre" (same), each forced into the LLM disambiguation
    triage on every turn instead of resolving deterministically on the first.
    """
    q = question.lower()
    return (any(_contains_token(ent, q) for ent in _extract_doc_entities(pages))
            or _question_names_corpus_doc_token(question, pages))


# Words that appear in this corpus's FILENAMES but identify nothing — folder
# names, document types, redaction/sample markers, extensions. A filename token
# is only a useful signal if it is the part that names a specific matter.
_DOC_TOKEN_STOPWORDS = frozenset({
    "legal", "service", "services", "agreement", "agreements", "contract",
    "judgment", "judgments", "judgement", "judgements", "court", "case",
    "cases", "document", "documents", "opinion", "opinions", "shareholder",
    "shareholders", "joint", "venture", "ventures", "confidentiality",
    "disclosure", "master", "statement", "work", "data", "processing",
    "redacted", "redact", "sample", "samples", "test", "final", "draft",
    "copy", "docx", "doc", "pdf", "txt", "file", "files", "version",
    "brand", "tool", "group", "type", "name", "misc", "other", "new", "old",
})


def _corpus_doc_name_tokens(pages: dict) -> set[str]:
    """Distinctive word tokens drawn from this corpus's source_doc filenames.

    Complements _extract_doc_entities, which mines PAGE TITLES. Ingest often
    abbreviates a counterparty in the title it synthesises ("HYD-LEX") while the
    filename keeps the name the user actually types ("Hyden-Lexus"), so a
    question naming that party matches nothing in the title-derived entity set.
    Confirmed live: "Hyden" appears in no page title in this corpus — the entity
    set holds "hyd-lex" — so every entity-based resolver returns empty for a
    question about Hyden Tech, even though retrieval finds the documents easily.
    """
    tokens: set[str] = set()
    for page in pages.values():
        if not isinstance(page, dict):
            continue
        sd = page.get("source_doc", "")
        if not sd:
            continue
        name = re.sub(r'^[a-f0-9-]{36}_', '', sd)
        name = re.sub(r'\.[A-Za-z0-9]{2,5}$', '', name)
        for tok in re.split(r'[^A-Za-z]+', name):
            t = tok.lower()
            if len(t) >= 4 and t not in _DOC_TOKEN_STOPWORDS:
                tokens.add(t)
    return tokens


def _question_names_corpus_doc_token(question: str, pages: dict) -> bool:
    """True if the question names a matter/party token from a document filename.

    Proper-noun-aware for the same reason _question_names_distinctive_entity is:
    a lowercase common word that happens to sit in a filename must not count.
    """
    for tok in _corpus_doc_name_tokens(pages):
        if _appears_as_proper_noun(tok, question):
            return True
    return False


def _appears_as_proper_noun(token: str, question: str) -> bool:
    """True if `token` appears in the ORIGINAL-case question capitalised as a
    deliberate proper noun — not as a lowercase common word, and not as a
    grammatically-forced capital at a sentence start.

    A distinctive party/entity name the user types to identify a document
    ("ReVolt", "Meridian", "SteelLoop") is a proper noun and is written
    capitalised; generic clause vocabulary that leaked into the entity set from a
    malformed/swapped-order title ("termination", "liability", "confidentiality")
    appears lowercase inside the question's prose ("… term, termination,
    liability …"). Checking case lets the disambiguation gate tell a real entity
    mention from an incidental keyword collision WITHOUT extending the
    hand-maintained _ENTITY_EXCLUDE stoplist, which by its own admission "can
    never keep up with every corpus".
    """
    for m in re.finditer(rf'\b{re.escape(token)}\b', question, re.IGNORECASE):
        start = m.start()
        if not question[start].isupper():
            continue
        prefix = question[:start].rstrip()
        if not prefix or prefix[-1] in '.!?\n':
            continue  # sentence-initial capital is grammar, not a proper-noun signal
        return True
    return False


def _question_names_distinctive_entity(question: str, pages: dict) -> bool:
    """Stricter _question_mentions_known_entity for the disambiguation gate: an
    entity token counts only when it ALSO appears as a capitalised proper noun in
    the question. Stops a generic dictionary word that leaked into the entity set
    from suppressing disambiguation on a genuinely vague query (e.g. "Summarize
    this document…" listing "term, termination, liability" — none proper nouns).

    This is the actual skip check classify_query uses before ever asking the LLM
    whether to disambiguate — so its blind spot is not cosmetic. It only reads
    _extract_doc_entities, which mines PAGE TITLES; the filename-token check
    below (_question_names_corpus_doc_token, already proper-noun-gated the same
    way) covers the party name as the user actually types it. Confirmed live:
    "Can the vendor or its model provider use client data to train or improve
    their AI models?" — asked, then "same document", then "the document I just
    spoke about in the previous question" — got the identical disambiguation
    prompt three times in one thread, because "Hyden" satisfies neither check
    without this addition, so nothing before the LLM triage could ever resolve
    it, and the triage has no memory of the reply already given.
    """
    q = question.lower()
    return (any(
        _contains_token(ent, q) and _appears_as_proper_noun(ent, question)
        for ent in _extract_doc_entities(pages)
    ) or _question_names_corpus_doc_token(question, pages))


def _pages_matching_question_entity(question: str, pages: dict) -> list[str]:
    """Return page titles whose document identifier contains an entity name
    mentioned in the question. Used to force-scope context to the right document.

    Prefers identifiers matching the MOST distinct hit tokens over identifiers
    matching just one. A question naming a party pair ("the Zephyr-Solaris NDA")
    yields two hit tokens ("zephyr", "solaris"); each name alone is common across
    a large synthetic corpus (100+ matches), but the *pair* together identifies
    one specific document precisely. Without this, a single-name flood (e.g.
    "zephyr" alone hitting 242 pages) buries the compound match and can push the
    total past ENTITY_MATCH_MAX_PAGES, abandoning force-include entirely.
    """
    q = question.lower()
    hits = {ent for ent in _extract_doc_entities(pages) if _contains_token(ent, q)}
    if not hits:
        return []
    by_match_count: dict[int, list[str]] = {}
    for title in pages:
        ident = _doc_identifier_part(title).lower()
        if not ident:
            continue
        n = sum(1 for h in hits if _contains_token(h, ident))
        if n > 0:
            by_match_count.setdefault(n, []).append(title)
    if not by_match_count:
        return []
    best = max(by_match_count)
    return by_match_count[best]


# Plural / collective family nouns that signal a question is asking ABOUT A SET
# of documents ("compare the NDAs", "summarize the agreements") rather than one.
# Combined with a family keyword (via _DOC_FAMILY_RULES) to resolve family scope.
_PLURAL_FAMILY_HINT_RE = re.compile(
    r'\b(ndas|agreements|judgments|judgements|opinions|pleadings|petitions|'
    r'contracts|ventures|leases)\b',
    re.IGNORECASE,
)


# Family keywords that do NOT, on their own, refer to a DOCUMENT. Against a
# doc_type string at ingest ("License Agreement") the bare noun is unambiguous,
# which is what _DOC_FAMILY_RULES was written for. Against a free-form QUESTION
# the same word is usually ordinary prose — "the Quantum-Mesh IP license", "a
# Letter of Intent for integrated design consultancy services" — and treating it
# as a document reference scopes the search to the wrong family, which now also
# EXCLUDES the document holding the answer. Measured on the live corpus: Q46
# (answer in a tax opinion) and Q105 (answer in a court judgment) were both
# pulled into the wrong family this way. These require an explicit document noun
# immediately after them; the multi-word and acronym rules stay as they were.
_GENERIC_FAMILY_WORDS = frozenset({
    "license", "licence", "service", "employment", "consulting", "supply",
    "opinion", "court",
})
_DOC_NOUN = r'(?:agreement|contract|deed|document)'


def _detect_question_family(question: str, available_families: set[str]) -> str | None:
    """Return the single document family a question refers to, or None.

    Reuses the same keyword→family rules as ingest-time normalization
    (_DOC_FAMILY_RULES). Returns a family only when EXACTLY ONE known family is
    referenced AND it actually exists in this session — a question naming two
    families ("the service agreements and the NDAs") is cross-family, so it
    stays unfiltered (None) rather than being wrongly narrowed to one.
    """
    if not available_families:
        return None
    q = question.lower()
    matched: set[str] = set()
    for keyword, family in _DOC_FAMILY_RULES:
        if family not in available_families:
            continue
        # Plural-tolerant, word-boundary match: a collective family question uses
        # the plural ("compare the NDAs", "the service agreements"), so allow an
        # optional trailing 's' on the keyword's final word — without it "nda"
        # would fail to match "NDAs" and silently drop the family.
        if keyword in _GENERIC_FAMILY_WORDS:
            pattern = rf'\b{re.escape(keyword)}s?\s+{_DOC_NOUN}s?\b'
        else:
            pattern = rf'\b{re.escape(keyword)}s?\b'
        if re.search(pattern, q):
            matched.add(family)
    return next(iter(matched)) if len(matched) == 1 else None


# A party the user names to identify an agreement almost always carries its
# corporate form ("… Private Limited", "… GmbH"). Capturing the capitalised
# words immediately BEFORE that suffix yields the distinctive party name
# ("SteelLoop Resource Recovery", "Cold Chain Energy Services") without dragging
# in surrounding prose — and the suffix gate keeps ordinary capitalised topic
# phrases ("Reserved Matters", "Joint Venture Agreement") from ever qualifying.
_CORP_SUFFIX_RE_STR = (
    r'(?:Private\s+Limited|Pvt\.?\s*Ltd\.?|Pte\.?\s*Ltd\.?|Limited|Ltd\.?|'
    r'LLP|LLC|FZE|FZC|Inc\.?|Corp(?:oration)?|PLC|GmbH|N\.?V\.?|S\.?A\.?)'
)
_PARTY_NAME_RE = re.compile(
    r'\b((?:[A-Z][A-Za-z0-9&.\-]+\s+){1,6}?)' + _CORP_SUFFIX_RE_STR + r'\b'
)

# A company name typed in shorthand ALL-CAPS carries no corporate suffix at all
# ("TATA POWER SOLAR", "JLR EUROPE") — _PARTY_NAME_RE's suffix anchor never
# fires on it. Without a second detector, resolve_scope's unresolved_party gate
# (below) sees no party name, treats the question as scopeless, and carries the
# PREVIOUS document's stale scope forward instead of searching for the company
# actually named — guaranteeing "not covered" for a real company the corpus may
# well have documents about, since retrieval never actually looked for it.
# Confirmed live: three turns into a Service Agreement 2 thread, "what
# information do we have about TATA POWER SOLAR" answered "not covered" while
# scoped to SA2, having never searched for Tata Power Solar at all. Requires 2+
# ALL-CAPS words specifically (not just Title Case) — ordinary capitalised legal
# vocabulary a user copies from a document ("Confidential Information", "Force
# Majeure") is essentially never typed in all caps, so this stays narrow.
# "in this agreement", "under that contract", "of the said deed" — a phrase
# that can only mean the document already being discussed. Requires a
# demonstrative: a bare "the agreement" is how a first question names a
# document type, not a back-reference.
_RX_DEMONSTRATIVE_DOC = re.compile(
    r"\b(?:in|under|of|for|about|within)\s+(?:this|that|the\s+said|the\s+same)\s+"
    r"(?:agreement|contract|document|deed|lease|instrument|sla|nda|msa|dpa|spa|"
    r"sow|licence|license|arrangement)\b",
    re.IGNORECASE)


_BARE_ALLCAPS_ENTITY_RE = re.compile(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,4}\b')

# A third naming style neither of the above catches: a single Title-Case word
# with NO corporate suffix and NO second ALL-CAPS word — someone's shorthand
# for a party ("Brackenpyre", "Hyden"), the way people actually refer to a
# counterparty in conversation rather than by its full registered name.
# Confirmed live: "the required timeframe for Brackenpyre to notify the
# Client" extracts zero candidates from _PARTY_NAME_RE (no suffix) or
# _BARE_ALLCAPS_ENTITY_RE (no second all-caps word) — even though the corpus
# holds exactly three documents mentioning "Brackenpyre" by name — so
# _resolve_docs_by_party never even tries a content search for it, and the
# question falls through to the LLM disambiguation triage every time.
#
# Deliberately the weakest signal of the three, so it is tried only as a last
# resort (see _resolve_docs_by_party) and leans on the SAME distinctiveness
# cap every candidate here is already subject to: a stopword that slips
# through just costs one wasted content-search query, filtered out for
# matching too many documents to resolve anything.
_BARE_PROPER_NOUN_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "did", "do", "is", "are", "was", "were", "can", "could", "would",
    "should", "will", "shall", "must", "may", "might", "have", "has", "had",
    "the", "this", "that", "these", "those", "their", "its", "our", "your",
    "under", "over", "before", "after", "between", "within", "during",
    "against", "please", "kindly", "also", "then", "there", "here",
    # Sentence-initial words. English capitalises the first word of every
    # sentence regardless of what it is, so position alone is no evidence of a
    # proper noun - but _BARE_PROPER_NOUN_RE only sees Title-Case and cannot
    # tell the two apart. Confirmed live: "Tell me more about the second one."
    # yielded "Tell" as a bare party-name candidate, whose content search
    # matched two unrelated legal opinions and pinned the whole follow-up to
    # them, discarding the document the conversation was actually on. The
    # conjunctions are the same failure one turn later ("And what happens
    # if..."). None of these is ever a party name on its own.
    "tell", "give", "show", "draft", "find", "identify", "outline", "walk",
    "and", "in", "of", "for", "as", "on", "at", "but", "so", "if",
    "client", "vendor", "party", "parties", "agreement", "agreements",
    "contract", "contracts", "document", "documents", "clause", "clauses",
    "section", "sections", "schedule", "schedules", "annexure", "annexures",
    "service", "services", "statement", "work", "data", "processing",
    "master", "regarding", "concerning", "according", "prepare", "provide",
    "explain", "describe", "summarize", "summarise", "compare", "list",
})
_BARE_PROPER_NOUN_RE = re.compile(r'\b[A-Z][a-z]{3,}\b')

# A run of 2-5 whitespace-separated Title-Case words with no corporate suffix
# ("Apex Lumendra Digital") — this corpus's dominant bare-name pattern, one
# level up from the single-word case above. _bare_proper_noun_candidates
# offers each word of a name like this SEPARATELY, which is fatal for a
# 3-word name built from common short words: "Apex" and "Digital" alone each
# hit dozens of unrelated documents (every "Apex *" company, every "* Digital"
# company), so both get discarded by _resolve_docs_by_party's max_docs cap and
# the name never resolves at all — confirmed live on "the Guarantee agreement,
# Apex Lumendra Digital, Jan 2021", which fell all the way through to
# unscoped corpus search and answered from an unrelated document. Trying the
# full 3-word phrase as ONE candidate first is what a real full-text search
# for the name would do, and resolves to exactly the one document.
_BARE_PROPER_NOUN_PHRASE_RE = re.compile(
    r'\b(?:[A-Z][a-z]{3,}\s+){1,4}[A-Z][a-z]{3,}\b'
)


def _bare_proper_noun_candidates(question: str) -> list[str]:
    """Single Title-Case words in ``question`` that aren't common English/legal
    vocabulary — candidate bare party-name shorthand for _resolve_docs_by_party.
    """
    seen: list[str] = []
    for m in _BARE_PROPER_NOUN_RE.finditer(question):
        tok = m.group(0)
        if tok.lower() in _BARE_PROPER_NOUN_STOPWORDS:
            continue
        if tok not in seen:
            seen.append(tok)
    return seen


def _bare_proper_noun_phrase_candidates(question: str) -> list[str]:
    """Multi-word runs of bare Title-Case words — the phrase-level sibling of
    _bare_proper_noun_candidates (see that function and _BARE_PROPER_NOUN_PHRASE_RE
    above). A run containing ANY stopword token is dropped whole rather than
    trimmed — e.g. "Guarantee Agreement Apex" never occurs in practice since
    "agreement" is lowercase mid-sentence, but if a stopword ever lands inside
    a matched run, guessing which end to trim risks cutting a real name in
    half, so the safer failure is no candidate at all.
    """
    seen: list[str] = []
    for m in _BARE_PROPER_NOUN_PHRASE_RE.finditer(question):
        phrase = m.group(0)
        words = phrase.split()
        if any(w.lower() in _BARE_PROPER_NOUN_STOPWORDS for w in words):
            continue
        if phrase not in seen:
            seen.append(phrase)
    return seen


# Any alphanumeric token worth checking against a filename during narrowing —
# deliberately loose (letters, digits, internal hyphens), since the whole
# point is to catch things _PARTY_NAME_RE/_bare_proper_noun_* never would:
# document codes like "IMG-4137", instrument words like "Guarantee" or "SOW".
_NARROW_TOKEN_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9\-]{2,}')

# Basic English function words that _NARROW_TOKEN_RE happily matches (it has
# no case requirement, unlike _BARE_PROPER_NOUN_STOPWORDS which was built for
# a Title-Case-only regex and never needed to exclude them). Confirmed live:
# "and" — from "... between Tata Projects Limited and Bhumika Motors Ltd
# ..." — matched the filename "Separation AND Release Agreement", an
# entirely unrelated document, and (being the single smallest-matching
# token) won the fallback below and was returned as the answer.
_NARROW_TOKEN_STOPWORDS = frozenset({
    "and", "for", "with", "from", "into", "provide", "provides", "dated",
    "between", "the", "that", "this", "was", "were", "has", "have", "had",
    "not", "but", "nor", "are", "will", "shall", "may", "can", "does", "did",
    "states", "state", "stated", "under", "about", "any", "all", "each",
    # Filing/drafting-status words this corpus's own filenames use as
    # boilerplate suffixes on a huge, unrelated cross-section of documents
    # ("... FINAL_FINAL.pdf", "... signed copy.pdf", "... - filed_ocr.pdf")
    # — near-universal, so they carry ~zero document-identifying signal, but
    # matched few enough documents to look "discriminating" by the bare
    # subset test. Confirmed live: "final" (from "Tax Deed FINAL_FINAL
    # agreement") uniquely matched a single unrelated Lease Deed document
    # whose filename happened to end "... FINAL_FINAL.pdf", won the
    # empty-intersection fallback, and was confidently returned as the
    # answer instead of the real Tax Deed.
    "final", "signed", "draft", "copy", "filed", "executed", "scanned",
    "redacted", "countersigned", "clean", "fully", "true",
})

# The corpus files a multi-word instrument type under its INITIALISM — "KERA",
# "SSA", "TSA", "SPA" — while questions spell the type out in full ("the Key
# Employee Retention Agreement"). Neither shares a token with the other, so
# filename narrowing sees the type words match nothing, discards them, and
# narrows on whatever generic word is left — usually "agreement", which then
# selects FOR the siblings whose filenames spell the type out and AGAINST the
# one document that uses the acronym. Confirmed live: "the Key Employee
# Retention Agreement between Apex Sagar Mobility Limited and Ashoka Travel
# Limited" narrowed an 8-document cluster to the 5 filenames containing
# "Agreement", dropping "MAT-2021-6077_Apex Sagar Mobility_KERA_2019-12-25.pdf"
# — the one document the question was actually about.
#
# Deriving the initialism from the spelled-out name closes that gap with the
# corpus's own naming convention. Only Title-Case runs ending in an instrument
# head noun qualify, so ordinary capitalised prose never manufactures a token.
# The head noun list covers what this corpus's instruments are actually called.
# It started at Agreement/Deed/Contract and missed "Board Resolution Approving
# Transaction" — whose document is filed as "..._BRAT_11Apr2019.pdf" — so a
# question naming it had no initialism to narrow 24 board resolutions with, and
# the scope stayed on five unrelated documents of the same parties.
_INSTRUMENT_INITIALISM_RE = re.compile(
    r'\b((?:[A-Z][a-z]+\s+){2,5}'
    r'(?:Agreement|Deed|Contract|Undertaking|Opinion|Memorandum|Notice|Policy|'
    r'Transaction|Resolutions?|Certificate|Letter|Statement|Sheet|Guarantee|'
    r'Consent|Plaint|Petition|Order|Work|Intent|Minutes|Schedule|Charter))\b'
)


def _instrument_initialisms(question: str) -> list[str]:
    """Lower-case initialisms of any spelled-out instrument type the question names."""
    out: list[str] = []
    for m in _INSTRUMENT_INITIALISM_RE.finditer(question):
        acronym = "".join(w[0] for w in m.group(1).split()).lower()
        if 3 <= len(acronym) <= 6 and acronym not in out:
            out.append(acronym)
    return out


def _shares_family(session_id: str, docs_a: set[str], docs_b: set[str]) -> bool:
    """Do the two document sets contain the same KIND of instrument?

    Used to reject a "second document reference" that is really a sibling of the
    one already resolved. Unknown families never block: a document the family
    classifier never labelled says nothing either way, and refusing on missing
    data would silently disable the branch this guards.
    """
    if not config.USE_DATABASE or not docs_a or not docs_b:
        return False
    try:
        families = _db.get_families_of_documents(
            _active_wiki_id(), session_id, sorted(docs_a | docs_b))
    except Exception as e:
        logger.warning("family comparison failed: %s", e)
        return False
    fam_a = {families[d] for d in docs_a if d in families}
    fam_b = {families[d] for d in docs_b if d in families}
    return bool(fam_a & fam_b)


# A matter or case number written the way a question writes it — "Appeal No.
# 113/2024", "C.S. No. 248/2026", "C.P. No. 499/2023". The filename writes the
# same number with the separator dropped ("... Appeal No. 1132024-20240619 signed
# copy.pdf"), and narrowing normalises punctuation away, so the two forms would
# match — except the question's own tokenizer splits on the slash into "113" and
# "2024", both purely numeric and therefore discarded as too weak to trust
# alone. Joined back up, the number is one of the most specific identifiers a
# question can carry.
_CASE_NUMBER_RE = re.compile(r'\b(\d{1,5})\s*[/\\]\s*(\d{2,4})\b')

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def _precise_filename_tokens(question: str, allow_initialism: bool = True) -> list[str]:
    """Tokens specific enough that a single filename match settles the question.

    Ordered most-specific first. Each is something the corpus writes into a
    filename verbatim but that ordinary word tokenizing cannot reconstruct: a
    case number split by its slash, a date written out in words, or an
    instrument type the filename abbreviates to its initialism.
    """
    out: list[str] = []

    def add(tok: str) -> None:
        if tok and tok not in out:
            out.append(tok)

    for m in _CASE_NUMBER_RE.finditer(question):
        add(f"{m.group(1)}{m.group(2)}")

    for m in _QUESTION_DATE_RE.finditer(question):
        parts = re.findall(r"[A-Za-z]+|\d+", m.group(0))
        month = day = year = None
        for p in parts:
            low = p.lower()
            if low in _MONTHS:
                month = _MONTHS[low]
            elif len(p) == 4 and p.isdigit():
                year = int(p)
            elif p.isdigit():
                day = int(p)
        if not (month and day and year):
            continue
        # The three orderings this corpus actually files under, plus the
        # written-month form ("19Dec2022"). Punctuation is normalised away
        # by the caller, so "2024-06-19" and "20240619" are the same token.
        add(f"{year:04d}{month:02d}{day:02d}")
        add(f"{day:02d}{month:02d}{year:04d}")
        add(f"{month:02d}{day:02d}{year:04d}")
        month_name = [k for k, v in _MONTHS.items() if v == month][0]
        add(f"{day:02d}{month_name[:3]}{year:04d}")
        add(f"{day:02d}{month_name}{year:04d}")

    if allow_initialism:
        for acronym in _instrument_initialisms(question):
            add(acronym)
    return out


def _narrow_by_question_tokens(question: str, candidate_docs: set[str],
                               exclude: str | None = None,
                               allow_precise: bool = True,
                               allow_initialism: bool = True,
                               precise_only: bool = False) -> set[str]:
    """Narrow a multi-document match using whatever ELSE the question names.

    A party name or matter reference that resolves to several documents isn't
    a dead end if the question also names something document-specific — an
    instrument type ("the Guarantee agreement"), a document code embedded in
    the filename ("IMG-4137"), a short form ("PPA"). Those live in the
    FILENAME, not the page content a phrase search already matched against,
    so this checks candidate filenames directly rather than repeating a
    content search.

    Confirmed live: "MAT-2021-7750 (IMG-4137 PPA)" resolves the matter number
    to several sibling documents (a whole deal's worth of instruments sharing
    one matter reference is normal, not ambiguous data) — "IMG-4137" and
    "PPA" both appear only in one sibling's filename, narrowing to exactly
    it. Same mechanism narrows "Guarantee agreement, Apex Lumendra Digital"
    — the party name alone resolves to several real, unrelated deals this
    company is party to, but only one of those filenames contains
    "Guarantee".

    Tokens that DISCRIMINATE (match a proper, non-empty subset of
    candidate_docs — not none of them, not all of them) are AND-ed together,
    not OR-ed: a token matching most of the candidates says nothing about
    which one is right, and OR-combining loose tokens accumulates false
    positives. Confirmed live: on a 20-document candidate set, OR-matching
    let generic filename words ("final", "signed", "draft") pull in two
    completely unrelated documents alongside the real one.

    If the AND intersection of every discriminating token comes back empty,
    that means the tokens DISAGREE — each is individually plausible but they
    don't point at the same document, which is a sign of noise, not a
    tie-breaker to resolve. This deliberately returns candidate_docs
    UNCHANGED in that case rather than trusting whichever token happened to
    match the fewest documents — confirmed live that "trust the smallest"
    is actively dangerous: "final" (a boilerplate drafting-status suffix on
    a huge, unrelated slice of this corpus's filenames) matched only one
    document by chance, disagreed with every other token, and would have
    been confidently returned as the answer instead of the real one. Every
    caller already only accepts a narrowing result at exactly length 1, so
    returning the unnarrowed set here correctly reads as "couldn't narrow,"
    not as a wrong answer.

    Two further collisions this guards against, both confirmed live:
    - A bare year ("Jan 2021" → token "2021") coincidentally matching an
      UNRELATED matter number embedded in a sibling's filename ("MAT-2021-
      6375"). Purely-numeric tokens are excluded — a year alone is never
      distinctive enough to trust here, unlike an alphanumeric code.
    - The resolved name/reference itself, when it's also literally embedded
      in every sibling's filename (a matter number folded into each of its
      own instrument's filenames), matching all of them and cancelling out
      the narrowing entirely. ``exclude`` is the resolved phrase/reference
      that produced candidate_docs — stripped from the token set so it can
      only narrow using signals OTHER than the one already used to find
      this candidate set.

    Returns candidate_docs unchanged (never widens, never guesses) unless a
    token narrows it to a strictly smaller, non-empty set.
    """
    if len(candidate_docs) <= 1:
        return candidate_docs
    exclude_norm = re.sub(r'[^a-z0-9]', '', exclude.lower()) if exclude else None
    tokens: list[str] = []
    for m in _NARROW_TOKEN_RE.finditer(question):
        raw = m.group(0)
        norm = re.sub(r'[^a-z0-9]', '', raw.lower())
        # Deliberately NOT filtering against _BARE_PROPER_NOUN_STOPWORDS here —
        # that set exists to keep instrument-type words ("agreement", "data",
        # "processing", "service", "master") OUT of party-name candidates, but
        # this function's whole purpose is narrowing BY instrument-type words
        # ("the Guarantee agreement" — see docstring). Reusing it silently
        # stripped exactly the tokens meant to discriminate. Confirmed live:
        # "the Data Processing Agreement between Tata Capital Limited and
        # Vishesh Motors Limited" lost "data"/"processing"/"agreement" to that
        # filter, leaving ordinary-English "term" (from "how is the term
        # 'Affiliate' defined") as the only surviving token — which then
        # collided with an unrelated sibling's "Term Sheet" filename and won.
        if len(norm) < 3 or raw.lower() in _NARROW_TOKEN_STOPWORDS:
            continue
        if norm.isdigit():
            continue
        if exclude_norm and norm in exclude_norm:
            continue
        if norm not in tokens:
            tokens.append(norm)
    haystacks = {d: re.sub(r'[^a-z0-9]', '', d.lower()) for d in candidate_docs}
    # A case number, a written-out date or an instrument initialism is a far
    # higher-precision signal than any single ordinary word: each is the
    # corpus's own way of writing something the question states exactly. When
    # one picks out a single candidate, take it outright rather than AND-ing it
    # with generic words that would only cancel it out — "kera" and "agreement"
    # intersect to nothing, and the empty-intersection guard below would then
    # discard both. Confirmed live: "the Affidavit in Support of the Plaint -
    # Appeal No. 113/2024 ... dated 19 June 2024" carried both the case number
    # and the date that its filename spells out ("... Appeal No.
    # 1132024-20240619 signed copy.pdf") and still fell back to answering
    # across all 28 Pleadings.
    for precise in (_precise_filename_tokens(question, allow_initialism) if allow_precise else []):
        if exclude_norm and precise in exclude_norm:
            continue
        hit = {d for d, h in haystacks.items() if precise in h}
        if len(hit) == 1:
            return hit
    # precise_only: the caller wants the high-precision tokens ONLY, and treats
    # "no precise match" as "could not narrow" rather than falling back to
    # ordinary words.
    if precise_only or not tokens:
        return candidate_docs
    discriminating: list[tuple[str, set[str]]] = []
    for t in tokens:
        matched = {d for d, h in haystacks.items() if t in h}
        if matched and len(matched) < len(candidate_docs):
            discriminating.append((t, matched))
    if not discriminating:
        return candidate_docs
    intersection: set[str] | None = None
    for _, matched in discriminating:
        intersection = matched if intersection is None else (intersection & matched)
    if not intersection:
        return candidate_docs
    if len(intersection) < len(candidate_docs):
        return intersection
    return candidate_docs


# An explicit calendar date typed in the question ("the SA dated 15 January
# 2026", "signed on August 28, 2025"). Two orderings: day-month-year (the
# convention this corpus's own documents use) and month-day-year.
_QUESTION_DATE_RE = re.compile(
    r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{4}\b'
    r'|\b(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2},?\s+\d{4}\b',
    re.IGNORECASE,
)


def _resolve_docs_by_effective_date(question: str, session_id: str) -> set[str]:
    """The one document whose stored effective_date is the date the question recites.

    _resolve_docs_by_date below searches PAGE TEXT for the date as a phrase,
    which only works when the document recites its own date in prose that
    survived ingest. Measured on the 200-question audit: of ten questions whose
    named document was never retrieved, seven recite an explicit date, and the
    content search found the right document for none of them.

    documents.effective_date answers the same question exactly. Fires ONLY on a
    unique match across the corpus - a date shared by several documents is not
    an identifier, which is why the party name outranks it everywhere else.
    """
    if not config.USE_DATABASE:
        return set()
    matches = [m.group(0) for m in _QUESTION_DATE_RE.finditer(question or "")]
    if not matches:
        return set()
    # Two dates means two documents. A comparison question ("the Legal Opinion
    # dated 31 October 2024 and the Legal Opinion dated 12 September 2025")
    # would otherwise be pinned to whichever of them happens to have a unique
    # date, turning a two-document comparison into a one-document answer —
    # a worse failure than the one this resolver exists to fix.
    if len({_db.parse_effective_date(m) for m in matches} - {None}) > 1:
        return set()
    for date_str in matches:
        parsed = _db.parse_effective_date(date_str)
        if not parsed:
            continue
        try:
            docs = _db.find_documents_by_effective_date(
                _active_wiki_id(), session_id, parsed.isoformat(), cap=3)
        except Exception as e:
            logger.error("resolve_scope: effective-date lookup failed for %r: %s",
                         date_str, e)
            continue
        if len(docs) != 1:
            continue
        # A unique date match is only an identifier if the RIGHT document has
        # its date stored. effective_date is empty on 143 documents even after
        # the filename backfill, so "the only document dated 24 October 2025"
        # can be an unrelated NDA while the Facility Agreement actually asked
        # about carries no date at all — which is exactly what happened, and
        # it turned a correct answer into one about the wrong document.
        #
        # So when the question names a party, the date match must agree with
        # it. A question naming no party ("the Master Services Agreement dated
        # 12 August 2020") has nothing to corroborate against and the unique
        # date stands on its own.
        named = [m.group(1).strip() for m in _PARTY_NAME_RE.finditer(question or "")]
        named = [n for n in named if len(n.split()) >= 2 and len(n) >= 6]
        if named:
            try:
                agree = _db.count_documents_by_party(
                    _active_wiki_id(), session_id, [named[0]], None, limit=1)
                if not agree.get("total"):
                    # The party is unknown to the index — nothing to check
                    # against, so fall through rather than veto on no evidence.
                    pass
                else:
                    hit = _db.count_documents_by_party(
                        _active_wiki_id(), session_id, [named[0]], None, limit=50)
                    if not any(d.get("source_doc") == docs[0]
                               for d in (hit.get("documents") or [])):
                        logger.info("Effective-date match %r rejected: %s does not "
                                    "name %r", date_str, _norm_doc_name(docs[0]), named[0])
                        continue
            except Exception as e:
                logger.error("resolve_scope: date/party corroboration failed: %s", e)
                continue
        logger.info("Effective-date match -> 1 document: %s (%r)",
                    _norm_doc_name(docs[0]), date_str)
        return set(docs)
    return set()


def _resolve_docs_by_date(question: str, session_id: str, max_docs: int = 1) -> set[str]:
    """Resolve the document a question names by an explicit date it recites.

    Mirrors _resolve_docs_by_party below, same content-FTS mechanism, for a
    different way lawyers pin a document with no filename and no party name in
    hand: by a date they already know ("the Service Agreement dated 15 January
    2026"). The date string is a phrase search against page content the same
    way a party name is — execution/effective dates are recited near-verbatim
    in a document's own recitals. Deliberately max_docs=1 by default (unlike
    the party resolver's small-cluster tolerance): two different documents
    sharing the exact same execution date is a real possibility in this corpus
    (batch-signed agreements), so this only fires on a genuinely unique hit,
    never picks among several dated the same day. Returns an empty set on any
    ambiguity, so it only ever ADDS a match other detectors miss.
    """
    if not config.USE_DATABASE:
        return set()
    dates = _QUESTION_DATE_RE.findall(question)
    # findall on an alternation with no groups returns the full match already;
    # re-run finditer defensively in case a future edit adds a capture group.
    if not dates:
        return set()
    matches = [m.group(0) for m in _QUESTION_DATE_RE.finditer(question)]
    for date_str in matches:
        try:
            docs = [d for d in _db.find_source_docs_mentioning_phrase(_active_wiki_id(), session_id, date_str, cap=max_docs + 1) if d]
        except Exception as e:
            logger.error("resolve_scope: date-content lookup failed for %r: %s", date_str, e)
            continue
        if len(docs) == 1:
            logger.info("Date content match → 1 document: %s (%r)",
                        _norm_doc_name(docs[0]), date_str)
            return set(docs)
    return set()


# A matter/reference number recited in the question ("MAT-2021-7750"). Looks
# unique but frequently ISN'T in this corpus — the same matter number is
# reused across several unrelated instruments for the same deal (a Loan
# Agreement, an IP Assignment, an Escrow Agreement, all filed under
# "MAT-2021-7750" with different parties on each). Deliberately max_docs=1:
# on a multi-hit it returns nothing rather than guessing among siblings, same
# fail-safe the date resolver above uses. Confirmed live: "the termination
# notice period in MAT-2021-7750 (IMG-4137 PPA)" resolved to NOTHING under the
# old resolvers (no filename match, no party name in the question at all) and
# fell through to unscoped corpus search, which answered from a different
# MAT-2021-7750 sibling with different parties and a different notice period.
_QUESTION_MATTER_REF_RE = re.compile(r'\bMAT-\d{4}-\d{3,6}\b', re.IGNORECASE)


def _resolve_docs_by_matter_reference(question: str, session_id: str) -> set[str]:
    """Resolve the document a question names by a matter/reference number it
    recites. Mirrors _resolve_docs_by_date's content-FTS mechanism, but a
    matter number routinely hits several sibling instruments of the same deal
    (see _QUESTION_MATTER_REF_RE above), so a multi-hit isn't a dead end —
    _narrow_by_question_tokens gets a chance to pin it down using whatever
    else the question names (a filename code, an instrument type) before
    this gives up and returns nothing.
    """
    if not config.USE_DATABASE:
        return set()
    refs = [m.group(0) for m in _QUESTION_MATTER_REF_RE.finditer(question)]
    if not refs:
        return set()
    # A matter can legitimately span a handful of instruments; capped well
    # above that so narrowing has the full sibling set to work with, not a
    # query-truncated slice of it.
    _MATTER_SCAN_CAP = 20
    for ref in refs:
        try:
            docs = [d for d in _db.find_source_docs_mentioning_phrase(_active_wiki_id(), session_id, ref, cap=_MATTER_SCAN_CAP) if d]
        except Exception as e:
            logger.error("resolve_scope: matter-reference lookup failed for %r: %s", ref, e)
            continue
        if not docs:
            continue
        if len(docs) == 1:
            logger.info("Matter-reference content match → 1 document: %s (%r)",
                        _norm_doc_name(docs[0]), ref)
            return set(docs)
        narrowed = _narrow_by_question_tokens(question, set(docs), exclude=ref)
        if len(narrowed) == 1:
            logger.info("Matter-reference content match, narrowed by question tokens → 1 document: %s (%r, %d siblings)",
                        _norm_doc_name(next(iter(narrowed))), ref, len(docs))
            return narrowed
    return set()


def _narrow_by_title_hint(session_id: str, candidate_docs: set[str], question: str,
                          exclude: str | None = None) -> set[str]:
    """Like _narrow_by_question_tokens, but checks page TITLES instead of
    filenames — catches the instrument-type label ingest assigns even when
    the filename itself is an opaque code the label never made it into.

    Confirmed live: a party-name resolver correctly finds a 7-document
    cluster for "Redgate Mobility" but the question's other document
    reference — "the Detailed Judgment and Final Order ... C.S. No.
    248/2026" — names an instrument type that never appears in the real
    judgment's filename ("...DJAFOCN - filed_ocr.pdf"), so filename-based
    narrowing finds nothing. Its page TITLES do carry the type
    ("Overview – C.S. No. 248/2026 (Court Judgment)") because ingest writes
    it there regardless of what the filename says. "judgment" alone narrows
    the 7-document cluster to 2 via title search.

    Tries each token independently (not AND-combined — one real hit is
    enough) and returns on the first that narrows to a smaller, non-empty
    subset. Returns candidate_docs unchanged otherwise.
    """
    if len(candidate_docs) <= 1:
        return candidate_docs
    exclude_norm = re.sub(r'[^a-z0-9]', '', exclude.lower()) if exclude else None
    tokens: list[str] = []
    for m in _NARROW_TOKEN_RE.finditer(question):
        raw = m.group(0)
        norm = re.sub(r'[^a-z0-9]', '', raw.lower())
        if len(norm) < 4 or raw.lower() in _BARE_PROPER_NOUN_STOPWORDS or raw.lower() in _NARROW_TOKEN_STOPWORDS:
            continue
        if norm.isdigit():
            continue
        if exclude_norm and norm in exclude_norm:
            continue
        if norm not in tokens:
            tokens.append(norm)
    for t in tokens:
        try:
            title_hits = set(_db.find_source_docs_by_title_tokens(_active_wiki_id(), session_id, [t], cap=200) or [])
        except Exception:
            continue
        narrowed = candidate_docs & title_hits
        if narrowed and len(narrowed) < len(candidate_docs):
            return narrowed
    return candidate_docs


def _with_canonical_party_names(candidates: list[str]) -> list[str]:
    """Each candidate, plus its canonical entity name where one is recorded.

    Order is preserved and the original is always kept first: the smallest-set
    selection downstream picks by how few documents a name matches, and a
    canonical name that matches a broader set must not displace the specific
    string the question actually used.
    """
    try:
        from services import backbone as _bb
        wiki_id = _active_wiki_id()
    except Exception:
        return candidates
    out, seen = [], set()
    for name in candidates:
        for n in (name, _canonical_party_name(_bb, wiki_id, name)):
            k = (n or "").strip().lower()
            if n and k not in seen:
                seen.add(k)
                out.append(n)
    return out


def _canonical_party_name(bb, wiki_id: str, name: str) -> str | None:
    try:
        row = bb.resolve_entity(wiki_id, name)
    except Exception as e:
        logger.debug("entity canonicalisation failed for %r: %s", name, e)
        return None
    if not row:
        return None
    canon = (row.get("canonical_name") or "").strip()
    if not canon or canon.strip().lower() == (name or "").strip().lower():
        return None
    logger.info("Party name canonicalised: %r -> %r", name, canon)
    return canon


def _resolve_docs_by_party(question: str, session_id: str, max_docs: int = 4) -> set[str]:
    """Resolve the document(s) of a PARTY NAME typed in the question.

    Lawyers name an agreement by its counterparty ("the JV with Cold Chain
    Energy Services"), not by the filename ("JVA 4") the corpus stores it under.
    The party name often survives only in the document BODY — the filename is a
    bare type+number, the page-title identifier can be an ingest-synthesised
    short-name ("SunBridge-JV"), and the parties metadata may be redaction-masked
    ("[Redacted Logistics Infrastructure Partner]"). So resolve it by a full-text
    CONTENT search on the distinctive party phrase.

    Returns the doc set of the MOST distinctive party named — the one hitting the
    fewest documents — provided that set is small (<= max_docs), OR narrows to
    exactly one via _narrow_by_question_tokens when it isn't. An umbrella name
    like "Tata Steel Limited" hits many documents; the specific counterparty
    resolves to one document ("SteelLoop Resource Recovery" → JVA 3) or, when
    the same two parties share several instruments, to that small cluster
    ("Tata Steel & NordForge Metallurgy" → the NDA + arbitration notice +
    Section 9 petition) — or, when the question ALSO names an instrument type
    or document code an umbrella name alone can't narrow ("Guarantee
    agreement, Apex Lumendra Digital" — the party sits on 6 unrelated real
    deals, but only one filename says "Guarantee"), to that single document.
    The caller decides, from how many instruments the question names, whether
    to pin the whole cluster or narrow to one. Returns an empty set on
    genuine ambiguity (no hit, or nothing narrows a large set down), so it
    only ever ADDS precise matches the filename/entity detectors miss.
    """
    if not config.USE_DATABASE:
        return set()
    # Suffix-derived names ("Charitra Metals LLC") are real, distinct legal
    # entities — safe to treat two of them as two different documents (see
    # the secondary-match scan below). Bare-word fallback candidates
    # ("Guarantee", "Apex", "Digital") are not: they're single common words,
    # not party identities, so the secondary scan is restricted to this set.
    suffix_candidates = [m.group(1).strip() for m in _PARTY_NAME_RE.finditer(question)]
    suffix_candidates = [c for c in suffix_candidates if len(c) >= 4]
    candidates = suffix_candidates
    if not candidates:
        # Phrase candidates first: a bare multi-word name ("Apex Lumendra
        # Digital") searched whole is far more distinctive than any one of
        # its words searched alone, so it gets first crack at the smallest-set
        # selection below — but every candidate is still tried, so a genuine
        # single bare word ("Brackenpyre") is never crowded out.
        candidates = _bare_proper_noun_phrase_candidates(question) + _bare_proper_noun_candidates(question)
    if not candidates:
        return set()

    # Canonicalise each candidate through the entity registry before searching.
    # backbone.resolve_entity maps a name or a recorded spelling to the party's
    # canonical form across 530 entities and 447 aliases; it has existed and
    # been populated since the Phase 0 backbone and nothing in the query path
    # ever read it, so scope resolution has been matching raw strings against
    # page text the whole time. The canonical form is ADDED rather than
    # substituted: an alias that resolves gives two chances to find the
    # document, and a name the registry has never seen behaves exactly as
    # before, so this can widen a match and cannot narrow one.
    candidates = _with_canonical_party_names(candidates)

    # Scanned well above max_docs so a multi-doc match has its FULL sibling
    # set available to narrow against below, not a query-truncated slice that
    # happens to omit the one sibling a filename token would have pinned.
    _PARTY_SCAN_CAP = 20
    resolved: list[tuple[str, set[str]]] = []
    for name in candidates:
        try:
            docs = {d for d in _db.find_source_docs_mentioning_phrase(_active_wiki_id(), session_id, name, cap=_PARTY_SCAN_CAP) if d}
        except Exception as e:
            logger.error("resolve_scope: party-content lookup failed for %r: %s", name, e)
            continue
        if docs:
            resolved.append((name, docs))
    if not resolved:
        return set()
    best_name, best_docs = min(resolved, key=lambda pair: len(pair[1]))
    best_n = len(best_docs)
    pinned_precisely = False
    if best_n <= max_docs:
        primary = best_docs
        # A small cluster used to be returned as-is, skipping narrowing
        # entirely — but a question stating a date or a case number has
        # already named ONE of these documents. Confirmed live: "the Facility
        # Agreement between Apex Devashri InfoSystems Limited and Amberline
        # Commodities Limited dated 30 June 2023" returned the loan agreement
        # plus two amendments, and the answer reported the amendments'
        # governing law because it could not tell which document was meant,
        # although "30062023" appears in exactly one of the three filenames.
        # Only precise tokens are allowed to narrow here: ordinary words are
        # what the else-branch below uses as a last resort on a cluster too
        # big to return, and are not strong enough to discard siblings from a
        # cluster already small enough to be a legitimate answer.
        if len(primary) > 1:
            pinpoint = _narrow_by_question_tokens(question, primary, exclude=best_name,
                                                  precise_only=True)
            if len(pinpoint) == 1 and pinpoint < primary:
                primary, pinned_precisely = pinpoint, True
                logger.info("Party-name content match pinned to 1 document by an "
                            "identifier the question states: %s (%d siblings)",
                            _norm_doc_name(next(iter(primary))), best_n)
        if not pinned_precisely:
            logger.info("Party-name content match → %d document(s): %s",
                        best_n, {_norm_doc_name(d) for d in primary})
    else:
        narrowed = _narrow_by_question_tokens(question, best_docs, exclude=best_name)
        if len(narrowed) == 1:
            primary = narrowed
            logger.info("Party-name content match, narrowed by question tokens → 1 document: %s (%d siblings)",
                        _norm_doc_name(next(iter(primary))), best_n)
        else:
            primary = set()
    if not primary:
        return set()
    if pinned_precisely:
        # The question named one document by an identifier only that document
        # carries. There is no second document to look for, and looking anyway
        # found an unrelated Tax Deed to sit alongside the Facility Agreement.
        return primary

    # A question can name TWO separate documents by two separate parties
    # ("the judgment between X and Y ... as stated in the SSA between A and
    # B") — every other candidate above was discarded once the smallest
    # (most distinctive) one won, but a discarded candidate whose OWN
    # documents are disjoint from the primary pick is real evidence of a
    # second document being named, not noise. Confirmed live: "Tata Power"/
    # "Charitra Metals" (the SSA) won as smallest and returned alone, while
    # "Redgate Mobility"/"Apex Zephyra Trading" (the judgment) were silently
    # dropped — the judgment genuinely exists, fully indexed, and the answer
    # falsely reported it as absent. Tries filename narrowing first, then
    # title-hint narrowing (catches an instrument type that never made it
    # into the filename) — accepts a candidate only if it narrows to
    # max_docs or fewer AND doesn't overlap the primary pick.
    #
    # Restricted to suffix_candidates ONLY — confirmed live this cannot run
    # over the bare-word fallback too: on "Guarantee agreement, Apex Lumendra
    # Digital", the bare candidate "Guarantee" (a document TYPE, not a party)
    # narrowed its own huge candidate pool down to two unrelated "Apex
    # Lumendra Digital" amendment documents and got merged in as a false
    # "second document" — three single common words standing in for a party
    # identity is not the same guarantee a real corporate-suffixed name is.
    for name, docs in resolved:
        if name == best_name or name not in suffix_candidates:
            continue
        remainder = docs - primary
        if not remainder:
            continue
        # allow_precise=False: the instrument type, case number and date the
        # question states have already been spent identifying the PRIMARY
        # document. Letting any of them pin a second one finds a sibling of the
        # primary rather than the different instrument this branch exists to
        # recover, and hands the answer LLM two documents to confuse. Confirmed
        # live: "Section 5 of the Share Subscription Agreement between Tata
        # Elxsi Limited and Vantara Vehicles LLC" correctly pinned the Vantara
        # SSA, then pulled in an unrelated Tata Elxsi/Tata Elxsi SSA on the
        # strength of "SSA" alone.
        secondary = _narrow_by_question_tokens(question, remainder, exclude=name,
                                               allow_precise=False)
        if not secondary or len(secondary) > max_docs:
            secondary = _narrow_by_title_hint(session_id, remainder, question, exclude=name)
        # Exactly one, and a DIFFERENT kind of instrument than the primary.
        # "A second document is named here" is a claim about one document, and
        # this branch exists for a question naming two different instruments
        # ("the judgment ... as stated in the SSA"); a same-family match is a
        # sibling of the primary, which only gives the answer LLM two documents
        # of one type to confuse. Confirmed live: "Section 2 (Issues) of the
        # Detailed Judgment and Final Order - Appeal No. 113/2024 between
        # Pacific Rim Capital Bank Ltd and Vantara InfoSystems LLC" pinned the
        # right judgment and then merged in a Detailed Judgment and Final Order
        # from an entirely different matter. And a narrowing that lands on
        # SEVERAL documents is the other party's own portfolio rather than a
        # named cross-reference at all: "Section 9 (Severability) of the Key
        # Employee Retention Agreement between Apex Sagar Mobility Limited and
        # Ashoka Travel Limited" pinned the right KERA, then merged in four
        # unrelated Ashoka Travel instruments (an Escrow, a TSA, an SPA and a
        # Shareholder Agreement) alongside it.
        if len(secondary) == 1 and secondary.isdisjoint(primary) \
                and not _shares_family(session_id, primary, secondary):
            logger.info("Party-name content match also found a second document reference "
                        "(%r) → %s", name, {_norm_doc_name(d) for d in secondary})
            return primary | secondary
    return primary


# A question can name SEVERAL documents by informal nickname — "the Amberline
# NDA, the Apex Cobalt NDA, and the Apex Falcora EV NDA" — with no corporate
# suffix on any of them at all, so _PARTY_NAME_RE never sees them and the
# combinatorial party-pairing above never fires. _resolve_docs_by_party's own
# second-document recovery is deliberately restricted to suffix_candidates
# ONLY (see its docstring: a bare word standing in for a party identity isn't
# the same guarantee a real corporate-suffixed name is), so three bare
# nicknames collapse onto whichever ONE is most distinctive and the other two
# are never looked for. Confirmed live: exactly this question answered as if
# only the Apex Cobalt NDA existed, though the Amberline and Apex Falcora EV
# NDAs were both real, indexed documents.
#
# This pattern is narrow enough to resolve safely on syntax alone: a run of
# capitalised words immediately followed by a naming word for the KIND of
# instrument, since a lawyer names a document that way ("the Amberline NDA")
# far more often than that phrase shape occurs by coincidence in ordinary
# prose.
_NAMED_INSTRUMENT_RE = re.compile(
    r'\b(?:the\s+)?((?:[A-Z][A-Za-z0-9&\'.\-]*\s+){0,3}[A-Z][A-Za-z0-9&\'.\-]*)\s+'
    r'(NDAs?|Non-Disclosure\s+Agreements?|Agreements?|Notices?|Petitions?|'
    r'Judg(?:e)?ments?|Affidavits?|Complaints?|Contracts?)\b'
)


# The instrument words a lawyer actually attaches to a document nickname. Far
# wider than _NAMED_INSTRUMENT_RE's list, and case-insensitive on the KIND only
# — a question says "the Voltas escrow agreement" in lower case as readily as
# "the Amberline NDA" in caps, while the NAME must stay capitalised or this
# would match "the payment agreement" and resolve on a word that names nothing.
_SINGLE_INSTRUMENT_KIND = (
    r"(?i:NDAs?|MSAs?|SLAs?|DPAs?|SPAs?|SSAs?|SHAs?|SOWs?|JVAs?|LOIs?|POAs?|TSAs?"
    r"|non-disclosure\s+agreements?|confidentiality\s+agreements?"
    r"|master\s+services?\s+agreements?|service\s+level\s+agreements?"
    r"|data\s+processing\s+agreements?|share\s+purchase\s+agreements?"
    r"|share\s+subscription\s+agreements?|shareholders?'?\s+agreements?"
    r"|joint\s+venture\s+(?:governance\s+)?agreements?|escrow\s+agreements?"
    r"|purchase\s+agreements?|supply\s+agreements?|licen[cs]e\s+agreements?"
    r"|consultancy\s+agreements?|consulting\s+agreements?"
    r"|employment\s+agreements?|facility\s+agreements?|framework\s+agreements?"
    r"|technical\s+services?\s+agreements?|transition\s+services?\s+agreements?"
    r"|amendment\s+agreements?|services?\s+agreements?"
    r"|statements?\s+of\s+work|terms?\s+sheets?|term\s+sheets?"
    r"|conditions\s+precedent\s+checklists?|disclosure\s+letters?"
    r"|closing\s+certificates?|side\s+letters?|letters?\s+of\s+intent"
    r"|lease\s+deeds?|tax\s+deeds?|legal\s+opinions?"
    r"|board\s+resolutions?|powers?\s+of\s+attorney"
    r"|agreements?|contracts?|deeds?|leases?|opinions?|judg(?:e)?ments?"
    r"|orders?|petitions?|plaints?|affidavits?|notices?|checklists?)"
)

# Anchored on a preposition so the name is being used to POINT at a document
# ("under the Voltas escrow agreement"), not merely mentioned in passing.
_SINGLE_NAMED_INSTRUMENT_RE = re.compile(
    r"\b(?:in|of|under|for|from|about|within)\s+(?:the\s+)?"
    r"((?:[A-Z][A-Za-z0-9&'.\-]*\s+){0,3}[A-Z][A-Za-z0-9&'.\-]*)\s+"
    + _SINGLE_INSTRUMENT_KIND + r"\b"
)

# A capitalised word that begins a sentence, or is a generic legal noun, is not
# a document nickname. Without this, "What is the liability cap in The Agreement"
# would search the corpus for a company called "The".
_NOT_A_NICKNAME = {
    "the", "this", "that", "our", "their", "its", "a", "an", "any", "all",
    "what", "which", "who", "when", "where", "how", "does", "do", "is", "are",
    "master", "mutual", "original", "executed", "signed", "draft", "final",
    "same", "above", "below", "said", "such", "each", "both", "either",
}

# Small on purpose. One document is a clean resolution and a handful is a real
# ambiguity the answer can report per document; beyond that the nickname did
# not actually narrow anything and scope should stay where it was.
_SINGLE_NAMED_MAX_DOCS = 3


# Words that ride along on the front of a captured party name and are not part
# of it — _PARTY_NAME_RE anchors on a capital letter, so a sentence-initial
# "From Apex Zephyra Trading Company" captures the "From" too.
_PARTY_LEAD_NOISE = re.compile(
    r"^(?:from|in|of|under|for|between|with|against|by|the|this|that|and|to)\s+",
    re.IGNORECASE)

# How many documents a party pair may share before the pair stops being a
# usable narrowing signal. Generous, because the alternative is the whole
# corpus: twenty candidate documents is a scope a retrieval pass can rank
# sensibly, 1,372 is not.
_PAIR_FAMILY_MAX_DOCS = 20


def _resolve_docs_by_party_pair_index(question: str, session_id: str,
                                      max_docs: int = _PAIR_FAMILY_MAX_DOCS) -> set[str]:
    """Documents naming EVERY party the question names, read from documents.parties.

    The existing pair resolvers intersect page TITLES and page CONTENT, and both
    return nothing when a pair shares a whole document family: the titles do not
    carry party names, and a content intersection over sixteen documents is not
    a narrowing. Scope then fell through to an unscoped corpus search, which is
    the worst available answer — the pair is a real, strong signal and it was
    being discarded because it did not resolve to exactly one document.

    ``documents.parties`` is the clean JSONB array the counting path already
    reads reliably, so the intersection is exact rather than inferred from text.
    Matching is substring and case-insensitive because the corpus stores full
    legal names while a question says "Nimbus Capital".

    Returns the whole shared set, not a guess at which one was meant. Sixteen
    documents is a scope; one document chosen from sixteen without saying so is
    a fabrication with a citation attached.
    """
    if not config.USE_DATABASE:
        return set()

    names: list[str] = []
    for raw in _PARTY_NAME_RE.findall(question or ""):
        name = _PARTY_LEAD_NOISE.sub("", str(raw).strip()).strip(" ,.;:'\"")
        # Two words minimum: a single capitalised token is far too weak to
        # intersect on, and would pull in every document sharing one word.
        if len(name.split()) >= 2 and len(name) >= 6:
            if name.lower() not in {n.lower() for n in names}:
                names.append(name)
    if len(names) < 2:
        return set()

    try:
        from services import wikis as _wikis
        result = _db.count_documents_by_party(
            _wikis.active_wiki_id(), session_id, names, None,
            limit=max_docs + 1)
    except Exception as e:
        logger.error("resolve_scope: party-pair index lookup failed: %s", e)
        return set()

    total = int(result.get("total") or 0)
    if not total or total > max_docs:
        return set()
    docs = {d["source_doc"] for d in (result.get("documents") or []) if d.get("source_doc")}
    if not docs:
        return set()
    logger.info("Party-pair index %s -> %d document(s)", names, len(docs))
    return docs


# A pasted document name has to be long enough that matching it cannot be an
# accident. Fifteen characters of a filename is already far more specific than
# any phrase a question would contain by chance.
_MIN_PASTED_NAME_CHARS = 15


# The ingest folder prefix the UI strips when it displays a document, so a
# pasted display name can be compared against the stored one.
_RX_INGEST_ONLY_PREFIX = re.compile(r"^pdfs[_ ]by[_ ]category[_ ]generated[_ ]")
_RX_DOC_EXTENSION = re.compile(r"\.(?:pdf|docx?|txt|rtf)$", re.IGNORECASE)
_RX_INGEST_FOLDER_PREFIX = re.compile(
    r"^(?:pdfs[_ ]by[_ ]category[_ ]generated[_ ])?(?:[A-Za-z][A-Za-z ]{2,40}?[_])?",
)


def _resolve_docs_by_display_name(question: str, session_id: str) -> set[str]:
    """The document whose own displayed name the question quotes back.

    The app shows a document as "Consulting Agreement / Consultancy Agreement -
    2024-11-17 (2).pdf" and lists it in References as "pdfs by category
    generated Consulting Agreement Consultancy Agreement - 2024-11-17 (2)".
    When asked which document they meant, a user pastes one of those. Neither
    resolved: every resolver here looks for party names, instrument types,
    dates or matter codes, and a filename is none of those.

    So the system answered "the retrieved context does not contain a file
    titled 'Consulting Agreement Consultancy Agreement - 2024-11-17 (2)'" about
    a document it had displayed moments earlier, and answered from unrelated
    documents instead. Telling someone a document they can see does not exist
    is the single worst thing this system can say.

    Matched in the forward direction only — a stored name appearing IN the
    question — which is safe because these names are long and specific. A
    document is accepted only when its name is matched in full and no other
    document's name matches, so an ambiguous paste resolves nothing rather
    than picking.
    """
    if not config.USE_DATABASE:
        return set()
    q_norm = _alnum_only(_norm_for_match(question or ""))
    if len(q_norm) < _MIN_PASTED_NAME_CHARS:
        return set()

    try:
        from sqlalchemy import text as _sql
        with _db.get_engine().connect() as conn:
            rows = conn.execute(_sql(
                "SELECT source_doc FROM documents WHERE wiki_id = :w AND session_id = :s"
            ), {"w": _active_wiki_id(), "s": session_id}).fetchall()
    except Exception as e:
        logger.error("resolve_scope: display-name lookup failed: %s", e)
        return set()

    hits: list[tuple[int, str]] = []
    for (sd,) in rows:
        if not sd:
            continue
        # Three spellings of the same document, because three are shown: the
        # normalised name used in References, the folder/basename pair the
        # Files tab and answers display, and the bare basename on its own.
        candidates = {_norm_doc_name(sd)}
        base = sd.split("_", 1)[-1]
        candidates.add(base)
        candidates.add(base.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
        # What the screen actually shows drops the ingest folder prefix, so a
        # paste of the displayed name matches none of the stored spellings
        # above. Derived by string surgery rather than by calling doc_paths for
        # every document: this runs over the whole corpus on every question,
        # and a per-document display() call made resolve_scope take minutes.
        # Two more spellings, both of which a user really pastes: the display
        # form with the ingest folder gone but the category folder kept
        # ("Consulting Agreement / Consultancy Agreement - 2024-11-17 (2)"),
        # and the bare filename. Extensions are stripped from every candidate
        # because the screen shows ".pdf" and a paste usually drops it.
        candidates.add(_RX_INGEST_ONLY_PREFIX.sub("", base))
        candidates.add(_RX_INGEST_FOLDER_PREFIX.sub("", base))
        candidates.add(sd.split("_")[-1])
        for cand in list(candidates):
            no_ext = _RX_DOC_EXTENSION.sub("", cand or "")
            if no_ext and no_ext != cand:
                candidates.add(no_ext)
        for cand in candidates:
            c_norm = _alnum_only(_norm_for_match(cand or ""))
            if len(c_norm) >= _MIN_PASTED_NAME_CHARS and c_norm in q_norm:
                hits.append((len(c_norm), sd))
                break

    if not hits:
        return set()
    # Longest match wins only when it is unambiguous: two documents whose names
    # both appear in full is a paste naming two documents, which is not a case
    # to guess on.
    best = max(h[0] for h in hits)
    winners = {sd for ln, sd in hits if ln == best}
    if len(winners) != 1:
        # Near-duplicates of one document (an OCR twin, a "(1)" copy) are the
        # common case here and are all genuinely named, so keep them; anything
        # wider than that is ambiguous.
        if len(winners) > 3:
            return set()
    logger.info("Display-name paste resolved %d document(s): %s",
                len(winners), {_norm_doc_name(w) for w in winners})
    return winners


def _resolve_docs_by_single_named_instrument(question: str, session_id: str,
                                             max_docs: int = _SINGLE_NAMED_MAX_DOCS) -> set[str]:
    """One document named by nickname plus instrument type, with no other signal.

    The multi-name resolver below requires TWO names, because it exists to
    answer "compare the Amberline NDA and the Apex Cobalt NDA". A question
    naming ONE document that way had no resolver at all, so "the total contract
    value of the Palladion Global purchase agreement" fell through to an
    unscoped corpus search while the same question with a date attached
    resolved immediately. That gap made a date feel mandatory in ordinary
    phrasing, which is a query language with extra steps.

    Deliberately runs LAST, after every party, matter-reference and date signal
    has already passed. It can only turn a fall-through into a resolution,
    never redirect one that already worked — the same constraint the
    Calculation Agent's identifier fallback carries, and for the same reason.

    Resolved the way the multi-name resolver resolves each of its names: a
    content search for the nickname, narrowed to whichever of those documents
    also carries the stated instrument word in its own page titles. Returns
    nothing rather than guessing when the result is empty or too broad.
    """
    if not config.USE_DATABASE:
        return set()

    candidates: list[tuple[str, str]] = []
    for m in _SINGLE_NAMED_INSTRUMENT_RE.finditer(question or ""):
        name = m.group(1).strip()
        head = name.split()[0].lower() if name.split() else ""
        if head in _NOT_A_NICKNAME or len(name) < 4:
            continue
        kind = m.group(0)[m.end(1) - m.start():].strip()
        candidates.append((name, kind))
    if len(candidates) != 1:
        # Zero means the question named no document this way. More than one is
        # the multi-name resolver's job, and it has already had its turn.
        return set()

    name, kind = candidates[0]
    try:
        content_docs = {d for d in _db.find_source_docs_mentioning_phrase(
            _active_wiki_id(), session_id, name, cap=40) if d}
    except Exception as e:
        logger.error("resolve_scope: single-named-instrument content lookup failed "
                     "for %r: %s", name, e)
        return set()
    if not content_docs:
        return set()

    kind_word = re.sub(r"\s+", " ", kind).strip().rstrip("s")
    try:
        title_docs = {d for d in _db.find_source_docs_by_title_tokens(
            _active_wiki_id(), session_id, [kind_word], cap=2000) if d}
    except Exception as e:
        logger.error("resolve_scope: single-named-instrument title lookup failed "
                     "for %r: %s", kind_word, e)
        return set()

    narrowed = (content_docs & title_docs) if title_docs else content_docs
    if not narrowed or len(narrowed) > max_docs:
        return set()
    logger.info("Single named instrument %r + %r -> %d document(s): %s",
                name, kind_word, len(narrowed), {_norm_doc_name(d) for d in narrowed})
    return narrowed


def _resolve_docs_by_named_instruments(question: str, session_id: str,
                                       max_docs: int = 6) -> set[str]:
    """Resolve a question naming several documents by nickname, each with no
    corporate suffix to anchor on — see the note above _NAMED_INSTRUMENT_RE.

    Each "<Name> <kind>" mention is resolved independently: a content search
    on the name, narrowed to whichever of those candidates ALSO carries the
    stated kind word in its own page titles. Accepts the whole result only
    when every distinct name resolves to a real, non-empty cluster and no two
    names collapse onto the same document — otherwise this is either not
    actually a multi-document question or genuinely ambiguous, and the
    existing single-name resolver is left to make its own, narrower call.
    """
    if not config.USE_DATABASE:
        return set()
    seen_names: list[str] = []
    kinds: dict[str, str] = {}
    for m in _NAMED_INSTRUMENT_RE.finditer(question):
        name, kind = m.group(1).strip(), m.group(2).strip()
        if name.lower() not in {n.lower() for n in seen_names}:
            seen_names.append(name)
            kinds[name] = kind
    if len(seen_names) < 2:
        return set()

    resolved: list[set[str]] = []
    for name in seen_names:
        try:
            content_docs = {d for d in _db.find_source_docs_mentioning_phrase(
                _active_wiki_id(), session_id, name, cap=20) if d}
        except Exception as e:
            logger.error("resolve_scope: named-instrument content lookup failed for %r: %s", name, e)
            return set()
        if not content_docs:
            return set()
        kind_word = re.sub(r'\s+', ' ', kinds[name]).rstrip('s')
        try:
            # Uncapped in effect (2000 comfortably exceeds this corpus's total
            # document count): the result is intersected with content_docs
            # below, which is already small, so a low cap here would only
            # truncate the WRONG set — confirmed live, a cap of 60 silently
            # excluded the one real NDA document this exact search needed,
            # since a common instrument word like "NDA" alone matches
            # hundreds of titles corpus-wide.
            title_docs = {d for d in _db.find_source_docs_by_title_tokens(
                _active_wiki_id(), session_id, [kind_word], cap=2000) if d}
        except Exception as e:
            logger.error("resolve_scope: named-instrument title lookup failed for %r: %s", name, e)
            return set()
        narrowed = (content_docs & title_docs) if title_docs else content_docs
        if not narrowed or len(narrowed) > max_docs:
            return set()
        resolved.append(narrowed)

    if len(set.union(*resolved)) != sum(len(r) for r in resolved):
        # Two different nicknames landed on the same document — either the
        # same document was named twice, or the nicknames aren't actually
        # distinct enough to trust; either way, not a case to guess on.
        return set()

    union: set[str] = set().union(*resolved)
    logger.info("Named-instrument list %s → %d document(s): %s",
                seen_names, len(union), {_norm_doc_name(d) for d in union})
    return union


# Ceiling on how many documents an umbrella party may hit before its doc set is
# treated as unusable even for intersection. Generous on purpose — the set is
# only ever intersected with a family below, never used on its own — but bounded,
# because a name matching hundreds of documents is truncated by the query cap and
# an intersection against a truncated list is arbitrary rather than wrong-looking.
_PARTY_FAMILY_SCAN_CAP = 80

# How many documents the party-and-instrument intersection may resolve to before
# it stops being an answerable set. One is a clean resolution; a handful is a
# genuine ambiguity the answer reports per document; more than this and nobody
# can read the result, so scope widens to the family as it did before.
_PARTY_IN_FAMILY_MAX_DOCS = 4


def _resolve_party_within_family(question: str, session_id: str,
                                 fam_docs: set[str]) -> set[str]:
    """Resolve an UMBRELLA party name against the family the question names.

    _resolve_docs_by_party deliberately gives up on a name that hits more than a
    handful of documents ("Tata Steel Limited", "Tata Sons Private Limited"),
    because on its own such a name cannot identify one document. But the question
    usually supplies a second constraint the resolver never spends: the
    INSTRUMENT. Neither signal is decisive alone; their intersection routinely is.

    Measured on Q21 ("the termination clause of Service Agreement of Tata Steel
    Limited"): "Tata Steel Limited" appears in 7 documents — over the standalone
    resolver's cap of 4, so it returned nothing — while the Service Agreement
    family holds 62. The intersection is exactly ONE document, Service Agreement 7,
    the correct one. Without this, scope fell through to the whole 62-document
    family flagged broad, retrieval diversified one page per document, and the
    single page of SA 7 that survived was about confidentiality; the answer then
    reported that no termination grounds existed when the document lists seven.

    Returns the smallest non-empty intersection across the question's party-name
    candidates, or an empty set when nothing intersects or the name is so common
    that its document list came back truncated (see _PARTY_FAMILY_SCAN_CAP).
    """
    if not config.USE_DATABASE or not fam_docs:
        return set()
    candidates = [m.group(1).strip() for m in _PARTY_NAME_RE.finditer(question)]
    candidates = [c for c in candidates if len(c) >= 4]
    if not candidates:
        return set()
    best: set[str] | None = None
    for name in candidates:
        try:
            docs = _db.find_source_docs_mentioning_phrase(
                _active_wiki_id(), session_id, name, cap=_PARTY_FAMILY_SCAN_CAP + 1)
        except Exception as e:
            logger.error("resolve_scope: party-in-family lookup failed for %r: %s", name, e)
            continue
        if not docs or len(docs) > _PARTY_FAMILY_SCAN_CAP:
            continue
        hit = {d for d in docs if d in fam_docs}
        if hit and (best is None or len(hit) < len(best)):
            best = hit
    return best or set()


# A quoted proper noun the question uses to name the SUBJECT of a dispute
# rather than its second party ("operators of 'Tata Restart' and related
# websites"). Matches straight and curly quote pairs.
_QUOTED_PHRASE_RE = re.compile(
    r"['‘’\"“”]([A-Za-z][A-Za-z0-9 &.\-]{2,40})['‘’\"“”]"
)


def _narrow_by_quoted_subject(question: str, session_id: str,
                              candidate_docs: set[str]) -> set[str]:
    """Narrow a document set using a quoted subject the question names.

    Court-document questions routinely distinguish between many judgments
    that share the same plaintiff ("Tata Sons Private Limited" is the
    plaintiff in every Tata Brand Judgment) by naming the infringing subject
    in quotes instead of a second party — "the suit ... against operators of
    'Tata Restart' and related websites" has no second capitalised party for
    _resolve_docs_by_party_pair to catch, so party resolution alone leaves
    every judgment in the family equally plausible.

    Ingest folds that subject into the matter's title short-name with spaces
    stripped ("TataRestart"), so the title ILIKE search below collapses the
    quoted phrase the same way — searching for "Tata Restart" WITH the space
    would match nothing.

    Confirmed on Q100: "Tata Sons Private Limited" alone spans every document
    in the Judgments family, so party-in-family resolution returns the whole
    family and the answer LLM guesses (measured: it picked Judgment 2 when
    the case number sat in Judgment 7). The quoted "'Tata Restart'" pins the
    one document whose title carries that short-name.

    Returns the first quoted phrase's hits (intersected with candidate_docs)
    that resolve to at least one document, or an empty set if no quoted
    phrase in the question matches anything.
    """
    if not config.USE_DATABASE or not candidate_docs:
        return set()
    for m in _QUOTED_PHRASE_RE.finditer(question or ''):
        phrase = m.group(1).strip()
        collapsed = re.sub(r'[^A-Za-z0-9]', '', phrase)
        if len(collapsed) < 4:
            continue
        try:
            docs = set(_db.find_source_docs_by_title_tokens(
                _active_wiki_id(), session_id, [collapsed], cap=25))
        except Exception as e:
            logger.error("resolve_scope: quoted-subject lookup failed for %r: %s",
                         phrase, e)
            continue
        hit = docs & candidate_docs
        if hit:
            return hit
    return set()


# Document-type words a question uses to single out ONE instrument between two
# parties, mapped to the parenthetical ingest appends to that document's page
# titles ("… – Aether-Helios (Verified Complaint)"). Ordered most-specific
# first: "amended complaint" and "verified complaint" must both be tested
# before the bare "complaint" that each of them contains, or the general
# pattern would claim the phrase and point at the wrong pleading.
_TITLE_KIND_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'amended\s+complaint', re.I),                  'Amended Complaint'),
    (re.compile(r'verified\s+complaint', re.I),                 'Verified Complaint'),
    # Both instruments of one dispute carry the same two party names, so the
    # party-pair lookup returns them together and only the instrument the
    # question names tells them apart. Confirmed on Q24, which asks about the
    # "Notice Invoking Arbitration" and was answered from the Section 9 petition
    # sitting beside it — preservation relief instead of the pleaded breaches.
    (re.compile(r'section\s+9\s+petition|petition\s+under\s+section\s+9', re.I),
                                                                'Section 9 Petition'),
    (re.compile(r'notice\s+invoking\s+arbitration|arbitration\s+notice', re.I),
                                                                'Arbitration Notice'),
    (re.compile(r'written\s+statement', re.I),                   'Written Statement'),
    (re.compile(r'opposition\s+brief', re.I),                   'Opposition Brief'),
    (re.compile(r'reply\s+brief', re.I),                        'Reply Brief'),
    (re.compile(r'(?:court\s+)?transcript', re.I),              'Transcript'),
    (re.compile(r'\baffidavit\b', re.I),                        'Affidavit'),
    (re.compile(r'settlement\s+agreement', re.I),               'Settlement'),
    (re.compile(r'preliminary\s+injunction|injunction\s+motion', re.I), 'Injunction'),
    (re.compile(r'judg[e]?ment', re.I),                         'Judgment'),
    (re.compile(r'joint\s+venture', re.I),                      'Joint Venture'),
    (re.compile(r'shareholders?\s+agreement', re.I),            'Shareholder'),
    (re.compile(r'non[-\s]?disclosure|\bnda\b', re.I),          'NDA'),
    (re.compile(r'services?\s+agreement', re.I),                'Services Agreement'),
    (re.compile(r'\bcounterclaims?\b|\banswer\b', re.I),        '(Answer)'),
    (re.compile(r'\bcomplaint\b', re.I),                        'Complaint'),
]

# A question can describe a clause by SUBJECT MATTER instead of naming the
# instrument that carries it ("the security incident notification
# obligations", never "the NDA"). _TITLE_KIND_HINTS only ever matches an
# instrument named outright, so it has nothing to fire on here — and a real
# M&A-style deal file can run to 15+ documents between the same two parties
# (NDA, Term Sheet, SPA, Disclosure Letter, KERA, TSA, IP Assignment, Escrow,
# Tax Deed, ...), too many for any resolver to return outright or narrow by
# filename tokens alone. Used only as a last-resort narrower in
# ``_content_pair_supplement`` when its own candidate pool is too large to
# return, and only in the SAME ILIKE-against-title-or-filename way
# ``kind_hint`` already narrows an explicit instrument mention — so a
# document ingest happened to title only by one party's name (confirmed
# live: an NDA's own page titles all read "NDA-Apex Zephyra", never
# "Nimbus", because ingest's short-naming picked one party) is still found,
# since the instrument-type word survives in that title regardless of which
# party's name it carries.
_SUBJECT_KIND_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'security\s+incident|data\s+breach|confidential(?:ity|\s+information)|'
                r'non[-\s]?disclosure', re.I),                          'NDA'),
    (re.compile(r'key\s+employee|retention\s+(?:bonus|payment|agreement)', re.I), 'KERA'),
    (re.compile(r'\bescrow\b', re.I),                                   'Escrow'),
    (re.compile(r'transition\s+services?', re.I),                       'TSA'),
    (re.compile(r'intellectual\s+property\s+assign|\bip\s+assign', re.I), 'IP Assign'),
    (re.compile(r'closing\s+certificate', re.I),                        'ClosCert'),
    (re.compile(r'disclosure\s+letter', re.I),                          'Disclosure'),
    (re.compile(r'conditions?\s+precedent', re.I),                      'Conditions Precedent'),
]

# Corporate-form and generic descriptor words that are NOT the distinctive part
# of a party name. Ingest coins its matter short-name from the first
# distinctive word ("Aether Technologies Inc." → "Aether"), so stripping these
# leaves the token that actually appears in the page titles.
_PARTY_GENERIC_WORDS = frozenset({
    'private', 'limited', 'ltd', 'pvt', 'pte', 'llp', 'llc', 'inc', 'corp',
    'corporation', 'plc', 'gmbh', 'company', 'co', 'group', 'holdings',
    'technologies', 'technology', 'systems', 'solutions', 'services',
    'energy', 'industries', 'international', 'global', 'partners', 'ventures',
    # This corpus's own conglomerate prefixes ("Apex Sagar Mobility", "Tata
    # Projects", "Apex Zephyra Trading Company") — real-world-style umbrella
    # names that head dozens of unrelated subsidiary parties, not a single
    # party's identity. Confirmed live, twice: "Apex Zephyra Trading Company"
    # reduced to token "Apex" — the single most common word in the whole
    # corpus — and party-pair resolution over- or under-matched on it in both
    # directions (a real Tax Deed excluded from its title-search cluster
    # entirely; a real Transition Services Agreement never found at all).
    # Ingest's own short-titling already drops these ("Sagar-Ashoka", "TPL"),
    # this just catches the token extractor up to match.
    'apex', 'tata',
})


def _distinctive_party_token(name: str) -> str:
    """The one word of a party name that identifies the party.

    "Aether Technologies Inc." → "Aether"; "Helios Energy Corporation" →
    "Helios". Ingest builds each document's matter short-name from this same
    leading distinctive word, which is what makes the token matchable against
    page titles. Returns "" when nothing distinctive survives (a name made
    entirely of generic words), so the caller can skip it rather than search
    for a word that would match half the corpus.
    """
    for word in re.split(r'[^A-Za-z0-9]+', name or ''):
        if len(word) >= 3 and word.lower() not in _PARTY_GENERIC_WORDS:
            return word
    return ""


# Capitalised words that are NOT party names, used to filter the bare-name
# fallback below. Question openers, forum/procedure vocabulary, and instrument
# abbreviations all get capitalised in ordinary legal phrasing ("in the Court
# of Chancery", "breach of the JVA") and would otherwise be mistaken for the
# second party of a pair.
_QUESTION_COMMON_WORDS = frozenset({
    'what', 'who', 'whom', 'whose', 'which', 'when', 'where', 'why', 'how',
    'the', 'this', 'that', 'these', 'those', 'and', 'but', 'for', 'from',
    'did', 'does', 'was', 'were', 'has', 'have', 'are', 'its', 'their',
    'court', 'chancery', 'state', 'delaware', 'district', 'supreme', 'high',
    'civil', 'criminal', 'action', 'suit', 'lawsuit', 'litigation', 'case',
    'matter', 'section', 'clause', 'schedule', 'exhibit', 'annexure',
    'agreement', 'contract', 'deed', 'amendment', 'addendum',
    'jva', 'nda', 'sha', 'sow', 'msa', 'spa', 'ira', 'llp', 'llc',
    'plaintiff', 'defendant', 'petitioner', 'respondent', 'appellant',
    'complaint', 'answer', 'counterclaim', 'counterclaims', 'defense',
    'defence', 'defenses', 'defences', 'motion', 'brief', 'affidavit',
    'judgment', 'judgement', 'order', 'decree', 'verdict', 'counsel',
    'esq', 'inc', 'ltd', 'corp', 'plc', 'gmbh', 'pvt', 'pte',
})

# An adversarial caption names both sides with no corporate suffix on either
# ("the Aether v. Helios litigation"). _PARTY_NAME_RE's suffix anchor cannot
# see them, so the pair resolver would find at most one party and give up.
_CASE_CAPTION_RE = re.compile(
    r'\b([A-Z][A-Za-z0-9&.\-]{2,})\s+(?:v\.?|vs\.?|versus)\s+([A-Z][A-Za-z0-9&.\-]{2,})\b'
)

# The other two ways a question puts two parties in an explicit relationship
# without captioning them: a claim asserted "against" the other side, and an
# instrument struck "between" them. Both keep the second party in the pattern;
# the first is recovered by scanning backwards (see _bare_party_tokens).
_AGAINST_RE = re.compile(r'\bagainst\s+([A-Z][A-Za-z0-9&.\-]{2,})')
_BETWEEN_AND_RE = re.compile(
    r'\bbetween\s+([A-Z][A-Za-z0-9&.\-]{2,})(?:[^.?!]{0,60}?)\s+and\s+([A-Z][A-Za-z0-9&.\-]{2,})'
)

# A directional obligation named "of X to Y" — "the notification obligations
# of Nidra Bhandari to Apex Suvarna...". Only the first (usually unsuffixed)
# party needs recovering here: the second routinely carries a corporate
# suffix and is already caught by _PARTY_NAME_RE. Confirmed live: a natural
# person named only this way ("Nidra Bhandari", no corporate suffix) left a
# 4-party compound comparison with just 3 _PARTY_NAME_RE hits — an odd count
# that made _resolve_docs_by_combinatorial_pairing decline outright, and the
# question fell back to independent per-name scoring, which silently missed
# the other pair's actual document entirely.
_OF_TO_RE = re.compile(
    r'\bof\s+([A-Z][A-Za-z0-9&.\-]{2,})\b(?=(?:\s+[A-Z][A-Za-z0-9&.\-]{2,}){0,3}\s+to\s+[A-Z])'
)

_CAPITALISED_WORD_RE = re.compile(r'\b[A-Z][A-Za-z0-9&.\-]*\b')


def _is_party_like(word: str) -> bool:
    """Could this capitalised word be a party's short-name?"""
    w = (word or '').rstrip('.').lower()
    return (len(w) >= 3
            and w not in _QUESTION_COMMON_WORDS
            and w not in _PARTY_GENERIC_WORDS)


def _bare_party_tokens(question: str) -> list[str]:
    """Party short-names a question states WITHOUT a corporate suffix.

    Complements ``_distinctive_party_token``, which only ever sees names
    carrying a corporate suffix. Lawyers drop the suffix once a matter is
    under discussion ("what damages did Helios claim against Aether"), and an
    adversarial caption never carries one at all.

    Only words standing in an EXPLICIT two-party relationship count — "X v.
    Y", a claim "against Y", an instrument "between X and Y". An earlier
    version simply harvested every capitalised word that was not a common
    legal term, on the theory that the caller's title-match would filter the
    rest; it does not. Capitalised topic words co-occur in page titles just
    fine, so "Summarize the Joint Venture Agreement" resolved <Joint, Venture>
    and pinned two arbitrary JVAs instead of scoping to the family, and "What
    are Reserved Matters and Board Approval thresholds?" pinned an unrelated
    judgment. Requiring a relational construction is what separates naming two
    PARTIES from naming one TOPIC in title case.
    """
    tokens: list[str] = []
    seen: set[str] = set()

    def add(word: str) -> None:
        w = (word or '').rstrip('.')
        if w and w.lower() not in seen and _is_party_like(w):
            seen.add(w.lower())
            tokens.append(w)

    for m in _CASE_CAPTION_RE.finditer(question or ''):
        add(m.group(1))
        add(m.group(2))
    for m in _BETWEEN_AND_RE.finditer(question or ''):
        add(m.group(1))
        add(m.group(2))
    for m in _AGAINST_RE.finditer(question or ''):
        add(m.group(1))
        # The claimant is whatever party-like name last appeared before
        # "against" — "damages did HELIOS claim in its counterclaim … against
        # Aether". Taking the nearest one avoids latching onto the sentence's
        # capitalised opening word ("What", "Who"), which is grammar.
        prior = [w for w in _CAPITALISED_WORD_RE.findall(question[:m.start()])
                 if _is_party_like(w)]
        if prior:
            add(prior[-1])
    for m in _OF_TO_RE.finditer(question or ''):
        add(m.group(1))
    return tokens


def _content_pair_supplement(session_id: str, tokens: list[str], full_names: list[str],
                             cluster: set[str], question: str, max_docs: int) -> set[str]:
    """Find a sibling instrument title search missed because ingest titled it
    under an arbitrary code name instead of either party's name.

    Confirmed live: a real Term Sheet and a real Share Purchase Agreement are
    both between "Tata Projects" and "Bhumika Motors". The SPA's every page
    title reads "... – Tata Projects/Bhumika (Share Purchase Agreement)" — a
    party-derived short name, so title search finds it. The Term Sheet's
    pages read "... – Matter Blue (Term Sheet)" — an arbitrary code name
    containing neither party — so title search cannot find it no matter what
    tokens it tries; the document is real and correctly indexed, it just
    isn't titled by party name. Full-text content confirms "Bhumika Motors"
    is genuinely discussed throughout it.

    _resolve_docs_by_party_pair's own docstring explains why it uses titles
    instead of raw content-intersection in the first place: on the whole
    corpus, intersecting two party names by content is too noisy (76 of ~115
    documents on the adversarial-litigation case that motivated it). This
    avoids that by anchoring on the RARER of the two party tokens alone (one
    content search, not an intersection), subtracting what title search
    already found, narrowing what's left by filename
    (_narrow_by_question_tokens — instrument type, document code), and only
    THEN, on that already-small remainder, verifying each survivor's own
    CONTENT for the other party token too. Content-intersecting a handful of
    pre-narrowed candidates is a different cost/noise profile than
    content-intersecting the whole corpus.

    Returns an empty set unless the anchor token resolves to something
    genuinely outside `cluster`, and that remainder narrows (by filename,
    then by content-confirming the other party) to no more than max_docs.

    ``full_names`` are the fuller party phrases the single-word ``tokens``
    were each reduced from ("Tata Projects", not just "Tata") — the final
    content-verification step needs that fuller phrase, not the bare word;
    "Tata" alone appears in most of this corpus and confirms nothing.
    """
    if not config.USE_DATABASE or len(tokens) < 2:
        return set()
    anchor_tok, anchor_full, other_tok, anchor_docs = None, None, None, None
    for i, t in enumerate(tokens[:2]):
        try:
            docs = set(_db.find_source_docs_mentioning_phrase(_active_wiki_id(), session_id, t, cap=100) or [])
        except Exception:
            docs = set()
        if docs and (anchor_docs is None or len(docs) < len(anchor_docs)):
            anchor_docs, anchor_tok = docs, t
            # The anchor's own FULL name, not just its bare distinctive token —
            # see the `spent` fix below for why this matters.
            anchor_full = (full_names[i] if i < len(full_names) else t)
            other_tok = (full_names[1 - i] if len(full_names) > 1 else None) or (tokens[1 - i] if len(tokens) > 1 else None)
    if not anchor_docs:
        return set()
    remainder = anchor_docs - cluster
    if not remainder:
        return set()

    # Content-verify against the OTHER party first, before any filename
    # narrowing — cheap here (remainder is already small, unlike the whole
    # corpus) and it must come first: both party names' own words are
    # legitimately present in EVERY sibling instrument's filename too (a
    # deal's Escrow, SPA, and Term Sheet filenames all say "Tata Projects"),
    # so if filename-narrowing runs first and includes party-name tokens,
    # those tokens discriminate toward whichever sibling happens to spell
    # the party out in ITS OWN filename — a real document, but not
    # necessarily the one the question's instrument type names. Confirmed
    # live: "tata"/"projects" alone pointed at the Escrow sibling (whose
    # filename literally reads "Tata Projects - Escrow - Agreement"), which
    # disagreed with "term"/"sheet" pointing at the real Term Sheet, and the
    # two cancelled out.
    if other_tok:
        try:
            other_docs = set(_db.find_source_docs_mentioning_phrase(_active_wiki_id(), session_id, other_tok, cap=200) or [])
        except Exception:
            other_docs = set()
        content_verified = remainder & other_docs
    else:
        content_verified = remainder
    if not content_verified:
        return set()

    # NOW narrow by filename — but the instrument type is the only signal
    # left to ask for; both party phrases already did their job above and
    # would just recreate the same cancel-out if left in the token set.
    #
    # Excludes each party's FULL name, not just the bare anchor token it was
    # reduced to for the content search above. Confirmed live: anchor_tok
    # "Nimbus" alone left "Capital" (from "Nimbus Capital") in the narrowing
    # pool as a live token — it matched 5 unrelated sibling documents, which
    # disagreed with "tax"/"deed" (which correctly, uniquely matched the real
    # Tax Deed) and cancelled the narrowing out via the empty-intersection
    # guard, silently returning the whole unnarrowed set instead of the one
    # real answer.
    spent = " ".join(n for n in (anchor_full, other_tok) if n)
    narrowed = _narrow_by_question_tokens(question, content_verified, exclude=spent)
    if narrowed and len(narrowed) <= max_docs:
        return narrowed
    if len(content_verified) <= max_docs:
        return content_verified

    # Still too large, and the question named no instrument by TYPE (that
    # would already have narrowed above) — try what it names by SUBJECT
    # instead. Matches the kind_hint word against each candidate's own
    # filename directly (no DB round-trip needed: ingest's document-type
    # word survives in the filename regardless of which party's name that
    # filename happens to carry — see _SUBJECT_KIND_HINTS).
    for rx, hint in _SUBJECT_KIND_HINTS:
        if not rx.search(question):
            continue
        by_subject = {d for d in content_verified if hint.lower() in (d or '').lower()}
        if by_subject and len(by_subject) <= max_docs:
            logger.info("Content-supplement candidates %d narrowed by subject %r → %d document(s): %s",
                        len(content_verified), hint, len(by_subject),
                        {_norm_doc_name(d) for d in by_subject})
            return by_subject
        break
    return set()


_BETWEEN_RE = re.compile(r'\bbetween\b', re.I)

# What joins one named instrument to the NEXT one in a question that names
# several ("... dated 25 December 2019 AND THE Key Employee Retention Agreement
# between ...", "... dated 05 July 2024 AS STATED IN THE Share Subscription
# Agreement between ..."). Everything after this marker introduces the next
# instrument, so it belongs to the next pair's sub-question, not this one.
# Requiring "the" after the connector is what keeps it from splitting on the
# "and" that joins the two parties of a single pair ("... Limited and Ashoka
# Travel Limited").
_PAIR_CONNECTOR_RE = re.compile(
    r'\b(?:and|or|as\s+(?:stated|set\s+out|provided|described|defined)\s+in|'
    r'versus|vs\.?|compared\s+(?:to|with))\s+the\b',
    re.I,
)


def _question_pair_segments(question: str) -> list[str]:
    """Split a question naming SEVERAL party-pairs into one sub-question per pair.

    A comparison question names two whole matters at once — "compare the
    governing law of the KERA between Apex Sagar Mobility Limited and Ashoka
    Travel Limited dated 25 December 2019 and the KERA between Apex Prisha
    Motors Limited and Northfield Mobility Private Limited dated 28 April
    2021". Every party-name detector in this module reads the question as one
    flat list of names, so the pair resolver below sees FOUR parties, truncates
    to three, and requires all three in a single document title. Nothing has
    all three, so it either matches nothing or — confirmed live on that exact
    question — matches two unrelated Apex Sagar/Ashoka Travel instruments (an
    Escrow Agreement and an SPA) while retrieving neither KERA the question
    actually asked about.

    Each returned segment is the question's head (which carries the instrument
    type and the task verb) plus one "between …" span, so the pair's own date
    and its own instrument words narrow it without the other pair's date
    cancelling them out.

    Returns [] unless there are 2+ "between" spans AND each one names two
    parties — one pair, or prose that merely uses the word, stays on the
    ordinary single-pair path.
    """
    starts = [m.start() for m in _BETWEEN_RE.finditer(question)]
    if len(starts) < 2:
        return []
    segments: list[str] = []
    # The instrument type sits BEFORE its "between", so each span's own head is
    # whatever preceded it: the question's opening for the first pair, and for
    # every later pair the tail the previous span handed over at its connector.
    head = question[:starts[0]]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(question)
        span = question[start:end]
        body, tail = span, ""
        if i + 1 < len(starts):
            cuts = list(_PAIR_CONNECTOR_RE.finditer(span))
            if cuts:
                body, tail = span[:cuts[-1].start()], span[cuts[-1].start():]
        if len(_PARTY_NAME_RE.findall(body)) < 2:
            return []
        segments.append(head + body)
        head = tail
    return segments


def _resolve_docs_by_party_pair(question: str, session_id: str,
                                max_docs: int = 6) -> set[str]:
    """Resolve documents named by their parties, one pair or several.

    Compound questions are resolved pair-by-pair and unioned; everything else
    goes straight to the single-pair resolver below.
    """
    if not config.USE_DATABASE:
        return set()
    segments = _question_pair_segments(question)
    if not segments:
        try:
            combinatorial = _resolve_docs_by_combinatorial_pairing(question, session_id, max_docs)
        except Exception as e:
            logger.error("resolve_scope: combinatorial party-pairing failed: %s", e)
            combinatorial = set()
        if combinatorial:
            return combinatorial
        return _resolve_one_party_pair(question, session_id, max_docs)

    union: set[str] = set()
    resolved = 0
    for segment in segments:
        try:
            got = _resolve_one_party_pair(segment, session_id, max_docs)
        except Exception as e:
            logger.error("resolve_scope: compound party-pair leg failed: %s", e)
            got = set()
        if got:
            union |= got
            resolved += 1
    if resolved == len(segments) and union:
        logger.info("Compound party-pair question: all %d pairs resolved → %d document(s): %s",
                    len(segments), len(union), {_norm_doc_name(d) for d in union})
        return union
    # Deliberately NOT falling back to the flat single-pair path: with four
    # party names in one question that path ANDs three of them together and
    # returns whatever coincidence survives. Returning nothing lets the weaker
    # but honest family/broad signals downstream handle it instead.
    logger.info("Compound party-pair question: only %d of %d pairs resolved — "
                "declining to scope on a partial pair set", resolved, len(segments))
    return set()


def _resolve_one_party_pair(question: str, session_id: str,
                            max_docs: int = 6) -> set[str]:
    """Resolve the documents of a matter named by BOTH of its parties.

    ``_resolve_docs_by_party`` above scores each party name INDEPENDENTLY and
    keeps the single most distinctive one — it never intersects them. That is
    the right call for "the JV with Cold Chain Energy Services", where one
    party is the whole signal, but it cannot resolve an adversarial pair:
    asked about "the lawsuit filed by Aether Technologies Inc. against Helios
    Energy Corporation", each name alone spans ~115 documents, both exceed the
    max_docs cap, and the function returns nothing. Scope then falls through
    to an unscoped corpus search.

    Confirmed live, and the reason this exists: that exact question answered
    "not covered" for a civil action number sitting in the retrieved corpus,
    and the companion question about plaintiff's counsel cited an unrelated
    NDA between entirely different parties. Both documents were indexed and
    both contained the answer verbatim — retrieval simply never scoped to them.

    Intersecting on page CONTENT does not fix this, because litigation
    documents recite the opposing side's officers in their discovery
    paragraphs: on this corpus a content-intersection of the two names still
    matched 76 of ~115 documents. Intersecting on page TITLES does, because
    ingest writes the matter's own short-name into every title it creates —
    that drops the same corpus to the 20 instruments actually between those
    parties, and adding the document-type word the question already supplies
    ("the verified complaint") pins it to exactly one.

    Returns an empty set unless at least two distinct parties are named AND
    the resolved cluster is non-empty and within ``max_docs`` — so it only
    ever ADDS precision where scope currently resolves to nothing at all.
    """
    if not config.USE_DATABASE:
        return set()
    names = [m.group(1).strip() for m in _PARTY_NAME_RE.finditer(question)]
    tokens: list[str] = []
    # Parallel to tokens — the fuller phrase each single-word token was
    # reduced from ("Tata Projects" for token "Tata"). _content_pair_supplement
    # needs the fuller phrase for content-verification; the bare word alone
    # is too common in this corpus to confirm anything.
    token_full: dict[str, str] = {}
    for n in names:
        tok = _distinctive_party_token(n)
        if tok and tok.lower() not in {t.lower() for t in tokens}:
            tokens.append(tok)
            token_full[tok] = n
    if len(tokens) < 2:
        # Only one side carried a corporate suffix (or neither did). Fall back
        # to bare capitalised short-names, which is how a matter gets referred
        # to once it is under discussion ("the Aether v. Helios litigation",
        # "what damages did Helios claim against Aether"). Suffix-derived
        # tokens stay first so the strongest signal still leads.
        for tok in _bare_party_tokens(question):
            if tok.lower() not in {t.lower() for t in tokens}:
                tokens.append(tok)
                token_full[tok] = tok
    if len(tokens) < 2:
        return set()
    # More than a handful of capitalised words means this is prose, not a
    # two-party reference — requiring ALL of them in one title would either
    # match nothing or match by accident.
    tokens = tokens[:3]
    full_names = [token_full.get(t, t) for t in tokens]
    return _resolve_docs_for_tokens(tokens, full_names, question, session_id, max_docs)


def _resolve_docs_for_tokens(tokens: list[str], full_names: list[str], question: str,
                             session_id: str, max_docs: int = 6) -> set[str]:
    """Resolve documents whose title (or, failing that, content) carries every
    one of ``tokens``.

    Split out of ``_resolve_one_party_pair`` so the same title-cluster,
    content-supplement, and narrowing logic can run on a token pair chosen by
    combinatorial pairing (``_resolve_docs_by_combinatorial_pairing``) as well
    as on the flat whole-question token extraction above it.
    """
    try:
        cluster = {d for d in _db.find_source_docs_by_title_tokens(
            _active_wiki_id(), session_id, tokens, cap=max_docs * 5) if d}
    except Exception as e:
        logger.error("resolve_scope: party-pair title lookup failed: %s", e)
        return set()

    # Title search under-recalls when ingest gave a sibling instrument an
    # arbitrary code name instead of a party-derived one — see
    # _content_pair_supplement's docstring for the confirmed live case. Runs
    # even when cluster already found something: the missing sibling doesn't
    # announce itself, so there's no signal to condition this on.
    try:
        supplement = _content_pair_supplement(session_id, tokens, full_names, cluster, question, max_docs)
    except Exception as e:
        logger.error("resolve_scope: party-pair content supplement failed: %s", e)
        supplement = set()
    if supplement:
        logger.info("Party-pair title match %s supplemented by content-verified match: %s",
                    tokens, {_norm_doc_name(d) for d in supplement})
        cluster |= supplement

    if not cluster:
        return set()
    # A cluster small enough to pin outright can still hold SEVERAL instruments
    # of one dispute — the same two parties sign the arbitration notice and the
    # Section 9 petition alike. When the question names exactly one of them, say
    # so before returning: pinning both leaves the answer LLM to choose, and on
    # Q24 it chose the petition and reported preservation relief for a question
    # about the notice's pleaded breaches. Only ever narrows, and only when the
    # question names ONE instrument — a question spanning several ("across the
    # NDA, the notice, and the petition") must keep the whole cluster.
    if len(cluster) > 1 and _count_instrument_mentions(question) <= 1:
        pinned = set()
        for rx, hint in _TITLE_KIND_HINTS:
            if not rx.search(question):
                continue
            try:
                pinned = {d for d in _db.find_source_docs_by_title_tokens(
                    _active_wiki_id(), session_id, tokens, kind_hint=hint, cap=max_docs * 5) if d}
            except Exception:
                pinned = set()
            pinned &= cluster
            if pinned and len(pinned) < len(cluster):
                logger.info("Party-pair title match %s narrowed by the instrument "
                            "named (%r) → %d document(s): %s",
                            tokens, hint, len(pinned),
                            {_norm_doc_name(d) for d in pinned})
                return pinned
            break
        if not pinned:
            # The question names an instrument type this curated list doesn't
            # cover ("Key Employee Retention Agreement", "Tax Deed") — try
            # filename-narrowing the same general mechanism uses everywhere
            # else, excluding the resolved party names so their own words
            # can't cancel out a real instrument-type token the way "Capital"
            # once did (see _content_pair_supplement). Confirmed live: a KERA
            # question's title cluster + content supplement correctly totalled
            # 11 real candidate documents, including the right one — too many
            # to return outright, and no curated kind-hint matched, so it fell
            # all the way through to an empty result instead of narrowing.
            spent = " ".join(full_names)
            filename_narrowed = _narrow_by_question_tokens(question, cluster, exclude=spent)
            if filename_narrowed and len(filename_narrowed) < len(cluster):
                logger.info("Party-pair title match %s narrowed by filename tokens "
                            "→ %d document(s): %s",
                            tokens, len(filename_narrowed),
                            {_norm_doc_name(d) for d in filename_narrowed})
                return filename_narrowed
    if len(cluster) <= max_docs:
        logger.info("Party-pair title match %s → %d document(s): %s",
                    tokens, len(cluster), {_norm_doc_name(d) for d in cluster})
        return cluster

    # Cluster is the whole matter (every instrument between these parties).
    # Narrow it the way the question already distinguishes them — first by the
    # instrument it names outright ("the verified complaint"), then, for a
    # question that names the matter but no instrument ("the LAWSUIT filed by
    # X against Y"), by the document family the phrasing implies.
    for rx, hint in _TITLE_KIND_HINTS:
        if not rx.search(question):
            continue
        try:
            narrowed = {d for d in _db.find_source_docs_by_title_tokens(
                _active_wiki_id(), session_id, tokens, kind_hint=hint, cap=max_docs * 5) if d}
        except Exception:
            narrowed = set()
        if narrowed and len(narrowed) <= max_docs:
            logger.info("Party-pair title match %s + kind %r → %d document(s): %s",
                        tokens, hint, len(narrowed),
                        {_norm_doc_name(d) for d in narrowed})
            return narrowed
        break

    try:
        available = set(_db.list_doc_families(_active_wiki_id(), session_id))
        fam = _detect_question_family(question, available)
        fam_docs = set(_db.get_documents_by_family(_active_wiki_id(), session_id, fam)) if fam else set()
    except Exception as e:
        logger.error("resolve_scope: party-pair family narrowing failed: %s", e)
        fam_docs = set()
    narrowed = cluster & fam_docs
    if narrowed and len(narrowed) <= max_docs:
        logger.info("Party-pair title match %s + family → %d document(s): %s",
                    tokens, len(narrowed), {_norm_doc_name(d) for d in narrowed})
        return narrowed
    return set()


def _pairings(items: list[str]):
    """Every way of grouping ``items`` into disjoint pairs, each yielded once.

    Standard recursive construction: fix the first item, pair it with each of
    the others in turn, and recurse on what's left. For n items this yields
    (n-1)!! matchings — 3 for four items, 15 for six — never duplicating a
    grouping under reordering, since the first item is always the one paired.
    """
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for i, other in enumerate(rest):
        remaining = rest[:i] + rest[i + 1:]
        for tail in _pairings(remaining):
            yield [(first, other)] + tail


def _resolve_docs_by_combinatorial_pairing(question: str, session_id: str,
                                           max_docs: int = 6) -> set[str]:
    """Resolve a question naming two (or three) whole matters that never
    repeats the word "between" for each one.

    ``_question_pair_segments`` only splits a question into per-matter
    segments when "between" introduces EACH pair ("the agreement between A
    and B ... the agreement between C and D"). Real questions often name two
    matters without repeating it — "what does A owe B that isn't in C and
    D's agreement", "can A and C each assign without B or D's consent" — and
    the second phrasing doesn't even keep each matter's two parties adjacent:
    it lists all the "first" parties together, then all the "second" parties,
    so naive positional pairing (1st with 2nd, 3rd with 4th) gets it wrong
    too. Confirmed live: this exact phrasing cost three real cross-document
    questions their second document entirely — each answered "the agreement
    isn't in the provided context" for a document that was in fact ingested,
    indexed, and sitting in the same wiki as the one it did find.

    Instead of parsing the sentence's grammar, this tries every way of
    pairing up the party names the question names at all, resolves each
    candidate pair through the exact same title/content/narrowing logic a
    single explicit pair gets (``_resolve_docs_for_tokens``), and accepts a
    grouping only if EVERY pair in it independently resolves to a real,
    distinct document cluster. Wrong groupings are expected to fail outright
    here, not just score worse — two parties who were never actually
    counterparties on anything share no document, so their pair resolves to
    nothing. If more than one grouping manages to resolve every pair, which
    pair is which is genuinely ambiguous from the names alone — decline
    rather than guess, the same rule the "between"-segment path already
    applies to a partially-resolved split.

    Capped at 6 tokens (three matters): combinatorial cost aside, a question
    naming more than three matters by name alone is rare enough that
    guessing the grouping is riskier than declining outright.
    """
    if not config.USE_DATABASE:
        return set()
    names = [m.group(1).strip() for m in _PARTY_NAME_RE.finditer(question)]
    tokens: list[str] = []
    token_full: dict[str, str] = {}
    for n in names:
        tok = _distinctive_party_token(n)
        if tok and tok.lower() not in {t.lower() for t in tokens}:
            tokens.append(tok)
            token_full[tok] = n
    # One side of a compound comparison can name its party without a
    # corporate suffix (a natural person, or a bare short-name once a matter
    # is under discussion) — _PARTY_NAME_RE alone then hands back an odd
    # count and every matching below is skipped before it's tried. Confirmed
    # live: "the notification obligations of Nidra Bhandari to X ... with
    # those of Y to Z" resolved only 3 suffixed names for a real 4-party
    # question. Supplementing with the same bare-token detectors the
    # single-pair resolver already falls back on recovers the 4th.
    for tok in _bare_party_tokens(question):
        if tok.lower() not in {t.lower() for t in tokens}:
            tokens.append(tok)
            token_full[tok] = tok
    if len(tokens) < 4 or len(tokens) % 2 or len(tokens) > 6:
        return set()

    valid_matchings: list[list[set[str]]] = []
    for matching in _pairings(tokens):
        resolved: list[set[str]] = []
        for a, b in matching:
            got = _resolve_docs_for_tokens(
                [a, b], [token_full.get(a, a), token_full.get(b, b)],
                question, session_id, max_docs)
            if not got:
                resolved = []
                break
            resolved.append(got)
        if not resolved:
            continue
        # Two different pairs landing on the same document is a sign the
        # grouping is wrong — two distinct matters don't share one instrument.
        if len(set.union(*resolved)) != sum(len(r) for r in resolved):
            continue
        valid_matchings.append(resolved)

    if len(valid_matchings) != 1:
        if len(valid_matchings) > 1:
            logger.info("Combinatorial party-pairing: %d groupings of %s all resolved — "
                        "ambiguous, declining", len(valid_matchings), tokens)
        return set()

    union: set[str] = set().union(*valid_matchings[0])
    logger.info("Combinatorial party-pairing: %s → %d document(s): %s",
                tokens, len(union), {_norm_doc_name(d) for d in union})
    return union


# "the original <TYPE> agreement" — names a document by TYPE alone, no party,
# because the question is naming it in contrast to an amendment of it named
# elsewhere in the same question. Non-greedy up to the first instrument-type
# suffix word, so "the original IT Outsourcing Agreement say" captures "IT
# Outsourcing Agreement" rather than running past it into the verb.
_ORIGINAL_TYPE_RE = re.compile(
    r'\boriginal\s+((?:[A-Za-z][A-Za-z&\'/-]*\s+){0,4}?(?:agreement|contract|deed|'
    r'lease|licen[cs]e|nda|mou|memorandum))\b',
    re.IGNORECASE,
)
_AMENDMENT_WORD_RE = re.compile(r'\bamendments?\b', re.IGNORECASE)


def _resolve_original_of_amendment(question: str, session_id: str,
                                   max_docs: int = 4) -> set[str]:
    """Resolve "the original X agreement ... the Y amendment" — a document
    named only by TYPE, referenced opposite an amendment the question names
    by an umbrella party with no corporate suffix at all.

    Confirmed live: "What did the original IT Outsourcing Agreement say about
    payment terms, and how does the Apex Meridian amendment change that?"
    resolved to neither document. "Apex Meridian" carries no suffix
    _PARTY_NAME_RE can anchor on, and it is genuinely ambiguous alone on this
    corpus — Apex Meridian Software, Apex Meridian Mobility, and Apex Meridian
    Travel are three unrelated real entities sharing that prefix — so every
    resolver gated on a distinctive single name declines, correctly, rather
    than guess which one. The question's own second constraint, "amendment",
    breaks that ambiguity the same way an instrument type breaks an umbrella
    party name elsewhere in this file: intersected with content matching
    "Apex Meridian", exactly one document survives.

    Once that amendment is pinned, ingest's own cross-reference resolution
    frequently cannot name what it amends either — an amendment stating "the
    agreement dated as referenced in the recitals below" gives the resolver
    no inline filename or date to match, leaving document_relations with an
    unresolved edge (from_doc set, to_doc NULL). The documents.parties column
    is populated independently of that resolution, from the same extraction
    that reads the amendment's own signature block, so the original is found
    directly by "which document names these same parties and has the type
    this question names" — see db.find_docs_sharing_parties.

    Requires both signals: an "original <type>" phrase AND the word
    "amendment" appearing elsewhere in the question. Returns a set only when
    both the amendment and its original resolve to exactly one document each.
    """
    if not config.USE_DATABASE:
        return set()
    m = _ORIGINAL_TYPE_RE.search(question)
    if not m or not _AMENDMENT_WORD_RE.search(question):
        return set()
    type_hint = re.sub(r'\s+', ' ', m.group(1)).strip()

    amendment_doc: str | None = None
    for phrase in _bare_proper_noun_phrase_candidates(question) + _bare_proper_noun_candidates(question):
        try:
            content_docs = set(_db.find_source_docs_mentioning_phrase(
                _active_wiki_id(), session_id, phrase, cap=30) or [])
        except Exception as e:
            logger.error("resolve_scope: original-of-amendment content lookup failed for %r: %s", phrase, e)
            continue
        if not content_docs:
            continue
        try:
            amendment_docs = {d for d in _db.find_source_docs_by_title_tokens(
                _active_wiki_id(), session_id, ['Amendment'], cap=2000) if d}
        except Exception as e:
            logger.error("resolve_scope: original-of-amendment title lookup failed: %s", e)
            continue
        hit = content_docs & amendment_docs
        if len(hit) == 1:
            amendment_doc = next(iter(hit))
            break
    if not amendment_doc:
        return set()

    try:
        originals = _db.find_docs_sharing_parties(
            _active_wiki_id(), session_id, amendment_doc, type_hint,
            exclude=amendment_doc, cap=max_docs)
    except Exception as e:
        logger.error("resolve_scope: original-of-amendment party lookup failed: %s", e)
        return set()
    if not originals:
        return set()
    # More than one survivor is usually the same real document ingested twice
    # under different filenames (a plain PDF and its separately-run OCR
    # twin) rather than genuinely different instruments — including both is
    # redundant, not wrong, unlike the different-referent ambiguity the rest
    # of this file declines on. find_docs_sharing_parties' own cap (max_docs)
    # already bounds how far that can run.
    result = {amendment_doc, *originals}
    logger.info("Original-of-amendment: type=%r amendment=%s → %d document(s): %s",
                type_hint, _norm_doc_name(amendment_doc), len(result),
                {_norm_doc_name(d) for d in result})
    return result


# Distinct legal-instrument categories a question may name. Counting how many
# DIFFERENT categories appear tells scope resolution whether a question about a
# named party wants ONE of its instruments (summarise "the NordForge NDA") or a
# cross-instrument view of SEVERAL ("across the NDA, the arbitration notice, and
# the Section 9 petition") — the difference between pinning one document and
# pinning the whole party cluster.
_INSTRUMENT_PATTERNS = [
    ("nda",                re.compile(r'\bnda\b|non[-\s]?disclosure', re.I)),
    ("arbitration_notice", re.compile(r'arbitration\s+notice|notice\s+of\s+arbitration|request\s+for\s+arbitration', re.I)),
    ("petition",           re.compile(r'section\s*9\s*petition|\bpetition\b', re.I)),
    ("sha",                re.compile(r'shareholders?\s+agreement|\bsha\b', re.I)),
    ("service_agreement",  re.compile(r'services?\s+agreement|master\s+services|\bmsa\b', re.I)),
    ("jva",                re.compile(r'joint\s+venture(?:\s+agreement)?|\bjva?\b', re.I)),
    ("judgment",           re.compile(r'judg[e]?ment', re.I)),
    ("opinion",            re.compile(r'legal\s+opinion', re.I)),
    ("pleading",           re.compile(r'\bpleading\b|\bcomplaint\b', re.I)),
]


def _count_instrument_mentions(question: str) -> int:
    """Number of DISTINCT legal-instrument categories the question names."""
    return sum(1 for _, rx in _INSTRUMENT_PATTERNS if rx.search(question))


def _docs_of_entity_pages(question: str, pages: dict) -> dict:
    """Map entity-matched page titles to their source_docs with page counts.

    Returns {source_doc: n_matched_pages}. Lets scope resolution pin the entity's
    dominant document (whole-document summary → strict scope) or all of them
    (multi-instrument question) instead of returning no concrete target at all.
    """
    counts: dict[str, int] = {}
    for title in _pages_matching_question_entity(question, pages):
        page = pages.get(title)
        sd = page.get("source_doc", "") if isinstance(page, dict) else ""
        if sd:
            counts[sd] = counts.get(sd, 0) + 1
    return counts


# Any document-TYPE word. A question naming a type is talking ABOUT that type —
# it is not silently continuing with the current document — so carryover must
# never fire into it ("what about our NDAs?" asked during a Service Agreement
# thread is a pivot, not a continuation).
_CARRYOVER_TYPE_RE = re.compile(
    r'\b(nda|non-?disclosure|confidentiality\s+agreement|service\s+agreement|'
    r'shareholders?\s+agreement|joint\s+venture|jva|sha|msa|dpa|sow|'
    r'statement\s+of\s+work|data\s+processing\s+agreement|deed|lease|licen[cs]e|'
    r'judgment|judgement|court\s+case|legal\s+opinion|arbitration|petition|'
    r'complaint|affidavit|notice|contract|agreement)\b',
    re.IGNORECASE,
)

# "Which agreement/contract/notice ..." (a superlative/comparison pick FROM the
# set already established by the prior turn, e.g. "Which agreement has the
# strictest requirement?") is not a type pivot the way "the agreement" or
# "this notice" is — it's referring back to the same set, not naming a new
# document type. Without this exception _CARRYOVER_TYPE_RE's bare
# agreement/contract/notice words fire on it, carryover bails to [], and the
# question (which usually carries no other topical anchor of its own) falls
# through to a fully unscoped corpus search — confirmed live: this is exactly
# what turned a confidentiality-comparison thread's third turn into an answer
# about an unrelated Shareholder Agreement's waiver clause.
_COMPARATIVE_TYPE_REF_RE = re.compile(
    r'\bwhich\s+(?:\w+\s+){0,2}(?:nda|agreement|contract|notice|judgment|judgement|'
    r'opinion|petition|lease)s?\b',
    re.IGNORECASE,
)

# A bare backward-referencing pronoun standing in for the parties already
# established this thread ("draft a shareholder agreement WITH THEM") is not
# naming a new document TYPE the way "the shareholder agreement" is — it is
# asking for a new instrument between the SAME parties. Without this exception
# _CARRYOVER_TYPE_RE's bare "shareholders agreement" match vetoes carryover,
# retrieval falls through to an unscoped corpus search for "shareholder
# agreement", and drafting silently borrows an unrelated real document's party
# names instead of the parties actually under discussion. Confirmed live: "help
# me draft a new shareholder agreement with them", asked right after a Tata
# Power Renewable Energy / Helios Grid Advisory service-agreement summary,
# drafted for "SolarNexus Semiconductor Holdings LLC" and "Zephyr Systems
# LLC" — names that never appeared anywhere in the conversation — with no
# disclosure that "them" hadn't actually resolved to anything.
_BACKREF_PRONOUN_RE = re.compile(
    r'\b(?:with|for|between|among)\s+(?:them|him|her|these\s+parties|those\s+parties)\b',
    re.IGNORECASE,
)

# Ordinary-English uses of "notice"/"agreement"/"contract" that are NOT naming a
# new document type, unlike _COMPARATIVE_TYPE_REF_RE's "which agreement..."
# exception, which points BACK at an established set. These instead use the
# word as a plain noun referring to the document already under discussion —
# "what notice do we need to provide" (asking about the notice OBLIGATION, not
# switching to a document called "a Notice"), "breach the agreement" / "under
# the agreement" / "terms of the agreement" (all definite-article references to
# the current document, not a pivot to a different one). Confirmed live: both
# phrasings tripped _CARRYOVER_TYPE_RE and bounced back to disambiguation mid-
# thread even after the carryover-aware fix above, on a conversation that had
# just established a single document's scope. Deliberately narrow — only the
# specific phrasings observed to false-positive, not a general "the X" carve-out
# (which would also swallow genuine pivots like "what about the NDA instead").
_ORDINARY_TYPE_USAGE_RE = re.compile(
    r'\b(?:give|provide|serve|receive|send)\s+(?:\w+\s+){0,2}notice\b'
    r'|\bnotice\s+(?:period|do\s+we\s+need|is\s+required|clause)\b'
    r'|\bbreach(?:es|ed|ing)?\s+the\s+(?:agreement|contract)\b'
    r'|\b(?:under|pursuant\s+to)\s+the\s+(?:agreement|contract)\b'
    r'|\bterms?\s+of\s+the\s+(?:agreement|contract)\b',
    re.IGNORECASE,
)

# A demonstrative pronoun ("that"/"this") + a SPECIFIC litigation-filing or
# instrument type word ("that petition", "this notice", "that matter") is a
# backreference to the document already established this thread, not a pivot
# to a new one — confirmed live: "What records are sought to be preserved
# under THAT PETITION?", asked right after a Section 9 petition was the
# established scope, still tripped _CARRYOVER_TYPE_RE's bare "petition" match
# and forced a needless disambiguation prompt mid-thread. Deliberately
# excludes the generic words "agreement"/"contract"/"document"/"nda" — those
# stay covered by _VAGUE_DOC_PATTERN's own, deliberately stricter safety net
# (see the original "top 10 risks in THIS document" bug this project's Phase 2
# is built around): a bare demonstrative + generic instrument word is exactly
# the ambiguous case that net exists to keep asking about, since it could
# equally mean a different document the user has in mind. A demonstrative +
# SPECIFIC filing-type word carries far less of that ambiguity.
_DEMONSTRATIVE_BACKREF_RE = re.compile(
    r'\b(?:that|this)\s+(?:petition|affidavit|notice|judgment|judgement|'
    r'opinion|complaint|motion|application|award|plaint|summons|suit|claim|'
    r'matter|proceedings?|dispute|arbitration)\b',
    re.IGNORECASE,
)


# Scope-resolution methods that pinned specific documents and are therefore safe
# to inherit. A "family"/"broad"/"default" answer has no specific scope to pass
# on. "carryover" is included so a multi-turn thread stays on the same document
# rather than only the first follow-up working — the type-word and broad-phrasing
# guards remain the exits, and every carried turn discloses itself. "carryover-set"
# is included for the same reason: a comparison thread that established its set
# should keep it for subsequent follow-ups, not only the turn that resolved it.
# Matched as a PREFIX, not as an exact string. Scope methods are composed: a
# resolver names itself and every downstream correction appends to it, so the
# real values are "party-multi-doctype", "party-pair-family-corrected-doctype",
# "effective-date-doctype-corrected". An exact-membership set knows none of
# those, and every one of them silently ended the thread's scope.
#
# Measured across the 200-question audit: 137 of 196 resolved turns produced a
# method this set did not recognise, so the NEXT question in the conversation
# inherited nothing and fell back to the whole corpus. That is the thing a user
# experiences as the system losing track of which document they mean once a
# conversation gets long and moves between documents.
#
# Deliberately still a whitelist. "family", "default" and "broad" stay out: a
# turn that answered across a family or the corpus is not a document the next
# question should silently inherit.
_CARRYOVER_FROM_PREFIXES = ("file", "display-name", "effective-date", "date",
                            "party", "entity", "carryover",
                            "named-instrument-single")


def _is_carryover_method(method: str) -> bool:
    m = (method or "").strip().lower()
    return any(m == p or m.startswith(p + "-") for p in _CARRYOVER_FROM_PREFIXES)

# How many answers back to look for the turn whose scope should be inherited.
# NOT a widening of what may be inherited — see _last_document_turn. Bounded so a
# run of canned turns eventually ends the thread's scope rather than dragging a
# document along forever.
_CARRYOVER_LOOKBACK = 4


def _last_document_turn(recent: list[dict]) -> dict | None:
    """The newest answer that was a DOCUMENT turn, skipping canned ones.

    Canned turns — help, greetings, and general legal knowledge — deliberately
    record a blank scope (intent_agent._canned_payload), because a turn that
    resolved no document must never BECOME the scope the next turn inherits.
    But the readers below only ever looked at the single newest answer, so a
    blank-scope turn sitting in that slot also BLOCKED inheritance from the real
    document turn before it. Confirmed live: three turns correctly pinned to
    Service Agreement 2, then "by the way, what is novation?", and the next
    question — "what does that mean for the termination clause?" — came back
    asking "Which agreement are you asking about?", having lost a scope the
    conversation had plainly established.

    Skips ONLY blank-method entries. Any entry with a real method terminates the
    scan even when it is not inheritable (a "broad" turn means the user
    deliberately widened, and reaching past it to an older document would be the
    topic-drift failure this module exists to prevent — not a fix for it).
    """
    for entry in recent or []:
        if entry.get("method"):
            return entry
    return None


def has_established_document_scope(session_id: str) -> bool:
    """True when this conversation is already pinned to specific document(s).

    Used to decide whether a definitional question asked mid-thread ("what does
    indemnify mean") should be answered purely generally, or from the document
    under discussion with a general-knowledge aside attached. Reads the same
    recorded scope as _carryover_scope, and skips canned turns the same way, so
    the two cannot disagree about whether a thread has a document.
    """
    if not config.USE_DATABASE:
        return False
    try:
        recent = _db.get_recent_answer_scope(session_id, n=_CARRYOVER_LOOKBACK)
    except Exception as e:
        logger.error("has_established_document_scope: could not read scope: %s", e)
        return False
    last = _last_document_turn(recent)
    return bool(last and _is_carryover_method(last.get("method")) and last.get("docs"))


def _carryover_scope(question: str, session_id: str) -> list[str]:
    """The document this conversation is already about, when the question names none.

    A follow-up like "list all obligations of each party" carries no document
    reference at all, so every detector above it returns nothing and the scope
    defaults to the whole corpus — which then answers from whatever unrelated
    documents happen to rank well. Confirmed live: that exact question, asked
    directly after a Service Agreement thread, was answered from NDAs 1/4/6 about
    a party appearing in none of the documents under discussion; the same gap
    produced false "this clause is missing" findings in an earlier batch, because
    relative to the documents it wrongly retrieved, the clauses genuinely were.

    Deliberately conservative — inherits ONLY when there is nothing to weigh it
    against and the prior scope is unambiguous:
      * the question names no document TYPE (a type word means a pivot),
      * it isn't broad/collective phrasing (that wants the corpus, by design),
      * the preceding answer's scope was RESOLVED to specific documents.
    Anything else returns [] and the pre-existing corpus default stands, so this
    can only ever narrow a question that had no scope of its own.

    Inherits the preceding turn's recorded SCOPE, not its file count. Counting
    files cannot distinguish "scoped to one named reference" from "synthesised
    across the corpus": in this corpus one numbered reference routinely resolves
    to two files (a real document plus its zero-padded Test_* sibling), so a
    "exactly one file" rule never fires on the very follow-ups it exists to catch
    (confirmed live: the SA 2 key-dates answer recorded 2 files, so the
    obligations follow-up after it fell through to the corpus anyway).
    """
    if not config.USE_DATABASE:
        return []
    # Same underscore normalisation as _numbered_docs_in — \b doesn't fire
    # between "sha" and "_01" since "_" is a \w char, so "sha_01" silently
    # bypassed this guard and fell through to a stale carried-over document
    # from an earlier, unrelated turn (confirmed live: "sha_01" reply to a
    # disambiguation prompt for the SHA-GridEdge question landed on whatever
    # document the PRIOR unrelated comparison had answered, with no scope
    # warning, because both the type-word guard and _DOC_NAME_PATTERN missed
    # the underscore-joined shorthand).
    if (_CARRYOVER_TYPE_RE.search(question.replace('_', ' '))
            and not _COMPARATIVE_TYPE_REF_RE.search(question)
            and not _BACKREF_PRONOUN_RE.search(question)
            and not _ORDINARY_TYPE_USAGE_RE.search(question)
            and not _DEMONSTRATIVE_BACKREF_RE.search(question)
            # "in this agreement", "under that contract" — the type word is the
            # thing being pointed AT, not a pivot to a new type.
            # _DEMONSTRATIVE_BACKREF_RE deliberately omits the generic words
            # (agreement/contract/document/nda) because a bare demonstrative
            # plus a generic word is ambiguous in a FRESH thread, and the
            # disambiguation prompt is the right answer there. It is not
            # ambiguous one turn after a document was pinned, and this guard
            # could not tell those apart: it refused the inheritance before the
            # established-scope check below ever ran. Letting the phrase through
            # restores that distinction without weakening the fresh-thread case,
            # because a thread with no resolved prior turn still falls out at
            # _is_carryover_method and returns [] exactly as before.
            # Confirmed live: "What are the biggest risks for Suvarna in this
            # agreement?", asked directly after a Consultancy Agreement was
            # pinned, searched all 1,372 documents and answered from a different
            # company's Joint Venture Agreement that merely shared the word
            # "Suvarna" - and turns 3 and 4 then inherited that wrong document.
            and not _RX_DEMONSTRATIVE_DOC.search(question)):
        return []
    if _BROAD_SCOPE_RE.search(question) or _PLURAL_FAMILY_HINT_RE.search(question):
        return []
    try:
        recent = _db.get_recent_answer_scope(session_id, n=_CARRYOVER_LOOKBACK)
    except Exception as e:
        logger.error("_carryover_scope: could not read recent answer scope: %s", e)
        return []
    last = _last_document_turn(recent)
    if not last:
        return []
    if _is_carryover_method(last.get("method")) and last.get("docs"):
        return list(last["docs"])
    return []


# Explicit references back to a set the conversation just established —
# "which of those", "among these", "out of them". The counterpart to
# _COMPARATIVE_TYPE_REF_RE ("which agreement…"), which names the type instead
# of pointing at the set.
_SET_REFERENCE_RE = re.compile(
    r'\b(?:of|among|amongst|between|from)\s+(?:those|them|these|the\s+(?:two|three|four))\b',
    re.IGNORECASE,
)

# Upper bound on an inherited comparison set. A previous turn that touched more
# documents than this was a corpus-wide sweep, not "the set we are discussing" —
# inheriting it would pin retrieval to a large arbitrary list rather than narrow
# anything, so those fall through to the existing corpus default instead.
_COMPARATIVE_SET_MAX = 12


def _carryover_comparative_set(question: str, session_id: str) -> list[str]:
    """The documents a comparative follow-up is comparing, when it names none.

    Fills the gap _carryover_scope cannot: that function inherits the previous
    turn's RESOLVED scope, but a "broad"/"default" turn resolves no documents at
    all. So after a multi-document answer, a comparative follow-up
    ("Which agreement has the strictest requirement?") had nothing to inherit
    and fell through to an unscoped corpus search — confirmed live: the third
    turn of a Tata confidentiality thread came back comparing waiver clauses in
    two unrelated Shareholder Agreements, having quietly abandoned both the
    topic and the documents under discussion.

    Inherits the previous answer's files_used, which is what such a question
    refers back to ("which of THOSE"). Deliberately narrow — every condition
    must hold:
      * the question explicitly refers to a set (a comparative type reference
        or an "of those/among these" pointer), so an ordinary follow-up is
        never affected,
      * the previous turn genuinely spanned SEVERAL documents (a single-document
        thread is already handled by _carryover_scope),
      * that span is small enough to be a set someone is choosing between,
        not a corpus-wide sweep.
    Anything else returns [] and the pre-existing corpus default stands, so this
    can only narrow a question that had no scope of its own.
    """
    if not config.USE_DATABASE:
        return []
    if not (_COMPARATIVE_TYPE_REF_RE.search(question)
            or _SET_REFERENCE_RE.search(question)):
        return []
    try:
        recent = _db.get_recent_answer_scope(session_id, n=_CARRYOVER_LOOKBACK)
    except Exception as e:
        logger.error("_carryover_comparative_set: could not read recent answer scope: %s", e)
        return []
    last = _last_document_turn(recent)
    if not last:
        return []
    files = last.get("files") or []
    if not 2 <= len(files) <= _COMPARATIVE_SET_MAX:
        return []
    return list(files)


def _question_family_scope(question: str, session_id: str) -> tuple[str | None, set[str]]:
    """The document family a question explicitly names, plus that family's members.

    Returns (None, set()) whenever the question names no single family — which
    leaves the guard below inert and preserves the pre-Phase-2 behaviour.
    """
    if not config.USE_DATABASE:
        return None, set()
    try:
        available = set(_db.list_doc_families(_active_wiki_id(), session_id))
    except Exception as e:
        logger.error("resolve_scope: list_doc_families failed: %s", e)
        return None, set()
    family = _detect_question_family(question, available)
    if not family:
        return None, set()
    try:
        docs = set(_db.get_documents_by_family(_active_wiki_id(), session_id, family))
    except Exception as e:
        logger.error("resolve_scope: get_documents_by_family failed: %s", e)
        return None, set()
    # A document whose ingest-time CONTENT classification came out generic
    # ("Agreement") is invisible to the doc_family lookup above even when it
    # was filed under this family's folder — confirmed on the real corpus: a
    # Legal Opinion whose actual text reads like a bare bilateral contract
    # (National Council for Consumer Protection / Apex Sagar Financial
    # Services) got doc_family=None and dropped out of every "Legal Opinion"
    # family question. folder_hint already carries this signal from ingest at
    # no extra cost, so union it in rather than leave the gap.
    try:
        keywords = [kw for kw, fam in _DOC_FAMILY_RULES if fam == family]
        docs |= set(_db.get_documents_by_folder_hint(_active_wiki_id(), session_id, keywords))
    except Exception as e:
        logger.warning("resolve_scope: folder_hint fallback failed for family %s: %s", family, e)
    return family, docs


def _enforce_question_family(scoped: dict, family: str | None,
                             fam_docs: set[str]) -> dict:
    """Reconcile a party/entity resolution with the INSTRUMENT the question names.

    The party, party-pair and entity resolvers match on party NAMES alone. A
    party appearing across several document types therefore resolves to whichever
    of its documents the content match ranked highest — which can be a different
    instrument than the one asked about. Confirmed on the production-representative
    corpus: "the NDA between Tata Steel and NordForge" resolved to the
    Tata-NordForge *arbitration notice*, "the VitalSpring … *joint venture*" to an
    NDA, and "the *Services Agreement* between Tata Sons and its service provider"
    to three brand *judgments* — each answering "not covered" from a document
    whose type the question had already ruled out.

    The question stated the instrument; this makes that statement authoritative:

      * partial overlap — narrow to the named family's members (strictly tighter)
      * no overlap      — the resolution contradicts the question, so drop it and
                          scope to the named family rather than answer from the
                          wrong instrument

    Never widens: with no family named, no known members, or a scope that resolved
    no documents at all, the scope is returned untouched.
    """
    if not family or not fam_docs:
        return scoped
    targets = scoped.get("target_docs") or []
    if not targets:
        return scoped
    kept = [d for d in targets if d in fam_docs]
    if len(kept) == len(targets):
        return scoped
    method = scoped.get("method", "")
    if kept:
        logger.info("Scope %s narrowed to the %s family the question names: %d of %d doc(s)",
                    method, family, len(kept), len(targets))
        return {**scoped, "target_docs": kept, "method": f"{method}-family"}
    logger.info("Scope %s resolved entirely outside the %s family the question names "
                "— scoping to that family instead", method, family)
    return {**scoped, "scope": "family", "target_docs": sorted(fam_docs),
            "target_family": family, "is_broad": True, "confidence": 0.6,
            "method": f"{method}-family-corrected"}


# The instrument a question names, as a phrase: the words between "of the" /
# "governs the" / "to the" and whatever ends the noun phrase — a case
# designation, the parties, the date, or the end of the sentence.
_INSTRUMENT_PHRASE_RE = re.compile(
    r'\b(?:of|governs|in|to|under|about|from)\s+the\s+(.{3,80}?)'
    r'(?=\s+-\s|\s+between\b|\s+involving\b|\s+dated\b|\s*\?|,)',
    re.IGNORECASE,
)

# Cap on how many same-type documents may be returned before the type is judged
# too broad to scope on by itself.
_DOC_TYPE_MAX_DOCS = 6


# Words shared by so many instrument names that matching on them says nothing
# about which instrument is meant.
_TYPE_GENERIC_WORDS = frozenset({
    'agreement', 'agreements', 'the', 'of', 'and', 'in', 'to', 'for', 'a', 'an',
    'or', 'on', 'by', 'with', 'document', 'draft', 'privileged', 'confidential',
})


def _norm_type_words(text: str) -> list[str]:
    """Type words, lower-cased and singularised.

    Singularising matters: the corpus records what a question calls a "Board
    Resolution" as "EXTRACT OF MINUTES / CERTIFIED BOARD RESOLUTIONS".
    """
    words = re.sub(r'[^a-z0-9 ]', ' ', (text or "").lower()).split()
    return [w[:-1] if len(w) > 3 and w.endswith('s') else w for w in words]


def _type_core(text: str) -> set[str]:
    """The words of an instrument name that actually identify it."""
    return {w for w in _norm_type_words(text) if w not in _TYPE_GENERIC_WORDS}


def _resolve_docs_by_doc_type(question: str, session_id: str) -> set[str]:
    """Documents whose ingest-recorded instrument type is the one the question names.

    Every name-based resolver in this module matches PARTIES, and a party with
    several instruments resolves to whichever one its content match ranked
    highest. `_enforce_question_family` already corrects the coarsest version of
    that error, but a family is a bucket ("Pleading") holding many distinct
    instruments — a Rejoinder, an Affidavit in Support, an Interim Application
    and a Reply all live in it, and the question names exactly one.

    `documents.doc_type` records that name verbatim, in the same words the
    question uses. Measured over the 500-question evaluation: of 27 failures
    where the right document was never retrieved at all, 21 are reachable this
    way — questions like "the Rejoinder in the Petition - Appeal No. 511/2026"
    whose document is filed under the corpus's own unexplained abbreviation
    ("MAT-2011-8187 RITPAN FINAL v2.pdf"), which no party or filename signal
    could ever connect to the words the question actually used.
    """
    if not config.USE_DATABASE:
        return set()
    phrases = [m.group(1).strip() for m in _INSTRUMENT_PHRASE_RE.finditer(question or "")]
    if not phrases:
        return set()
    try:
        types = _db.get_document_types(_active_wiki_id(), session_id)
    except Exception as e:
        logger.error("resolve_scope: doc_type lookup failed: %s", e)
        return set()
    by_type: dict[str, set[str]] = {}
    for _sd, _dt in types.items():
        core = frozenset(_type_core(_dt))
        if core:
            by_type.setdefault(core, set()).add(_sd)
    for phrase in phrases:
        wordset = _type_core(phrase)
        if not wordset:
            continue
        # Deliberately NOT short-circuiting on an exact type match. The recorded
        # types vary in granularity for one instrument — "Legal Opinion" and
        # "Privileged & Confidential Legal Opinion", "Board Resolution Approving
        # Transaction" and "EXTRACT OF MINUTES / CERTIFIED BOARD RESOLUTIONS" —
        # so returning only the exact spelling drops most of the instrument's
        # own documents. Measured: exact-first matching found the golden source
        # in 89.3% of questions; unioning every compatible spelling is what
        # makes the result trustworthy enough to correct a scope with.
        hits: set[str] = set()
        for core, docs in by_type.items():
            shared = core & wordset
            if not shared:
                continue
            # A short name must match ENTIRELY. Half of a two-word name is one
            # word, and in this corpus's pleadings that one word is the family
            # noun, not the instrument: "Rejoinder in the Petition" would match
            # "Petition in the matter of" on "petition" alone and pull in all 18
            # pleadings, which is both wrong and too broad to correct with.
            # Longer names may match on half, which is what links a record of one
            # instrument written two ways ("Board Resolution Approving
            # Transaction" / "EXTRACT OF MINUTES / CERTIFIED BOARD RESOLUTIONS").
            shorter = min(len(core), len(wordset))
            if shared == core or shared == wordset:
                hits |= docs
            elif shorter > 2 and len(shared) * 2 >= shorter:
                hits |= docs
        if hits:
            return hits
    return set()


def _enforce_question_doc_type(scoped: dict, question: str, session_id: str) -> dict:
    """Reconcile a resolved scope with the INSTRUMENT TYPE the question names.

    Same contract as _enforce_question_family, one level finer: partial overlap
    narrows, no overlap means the resolution contradicts the question. The
    no-overlap case only replaces the scope when the named type resolves to a
    workably small set — correcting one wrong document into twenty right-typed
    ones would trade a precise wrong answer for an unfocused one.
    """
    targets = scoped.get("target_docs") or []
    if not targets:
        return scoped
    try:
        type_docs = _resolve_docs_by_doc_type(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: doc-type enforcement failed: %s", e)
        return scoped
    if not type_docs:
        return scoped
    method = scoped.get("method", "")
    kept = [d for d in targets if d in type_docs]
    if kept:
        if len(kept) == len(targets):
            return scoped
        logger.info("Scope %s narrowed to the %d document(s) of the instrument type "
                    "the question names", method, len(kept))
        return {**scoped, "target_docs": sorted(kept), "method": f"{method}-doctype"}
    # No overlap has two very different causes, and only one of them is a wrong
    # answer. If a scoped document's OWN recorded type shares an identifying
    # word with what the question asked for, this is the corpus wording one
    # instrument two ways — the resolution is probably right and the matcher
    # merely failed to connect the spellings. Correcting there is what turned 35
    # already-correct scopes into wrong ones on the first attempt at this.
    # A genuine contradiction shares nothing: the question says "Rejoinder in
    # the Petition" and the resolution produced a Master Services Agreement.
    try:
        scoped_types = _db.get_document_types(_active_wiki_id(), session_id)
    except Exception:
        scoped_types = {}
    question_core: set[str] = set()
    initialisms: set[str] = set()
    for m in _INSTRUMENT_PHRASE_RE.finditer(question or ""):
        question_core |= _type_core(m.group(1))
        words = [w for w in re.split(r'[\s/]+', m.group(1)) if w and w[0].isalpha()]
        acronym = "".join(w[0] for w in words).lower()
        if 4 <= len(acronym) <= 12:
            initialisms.add(acronym)
    for d in targets:
        recorded = scoped_types.get(d, "")
        if not recorded.strip():
            # Two of this corpus's documents carry no recorded type at all.
            # Absence of evidence is not contradiction — never correct one away.
            logger.info("Scope %s kept: %s has no recorded instrument type to contradict "
                        "the question", method, _norm_doc_name(d))
            return scoped
        if _type_core(recorded) & question_core:
            logger.info("Scope %s sits outside the matched type set, but %s is recorded "
                        "as a compatible instrument — leaving the scope alone",
                        method, _norm_doc_name(d))
            return scoped
        # The filename is a second, independent record of the type: this corpus
        # files an instrument under the initialism of its full name ("WCILOM"
        # for a Written Consent in Lieu of Meeting). When that agrees with the
        # question, ingest's classification is what is wrong — confirmed live on
        # exactly that document, recorded as "EXTRACT OF MINUTES / CERTIFIED
        # BOARD RESOLUTIONS" and correctly resolved by the party branch.
        haystack = re.sub(r'[^a-z0-9]', '', _norm_doc_name(d).lower())
        if any(a in haystack for a in initialisms):
            logger.info("Scope %s kept: %s is filed under the initialism of the "
                        "instrument the question names", method, _norm_doc_name(d))
            return scoped

    narrowed = set(type_docs)
    if len(narrowed) > _DOC_TYPE_MAX_DOCS:
        narrowed = _narrow_by_question_tokens(question, set(type_docs)) or set(type_docs)
    if len(narrowed) > _DOC_TYPE_MAX_DOCS:
        logger.info("Scope %s sits outside the instrument type the question names, but "
                    "that type spans %d documents — leaving the scope alone",
                    method, len(narrowed))
        return scoped
    logger.info("Scope %s resolved entirely outside the instrument type the question "
                "names — scoping to that type's %d document(s) instead",
                method, len(narrowed))
    return {**scoped, "scope": "single_doc", "target_docs": sorted(narrowed),
            "is_broad": False, "confidence": 0.7,
            "method": f"{method}-doctype-corrected"}


# ---------------------------------------------------------------------------
# Amendment families — the question names two documents, the second answers it
# ---------------------------------------------------------------------------
# "What is the CURRENT value of notice days under the agreement family comprising
# the Cloud Services Agreement between A and B dated 24 July 2021 AND THE
# AMENDMENT RECORDED IN the Amendment Agreement between A and B dated 24 May
# 2021, after giving effect to this amendment?"
#
# Two documents are named, by the same party pair, distinguished only by date.
# The one that answers the question is the SECOND: the amendment states the value
# that now governs, and the original states the one it replaced. Every compound
# resolver in front of this narrows a multi-document match down to one and keeps
# the first — so the amendment was dropped before retrieval ever saw it and the
# answer confidently reported the superseded figure.
#
# Measured over both evaluation sets: 6 of the 9 questions in this shape
# retrieved only the original. The mirror-image shape ("in the original
# agreement, BEFORE it was amended by ...") already resolves correctly 7 times
# out of 7 — it wants the first document, which is what the existing resolvers
# already return, so it is deliberately left alone here.
_AMENDMENT_TAIL_RE = re.compile(
    r'\band\s+the\s+amendments?\s+recorded\s+in\s+',
    re.IGNORECASE,
)

# An amendment family is two documents, occasionally three. Past that the tail
# resolved to a party's whole book of business rather than the one instrument it
# names, and adding all of it would bury the original the question also asked
# about.
_AMENDMENT_FAMILY_MAX_DOCS = 4


def _expand_amendment_family(scoped: dict, question: str, session_id: str) -> dict:
    """Put the amendment a "family comprising ..." question names back in scope.

    Resolves the text AFTER "and the amendment recorded in" as a scope question
    in its own right — it is a complete document reference (instrument type,
    party pair, date), and resolving it separately is what stops the compound
    resolvers from having to choose between the two documents named.
    """
    m = _AMENDMENT_TAIL_RE.search(question or "")
    if not m:
        return scoped
    targets = list(scoped.get("target_docs") or [])
    if not targets:
        return scoped
    tail = question[m.end():]
    try:
        # Uncorrected: the tail names an Amendment Agreement, so doc-type
        # enforcement over it would be measuring the tail's own instrument
        # against itself — and the correction it can make (replacing the scope
        # wholesale) is not one this caller wants applied to a sub-clause.
        amended = _resolve_scope_uncorrected(tail, session_id)
    except Exception as e:
        logger.error("resolve_scope: amendment-family expansion failed: %s", e)
        return scoped
    amend_docs = list((amended or {}).get("target_docs") or [])
    if not amend_docs:
        return scoped
    # The tail names ONE amendment, by party pair AND date. A party-only match
    # returns every amendment those two parties ever signed, and a sibling
    # amendment answers this question with a real but wrong figure — so narrow
    # on the date the tail states before adding anything.
    if len(amend_docs) > 1:
        try:
            pinned = _narrow_by_question_tokens(tail, set(amend_docs),
                                                precise_only=True)
        except Exception:
            pinned = set()
        if pinned:
            amend_docs = sorted(pinned)
    added = [d for d in amend_docs if d not in targets]
    if not added:
        return scoped
    if len(targets) + len(added) > _AMENDMENT_FAMILY_MAX_DOCS:
        logger.info("Amendment-family expansion skipped: the tail resolved to "
                    "%d document(s), too many to be the one amendment named",
                    len(added))
        return scoped
    logger.info("Amendment family: added %d amending document(s) to scope — %s",
                len(added), [_norm_doc_name(d) for d in added])
    return {**scoped,
            "target_docs": targets + added,
            "amendment_docs": added,
            "method": f"{scoped.get('method', '')}-amendment-family"}


# ---------------------------------------------------------------------------
# Content-descriptive ambiguity — one description, several documents
# ---------------------------------------------------------------------------
# Every resolver above pins a document by its NAME (a filename, a number, a title
# identifier), by a party name, by a party PAIR, or by a date it recites. None of
# them notices the opposite failure: a question that identifies its document by a
# descriptive phrase drawn from the document's own TEXT, where that exact phrase
# is boilerplate repeated verbatim across several agreements with the same party.
#
# Confirmed live (Q85, scoring 6/10 through v3 and v4): "the services agreement
# entered into by Tata Sons Private Limited having its registered office at Bombay
# House, 24 Homi Mody Street, Mumbai" — that registered-office block is stated
# IDENTICALLY in Service Agreement 2 and Service Agreement 4, so the question has
# two equally valid answers (execution dates 18 July 2025 and 28 August 2025).
# The system answered "28 August 2025", cited SA 4, and stopped: correct FOR SA 4,
# but the reader gets one confident date for a question the description cannot
# resolve. Every existing signal passes it by — _resolve_docs_by_party gives up
# ("Tata Sons" is an umbrella name over the cap), _resolve_docs_by_party_pair sees
# only one party, _resolve_docs_by_date finds no recited date (the date is what is
# being ASKED for), and the entity branch below collapses its matches to ONE
# winner via max(ent_counts). So the second candidate never surfaced anywhere.
#
# The corporate-boilerplate framing is deliberate and narrow. A registered-office
# block is the one descriptor a drafter copies verbatim between every instrument
# with the same counterparty, which is exactly what makes it non-identifying —
# and equally what makes the user believe it identifies one document. Free-form
# subject-matter paraphrases ("the wastewater-dosing NDA") are NOT handled here:
# they are genuinely distinguishing more often than not, and the existing
# paraphrase path in check_disambiguation_node already covers them.

# The office noun that marks a party-address descriptor. Matching on this
# specific vocabulary — rather than any "located at X" phrase — keeps the
# detector off questions ASKING about a location inside a document ("what does
# the lease say about the premises at 5 Main Street") and on questions USING an
# address to name which document they mean.
_OFFICE_NOUN_RE_STR = (
    r'(?:registered\s+(?:office|address)|principal\s+place\s+of\s+business|'
    r'(?:principal|head|corporate|registered|branch)\s+office|office)'
)

_DESCRIPTIVE_IDENTIFIER_RES = (
    # "… having its registered office at X", "… with registered office at X"
    re.compile(
        r'\b(?:having|with|has)\s+(?:its|their|a|the)?\s*' + _OFFICE_NOUN_RE_STR +
        r'\s+(?:situated\s+|located\s+)?(?:at|in)\s+(?P<desc>[^?;]+)',
        re.IGNORECASE,
    ),
    # "… whose registered office is at X", "… registered office at X" — anchored
    # on the unmistakable corporate-boilerplate noun, so it is safe standing alone
    # without a "having"/"with" frame in front of it.
    re.compile(
        r'\b(?:whose\s+)?(?:registered\s+(?:office|address)|principal\s+place\s+of\s+business)\s+'
        r'(?:is\s+)?(?:situated\s+|located\s+)?(?:at|in)\s+(?P<desc>[^?;]+)',
        re.IGNORECASE,
    ),
)

_DESC_MIN_TOKENS = 4


# The captures above run to the end of the sentence, so a question that appends
# another clause ("… registered office at X, and when does it expire?") drags
# that clause in too. Cut at the join — an address never continues into an
# interrogative or an auxiliary verb.
_DESC_TAIL_RE = re.compile(
    r'\b(?:and|but|or)\s+(?:what|when|who|whom|which|where|how|why|is|are|was|were|'
    r'does|do|did|has|have|had|can|could|should|would|will)\b',
    re.IGNORECASE,
)

# An address never begins with one of these; a relative clause the capture ran
# into always does ("… registered office AT WHICH notices must be served").
_DESC_RELATIVE_PRONOUNS = frozenset({
    "which", "whom", "what", "whose", "that", "who", "where", "whether",
})


def _extract_descriptive_identifier(question: str) -> str:
    """The party-address descriptor a question uses to identify its document.

    Returns the raw descriptor text ("Bombay House, 24 Homi Mody Street,
    Mumbai") or "" when the question carries no such phrase.

    The result must LOOK like an address: two or more capitalised tokens, and not
    opening on a relative pronoun. Without that gate the office-noun frames also
    fire on a question ASKING about a registered office rather than identifying a
    document by one ("what is the registered office AT WHICH notices must be
    served under NDA 3"), and the ordinary prose that follows would then be
    looked up as if it were a distinguishing phrase — a run of common legal words
    matches a handful of documents by accident, manufacturing an ambiguity that
    isn't there. Both conditions are needed: that example clears a
    capitalised-token count on its own ("NDA") and is caught only by the pronoun
    check, while a lowercase clause is caught only by the count.
    """
    for rx in _DESCRIPTIVE_IDENTIFIER_RES:
        m = rx.search(question)
        if not m:
            continue
        desc = (m.group("desc") or "").strip()
        cut = _DESC_TAIL_RE.search(desc)
        if cut:
            desc = desc[:cut.start()]
        desc = desc.strip().strip('.,:;"\'')
        tokens = desc.split()
        if len(tokens) < _DESC_MIN_TOKENS:
            continue
        if tokens[0].lower().strip('.,') in _DESC_RELATIVE_PRONOUNS:
            continue
        if sum(1 for t in tokens if t[:1].isupper()) >= 2:
            return desc
    return ""


def resolve_scope(question: str, session_id: str, pages: dict | None = None,
                  chat_session_id: str | None = None) -> dict:
    """Resolve a question's retrieval scope, then hold it to the instrument named.

    The resolution itself is _resolve_scope_uncorrected below; this wrapper
    applies the one check that has to see the FINAL answer rather than any
    single branch's — that the documents resolved are actually of the
    instrument type the question asked about (see _enforce_question_doc_type).

    Two exemptions. A question naming a FILE outright has said something
    stronger than a type and must never be overridden by one. A carried-over
    scope names no instrument at all — the type words belong to the earlier
    turn, not this one, so applying them here would silently re-scope a
    follow-up onto a different document.
    """
    scoped = _resolve_scope_uncorrected(question, session_id, pages, chat_session_id)
    method = (scoped or {}).get("method", "")
    if not scoped or method == "file" or "carryover" in method:
        return scoped
    scoped = _enforce_question_doc_type(scoped, question, session_id)
    # Last, so it adds the amendment back whatever narrowing ran above: the two
    # documents in an amendment family are different instrument types, and
    # doc-type enforcement is entitled to drop one of them.
    return _expand_amendment_family(scoped, question, session_id)


def _resolve_scope_uncorrected(question: str, session_id: str, pages: dict | None = None,
                               chat_session_id: str | None = None) -> dict:
    """Resolve the retrieval scope of a question in ONE place (Phase 2).

    Consolidates the three previously-scattered scope signals — named-document
    detection, known-entity matching, and broad/collective phrasing — into a
    single decision object the retrieval node acts on, instead of each stage
    re-deriving scope with its own regex.

    Returns a dict:
      scope         : "single_doc" | "family" | "corpus"
      target_docs   : concrete source_doc names when known (single_doc/family)
      target_family : canonical family label when scope == "family", else None
      is_broad      : True for family / broad cross-document questions
      confidence    : rough 0-1 confidence in the scope call
      method        : which signal decided it ("file"/"entity"/"family"/"broad"/"default")

    Deliberately deterministic (no LLM call) — it composes the existing
    detectors, matching how the rest of this pipeline resolves scope, and stays
    conservative: anything it isn't sure about falls through to "corpus"
    (unfiltered whole-session search), which is the pre-Phase-2 behaviour and
    can never wrongly starve retrieval of a relevant document.
    """
    if pages is None:
        try:
            pages = _load_index(session_id).get("pages", {})
        except Exception as e:
            logger.error("resolve_scope: could not load index: %s", e)
            pages = {}

    # A document the question quotes back by its own displayed or referenced
    # name — the strongest identifier a question can carry, and exactly what a
    # user supplies when the system has just asked which document they meant.
    #
    # Ahead of _detect_mentioned_files below, which is looser by design: asked
    # about "Palladion Global PurcAgre 26-05-2022 - filed.pdf" it matched on
    # the word "filed" and pinned thirty documents. A name matched in full
    # should win over a name matched in part.

    # An explicit date matching exactly one document in the whole corpus.
    # Placed here with the filename match rather than down with the other date
    # handling, because a UNIQUE date is a precise identifier and not the weak
    # signal that ordering assumes. Measured on the 200-question audit: of ten
    # questions whose named document was never retrieved, party-family
    # resolution answered them with the wrong member of the right family - a
    # different Legal Opinion, a different Pleading - while the date named
    # exactly one document.
    #
    # Still only fires on a unique hit. A date shared by several documents is
    # not an identifier, which is precisely why the party name outranks it
    # everywhere else.
    try:
        _dated = _resolve_docs_by_effective_date(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: effective-date resolution failed: %s", e)
        _dated = set()
    if _dated:
        return {"scope": "single_doc", "target_docs": sorted(_dated),
                "target_family": None, "is_broad": False,
                "confidence": 0.88, "method": "effective-date"}
    try:
        pasted = _resolve_docs_by_display_name(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: display-name resolution failed: %s", e)
        pasted = set()
    if pasted:
        return {"scope": "single_doc", "target_docs": sorted(pasted),
                "target_family": None, "is_broad": False,
                "confidence": 0.92, "method": "display-name"}

    # 1. Single specific document — a named file or a distinctive known entity.
    #    Mirrors get_context's own force-include logic; here it only records the
    #    decision (get_context still does the actual page scoping for this case).
    try:
        mentioned = _detect_mentioned_files(question, pages)
    except Exception:
        mentioned = set()
    if mentioned:
        # A numbered reference ("SHA 1") can match BOTH the real document and a
        # synthetic Test_* sibling of the same type+number — both get pinned, and
        # the answer LLM tends to answer from whichever is richer (usually the
        # synthetic one) with no indication it switched. Detect the collision so
        # the answer can warn; a read-only overlay, target_docs is unchanged.
        try:
            collisions = _numbered_doc_collisions(question, _distinct_source_docs(pages))
        except Exception:
            collisions = []
        return {"scope": "single_doc", "target_docs": sorted(mentioned),
                "target_family": None, "is_broad": False,
                "confidence": 0.9, "method": "file",
                "doc_collisions": collisions}
    # The instrument the question names, resolved once and applied to every
    # name-based branch below (party / party-pair / entity). Deliberately AFTER
    # the explicit-filename branch above: a user who names a file outright has
    # said something stronger than a document type, and must never be overridden
    # by it.
    _fam_name, _fam_docs = _question_family_scope(question, session_id)

    # A question naming SEVERAL party-pairs ("compare X between A and B with Y
    # between C and D") has to be resolved pair by pair — every branch below
    # reads the question as one flat list of names and would answer from
    # whichever pair its scoring happened to favour, silently dropping the
    # other side of the comparison. Runs ahead of the single-party branch
    # precisely because that branch DOES resolve such questions: confirmed live,
    # a two-SSA comparison resolved "party" to the Tata Steel agreement alone
    # and the answer compared it against nothing. Only fires when every pair
    # resolves, so it never trades a real single-document match for a partial one.
    #
    # Not every compound question repeats "between" once per matter, though —
    # see _resolve_docs_by_combinatorial_pairing's docstring — so this tries
    # that too when the explicit split finds nothing. Both attempts have to
    # run HERE, ahead of the single-party branch below, not merely inside
    # _resolve_docs_by_party_pair: confirmed live, a 4-party question with no
    # repeated "between" reached the single-party branch first, which found
    # one party's name resolved to exactly one document and returned
    # immediately — the compound path never got a turn at all, regardless of
    # what it would itself have found.
    _pair_segments = _question_pair_segments(question)
    compound_docs: set[str] = set()
    if _pair_segments:
        try:
            compound_docs = _resolve_docs_by_party_pair(question, session_id)
        except Exception as e:
            logger.error("resolve_scope: compound party-pair resolution failed: %s", e)
    else:
        try:
            compound_docs = _resolve_docs_by_combinatorial_pairing(question, session_id)
        except Exception as e:
            logger.error("resolve_scope: combinatorial party-pairing failed: %s", e)
    # Neither pairing approach sees a question naming several documents by
    # bare nickname ("the Amberline NDA, the Apex Cobalt NDA, and the Apex
    # Falcora EV NDA") — no corporate suffix on any of them for _PARTY_NAME_RE
    # to anchor on. Tried at this same priority, ahead of single-party
    # resolution, for the identical reason the pairing attempts are: letting
    # one name's own resolver run first and return immediately on its single
    # best match never gives this a turn, regardless of what it would find.
    if not compound_docs:
        try:
            compound_docs = _resolve_docs_by_named_instruments(question, session_id)
        except Exception as e:
            logger.error("resolve_scope: named-instrument list resolution failed: %s", e)
    # A third shape neither of the above reaches: a document named by TYPE
    # alone ("the original IT Outsourcing Agreement"), opposite an amendment
    # named by an umbrella party with no corporate suffix ("the Apex Meridian
    # amendment") — genuinely ambiguous alone on this corpus, broken only by
    # intersecting with "amendment" the same way an instrument type breaks an
    # umbrella party elsewhere in this file. Same priority, same reasoning.
    if not compound_docs:
        try:
            compound_docs = _resolve_original_of_amendment(question, session_id)
        except Exception as e:
            logger.error("resolve_scope: original-of-amendment resolution failed: %s", e)
    if compound_docs:
        return _enforce_question_family(
            {"scope": "single_doc", "target_docs": sorted(compound_docs),
             "target_family": None, "is_broad": False,
             "confidence": 0.8, "method": "party-pair-compound"},
            _fam_name, _fam_docs)

    # Party-name → document via full-text content match. Catches the case the
    # filename/entity detectors miss: the user names the counterparty ("SteelLoop
    # Resource Recovery", "Cold Chain Energy Services") but the corpus files the
    # document under a bare type+number and masks the party in metadata. Only
    # fires on an unambiguous single-document hit, so it's safe to prefer over the
    # weaker entity heuristic below (which resolves no concrete target_docs).
    try:
        party_docs = _resolve_docs_by_party(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: party resolution failed: %s", e)
        party_docs = set()
    if party_docs:
        n_instr = _count_instrument_mentions(question)
        if len(party_docs) == 1 or n_instr >= 2:
            # One document, OR a multi-instrument question naming this party's
            # several instruments ("across the NDA, the arbitration notice, and
            # the Section 9 petition") — pin the whole resolved cluster so every
            # named instrument is retrieved, not just the ones a single semantic
            # pass happened to surface.
            return _enforce_question_family(
                {"scope": "single_doc", "target_docs": sorted(party_docs),
                 "target_family": None, "is_broad": False,
                 "confidence": 0.85 if len(party_docs) == 1 else 0.8,
                 "method": "party" if len(party_docs) == 1 else "party-multi"},
                _fam_name, _fam_docs)
        # Party spans several documents. If the question ALSO names exactly one
        # instrument type ("the SOW with Cindercast"), narrow to that specific
        # document within the resolved family — sharper than answering across
        # all of them when the user asked for one.
        try:
            available = set(_db.list_doc_families(_active_wiki_id(), session_id)) if config.USE_DATABASE else set()
            fam = _detect_question_family(question, available)
            fam_docs = set(_db.get_documents_by_family(_active_wiki_id(), session_id, fam)) if fam else set()
        except Exception:
            fam_docs = set()
        narrowed = party_docs & fam_docs
        if len(narrowed) == 1:
            return {"scope": "single_doc", "target_docs": sorted(narrowed),
                    "target_family": None, "is_broad": False,
                    "confidence": 0.8, "method": "party"}
        # No instrument named, or naming one didn't narrow further — used to
        # fall through here to carryover/corpus-default instead of using the
        # match, on the reasoning that an unnarrowed multi-document party hit
        # was too uncertain to commit to. But _resolve_docs_by_party's own
        # contract already rules that out: it gives up and returns empty
        # whenever the smallest candidate set exceeds max_docs (default 4), so
        # a non-empty party_docs here is ALREADY a small, coherent instrument
        # family (a deal's own MSA+SOW+DPA), not an arbitrary pile. Confirmed
        # live: "Can Cindercast use Torvald's data to train AI models?" names
        # no instrument type, so this used to fall through — past the party
        # match that correctly found CND-TOR-MSA/SOW/DPA — all the way to
        # carryover, which then answered from "Tata Brand Judgment 3", left
        # over from an unrelated earlier question in the same thread. Silent
        # and wrong, which is worse than the disambiguation prompt this branch
        # exists to avoid. Answer across the whole resolved family instead.
        return _enforce_question_family(
            {"scope": "single_doc", "target_docs": sorted(party_docs),
             "target_family": None, "is_broad": False,
             "confidence": 0.75, "method": "party-multi"},
            _fam_name, _fam_docs)

    # Adversarial / two-sided matter ("Aether Technologies Inc. against Helios
    # Energy Corporation"). The single-party resolver above cannot reach this:
    # each name on its own spans far more documents than its cap allows, so it
    # returns nothing and scope would fall through to an unscoped corpus search
    # that has repeatedly failed to surface the right document. Runs only after
    # every stronger single-document signal has passed, and returns a set at
    # all only when both parties resolve to a small shared cluster.
    try:
        pair_docs = _resolve_docs_by_party_pair(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: party-pair resolution failed: %s", e)
        pair_docs = set()
    if pair_docs:
        # An ambiguous multi-doc pair result can still be pinned by an
        # explicit date the question recites — a stronger, cheaper signal
        # than leaving several documents for the answer LLM to sort out.
        # Confirmed live: a Board Resolution naming only ONE of its two
        # parties (common in this corpus — a board resolves in its own name,
        # never the counterparty's) can never appear in a two-party title or
        # content match at all, so the party-pair cluster this question
        # produces is real siblings that don't include the actual right
        # answer. Before the date check existed here, this exact case
        # resolved correctly via the date resolver below — party-pair
        # returning early on ANY non-empty result, even a wrong-ish
        # ambiguous one, silently took that away.
        # Never on a compound question: it recites one date PER pair, so pinning
        # the whole result to a single date would drop the other pair's document.
        if len(pair_docs) > 1 and not _pair_segments:
            try:
                date_docs = _resolve_docs_by_date(question, session_id)
            except Exception as e:
                logger.error("resolve_scope: date check on party-pair result failed: %s", e)
                date_docs = set()
            if date_docs and len(date_docs) == 1:
                # A unique date match is a strong signal ONLY when the matched
                # document is actually about one of the two named parties —
                # otherwise it may just be a coincidence (a Board Resolution
                # for an entirely different deal that happens to recite the
                # same calendar date somewhere in its own text). Confirmed
                # live: "24 March 2024" uniquely matched an unrelated Board
                # Resolution with no connection to either party the question
                # named, silently replacing a correct 5-document Tata Elxsi
                # cluster (containing the real answer) with the wrong single
                # document.
                date_doc = next(iter(date_docs))
                names = [m.group(1).strip() for m in _PARTY_NAME_RE.finditer(question)]
                relevant = False
                for n in names:
                    try:
                        if date_doc in set(_db.find_source_docs_mentioning_phrase(
                                _active_wiki_id(), session_id, n, cap=500) or []):
                            relevant = True
                            break
                    except Exception:
                        continue
                if relevant:
                    logger.info("Party-pair match %d document(s) pinned to 1 by recited date: %s",
                                len(pair_docs), {_norm_doc_name(d) for d in date_docs})
                    pair_docs = date_docs
                else:
                    logger.info("Date match %s discarded — mentions neither party named "
                                "in the question", _norm_doc_name(date_doc))
        return _enforce_question_family(
            {"scope": "single_doc", "target_docs": sorted(pair_docs),
             "target_family": None, "is_broad": False,
             "confidence": 0.82 if len(pair_docs) == 1 else 0.75,
             "method": "party-pair"},
            _fam_name, _fam_docs)

    # A matter/reference number the question recites ("MAT-2021-7750")
    # resolving to exactly one document. Runs after every party-name signal,
    # before the date check below — see _resolve_docs_by_matter_reference for
    # why a matter number only counts on a genuinely unique hit.
    try:
        matter_docs = _resolve_docs_by_matter_reference(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: matter-reference resolution failed: %s", e)
        matter_docs = set()
    if matter_docs:
        return {"scope": "single_doc", "target_docs": sorted(matter_docs),
                "target_family": None, "is_broad": False,
                "confidence": 0.85, "method": "matter-reference"}

    # An explicit date the question recites ("the SA dated 15 January 2026")
    # resolving to exactly one document. Runs after every party-name signal
    # (a party name is a stronger identifier than a shared execution date), but
    # still before the vaguer family/broad/carryover branches below — a
    # question that names nothing else but a specific date it already knows
    # should resolve directly rather than asking "which document?" (confirmed
    # live: this exact phrasing needlessly disambiguated).
    try:
        date_docs = _resolve_docs_by_date(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: date resolution failed: %s", e)
        date_docs = set()
    if date_docs:
        return {"scope": "single_doc", "target_docs": sorted(date_docs),
                "target_family": None, "is_broad": False,
                "confidence": 0.8, "method": "date"}

    # One document named by nickname plus instrument type ("the Voltas escrow
    # agreement", "the Palladion Global purchase agreement"). Placed here, after
    # every party, matter-reference and date signal has already declined, so it
    # can only turn a fall-through into a resolution. Before this, the same
    # question resolved only when the user also recited a date — which made a
    # date feel mandatory in ordinary phrasing.
    try:
        named_one = _resolve_docs_by_single_named_instrument(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: single named-instrument resolution failed: %s", e)
        named_one = set()
    if named_one:
        return _enforce_question_family(
            {"scope": "single_doc", "target_docs": sorted(named_one),
             "target_family": None, "is_broad": False,
             "confidence": 0.78, "method": "named-instrument-single"},
            _fam_name, _fam_docs)

    # A party pair that shares a whole document family. The title- and
    # content-intersection resolvers above both decline here, because sixteen
    # documents is not a narrowing to them — and scope then fell all the way
    # through to an unscoped corpus search, discarding a strong signal for
    # being insufficiently precise. Reading documents.parties gives the shared
    # set exactly; retrieval ranking over sixteen candidates is a far better
    # position than over 1,372.
    #
    # Returned as a pinned set rather than a single document on purpose. One
    # document chosen from sixteen, without saying so, is a fabrication with a
    # citation attached.
    try:
        pair_family = _resolve_docs_by_party_pair_index(question, session_id)
    except Exception as e:
        logger.error("resolve_scope: party-pair index resolution failed: %s", e)
        pair_family = set()
    if pair_family:
        _single = len(pair_family) == 1
        return {"scope": "single_doc" if _single else "family",
                "target_docs": sorted(pair_family),
                "target_family": None, "is_broad": not _single,
                "confidence": 0.8 if _single else 0.7,
                "method": "party-pair-index"}

    if _question_names_a_document(question, []) and _question_mentions_known_entity(question, pages):
        # Resolve the concrete document(s) the entity points at so retrieval can
        # scope STRICTLY to them. A single-instrument question (e.g. "summarise
        # the NordForge NDA") pins the entity's dominant document — dropping the
        # supplementary cross-document pages that otherwise dilute a whole-document
        # summary into half "Not covered"; a multi-instrument question keeps all
        # matched documents.
        ent_counts = _docs_of_entity_pages(question, pages)
        if ent_counts:
            if _count_instrument_mentions(question) >= 2:
                ent_targets = sorted(ent_counts)
            else:
                ent_targets = [max(ent_counts, key=ent_counts.get)]
            return _enforce_question_family(
                {"scope": "single_doc", "target_docs": ent_targets,
                 "target_family": None, "is_broad": False,
                 "confidence": 0.72, "method": "entity"},
                _fam_name, _fam_docs)
        return {"scope": "single_doc", "target_docs": [],
                "target_family": None, "is_broad": False,
                "confidence": 0.7, "method": "entity"}

    # 2. Family scope — a COLLECTIVE reference to one document family that
    #    actually exists in this session. Requires both a collective marker
    #    (broad phrasing or a plural family noun) and a single resolved family,
    #    so a narrow single-clause question is never wrongly filtered.
    collective = bool(_BROAD_SCOPE_RE.search(question) or _PLURAL_FAMILY_HINT_RE.search(question))
    if collective and _fam_name:
        return {"scope": "family", "target_docs": sorted(_fam_docs),
                "target_family": _fam_name, "is_broad": True,
                "confidence": 0.75, "method": "family"}

    # 3. Broad cross-document question with no single resolvable family.
    if _BROAD_SCOPE_RE.search(question):
        return {"scope": "corpus", "target_docs": [], "target_family": None,
                "is_broad": True, "confidence": 0.6, "method": "broad"}

    # Did the question NAME a specific counterparty that simply couldn't be
    # pinned to one document? Reaching here means every detector above passed —
    # including the party-content match, which returns empty when the named
    # party spans too many documents to disambiguate (an umbrella name like
    # "Tata Sons Private Limited" appearing across many NDAs/SHAs/JVAs/
    # Judgments). Computed BEFORE carryover, and used to GATE it below: a
    # question that names a party is a fresh topic reference, even when that
    # name is too ambiguous to resolve to one document — it must never be
    # confused with "names no document at all" and silently answered from
    # whatever document the previous, unrelated question happened to land on
    # (confirmed live: a Judgment-thread's carried scope kept answering brand-new
    # Legal-Opinion questions about Tata Sons/Consumer Products/Motors from the
    # stale Judgment 6 context, because none of those questions contained an
    # explicit document-TYPE word for _CARRYOVER_TYPE_RE to catch).
    unresolved_party = ""
    try:
        _cands = [m.group(1).strip() for m in _PARTY_NAME_RE.finditer(question)]
        _cands = [c for c in _cands if len(c) >= 4]
        if not _cands:
            _cands = [m.group(0).strip() for m in _BARE_ALLCAPS_ENTITY_RE.finditer(question)]
        if _cands:
            unresolved_party = _cands[0]
    except Exception:
        unresolved_party = ""

    # 4. Conversational carryover — the question names no document, no party
    #    (resolved OR unresolved), no entity, no family, and isn't broad. If the
    #    conversation is demonstrably already about ONE document, stay there
    #    rather than silently widening to the whole corpus. Runs last, so it can
    #    only claim questions every real detector above has already passed on
    #    (see _carryover_scope for guards) AND that named no party at all —
    #    unresolved_party being non-empty means this IS a fresh topic reference,
    #    just an ambiguous one, so it falls through to the corpus-default branch
    #    below (which discloses the ambiguity via a [SCOPE WARNING]) instead of
    #    ever reaching carryover.
    # Carryover reads THIS conversation's recent scope, so it must use the chat
    # session (where the thread's messages are stored), NOT the wiki/doc session
    # (where pages live). They diverge when a fixed main wiki is served: retrieval
    # runs against the doc session, but each chat thread's answers are saved under
    # its own session_id. Reading the doc session here made carryover inherit a
    # stale answer frozen in the ingest session (confirmed live: every no-doc
    # question in every new chat defaulted to a July-17 JVA7 answer). Falls back
    # to session_id when no separate chat session is passed (dev single-session).
    # A demonstrative pointing AT a document overrides the party gate. "What
    # are the biggest risks for Suvarna in this agreement?" names a party, so
    # the rule above treated it as a fresh topic and searched all 1,372
    # documents — but "in this agreement" is an explicit back-reference, and the
    # party name is qualifying the document already under discussion rather
    # than introducing a new one. The gate exists to stop a NEW party name
    # inheriting a stale document; it was never meant to catch a question that
    # says outright which document it means.
    _points_back = bool(_RX_DEMONSTRATIVE_DOC.search(question or ""))
    if _points_back and unresolved_party:
        logger.info("Carryover party gate overridden: question points back at "
                    "the document under discussion (%r)", unresolved_party)
    if not unresolved_party or _points_back:
        carried = _carryover_scope(question, chat_session_id or session_id)
        if carried:
            logger.info("Scope carried over from the conversation: %s",
                        _norm_doc_name(carried[0]))
            return {"scope": "single_doc", "target_docs": carried,
                    "target_family": None, "is_broad": False,
                    "confidence": 0.55, "method": "carryover"}

        # 4b. Comparative follow-up after a MULTI-document turn. The single-
        #     document carryover above cannot help here: the previous turn
        #     resolved no target_docs (it was broad/default), so without this the
        #     question widens to the whole corpus and drifts off both the topic
        #     and the documents under discussion. Same "single_doc" shape as the
        #     party-multi branch above, which likewise pins several documents, so
        #     retrieval scopes to exactly the set being compared.
        carried_set = _carryover_comparative_set(question, chat_session_id or session_id)
        if carried_set:
            logger.info("Comparison set carried over from the conversation: %d document(s): %s",
                        len(carried_set),
                        ", ".join(_norm_doc_name(d) for d in carried_set[:4]))
            return {"scope": "single_doc", "target_docs": carried_set,
                    "target_family": None, "is_broad": False,
                    "confidence": 0.5, "method": "carryover-set"}

    # 4c. Nothing pinned a single document, but the question DID name an
    #     instrument ("the Service Agreement of Tata Steel Limited"). Falling
    #     through to a whole-corpus search discards a signal the user actually
    #     gave: confirmed on the production-representative corpus, that question
    #     was answered from Service Agreement 2 when it meant Service Agreement 7.
    #     Narrow to the named family, and carry unresolved_party through so the
    #     ambiguity is still disclosed — the party genuinely was not pinned, and
    #     narrowing the search does not make it certain which document was meant.
    #
    #     Gated on unresolved_party deliberately. _detect_question_family matches
    #     its keywords anywhere in the sentence, including inside ordinary prose:
    #     "a Letter of Intent ... for integrated design consultancy SERVICES"
    #     matches the Service Agreement family although the answer lives in a
    #     court judgment, and hard-narrowing there would exclude the very document
    #     that holds it. Requiring a named-but-unpinned party keeps this to the
    #     case it was built for — the user identified a company AND an instrument
    #     type, so only WHICH document of that type is in doubt. Questions naming
    #     no party at all keep the whole-corpus search they already answer well
    #     from (measured: Q14/Q67/Q70 score 8-9 on the unscoped path).
    if _fam_name and unresolved_party:
        #     Before widening to the whole family, spend the party name against it.
        #     An umbrella party is unusable alone and a family is unusable alone,
        #     but their intersection is often a single document (see
        #     _resolve_party_within_family). Pinning that document — rather than
        #     handing 60+ siblings to a broad, diversified retrieval that returns
        #     roughly one page each — is what lets the answer see the whole
        #     instrument instead of an arbitrary page of it.
        _narrowed = _resolve_party_within_family(question, session_id, _fam_docs)
        #     The party alone often isn't enough within Judgments/Court-case
        #     families, where the same plaintiff files against many unrelated
        #     defendants — the party-in-family intersection there is the whole
        #     family, not one document. A quoted subject the question names
        #     ("operators of 'Tata Restart'") is the second signal those
        #     questions actually carry; spend it before giving up to the
        #     unresolved whole-family fallback below. See
        #     _narrow_by_quoted_subject.
        if len(_narrowed) != 1:
            _quoted_narrowed = _narrow_by_quoted_subject(
                question, session_id, _narrowed or _fam_docs)
            if _quoted_narrowed and len(_quoted_narrowed) <= _PARTY_IN_FAMILY_MAX_DOCS:
                logger.info("Quoted subject in %r narrowed party-in-family %d(%s"
                            ") document(s) to %d: %s",
                            question, len(_narrowed or _fam_docs), _fam_name,
                            len(_quoted_narrowed),
                            ", ".join(sorted(_norm_doc_name(d) for d in _quoted_narrowed)))
                _narrowed = _quoted_narrowed
        if _narrowed and len(_narrowed) <= 4:
            logger.info("Party %r within the %s family → %d document(s): %s",
                        unresolved_party, _fam_name, len(_narrowed),
                        ", ".join(sorted(_norm_doc_name(d) for d in _narrowed)))
            #     One document is a genuine resolution, so it carries no warning.
            #     Several means the instrument type still hides which one was
            #     meant, so unresolved_party rides along and the answer discloses.
            _scoped = {"scope": "single_doc", "target_docs": sorted(_narrowed),
                       "target_family": None, "is_broad": False,
                       "confidence": 0.78 if len(_narrowed) == 1 else 0.65,
                       "method": "party-in-family"}
            if len(_narrowed) > 1:
                # Several documents fit the question equally: the party is in all
                # of them and so is the instrument type, and the question offers
                # nothing else to choose by. Reporting the top-ranked one as THE
                # answer is what scored Q85 6/10 twice — "28 August 2025" from
                # Service Agreement 4, with no sign that Service Agreement 2
                # answers the same question with 18 July 2025. Hand the answer
                # side the matched set so every value it reports is attributed to
                # the document it came from (see the ambiguity_directive in
                # intent_agent.generate_answer_node).
                #
                # unresolved_party rides along as it always did, so if the
                # directive is ever dropped the weaker warning still fires.
                _scoped["unresolved_party"] = unresolved_party
                _what = f'"{unresolved_party}" in the {_fam_name} family'
                _desc = _extract_descriptive_identifier(question)
                if _desc:
                    # The question also recited boilerplate — a registered-office
                    # block — which the reader almost certainly believed was
                    # doing the identifying. Name it, so the disclosure answers
                    # the question they will actually ask: why wasn't that enough?
                    _what += f' (the address you gave, "{_desc}", does not '
                    _what += 'separate them)'
                _scoped["ambiguous_match"] = {"description": _what,
                                              "docs": sorted(_narrowed)}
            return _scoped
        return {"scope": "family", "target_docs": sorted(_fam_docs),
                "target_family": _fam_name, "is_broad": True,
                "confidence": 0.55, "method": "default-family",
                "unresolved_party": unresolved_party}

    # 5. Default — search the whole corpus, narrow (unchanged pre-Phase-2 path).
    return {"scope": "corpus", "target_docs": [], "target_family": None,
            "is_broad": False, "confidence": 0.5, "method": "default",
            "unresolved_party": unresolved_party}


def classify_query(question: str, session_id: str) -> dict:
    """Determine if the query targets a specific unnamed document.

    Uses a fast LLM call. Returns {needs_disambiguation, documents}.
    """
    index = _load_index(session_id)
    pages = index.get("pages", {})

    # Skip if the question already names a specific file
    mentioned = _detect_mentioned_files(question, pages)
    if mentioned:
        return {"needs_disambiguation": False, "documents": []}

    # Get distinct source documents
    if config.USE_DATABASE:
        docs = _db.get_source_docs(_active_wiki_id(), session_id)
    else:
        docs = list({
            p.get("source_doc", "") for p in pages.values()
            if isinstance(p, dict) and p.get("source_doc")
        })

    if len(docs) <= 1:
        return {"needs_disambiguation": False, "documents": docs}

    # Deliberately broad phrasing ("across all NDAs", "across the court case
    # documents") is not an ambiguous reference to ONE document — it asks for a
    # synthesis over many, and resolve_scope already knows how to scope it that
    # way. Without this, _VAGUE_DOC_PATTERN below matches the type word inside
    # the broad phrase ("the court case …") and prompts "which one?" for a
    # question that named no single document and wanted none (confirmed live on
    # Q40, which then got answered from a single CCD instead of across them).
    # Checked before the party resolver: explicit broad phrasing outranks an
    # incidental party mention.
    #
    # A bare plural with no breadth word ("our NDAs", "these judgments") means
    # the same thing but doesn't match _BROAD_SCOPE_RE — that's what
    # _PLURAL_FAMILY_HINT_RE is for, and resolve_scope already ORs the two
    # together for its own family check (see below). This gate had only checked
    # _BROAD_SCOPE_RE, so "our NDAs" fell through to the vague-reference check
    # and asked "which one?" for a question that named no single document
    # (confirmed live: resolve_scope handles it fine once it gets there — this
    # gate just never let it).
    if _BROAD_SCOPE_RE.search(question) or _PLURAL_FAMILY_HINT_RE.search(question):
        logger.info("Broad/plural-family phrasing → skip disambiguation")
        return {"needs_disambiguation": False, "documents": docs}

    # A named party that resolves via full-text content search is an unambiguous
    # document reference — skip disambiguation and let resolve_scope (which runs
    # the SAME resolver later) pin it. This catches the common case the
    # distinctive-entity check above misses: a counterparty named by its full
    # corporate name ("Helios Grid Advisory Private Limited"), or by bare
    # shorthand with no suffix at all ("Brackenpyre"), whose name lives only in
    # the document BODY, not the filename or the page-title identifier tokens
    # _extract_doc_entities mines — so neither entity check above fires and the
    # question would otherwise trigger a needless "which document?" prompt even
    # though the party pins it precisely (confirmed live: SA5/Helios and
    # SA6/Meridian questions both disambiguated despite each party name resolving
    # to a single Service Agreement).
    #
    # ANY non-empty result skips here, not just a single document. That used to
    # require len == 1, on the reasoning that "a party shared across several
    # documents stays genuinely ambiguous" — true for an UNBOUNDED umbrella name,
    # but _resolve_docs_by_party's own contract already rules that case out
    # before ever returning: it gives up and returns empty whenever the smallest
    # candidate set exceeds max_docs (default 4). So a non-empty result here is
    # already guaranteed small and coherent — a deal's own linked instrument
    # family (MSA+SOW+DPA), not an unrelated pile of documents that happen to
    # share a common name. Confirmed live: "Brackenpyre" resolves to exactly its
    # own 3 documents and was still disambiguated under the == 1 version, because
    # a bare-name party question almost never resolves to just one instrument —
    # requiring that never fires for the shape this whole check exists to catch.
    #
    # _resolve_docs_by_party is DB-gated and returns an empty set with no party
    # phrase present, so questions that name no corporate party incur no extra
    # cost.
    try:
        party_docs = _resolve_docs_by_party(question, session_id)
    except Exception as e:
        logger.error("classify_query: party resolution failed: %s", e)
        party_docs = set()
    if party_docs:
        logger.info("Named party resolves to %d document(s) → skip disambiguation: %s",
                    len(party_docs), {_norm_doc_name(d) for d in party_docs})
        return {"needs_disambiguation": False, "documents": docs}

    # Same idea, for a question that names no party but does recite an explicit
    # date it already knows ("the SA dated 15 January 2026", "the Legal Opinion
    # from 3 March 2025"). Confirmed live: several date-only and keyword-only
    # document references (a distinctive date, "EV charging infrastructure")
    # needlessly disambiguated despite uniquely identifying one document —
    # resolve_scope runs the SAME date resolver later via _resolve_docs_by_date,
    # so this only skips the redundant round-trip when it's already unambiguous.
    try:
        date_docs = _resolve_docs_by_date(question, session_id)
    except Exception as e:
        logger.error("classify_query: date resolution failed: %s", e)
        date_docs = set()
    if len(date_docs) == 1:
        logger.info("Dated reference resolves to a single document → skip disambiguation: %s",
                    _norm_doc_name(next(iter(date_docs))))
        return {"needs_disambiguation": False, "documents": docs}

    # Same idea a third time, for the pairing that actually identifies Q85's
    # document: an UMBRELLA party name plus the instrument the question names.
    # Neither is usable alone — "Tata Sons Private Limited" appears in 10
    # documents, the Service Agreement family holds 62 — but their intersection
    # is exactly the two service agreements the question could mean. The vague
    # branch below claimed the question first on the type word alone and asked
    # "which of 68 service agreements?", so that intersection was never spent.
    #
    # Skips whether the intersection is one document or several, and the
    # difference is the whole point: one means resolve_scope will pin it, and
    # several means the question has that many valid answers, which resolve_scope
    # surfaces with each value attributed to its own document. Asking "which
    # agreement?" is wrong in both cases — in the first the user already said,
    # and in the second the honest reply is that their description names more
    # than one. Suppression only; this can never RAISE a prompt, so it cannot
    # reopen the documented disambiguation stuck-loop.
    try:
        _fam_docs_gate = _question_family_scope(question, session_id)[1]
        party_in_fam = (_resolve_party_within_family(question, session_id, _fam_docs_gate)
                        if _fam_docs_gate else set())
    except Exception as e:
        logger.error("classify_query: party-in-family resolution failed: %s", e)
        party_in_fam = set()
    if party_in_fam and len(party_in_fam) <= _PARTY_IN_FAMILY_MAX_DOCS:
        logger.info("Party within the named instrument family resolves to %d "
                    "document(s) → skip disambiguation, resolve_scope will scope it: %s",
                    len(party_in_fam),
                    ", ".join(sorted(_norm_doc_name(d) for d in party_in_fam)))
        return {"needs_disambiguation": False, "documents": docs}

    # Vague singular reference ("this NDA", "the agreement") with multiple matching
    # documents → ask which one. Narrow to the named type when possible.
    vmatch = _VAGUE_DOC_PATTERN.search(question)
    if vmatch and not _question_names_a_document(question, docs) \
            and not _question_names_distinctive_entity(question, pages):
        vtype = re.sub(r'\s+', ' ', vmatch.group(1).lower().strip())
        if vtype in _GENERIC_VAGUE_WORDS:
            logger.info("Vague document reference '%s' → disambiguate (all docs)", vtype)
            return {"needs_disambiguation": True, "documents": _diversify_doc_order(docs)}
        filt = _VAGUE_TYPE_FILTER.get(vtype)
        if filt:
            type_docs = [
                d for d in docs
                if filt in re.sub(r'^[a-f0-9-]{36}_', '', d).replace('_', ' ').lower()
            ]
            if len(type_docs) > 1:
                logger.info("Vague '%s' reference → disambiguate among %d %s docs",
                            vtype, len(type_docs), filt)
                return {"needs_disambiguation": True, "documents": type_docs}
        elif vtype.split()[0] in _VAGUE_TITLE_TYPE_WORDS or vtype in _VAGUE_TITLE_TYPE_WORDS:
            # Litigation-filing types (Motion, Complaint, Stipulation, ...) aren't
            # encoded in the raw filename — resolve via page-title prefixes instead.
            type_docs = _source_docs_by_title_type(pages, vtype)
            if len(type_docs) > 1:
                logger.info("Vague '%s' reference → disambiguate among %d filings",
                            vtype, len(type_docs))
                return {"needs_disambiguation": True, "documents": type_docs}

    # Skip if the question contains a recognizable document name pattern
    if _question_names_a_document(question, docs):
        return {"needs_disambiguation": False, "documents": docs}

    # Skip if the question mentions a distinctive entity/party name from the wiki.
    # Use the proper-noun-aware check so an incidental lowercase clause word that
    # leaked into the entity set can't short-circuit disambiguation here either;
    # a genuine (even lowercase) entity still falls through to the LLM triage below.
    if _question_names_distinctive_entity(question, pages):
        return {"needs_disambiguation": False, "documents": docs}

    # Clean document names for display
    clean_docs = [re.sub(r'^[a-f0-9-]{36}_', '', d) for d in docs]
    doc_list_str = "\n".join(f"- {d}" for d in clean_docs)

    prompt = (
        "You are a triage system for a legal document Q&A platform.\n"
        "Determine if the user's question is about a SPECIFIC document without naming it.\n\n"
        f"Available documents:\n{doc_list_str}\n\n"
        f"Question: {question}\n\n"
        "A question DOES NOT need disambiguation when:\n"
        "- It names or numbers a specific document (e.g. 'service agreement 1', 'NDA 3', 'the SHA')\n"
        "- It mentions specific party names, entity names, or company names (e.g. 'the ReVolt JV Agreement', 'Meridian service agreement', 'agreement between Tata Motors and ReVolt')\n"
        "- It's a cross-document comparison or general legal question\n"
        "- It mentions a document type with a number, identifier, or distinctive party/entity name\n"
        "- It mentions a distinctive project/deal codename (e.g. a named initiative or "
        "project) that only one document's content would use\n"
        "- The clause it asks about would be covered by a linked family of instruments "
        "that explicitly incorporate and cross-reference one another (e.g. an MSA plus "
        "its DPA and SOW) — that's a valid answer spanning the linked set, not an "
        "ambiguous reference needing a document pick\n\n"
        "A question NEEDS disambiguation when:\n"
        "- It uses vague references like 'this document', 'summarize it', 'the agreement' "
        "without ANY identifier, number, or party name\n"
        "- It refers to a specific clause, provision, or section (e.g. 'the indemnity clause', "
        "'limitation of liability', 'termination provisions') without specifying WHICH document "
        "contains that clause, where the candidates are UNRELATED documents (not a linked "
        "instrument family) and multiple of them could have such a clause\n\n"
        "A question does NOT need disambiguation when it is a cross-document comparison or "
        "general legal question that intentionally spans all documents.\n\n"
        "Respond with JSON only:\n"
        '{"needs_disambiguation": bool, "reason": "one sentence"}'
    )
    try:
        raw, _ = llm.ask(prompt, fast=True, max_tokens=config.MAX_TOKENS_DISAMBIGUATION)
        parsed = _parse_json_safe(raw)
        if parsed and parsed.get("needs_disambiguation"):
            return {"needs_disambiguation": True, "documents": _diversify_doc_order(docs)}
    except Exception as e:
        logger.error("classify_query failed: %s", e)

    return {"needs_disambiguation": False, "documents": docs}


def _bucket_docs_by_type(docs: list[str]) -> list[list[str]]:
    """Group source_doc filenames into buckets by known document type keywords
    (reusing _VAGUE_TYPE_FILTER's canonical types; anything unmatched goes in
    "other"). Pure string matching — no DB round-trips — so it stays cheap even
    at tens of thousands of documents.
    """
    buckets: dict[str, list[str]] = {}
    for d in docs:
        norm = re.sub(r'^[a-f0-9-]{36}_', '', d).replace('_', ' ').lower()
        bucket_key = next((v for k, v in _VAGUE_TYPE_FILTER.items() if k in norm), "other")
        buckets.setdefault(bucket_key, []).append(d)
    return list(buckets.values())


def _round_robin_buckets(bucket_lists: list[list[str]], limit: int | None = None) -> list[str]:
    """Interleave a list of buckets round-robin so the result spans every
    bucket early on, instead of exhausting one bucket before the next starts.
    """
    if not bucket_lists:
        return []
    result: list[str] = []
    idx = 0
    max_len = max(len(bl) for bl in bucket_lists)
    while idx < max_len and (limit is None or len(result) < limit):
        for bl in bucket_lists:
            if idx < len(bl):
                result.append(bl[idx])
                if limit is not None and len(result) >= limit:
                    break
        idx += 1
    return result


def _diverse_doc_sample(docs: list[str], limit: int) -> list[str]:
    """Sample up to `limit` docs spread across distinct document types.

    A raw docs[:N] head-slice is unrepresentative at scale: if the corpus is
    loaded/ordered by folder (as batch ingests typically are), a head slice can
    be dozens of the same document type, giving the ambiguity-check LLM a
    skewed sense of what's actually in the corpus — a genuinely ambiguous
    question touching document types outside that slice would never trigger
    clarification. Buckets by type and round-robins across buckets instead, so
    the sample spans the real variety of the corpus regardless of ingest order.
    """
    if len(docs) <= limit:
        return docs
    return _round_robin_buckets(_bucket_docs_by_type(docs), limit)


def _diversify_doc_order(docs: list[str]) -> list[str]:
    """Reorder (never drops) a full document list so distinct types appear
    early, instead of clustering by ingest order (e.g. 25 "Court Case
    Documents" in a row before any NDA/SHA/Service Agreement appears). Used
    for disambiguation chip lists, where every document must still be shown —
    only the order changes, so a user isn't stuck scrolling through one type
    to find another.
    """
    return _round_robin_buckets(_bucket_docs_by_type(docs))


def check_ambiguity(question: str, session_id: str, conversation_context: str = "") -> dict:
    """Determine if the query needs clarification before answering.

    Uses a fast LLM call. Returns {needs_clarification, question, options}.
    """
    if not config.ENABLE_CLARIFICATION:
        return {"needs_clarification": False}

    # Get doc types for context
    if config.USE_DATABASE:
        docs = _db.get_source_docs(_active_wiki_id(), session_id)
    else:
        index = _load_index(session_id)
        pages = index.get("pages", {})
        docs = list({
            p.get("source_doc", "") for p in pages.values()
            if isinstance(p, dict) and p.get("source_doc")
        })
    sampled_docs = _diverse_doc_sample(docs, config.AMBIGUITY_DOC_SAMPLE_CAP)
    clean_docs = [re.sub(r'^[a-f0-9-]{36}_', '', d) for d in sampled_docs]

    conv_snippet = ""
    if conversation_context:
        conv_snippet = f"\nRecent conversation:\n{conversation_context[:500]}\n"

    prompt = (
        "You are a legal assistant triage system. Determine if the user's question "
        "is clear enough to answer directly or needs ONE clarifying question.\n\n"
        "If it does, phrase clarification_question the way a knowledgeable colleague "
        "would ask across a desk — plain, direct, a little informal — not like a form "
        "or a system prompt. Avoid phrasing like \"Please specify\" or \"Could you "
        "clarify the scope of your request\"; prefer something like \"Are you asking "
        "about the payment terms or the whole agreement?\"\n\n"
        "A question needs clarification when:\n"
        "- It could mean multiple very different things\n"
        "- The scope is unclear (e.g., 'summarize' without specifying focus area)\n"
        "- Key terms are ambiguous in context\n\n"
        "A question does NOT need clarification when:\n"
        "- It's straightforward even if broad\n"
        "- It names a specific document AND states what to do with it\n"
        "- Standard legal analysis is implied\n"
        "- The intent is obvious from context or conversation history\n"
        "- It asks for a specific deliverable (table, list, summary, review, recommendation)\n"
        "- It asks about a clause/topic that a linked family of instruments would "
        "cover together (e.g. an MSA plus its DPA and SOW, which explicitly "
        "incorporate and cross-reference one another) — answer using all of "
        "them rather than asking which one\n\n"
        "When in doubt, answer directly — do NOT ask for clarification.\n\n"
        f"Available documents: {', '.join(clean_docs)}\n"
        f"{conv_snippet}\n"
        f"Question: {question}\n\n"
        "Respond with JSON only:\n"
        '{"needs_clarification": bool, "clarification_question": "string or null", '
        '"options": ["option1", "option2", "option3"] or null}'
    )
    try:
        raw, _ = llm.ask(prompt, fast=True, max_tokens=config.MAX_TOKENS_AMBIGUITY_CHECK)
        parsed = _parse_json_safe(raw)
        if parsed and parsed.get("needs_clarification"):
            return {
                "needs_clarification": True,
                "question": parsed.get("clarification_question", "Could you clarify your question?"),
                "options": parsed.get("options") or [],
            }
    except Exception as e:
        logger.error("check_ambiguity failed: %s", e)

    return {"needs_clarification": False}


# How much of the conversation the ANSWER MODEL gets to see. Distinct from scope
# carryover, which is unbounded (a "carryover" turn is itself inheritable, so the
# document chains forward indefinitely). That asymmetry was the real gap: by turn
# 8 retrieval still targeted the right document while the model could no longer
# see the turn where "the second one" was named.
#
# Widened 6→10 messages (3 exchanges → 5). Deliberately modest: this text feeds
# the answer prompt, and more prior conversation is also more opportunity for the
# model to answer FROM the conversation instead of from the retrieved documents —
# the drift failure mode this codebase has repeatedly had to close. Answers stay
# clipped to a short trail rather than reproduced in full, so the window grows in
# turns without growing much in tokens.
_CONV_CONTEXT_MESSAGES = 10
_CONV_CONTEXT_BUDGET = 3000
_CONV_ANSWER_CLIP = 300


def build_conversation_context(session_id: str) -> str:
    """Build a conversation context string from recent chat messages."""
    if not config.USE_DATABASE:
        return ""
    try:
        recent = _db.get_recent_context(session_id, n=_CONV_CONTEXT_MESSAGES)
    except Exception:
        return ""

    if not recent:
        return ""

    # Built NEWEST-FIRST, then reversed. The previous version walked oldest→newest
    # and broke at the budget, so whenever the budget actually bound it kept the
    # OLDEST messages and dropped the most recent ones — backwards for resolving
    # "that clause"/"the second one", which always refer to the latest turns. The
    # bug was mostly dormant at n=6 (~1.1k chars, well under budget) and starts
    # firing as soon as the window widens, so it is fixed here first.
    parts = []
    total_chars = 0
    for msg in reversed(recent):
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if msg["msg_type"] == "answer" and len(content) > _CONV_ANSWER_CLIP:
            content = content[:_CONV_ANSWER_CLIP] + "..."
        line = f"{role}: {content}"
        if total_chars + len(line) > _CONV_CONTEXT_BUDGET:
            break
        parts.append(line)
        total_chars += len(line)

    return "\n".join(reversed(parts))


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


def _rrf_fuse(rankings: list[list[str]], k: int = 60, limit: int | None = None) -> list[str]:
    """Reciprocal Rank Fusion of several ranked title lists into one (Phase 3).

    RRF score for a title = sum over each list of 1/(k + rank), rank 1-based. A
    title ranked highly by EITHER retriever (vector OR BM25) rises; one ranked
    well by both rises most. This properly merges the two hybrid rankings —
    unlike the previous "all vector results, then BM25 appended" which buried a
    strong keyword-only match below every semantic hit. Zero LLM calls. Ties
    (same fused score) preserve first-seen order via a stable sort.
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    seq = 0
    for ranking in rankings:
        for rank, title in enumerate(ranking, start=1):
            scores[title] = scores.get(title, 0.0) + 1.0 / (k + rank)
            if title not in order:
                order[title] = seq
                seq += 1
    fused = sorted(scores, key=lambda t: (-scores[t], order[t]))
    return fused[:limit] if limit else fused


def _rerank_pages(question: str, candidate_titles: list[str], pages: dict,
                  limit: int | None = None) -> list[str]:
    """Optional fast-model relevance rerank of already-retrieved candidates.

    Gated behind config.ENABLE_RERANK and only applied to broad/family queries
    by the caller. Reuses the same compact "- title: summary" candidate index as
    PAGE_SELECT_PROMPT, but asks the model to ORDER by relevance rather than
    select a subset. Fail-safe: on any error or unparseable output, returns the
    input order unchanged (never drops the pipeline into a worse state).
    """
    if not candidate_titles:
        return candidate_titles
    index_lines = []
    for t in candidate_titles:
        page = pages.get(t)
        summary = page.get("summary", "") if isinstance(page, dict) else ""
        index_lines.append(f"- {t}: {summary}" if summary else f"- {t}")
    prompt = (
        "Rank these legal wiki pages by how directly relevant each is to answering "
        "the QUESTION. Respond with ONLY valid JSON, no other text:\n"
        '{"ranking": ["<most relevant title>", "<next>", ...]}\n'
        "Include every title exactly once, most relevant first.\n\n"
        f"QUESTION: {question}\n\nPAGES:\n" + "\n".join(index_lines)
    )
    try:
        raw, _usage = llm.ask(prompt, fast=True, max_tokens=config.MAX_TOKENS_RERANK)
        # The fast gpt-oss reasoning model can spend its whole budget on hidden
        # reasoning and emit nothing when the cap is too low (same failure mode as
        # the grounding check). One doubling retry recovers those cases before
        # falling back to fusion order.
        if _usage.get("finish_reason") == "length":
            raw, _usage = llm.ask(prompt, fast=True, max_tokens=config.MAX_TOKENS_RERANK * 2)
        parsed = _parse_json_safe(raw)
        ranking = parsed.get("ranking") if isinstance(parsed, dict) else None
        if isinstance(ranking, list):
            cand_set = set(candidate_titles)
            valid = [t for t in ranking if t in cand_set]
            # Append any titles the model dropped, preserving their prior order,
            # so a lazy/truncated ranking never silently loses candidates.
            seen = set(valid)
            ranked = valid + [t for t in candidate_titles if t not in seen]
            return ranked[:limit] if limit else ranked
    except Exception as e:
        logger.warning("LLM rerank failed, keeping fusion order: %s", e)
    return candidate_titles[:limit] if limit else candidate_titles


def _select_relevant_pages(
    pages: dict, question: str, session_id: str | None = None,
    doc_family: "str | list[str] | None" = None, force_broad: bool = False,
    exclude_cached_answers: bool = False,
) -> tuple[list[str], dict]:
    """Select the most relevant pages for a question.

    Priority order:
      1. pgvector cosine similarity search (DB mode only, Phase 3) — 0 LLM calls, ~5ms
      2. BM25 pre-filter + LLM selection (fallback or file mode)
      3. BM25 keyword-only (if LLM selection fails)

    doc_family (Phase 1): when scope resolution (Phase 2) narrows a question to a
    document family, it's passed here to pre-filter the pgvector search to that
    family's embeddings. None = unfiltered whole-session search (default).

    exclude_cached_answers: forwarded to the pgvector search so cached "Q:" pages
    are excluded by the SQL rather than discarded after the LIMIT. The caller has
    already dropped them from `pages`, and filtering only afterwards costs the
    entire vector budget — see search_similar_pages for the measured case.

    force_broad (Phase 2): when scope resolution classifies a question as family
    or broad, force the wide+diversified candidate path even if the local
    _BROAD_SCOPE_RE regex wouldn't have fired on the phrasing (e.g. "compare the
    NDAs"). Only ever WIDENS — never narrows — so it can't regress existing broad
    detection.

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
    if not config.USE_DATABASE:
        logger.info(
            "Hybrid retrieval unavailable: app is in FILE mode (DATABASE_URL not set). "
            "pgvector search requires PostgreSQL — using BM25+LLM page selection."
        )
    if config.USE_DATABASE and session_id:
        try:
            _wiki_id_hr = _active_wiki_id()
            emb_count = _db.count_embeddings(_wiki_id_hr, session_id)
            if emb_count == 0:
                logger.info(
                    "Hybrid retrieval skipped: 0 embeddings in DB for session %s "
                    "(embeddings absent — likely failed/rate-limited at ingest). "
                    "Using BM25+LLM. Run backfill_embeddings() to enable pgvector search.",
                    session_id,
                )
            if emb_count > 0:
                from services import embedder as _embedder
                q_embedding = _embedder.embed(question, is_query=True)
                is_broad = force_broad or bool(_BROAD_SCOPE_RE.search(question))
                vector_limit = config.BROAD_QUESTION_VECTOR_TOP_K if is_broad else config.VECTOR_SEARCH_TOP_K
                vector_titles = _db.search_similar_pages(
                    _wiki_id_hr, session_id, q_embedding, limit=vector_limit, doc_family=doc_family,
                    exclude_cached=exclude_cached_answers,
                )
                # Validate titles against the in-memory pages dict (guards against
                # stale embeddings pointing at deleted pages)
                valid_vector = [t for t in vector_titles if t in pages]

                # BM25 ranking for fusion — pull a comparable-length keyword list
                # (not just the old small supplement) so RRF has two real rankings
                # to merge, not one ranking plus a handful of extras.
                bm25_ranking = [
                    t for t in _keyword_fallback_pages(pages, question, n=vector_limit)
                    if t in pages
                ]

                # Hypothetical-question vectors as a THIRD ranking. Ingest writes
                # a set of questions each page can answer and embeds them (stage
                # 06); this corpus holds 16,042 of them, and until now nothing
                # read them back — the search function existed with no caller.
                # They match a different thing from the page embedding: the page
                # vector encodes what a page SAYS, the question vector what it can
                # be ASKED, so a query phrased as a question lands closer to them.
                # Costs no extra embedding call — q_embedding is already made.
                #
                # OFF by default (config.USE_QUESTION_EMBEDDINGS). The questions
                # this corpus's ingest produced discriminate TOPIC, not document:
                # "How does the Agreement define 'Confidential Information'?" is
                # the stored question for 124 separate pages, all scoring
                # identically. RRF promotes whatever any channel ranks highly, so
                # feeding it a ranking that orders documents arbitrarily is a
                # route to an unrelated agreement's page in the context — the
                # exact failure the scope work has been closing. Turning it on
                # needs a live retrieval comparison, not a reading of the code.
                question_ranking = []
                if config.USE_QUESTION_EMBEDDINGS:
                    try:
                        question_ranking = [
                            r["title"] for r in _db.search_similar_questions(
                                _wiki_id_hr, session_id, q_embedding,
                                limit=vector_limit, doc_family=doc_family,
                                max_pages_sharing=config.QUESTION_MAX_PAGES_SHARING)
                            if r["title"] in pages
                        ]
                    except Exception as _qe:
                        # Never fatal: this is a third opinion on top of two
                        # rankings that already work on their own.
                        logger.warning("question-embedding search failed: %s", _qe)

                # Phase 3: Reciprocal Rank Fusion of the vector and BM25 rankings,
                # replacing the previous "all vector, then BM25 appended" order —
                # a strong keyword-only match now ranks on its own merit instead of
                # sitting below every semantic hit. Zero LLM calls.
                _rankings = [valid_vector, bm25_ranking]
                if question_ranking:
                    _rankings.append(question_ranking)
                if is_broad:
                    # Fuse first, THEN diversify: the per-document cap + Parties-page
                    # force-include operate on a better-ordered base list, but the
                    # breadth guarantee for "across all X" questions is unchanged.
                    fused = _rrf_fuse(_rankings, k=config.RRF_K)
                    hybrid = _diversify_by_document(
                        fused, pages,
                        config.BROAD_QUESTION_PER_DOC_CAP, config.BROAD_QUESTION_TOTAL_CAP,
                    )
                    logger.info(
                        "Broad question — widened to %d vector candidates, RRF-fused with "
                        "BM25, diversified to %d pages across documents",
                        vector_limit, len(hybrid),
                    )
                else:
                    hybrid = _rrf_fuse(
                        _rankings,
                        k=config.RRF_K, limit=config.HYBRID_FUSION_TOP_K,
                    )

                # Optional LLM rerank (off by default) — only for broad/family
                # queries, where retrieval precision matters most and the extra
                # fast-model call is worth its latency. RRF already gives a strong
                # base order, so this is a refinement, not a dependency.
                if hybrid and config.ENABLE_RERANK and is_broad:
                    before = len(hybrid)
                    hybrid = _rerank_pages(question, hybrid, pages, limit=before)
                    logger.info("LLM rerank applied to %d broad-query candidates", before)

                if hybrid:
                    logger.info(
                        "Page selection: %d pages via hybrid RRF fusion "
                        "(vector=%d, bm25=%d, questions=%d, embeddings_in_db=%d, broad=%s)",
                        len(hybrid), len(valid_vector), len(bm25_ranking),
                        len(question_ranking), emb_count, is_broad,
                    )
                    _trace = tracing.get_trace()
                    if _trace:
                        _trace.log_page_selection(
                            "vector+bm25+questions RRF fusion",
                            vector=valid_vector, bm25=bm25_ranking,
                            questions=question_ranking,
                            selected=hybrid, embeddings_in_db=emb_count, is_broad=is_broad,
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
                _trace = tracing.get_trace()
                if _trace:
                    _trace.log_page_selection(
                        "BM25 prefilter + LLM select", candidates=candidate_titles, selected=valid,
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
    _trace = tracing.get_trace()
    if _trace:
        _trace.log_page_selection("BM25 keyword-only fallback", selected=fallback)
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
