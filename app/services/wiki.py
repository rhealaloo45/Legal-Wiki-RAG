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
from functools import lru_cache

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
    clean = re.sub(r'\.pdf$', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\.txt$', '', clean, flags=re.IGNORECASE)
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
    from services.reader import read_file_with_positions as _read_with_pos
    result = _read_with_pos(file_path)
    text = result["text"]
    page_map = result["page_map"]
    doc_name = os.path.basename(file_path)

    # Store page-level positions for citation location support
    if config.USE_DATABASE and page_map:
        try:
            _db.store_page_map(session_id, doc_name, page_map)
        except Exception as _pm_err:
            logger.warning("Failed to store page map for %s: %s", doc_name, _pm_err)

    logger.info("Wiki ingest: %s (%d chars, %d pages)", doc_name, len(text), len(page_map))

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

            existing = _db.get_page(session_id, title)

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
    _embed_pages_batch(session_id, pages_to_embed, doc_family_for_batch)

    return pages_updated, new_rels_count, len(contradictions_found)


# ---------------------------------------------------------------------------
# Embedding helper (Phase 3) — called OUTSIDE the session lock
# ---------------------------------------------------------------------------
def _embed_pages_batch(session_id: str, pages_to_embed: list[tuple[str, str]],
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
            _db.upsert_embedding(session_id, title, embedding, doc_family)
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

    pages = _db.get_pages(session_id)
    if not pages:
        return {"ok": False, "reason": "no pages in session", "embedded": 0}

    existing = _db.count_embeddings(session_id)

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
        _embed_pages_batch(session_id, chunk)  # logs + swallows failures per batch
        embedded += len(chunk)

    final = _db.count_embeddings(session_id)
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


def get_context(question: str, session_id: str, target_doc: str = "", retrieval_hints: dict = None,
                 exclude_cached_answers: bool = False,
                 doc_family: "str | list[str] | None" = None, force_broad: bool = False,
                 force_docs: "list[str] | None" = None) -> tuple[str, list]:
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
    fresh generation to actually observe whether behavior changed.
    """
    index = _load_index(session_id)
    pages = index.get("pages", {})
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
        logger.info("Single-document scope (%s): scoped to %d page(s), supplementary retrieval skipped",
                     target_doc or f"party:{sorted(forced_set)}", len(file_pages))
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
            llm_selected, page_selection_usage = _select_relevant_pages(pages_for_llm, question, session_id)
            supplementary = _drop_colliding(llm_selected)
            selected_titles = file_pages + supplementary
        logger.info("File-focused query: %d pages from mentioned file(s), %d total selected",
                     len(file_pages), len(selected_titles))
    else:
        # No file mentioned — original behaviour, now with optional family
        # pre-filter + broad-widen forwarded from the resolved scope (Phase 2).
        if len(pages) <= 20:
            selected_titles = list(pages.keys())
        else:
            selected_titles, page_selection_usage = _select_relevant_pages(
                pages_for_llm, question, session_id,
                doc_family=doc_family, force_broad=force_broad,
            )

    # --- Step 2: Build context string from selected pages ---
    # Q: pages are cached prior answers — cap so they don't crowd out source content.
    # Regular pages: cap at MAX_PAGE_CONTEXT_CHARS to bound total prompt size.
    # Shared concept pages (no case prefix, multi-source) can grow large through merges;
    # capping ensures one bloated page doesn't consume half the context window.
    _QPAGE_CAP = config.MAX_QPAGE_CONTEXT_CHARS
    _PAGE_CAP  = config.MAX_PAGE_CONTEXT_CHARS

    wiki_parts = []
    # When retrieval is file-focused, prepend a header so the LLM knows which
    # document the pages come from (handles "Services Agreement" vs "Service Agreement").
    if file_pages and mentioned_files:
        doc_names = [re.sub(r'\b(redacted|Redacted|_)\b', ' ', os.path.splitext(
            d.replace("\\", "/").rsplit("/", 1)[-1])[0]).strip()
            for d in mentioned_files]
        doc_names = [re.sub(r'\s+', ' ', d) for d in doc_names]
        wiki_parts.append(f"[The following pages are from: {', '.join(doc_names)}]\n")

    _TOTAL_CAP = config.MAX_TOTAL_CONTEXT_CHARS
    total_chars = sum(len(p) for p in wiki_parts)
    pages_omitted = 0

    for title in selected_titles:
        if title in pages:
            if total_chars >= _TOTAL_CAP:
                pages_omitted += 1
                continue

            page = pages[title]
            content = page.get("content", "") if isinstance(page, dict) else page

            if isinstance(page, dict) and page.get("contradiction_flagged"):
                content = "[WARNING: This page contains conflicting claims. Surface the conflict explicitly in your answer. Do not resolve it.]\n" + content

            if title.startswith("Q:") and len(content) > _QPAGE_CAP:
                content = content[:_QPAGE_CAP] + "\n[...truncated — cached answer summary only]"
            elif len(content) > _PAGE_CAP:
                content = content[:_PAGE_CAP] + "\n[...truncated]"

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
                meta = _db.get_metadata(session_id, doc)
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
_QUOTE_SPAN_RE = re.compile(r'["“]([^"“”|\n]{15,500})["”]')

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


def _strict_verification_corpus(context: str) -> str:
    """Build the citation-verification text corpus, restricted to Supporting
    Quotes blocks per page (see _block_verification_text). Falls back to the
    raw context unchanged if no '## Title' page blocks are found at all.
    """
    blocks = _PAGE_BLOCK_RE.findall(context)
    if not blocks:
        return context
    return '\n'.join(_block_verification_text(title, body) for title, body in blocks)


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


def _strip_placeholder_quotes(answer: str) -> str:
    """Remove quote-wrapped placeholder stand-ins (e.g. "Not provided in excerpt")
    from the ends of reference/citation lines. Leaves the rest of the line — the
    real FileName + Clause/Section citation — intact, so a reference with no
    verbatim quote simply ends after its clause reference, as the prompts require.
    """
    if not answer or '"' not in answer and "'" not in answer and '(' not in answer:
        return answer
    return _PLACEHOLDER_QUOTE_RE.sub('', answer)


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


def generate_answer(question: str, wiki_content: str, selected_titles: list, session_id: str, bm25_count: int = 0, page_selection_usage: dict = None, conversation_context: str = "", intent: str = "factual", unconfirmed_doc_reference: bool = False, scope_note: str = "", scope_warning: str = "", clause_directive: str = "") -> dict:
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

    # Pick prompt based on the classified lawyer intent (intent_agent upstream)
    _intent_prompt_map = {
        "factual": ANSWER_PROMPT,
        "risk_assessment": ASSESSMENT_PROMPT,
        "comparison": COMPARISON_PROMPT,
        "obligation": OBLIGATION_PROMPT,
        "drafting": DRAFTING_PROMPT,
    }
    prompt_template = _intent_prompt_map.get(intent, ANSWER_PROMPT)
    prompt = _unconfirmed_doc_note + _clause_directive_note + prompt_template.format(
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
    _was_empty = len(answer.strip()) < _MIN_VIABLE_ANSWER_CHARS
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
        })
        _retry_ok = retry_usage.get("finish_reason") != "length" and len(retry_answer.strip()) >= _MIN_VIABLE_ANSWER_CHARS
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

    # Strip quote-wrapped placeholder stand-ins ("Not provided in excerpt", "(not
    # provided here)", etc.) from reference lines BEFORE the integrity checks — they
    # are a known nano non-compliance the prompts already forbid, and stripping them
    # here both cleans the output and stops them from tripping the citation check
    # (which correctly flags "Not provided in excerpt" as a non-verbatim quote).
    answer = _strip_placeholder_quotes(answer)

    # Deterministic citation-integrity checks: flag any quoted span the model
    # presented as verbatim that doesn't actually appear in the retrieved
    # context (paraphrase dressed up as an exact quote), and any quote
    # attributed to the wrong document.
    _unverified_quotes = _verify_answer_citations(answer, wiki_content, question)
    _misattributed = _verify_citation_attribution(answer, wiki_content)

    # Corrective retry: give the model one chance to fix flagged quotes — either
    # match them verbatim to context or drop the quotation marks — instead of
    # just warning the user after the fact. Only retried once; if the retry
    # doesn't measurably improve things, the original answer is kept and a
    # warning is appended as before.
    if _unverified_quotes or _misattributed:
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
        _retry_prompt = prompt + _CITATION_RETRY_ADDENDUM.format(flagged="\n".join(_flag_lines))
        retry_answer, retry_usage, retry_score, retry_reason = _run_generation_pass(_retry_prompt)
        retry_unverified = _verify_answer_citations(retry_answer, wiki_content, question)
        retry_misattributed = _verify_citation_attribution(retry_answer, wiki_content)

        token_breakdown.append({
            "call": "citation_retry",
            "model": llm.active_model(fast=False),
            "prompt_tokens": retry_usage.get("prompt_tokens", 0),
            "completion_tokens": retry_usage.get("completion_tokens", 0),
            "total_tokens": retry_usage.get("prompt_tokens", 0) + retry_usage.get("completion_tokens", 0),
        })

        _fewer_issues = len(retry_unverified) + len(retry_misattributed) < len(_unverified_quotes) + len(_misattributed)
        # A citation fix must not gut the answer. The retry sometimes comes back
        # far shorter — e.g. only the reasoning plan, or a truncated table — which
        # trivially has "fewer" unverified quotes simply because it has fewer
        # quotes (or none). Confirmed live: a 9,974-char comparison answer was
        # replaced by an 818-char plan-only answer that "passed" this check.
        # Require the retry to retain most of the original's length to count.
        _retained_length = len(retry_answer) >= 0.6 * len(answer)
        if _fewer_issues and _retained_length:
            logger.info(
                "Citation retry improved answer: %d->%d unverified quotes, %d->%d misattributed",
                len(_unverified_quotes), len(retry_unverified),
                len(_misattributed), len(retry_misattributed),
            )
            answer, usage, confidence_score, confidence_reason = retry_answer, retry_usage, retry_score, retry_reason
            _unverified_quotes, _misattributed = retry_unverified, retry_misattributed
        elif _fewer_issues and not _retained_length:
            logger.info(
                "Citation retry had fewer issues but was drastically shorter "
                "(%d vs %d chars) — keeping fuller original answer",
                len(retry_answer), len(answer),
            )
        else:
            logger.info("Citation retry did not improve verification — keeping original answer")

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
            answer += (
                f"\n\n[CITATION WARNING: {len(_absent_quotes)} quoted passage(s) above do not "
                f"appear anywhere in the retrieved source text — do not rely on them as quotes "
                f"without checking the document: {_preview_of(_absent_quotes)}]"
            )
        if _prose_quotes:
            answer += (
                f"\n\n[CITATION NOTE: {len(_prose_quotes)} passage(s) above match the retrieved "
                f"material but not its verified excerpts — read them as paraphrase rather than "
                f"exact wording: {_preview_of(_prose_quotes)}]"
            )

    if _misattributed:
        logger.warning("Citation-attribution check: %d quote(s) attributed to the wrong document: %s",
                        len(_misattributed), _misattributed)
        answer += (
            f"\n\n[ATTRIBUTION WARNING: {len(_misattributed)} quote(s) above appear to be attributed "
            f"to the wrong document — {'; '.join(_misattributed[:2])}]"
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
            f"NO ingested content and contributed nothing to this answer — {_names}. What you "
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
    if confidence["score"] >= 80 and not _unverified_quotes and not _misattributed:
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
        "selected_titles": selected_titles,
        "relations": relations,
        "usage": usage,
        "confidence_score": confidence["score"],
        "confidence_reason": confidence["reason"],
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
            "total": len(_QUOTE_SPAN_RE.findall(answer)),
            "unverified": len(_unverified_quotes),
            "misattributed": len(_misattributed),
        },
        "token_breakdown": token_breakdown,
        "token_total": token_total,
    }


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
    document identifier (e.g. "ReVolt", "Meridian", "Yuvraj Kanther")."""
    q = question.lower()
    return any(_contains_token(ent, q) for ent in _extract_doc_entities(pages))


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
    this document…" listing "term, termination, liability" — none proper nouns)."""
    q = question.lower()
    return any(
        _contains_token(ent, q) and _appears_as_proper_noun(ent, question)
        for ent in _extract_doc_entities(pages)
    )


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
        if re.search(rf'\b{re.escape(keyword)}s?\b', q):
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
    r'LLP|LLC|Inc\.?|Corp(?:oration)?|PLC|GmbH|N\.?V\.?|S\.?A\.?)'
)
_PARTY_NAME_RE = re.compile(
    r'\b((?:[A-Z][A-Za-z0-9&.\-]+\s+){1,6}?)' + _CORP_SUFFIX_RE_STR + r'\b'
)


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
    fewest documents — provided that set is small (<= max_docs). An umbrella name
    like "Tata Steel Limited" hits many documents and is correctly ignored; the
    specific counterparty resolves to one document ("SteelLoop Resource Recovery"
    → JVA 3) or, when the same two parties share several instruments, to that
    small cluster ("Tata Steel & NordForge Metallurgy" → the NDA + arbitration
    notice + Section 9 petition). The caller decides, from how many instruments
    the question names, whether to pin the whole cluster or narrow to one. Returns
    an empty set on ambiguity (no hit, or the smallest set exceeds max_docs), so
    it only ever ADDS precise matches the filename/entity detectors miss.
    """
    if not config.USE_DATABASE:
        return set()
    candidates = [m.group(1).strip() for m in _PARTY_NAME_RE.finditer(question)]
    candidates = [c for c in candidates if len(c) >= 4]
    if not candidates:
        return set()
    best_docs: set[str] | None = None
    best_n = 1 << 30
    for name in candidates:
        try:
            docs = [d for d in _db.find_source_docs_mentioning_phrase(session_id, name, cap=max_docs + 2) if d]
        except Exception as e:
            logger.error("resolve_scope: party-content lookup failed for %r: %s", name, e)
            continue
        if docs and len(docs) < best_n:
            best_n, best_docs = len(docs), set(docs)
    if best_docs is not None and best_n <= max_docs:
        logger.info("Party-name content match → %d document(s): %s",
                    best_n, {_norm_doc_name(d) for d in best_docs})
        return best_docs
    return set()


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
    r'shareholders?\s+agreement|joint\s+venture|jva|sha|msa|deed|lease|licen[cs]e|'
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


# Scope-resolution methods that pinned specific documents and are therefore safe
# to inherit. A "family"/"broad"/"default" answer has no specific scope to pass
# on. "carryover" is included so a multi-turn thread stays on the same document
# rather than only the first follow-up working — the type-word and broad-phrasing
# guards remain the exits, and every carried turn discloses itself. "carryover-set"
# is included for the same reason: a comparison thread that established its set
# should keep it for subsequent follow-ups, not only the turn that resolved it.
_CARRYOVER_FROM_METHODS = frozenset({"file", "party", "party-multi", "entity",
                                     "carryover", "carryover-set"})


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
            and not _COMPARATIVE_TYPE_REF_RE.search(question)):
        return []
    if _BROAD_SCOPE_RE.search(question) or _PLURAL_FAMILY_HINT_RE.search(question):
        return []
    try:
        recent = _db.get_recent_answer_scope(session_id, n=1)
    except Exception as e:
        logger.error("_carryover_scope: could not read recent answer scope: %s", e)
        return []
    if not recent:
        return []
    last = recent[0]
    if last.get("method") in _CARRYOVER_FROM_METHODS and last.get("docs"):
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
        recent = _db.get_recent_answer_scope(session_id, n=1)
    except Exception as e:
        logger.error("_carryover_comparative_set: could not read recent answer scope: %s", e)
        return []
    if not recent:
        return []
    files = recent[0].get("files") or []
    if not 2 <= len(files) <= _COMPARATIVE_SET_MAX:
        return []
    return list(files)


def resolve_scope(question: str, session_id: str, pages: dict | None = None,
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
            return {"scope": "single_doc", "target_docs": sorted(party_docs),
                    "target_family": None, "is_broad": False,
                    "confidence": 0.85 if len(party_docs) == 1 else 0.8,
                    "method": "party" if len(party_docs) == 1 else "party-multi"}
        # Party spans several documents but the question names only one instrument
        # type — narrow to that family when it resolves cleanly, else fall through
        # rather than guess.
        try:
            available = set(_db.list_doc_families(session_id)) if config.USE_DATABASE else set()
            fam = _detect_question_family(question, available)
            fam_docs = set(_db.get_documents_by_family(session_id, fam)) if fam else set()
        except Exception:
            fam_docs = set()
        narrowed = party_docs & fam_docs
        if len(narrowed) == 1:
            return {"scope": "single_doc", "target_docs": sorted(narrowed),
                    "target_family": None, "is_broad": False,
                    "confidence": 0.8, "method": "party"}
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
            return {"scope": "single_doc", "target_docs": ent_targets,
                    "target_family": None, "is_broad": False,
                    "confidence": 0.72, "method": "entity"}
        return {"scope": "single_doc", "target_docs": [],
                "target_family": None, "is_broad": False,
                "confidence": 0.7, "method": "entity"}

    # 2. Family scope — a COLLECTIVE reference to one document family that
    #    actually exists in this session. Requires both a collective marker
    #    (broad phrasing or a plural family noun) and a single resolved family,
    #    so a narrow single-clause question is never wrongly filtered.
    collective = bool(_BROAD_SCOPE_RE.search(question) or _PLURAL_FAMILY_HINT_RE.search(question))
    if collective:
        try:
            available = set(_db.list_doc_families(session_id)) if config.USE_DATABASE else set()
        except Exception as e:
            logger.error("resolve_scope: list_doc_families failed: %s", e)
            available = set()
        family = _detect_question_family(question, available)
        if family:
            try:
                docs = _db.get_documents_by_family(session_id, family)
            except Exception as e:
                logger.error("resolve_scope: get_documents_by_family failed: %s", e)
                docs = []
            return {"scope": "family", "target_docs": docs,
                    "target_family": family, "is_broad": True,
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
    if not unresolved_party:
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
        docs = _db.get_source_docs(session_id)
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

    # A named party that resolves to exactly ONE document via full-text content
    # search is an unambiguous document reference — skip disambiguation and let
    # resolve_scope (which runs the SAME resolver later) pin it. This catches the
    # common case the distinctive-entity check below misses: a counterparty named
    # by its full corporate name ("Helios Grid Advisory Private Limited") whose
    # name lives only in the document BODY / redaction-masked metadata, not in the
    # page-title identifier tokens _extract_doc_entities mines — so
    # _question_names_distinctive_entity returns False and the vague "the Services
    # Agreement between X and Y" phrasing would otherwise trigger a needless
    # "which document?" prompt even though the party pins it uniquely (confirmed
    # live: SA5/Helios and SA6/Meridian questions both disambiguated despite each
    # party name resolving to a single Service Agreement). Only a UNIQUE hit skips
    # here; a party shared across several documents stays genuinely ambiguous and
    # falls through to normal disambiguation. _resolve_docs_by_party is DB-gated
    # and returns an empty set with no party phrase present, so questions that
    # name no corporate party incur no extra cost.
    try:
        party_docs = _resolve_docs_by_party(question, session_id)
    except Exception as e:
        logger.error("classify_query: party resolution failed: %s", e)
        party_docs = set()
    if len(party_docs) == 1:
        logger.info("Named party resolves to a single document → skip disambiguation: %s",
                    _norm_doc_name(next(iter(party_docs))))
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
        "- It mentions a document type with a number, identifier, or distinctive party/entity name\n\n"
        "A question NEEDS disambiguation when:\n"
        "- It uses vague references like 'this document', 'summarize it', 'the agreement' "
        "without ANY identifier, number, or party name\n"
        "- It refers to a specific clause, provision, or section (e.g. 'the indemnity clause', "
        "'limitation of liability', 'termination provisions') without specifying WHICH document "
        "contains that clause — and multiple documents in the list could have such a clause\n\n"
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
        docs = _db.get_source_docs(session_id)
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
        "- It asks for a specific deliverable (table, list, summary, review, recommendation)\n\n"
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


def build_conversation_context(session_id: str) -> str:
    """Build a conversation context string from recent chat messages."""
    if not config.USE_DATABASE:
        return ""
    try:
        recent = _db.get_recent_context(session_id, n=6)
    except Exception:
        return ""

    if not recent:
        return ""

    parts = []
    total_chars = 0
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if msg["msg_type"] == "answer" and len(content) > 300:
            content = content[:300] + "..."
        line = f"{role}: {content}"
        if total_chars + len(line) > 2000:
            break
        parts.append(line)
        total_chars += len(line)

    return "\n".join(parts)


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
) -> tuple[list[str], dict]:
    """Select the most relevant pages for a question.

    Priority order:
      1. pgvector cosine similarity search (DB mode only, Phase 3) — 0 LLM calls, ~5ms
      2. BM25 pre-filter + LLM selection (fallback or file mode)
      3. BM25 keyword-only (if LLM selection fails)

    doc_family (Phase 1): when scope resolution (Phase 2) narrows a question to a
    document family, it's passed here to pre-filter the pgvector search to that
    family's embeddings. None = unfiltered whole-session search (default).

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
            emb_count = _db.count_embeddings(session_id)
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
                    session_id, q_embedding, limit=vector_limit, doc_family=doc_family
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

                # Phase 3: Reciprocal Rank Fusion of the vector and BM25 rankings,
                # replacing the previous "all vector, then BM25 appended" order —
                # a strong keyword-only match now ranks on its own merit instead of
                # sitting below every semantic hit. Zero LLM calls.
                if is_broad:
                    # Fuse first, THEN diversify: the per-document cap + Parties-page
                    # force-include operate on a better-ordered base list, but the
                    # breadth guarantee for "across all X" questions is unchanged.
                    fused = _rrf_fuse([valid_vector, bm25_ranking], k=config.RRF_K)
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
                        [valid_vector, bm25_ranking],
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
                        "(vector=%d, bm25=%d, embeddings_in_db=%d, broad=%s)",
                        len(hybrid), len(valid_vector), len(bm25_ranking), emb_count, is_broad,
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
