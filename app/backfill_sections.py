"""
backfill_sections.py — recover numbered sections the ingest extraction skipped

Ingest asks the model to extract "every clause you can identify", but on a long
contract it reliably skips the back-half boilerplate: the numbered sections that
carry no negotiated value but are exactly what a question about "Section 12
(Relationship Of Parties)" needs. Confirmed live on this corpus: 796 numbered
sections across 191 documents have neither a wiki page nor a clause row —
"Relationship Of Parties" alone is missing from 77 documents, "Representations
General" from 80, "Compliance With Laws" from 64.

The documents' own structure makes this recoverable with no LLM call at all: a
section is a line of the form "12. Relationship Of Parties" followed by its
text. This reads the ORIGINAL uploaded file, pulls each numbered section's
heading and first paragraph verbatim, and writes the ones nothing already covers
into `clauses` — where get_context already surfaces them alongside the prose
pages. Same approach, and the same reasoning, as the Definitions backfill that
preceded it.

Also recovers the document header line ("Effective Date: ... | Governing Law:
... | Matter Reference: ...") for documents whose ingest produced no overview
page — the reason "what is the effective date of the Statement of Work between
Apex Novantis EPC Limited and Greystone Data Centers PLC" could not be answered
from a document that states it on page 1.

Recovers defined terms the same way ('"Applicable Law" means ...'), which
matters for a document ingest read as a scan: with no text layer there was
nothing for the earlier Definitions backfill to read.

Deterministic and idempotent: verbatim text only, nothing inferred, and a
section already represented by a page title or clause type is skipped, so a
second run inserts nothing. No LLM calls and no cost unless --ocr is passed.

--ocr reads scanned pages through the configured OCR engine. With
OCR_ENGINE=azure_vision that is a model call per scanned page, which is why it
is opt-in and why --only exists to scope it to named documents.

Usage:
    cd app
    python3 backfill_sections.py                       # all sessions
    python3 backfill_sections.py <session_id>          # single session
    python3 backfill_sections.py <session_id> --dry-run
    python3 backfill_sections.py <session_id> --only "LoanAgt - 30-06-2023" --ocr
"""

from __future__ import annotations

import logging
import os
import re
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("backfill_sections")

# "12. Relationship Of Parties" on a line of its own. Bounded length and a
# leading capital keep it from matching a numbered list item inside a paragraph.
_HEADING_RE = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][A-Za-z][A-Za-z ,/&'()\-]{2,58})\s*$")
_PAGE_MARKER_RE = re.compile(r"^\s*Page\s+\d+\s*$")
_UUID_PREFIX_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_(.*)$"
)
# The header line these documents print under the title on page 1.
_HEADER_RE = re.compile(
    r"(Effective\s+Date:\s*[^|\n]{3,60}(?:\|[^|\n]{3,80}){0,4})", re.IGNORECASE
)

# A defined term as this corpus writes it: '"Applicable Law" means, in relation
# to ...'. The optional comma matters — several of the boilerplate definitions
# read "means," rather than "means ", and requiring whitespace silently dropped
# them.
_DEFINITION_RE = re.compile(r'"([A-Z][A-Za-z /\-]{2,40})"\s+means,?\s+')

# The same numbered heading as _HEADING_RE, but seen mid-line — once a page's
# line breaks are collapsed into one string, a section heading is all that
# marks where the definitions block ends.
_INLINE_HEADING_RE = re.compile(r"\s\d{1,2}\.\s+[A-Z][A-Za-z]")

# A section body shorter than this is a stray heading match, not a clause.
_MIN_BODY_CHARS = 40
# Sections run to a few hundred characters; the cap stops a mis-detected
# heading from swallowing pages of text into one row.
_MAX_BODY_CHARS = 1200

# Words that carry no signal when deciding whether a heading is already covered
# by an existing page title or clause type.
_COVERAGE_STOPWORDS = {
    "the", "of", "and", "or", "to", "in", "a", "an", "for", "with", "on", "by",
    "clause", "section", "agreement", "provisions", "provision", "general",
}


def _coverage_tokens(s: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]+", (s or "").lower())
        if w not in _COVERAGE_STOPWORDS and len(w) > 2
    }


def _build_file_index(upload_path: str) -> dict[str, str]:
    """Map a document's name (with or without its upload UUID prefix) to a path.

    A document ingested in one session can sit on disk under a different
    session's UUID prefix, so the suffix is what actually identifies the file.
    """
    index: dict[str, str] = {}
    if not os.path.isdir(upload_path):
        return index
    for name in os.listdir(upload_path):
        if not name.lower().endswith(".pdf"):
            continue
        m = _UUID_PREFIX_RE.match(name)
        index.setdefault(m.group(1) if m else name, os.path.join(upload_path, name))
    return index


def _find_file(index: dict[str, str], source_doc: str) -> str | None:
    m = _UUID_PREFIX_RE.match(source_doc)
    return index.get(m.group(1) if m else source_doc)


# Below this, a page has no usable text layer — it is a scan.
_SCANNED_PAGE_CHARS = 50


# Keyed by (file path, page number). The three extraction passes each reopen the
# document, and with OCR_ENGINE=azure_vision a page re-read is a model call — so
# a page is OCR'd once per run, not once per pass.
_OCR_CACHE: dict[tuple[str, int], str] = {}


def _page_text(page, ocr: bool, path: str = "", page_num: int = 0) -> str:
    """A page's text, falling back to OCR for a scanned page when allowed.

    OCR is opt-in (--ocr) and never the default: with OCR_ENGINE=azure_vision
    it is a model call per page, and this script otherwise costs nothing at all.
    """
    text = page.get_text()
    if len(text.strip()) >= _SCANNED_PAGE_CHARS or not ocr:
        return text
    key = (path, page_num)
    if key in _OCR_CACHE:
        return _OCR_CACHE[key]
    try:
        from services import reader
        text = reader._ocr_page(page)
    except Exception as e:
        logger.warning("  OCR failed on page %d: %s", page_num, e)
    _OCR_CACHE[key] = text
    return text


def _extract_sections(path: str, ocr: bool = False) -> list[dict]:
    """Every numbered section's heading and first paragraph, verbatim."""
    import fitz

    out: list[dict] = []
    doc = fitz.open(path)
    try:
        for page_num, page in enumerate(doc, 1):
            lines = _page_text(page, ocr, path, page_num).splitlines()
            for i, line in enumerate(lines):
                m = _HEADING_RE.match(line)
                if not m:
                    continue
                body: list[str] = []
                for nxt in lines[i + 1:]:
                    stripped = nxt.strip()
                    # The section's own text ends at the first blank line, page
                    # marker or next heading. What follows in these documents is
                    # padded restatements of the same provision ("Notwithstanding
                    # anything to the contrary ..."), not new content.
                    if not stripped or _PAGE_MARKER_RE.match(nxt) or _HEADING_RE.match(nxt):
                        break
                    body.append(stripped)
                    if len(" ".join(body)) > _MAX_BODY_CHARS:
                        break
                text = " ".join(body).strip()
                if len(text) < _MIN_BODY_CHARS:
                    continue
                out.append({"heading": m.group(2).strip(), "text": text, "page": page_num})
    finally:
        doc.close()
    return out


def _extract_definitions(path: str, ocr: bool = False) -> list[dict]:
    """Every '"Term" means ...' definition in the document, verbatim.

    A scanned document has no text layer, so the earlier Definitions backfill
    skipped it entirely — which is why "how is the term Applicable Law defined
    in the Facility Agreement between Apex Devashri InfoSystems Limited and
    Amberline Commodities Limited" had nothing to answer from. With --ocr the
    same extraction runs on the OCR'd text.
    """
    import fitz

    out: list[dict] = []
    seen: set[str] = set()
    doc = fitz.open(path)
    try:
        for page_num, page in enumerate(doc, 1):
            text = " ".join(_page_text(page, ocr, path, page_num).split())
            for m in _DEFINITION_RE.finditer(text):
                term = m.group(1).strip()
                if term.lower() in seen:
                    continue
                # Run to the end of the sentence, or to the next defined term
                # if the sentence's own full stop is missing (common in OCR).
                tail = text[m.start():m.start() + 1400]
                nxt = _DEFINITION_RE.search(tail, m.end() - m.start())
                if nxt:
                    tail = tail[:nxt.start()]
                # The last definition in a block runs into the section that
                # follows it, because nothing separates them once the page's
                # line breaks are collapsed. A numbered heading is that
                # boundary: without it, '"Term" means ...' swallowed the
                # Governing Law section that came next.
                head = _INLINE_HEADING_RE.search(tail, m.end() - m.start())
                if head:
                    tail = tail[:head.start()]
                end = tail.rfind(". ")
                verbatim = (tail[:end + 1] if end > 40 else tail).strip()
                if len(verbatim) < _MIN_BODY_CHARS:
                    continue
                seen.add(term.lower())
                out.append({"term": term, "text": verbatim, "page": page_num})
    finally:
        doc.close()
    return out


def _extract_header(path: str, ocr: bool = False) -> str | None:
    """The document's page-1 "Effective Date: ... | Governing Law: ..." line."""
    import fitz

    doc = fitz.open(path)
    try:
        if not len(doc):
            return None
        m = _HEADER_RE.search(_page_text(doc[0], ocr, path, 1))
        return m.group(1).strip() if m else None
    finally:
        doc.close()


def _existing_coverage(conn, session_id: str) -> tuple[dict[str, list[set[str]]],
                                                       dict[str, set[str]]]:
    """Per document: the token sets of its labels, and its exact clause types.

    Two views because the two kinds of row need different tests. A section
    heading is matched loosely (a page titled "Termination Rights (Convenience
    and for Cause)" already covers a section headed "Termination For Cause"),
    but a defined term must be matched EXACTLY — "Definition - Governing Law"
    and "Definition - Applicable Law" share two meaningful words and the loose
    test would treat one as covering the other.
    """
    from sqlalchemy import text

    coverage: dict[str, list[set[str]]] = {}
    exact: dict[str, set[str]] = {}
    for source_doc, label in conn.execute(
        text("SELECT source_doc, title FROM pages WHERE session_id = :sid"),
        {"sid": session_id},
    ):
        coverage.setdefault(source_doc, []).append(_coverage_tokens(label))
    for source_doc, label in conn.execute(
        text("SELECT source_doc, clause_type FROM clauses WHERE session_id = :sid"),
        {"sid": session_id},
    ):
        coverage.setdefault(source_doc, []).append(_coverage_tokens(label))
        exact.setdefault(source_doc, set()).add((label or "").strip().lower())
    return coverage, exact


# "The following table sets out the financial covenants table applicable under
# this Agreement." — how this corpus introduces every schedule it prints.
_TABLE_INTRO_RE = re.compile(
    r"The following table sets out the ([^.\n]{3,60})\.", re.IGNORECASE)

# A schedule runs to a few dozen short cells; past this a mis-detected end has
# swallowed the body text that follows it.
_MAX_TABLE_CHARS = 1500


def _extract_labelled_tables(path: str, ocr: bool = False) -> list[dict]:
    """Schedules a document prints that ingest's table extraction missed.

    Deliberately captures the table's CELLS VERBATIM as one block rather than
    reconstructing rows and columns. A flat PDF text layer gives no column
    count — the cells arrive as a bare sequence — and guessing one wrong would
    file confident, wrongly-aligned values into the typed `tables` store, which
    is a worse failure than the prose-only status quo. The block goes to
    `clauses` under the schedule's own caption, where the answer model reads the
    layout for itself.

    Ingest captures 168 of the 176 documents that print such a table; this is
    for the remaining 8. Measured cost of the gap: the Facility Agreement
    between Everbright Capital Bank Ltd and Apex Meridian Mobility states a
    Debt Service Coverage Ratio of ">= 1.4x" in its covenants table and "not
    less than 1.21" in the padded prose repeated five times around it. Only the
    prose reached retrieval, so 1.21 is what the answer reported.
    """
    import fitz

    out: list[dict] = []
    doc = fitz.open(path)
    try:
        for page_num, page in enumerate(doc, 1):
            lines = _page_text(page, ocr, path, page_num).splitlines()
            for i, line in enumerate(lines):
                m = _TABLE_INTRO_RE.search(line)
                if not m:
                    continue
                cells: list[str] = []
                for nxt in lines[i + 1:]:
                    stripped = nxt.strip()
                    if (not stripped or _PAGE_MARKER_RE.match(nxt)
                            or _HEADING_RE.match(nxt) or _TABLE_INTRO_RE.search(nxt)):
                        break
                    cells.append(stripped)
                    if len(" | ".join(cells)) > _MAX_TABLE_CHARS:
                        break
                if len(cells) < 4:
                    continue
                out.append({"caption": m.group(1).strip().title(),
                            "text": " | ".join(cells), "page": page_num})
    finally:
        doc.close()
    return out


def _covers_header(text: str) -> bool:
    """Whether a document's stored text already states its header facts."""
    return "effective date:" in (text or "").lower()


def _header_text(conn, session_id: str) -> dict[str, str]:
    """Per document, all stored page and clause text — the corpus of _covers_header.

    An Overview PAGE is not evidence the header's facts were captured: ingest
    writes one for most documents and it summarises the agreement's substance
    without necessarily restating the effective date printed above it.
    """
    from sqlalchemy import text

    blob: dict[str, list[str]] = {}
    for source_doc, body in conn.execute(
        text("SELECT source_doc, content FROM pages WHERE session_id = :sid"),
        {"sid": session_id},
    ):
        blob.setdefault(source_doc, []).append(body or "")
    for source_doc, body in conn.execute(
        text("SELECT source_doc, verbatim_text FROM clauses WHERE session_id = :sid"),
        {"sid": session_id},
    ):
        blob.setdefault(source_doc, []).append(body or "")
    return {k: "\n".join(v) for k, v in blob.items()}


def _is_covered(heading: str, known: list[set[str]]) -> bool:
    tokens = _coverage_tokens(heading)
    if not tokens:
        return True
    # Covered when an existing label contains the whole heading, or shares two
    # meaningful words with it ("Termination Rights (Convenience and for Cause)"
    # already covers "Termination For Cause").
    return any(tokens <= k or len(tokens & k) >= 2 for k in known)


def _sessions(conn, target_session: str | None) -> list[str]:
    from sqlalchemy import text

    if target_session:
        return [target_session]
    return [r[0] for r in conn.execute(
        text("SELECT DISTINCT session_id FROM pages WHERE session_id IS NOT NULL")
    )]


def backfill(target_session: str | None = None, dry_run: bool = False,
             ocr: bool = False, only: str | None = None) -> None:
    import config

    if not config.USE_DATABASE:
        logger.error("DATABASE_URL is not set — backfill only applies to PostgreSQL mode.")
        sys.exit(1)

    from services import db as _db
    from sqlalchemy import text as _sql

    file_index = _build_file_index(config.UPLOAD_PATH)
    if not file_index:
        logger.error("No uploaded PDFs found under %s — nothing to read.", config.UPLOAD_PATH)
        sys.exit(1)

    engine = _db.get_engine()
    total_rows = total_docs = total_missing_file = 0

    with engine.connect() as conn:
        sessions = _sessions(conn, target_session)

    if ocr:
        logger.warning("--ocr is ON: scanned pages will be read with OCR_ENGINE=%s. "
                       "With azure_vision that is a model call per scanned page.",
                       config.OCR_ENGINE)

    for session_id in sessions:
        with engine.connect() as conn:
            coverage, exact_types = _existing_coverage(conn, session_id)
            header_blobs = _header_text(conn, session_id)
            docs_with_tables = {r[0] for r in conn.execute(
                _sql("SELECT DISTINCT source_doc FROM tables WHERE session_id = :sid"),
                {"sid": session_id})}
            docs = [r[0] for r in conn.execute(
                _sql("SELECT DISTINCT source_doc FROM pages WHERE session_id = :sid"),
                {"sid": session_id},
            ) if r[0] and (not only or only.lower() in r[0].lower())]
            wiki_row = conn.execute(
                _sql("SELECT wiki_id FROM pages WHERE session_id = :sid LIMIT 1"),
                {"sid": session_id},
            ).first()
        wiki_id = wiki_row[0] if wiki_row and wiki_row[0] else _db.DEFAULT_WIKI_ID

        for source_doc in sorted(docs):
            path = _find_file(file_index, source_doc)
            if not path:
                total_missing_file += 1
                continue
            known = coverage.get(source_doc, [])
            known_exact = exact_types.get(source_doc, set())
            known_text = header_blobs.get(source_doc, "")
            has_table = source_doc in docs_with_tables
            try:
                sections = _extract_sections(path, ocr)
                definitions = _extract_definitions(path, ocr)
                header = _extract_header(path, ocr)
                schedules = _extract_labelled_tables(path, ocr) if not has_table else []
            except Exception as e:
                logger.error("  [%s] could not read file: %s", source_doc, e)
                continue

            clauses = [
                {"type": s["heading"], "text": s["text"], "confidence": 1.0, "page": s["page"]}
                for s in sections
                if not _is_covered(s["heading"], known)
            ]
            clauses.extend(
                {"type": f"Definition - {d['term']}", "text": d["text"],
                 "confidence": 1.0, "page": d["page"]}
                for d in definitions
                if f"definition - {d['term'].lower()}" not in known_exact
            )
            # An Overview page is NOT evidence the header's facts were captured:
            # ingest writes one for most documents, and it summarises the
            # agreement's substance without necessarily restating the effective
            # date, governing law or matter reference printed above it. Measured
            # after the first run of this script: 44 documents had an Overview
            # page and still stated their effective date nowhere the pipeline
            # could reach, which is why "what is the effective date of the
            # Facility Agreement between National Trust Financial Corporation
            # and Apex Junoon Alloys PLC" answered that no date was given while
            # page 1 of that PDF reads "Effective Date: 21 November 2024".
            # What counts as coverage is the header TEXT, wherever it lives.
            clauses.extend(
                {"type": sch["caption"], "text": sch["text"],
                 "confidence": 1.0, "page": sch["page"]}
                for sch in schedules
                if not _is_covered(sch["caption"], known)
            )
            if header and not _covers_header(known_text) and not _is_covered("Document Header", known):
                clauses.append({"type": "Document Header", "text": header,
                                "confidence": 1.0, "page": 1})
            if not clauses:
                continue

            total_docs += 1
            if dry_run:
                total_rows += len(clauses)
                logger.info("  [%s] would add %d: %s", source_doc, len(clauses),
                            ", ".join(c["type"] for c in clauses[:6]))
                continue
            try:
                n = _db.insert_clauses(wiki_id, session_id, source_doc, clauses)
            except Exception as e:
                logger.error("  [%s] insert failed: %s", source_doc, e)
                continue
            total_rows += n
            logger.info("  [%s] +%d section(s)", source_doc, n)

    logger.info(
        "%s — %d clause rows across %d documents (%d documents had no file on disk).",
        "Dry run complete" if dry_run else "Backfill complete",
        total_rows, total_docs, total_missing_file,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    only = None
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = argv[i + 1]
    positional = [a for a in argv if not a.startswith("--") and a != only]
    backfill(target_session=positional[0] if positional else None,
             dry_run="--dry-run" in argv,
             ocr="--ocr" in argv,
             only=only)
