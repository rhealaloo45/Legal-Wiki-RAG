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

Deterministic and idempotent: verbatim text only, nothing inferred, and a
section already represented by a page title or clause type is skipped, so a
second run inserts nothing. No LLM calls, no cost.

Usage:
    cd app
    python3 backfill_sections.py                       # all sessions
    python3 backfill_sections.py <session_id>          # single session
    python3 backfill_sections.py <session_id> --dry-run
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


def _extract_sections(path: str) -> list[dict]:
    """Every numbered section's heading and first paragraph, verbatim."""
    import fitz

    out: list[dict] = []
    doc = fitz.open(path)
    try:
        for page_num, page in enumerate(doc, 1):
            lines = page.get_text().splitlines()
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


def _extract_header(path: str) -> str | None:
    """The document's page-1 "Effective Date: ... | Governing Law: ..." line."""
    import fitz

    doc = fitz.open(path)
    try:
        if not len(doc):
            return None
        m = _HEADER_RE.search(doc[0].get_text())
        return m.group(1).strip() if m else None
    finally:
        doc.close()


def _existing_coverage(conn, session_id: str) -> dict[str, list[set[str]]]:
    """Per document, the token sets of every page title and clause type it has."""
    from sqlalchemy import text

    coverage: dict[str, list[set[str]]] = {}
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
    return coverage


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


def backfill(target_session: str | None = None, dry_run: bool = False) -> None:
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

    for session_id in sessions:
        with engine.connect() as conn:
            coverage = _existing_coverage(conn, session_id)
            docs = [r[0] for r in conn.execute(
                _sql("SELECT DISTINCT source_doc FROM pages WHERE session_id = :sid"),
                {"sid": session_id},
            ) if r[0]]
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
            try:
                sections = _extract_sections(path)
                header = _extract_header(path)
            except Exception as e:
                logger.error("  [%s] could not read file: %s", source_doc, e)
                continue

            clauses = [
                {"type": s["heading"], "text": s["text"], "confidence": 1.0, "page": s["page"]}
                for s in sections
                if not _is_covered(s["heading"], known)
            ]
            # The header carries the effective date, governing law and matter
            # reference. Only worth adding when the document has no overview
            # page holding the same thing.
            if (header
                    and not any("overview" in " ".join(k) for k in known)
                    and not _is_covered("Document Header", known)):
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
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    backfill(target_session=args[0] if args else None,
             dry_run="--dry-run" in sys.argv[1:])
