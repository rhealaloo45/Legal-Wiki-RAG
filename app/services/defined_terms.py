"""
Defined-term cross-reference (target architecture § Phase 4).

A clause defining "Affiliate" in §1.2 and every clause that later relies on
that term are structurally unlinked: the term is prose on both sides. This
module builds the link.

The roadmap specified this as new edges written during the ingest synthesis
call. It is built here WITHOUT one, because the corpus already carries the
data: 3,636 clauses classify as `definition` under the Phase 3.5c vocabulary,
and legal definitions follow a rigid form — `"Term" means ...` — that a regex
reads more reliably than a model would. That makes this work on the 1,372
documents already ingested rather than only on documents ingested from now on,
which is the difference between a feature and a promise. The ingest prompt is
left alone.

Two directions, and the second is the one that pays:

  defines           term  ->  the clause that defines it
  references-term   term  ->  the clauses that use it

The valuable question is the absence one — "is this term actually defined
anywhere in this document" — which today can only be answered by an LLM
re-reading the whole document, and which a lawyer asks when a contract uses a
capitalised term it never defines. That is a real drafting defect, and it is
invisible to retrieval.
"""

import logging
import re

logger = logging.getLogger(__name__)

# `"Term" means`, `"Term" shall mean`, `'Term' means`, and the curly-quote
# variants the corpus actually contains. The quoted form is required: an
# unquoted "affiliate means" is prose, not a definition, and admitting it
# produces junk terms at volume.
_RX_DEFINITION = re.compile(
    r'["“‘\']([A-Z][^"”’\']{1,60}?)["”’\']\s*'
    r'(?:shall\s+)?(?:mean|means|has\s+the\s+meaning|refers\s+to|is\s+defined)',
)
# Fallback for clause_type "Definition - Affiliate" when the text itself does
# not carry the quoted form.
_RX_TYPE_TERM = re.compile(r'^definitions?\s*[-–:]\s*(.{2,60})$', re.IGNORECASE)

# Terms too generic to be worth tracking as defined terms — they appear in
# almost every contract and linking them adds noise rather than signal.
_SKIP_TERMS = {
    "agreement", "party", "parties", "the parties", "this agreement",
    "company", "term", "terms", "person", "day", "days",
}


def _enabled() -> bool:
    import config
    return bool(getattr(config, "USE_DATABASE", False))


def term_key(term: str) -> str:
    """Case- and punctuation-insensitive key for matching a term to its uses."""
    return re.sub(r"[^a-z0-9]+", " ", (term or "").lower()).strip()


def extract_term(clause_type: str, verbatim_text: str) -> str | None:
    """The term a definition clause defines, or None.

    Text first, clause_type second: the text carries the term as the drafter
    wrote it, while the type carries whatever the extracting model chose to
    call the clause.
    """
    m = _RX_DEFINITION.search(verbatim_text or "")
    if m:
        term = m.group(1).strip()
    else:
        m2 = _RX_TYPE_TERM.match((clause_type or "").strip())
        if not m2:
            return None
        term = m2.group(1).strip()
    term = re.sub(r"\s+", " ", term).strip(" .,:;\"'")
    if not term or len(term) < 2 or term_key(term) in _SKIP_TERMS:
        return None
    # A "term" carrying sentence punctuation is a mis-parse, not a term.
    if re.search(r"[.;]|\b(?:means|shall)\b", term, re.IGNORECASE):
        return None
    return term


def build(wiki_id: str, session_id: str, dry_run: bool = False) -> dict:
    """Populate defined_terms and term_references from existing clause rows.

    Deterministic and re-runnable — no model call. Replaces this session's rows
    wholesale so a re-run after a parser change cannot leave stale terms behind.
    """
    if not _enabled():
        return {"error": "database not configured"}
    from sqlalchemy import text
    from services import db

    with db.get_engine().connect() as conn:
        _init(conn, text)
        conn.commit()

        rows = conn.execute(text("""
            SELECT id, source_doc, clause_type, verbatim_text, page_num
            FROM clauses
            WHERE wiki_id = :w AND session_id = :sid AND clause_type_canon = 'definition'
        """), {"w": wiki_id, "sid": session_id}).fetchall()

        found: list[dict] = []
        for cid, doc, ctype, vtext, page in rows:
            term = extract_term(ctype, vtext)
            if not term:
                continue
            found.append({"clause_id": int(cid), "source_doc": doc, "term": term,
                          "term_key": term_key(term),
                          "definition": (vtext or "")[:2000], "page_num": page})

        if dry_run:
            return {"dry_run": True, "definition_clauses": len(rows),
                    "terms_found": len(found),
                    "distinct_terms": len({f["term_key"] for f in found})}

        conn.execute(text("DELETE FROM defined_terms WHERE wiki_id=:w AND session_id=:sid"),
                     {"w": wiki_id, "sid": session_id})
        for f in found:
            conn.execute(text("""
                INSERT INTO defined_terms
                    (wiki_id, session_id, source_doc, term, term_key, definition,
                     clause_id, page_num)
                VALUES (:w,:sid,:d,:t,:k,:def,:cid,:p)
                ON CONFLICT (wiki_id, session_id, source_doc, term_key) DO NOTHING
            """), {"w": wiki_id, "sid": session_id, "d": f["source_doc"],
                   "t": f["term"], "k": f["term_key"], "def": f["definition"],
                   "cid": f["clause_id"], "p": f["page_num"]})
        conn.commit()

        refs = _link_references(conn, text, wiki_id, session_id)

    return {"dry_run": False, "definition_clauses": len(rows),
            "terms_stored": len(found),
            "distinct_terms": len({f["term_key"] for f in found}),
            "references_linked": refs}


def _link_references(conn, text, wiki_id: str, session_id: str) -> int:
    """Link each defined term to the other clauses in its own document that use it.

    Scoped to the SAME document deliberately. A term defined in one contract
    does not govern its use in an unrelated one, and linking across documents
    would assert a relationship the drafters never made.
    """
    conn.execute(text("DELETE FROM term_references WHERE wiki_id=:w AND session_id=:sid"),
                 {"w": wiki_id, "sid": session_id})
    # One statement per document rather than per term: the term list per
    # document is small, and this keeps the clause text scan to a single pass.
    inserted = conn.execute(text("""
        INSERT INTO term_references
            (wiki_id, session_id, source_doc, term_key, clause_id)
        SELECT DISTINCT dt.wiki_id, dt.session_id, dt.source_doc, dt.term_key, c.id
        FROM defined_terms dt
        JOIN clauses c
          ON c.wiki_id = dt.wiki_id AND c.session_id = dt.session_id
         AND c.source_doc = dt.source_doc
         AND c.id <> dt.clause_id
         AND c.verbatim_text ILIKE '%' || dt.term || '%'
        WHERE dt.wiki_id = :w AND dt.session_id = :sid
        ON CONFLICT DO NOTHING
    """), {"w": wiki_id, "sid": session_id}).rowcount or 0
    conn.commit()
    return int(inserted)


def _init(conn, text) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS defined_terms (
            id          BIGSERIAL PRIMARY KEY,
            wiki_id     TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            source_doc  TEXT NOT NULL,
            term        TEXT NOT NULL,
            term_key    TEXT NOT NULL,
            definition  TEXT,
            clause_id   BIGINT,
            page_num    INT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (wiki_id, session_id, source_doc, term_key)
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS term_references (
            id          BIGSERIAL PRIMARY KEY,
            wiki_id     TEXT NOT NULL,
            session_id  TEXT NOT NULL,
            source_doc  TEXT NOT NULL,
            term_key    TEXT NOT NULL,
            clause_id   BIGINT NOT NULL,
            UNIQUE (wiki_id, session_id, source_doc, term_key, clause_id)
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS defined_terms_doc_idx
        ON defined_terms (wiki_id, session_id, source_doc)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS defined_terms_key_idx
        ON defined_terms (wiki_id, session_id, term_key)
    """))


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def terms_in_document(wiki_id: str, session_id: str, source_doc: str) -> list[dict]:
    """Every term this document defines, with how many clauses rely on each."""
    if not _enabled():
        return []
    from sqlalchemy import text
    from services import db
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT dt.term, dt.definition, dt.page_num,
                   (SELECT count(*) FROM term_references tr
                     WHERE tr.wiki_id = dt.wiki_id AND tr.session_id = dt.session_id
                       AND tr.source_doc = dt.source_doc AND tr.term_key = dt.term_key)
            FROM defined_terms dt
            WHERE dt.wiki_id = :w AND dt.session_id = :sid AND dt.source_doc = :d
            ORDER BY 4 DESC, dt.term
        """), {"w": wiki_id, "sid": session_id, "d": source_doc}).fetchall()
    return [{"term": r[0], "definition": (r[1] or "")[:400],
             "page_num": r[2], "used_in_clauses": int(r[3])} for r in rows]


def find_definition(wiki_id: str, session_id: str, term: str,
                    source_doc: str | None = None) -> list[dict]:
    """Where a term is defined — in one document, or across the corpus.

    Returned as a list because the same term is routinely defined differently
    in different agreements, and collapsing that to one answer would hide the
    variation a lawyer is usually asking about.
    """
    if not _enabled():
        return []
    from sqlalchemy import text
    from services import db
    params = {"w": wiki_id, "sid": session_id, "k": term_key(term)}
    where = "wiki_id = :w AND session_id = :sid AND term_key = :k"
    if source_doc:
        params["d"] = source_doc
        where += " AND source_doc = :d"
    with db.get_engine().connect() as conn:
        rows = conn.execute(text(f"""
            SELECT source_doc, term, definition, page_num
            FROM defined_terms WHERE {where} ORDER BY source_doc LIMIT 50
        """), params).fetchall()
    return [{"source_doc": r[0], "term": r[1], "definition": (r[2] or "")[:800],
             "page_num": r[3]} for r in rows]


# A capitalised phrase in quotes is not automatically a term of art. Contracts
# introduce party short-names the same way — Apex Meridian Software Private
# Limited ("Meridian") — and a person's name in a signature block looks
# identical to a regex. Both filters below were added after the first run of
# this check reported "Deborah Rodriguez", "Voltas" and "Udaan" as undefined
# terms: correct pattern match, useless finding.
_MIN_UNDEFINED_OCCURRENCES = 2


def undefined_terms(wiki_id: str, session_id: str, source_doc: str,
                    limit: int = 25) -> dict:
    """Capitalised terms a document USES in quotes but never defines.

    The drafting-defect check, and the reason this module exists. Two filters
    keep it usable rather than merely correct:

      Quoted form is required. Treating every capitalised phrase as a term of
      art reports statute titles and headings as defects.

      A term must appear at least twice. A genuine undefined term of art is
      RELIED ON — it recurs. A quoted capitalised phrase appearing exactly
      once is nearly always an introduction (a party short-name, a signatory),
      which is the opposite of a defect. This single threshold removed every
      false positive in the first real run.
    """
    if not _enabled():
        return {"error": "database not configured"}
    from sqlalchemy import text
    from services import db
    with db.get_engine().connect() as conn:
        defined = {r[0] for r in conn.execute(text(
            "SELECT term_key FROM defined_terms WHERE wiki_id=:w AND session_id=:sid "
            "AND source_doc=:d"),
            {"w": wiki_id, "sid": session_id, "d": source_doc})}
        texts = [r[0] or "" for r in conn.execute(text(
            "SELECT verbatim_text FROM clauses WHERE wiki_id=:w AND session_id=:sid "
            "AND source_doc=:d"),
            {"w": wiki_id, "sid": session_id, "d": source_doc})]
        # The document's own parties, so a party short-name is never reported.
        prow = conn.execute(text(
            "SELECT parties FROM documents WHERE wiki_id=:w AND session_id=:sid "
            "AND source_doc=:d"),
            {"w": wiki_id, "sid": session_id, "d": source_doc}).fetchone()

    party_keys: set[str] = set()
    if prow and isinstance(prow[0], list):
        for p in prow[0]:
            k = term_key(str(p))
            if k:
                party_keys.add(k)
                # Party names are introduced by a short form drawn from the
                # full name, so each word is also a candidate short-name.
                for w in k.split():
                    if len(w) > 3:
                        party_keys.add(w)

    rx = re.compile(r'["“‘]([A-Z][A-Za-z ]{2,40}?)["”’]')
    seen: dict[str, int] = {}
    for t in texts:
        for m in rx.finditer(t):
            cand = m.group(1).strip()
            k = term_key(cand)
            if k in defined or k in _SKIP_TERMS or len(k) < 3:
                continue
            if k in party_keys or any(w in party_keys for w in k.split()):
                continue
            seen[cand] = seen.get(cand, 0) + 1
    ranked = [(t, n) for t, n in sorted(seen.items(), key=lambda kv: -kv[1])
              if n >= _MIN_UNDEFINED_OCCURRENCES][:limit]
    return {
        "source_doc": source_doc,
        "defined_count": len(defined),
        "undefined": [{"term": t, "occurrences": n} for t, n in ranked],
        "note": (f"Terms used at least {_MIN_UNDEFINED_OCCURRENCES} times in the quoted "
                 f"form a definition would use, that this document never defines, "
                 f"excluding its own party names. Both filters are deliberate: a "
                 f"once-quoted capitalised phrase is nearly always an introduction "
                 f"rather than a reliance, and party short-names look identical to "
                 f"terms of art to any pattern match."),
    }
