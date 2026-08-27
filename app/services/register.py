"""Phase 1 — the Contract Register, the Obligation tracker, and the shared
typed-field reader behind both.

Target architecture § 06: "Every document appears automatically as a row the
moment it finishes ingest." That is the whole feature — no extraction, no
LLM call, no background job. The rows already exist in `documents` and the
five family tables; the register is the read nobody had written.

Two decisions worth stating.

The register is **family-aware rather than contract-shaped**, even though it
is called the Contract Register. A litigation document has no liability cap
and a power of attorney has no governing law, and giving every document the
same twelve contract columns is the exact mistake § 06 calls out. The column
set therefore comes from the schema registry, per family — see
`family_columns` for which of the registry's two field lists it reads and
why — and the all-families view falls back to one party column resolved
across families.

`document_fields` exists here rather than in `db.py` because Review Table
needs the same per-document typed values the register shows, keyed by field
name. Putting one reader behind both is the point: a column the register can
display is by definition a column Review Table should never spend an LLM call
on.

The Obligation tracker sits at the bottom of this module for the same reason:
it is the same kind of read, over `obligations` instead of the family tables,
and it shares the value flattening. It reports its own emptiness honestly —
a corpus ingested before obligation extraction was wired has no duties to
show, and saying "nothing found" would misreport that as a corpus with no
obligations in it.
"""
from __future__ import annotations

import logging

from services import db, schema_registry

logger = logging.getLogger(__name__)

# Where each family's typed row lands. Read together and matched on
# source_doc rather than joined per family — the same approach
# db.get_review_documents takes, for the same reason: which table holds a
# document is not known until its row has been read.
_TYPED_TABLES = ("contracts", "litigation_facts", "authorizations", "opinions")

# Shown as their own columns, so they would be duplicated as metadata fields.
_STRUCTURAL_FIELDS = ("doc_type", "doc_family", "jurisdiction")


def family_columns(family_key: str | None) -> list[str]:
    """The standard column set for one family, from the schema registry.

    Not a fixed list. This is the mechanism § 06 asks for — adding a family
    to the registry gives it register columns and Review Table columns with
    no further code.

    The set is drawn from `extraction_fields`, ordered by `metadata_fields`.
    Those two registry lists have drifted apart: a family's metadata list
    names fields nothing extracts (`auto_renewal` and `termination_notice`
    for contracts — the extraction calls them `renewal_terms` and
    `termination`) and omits fields that are extracted (`holding`,
    `relief_granted`, `scope_of_authority`, `conclusion`). Driving columns
    off the metadata list alone would print columns that are empty by
    construction and hide values the pipeline really produced, so the column
    set is what the family actually extracts, in the order the metadata list
    prefers.
    """
    extracted = list(schema_registry.get(family_key).extraction_fields)
    preferred = [f for f in schema_registry.metadata_field_list(family_key)
                 if f in extracted]
    rest = [f for f in extracted if f not in preferred]
    return [f for f in preferred + rest if f not in _STRUCTURAL_FIELDS]


# The all-families view cannot show per-family terms — a judgment has no
# liability cap, and a liability-cap column that is empty for a reason reads
# as missing data. What every family does have is someone the document is
# about, under a different field name each time. This resolves that one
# column across families rather than showing nothing.
_PARTY_FALLBACK = ("parties", "plaintiffs", "defendants", "grantor",
                   "grantee", "addressee")


def all_family_columns() -> list[str]:
    """Columns for the unfiltered view: only what resolves for every family.

    Currently just the synthetic party column. The structural fields
    (document, family, type, jurisdiction, counts) are returned as their own
    top-level keys and are not listed here.
    """
    return ["parties"]


def display_value(value):
    """Flatten a validated value into something a table cell can show.

    The duration and currency coercers store a normalized object so a later
    feature can sort and compare on it; a register cell wants the words the
    document used. Both are kept — this reads `raw` and leaves the stored
    object alone.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        if value.get("raw") is not None:
            return str(value["raw"])
        return "; ".join(f"{k}: {v}" for k, v in value.items() if v is not None) or None
    if isinstance(value, (list, tuple)):
        parts = [display_value(v) for v in value]
        return "; ".join(p for p in parts if p) or None
    if isinstance(value, str):
        # A coerced duration or currency written into a TEXT column comes back
        # as its JSON text, not as an object — obligations.notice_period and
        # contracts.term_length both do this. Unwrapping only a leading brace
        # that parses to an object leaves real prose alone; a clause that
        # happens to start with "{" and is not JSON fails to parse and is
        # returned unchanged.
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                import json
                parsed = json.loads(s)
            except Exception:
                return value
            if isinstance(parsed, dict):
                return display_value(parsed)
    return str(value)


def _typed_by_doc(conn, wiki_id: str, session_id: str) -> dict[str, tuple]:
    from sqlalchemy import text
    out: dict[str, tuple] = {}
    for tbl in _TYPED_TABLES:
        try:
            for r in conn.execute(text(
                f"SELECT source_doc, typed_value, confidence FROM {tbl} "
                f"WHERE wiki_id = :w AND session_id = :s"
            ), {"w": wiki_id, "s": session_id}).fetchall():
                out[r[0]] = (r[1], r[2])
        except Exception as err:
            logger.debug("Register could not read %s: %s", tbl, err)
    return out


def _counts(conn, table: str, wiki_id: str, session_id: str) -> dict[str, int]:
    from sqlalchemy import text
    try:
        return {r[0]: int(r[1]) for r in conn.execute(text(
            f"SELECT source_doc, COUNT(*) FROM {table} "
            f"WHERE wiki_id = :w AND session_id = :s GROUP BY source_doc"
        ), {"w": wiki_id, "s": session_id}).fetchall()}
    except Exception as err:
        logger.debug("Register could not count %s: %s", table, err)
        return {}


def _fields_map(raw_typed) -> dict[str, dict]:
    """`db._review_fields` keyed by field name rather than ordered by doubt."""
    rows = db._review_fields(raw_typed, db._crypto())
    return {r["name"]: r for r in rows}


def document_fields(wiki_id: str, session_id: str, source_doc: str) -> dict[str, dict]:
    """Every typed metadata field known for one document, by field name.

    Returns {} when the document has no typed row — which is a real answer,
    not an error: a document ingested before the backbone existed, or one
    whose family has no typed table, genuinely has no typed fields.
    """
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        for tbl in _TYPED_TABLES:
            try:
                row = conn.execute(text(
                    f"SELECT typed_value FROM {tbl} WHERE wiki_id = :w "
                    f"AND session_id = :s AND source_doc = :d"
                ), {"w": wiki_id, "s": session_id, "d": source_doc}).fetchone()
            except Exception as err:
                logger.debug("Register could not read %s for %s: %s", tbl, source_doc, err)
                continue
            if row and row[0] is not None:
                return _fields_map(row[0])
    return {}


def _cell(fields: dict[str, dict], name: str) -> dict:
    f = fields.get(name) or {}
    return {"value": display_value(f.get("value")),
            "confidence": f.get("confidence"),
            "flagged": bool(f.get("flagged")),
            "edited": bool(f.get("edited"))}


def _party_cell(fields: dict[str, dict]) -> dict:
    """The all-families party column, resolved through the fallback chain.

    A litigation row shows its plaintiffs and defendants together, because
    "who is this document about" is one question even where the schema
    splits the answer in two.
    """
    parts, confs, flagged = [], [], False
    for name in _PARTY_FALLBACK:
        cell = _cell(fields, name)
        if cell["value"]:
            parts.append(cell["value"])
            if cell["confidence"] is not None:
                confs.append(cell["confidence"])
            flagged = flagged or cell["flagged"]
            # `parties` is the whole answer where a family has it; the split
            # fields only stand in for families that don't.
            if name == "parties":
                break
    return {"value": "; ".join(parts) or None,
            "confidence": min(confs) if confs else None,
            "flagged": flagged, "edited": False}


def register_rows(wiki_id: str, session_id: str, *,
                  family: str | None = None,
                  search: str | None = None,
                  limit: int = 500, offset: int = 0) -> dict:
    """One row per ingested document, with its family's standard fields.

    Filtering and paging happen in Python rather than SQL because the typed
    values being filtered on live inside an encrypted JSON column — a WHERE
    clause cannot see them. At corpus scale (hundreds of documents, not
    millions) reading the registry and filtering it costs less than the
    machinery to avoid doing so would.
    """
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        docs = conn.execute(text("""
            SELECT source_doc, doc_family, doc_type, jurisdiction, lifecycle,
                   family_confidence, family_method, schema_version, created_at
            FROM documents
            WHERE wiki_id = :w AND session_id = :s
            ORDER BY created_at DESC, source_doc
        """), {"w": wiki_id, "s": session_id}).fetchall()
        typed = _typed_by_doc(conn, wiki_id, session_id)
        pages = _counts(conn, "pages", wiki_id, session_id)
        clauses = _counts(conn, "clauses", wiki_id, session_id)
        obligations = _counts(conn, "obligations", wiki_id, session_id)
        citations = _counts(conn, "citations", wiki_id, session_id)

    columns = family_columns(family) if family else all_family_columns()
    needle = (search or "").strip().lower()

    rows: list[dict] = []
    for d in docs:
        doc = d[0]
        if family and (d[1] or "generic") != family:
            continue
        raw_typed, row_conf = typed.get(doc, (None, None))
        fields = _fields_map(raw_typed)
        values = {name: _cell(fields, name) for name in columns}
        if not family:
            values["parties"] = _party_cell(fields)
        if needle:
            hay = " ".join(
                [doc, d[1] or "", d[2] or "", d[3] or ""]
                + [v["value"] or "" for v in values.values()]
            ).lower()
            if needle not in hay:
                continue
        rows.append({
            "source_doc": doc,
            "doc_family": d[1],
            "doc_type": d[2],
            "jurisdiction": d[3],
            "lifecycle": d[4],
            "family_confidence": d[5],
            "family_method": d[6],
            "schema_version": d[7],
            "created_at": d[8].isoformat() if d[8] else None,
            "metadata_confidence": row_conf,
            "values": values,
            "page_count": pages.get(doc, 0),
            "clause_count": clauses.get(doc, 0),
            "obligation_count": obligations.get(doc, 0),
            "citation_count": citations.get(doc, 0),
        })

    total = len(rows)
    window = rows[offset:offset + limit] if limit else rows[offset:]
    return {
        "columns": columns,
        "rows": window,
        "total": total,
        "offset": offset,
        "limit": limit,
        "family": family,
        "families": [
            {"key": k, "label": schema_registry.get(k).label,
             "count": sum(1 for d in docs if (d[1] or "generic") == k)}
            for k in schema_registry.family_keys()
        ],
    }


# ---------------------------------------------------------------------------
# Obligation tracker
# ---------------------------------------------------------------------------

_OBLIGATION_COLUMNS = ("obligated_party", "duty", "trigger", "deadline",
                       "notice_period", "consequence")


def obligation_rows(wiki_id: str, session_id: str, *,
                    party: str | None = None,
                    source_doc: str | None = None,
                    search: str | None = None,
                    with_deadline: bool = False,
                    limit: int = 1000, offset: int = 0) -> dict:
    """Every extracted duty for the active wiki, filterable by who bears it.

    `coverage` is the part that matters as much as the rows. Obligation
    extraction was wired into ingest only in this phase, so a corpus ingested
    before it has zero duties — and an empty table rendered as "no results"
    would read as "these documents impose no obligations", which is a false
    statement about the documents rather than a true one about the pipeline.
    The caller gets the numbers to say which it is.
    """
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT source_doc, obligated_party, duty, trigger, deadline,
                   notice_period, consequence, verbatim_text, page_num,
                   confidence, review_status
            FROM obligations
            WHERE wiki_id = :w AND session_id = :s
            ORDER BY source_doc, id
        """), {"w": wiki_id, "s": session_id}).fetchall()
        # Documents that could carry duties at all — the tracker's denominator.
        # A judgment has none by design, so counting every document would
        # understate coverage rather than overstate it.
        eligible = conn.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE wiki_id = :w AND session_id = :s
              AND COALESCE(doc_family, 'generic') IN ('contract', 'term_sheet', 'generic')
        """), {"w": wiki_id, "s": session_id}).scalar() or 0

    out: list[dict] = []
    parties: dict[str, int] = {}
    docs_with_rows: set[str] = set()
    for r in rows:
        item = {
            "source_doc": r[0],
            "obligated_party": display_value(r[1]),
            "duty": display_value(r[2]),
            "trigger": display_value(r[3]),
            "deadline": display_value(r[4]),
            "notice_period": display_value(r[5]),
            "consequence": display_value(r[6]),
            "verbatim_text": r[7],
            "page_num": r[8],
            "confidence": r[9],
            "review_status": r[10],
        }
        docs_with_rows.add(r[0])
        if item["obligated_party"]:
            parties[item["obligated_party"]] = parties.get(item["obligated_party"], 0) + 1
        out.append(item)

    needle = (search or "").strip().lower()
    filtered = [
        o for o in out
        if (not party or o["obligated_party"] == party)
        and (not source_doc or o["source_doc"] == source_doc)
        and (not with_deadline or o["deadline"])
        and (not needle or needle in " ".join(
            str(o.get(k) or "") for k in
            ("source_doc", "obligated_party", "duty", "trigger", "deadline",
             "consequence", "verbatim_text")).lower())
    ]

    return {
        "columns": list(_OBLIGATION_COLUMNS),
        "rows": filtered[offset:offset + limit] if limit else filtered[offset:],
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
        "party": party,
        "parties": [{"name": n, "count": c} for n, c in
                    sorted(parties.items(), key=lambda kv: (-kv[1], kv[0]))],
        "coverage": {
            "obligations": len(out),
            "documents_with_obligations": len(docs_with_rows),
            "eligible_documents": int(eligible),
            "with_deadline": sum(1 for o in out if o["deadline"]),
        },
    }


# ---------------------------------------------------------------------------
# Review Table — standard columns served from the backbone, not the model
# ---------------------------------------------------------------------------
#
# § 06: "The existing page_metadata cache is fixed to 12 hardcoded,
# contract-shaped fields. A litigation doc doesn't have a liability cap, it
# has a case number, court, and disposition. The standard-field list becomes
# per-family, pulled from the schema registry."
#
# So this resolves a Review Table column in two steps, both free:
#   1. the document's own family metadata, matched by registry field name;
#   2. its extracted clauses, matched on clause_type.
# Only a column neither step can answer reaches the LLM.

import re as _re


def _norm(name: str) -> str:
    """Collapse a column or field name to letters and digits.

    "Governing Law", "governing_law" and "governing-law" are one column; a
    reviewer typing a header should not have to guess the registry's spelling.
    """
    return _re.sub(r"[^a-z0-9]+", "", (name or "").lower())


# Words that carry no distinguishing weight in a column or clause name.
_STOPWORDS = frozenset({"a", "an", "the", "of", "and", "or", "for", "to",
                        "in", "on", "by", "with", "clause", "provision"})


def _tokens(name: str) -> frozenset[str]:
    """Significant whole words in a column or clause name."""
    return frozenset(
        w for w in _re.split(r"[^a-z0-9]+", (name or "").lower())
        if w and w not in _STOPWORDS
    )


# Column wordings a lawyer types that do not match the registry field name.
# Deliberately narrow: each entry is a synonym for the *same* fact, never a
# near-neighbour. "Termination notice" and "termination" are not synonyms —
# one is a period and the other is a right — so they are not listed together.
_COLUMN_SYNONYMS = {
    "law": "governing_law",
    "choiceoflaw": "governing_law",
    "applicablelaw": "governing_law",
    "cap": "liability_cap",
    "limitationofliability": "liability_cap",
    "liabilitycapamount": "liability_cap",
    "term": "term_length",
    "duration": "term_length",
    "renewal": "renewal_terms",
    "autorenewal": "renewal_terms",
    "autorenew": "renewal_terms",
    "terminationrights": "termination",
    "terminationprovision": "termination",
    "ip": "ip_ownership",
    "intellectualproperty": "ip_ownership",
    "intellectualpropertyownership": "ip_ownership",
    "confidentialityobligation": "confidentiality",
    "nda": "confidentiality",
    "commencement": "effective_date",
    "startdate": "effective_date",
    "expiry": "expiry_date",
    "enddate": "expiry_date",
    "caseno": "case_number",
    "suitnumber": "case_number",
    "forum": "court",
    "outcome": "disposition",
    "result": "disposition",
    "judgmentdate": "decided_date",
    "dateofjudgment": "decided_date",
    "posture": "procedural_posture",
    "reliefsought": "relief_sought",
    "reliefgranted": "relief_granted",
    "scope": "scope_of_authority",
    "authority": "scope_of_authority",
    "poagrantor": "grantor",
    "poagrantee": "grantee",
    "attorney": "grantee",
    "opinionaddressee": "addressee",
    "dateofopinion": "opinion_date",
    "binding": "binding_status",
}

# A clause hit is a weaker answer than a typed field: it says the document
# has a clause of that name and quotes it, not that a validated value was
# extracted. Reported below the typed-field confidence so a reviewer sorting
# by doubt meets these first.
_CLAUSE_HIT_CEILING = 0.85
_CLAUSE_VALUE_MAX = 400


def _flatten_typed_value(raw, column: str = "") -> str | None:
    """Render a clause's typed_value as one cell.

    The key name is kept unless the column already says it. An Insurance
    clause storing {"coverage_minimum": "Rs. 84,976,789"} rendered as the bare
    figure reads as though the column asked for a number; "coverage minimum:
    Rs. 84,976,789" says what the number is.
    """
    if raw is None:
        return None
    value = raw
    if isinstance(value, str):
        try:
            import json
            value = json.loads(value)
        except Exception:
            return value.strip() or None
    if isinstance(value, dict):
        want = _tokens(column)
        parts = []
        for k, v in value.items():
            shown = display_value(v)
            if shown is None:
                continue
            parts.append(shown if _tokens(k) <= want and want
                         else f"{k.replace('_', ' ')}: {shown}")
        return "; ".join(parts) or None
    return display_value(value)


def resolve_source_doc(wiki_id: str, session_id: str, doc_name: str) -> str | None:
    """Map a Review Table document name onto a `documents.source_doc`.

    Review Table keys rows by the on-disk name, which may or may not still
    carry the session prefix depending on which session uploaded the file.
    Rather than reconstructing the prefix, this matches on the normalized
    tail — the filename is what both spellings agree on.
    """
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        rows = [r[0] for r in conn.execute(text(
            "SELECT source_doc FROM documents WHERE wiki_id = :w AND session_id = :s"
        ), {"w": wiki_id, "s": session_id}).fetchall()]
    if doc_name in rows:
        return doc_name
    flat = _norm(doc_name.replace("/", "_").replace("\\", "_"))
    if not flat:
        return None
    for candidate in rows:
        norm = _norm(candidate)
        if norm == flat or norm.endswith(flat):
            return candidate
    return None


def _clause_cell(wiki_id: str, session_id: str, source_doc: str,
                 column: str) -> dict | None:
    from sqlalchemy import text
    want = _norm(column)
    if not want:
        return None
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT clause_type, verbatim_text, typed_value, confidence
            FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
              AND review_status <> 'rejected'
        """), {"w": wiki_id, "s": session_id, "d": source_doc}).fetchall()

    want_tokens = _tokens(column)
    exact, partial = [], []
    for r in rows:
        norm = _norm(r[0])
        if not norm:
            continue
        if norm == want:
            exact.append(r)
        elif want_tokens and want_tokens <= _tokens(r[0]):
            partial.append(r)
    # An exact clause_type match is the answer; a partial one is the weaker
    # claim that a "Termination" column is answered by a "Termination for
    # Convenience" clause, so it is only used when nothing matched outright.
    #
    # Containment runs one way only, and on whole words. Raw substring
    # matching in the other direction let a "Compliance" clause answer a
    # "Zorblatt compliance rating" column — a wrong answer costs far more
    # than the LLM call it saved, and falling through to the model is the
    # safe direction to fail in.
    pool = exact or partial
    if not pool:
        return None
    best = max(pool, key=lambda r: (r[3] if r[3] is not None else 0, len(r[1] or "")))
    value = _flatten_typed_value(best[2], column) or (best[1] or "").strip()
    if not value:
        return None
    if len(value) > _CLAUSE_VALUE_MAX:
        value = value[:_CLAUSE_VALUE_MAX].rstrip() + "…"
    conf = best[3] if best[3] is not None else 0.7
    return {"value": value,
            "confidence": round(min(float(conf), _CLAUSE_HIT_CEILING)
                                * (1.0 if exact else 0.85), 3),
            "quote": best[1],
            "source": f"clause: {best[0]}"}


def standard_cell(wiki_id: str, session_id: str, doc_name: str,
                  column: str) -> dict | None:
    """Answer one Review Table cell from the backbone, or return None.

    None means "spend the LLM call" — it is never a claim that the document
    is silent on the column, only that nothing typed was extracted for it.
    """
    source_doc = resolve_source_doc(wiki_id, session_id, doc_name)
    if not source_doc:
        return None

    want = _norm(column)
    target = _COLUMN_SYNONYMS.get(want, want)
    fields = document_fields(wiki_id, session_id, source_doc)
    for name, field in fields.items():
        if _norm(name) not in (want, target):
            continue
        value = display_value(field.get("value"))
        if value:
            return {"value": value,
                    "confidence": field.get("confidence"),
                    "quote": None,
                    "source": f"registry field: {name}"}

    return _clause_cell(wiki_id, session_id, source_doc, column)
