"""Phase 1 — the Contract Register, and the shared typed-field reader.

Target architecture § 06: "Every document appears automatically as a row the
moment it finishes ingest." That is the whole feature — no extraction, no
LLM call, no background job. The rows already exist in `documents` and the
five family tables; the register is the read nobody had written.

Two decisions worth stating.

The register is **family-aware rather than contract-shaped**, even though it
is called the Contract Register. A litigation document has no liability cap
and a power of attorney has no governing law, and giving every document the
same twelve contract columns is the exact mistake § 06 calls out. The column
set therefore comes from the schema registry's `metadata_fields` for whatever
family is being listed, and the all-families view falls back to the fields
every family shares.

`document_fields` exists here rather than in `db.py` because Review Table
needs the same per-document typed values the register shows, keyed by field
name. Putting one reader behind both is the point: a column the register can
display is by definition a column Review Table should never spend an LLM call
on.
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
