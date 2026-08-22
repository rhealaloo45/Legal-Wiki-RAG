"""
Persistence for the Phase 0 typed tables (target architecture § 03).

Everything an ingest extracts that isn't wiki prose lands here: the
`documents` registry row, the family-specific typed row, citations,
structural anchors, entities, tables and figures.

Three properties this layer is responsible for, all of which the
architecture doc calls out as hardening items rather than nice-to-haves:

  * **wiki_id is a predicate, not a filter.** Every read and write below
    carries it in the WHERE/INSERT itself. A caller cannot accidentally
    query across wikis, because there is no code path here that omits it.

  * **Entity canonicalization is race-safe.** Two documents ingesting in
    parallel, both naming "Acme Corp" for the first time, must not create
    two rows. The UNIQUE (wiki_id, canonical_key) constraint plus an
    ON CONFLICT upsert does that structurally — not a check-then-act that
    happens to usually win.

  * **Re-ingest replaces rather than blends.** Every write for a document
    goes through `replace_document_rows`, which deletes that document's
    prior typed rows inside the same transaction as the insert. The doc is
    explicit that re-ingest must be a swap, not a merge; making that the
    only available write path means it can't be got wrong by a caller.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

import config
from services import db, schema_registry

logger = logging.getLogger(__name__)

# Family key -> its typed table and the columns that table accepts. Kept here
# rather than derived from the registry so a schema-registry override can add
# an extraction *field* without silently implying a new DB column.
_FAMILY_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "contract": ("contracts", (
        "governing_law", "liability_cap", "term_length", "renewal_terms",
        "termination", "binding_status", "exclusivity",
    )),
    "term_sheet": ("contracts", (
        "governing_law", "liability_cap", "term_length", "renewal_terms",
        "termination", "binding_status", "exclusivity",
    )),
    "litigation": ("litigation_facts", (
        "court", "case_number", "plaintiffs", "defendants",
        "procedural_posture", "holding", "relief_granted", "disposition",
        "decided_date",
    )),
    "authorization": ("authorizations", (
        "grantor", "grantee", "scope_of_authority", "limitations",
        "resolving_body", "effective_date", "expiry_date",
    )),
    "opinion": ("opinions", (
        "addressee", "matters_opined", "assumptions", "qualifications",
        "conclusion", "reliance_limitation", "opinion_date",
    )),
}

# Columns declared JSONB in the schema — a Python list/dict must be dumped
# for these and left alone for the rest.
_JSON_COLUMNS = frozenset({
    "plaintiffs", "defendants", "matters_opined", "assumptions",
    "qualifications", "parties", "typed_value", "columns", "rows",
})

# Company-form suffixes stripped when canonicalizing an entity name. "Acme
# Corp" and "Acme Corporation" are the same party; "Acme Holdings" is not,
# so only true form suffixes are listed here, never descriptive words.
_ENTITY_SUFFIXES = (
    "private limited", "pvt ltd", "pvt. ltd.", "pvt limited",
    "public limited company", "limited liability partnership",
    "incorporated", "corporation", "company", "limited", "llp", "llc",
    "plc", "ltd", "inc", "corp", "co", "gmbh", "s.a.", "sa", "bv", "nv",
    "pte", "pty", "ag",
)
_ENTITY_NOISE = re.compile(r"[^a-z0-9&\s]")
_WS = re.compile(r"\s+")

# A party's *description* is not a party. Legal drafting writes
# "Acme Corp, a company incorporated under the laws of India", and any
# extraction that hands the second half over as a name pollutes the canonical
# registry with entries that read as real parties to everything downstream.
_ENTITY_DESCRIPTOR = re.compile(
    r"^(a|an|the)\s+.*\b(company|corporation|partnership|entity|firm|"
    r"body corporate|limited liability|incorporated|organized|organised|"
    r"registered|existing|duly)\b",
    re.IGNORECASE,
)
# Contractual role labels. Real and useful in the document, but they name a
# position in the agreement, not a legal person — and the same label refers
# to different companies in different documents, so canonicalizing on it
# would merge unrelated parties into one entity.
_ENTITY_ROLE_WORDS = frozenset({
    "service provider", "provider", "supplier", "customer", "client",
    "vendor", "purchaser", "buyer", "seller", "licensor", "licensee",
    "lessor", "lessee", "borrower", "lender", "disclosing party",
    "receiving party", "party", "parties", "company", "counterparty",
    "employer", "employee", "contractor", "consultant", "the company",
    "first party", "second party", "plaintiff", "defendant", "petitioner",
    "respondent", "grantor", "grantee", "assignor", "assignee",
})


def is_probable_entity_name(name: str) -> bool:
    """Whether a string looks like a legal person rather than a description
    or a role label.

    Conservative in the direction of *rejecting*: a missing entity is a gap
    someone can notice and fill, while a bogus entity silently becomes an
    authoritative party that clauses and obligations get attributed to.
    """
    s = (name or "").strip()
    if len(s) < 2 or len(s) > 200:
        return False
    # A party whose name was redacted out of the source is rendered by the
    # document's own defined term ("Participant (name redacted)"). That label
    # identifies a role within ONE agreement — two documents' "Participant"
    # are different companies — so canonicalizing on it would merge unrelated
    # parties into a single entity. The value is still shown on the document;
    # it just does not become a corpus-wide identity.
    if "(name redacted)" in s.lower() or s.startswith("[unnamed party"):
        return False
    if _ENTITY_DESCRIPTOR.match(s):
        return False
    if s.lower().strip(" .") in _ENTITY_ROLE_WORDS:
        return False
    # Must contain a capitalised token or a bracketed redaction marker —
    # this corpus redacts party names as "[Redacted Financial Investor]",
    # which is a real (if anonymised) party and must survive.
    if s.startswith("[") or re.search(r"\b[A-Z][A-Za-z&.-]", s):
        return True
    return False


def _text():
    from sqlalchemy import text
    return text


def _enabled() -> bool:
    return bool(config.USE_DATABASE)


# Columns holding the client's own words rather than derived structure.
# Encrypted at rest (§ 01.6). Names and dates stay in the clear so the
# registry remains queryable — a fully-encrypted documents table could not
# answer "which contracts expire this quarter" without decrypting every row.
_ENCRYPTED_COLUMNS = frozenset({
    "verbatim_text", "typed_value", "holding", "relief_granted",
    "scope_of_authority", "limitations", "conclusion", "assumptions",
    "qualifications", "reliance_limitation", "duty", "consequence",
    "description", "caption", "rows", "columns", "heading_text",
})


def _coerce_param(column: str, value: Any) -> Any:
    if value is None:
        return None
    from services import crypto

    if column in _JSON_COLUMNS and not isinstance(value, str):
        if column in _ENCRYPTED_COLUMNS:
            return json.dumps(crypto.encrypt_json(value), ensure_ascii=False,
                              default=str)
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, (dict, list)):
        if column in _ENCRYPTED_COLUMNS:
            return json.dumps(crypto.encrypt_json(value), ensure_ascii=False,
                              default=str)
        return json.dumps(value, ensure_ascii=False, default=str)
    if column in _ENCRYPTED_COLUMNS and isinstance(value, str):
        return crypto.encrypt(value)
    return value


# ---------------------------------------------------------------------------
# Entity canonicalization
# ---------------------------------------------------------------------------

def canonical_key(name: str) -> str:
    """Normalize a party name to its canonical lookup key.

    Deliberately conservative. Stripping "Limited" merges two spellings of
    one company; stripping anything more aggressive would merge two different
    companies, and a wrongly-merged party silently attributes one entity's
    obligations to another. Under-merging leaves a duplicate row a human can
    see and fix; over-merging produces a confident wrong answer nobody sees.
    """
    s = _ENTITY_NOISE.sub(" ", (name or "").lower())
    s = _WS.sub(" ", s).strip()
    changed = True
    while changed and s:
        changed = False
        for suffix in _ENTITY_SUFFIXES:
            if s.endswith(" " + suffix):
                s = s[: -(len(suffix) + 1)].strip()
                changed = True
                break
    return s or _WS.sub(" ", (name or "").lower()).strip()


def upsert_entity(wiki_id: str, name: str, entity_type: str | None = None,
                  source_doc: str | None = None, conn=None) -> int | None:
    """Resolve a party name to a canonical entity id, creating it if new.

    Race-safe by construction: the INSERT ... ON CONFLICT DO UPDATE always
    returns a row, so two parallel ingests naming the same party for the
    first time converge on one id instead of both inserting. A plain
    look-up-else-create here would be a check-then-act that loses under
    exactly the concurrency bulk upload creates.
    """
    if not _enabled() or not (name or "").strip():
        return None
    if not is_probable_entity_name(name):
        logger.debug("Not registering %r as an entity — reads as a description "
                     "or role label, not a party name", name[:80])
        return None
    key = canonical_key(name)
    if not key:
        return None
    text = _text()

    def _run(c) -> int:
        row = c.execute(text("""
            INSERT INTO entities (wiki_id, canonical_name, canonical_key, entity_type)
            VALUES (:w, :n, :k, :t)
            ON CONFLICT (wiki_id, canonical_key)
            DO UPDATE SET entity_type = COALESCE(entities.entity_type, EXCLUDED.entity_type)
            RETURNING id
        """), {"w": wiki_id, "n": name.strip(), "k": key, "t": entity_type}).fetchone()
        entity_id = int(row[0])
        alias_key = _WS.sub(" ", _ENTITY_NOISE.sub(" ", name.lower())).strip()
        if alias_key and alias_key != key:
            c.execute(text("""
                INSERT INTO entity_aliases (wiki_id, entity_id, alias, alias_key, source_doc)
                VALUES (:w, :e, :a, :ak, :s)
                ON CONFLICT (wiki_id, alias_key) DO NOTHING
            """), {"w": wiki_id, "e": entity_id, "a": name.strip(),
                   "ak": alias_key, "s": source_doc})
        return entity_id

    if conn is not None:
        return _run(conn)
    with db.get_engine().connect() as c:
        entity_id = _run(c)
        c.commit()
    return entity_id


def resolve_entity(wiki_id: str, name: str) -> dict | None:
    """Look up an existing entity by canonical name or any known alias."""
    if not _enabled():
        return None
    text = _text()
    key = canonical_key(name)
    alias_key = _WS.sub(" ", _ENTITY_NOISE.sub(" ", (name or "").lower())).strip()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT id, canonical_name, entity_type FROM entities
            WHERE wiki_id = :w AND canonical_key = :k
        """), {"w": wiki_id, "k": key}).fetchone()
        if row is None and alias_key:
            row = c.execute(text("""
                SELECT e.id, e.canonical_name, e.entity_type
                FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
                WHERE a.wiki_id = :w AND a.alias_key = :ak
            """), {"w": wiki_id, "ak": alias_key}).fetchone()
    if row is None:
        return None
    return {"id": int(row[0]), "canonical_name": row[1], "entity_type": row[2]}


# ---------------------------------------------------------------------------
# documents registry
# ---------------------------------------------------------------------------

def upsert_document(wiki_id: str, session_id: str, source_doc: str,
                    **fields: Any) -> int | None:
    """Create or update the registry row for a document. Returns document_id.

    Bumps `schema_version` on update so a re-ingest is distinguishable from
    the original ingest without a separate audit table — that stamp is what
    the doc gates single-document re-ingest on.
    """
    if not _enabled():
        return None
    allowed = (
        "doc_family", "doc_type", "jurisdiction", "parties", "effective_date",
        "expiry_date", "status", "lifecycle", "role", "binding_status",
        "family_confidence", "family_method", "folder_hint",
    )
    cols = {k: _coerce_param(k, v) for k, v in fields.items()
            if k in allowed and v is not None}
    text = _text()
    set_clause = ", ".join(f"{k} = :{k}" for k in cols)
    insert_cols = ", ".join(cols)
    insert_vals = ", ".join(f":{k}" for k in cols)
    sql = f"""
        INSERT INTO documents (wiki_id, session_id, source_doc{',' + insert_cols if cols else ''})
        VALUES (:w, :s, :d{',' + insert_vals if cols else ''})
        ON CONFLICT (wiki_id, session_id, source_doc) DO UPDATE SET
            schema_version = documents.schema_version + 1
            {',' + set_clause if cols else ''}
        RETURNING id
    """
    params = {"w": wiki_id, "s": session_id, "d": source_doc, **cols}
    with db.get_engine().connect() as c:
        row = c.execute(text(sql), params).fetchone()
        c.commit()
    return int(row[0]) if row else None


def get_document(wiki_id: str, session_id: str, source_doc: str) -> dict | None:
    if not _enabled():
        return None
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT id, doc_family, doc_type, jurisdiction, lifecycle, role,
                   binding_status, family_confidence, family_method,
                   folder_hint, schema_version
            FROM documents
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
        """), {"w": wiki_id, "s": session_id, "d": source_doc}).fetchone()
    if not row:
        return None
    keys = ("id", "doc_family", "doc_type", "jurisdiction", "lifecycle", "role",
            "binding_status", "family_confidence", "family_method",
            "folder_hint", "schema_version")
    return dict(zip(keys, row))


# ---------------------------------------------------------------------------
# Typed rows — swap, never blend
# ---------------------------------------------------------------------------

# Every table holding per-document typed rows. Listed so a re-ingest clears
# all of them; a table missing here would leave stale rows behind a swap,
# which is the "blend" failure the doc explicitly forbids.
_PER_DOCUMENT_TABLES = (
    "contracts", "obligations", "litigation_facts", "authorizations",
    "opinions", "citations", "structural_anchors", "tables", "figures",
)


def clear_document_rows(wiki_id: str, session_id: str, source_doc: str,
                        conn=None) -> dict[str, int]:
    """Delete every typed row for one document. Used by the swap path."""
    text = _text()
    deleted: dict[str, int] = {}

    def _run(c):
        for tbl in _PER_DOCUMENT_TABLES:
            res = c.execute(text(
                f"DELETE FROM {tbl} WHERE wiki_id = :w AND session_id = :s "
                f"AND source_doc = :d"
            ), {"w": wiki_id, "s": session_id, "d": source_doc})
            if res.rowcount:
                deleted[tbl] = res.rowcount

    if conn is not None:
        _run(conn)
        return deleted
    with db.get_engine().connect() as c:
        _run(c)
        c.commit()
    return deleted


def _insert_rows(c, table: str, wiki_id: str, session_id: str, source_doc: str,
                 document_id: int | None, rows: Iterable[dict],
                 columns: tuple[str, ...]) -> int:
    text = _text()
    n = 0
    for row in rows:
        payload = {k: _coerce_param(k, row.get(k)) for k in columns}
        payload.update({"w": wiki_id, "s": session_id, "d": source_doc,
                        "doc_id": document_id})
        col_sql = ", ".join(columns)
        val_sql = ", ".join(f":{k}" for k in columns)
        c.execute(text(f"""
            INSERT INTO {table} (wiki_id, session_id, source_doc, document_id, {col_sql})
            VALUES (:w, :s, :d, :doc_id, {val_sql})
        """), payload)
        n += 1
    return n


def replace_document_rows(wiki_id: str, session_id: str, source_doc: str,
                          document_id: int | None = None,
                          family_row: dict | None = None,
                          family_key: str | None = None,
                          obligations: list[dict] | None = None,
                          citations: list[dict] | None = None,
                          anchors: list[dict] | None = None,
                          tables: list[dict] | None = None,
                          figures: list[dict] | None = None) -> dict[str, int]:
    """Swap in a document's typed rows, in one transaction.

    The delete and every insert share a transaction, so a failure part-way
    leaves the previous ingest's rows intact rather than a half-replaced
    document. This is also the retry/idempotency path the doc asks for: a
    call that times out mid-write can simply be re-run, because it always
    clears before it writes and can therefore never double-insert.
    """
    if not _enabled():
        return {}
    text = _text()
    written: dict[str, int] = {}
    with db.get_engine().connect() as c:
        try:
            clear_document_rows(wiki_id, session_id, source_doc, conn=c)

            if family_row and family_key:
                spec = _FAMILY_TABLES.get(family_key)
                if spec:
                    table, cols = spec
                    payload = {k: family_row.get(k) for k in cols}
                    payload["typed_value"] = family_row.get("typed_value") or {
                        k: v for k, v in family_row.items() if k not in cols
                    }
                    payload["confidence"] = family_row.get("confidence")
                    written[table] = _insert_rows(
                        c, table, wiki_id, session_id, source_doc, document_id,
                        [payload], cols + ("typed_value", "confidence"),
                    )

            if obligations:
                written["obligations"] = _insert_rows(
                    c, "obligations", wiki_id, session_id, source_doc, document_id,
                    obligations,
                    ("obligated_party", "entity_id", "duty", "trigger", "deadline",
                     "notice_period", "consequence", "verbatim_text", "page_num",
                     "confidence"),
                )
            if citations:
                written["citations"] = _insert_rows(
                    c, "citations", wiki_id, session_id, source_doc, document_id,
                    citations,
                    ("citation_text", "authority_type", "normalized_form",
                     "page_title", "page_num", "confidence"),
                )
            if anchors:
                written["structural_anchors"] = _insert_rows(
                    c, "structural_anchors", wiki_id, session_id, source_doc,
                    document_id, anchors,
                    ("anchor_label", "anchor_kind", "heading_text", "char_start",
                     "char_end", "page_num", "page_title", "ordinal"),
                )
            if tables:
                written["tables"] = _insert_rows(
                    c, "tables", wiki_id, session_id, source_doc, document_id,
                    tables,
                    ("page_num", "page_title", "caption", "columns", "rows",
                     "confidence", "extraction_method"),
                )
            if figures:
                written["figures"] = _insert_rows(
                    c, "figures", wiki_id, session_id, source_doc, document_id,
                    figures,
                    ("page_num", "page_title", "figure_kind", "description",
                     "image_ref", "confidence", "extraction_method"),
                )
            c.commit()
        except Exception:
            c.rollback()
            logger.exception("Typed-row swap failed for %s — previous rows kept",
                             source_doc)
            raise
    return written


# ---------------------------------------------------------------------------
# Cross-segment reconciliation (§ 01 stage 04)
# ---------------------------------------------------------------------------

_DEDUP_WS = re.compile(r"\s+")


def _dedup_key(row: dict, fields: tuple[str, ...]) -> str:
    parts = []
    for f in fields:
        v = row.get(f)
        if v is None:
            continue
        parts.append(_DEDUP_WS.sub(" ", str(v).strip().lower()))
    return "|".join(parts)


def reconcile_rows(rows: list[dict], key_fields: tuple[str, ...],
                   confidence_field: str = "confidence") -> list[dict]:
    """Merge duplicate structured rows produced by adjacent segments.

    Segment boundaries now land on real structure, which removes most
    straddling — but a clause quoted in one section and cross-referenced in
    the next still yields two rows, and the overlap tier can still duplicate
    outright. Highest confidence wins; ties keep the longer text, on the
    reasoning that a truncated extraction is the more likely of two variants
    to be the damaged one.
    """
    best: dict[str, dict] = {}
    for row in rows:
        key = _dedup_key(row, key_fields)
        if not key:
            continue
        prior = best.get(key)
        if prior is None:
            best[key] = row
            continue
        c_new = row.get(confidence_field) or 0
        c_old = prior.get(confidence_field) or 0
        if c_new > c_old:
            best[key] = row
        elif c_new == c_old:
            len_new = sum(len(str(row.get(f) or "")) for f in row)
            len_old = sum(len(str(prior.get(f) or "")) for f in prior)
            if len_new > len_old:
                best[key] = row
    return list(best.values())


def document_summary(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """Row counts per typed table for one document — used by the ingest
    evaluation gate and the admin panel."""
    if not _enabled():
        return {}
    text = _text()
    out: dict[str, int] = {}
    with db.get_engine().connect() as c:
        for tbl in _PER_DOCUMENT_TABLES:
            n = c.execute(text(
                f"SELECT COUNT(*) FROM {tbl} WHERE wiki_id = :w "
                f"AND session_id = :s AND source_doc = :d"
            ), {"w": wiki_id, "s": session_id, "d": source_doc}).scalar()
            if n:
                out[tbl] = int(n)
    return out
