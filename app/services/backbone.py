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
        "family_confidence", "family_method", "folder_hint", "content_hash",
        "file_hash",
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


def find_by_content_hash(wiki_id: str, content_hash: str) -> dict | None:
    """The existing document this content hash already belongs to, if any.

    Scoped to `wiki_id` only, deliberately not `session_id` — the whole
    point is catching a document re-uploaded under a *different* session
    (a fresh folder upload gets a fresh session_id for every file in it).
    Two separate wikis are deliberately isolated corpora, so the same file
    uploaded into both is not a duplicate of anything; it belongs in both.

    Ties are broken by taking the oldest row (`ORDER BY created_at`), so a
    duplicate always resolves to the original rather than to whichever
    re-upload happened to run last.
    """
    if not _enabled() or not content_hash:
        return None
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT session_id, source_doc, doc_family, created_at
            FROM documents
            WHERE wiki_id = :w AND content_hash = :h
            ORDER BY created_at ASC
            LIMIT 1
        """), {"w": wiki_id, "h": content_hash}).fetchone()
    if not row:
        return None
    return {"session_id": row[0], "source_doc": row[1], "doc_family": row[2],
            "created_at": row[3].isoformat() if row[3] else None}


def documents_missing_hash(wiki_id: str) -> list[dict]:
    """Documents ingested before content_hash existed, or whose extracted
    text was too short to hash (see wiki._MIN_HASH_CHARS) — the backfill
    target. A row here is not necessarily fixable: it may have no file left
    on disk, which the caller has to check separately."""
    if not _enabled():
        return []
    text = _text()
    with db.get_engine().connect() as c:
        rows = c.execute(text("""
            SELECT session_id, source_doc FROM documents
            WHERE wiki_id = :w AND content_hash IS NULL
        """), {"w": wiki_id}).fetchall()
    return [{"session_id": r[0], "source_doc": r[1]} for r in rows]


def backfill_content_hash(wiki_id: str, session_id: str, source_doc: str,
                          content_hash: str) -> bool:
    """Set content_hash on an existing row, and nothing else.

    Deliberately not routed through upsert_document, which bumps
    schema_version on every write — that stamp is what gates single-document
    re-ingest elsewhere in this doc, and a pure hash backfill making a
    document look freshly re-ingested would be a real, if quiet, correctness
    bug for that feature. Only ever touches a row that still has no hash, so
    calling this twice on the same document is always a no-op the second time.
    """
    if not _enabled() or not content_hash:
        return False
    text = _text()
    with db.get_engine().connect() as c:
        res = c.execute(text("""
            UPDATE documents SET content_hash = :h
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
              AND content_hash IS NULL
        """), {"h": content_hash, "w": wiki_id, "s": session_id, "d": source_doc})
        c.commit()
    return res.rowcount > 0


def find_by_file_hash(wiki_id: str, file_hash: str) -> dict | None:
    """The existing document this raw-file hash already belongs to, if any.

    This is the upload-time check: file_hash is SHA-256 of the raw uploaded
    bytes, computed before any text extraction or OCR runs, so this lookup
    is the one that actually saves the extraction cost — find_by_content_hash
    only fires after extraction has already happened. Same scoping/tie-break
    rules as find_by_content_hash: wiki_id only, oldest row wins.
    """
    if not _enabled() or not file_hash:
        return None
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT session_id, source_doc, doc_family, created_at
            FROM documents
            WHERE wiki_id = :w AND file_hash = :h
            ORDER BY created_at ASC
            LIMIT 1
        """), {"w": wiki_id, "h": file_hash}).fetchone()
    if not row:
        return None
    return {"session_id": row[0], "source_doc": row[1], "doc_family": row[2],
            "created_at": row[3].isoformat() if row[3] else None}


def documents_missing_file_hash(wiki_id: str) -> list[dict]:
    """Documents ingested before file_hash existed — the backfill target for
    the cheap, no-OCR upload-time dedup signal. Unlike documents_missing_hash
    (content_hash), backfilling this needs only the raw file bytes, never
    text extraction — safe to run on scanned/image documents at zero cost."""
    if not _enabled():
        return []
    text = _text()
    with db.get_engine().connect() as c:
        rows = c.execute(text("""
            SELECT session_id, source_doc FROM documents
            WHERE wiki_id = :w AND file_hash IS NULL
        """), {"w": wiki_id}).fetchall()
    return [{"session_id": r[0], "source_doc": r[1]} for r in rows]


def backfill_file_hash(wiki_id: str, session_id: str, source_doc: str,
                       file_hash: str) -> bool:
    """Set file_hash on an existing row, and nothing else. Same rationale as
    backfill_content_hash: not routed through upsert_document, so this never
    bumps schema_version. No-op if the row already has a hash."""
    if not _enabled() or not file_hash:
        return False
    text = _text()
    with db.get_engine().connect() as c:
        res = c.execute(text("""
            UPDATE documents SET file_hash = :h
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
              AND file_hash IS NULL
        """), {"h": file_hash, "w": wiki_id, "s": session_id, "d": source_doc})
        c.commit()
    return res.rowcount > 0


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


def update_metadata_field(wiki_id: str, session_id: str, source_doc: str,
                          field_name: str, new_value, family_key: str | None = None) -> dict:
    """Record a reviewer's correction to one extracted metadata field.

    Provenance is the point. The corrected value replaces the extracted one,
    but the extraction is kept as `previous_value` and the field is stamped
    `edited: true` — a human-corrected field and a model-extracted field that
    happen to hold the same string are not the same fact, and a reviewer
    coming back later needs to see which is which.

    Confidence goes to 1.0 because a person read the document and decided.
    That is the one place in this pipeline where a confidence of 1.0 is
    earned rather than self-reported.
    """
    if not _enabled():
        return {}
    from datetime import datetime, timezone
    from services import crypto

    spec = _FAMILY_TABLES.get(family_key or "")
    if not spec:
        raise ValueError(f"No typed table for family {family_key!r}")
    table, columns = spec
    text = _text()

    with db.get_engine().connect() as c:
        row = c.execute(text(
            f"SELECT typed_value FROM {table} WHERE wiki_id = :w "
            f"AND session_id = :s AND source_doc = :d"
        ), {"w": wiki_id, "s": session_id, "d": source_doc}).fetchone()
        if not row:
            raise ValueError("No extracted metadata row for this document")

        decoded = crypto.decrypt_json(row[0])
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
        if not isinstance(decoded, dict):
            decoded = {}
        fields = decoded.get("fields")
        if not isinstance(fields, dict):
            fields = {}
        prior = fields.get(field_name) if isinstance(fields.get(field_name), dict) else {}

        fields[field_name] = {
            **prior,
            "value": new_value,
            "confidence": 1.0,
            "flagged": False,
            "reason": None,
            "edited": True,
            "edited_at": datetime.now(timezone.utc).isoformat(),
            # Only capture the original extraction once — a second edit must
            # not overwrite the model's output with the first correction, or
            # the audit trail silently becomes "a human said this twice".
            "previous_value": prior.get("previous_value",
                                        prior.get("value")) if prior else None,
        }
        decoded["fields"] = fields
        validated = decoded.get("validated")
        if isinstance(validated, dict):
            validated[field_name] = new_value
            decoded["validated"] = validated
        decoded["flagged_fields"] = [
            f for f in (decoded.get("flagged_fields") or []) if f != field_name
        ]

        params = {"w": wiki_id, "s": session_id, "d": source_doc,
                  "tv": json.dumps(crypto.encrypt_json(decoded),
                                   ensure_ascii=False, default=str)}
        set_col = ""
        if field_name in columns:
            set_col = f", {field_name} = :col"
            params["col"] = _coerce_param(field_name, new_value)
        c.execute(text(
            f"UPDATE {table} SET typed_value = :tv{set_col} "
            f"WHERE wiki_id = :w AND session_id = :s AND source_doc = :d"
        ), params)
        c.commit()

    logger.info("Reviewer corrected %s.%s on %s", table, field_name, source_doc)
    return {"field": field_name, "value": new_value, "edited": True}


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
