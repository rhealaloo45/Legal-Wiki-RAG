"""Precedent layer — role-tagged documents + clause-level embeddings.

The retrieval mechanism Draft Mode reads from (§ Phase 2). Drafting needs the
CLAUSE that solves a problem, not the page that discusses it: a page vector
averages a whole topic, so "liability capped at fees paid" ranks a page
*about* liability above the clause that actually says it. Clause vectors are
short and specific, which is what makes them rankable for drafting.

Roles answer "is this document a usable drafting precedent?":

    precedent  clause-bearing instruments — contracts, term sheets. These are
               what you draft FROM.
    reference  judgments, opinions, authorisations. Citable, and actively
               wrong to copy clause language out of: a judgment quoting a
               liability cap is describing someone else's contract, not
               offering a house position.
    excluded   explicitly withheld by an admin.

Derived per family, and overridable per document — a firm's gold-standard NDA
is a stronger precedent than the other 200, and only a human knows that.
"""
from __future__ import annotations

import logging

from services import db

logger = logging.getLogger(__name__)

ROLES = ("precedent", "reference", "excluded")

# Family -> default role. Clause-bearing families are draftable; the rest
# describe or authorise rather than contract, so their clause language must
# not surface as drafting material.
_FAMILY_ROLE = {
    "contract": "precedent",
    "term_sheet": "precedent",
    "litigation": "reference",
    "opinion": "reference",
    "authorization": "reference",
    "generic": "reference",
}


def _text():
    from sqlalchemy import text
    return text


def _enabled() -> bool:
    import config
    return bool(config.USE_DATABASE)


# ---------------------------------------------------------------------------
# role tagging
# ---------------------------------------------------------------------------

def derive_roles(wiki_id: str, overwrite: bool = False) -> dict:
    """Set documents.role from doc_family. Zero LLM calls.

    `overwrite=False` by default so an admin's explicit tag is never silently
    reverted by a later backfill — a manual role is a judgement the derivation
    cannot reproduce.
    """
    if not _enabled():
        return {"updated": 0}
    text = _text()
    guard = "" if overwrite else "AND role IS NULL"
    updated = {}
    with db.get_engine().connect() as c:
        for family, role in _FAMILY_ROLE.items():
            res = c.execute(text(f"""
                UPDATE documents SET role = :r
                WHERE wiki_id = :w AND doc_family = :f {guard}
            """), {"r": role, "w": wiki_id, "f": family})
            if res.rowcount:
                updated[family] = res.rowcount
        c.commit()
    return {"updated": sum(updated.values()), "by_family": updated}


def set_role(wiki_id: str, source_doc: str, role: str) -> bool:
    """Explicitly tag one document. Overrides the derived value."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    text = _text()
    with db.get_engine().connect() as c:
        res = c.execute(text("""
            UPDATE documents SET role = :r WHERE wiki_id = :w AND source_doc = :d
        """), {"r": role, "w": wiki_id, "d": source_doc})
        c.commit()
    return (res.rowcount or 0) > 0


def role_summary(wiki_id: str) -> dict:
    if not _enabled():
        return {}
    text = _text()
    with db.get_engine().connect() as c:
        rows = c.execute(text("""
            SELECT COALESCE(role, '(untagged)'), count(*) FROM documents
            WHERE wiki_id = :w GROUP BY 1 ORDER BY 2 DESC
        """), {"w": wiki_id}).fetchall()
    return {r[0]: int(r[1]) for r in rows}


# ---------------------------------------------------------------------------
# clause embeddings
# ---------------------------------------------------------------------------

def clauses_missing_embeddings(wiki_id: str, session_id: str,
                               roles: tuple[str, ...] = ("precedent",),
                               limit: int | None = None) -> list[dict]:
    """Clauses with no vector yet, restricted to the roles worth embedding.

    Reference-role documents are skipped deliberately: embedding a judgment's
    clause language would put it in the pool Draft Mode ranks, and copying
    contract wording out of a judgment is the failure this layer exists to
    prevent.
    """
    if not _enabled():
        return []
    text = _text()
    tbl = db._clause_table_name()
    lim = f"LIMIT {int(limit)}" if limit else ""
    with db.get_engine().connect() as c:
        rows = c.execute(text(f"""
            SELECT cl.id, cl.source_doc, cl.clause_type, cl.verbatim_text,
                   d.doc_family, d.role
            FROM clauses cl
            JOIN documents d
              ON d.wiki_id = cl.wiki_id AND d.source_doc = cl.source_doc
            LEFT JOIN {tbl} e ON e.clause_id = cl.id
            WHERE cl.wiki_id = :w AND cl.session_id = :s
              AND e.id IS NULL
              AND d.role = ANY(:roles)
              AND cl.review_status <> 'rejected'
              AND length(cl.verbatim_text) > 40
            ORDER BY cl.id {lim}
        """), {"w": wiki_id, "s": session_id, "roles": list(roles)}).fetchall()
    return [{"clause_id": int(r[0]), "source_doc": r[1], "clause_type": r[2],
             "text": r[3], "doc_family": r[4], "role": r[5]} for r in rows]


def store_embeddings(wiki_id: str, session_id: str, rows: list[dict],
                     vectors: list[list[float]]) -> int:
    """Persist clause vectors. Keyed on clause_id, so a re-ingest that
    replaces a clause replaces its vector rather than leaving a stale one
    pointing at text that no longer exists."""
    if not rows:
        return 0
    text = _text()
    tbl = db._clause_table_name()
    n = 0
    with db.get_engine().connect() as c:
        for row, vec in zip(rows, vectors):
            if not vec:
                continue
            c.execute(text(f"""
                INSERT INTO {tbl} (wiki_id, session_id, clause_id, source_doc,
                                   clause_type, doc_family, role, embedding)
                VALUES (:w, :s, :cid, :d, :ct, :f, :r, :e)
                ON CONFLICT (clause_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding, role = EXCLUDED.role,
                    clause_type = EXCLUDED.clause_type
            """), {"w": wiki_id, "s": session_id, "cid": row["clause_id"],
                   "d": row["source_doc"], "ct": row["clause_type"],
                   "f": row["doc_family"], "r": row["role"],
                   "e": "[" + ",".join(f"{x:.8f}" for x in vec) + "]"})
            n += 1
        c.commit()
    return n


def embed_pending(wiki_id: str, session_id: str, batch: int = 128,
                  max_clauses: int | None = None, progress=None) -> dict:
    """Embed clauses that have no vector. One embedding call per batch."""
    from services import embedder
    pending = clauses_missing_embeddings(wiki_id, session_id, limit=max_clauses)
    total = len(pending)
    done = 0
    for i in range(0, total, batch):
        chunk = pending[i:i + batch]
        texts = [f"{r['clause_type']}: {r['text']}"[:2000] for r in chunk]
        try:
            vecs = embedder.embed_batch(texts, is_query=False)
        except Exception as e:
            logger.error("Clause embedding batch failed at %d: %s", i, e)
            continue
        done += store_embeddings(wiki_id, session_id, chunk, vecs)
        if progress:
            progress(done, total)
    return {"embedded": done, "pending_before": total}


def coverage(wiki_id: str, session_id: str) -> dict:
    text = _text()
    tbl = db._clause_table_name()
    with db.get_engine().connect() as c:
        embedded = c.execute(text(
            f"SELECT count(*) FROM {tbl} WHERE wiki_id=:w AND session_id=:s"),
            {"w": wiki_id, "s": session_id}).scalar() or 0
        eligible = c.execute(text("""
            SELECT count(*) FROM clauses cl
            JOIN documents d ON d.wiki_id=cl.wiki_id AND d.source_doc=cl.source_doc
            WHERE cl.wiki_id=:w AND cl.session_id=:s AND d.role='precedent'
              AND cl.review_status <> 'rejected' AND length(cl.verbatim_text) > 40
        """), {"w": wiki_id, "s": session_id}).scalar() or 0
    return {"embedded": int(embedded), "eligible": int(eligible),
            "pending": max(0, int(eligible) - int(embedded))}


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

def search_clauses(wiki_id: str, session_id: str, query: str, limit: int = 12,
                   clause_type: str | None = None,
                   roles: tuple[str, ...] = ("precedent",),
                   exclude_docs: tuple[str, ...] = ()) -> list[dict]:
    """Rank precedent clauses against a drafting request.

    This is what replaces Draft Mode's page dump: it returns the specific
    clauses most similar to what is being drafted, already scoped to
    role-tagged precedent documents, so nothing has to be truncated to fit.
    """
    if not _enabled() or not (query or "").strip():
        return []
    from services import embedder
    text = _text()
    tbl = db._clause_table_name()
    try:
        vec = embedder.embed(query, is_query=True)
    except Exception as e:
        logger.error("Clause search embedding failed: %s", e)
        return []
    emb = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
    params = {"w": wiki_id, "s": session_id, "e": emb, "l": limit,
              "roles": list(roles)}
    type_clause = ""
    if clause_type:
        type_clause = "AND e.clause_type ILIKE :ct"
        params["ct"] = f"%{clause_type}%"
    excl = ""
    if exclude_docs:
        excl = "AND e.source_doc <> ALL(:ex)"
        params["ex"] = list(exclude_docs)
    with db.get_engine().connect() as c:
        rows = c.execute(text(f"""
            SELECT e.clause_id, e.source_doc, e.clause_type, e.doc_family,
                   cl.verbatim_text, cl.page_num,
                   1 - (e.embedding <=> CAST(:e AS vector)) AS score
            FROM {tbl} e
            JOIN clauses cl ON cl.id = e.clause_id
            WHERE e.wiki_id = :w AND e.session_id = :s
              AND e.role = ANY(:roles) {type_clause} {excl}
            ORDER BY e.embedding <=> CAST(:e AS vector)
            LIMIT :l
        """), params).fetchall()
    return [{"clause_id": int(r[0]), "source_doc": r[1], "clause_type": r[2],
             "doc_family": r[3], "text": r[4], "page_num": r[5],
             "score": round(float(r[6]), 4)} for r in rows]
