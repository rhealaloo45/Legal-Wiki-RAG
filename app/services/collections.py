"""Collections — a named, wiki-scoped set of documents.

Shared plumbing under Playbooks and the Deviation Dashboard (§ Phase 2). Both
need "run this over these documents", and neither should carry its own idea of
what a document set is.

Membership is stored explicitly, never as a saved filter. A filter re-evaluates
on every read, so a playbook run and the dashboard row recording it could cover
different documents as the corpus grows — and a recorded run that cannot be
reproduced is not evidence of anything. Filters populate the list (see
`add_by_filter`); they are not the list.
"""
from __future__ import annotations

import logging
from typing import Any

from services import db

logger = logging.getLogger(__name__)


def _text():
    from sqlalchemy import text
    return text


def _enabled() -> bool:
    import config
    return bool(config.USE_DATABASE)


class CollectionError(Exception):
    """Raised for conflicts a caller should report rather than swallow."""


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create(wiki_id: str, session_id: str, name: str,
           description: str | None = None) -> dict:
    """Create an empty collection. Name must be unique within the wiki."""
    name = (name or "").strip()
    if not name:
        raise CollectionError("Collection name cannot be empty")
    if not _enabled():
        raise CollectionError("Database not configured")
    text = _text()
    with db.get_engine().connect() as c:
        exists = c.execute(text(
            "SELECT id FROM collections WHERE wiki_id = :w AND name = :n"),
            {"w": wiki_id, "n": name}).scalar()
        if exists:
            raise CollectionError(f"A collection named {name!r} already exists in this wiki")
        row = c.execute(text("""
            INSERT INTO collections (wiki_id, session_id, name, description)
            VALUES (:w, :s, :n, :d) RETURNING id, created_at
        """), {"w": wiki_id, "s": session_id, "n": name,
               "d": (description or "").strip() or None}).fetchone()
        c.commit()
    return {"id": int(row[0]), "name": name, "description": description,
            "document_count": 0, "created_at": row[1].isoformat() if row[1] else None}


def list_all(wiki_id: str) -> list[dict]:
    """Every collection in this wiki, with its document count."""
    if not _enabled():
        return []
    text = _text()
    with db.get_engine().connect() as c:
        rows = c.execute(text("""
            SELECT c.id, c.name, c.description, c.created_at, c.updated_at,
                   count(d.id) AS n
            FROM collections c
            LEFT JOIN collection_documents d ON d.collection_id = c.id
            WHERE c.wiki_id = :w
            GROUP BY c.id ORDER BY c.name
        """), {"w": wiki_id}).fetchall()
    return [{"id": int(r[0]), "name": r[1], "description": r[2],
             "created_at": r[3].isoformat() if r[3] else None,
             "updated_at": r[4].isoformat() if r[4] else None,
             "document_count": int(r[5])} for r in rows]


def get(wiki_id: str, collection_id: int, with_documents: bool = True) -> dict | None:
    """One collection and, by default, its member documents."""
    if not _enabled():
        return None
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT id, name, description, session_id, created_at, updated_at
            FROM collections WHERE wiki_id = :w AND id = :i
        """), {"w": wiki_id, "i": collection_id}).fetchone()
        if not row:
            return None
        out = {"id": int(row[0]), "name": row[1], "description": row[2],
               "session_id": row[3],
               "created_at": row[4].isoformat() if row[4] else None,
               "updated_at": row[5].isoformat() if row[5] else None}
        if with_documents:
            docs = c.execute(text("""
                SELECT cd.source_doc, d.doc_family, d.doc_type, d.effective_date
                FROM collection_documents cd
                LEFT JOIN documents d
                       ON d.wiki_id = cd.wiki_id AND d.source_doc = cd.source_doc
                WHERE cd.collection_id = :i
                ORDER BY cd.source_doc
            """), {"i": collection_id}).fetchall()
            out["documents"] = [{"source_doc": r[0], "doc_family": r[1],
                                 "doc_type": r[2], "effective_date": r[3]}
                                for r in docs]
        out["document_count"] = len(out.get("documents", [])) if with_documents else None
    return out


def rename(wiki_id: str, collection_id: int, name: str | None = None,
           description: str | None = None) -> bool:
    if not _enabled():
        return False
    sets, params = [], {"w": wiki_id, "i": collection_id}
    if name is not None:
        n = name.strip()
        if not n:
            raise CollectionError("Collection name cannot be empty")
        sets.append("name = :n")
        params["n"] = n
    if description is not None:
        sets.append("description = :d")
        params["d"] = description.strip() or None
    if not sets:
        return False
    sets.append("updated_at = now()")
    text = _text()
    with db.get_engine().connect() as c:
        if name is not None:
            clash = c.execute(text("""
                SELECT id FROM collections
                WHERE wiki_id = :w AND name = :n AND id <> :i
            """), params).scalar()
            if clash:
                raise CollectionError(f"A collection named {name!r} already exists")
        res = c.execute(text(
            f"UPDATE collections SET {', '.join(sets)} WHERE wiki_id = :w AND id = :i"),
            params)
        c.commit()
    return (res.rowcount or 0) > 0


def delete(wiki_id: str, collection_id: int) -> bool:
    """Delete a collection. Membership rows cascade; DOCUMENTS ARE NOT TOUCHED —
    a collection is a label over the corpus, not a container of it."""
    if not _enabled():
        return False
    text = _text()
    with db.get_engine().connect() as c:
        res = c.execute(text("DELETE FROM collections WHERE wiki_id = :w AND id = :i"),
                        {"w": wiki_id, "i": collection_id})
        c.commit()
    return (res.rowcount or 0) > 0


# ---------------------------------------------------------------------------
# membership
# ---------------------------------------------------------------------------

def add_documents(wiki_id: str, session_id: str, collection_id: int,
                  source_docs: list[str]) -> dict:
    """Add documents. Only documents that exist in this wiki are accepted.

    Silently skipping unknown names would let a typo produce a collection that
    looks populated and runs over nothing, so they are counted and returned.
    """
    if not _enabled() or not source_docs:
        return {"added": 0, "already_present": 0, "unknown": []}
    text = _text()
    with db.get_engine().connect() as c:
        known = {r[0] for r in c.execute(text("""
            SELECT source_doc FROM documents
            WHERE wiki_id = :w AND source_doc = ANY(:d)
        """), {"w": wiki_id, "d": list(source_docs)})}
        unknown = [d for d in source_docs if d not in known]
        added = 0
        already = 0
        for d in sorted(known):
            res = c.execute(text("""
                INSERT INTO collection_documents
                    (collection_id, wiki_id, session_id, source_doc)
                VALUES (:i, :w, :s, :d)
                ON CONFLICT (collection_id, source_doc) DO NOTHING
            """), {"i": collection_id, "w": wiki_id, "s": session_id, "d": d})
            if res.rowcount:
                added += 1
            else:
                already += 1
        c.execute(text("UPDATE collections SET updated_at = now() WHERE id = :i"),
                  {"i": collection_id})
        c.commit()
    return {"added": added, "already_present": already, "unknown": unknown}


def remove_documents(wiki_id: str, collection_id: int,
                     source_docs: list[str]) -> int:
    if not _enabled() or not source_docs:
        return 0
    text = _text()
    with db.get_engine().connect() as c:
        res = c.execute(text("""
            DELETE FROM collection_documents
            WHERE collection_id = :i AND wiki_id = :w AND source_doc = ANY(:d)
        """), {"i": collection_id, "w": wiki_id, "d": list(source_docs)})
        c.execute(text("UPDATE collections SET updated_at = now() WHERE id = :i"),
                  {"i": collection_id})
        c.commit()
    return res.rowcount or 0


def add_by_filter(wiki_id: str, session_id: str, collection_id: int,
                  doc_family: str | None = None, doc_type: str | None = None,
                  name_contains: str | None = None) -> dict:
    """Populate from a filter — evaluated ONCE, at call time.

    The resulting membership is a fixed list, deliberately: a collection that
    re-evaluated its filter on every read would change under a playbook run,
    and the run record would no longer describe what was actually processed.
    """
    if not _enabled():
        return {"added": 0, "already_present": 0, "unknown": [], "matched": 0}
    text = _text()
    clauses, params = ["wiki_id = :w"], {"w": wiki_id}
    if doc_family:
        clauses.append("doc_family = :f")
        params["f"] = doc_family
    if doc_type:
        clauses.append("doc_type = :t")
        params["t"] = doc_type
    if name_contains:
        clauses.append("source_doc ILIKE :n")
        params["n"] = f"%{name_contains}%"
    with db.get_engine().connect() as c:
        docs = [r[0] for r in c.execute(text(
            f"SELECT source_doc FROM documents WHERE {' AND '.join(clauses)}"), params)]
    result = add_documents(wiki_id, session_id, collection_id, docs)
    result["matched"] = len(docs)
    return result


def documents_in(wiki_id: str, collection_id: int) -> list[str]:
    """Member document names — what a playbook run iterates."""
    if not _enabled():
        return []
    text = _text()
    with db.get_engine().connect() as c:
        return [r[0] for r in c.execute(text("""
            SELECT source_doc FROM collection_documents
            WHERE collection_id = :i AND wiki_id = :w ORDER BY source_doc
        """), {"i": collection_id, "w": wiki_id})]


def resolve(wiki_id: str, ref: Any) -> int | None:
    """Accept a collection id or a name and return its id."""
    if ref is None:
        return None
    text = _text()
    with db.get_engine().connect() as c:
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            return c.execute(text(
                "SELECT id FROM collections WHERE wiki_id = :w AND id = :i"),
                {"w": wiki_id, "i": int(ref)}).scalar()
        return c.execute(text(
            "SELECT id FROM collections WHERE wiki_id = :w AND name = :n"),
            {"w": wiki_id, "n": str(ref).strip()}).scalar()
