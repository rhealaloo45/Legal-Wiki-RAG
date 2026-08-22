"""
Wikis — isolated corpora (target architecture § 01.5).

A wiki is a container, nothing more: id, name, status. What makes it an
isolation boundary rather than a label is that `wiki_id` is a mandatory
predicate inside every wiki-scoped query, not a filter applied to results
after they come back. See `active_wiki_id()` for how a request gets its
one bound value.

Scope of isolation as it actually stands right now — stated plainly because
the doc's warning ("isolation has to be real, not a display filter") is only
meaningful if the gap is named:

  * The Phase 0 typed tables (documents, contracts, obligations,
    litigation_facts, authorizations, opinions, citations,
    structural_anchors, entities, entity_aliases, tables, figures) are
    keyed by wiki_id and every read in services/backbone.py carries it.
  * The legacy tables (pages, relations, page_metadata, clauses,
    contradictions, clause_map, document_status, source_positions,
    question_embeddings_*, page_embeddings_*) carry wiki_id, it is stamped
    on every write, and every read/update/delete in db.py now predicates on
    it too — not just session_id. Switching the active wiki (services/
    wikis.py's set_active_wiki) now actually changes what every wiki-scoped
    view sees, including a brand-new empty wiki with no session of its own
    yet — it no longer falls through to whatever session happened to be
    loaded before the switch.
"""
from __future__ import annotations

import logging
import uuid

import config
from services import db

logger = logging.getLogger(__name__)

DEFAULT_WIKI_ID = db.DEFAULT_WIKI_ID
_ACTIVE_KEY = "active_wiki_id"


def _sql():
    from sqlalchemy import text
    return text


def active_wiki_id() -> str:
    """The system-level active-wiki pointer (§ Wikis — switch-based).

    Falls back to the default wiki rather than raising: a missing pointer row
    means a fresh or partially-migrated DB, and failing every request over it
    would be worse than resolving to the corpus that was already there.
    """
    if not config.USE_DATABASE:
        return DEFAULT_WIKI_ID
    text = _sql()
    try:
        with db.get_engine().connect() as conn:
            row = conn.execute(
                text("SELECT value FROM app_settings WHERE key = :k"),
                {"k": _ACTIVE_KEY},
            ).fetchone()
        if row and row[0]:
            return row[0]
    except Exception as err:
        logger.warning("Could not read active wiki pointer, using default: %s", err)
    return DEFAULT_WIKI_ID


def set_active_wiki(wiki_id: str) -> None:
    """Point the system at a different wiki. Refuses an unknown or archived
    id — switching to a wiki that doesn't exist would silently strand every
    subsequent read on an empty corpus."""
    text = _sql()
    with db.get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT status FROM wikis WHERE id = :id"), {"id": wiki_id}
        ).fetchone()
        if not row:
            raise ValueError(f"No such wiki: {wiki_id}")
        if row[0] != "active":
            raise ValueError(f"Wiki {wiki_id} is {row[0]}, not active")
        conn.execute(text("""
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (:k, :v, now())
            ON CONFLICT (key) DO UPDATE SET value = :v, updated_at = now()
        """), {"k": _ACTIVE_KEY, "v": wiki_id})
        conn.commit()
    logger.info("Active wiki switched to %s", wiki_id)


def list_wikis(include_archived: bool = True) -> list[dict]:
    text = _sql()
    active = active_wiki_id()
    sql = "SELECT id, name, status, created_at, created_by FROM wikis"
    if not include_archived:
        sql += " WHERE status = 'active'"
    sql += " ORDER BY created_at"
    with db.get_engine().connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
        counts = dict(conn.execute(text(
            "SELECT wiki_id, COUNT(*) FROM pages GROUP BY wiki_id"
        )).fetchall())
        doc_counts = dict(conn.execute(text(
            "SELECT wiki_id, COUNT(*) FROM documents GROUP BY wiki_id"
        )).fetchall())
    return [
        {
            "id": r[0],
            "name": r[1],
            "status": r[2],
            "created_at": r[3].isoformat() if r[3] else None,
            "created_by": r[4],
            "is_active": r[0] == active,
            "page_count": int(counts.get(r[0], 0)),
            "document_count": int(doc_counts.get(r[0], 0)),
        }
        for r in rows
    ]


def create_wiki(name: str, created_by: str | None = None) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Wiki name is required")
    wiki_id = str(uuid.uuid4())
    text = _sql()
    with db.get_engine().connect() as conn:
        conn.execute(text("""
            INSERT INTO wikis (id, name, status, created_by)
            VALUES (:id, :name, 'active', :by)
        """), {"id": wiki_id, "name": name, "by": created_by})
        conn.commit()
    logger.info("Created wiki %s (%s)", wiki_id, name)
    return wiki_id


def archive_wiki(wiki_id: str) -> None:
    """Archive a wiki. The default wiki and the currently-active wiki can't be
    archived — the first because every pre-backbone row lives there, the second
    because it would leave the system pointing at a corpus it just closed."""
    if wiki_id == DEFAULT_WIKI_ID:
        raise ValueError("The default wiki cannot be archived")
    if wiki_id == active_wiki_id():
        raise ValueError("Switch to another wiki before archiving this one")
    text = _sql()
    with db.get_engine().connect() as conn:
        res = conn.execute(
            text("UPDATE wikis SET status = 'archived' WHERE id = :id"),
            {"id": wiki_id},
        )
        if res.rowcount == 0:
            raise ValueError(f"No such wiki: {wiki_id}")
        conn.commit()
