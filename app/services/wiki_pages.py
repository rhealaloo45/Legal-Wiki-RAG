"""
Wiki page admin mutations — target architecture § Admin & Wiki Management.

Three actions on a single wiki page, independent of the whole-document
lifecycle in services/documents.py: Rename, Merge, Delete. None of these
existed anywhere in the codebase before this module — see services/db.py's
rename_page/delete_page/merge_pages for why a page's title can't just be
UPDATEd in the `pages` table alone (relations/clause_map/contradictions and
every page_embeddings* table all reference it by title text, no FKs, so a
rename/merge/delete has to walk all of them in one transaction).

This module is the thin validation/orchestration layer over those db.py
functions — mirrors services/documents.py's shape and division of labour
(db.py owns the SQL, this owns input validation and turning a bool result
into a raised ValueError the route can turn into a 400/404).
"""

import logging

from services import db

logger = logging.getLogger(__name__)


def rename_page(wiki_id: str, session_id: str, old_title: str, new_title: str) -> dict:
    old_title = (old_title or "").strip()
    new_title = (new_title or "").strip()
    if not old_title or not new_title:
        raise ValueError("old_title and new_title are both required")
    if old_title == new_title:
        raise ValueError("new_title is the same as old_title")
    ok = db.rename_page(wiki_id, session_id, old_title, new_title)
    if not ok:
        raise ValueError(
            f"Rename failed — either {old_title!r} doesn't exist, "
            f"or {new_title!r} is already taken in this session"
        )
    logger.info("Renamed page %r -> %r in session %r", old_title, new_title, session_id)
    return {"status": "renamed", "old_title": old_title, "new_title": new_title}


def delete_page(wiki_id: str, session_id: str, title: str) -> dict:
    title = (title or "").strip()
    if not title:
        raise ValueError("title is required")
    ok = db.delete_page(wiki_id, session_id, title)
    if not ok:
        raise ValueError(f"Page {title!r} does not exist in this session")
    logger.info("Deleted page %r in session %r", title, session_id)
    return {"status": "deleted", "title": title}


def merge_pages(wiki_id: str, session_id: str, source_title: str, target_title: str) -> dict:
    source_title = (source_title or "").strip()
    target_title = (target_title or "").strip()
    if not source_title or not target_title:
        raise ValueError("source_title and target_title are both required")
    if source_title == target_title:
        raise ValueError("source_title and target_title must be different pages")
    ok = db.merge_pages(wiki_id, session_id, source_title, target_title)
    if not ok:
        raise ValueError(
            f"Merge failed — one of {source_title!r} / {target_title!r} "
            f"does not exist in this session"
        )
    logger.info("Merged page %r into %r in session %r", source_title, target_title, session_id)
    return {"status": "merged", "source_title": source_title, "target_title": target_title}
