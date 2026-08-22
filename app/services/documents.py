"""
Admin document lifecycle — target architecture § 01.4.

Three actions: Add, Archive, Hard-delete. Add is the existing per-document
ingest pipeline (app.py's /upload route + services/wiki.py) — nothing here
duplicates it, this module only covers the other two.

Both Archive and Hard-delete key off `source_doc`, the flattened
"<session_id>_<relative/path/with/slashes/as/underscores>" string that
already identifies a document across pages, clause_map, and source_positions
— and, not by coincidence, is also that document's literal filename inside
config.UPLOAD_PATH. flatten_doc_key() below is the single place that string
gets built, shared with app.py's /files route so the two can never drift
apart into disagreeing about what a given upload's key is.

KNOWN LIMITATION — disclosed here once rather than re-explained at every call
site: pages.source_doc is a single column. A shared/merged concept page (a
statute, a clause type several documents reference) can only record ONE
contributing document — see wiki.py's merge-guard comment near "silently
overwrites source_doc". Archiving or deleting a document therefore reliably
removes everything exclusively attributable to it, but cannot retroactively
strip its facts back out of a page that later merged with a different
document's ingest and now attributes to that document instead. Closing this
gap for real needs the target architecture's per-document structured tables
(clauses/obligations, Phase 0 backbone) — not built yet. This is exactly why
Archive is the doc-specified default over Hard-delete: it's fully reversible,
where a merge is not.
"""

import logging
import os

import config
from services import db

logger = logging.getLogger(__name__)


def flatten_doc_key(session_id: str, relative_path: str) -> str:
    """The on-disk filename AND the source_doc DB key for one upload.

    Matches app.py's /files reconstruction exactly — kept as one function so
    the two can't silently diverge.
    """
    return f"{session_id}_" + relative_path.replace("\\", "/").replace("/", "_")


def _assert_ownership(session_id: str, source_doc: str) -> None:
    """Refuse to act unless source_doc is actually this session's own key.

    source_doc embeds its owning session_id as a literal prefix (see
    flatten_doc_key), which means hard_delete_document's file-removal path
    below can resolve to a real file on disk from session_id and source_doc
    ALONE — it doesn't need the two to agree to find something to delete.
    Without this check, a caller passing a session_id that doesn't match
    source_doc's real owner (stale client state, a redirect elsewhere in the
    app resolving session_id differently than the value a source_doc was
    fetched under, or simple caller error) would have every DB delete
    silently no-op on the session_id mismatch while the file removal
    succeeded anyway — wrong file gone, no error raised. Caught exactly this
    way in testing before it shipped; see the git history on this file.
    """
    prefix = f"{session_id}_"
    if not source_doc.startswith(prefix):
        raise ValueError(
            f"source_doc {source_doc!r} does not belong to session_id {session_id!r} "
            f"(expected it to start with {prefix!r}) — refusing to act on it"
        )


def archive_document(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """Hide a document from search/chat/registers without deleting anything.

    Reversible by design (see module docstring on why this is the default
    over hard-delete) — the uploaded file and every DB row are left exactly
    as they were, only a status row is written.
    """
    _assert_ownership(session_id, source_doc)
    db.archive_document(wiki_id, session_id, source_doc)
    logger.info("Archived document %r in session %r", source_doc, session_id)
    return {"status": "archived", "source_doc": source_doc}


def unarchive_document(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """Restore an archived document to active."""
    _assert_ownership(session_id, source_doc)
    db.unarchive_document(wiki_id, session_id, source_doc)
    logger.info("Unarchived document %r in session %r", source_doc, session_id)
    return {"status": "active", "source_doc": source_doc}


def hard_delete_document(wiki_id: str, session_id: str, source_doc: str, sessions: dict) -> dict:
    """Permanently remove a document: DB rows, the uploaded file, and its
    sessions.json file_paths entry.

    `sessions` is the already-loaded sessions dict (app.py's load_sessions())
    — mutated in place with the file_paths entry removed; the caller is
    responsible for save_sessions() afterward. Not done inside this function
    so a caller batching multiple deletes writes the file once, not per-call.

    Returns a report — see db.delete_document_data for exactly what's
    covered and the merged-page limitation that isn't.
    """
    _assert_ownership(session_id, source_doc)
    report = db.delete_document_data(wiki_id, session_id, source_doc)

    file_path = os.path.join(config.UPLOAD_PATH, source_doc)
    file_removed = False
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            file_removed = True
        except Exception as e:
            logger.error("Failed to remove uploaded file %r: %s", file_path, e)
    report["file_removed"] = file_removed

    session_entry = sessions.get(session_id)
    if session_entry and "file_paths" in session_entry:
        before = len(session_entry["file_paths"])
        session_entry["file_paths"] = [
            p for p in session_entry["file_paths"]
            if flatten_doc_key(session_id, p) != source_doc
        ]
        report["session_file_paths_removed"] = before - len(session_entry["file_paths"])
    else:
        report["session_file_paths_removed"] = 0

    logger.info("Hard-deleted document %r in session %r: %s", source_doc, session_id, report)
    report["source_doc"] = source_doc
    return report
