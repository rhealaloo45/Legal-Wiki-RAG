"""Recover the folder tree an upload was flattened from, and render a document
name the way a person filed it.

Upload flattens a relative path into the stored filename by replacing the
separator with "_" ("NDA/foo.pdf" -> "NDA_foo.pdf"), which is lossy: "_" is
also a legitimate filename character, so the boundary cannot be recovered from
any one name in isolation. It CAN be recovered from the corpus as a whole,
because a folder name is precisely the prefix many different files share.

Kept separate from ``wiki._norm_doc_name`` on purpose. That function is a
MATCHING key — lowercased, with qualifiers like "final"/"draft"/"v2" stripped
— and scope resolution compares against it throughout, so it must not become
a display string. This module is display-only.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import Counter

logger = logging.getLogger(__name__)

# Minimum files sharing a prefix before it is believed to be a folder rather
# than a filename that happens to start the same way.
FOLDER_MIN_FILES = 3

_SESSION_PREFIX_RE = re.compile(r'^[a-f0-9-]{36}_', re.IGNORECASE)

# (wiki_id, session_id) -> (document_count, {source_doc: "Folder/leaf.pdf"}).
# Keyed on the count as well so an ingest that adds documents rebuilds the map
# instead of leaving the new ones unresolvable.
_cache: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}
_cache_lock = threading.Lock()


def derive_upload_folders(names: list[str]) -> tuple[str, set[str]]:
    """Return (top_folder, {category folders}) derived from the names alone.

    Two passes, both driven by the data rather than by a list of folder names
    this deployment happens to know about:

    1. The TOP folder is the most widely shared first segment, extended one
       segment at a time for as long as extending it does not lose any files.
       This is what recovers a top folder whose own name contains underscores
       ("pdfs_by_category_generated"): every extension of "pdfs" still covers
       all 484 of its files, so extension continues, and stops only at the
       first segment that splits them (the category).
    2. The CATEGORY vocabulary is then read off directly — it is the set of
       segments sitting immediately after that top folder, which is exact
       rather than inferred. Applying that same vocabulary to the files that
       carry no top-folder prefix is what re-unites two upload batches of the
       same tree (one rooted at the tree, one rooted inside it) into a single
       correct hierarchy.

    Returns ("", set()) when nothing is shared widely enough to be a folder,
    which is the correct answer for a genuinely flat upload.
    """
    counts: Counter[str] = Counter()
    for n in names:
        segs = n.split("_")
        for i in range(1, len(segs)):
            counts["_".join(segs[:i])] += 1
    if not counts:
        return "", set()

    top = max(counts, key=lambda k: (counts[k], -k.count("_")))
    if counts[top] < FOLDER_MIN_FILES:
        return "", set()
    while True:
        depth = top.count("_") + 1
        ext = [k for k in counts
               if k.startswith(top + "_") and k.count("_") == depth
               and counts[k] == counts[top]]
        if len(ext) != 1:
            break
        top = ext[0]

    cats = {n[len(top) + 1:].split("_")[0]
            for n in names if n.startswith(top + "_")}
    return top, {c for c in cats if c}


def split_flat_name(name: str, top: str, cats: set[str]) -> tuple[str, str]:
    """Split one flattened name into (folder, filename).

    Longest category wins, so "Service Level Agreement" is not mistaken for
    "Service Agreement". Returns ("", name) for a file that sat at the top of
    the uploaded tree.
    """
    rest = name[len(top) + 1:] if top and name.startswith(top + "_") else name
    hit = ""
    for c in cats:
        if rest.startswith(c + "_") and len(c) > len(hit):
            hit = c
    return (hit, rest[len(hit) + 1:]) if hit else ("", rest)


def _strip_key(source_doc: str) -> str:
    """The stored key reduced to the flattened relative name."""
    s = (source_doc or "").replace("\\", "/").rsplit("/", 1)[-1]
    return _SESSION_PREFIX_RE.sub("", s)


def _build_map(wiki_id: str, session_id: str) -> dict[str, str]:
    from services import db as _db
    from sqlalchemy import text
    with _db.get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT source_doc FROM pages "
            "WHERE wiki_id = :w AND session_id = :s AND source_doc IS NOT NULL"
        ), {"w": wiki_id, "s": session_id})
        docs = [r[0] for r in rows]
    flat = {d: _strip_key(d) for d in docs}
    top, cats = derive_upload_folders(list(flat.values()))
    out: dict[str, str] = {}
    for doc, rel in flat.items():
        folder, leaf = split_flat_name(rel, top, cats)
        out[doc] = f"{folder}/{leaf}" if folder else leaf
    return out


def folder_map(wiki_id: str, session_id: str) -> dict[str, str]:
    """{source_doc: "Folder/leaf.pdf"} for every indexed document, cached."""
    key = (wiki_id or "", session_id or "")
    try:
        from services import db as _db
        from sqlalchemy import text
        with _db.get_engine().connect() as conn:
            n = conn.execute(text(
                "SELECT count(DISTINCT source_doc) FROM pages "
                "WHERE wiki_id = :w AND session_id = :s"
            ), {"w": wiki_id, "s": session_id}).scalar() or 0
    except Exception as e:
        logger.warning("doc_paths: could not count documents: %s", e)
        return {}
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] == n:
            return hit[1]
    try:
        built = _build_map(wiki_id, session_id)
    except Exception as e:
        logger.warning("doc_paths: could not build folder map: %s", e)
        return {}
    with _cache_lock:
        _cache[key] = (n, built)
    return built


def display(source_doc: str, wiki_id: str = "", session_id: str = "",
            with_folder: bool = True) -> str:
    """Render one stored document key the way a person filed it.

    "<uuid>_pdfs_by_category_generated_NDA_APEX ZEPHYRA_NDA_20-02-2024.pdf"
    becomes "NDA / APEX ZEPHYRA_NDA_20-02-2024.pdf".

    Degrades to the bare filename whenever the folder cannot be established —
    no wiki/session given, an unindexed document, or a flat upload. Never
    raises: a display helper failing must not take an answer down with it.
    """
    if not source_doc:
        return ""
    rel = ""
    if with_folder and wiki_id and session_id:
        try:
            rel = folder_map(wiki_id, session_id).get(source_doc, "")
        except Exception:
            rel = ""
    if not rel:
        rel = _strip_key(source_doc)
    folder, _, leaf = rel.rpartition("/")
    return f"{folder} / {leaf}" if folder else leaf


def display_many(docs, wiki_id: str = "", session_id: str = "") -> list[str]:
    """``display`` over a list, resolving the folder map once."""
    return [display(d, wiki_id, session_id) for d in (docs or [])]


def invalidate(wiki_id: str = "", session_id: str = "") -> None:
    """Drop cached maps — call after an ingest or a document deletion."""
    with _cache_lock:
        if wiki_id and session_id:
            _cache.pop((wiki_id, session_id), None)
        else:
            _cache.clear()
