"""
Answer handoffs (target architecture § Phase 3.5b).

When a question is better served in depth by a surface that already exists,
the answer should say so and name it. Four features — the Deviation Dashboard,
the Obligation tracker, Collections and Playbooks — are shipped and effectively
invisible: nothing in a conversational answer ever mentions them, so a lawyer
finds them only by opening the admin panel and reading the nav.

This is deliberately the cheapest thing in the phase: a routing table plus one
rendered line. It adds no extraction, no model call, and never edits the
answer text — a handoff is attached as its own field so it cannot be mistaken
for something the model asserted about the documents.

Two rules the suggestions follow:

  Only suggest a surface that has something to show. A handoff to an empty
  Deviation Dashboard teaches the user that these links waste their time, and
  they will stop reading them. Every rule below checks for real rows first.

  Suggest at most one. A list of places to go instead of an answer is not
  helpful, and ranking them honestly is not possible from here.
"""

import logging
import re

import config

logger = logging.getLogger(__name__)

# Target ids match templates/index.html's switchAdminTab() — a handoff naming
# a surface that does not exist is worse than none at all.
_ADMIN = "admin"

_RX_OBLIGATION = re.compile(
    r"\b(?:obligation|deadline|due\s+date|notice\s+period|when\s+(?:is|does|must)"
    r"|renewal|expir(?:y|es|ation)|milestone|deliverable)\b", re.IGNORECASE)
_RX_DEVIATION = re.compile(
    r"\b(?:deviat|non[- ]?standard|off[- ]?market|unusual|risk|red\s+flag"
    r"|compliance|acceptable|our\s+standard|playbook)\b", re.IGNORECASE)


def _enabled() -> bool:
    return bool(getattr(config, "USE_DATABASE", False))


def suggest(question: str, payload: dict, session_id: str) -> dict | None:
    """The one surface worth naming for this answer, or None.

    Never raises: a handoff is a convenience, and a bug in it must not take
    down the answer it was going to be attached to.
    """
    if not _enabled() or not isinstance(payload, dict):
        return None
    try:
        return _suggest_inner(question or "", payload, session_id)
    except Exception as e:
        logger.error("Handoff suggestion failed: %s", e)
        return None


def _suggest_inner(question: str, payload: dict, session_id: str) -> dict | None:
    from services import db as _db, wikis as _wikis
    wiki_id = _wikis.active_wiki_id()
    docs = list(payload.get("files_used") or [])

    # 1. Deviation Dashboard — only when a playbook has actually assessed the
    #    documents this answer used, and found something to say about them.
    if _RX_DEVIATION.search(question) and docs:
        n = _count_findings(_db, wiki_id, docs)
        if n:
            return _mk("deviation",
                       f"{n} playbook finding(s) already exist for the document(s) "
                       f"used here. The Deviation Dashboard breaks them down by "
                       f"severity and document.",
                       "Deviation Dashboard")

    # 2. Obligation tracker — only when obligations exist for these documents.
    if _RX_OBLIGATION.search(question) and docs:
        n = _count_obligations(_db, wiki_id, session_id, docs)
        if n:
            return _mk("obligations",
                       f"{n} obligation(s) are tracked on the document(s) used here. "
                       f"The Obligation tracker lists them with their deadlines.",
                       "Obligation tracker")

    # 3. A broad answer over many documents is what Collections exist for —
    #    the user is working with a set, and a set is a first-class object here.
    if len(docs) >= 8:
        return _mk("collections",
                   f"This answer drew on {len(docs)} documents. Saving them as a "
                   f"Collection lets you run a playbook across the whole set at once.",
                   "Collections")

    # 4. An abstention is the most useful place to point somewhere else: the
    #    answer is "not in these documents", and naming where it COULD be
    #    established beats the refusal standing alone.
    if payload.get("not_covered"):
        return _mk("documents",
                   "Nothing in the documents in scope establishes this. If the "
                   "document you expected is missing, Admin → Documents shows "
                   "everything currently ingested.",
                   "Admin → Documents")
    return None


def _mk(target: str, message: str, label: str) -> dict:
    return {"target": target, "panel": _ADMIN, "label": label, "message": message}


def _count_findings(_db, wiki_id: str, docs: list[str]) -> int:
    from sqlalchemy import text
    with _db.get_engine().connect() as c:
        return int(c.execute(text("""
            SELECT count(*) FROM playbook_findings
            WHERE wiki_id = :w AND source_doc = ANY(:docs)
              AND verdict IN ('unacceptable', 'missing', 'fallback')
        """), {"w": wiki_id, "docs": docs}).scalar() or 0)


def _count_obligations(_db, wiki_id: str, session_id: str, docs: list[str]) -> int:
    from sqlalchemy import text
    with _db.get_engine().connect() as c:
        return int(c.execute(text("""
            SELECT count(*) FROM obligations
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = ANY(:docs)
        """), {"w": wiki_id, "sid": session_id, "docs": docs}).scalar() or 0)
