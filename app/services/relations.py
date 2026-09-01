"""
Named, bounded-hop relation traversal (target architecture § Phase 4).

The roadmap describes this as replacing open joins on the relation fast path
with typed operations. Enumerating the existing consumers first — as this
document's own risk note for the item required — found that nothing currently
traverses more than a single hop: `db.get_document_relations` reads one hop in
both directions, `doc_references.amendment_partners` reads one hop for
contradiction detection, and that is the whole set. So there is no open-ended
traversal here to replace, and nothing to break by capping one.

This module is therefore ADDITIVE: it provides the multi-hop operations that
did not exist, each with its depth fixed at the call site rather than chosen
per question. That is the governance the roadmap actually wanted — a finite,
named set of traversals whose worst-case cost is knowable — without the
regression risk of narrowing a query surface something already depends on.

Depth budget, as specified:
    0 hops  a direct lookup (the document itself)
    1 hop   a basic relationship (what this amends, what amends this)
    2 hops  the common multi-hop cases — an amendment chain, a transaction
            cluster where A amends B which supersedes C
    3 hops  hard ceiling, never exceeded regardless of what a caller passes

Every result reports the hop distance of each document it found and whether
the traversal stopped because it ran out of edges or because it hit the cap.
A caller that cannot tell those apart cannot tell a complete answer from a
truncated one.
"""

import logging
from collections import deque

logger = logging.getLogger(__name__)

MAX_HOPS_CEILING = 3

# Edge labels that constitute a version chain, as opposed to a mere citation.
# Mirrors doc_references._AMENDMENT_LABELS deliberately rather than importing
# it, so a change there is a visible change here rather than a silent one.
CHAIN_LABELS = ("amends", "amended-by", "supersedes", "superseded-by",
                "restates", "novates")


def _enabled() -> bool:
    import config
    return bool(getattr(config, "USE_DATABASE", False))


def _neighbours(conn, text, wiki_id: str, session_id: str, doc: str,
                labels: tuple[str, ...] | None) -> list[tuple[str, str]]:
    """One hop out from `doc`, in both directions. Returns (doc, label) pairs.

    Both directions matter and only one of them is visible from a document's
    own text: "what does this amend" is written into this document, while
    "what amends this" exists only as a row created when the OTHER document
    was ingested.
    """
    params = {"w": wiki_id, "sid": session_id, "d": doc}
    label_sql = ""
    if labels:
        params["labels"] = list(labels)
        label_sql = " AND label = ANY(:labels)"
    rows = conn.execute(text(f"""
        SELECT to_doc AS other, label FROM document_relations
         WHERE wiki_id = :w AND session_id = :sid AND from_doc = :d
           AND resolved AND to_doc IS NOT NULL{label_sql}
        UNION
        SELECT from_doc AS other, label FROM document_relations
         WHERE wiki_id = :w AND session_id = :sid AND to_doc = :d
           AND resolved AND from_doc IS NOT NULL{label_sql}
    """), params).fetchall()
    return [(r[0], r[1]) for r in rows if r[0]]


def traverse(wiki_id: str, session_id: str, source_doc: str,
             max_hops: int = 2, labels: tuple[str, ...] | None = None,
             max_docs: int = 40) -> dict:
    """Breadth-first traversal from one document, hard-capped at 3 hops.

    Breadth-first rather than depth-first on purpose: with a hop cap, BFS
    guarantees every document is reported at its true shortest distance, where
    DFS can reach a document by a long path first and record it as further away
    than it is.
    """
    if not _enabled():
        return {"error": "database not configured"}
    hops = max(0, min(int(max_hops), MAX_HOPS_CEILING))
    capped_by_ceiling = int(max_hops) > MAX_HOPS_CEILING

    from sqlalchemy import text
    from services import db

    seen: dict[str, int] = {source_doc: 0}
    edges: list[dict] = []
    frontier_truncated = False
    hit_hop_cap = False

    with db.get_engine().connect() as conn:
        queue = deque([(source_doc, 0)])
        while queue:
            doc, dist = queue.popleft()
            if dist >= hops:
                # There may be more graph beyond here; say so rather than
                # letting the caller read an exhausted frontier as a complete
                # picture.
                if _neighbours(conn, text, wiki_id, session_id, doc, labels):
                    hit_hop_cap = True
                continue
            for other, label in _neighbours(conn, text, wiki_id, session_id, doc, labels):
                edges.append({"from": doc, "to": other, "label": label,
                              "hop": dist + 1})
                if other in seen:
                    continue
                if len(seen) >= max_docs:
                    frontier_truncated = True
                    continue
                seen[other] = dist + 1
                queue.append((other, dist + 1))

    from services import wiki as _wiki
    documents = [{"source_doc": d, "name": _wiki._norm_doc_name(d), "hops": h}
                 for d, h in sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))
                 if d != source_doc]
    return {
        "anchor": source_doc,
        "anchor_name": _wiki._norm_doc_name(source_doc),
        "max_hops": hops,
        "hop_ceiling_applied": capped_by_ceiling,
        "documents": documents,
        "edges": edges,
        "complete": not (hit_hop_cap or frontier_truncated),
        "stopped_at_hop_cap": hit_hop_cap,
        "stopped_at_doc_cap": frontier_truncated,
        "note": ("Complete — the traversal ran out of edges before reaching its "
                 f"{hops}-hop limit."
                 if not (hit_hop_cap or frontier_truncated)
                 else f"Truncated at the {hops}-hop limit; documents further than "
                      f"{hops} hop(s) from the anchor are not shown."),
    }


# ---------------------------------------------------------------------------
# The named operations. Each fixes its own depth; none takes it from a question.
# ---------------------------------------------------------------------------

def find_amendment_chain(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """The full version chain around a document — 2 hops.

    Two rather than one because a chain is commonly original → amendment →
    second amendment, and answering "what is the current state of this
    agreement" from one hop would show the first amendment while silently
    omitting the one that superseded it.
    """
    out = traverse(wiki_id, session_id, source_doc, max_hops=2, labels=CHAIN_LABELS)
    out["operation"] = "find_amendment_chain"
    return out


def find_related_documents(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """Everything one hop away by any edge — the basic "what is connected" call."""
    out = traverse(wiki_id, session_id, source_doc, max_hops=1, labels=None)
    out["operation"] = "find_related_documents"
    return out


def find_transaction_cluster(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """The deal a document belongs to — 2 hops over all edge types.

    An MSA, its SOW, and the DPA the SOW references are one transaction; the
    third document is two hops from the first and invisible at one.
    """
    out = traverse(wiki_id, session_id, source_doc, max_hops=2, labels=None)
    out["operation"] = "find_transaction_cluster"
    return out


def find_agreements_by_party(wiki_id: str, session_id: str, parties: list[str],
                             doc_type: str | None = None, limit: int = 50) -> dict:
    """Zero-hop: documents naming these parties, straight from documents.parties.

    No graph traversal at all — included in this module because it is the
    0-hop member of the same named set, and because routing it here keeps
    "find me the agreements with X" from being answered by a graph walk that
    would only find documents that happen to cite something.
    """
    if not _enabled():
        return {"error": "database not configured"}
    from services import db
    result = db.count_documents_by_party(wiki_id, session_id, parties, doc_type, limit)
    result["operation"] = "find_agreements_by_party"
    result["hops"] = 0
    return result
