"""Deviation Dashboard — a SQL aggregation over playbook_findings.

No new tables, no LLM calls: everything here reads rows a playbook run already
wrote. The dashboard's whole job is turning "25 rows in a table" into "3
documents need attention, ranked, with why".

Two read shapes:

  overview(wiki_id)              one line per playbook — its latest complete
                                  run and that run's verdict counts. What you
                                  see landing on the dashboard.

  dashboard(wiki_id, playbook_id, run_id=None)
                                  one playbook's latest run (or an explicit
                                  past run), broken down three ways: by clause
                                  type, by document (worst-first), and a
                                  priority list of the findings that actually
                                  need a human.

A run's own `documents_covered` (frozen at run time) is compared against the
collection's CURRENT membership to find "stale" documents — ones added since
the run and never assessed. That comparison is why runs freeze their own
scope rather than pointing at a live collection: without a frozen list there
would be nothing to compare the present against.
"""
from __future__ import annotations

from services import db

# Worst-first ordering. `standard` sorts last on purpose — a document with
# only standard clauses needs no attention, so it belongs at the bottom of a
# list meant to surface what does.
_VERDICT_RANK = {"unacceptable": 0, "missing": 1, "fallback": 2, "unclear": 3,
                 "standard": 4}
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, None: 3}


def _text():
    from sqlalchemy import text
    return text


def _enabled() -> bool:
    import config
    return bool(config.USE_DATABASE)


def latest_complete_run(wiki_id: str, playbook_id: int) -> int | None:
    """Most recent finished run for a playbook. A run still `running` is
    deliberately excluded — its findings are partial, and aggregating a
    partial run would show a document as compliant simply because its clauses
    have not been reached yet."""
    text = _text()
    with db.get_engine().connect() as c:
        return c.execute(text("""
            SELECT id FROM playbook_runs
            WHERE wiki_id = :w AND playbook_id = :p AND status = 'complete'
            ORDER BY started_at DESC LIMIT 1
        """), {"w": wiki_id, "p": playbook_id}).scalar()


def overview(wiki_id: str) -> list[dict]:
    """One row per playbook that has at least one completed run."""
    if not _enabled():
        return []
    text = _text()
    with db.get_engine().connect() as c:
        playbooks = c.execute(text(
            "SELECT id, name FROM playbooks WHERE wiki_id = :w ORDER BY name"),
            {"w": wiki_id}).fetchall()
        out = []
        for pid, name in playbooks:
            run_id = latest_complete_run(wiki_id, pid)
            if not run_id:
                continue
            run = c.execute(text("""
                SELECT collection_name, documents_total, findings_total, finished_at
                FROM playbook_runs WHERE id = :r
            """), {"r": run_id}).fetchone()
            verdicts = dict(c.execute(text("""
                SELECT verdict, count(*) FROM playbook_findings
                WHERE run_id = :r GROUP BY verdict
            """), {"r": run_id}).fetchall())
            out.append({
                "playbook_id": pid, "playbook": name, "run_id": run_id,
                "collection": run[0], "documents_total": run[1],
                "findings_total": run[2],
                "finished_at": run[3].isoformat() if run[3] else None,
                "verdicts": {k: int(v) for k, v in verdicts.items()},
                "attention_needed": int(verdicts.get("unacceptable", 0))
                                    + int(verdicts.get("missing", 0)),
            })
    return sorted(out, key=lambda r: -r["attention_needed"])


def dashboard(wiki_id: str, playbook_id: int, run_id: int | None = None) -> dict | None:
    """Full breakdown for one playbook's run. Defaults to the latest complete
    run; pass run_id to inspect a specific past run instead."""
    if not _enabled():
        return None
    text = _text()
    run_id = run_id or latest_complete_run(wiki_id, playbook_id)
    if not run_id:
        return None

    with db.get_engine().connect() as c:
        run = c.execute(text("""
            SELECT r.id, r.collection_id, r.collection_name, r.documents_total,
                   r.findings_total, r.started_at, r.finished_at, r.documents_covered,
                   p.name
            FROM playbook_runs r JOIN playbooks p ON p.id = r.playbook_id
            WHERE r.id = :r AND r.wiki_id = :w
        """), {"r": run_id, "w": wiki_id}).fetchone()
        if not run:
            return None

        import json as _json
        covered = run[7] if isinstance(run[7], list) else _json.loads(run[7] or "[]")

        verdicts = dict(c.execute(text("""
            SELECT verdict, count(*) FROM playbook_findings
            WHERE run_id = :r GROUP BY verdict
        """), {"r": run_id}).fetchall())

        by_type_rows = c.execute(text("""
            SELECT clause_type, verdict, count(*) FROM playbook_findings
            WHERE run_id = :r GROUP BY clause_type, verdict
        """), {"r": run_id}).fetchall()
        by_clause_type: dict[str, dict] = {}
        for ct, v, n in by_type_rows:
            by_clause_type.setdefault(ct, {}).__setitem__(v, int(n))

        doc_rows = c.execute(text("""
            SELECT source_doc, verdict, severity FROM playbook_findings
            WHERE run_id = :r
        """), {"r": run_id}).fetchall()

        priority = c.execute(text("""
            SELECT source_doc, clause_type, verdict, severity, rationale, redline
            FROM playbook_findings
            WHERE run_id = :r AND verdict IN ('unacceptable', 'missing')
            ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1
                                    WHEN 'low' THEN 2 ELSE 3 END,
                     CASE verdict WHEN 'unacceptable' THEN 0 ELSE 1 END
        """), {"r": run_id}).fetchall()

        stale = []
        if run[1]:  # collection_id
            current = set(c.execute(text("""
                SELECT source_doc FROM collection_documents WHERE collection_id = :i
            """), {"i": run[1]}).scalars())
            stale = sorted(current - set(covered))

    by_document: dict[str, dict] = {}
    for doc, verdict, severity in doc_rows:
        rec = by_document.setdefault(doc, {"source_doc": doc, "worst": "standard",
                                           "counts": {}})
        rec["counts"][verdict] = rec["counts"].get(verdict, 0) + 1
        if _VERDICT_RANK.get(verdict, 9) < _VERDICT_RANK.get(rec["worst"], 9):
            rec["worst"] = verdict
    documents = sorted(by_document.values(),
                       key=lambda r: _VERDICT_RANK.get(r["worst"], 9))

    return {
        "run_id": run[0], "playbook_id": playbook_id, "playbook": run[8],
        "collection": run[2], "documents_total": run[3], "findings_total": run[4],
        "started_at": run[5].isoformat() if run[5] else None,
        "finished_at": run[6].isoformat() if run[6] else None,
        "verdicts": {k: int(v) for k, v in verdicts.items()},
        "by_clause_type": [{"clause_type": ct, **counts}
                           for ct, counts in sorted(by_clause_type.items())],
        "by_document": documents,
        "priority_findings": [
            {"source_doc": r[0], "clause_type": r[1], "verdict": r[2],
             "severity": r[3], "rationale": r[4], "redline": r[5]}
            for r in priority],
        "stale_documents": stale,
    }
