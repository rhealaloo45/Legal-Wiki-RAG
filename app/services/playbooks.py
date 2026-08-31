"""Playbooks — house positions per clause type, run over a Collection.

Per the target architecture: "pull clause → classify vs. standard / fallback /
unacceptable → suggest redline → citation check", feeding the Deviation
Dashboard in Phase 3.

Two things shape the implementation:

* **Pulling a clause is a DB read, not an LLM call.** Ingest already extracted
  and persisted clauses, so a run spends inference only on the judgement.

* **Rules match clause types by keyword, not equality.** Extraction writes
  free-form type names — 5,957 distinct ones across this corpus, with
  "liability" alone appearing under 128 spellings ("Liability Cap", "Liability
  Cap Reference", "Limitation of Liability"). A rule keyed to one exact string
  would silently assess nothing, which reads identically to full compliance.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from services import db

logger = logging.getLogger(__name__)

VERDICTS = ("standard", "fallback", "unacceptable", "missing", "unclear")
SEVERITIES = ("low", "medium", "high")


class PlaybookError(Exception):
    """Conflicts a caller should report rather than swallow."""


def _text():
    from sqlalchemy import text
    return text


def _enabled() -> bool:
    import config
    return bool(config.USE_DATABASE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


def _keywords(clause_type: str) -> list[str]:
    """Match terms for a rule's clause type.

    The distinctive words are what carry it: a rule for "Liability Cap" should
    reach "Liability Cap Reference" and "Limitation of Liability", so each
    content word is a candidate and a type matching ANY of them is assessed.
    """
    stop = {"the", "of", "and", "or", "a", "an", "to", "for", "in", "on", "clause"}
    return [w for w in _norm(clause_type).split() if w not in stop and len(w) > 3]


# ---------------------------------------------------------------------------
# playbook CRUD
# ---------------------------------------------------------------------------

def create(wiki_id: str, session_id: str, name: str,
           description: str | None = None) -> dict:
    name = (name or "").strip()
    if not name:
        raise PlaybookError("Playbook name cannot be empty")
    if not _enabled():
        raise PlaybookError("Database not configured")
    text = _text()
    with db.get_engine().connect() as c:
        if c.execute(text("SELECT id FROM playbooks WHERE wiki_id=:w AND name=:n"),
                     {"w": wiki_id, "n": name}).scalar():
            raise PlaybookError(f"A playbook named {name!r} already exists in this wiki")
        row = c.execute(text("""
            INSERT INTO playbooks (wiki_id, session_id, name, description)
            VALUES (:w, :s, :n, :d) RETURNING id
        """), {"w": wiki_id, "s": session_id, "n": name,
               "d": (description or "").strip() or None}).fetchone()
        c.commit()
    return {"id": int(row[0]), "name": name, "description": description, "rules": 0}


def list_all(wiki_id: str) -> list[dict]:
    if not _enabled():
        return []
    text = _text()
    with db.get_engine().connect() as c:
        rows = c.execute(text("""
            SELECT p.id, p.name, p.description, p.created_at, count(r.id)
            FROM playbooks p LEFT JOIN playbook_rules r ON r.playbook_id = p.id
            WHERE p.wiki_id = :w GROUP BY p.id ORDER BY p.name
        """), {"w": wiki_id}).fetchall()
    return [{"id": int(r[0]), "name": r[1], "description": r[2],
             "created_at": r[3].isoformat() if r[3] else None,
             "rules": int(r[4])} for r in rows]


def get(wiki_id: str, playbook_id: int) -> dict | None:
    if not _enabled():
        return None
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT id, name, description FROM playbooks WHERE wiki_id=:w AND id=:i
        """), {"w": wiki_id, "i": playbook_id}).fetchone()
        if not row:
            return None
        rules = c.execute(text("""
            SELECT id, clause_type, standard, fallback, unacceptable, guidance,
                   severity, ordinal
            FROM playbook_rules WHERE playbook_id=:i ORDER BY ordinal, clause_type
        """), {"i": playbook_id}).fetchall()
    return {
        "id": int(row[0]), "name": row[1], "description": row[2],
        "rules": [{"id": int(r[0]), "clause_type": r[1], "standard": r[2],
                   "fallback": r[3], "unacceptable": r[4], "guidance": r[5],
                   "severity": r[6], "ordinal": r[7]} for r in rules],
    }


def delete(wiki_id: str, playbook_id: int) -> bool:
    """Delete a playbook, its rules and its past runs.

    Refuses while a run is still in flight. The cascade would otherwise remove
    the run row out from under the worker still writing findings against it,
    which fails on the foreign key mid-run — the run dies partway through and
    the findings it had already produced go with it.
    """
    if not _enabled():
        return False
    text = _text()
    with db.get_engine().connect() as c:
        active = c.execute(text("""
            SELECT count(*) FROM playbook_runs
            WHERE playbook_id = :i AND status = 'running'
        """), {"i": playbook_id}).scalar() or 0
        if active:
            raise PlaybookError(
                f"{active} run(s) still in progress — wait for them to finish "
                f"before deleting this playbook")
        res = c.execute(text("DELETE FROM playbooks WHERE wiki_id=:w AND id=:i"),
                        {"w": wiki_id, "i": playbook_id})
        c.commit()
    return (res.rowcount or 0) > 0


def run_exists(run_id: int) -> bool:
    """Whether the run row is still there — it is not, if the playbook was
    deleted underneath a run in flight."""
    text = _text()
    with db.get_engine().connect() as c:
        return bool(c.execute(text("SELECT 1 FROM playbook_runs WHERE id=:r"),
                              {"r": run_id}).scalar())


def add_rule(wiki_id: str, playbook_id: int, clause_type: str, standard: str,
             fallback: str | None = None, unacceptable: str | None = None,
             guidance: str | None = None, severity: str = "medium",
             ordinal: int = 0) -> dict:
    """Add or replace the rule for one clause type.

    `standard` is required: a rule with no stated house position gives the
    classifier nothing to compare against, so it could only ever return
    "unclear".
    """
    clause_type = (clause_type or "").strip()
    standard = (standard or "").strip()
    if not clause_type:
        raise PlaybookError("clause_type is required")
    if not standard:
        raise PlaybookError("A rule needs a standard position to compare against")
    if severity not in SEVERITIES:
        raise PlaybookError(f"severity must be one of {SEVERITIES}")
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            INSERT INTO playbook_rules (playbook_id, wiki_id, clause_type, standard,
                                        fallback, unacceptable, guidance, severity, ordinal)
            VALUES (:p, :w, :c, :s, :f, :u, :g, :sev, :o)
            ON CONFLICT (playbook_id, clause_type) DO UPDATE SET
                standard = EXCLUDED.standard, fallback = EXCLUDED.fallback,
                unacceptable = EXCLUDED.unacceptable, guidance = EXCLUDED.guidance,
                severity = EXCLUDED.severity, ordinal = EXCLUDED.ordinal
            RETURNING id
        """), {"p": playbook_id, "w": wiki_id, "c": clause_type, "s": standard,
               "f": fallback, "u": unacceptable, "g": guidance,
               "sev": severity, "o": ordinal}).fetchone()
        c.execute(text("UPDATE playbooks SET updated_at=now() WHERE id=:i"),
                  {"i": playbook_id})
        c.commit()
    return {"id": int(row[0]), "clause_type": clause_type}


def remove_rule(wiki_id: str, playbook_id: int, clause_type: str) -> bool:
    text = _text()
    with db.get_engine().connect() as c:
        res = c.execute(text("""
            DELETE FROM playbook_rules
            WHERE playbook_id=:p AND wiki_id=:w AND clause_type=:c
        """), {"p": playbook_id, "w": wiki_id, "c": clause_type})
        c.commit()
    return (res.rowcount or 0) > 0


def resolve(wiki_id: str, ref: Any) -> int | None:
    if ref is None:
        return None
    text = _text()
    with db.get_engine().connect() as c:
        if isinstance(ref, int) or (isinstance(ref, str) and str(ref).isdigit()):
            return c.execute(text("SELECT id FROM playbooks WHERE wiki_id=:w AND id=:i"),
                             {"w": wiki_id, "i": int(ref)}).scalar()
        return c.execute(text("SELECT id FROM playbooks WHERE wiki_id=:w AND name=:n"),
                         {"w": wiki_id, "n": str(ref).strip()}).scalar()


# ---------------------------------------------------------------------------
# matching clauses to rules
# ---------------------------------------------------------------------------

def clauses_for_rule(wiki_id: str, source_doc: str, clause_type: str) -> list[dict]:
    """Clauses in a document that a rule for `clause_type` should assess.

    Keyword match, not equality — see the module docstring. Rejected clauses
    are excluded: a reviewer has already said that row is wrong, and assessing
    it would report a deviation that rests on discredited text.

    STILL THE KEYWORD MATCHER. The canonical vocabulary (§ Phase 3.5c) exists
    and is backfilled, but this function has deliberately not been switched
    over to it yet — see clauses_for_rule_canon and compare_rule_matching
    below. Today's matcher OVER-matches, which shows up as a visibly wrong
    finding a reviewer can catch; a canonical matcher with a bad mapping
    UNDER-matches, and a clause that is silently never assessed produces no
    finding at all for anyone to notice. Trading a loud failure for a quiet
    one is a regression even if the match rate improves, so the cutover waits
    on a reviewed diff rather than on the new code merely existing.
    """
    if not _enabled():
        return []
    kws = _keywords(clause_type)
    if not kws:
        return []
    text = _text()
    with db.get_engine().connect() as c:
        rows = c.execute(text("""
            SELECT id, clause_type, verbatim_text, page_num, confidence, review_status
            FROM clauses
            WHERE wiki_id = :w AND source_doc = :d
              AND review_status <> 'rejected'
        """), {"w": wiki_id, "d": source_doc}).fetchall()
    out = []
    for r in rows:
        n = _norm(r[1])
        if any(k in n for k in kws):
            out.append({"id": int(r[0]), "clause_type": r[1], "text": r[2],
                        "page_num": r[3], "confidence": r[4],
                        "review_status": r[5]})
    return out


def clauses_for_rule_canon(wiki_id: str, source_doc: str, clause_type: str) -> list[dict]:
    """Canonical-type equivalent of clauses_for_rule. Not yet wired into runs.

    Selects on clauses.clause_type_canon, so a "Liability Cap" rule matches
    `liability_cap` and NOT `liability_cap_exclusion` — the carve-out is a
    different clause with a different standard, and assessing it against the
    cap rule is the defect this replaces.

    Returns [] when the rule's own name does not map to a canonical type: a
    rule the vocabulary cannot place must select nothing rather than fall back
    to keywords, or the comparison below would silently measure the old
    matcher twice.
    """
    if not _enabled():
        return []
    from services import clause_vocab
    canon = clause_vocab.canonical(clause_type)
    if not canon:
        return []
    text = _text()
    with db.get_engine().connect() as c:
        rows = c.execute(text("""
            SELECT id, clause_type, verbatim_text, page_num, confidence, review_status
            FROM clauses
            WHERE wiki_id = :w AND source_doc = :d
              AND review_status <> 'rejected'
              AND clause_type_canon = :canon
        """), {"w": wiki_id, "d": source_doc, "canon": canon}).fetchall()
    return [{"id": int(r[0]), "clause_type": r[1], "text": r[2], "page_num": r[3],
             "confidence": r[4], "review_status": r[5]} for r in rows]


def compare_rule_matching(wiki_id: str, source_docs: list[str],
                          clause_types: list[str]) -> dict:
    """Diff the keyword matcher against the canonical matcher, without running
    either against a model.

    This is the gate on the cutover. Three numbers matter, and they are not
    equally important:

      dropped   clauses the keyword matcher assessed that the canonical one
                does not. Mostly the intended fix (carve-outs leaving a cap
                rule), but a genuine cap the vocabulary failed to map would
                also land here — every entry needs a human eye.
      added     clauses the canonical matcher picks up that keywords missed.
      unmapped  rules whose own name does not map to a canonical type. These
                would select nothing after a cutover, which is the loudest
                possible under-match, so a non-empty list here blocks it.

    Zero LLM calls: this compares selection, not verdicts.
    """
    if not _enabled():
        return {"error": "playbooks disabled"}
    from services import clause_vocab
    dropped, added = [], []
    unmapped_rules = [ct for ct in clause_types if not clause_vocab.canonical(ct)]
    kw_total = canon_total = 0

    for doc in source_docs:
        for ct in clause_types:
            kw = {c["id"]: c for c in clauses_for_rule(wiki_id, doc, ct)}
            cn = {c["id"]: c for c in clauses_for_rule_canon(wiki_id, doc, ct)}
            kw_total += len(kw)
            canon_total += len(cn)
            for cid in kw.keys() - cn.keys():
                dropped.append({"rule": ct, "source_doc": doc, "clause_id": cid,
                                "clause_type": kw[cid]["clause_type"]})
            for cid in cn.keys() - kw.keys():
                added.append({"rule": ct, "source_doc": doc, "clause_id": cid,
                              "clause_type": cn[cid]["clause_type"]})

    def _by_type(items):
        agg: dict[str, int] = {}
        for i in items:
            agg[i["clause_type"]] = agg.get(i["clause_type"], 0) + 1
        return sorted(agg.items(), key=lambda x: -x[1])

    return {
        "documents": len(source_docs), "rules": len(clause_types),
        "keyword_selected": kw_total, "canon_selected": canon_total,
        "dropped_count": len(dropped), "added_count": len(added),
        "dropped_by_type": _by_type(dropped), "added_by_type": _by_type(added),
        "unmapped_rules": unmapped_rules,
        "safe_to_cut_over": not unmapped_rules,
        "dropped_sample": dropped[:40], "added_sample": added[:40],
    }


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def start_run(wiki_id: str, session_id: str, playbook_id: int,
              documents: list[str], collection_id: int | None = None,
              collection_name: str | None = None) -> int:
    """Open a run row, freezing the document list it will cover.

    The covered list is stored on the run rather than resolved from the
    collection later: membership can change after a run, and a run whose scope
    drifts is not reproducible evidence of anything.
    """
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            INSERT INTO playbook_runs
                (wiki_id, session_id, playbook_id, collection_id, collection_name,
                 documents_total, documents_covered)
            VALUES (:w, :s, :p, :c, :cn, :n, :docs) RETURNING id
        """), {"w": wiki_id, "s": session_id, "p": playbook_id, "c": collection_id,
               "cn": collection_name, "n": len(documents),
               "docs": json.dumps(documents)}).fetchone()
        c.commit()
    return int(row[0])


def record_finding(run_id: int, wiki_id: str, source_doc: str, clause_type: str,
                   verdict: str, **fields: Any) -> None:
    if verdict not in VERDICTS:
        raise PlaybookError(f"verdict must be one of {VERDICTS}")
    text = _text()
    with db.get_engine().connect() as c:
        c.execute(text("""
            INSERT INTO playbook_findings
                (run_id, wiki_id, source_doc, clause_type, clause_id, verdict,
                 severity, rationale, redline, clause_text, grounded, confidence)
            VALUES (:r, :w, :d, :c, :cid, :v, :sev, :rat, :red, :txt, :g, :conf)
        """), {"r": run_id, "w": wiki_id, "d": source_doc, "c": clause_type,
               "cid": fields.get("clause_id"), "v": verdict,
               "sev": fields.get("severity"), "rat": fields.get("rationale"),
               "red": fields.get("redline"), "txt": (fields.get("clause_text") or "")[:4000],
               "g": fields.get("grounded"), "conf": fields.get("confidence")})
        c.execute(text("""
            UPDATE playbook_runs SET findings_total = findings_total + 1 WHERE id = :r
        """), {"r": run_id})
        c.commit()


def mark_document_done(run_id: int) -> None:
    text = _text()
    with db.get_engine().connect() as c:
        c.execute(text(
            "UPDATE playbook_runs SET documents_done = documents_done + 1 WHERE id=:r"),
            {"r": run_id})
        c.commit()


def finish_run(run_id: int, status: str = "complete", error: str | None = None) -> None:
    text = _text()
    with db.get_engine().connect() as c:
        c.execute(text("""
            UPDATE playbook_runs SET status=:s, error=:e, finished_at=now() WHERE id=:r
        """), {"s": status, "e": error, "r": run_id})
        c.commit()


def get_run(wiki_id: str, run_id: int, with_findings: bool = True) -> dict | None:
    if not _enabled():
        return None
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT r.id, r.playbook_id, p.name, r.collection_name, r.status,
                   r.documents_total, r.documents_done, r.findings_total,
                   r.started_at, r.finished_at, r.error, r.documents_covered
            FROM playbook_runs r JOIN playbooks p ON p.id = r.playbook_id
            WHERE r.wiki_id = :w AND r.id = :i
        """), {"w": wiki_id, "i": run_id}).fetchone()
        if not row:
            return None
        out = {"id": int(row[0]), "playbook_id": int(row[1]), "playbook": row[2],
               "collection": row[3], "status": row[4],
               "documents_total": row[5], "documents_done": row[6],
               "findings_total": row[7],
               "started_at": row[8].isoformat() if row[8] else None,
               "finished_at": row[9].isoformat() if row[9] else None,
               "error": row[10],
               "documents_covered": row[11] if isinstance(row[11], list)
                                    else json.loads(row[11] or "[]")}
        counts = c.execute(text("""
            SELECT verdict, count(*) FROM playbook_findings
            WHERE run_id = :i GROUP BY verdict
        """), {"i": run_id}).fetchall()
        out["verdicts"] = {r[0]: int(r[1]) for r in counts}
        if with_findings:
            f = c.execute(text("""
                SELECT source_doc, clause_type, verdict, severity, rationale,
                       redline, grounded, confidence, clause_id
                FROM playbook_findings WHERE run_id = :i
                ORDER BY CASE verdict WHEN 'unacceptable' THEN 0 WHEN 'missing' THEN 1
                                      WHEN 'fallback' THEN 2 WHEN 'unclear' THEN 3
                                      ELSE 4 END, source_doc, clause_type
            """), {"i": run_id}).fetchall()
            out["findings"] = [{"source_doc": r[0], "clause_type": r[1],
                                "verdict": r[2], "severity": r[3], "rationale": r[4],
                                "redline": r[5], "grounded": r[6],
                                "confidence": r[7], "clause_id": r[8]} for r in f]
    return out


_CLASSIFY_PROMPT = """You are auditing one contract clause against a house playbook.

CLAUSE TYPE: {clause_type}

HOUSE POSITIONS
Standard (preferred):  {standard}
Fallback (acceptable): {fallback}
Unacceptable:          {unacceptable}
{guidance}

CLAUSE AS DRAFTED:
\"\"\"{clause_text}\"\"\"

Classify the drafted clause against the house positions and, when it is not
standard, give the minimal redline that would bring it to standard or fallback.

Rules:
- "standard" only if it matches the standard position in substance.
- "fallback" if it departs from standard but stays within the fallback.
- "unacceptable" if it matches the unacceptable position, or is worse than fallback.
- "unclear" if the clause text does not settle the question.
- Judge substance, not wording.
- Quote nothing that is not in the clause text above.

Strict JSON only:
{{"verdict": "standard|fallback|unacceptable|unclear",
  "rationale": "<20 words max>",
  "redline": "<the changed wording, or null if standard>"}}"""


def assess_clause(rule: dict, clause_text: str, ask=None) -> dict:
    """Classify one clause against one rule. The only LLM call in a run.

    A malformed or failed response returns "unclear" rather than a guess: on a
    deviation report, a wrong verdict is worse than an absent one, because it
    is acted on.
    """
    from services import llm as _llm
    ask = ask or (lambda p, **kw: _llm.ask(p, **kw))
    prompt = _CLASSIFY_PROMPT.format(
        clause_type=rule["clause_type"],
        standard=rule.get("standard") or "(not stated)",
        fallback=rule.get("fallback") or "(none defined)",
        unacceptable=rule.get("unacceptable") or "(none defined)",
        guidance=f"Guidance: {rule['guidance']}" if rule.get("guidance") else "",
        clause_text=(clause_text or "")[:3000],
    )
    try:
        import config
        raw, _ = ask(prompt, fast=True,
                     max_tokens=getattr(config, "MAX_TOKENS_RERANK", 2048))
        m = re.search(r"\{.*\}", raw or "", re.S)
        data = json.loads(m.group(0)) if m else {}
        verdict = str(data.get("verdict", "")).lower().strip()
        if verdict not in ("standard", "fallback", "unacceptable", "unclear"):
            return {"verdict": "unclear", "rationale": "unparseable classifier output",
                    "redline": None}
        return {"verdict": verdict,
                "rationale": str(data.get("rationale") or "")[:300],
                "redline": (str(data.get("redline")) if data.get("redline") else None)}
    except Exception as e:
        logger.warning("Playbook classify failed: %s", e)
        return {"verdict": "unclear", "rationale": f"classifier error: {type(e).__name__}",
                "redline": None}


def run(wiki_id: str, session_id: str, playbook_id: int, documents: list[str],
        collection_id: int | None = None, collection_name: str | None = None,
        ask=None, progress=None) -> int:
    """Run a playbook over documents. Returns the run id.

    Per document, per rule: pull the matching clauses (a DB read — ingest
    already extracted them), classify each against the house position, and
    record the verdict. A rule with no matching clause records `missing`,
    which is a finding in its own right: a contract with no liability cap at
    all is exactly what a deviation report should surface, and skipping it
    would make silence look like compliance.

    Every finding carries `grounded` — whether the assessed text is really in
    the stored clause — so the dashboard can separate a deviation from a
    classification made against text that does not exist.
    """
    book = get(wiki_id, playbook_id)
    if not book:
        raise PlaybookError("Playbook not found")
    if not book["rules"]:
        raise PlaybookError("Playbook has no rules to run")

    run_id = start_run(wiki_id, session_id, playbook_id, documents,
                       collection_id, collection_name)
    try:
        for doc in documents:
            # The playbook can be deleted while this is running, which cascades
            # the run row away. Stop cleanly rather than failing on the foreign
            # key for every remaining clause.
            if not run_exists(run_id):
                logger.info("Playbook run %s was deleted mid-run — stopping", run_id)
                return run_id
            for rule in book["rules"]:
                hits = clauses_for_rule(wiki_id, doc, rule["clause_type"])
                if not hits:
                    record_finding(run_id, wiki_id, doc, rule["clause_type"],
                                   "missing", severity=rule.get("severity"),
                                   rationale="No clause of this type found in the document",
                                   grounded=True)
                    continue
                for h in hits:
                    res = assess_clause(rule, h["text"], ask=ask)
                    record_finding(
                        run_id, wiki_id, doc, rule["clause_type"], res["verdict"],
                        clause_id=h["id"], severity=rule.get("severity"),
                        rationale=res["rationale"], redline=res["redline"],
                        clause_text=h["text"],
                        # The clause came from the stored extraction, so it is
                        # grounded by construction; recorded explicitly so the
                        # dashboard never has to assume it.
                        grounded=True, confidence=h.get("confidence"))
            mark_document_done(run_id)
            if progress:
                progress(doc)
        finish_run(run_id, "complete")
    except Exception as e:
        logger.error("Playbook run %s failed: %s", run_id, e, exc_info=True)
        finish_run(run_id, "error", str(e)[:400])
        raise
    return run_id


def list_runs(wiki_id: str, playbook_id: int | None = None,
              limit: int = 25) -> list[dict]:
    if not _enabled():
        return []
    text = _text()
    clause = "AND r.playbook_id = :p" if playbook_id else ""
    params: dict = {"w": wiki_id, "l": limit}
    if playbook_id:
        params["p"] = playbook_id
    with db.get_engine().connect() as c:
        rows = c.execute(text(f"""
            SELECT r.id, p.name, r.collection_name, r.status, r.documents_total,
                   r.documents_done, r.findings_total, r.started_at
            FROM playbook_runs r JOIN playbooks p ON p.id = r.playbook_id
            WHERE r.wiki_id = :w {clause}
            ORDER BY r.started_at DESC LIMIT :l
        """), params).fetchall()
    return [{"id": int(r[0]), "playbook": r[1], "collection": r[2], "status": r[3],
             "documents_total": r[4], "documents_done": r[5],
             "findings_total": r[6],
             "started_at": r[7].isoformat() if r[7] else None} for r in rows]
