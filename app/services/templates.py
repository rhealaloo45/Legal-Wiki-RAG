"""Wording templates: which documents state a clause in the same words.

"Which other documents carry that same restriction, and how many?" is a count
of documents per wording. Counting it from whichever clauses a similarity
search returned made the answer depend on the search: eight copies of one
template measured that template alone, and a template the search ranked ninth
was never counted.

Every clause, and every verbatim quote line on a wiki page, is instead given a
fingerprint at ingest: its operative wording with names, figures and places
masked out, hashed. Documents that state a clause the same way share a
fingerprint whatever their parties or numbers, so a count is the number of
distinct documents under one fingerprint. Exact, and the same tomorrow as
today.

Quote lines on pages are fingerprinted as well as clauses because the clause
extractor does not cut a clause from every passage that states a term. On the
test index, counting clauses alone found 71 documents for a wording that 85
carry, the missing 14 having it only in page text.

Two fingerprints can still state one commitment in different words. Those are
joined into a family by an optional merge pass (see merge_families); a count
then covers the family. Nothing here calls a model except that pass.

Only hashes and document references are stored, never wording, so the table
adds no readable contract text beyond what the clauses table already holds.
"""
from __future__ import annotations

import hashlib
import logging
import re

from services import db

logger = logging.getLogger(__name__)

_MIN_QUOTE_CHARS = 40
_BATCH = 2000

# A sentence end, kept away from the abbreviations and numbering that fill a
# contract: "No. 5.", "Ltd.", "Section 3.2", "(a)." and the like.
# Each lookbehind ends where the full stop BEGINS, so it holds the
# abbreviation without its dot.
_RX_SENTENCE = re.compile(r"""
    (?<![A-Z][a-z])              # a two-letter abbreviation: No. Co. Mr. St.
    (?<!\bPte)(?<!\bLtd)(?<!\bPvt)(?<!\bInc)(?<!\bLLP)(?<!\bPLC)(?<!\bLLC)
    (?<!\bArt)(?<!\bart)(?<!\bSec)(?<!\bsec)(?<!\bCl)(?<!\bcl)
    (?<!\bNos)(?<!\bnos)(?<!\bPara)(?<!\bpara)
    [.;]\s+(?=[A-Z(\"'“])
""", re.X)


def key(text: str) -> str | None:
    """The masked operative wording of a clause, or None if it has none."""
    from services import survey
    return survey._template_key(text)


def fingerprint(text: str) -> str | None:
    """The fingerprint of a passage's first operative sentence, or None.

    Sentence-scoped so that it matches what the index stores: a lookup made
    from a retrieved clause has to hash the same way the clause was indexed.
    """
    found = wordings(text)
    return found[0][0] if found else None


def fingerprints(text: str) -> list[str]:
    """One fingerprint per operative sentence of a passage, in order.

    A fingerprint taken from the whole passage starts at its FIRST operative
    verb, so the same commitment fingerprints differently depending on what
    the extractor happened to cut with it. Measured on the test index: the
    sentence "Confidentiality obligations shall survive for so long as the
    information remains confidential" carried one fingerprint in 27 documents
    and a second in 23, and the only difference was that in those 23 the
    extractor had included the preceding sentence about the term of the
    agreement — so "shall remain effective for three years" became the
    operative wording and the survival sentence was never fingerprinted at
    all. Counting the two together gives 50, which is what the index actually
    holds.

    Sentence by sentence, the commitment is fingerprinted wherever it sits.
    A passage with one operative sentence yields exactly what it did before.
    """
    return [h for h, _ in wordings(text)]


def wordings(text: str) -> list[tuple[str, str]]:
    """(fingerprint, sentence) for each distinct operative sentence, in order."""
    out, seen = [], set()
    for sentence in _RX_SENTENCE.split(text or ""):
        # The split takes the stop with the separator, so every sentence but
        # the last has already lost it. Dropping it from the last one too
        # keeps one wording from fingerprinting two ways by its punctuation.
        sentence = sentence.strip().rstrip(".;, ")
        k = key(sentence)
        if not k:
            continue
        h = hashlib.md5(k.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append((h, " ".join(sentence.split())))
    return out


def quote_lines(content: str) -> list[str]:
    """The verbatim lines of a wiki page (its "> " supporting quotes)."""
    out = []
    for line in (content or "").splitlines():
        if line.startswith(">"):
            q = line[1:].strip()
            if len(q) >= _MIN_QUOTE_CHARS:
                out.append(q)
    return out


def _rows_for(wiki_id: str, session_id: str, clauses, pages) -> list[dict]:
    rows = []
    crypto = db._crypto()
    for cid, src, vtext in clauses:
        for i, h in enumerate(fingerprints(crypto.decrypt_safe(vtext, default=vtext) or "")):
            rows.append({"w": wiki_id, "s": session_id, "d": src, "k": "clause",
                         "u": f"c:{cid}:{i}", "h": h})
    for pid, src, content in pages:
        for i, q in enumerate(quote_lines(content)):
            for j, h in enumerate(fingerprints(q)):
                rows.append({"w": wiki_id, "s": session_id, "d": src, "k": "page",
                             "u": f"p:{pid}:{i}:{j}", "h": h})
    return rows


def _insert(conn, rows: list[dict]) -> None:
    from sqlalchemy import text
    for i in range(0, len(rows), _BATCH):
        conn.execute(text("""
            INSERT INTO clause_templates
                (wiki_id, session_id, source_doc, kind, unit_ref, template_hash)
            VALUES (:w, :s, :d, :k, :u, :h)
            ON CONFLICT (wiki_id, kind, unit_ref) DO UPDATE SET
                source_doc = EXCLUDED.source_doc, template_hash = EXCLUDED.template_hash
        """), rows[i:i + _BATCH])


def index_document(wiki_id: str, session_id: str, source_doc: str) -> int:
    """(Re)fingerprint one document. Called when a document finishes ingest,
    after its clauses and pages exist, so a new document is counted at once."""
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        conn.execute(text("DELETE FROM clause_templates WHERE wiki_id = :w "
                          "AND session_id = :s AND source_doc = :d"),
                     {"w": wiki_id, "s": session_id, "d": source_doc})
        clauses = conn.execute(text("""
            SELECT id, source_doc, verbatim_text FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
              AND review_status <> 'rejected'"""),
            {"w": wiki_id, "s": session_id, "d": source_doc}).fetchall()
        pages = conn.execute(text("""
            SELECT id, source_doc, content FROM pages
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d"""),
            {"w": wiki_id, "s": session_id, "d": source_doc}).fetchall()
        rows = _rows_for(wiki_id, session_id, clauses, pages)
        _insert(conn, rows)
        conn.commit()
    return len(rows)


def backfill(wiki_id: str, session_id: str) -> dict:
    """Fingerprint every clause and page quote line in a wiki session.

    Replaces the session's rows in one transaction, so it can be re-run at any
    time. Zero model calls: it reads text already stored.
    """
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        clauses = conn.execute(text("""
            SELECT id, source_doc, verbatim_text FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND review_status <> 'rejected'"""),
            {"w": wiki_id, "s": session_id}).fetchall()
        pages = conn.execute(text("""
            SELECT id, source_doc, content FROM pages
            WHERE wiki_id = :w AND session_id = :s AND source_doc IS NOT NULL"""),
            {"w": wiki_id, "s": session_id}).fetchall()
        rows = _rows_for(wiki_id, session_id, clauses, pages)
        conn.execute(text("DELETE FROM clause_templates WHERE wiki_id = :w AND session_id = :s"),
                     {"w": wiki_id, "s": session_id})
        _insert(conn, rows)
        conn.commit()
    return {"clauses_read": len(clauses), "pages_read": len(pages),
            "rows_written": len(rows),
            "distinct_templates": len({r["h"] for r in rows}),
            "documents": len({r["d"] for r in rows})}


def is_populated(wiki_id: str, session_id: str) -> bool:
    from sqlalchemy import text
    try:
        with db.get_engine().connect() as conn:
            return bool(conn.execute(text(
                "SELECT 1 FROM clause_templates WHERE wiki_id = :w AND session_id = :s LIMIT 1"),
                {"w": wiki_id, "s": session_id}).fetchone())
    except Exception as e:
        logger.warning("template table unavailable: %s", e)
        return False


def count_documents(wiki_id: str, session_id: str, template_hash: str) -> int:
    """Distinct documents whose text states this wording (or any wording in
    its family, once families have been merged)."""
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        fam = conn.execute(text("""
            SELECT family FROM template_families
            WHERE wiki_id = :w AND template_hash = :h"""),
            {"w": wiki_id, "h": template_hash}).scalar() or template_hash
        return int(conn.execute(text("""
            SELECT count(DISTINCT t.source_doc)
            FROM clause_templates t
            JOIN documents d ON d.wiki_id = t.wiki_id AND d.source_doc = t.source_doc
            LEFT JOIN template_families f
                   ON f.wiki_id = t.wiki_id AND f.template_hash = t.template_hash
            WHERE t.wiki_id = :w AND t.session_id = :s
              AND COALESCE(f.family, t.template_hash) = :fam"""),
            {"w": wiki_id, "s": session_id, "fam": fam}).scalar() or 0)


def family_of(wiki_id: str, template_hash: str) -> str:
    """The family a fingerprint belongs to, or the fingerprint itself."""
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        return conn.execute(text("""
            SELECT family FROM template_families
            WHERE wiki_id = :w AND template_hash = :h"""),
            {"w": wiki_id, "h": template_hash}).scalar() or template_hash


def count_any(wiki_id: str, session_id: str, template_hashes: list[str]) -> int:
    """Distinct documents stating ANY of these wordings (or their families)."""
    from sqlalchemy import text
    if not template_hashes:
        return 0
    fams = list({family_of(wiki_id, h) for h in template_hashes})
    with db.get_engine().connect() as conn:
        return int(conn.execute(text("""
            SELECT count(DISTINCT t.source_doc)
            FROM clause_templates t
            JOIN documents d ON d.wiki_id = t.wiki_id AND d.source_doc = t.source_doc
            LEFT JOIN template_families f
                   ON f.wiki_id = t.wiki_id AND f.template_hash = t.template_hash
            WHERE t.wiki_id = :w AND t.session_id = :s
              AND COALESCE(f.family, t.template_hash) = ANY(:fams)"""),
            {"w": wiki_id, "s": session_id, "fams": fams}).scalar() or 0)


def docs_by_hash(wiki_id: str, session_id: str, source_docs: list[str],
                 ) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Within these documents: which documents state each wording, and how many
    fingerprinted passages each document has at all.

    The second number is what separates a document that omits a term from a
    document the extractor barely read: a scanned copy that produced four
    passages is missing most wordings in the population, and none of those
    absences mean anything.
    """
    from sqlalchemy import text
    if not source_docs:
        return {}, {}
    by_hash: dict[str, set[str]] = {}
    units: dict[str, int] = {d: 0 for d in source_docs}
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT COALESCE(f.family, t.template_hash) AS h, t.source_doc, count(*) AS n
            FROM clause_templates t
            LEFT JOIN template_families f
                   ON f.wiki_id = t.wiki_id AND f.template_hash = t.template_hash
            WHERE t.wiki_id = :w AND t.session_id = :s AND t.source_doc = ANY(:d)
            GROUP BY 1, 2"""),
            {"w": wiki_id, "s": session_id, "d": list(source_docs)}).fetchall()
    for h, src, n in rows:
        by_hash.setdefault(h, set()).add(src)
        units[src] = units.get(src, 0) + int(n)
    return by_hash, units


def example_text(wiki_id: str, session_id: str, template_hash: str) -> str | None:
    """One passage carrying this wording, read back through the unit it came
    from — the table stores hashes, so the words have to come from the source."""
    from sqlalchemy import text
    crypto = db._crypto()
    with db.get_engine().connect() as conn:
        refs = conn.execute(text("""
            SELECT kind, unit_ref FROM clause_templates
            WHERE wiki_id = :w AND session_id = :s AND template_hash = :h
            ORDER BY kind LIMIT 8"""),
            {"w": wiki_id, "s": session_id, "h": template_hash}).fetchall()
        for kind, ref in refs:
            try:
                parts = ref.split(":")
                if kind == "clause":
                    raw = conn.execute(text(
                        "SELECT verbatim_text FROM clauses WHERE id = :i"),
                        {"i": int(parts[1])}).scalar()
                    passage = crypto.decrypt_safe(raw, default=raw) if raw else ""
                    nth = int(parts[2]) if len(parts) > 2 else 0
                else:
                    content = conn.execute(text(
                        "SELECT content FROM pages WHERE id = :i"),
                        {"i": int(parts[1])}).scalar()
                    lines = quote_lines(content or "")
                    idx = int(parts[2])
                    passage = lines[idx] if idx < len(lines) else ""
                    nth = int(parts[3]) if len(parts) > 3 else 0
                # The fingerprint belongs to one sentence of the passage, so
                # that sentence is the example — not the whole extracted
                # chunk, which may carry unrelated wording around it.
                found = wordings(passage)
                if nth < len(found):
                    return found[nth][1]
                if passage:
                    return passage
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Merging fingerprints that state one commitment in different words
# ---------------------------------------------------------------------------

_MIN_DOCS_TO_MERGE = 5
_MERGE_BATCH = 25
_SAMPLE_CHARS = 350

_MERGE_PROMPT = """\
You compare contract clause wordings. Below are {n} wordings taken from documents of one clause type, each with an id. Party names and figures differ between documents and do not matter.

Group ids that state the SAME obligation with the SAME scope and exceptions: a lawyer would say they impose one commitment in different words. Do NOT group wordings that differ in who is bound, what is restricted, whether the duration is fixed or open-ended, what exceptions apply, or whether consent is needed. When unsure, leave them apart: a wrong group overstates how widely a term is used.

Return JSON only, no other text: {{"groups": [[id, id], ...]}}, listing only groups of two or more, using the ids given.

{items}"""


def _candidates(wiki_id: str, session_id: str, min_docs: int) -> dict[str, list[dict]]:
    """Templates worth comparing, bucketed by clause type: the clause type the
    template's clauses most often carry, the documents it covers, and one
    sample clause text."""
    from sqlalchemy import text
    crypto = db._crypto()
    out: dict[str, list[dict]] = {}
    with db.get_engine().connect() as conn:
        docs = {r[0]: int(r[1]) for r in conn.execute(text("""
            SELECT template_hash, count(DISTINCT source_doc) FROM clause_templates
            WHERE wiki_id = :w AND session_id = :s GROUP BY 1 HAVING count(DISTINCT source_doc) >= :m"""),
            {"w": wiki_id, "s": session_id, "m": min_docs}).fetchall()}
        rows = conn.execute(text("""
            SELECT t.template_hash, COALESCE(cl.clause_type_canon, '(untyped)') AS canon,
                   count(*) AS n, min(cl.verbatim_text) AS sample
            FROM clause_templates t
            JOIN clauses cl ON t.kind = 'clause' AND t.unit_ref = 'c:' || cl.id
            WHERE t.wiki_id = :w AND t.session_id = :s AND t.template_hash = ANY(:hs)
            GROUP BY 1, 2"""), {"w": wiki_id, "s": session_id, "hs": list(docs)}).fetchall()
    best: dict[str, tuple] = {}
    for h, canon, n, sample in rows:
        if h not in best or n > best[h][0]:
            best[h] = (n, canon, sample)
    for h, (_, canon, sample) in best.items():
        txt = " ".join((crypto.decrypt_safe(sample, default=sample) or "").split())[:_SAMPLE_CHARS]
        if txt:
            out.setdefault(canon, []).append({"hash": h, "docs": docs[h], "text": txt})
    for v in out.values():
        v.sort(key=lambda x: -x["docs"])
    return out


def propose_families(wiki_id: str, session_id: str,
                     min_docs: int = _MIN_DOCS_TO_MERGE) -> dict:
    """Ask the chat model which wordings of one clause type state one
    commitment. Writes nothing: returns the proposed groups with their
    document counts and sample wording, and the tokens spent, for review.

    Only templates covering at least `min_docs` documents are compared, and only
    within one clause type, so the model never sees an unrelated pair and the
    call count stays small."""
    import json
    import re
    from services import llm
    cands = _candidates(wiki_id, session_id, min_docs)
    groups, tokens, calls, bad = [], {"prompt": 0, "completion": 0}, 0, 0
    for canon, items in sorted(cands.items()):
        if len(items) < 2:
            continue
        for i in range(0, len(items), _MERGE_BATCH):
            chunk = items[i:i + _MERGE_BATCH]
            if len(chunk) < 2:
                continue
            prompt = _MERGE_PROMPT.format(
                n=len(chunk),
                items="\n".join(f"[{j}] {c['text']}" for j, c in enumerate(chunk)))
            raw, usage = llm.fast_ask(prompt, max_tokens=2048)
            if not (raw or "").strip():
                raw, usage2 = llm.fast_ask(prompt, max_tokens=4096)
                usage = {k: (usage or {}).get(k, 0) + (usage2 or {}).get(k, 0)
                         for k in ("prompt_tokens", "completion_tokens")}
            calls += 1
            tokens["prompt"] += (usage or {}).get("prompt_tokens", 0)
            tokens["completion"] += (usage or {}).get("completion_tokens", 0)
            try:
                parsed = json.loads(re.sub(r"```(?:json)?", "", raw or "").strip())
                found = parsed.get("groups", [])
            except Exception:
                bad += 1
                continue
            used = set()
            for g in found:
                idx = [x for x in g if isinstance(x, int) and 0 <= x < len(chunk) and x not in used]
                if len(idx) < 2:
                    continue
                used.update(idx)
                members = [chunk[x] for x in idx]
                groups.append({"clause_type": canon,
                               "members": [{"hash": m["hash"], "docs": m["docs"],
                                            "sample": m["text"]} for m in members]})
    return {"groups": groups, "llm_calls": calls, "unparsed_replies": bad,
            "tokens": tokens, "clause_types_compared": sum(1 for v in cands.values() if len(v) >= 2),
            "templates_considered": sum(len(v) for v in cands.values())}


def save_families(wiki_id: str, groups: list[dict]) -> int:
    """Store reviewed groups: every member of a group points at its first
    member's fingerprint as the family id."""
    from sqlalchemy import text
    rows = []
    for g in groups:
        fam = g["members"][0]["hash"]
        rows += [{"w": wiki_id, "h": m["hash"], "f": fam} for m in g["members"]]
    with db.get_engine().connect() as conn:
        for r in rows:
            conn.execute(text("""
                INSERT INTO template_families (wiki_id, template_hash, family)
                VALUES (:w, :h, :f)
                ON CONFLICT (wiki_id, template_hash) DO UPDATE SET family = EXCLUDED.family"""), r)
        conn.commit()
    return len(rows)
