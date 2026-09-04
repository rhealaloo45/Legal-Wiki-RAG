"""Check the 200-question test set against the documents it claims to describe.

Nine of its expected answers are already known wrong, which puts about 4.5
points of error in every accuracy figure measured with it. This finds the rest
without a model call: the expected answer's own figures and dates either appear
in the named document's stored content or they do not.

Two verdicts matter and they are asymmetric.

  disagrees-positive   the expected answer states a figure the document does
                       not contain anywhere
  disagrees-negative   the expected answer says the document does NOT address
                       something, and the document plainly does - the shape of
                       every one of the nine already found

Everything else is reported as agrees or undecidable, and undecidable is not a
criticism: an expected answer that is pure prose has nothing this can check.
"""
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join("C:/Users/Rhea/Desktop/Tasks/Legal-wiki-RAG", "app"))
S = ("C:/Users/Rhea/AppData/Local/Temp/claude"
     "/C--Users-Rhea-Desktop-Tasks-Legal-wiki-RAG"
     "/12a38837-56f3-4107-951d-ab5ae7a1a913/scratchpad")
SID = "57983304-3d63-40dc-bbd9-c9ff55a75232"

from services import db as DB
import services.wiki as W
from sqlalchemy import text

WID = W._active_wiki_id()

# Figures and dates carry the meaning of a legal answer; prose is paraphrased
# freely and checking it would measure wording rather than truth.
_RX_FACT = re.compile(
    r"(?:Rs\.?|INR|USD|\$|EUR|GBP)\s*[\d,]+(?:\.\d+)?"
    r"|\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b"
    r"|\b\d+(?:\.\d+)?\s*%"
    r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b"
    r"|\b\d+\s*(?:days?|months?|years?|weeks?)\b",
    re.IGNORECASE)

# An expected answer asserting the document is silent about something.
_RX_NEGATIVE = re.compile(
    r"\b(?:does not|doesn't|no such|not addressed|not contain|is silent|"
    r"not covered|not stated|not specified)\b", re.IGNORECASE)

# The subject of that denial: "does not contain a 'rent escalation' clause".
# A quoted subject is taken as-is; otherwise the words immediately before the
# instrument noun. Deliberately strict, because a loose version of this read
# "does not contain the FIRST document's governing law clause" as a denial
# about the word "first" and flagged three correct conflation rows.
_RX_DENIED_QUOTED = re.compile(r"[\"'‘“]([a-z][a-z /&'-]{4,40})[\"'’”]",
                               re.IGNORECASE)
_RX_DENIED_SUBJECT = re.compile(
    '((?:[a-z][a-z-]{2,}\\s+){1,3}?)(?:clause|provision|section|obligation)',
    re.IGNORECASE)

# Words that carry no subject on their own.
_SUBJ_STOP = {"first", "second", "third", "the", "a", "an", "any", "this", "that",
              "such", "document", "documents", "other", "same", "said", "its"}


# A conflation row denies that one document states ANOTHER document's clause.
# That is an assertion about the question, not about the document's contents,
# and reading it as "governing law is absent" flags three correct rows - every
# contract has a governing law clause.
_RX_CROSS_REF_DENIAL = re.compile(
    r"conflates|unrelated documents|the (?:first|second|other) document", re.IGNORECASE)


def denied_subject(expected):
    """The thing an expected answer says is absent, or "" if not readable."""
    if _RX_CROSS_REF_DENIAL.search(expected or ""):
        return ""
    m = _RX_DENIED_QUOTED.search(expected)
    cand = m.group(1) if m else ""
    if not cand:
        m2 = _RX_DENIED_SUBJECT.search(expected)
        cand = m2.group(1) if m2 else ""
    words = [w for w in re.findall(r"[a-z][a-z-]*", (cand or "").lower())
             if w not in _SUBJ_STOP]
    if len(words) < 2 and not (len(words) == 1 and len(words[0]) >= 8):
        return ""
    return " ".join(words)


def norm(s):
    s = (s or "").lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[,]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def doc_tail(path):
    base = (path or "").split("/")[-1].strip()
    return norm(re.sub(r"\.(pdf|docx?|txt)$", "", base, flags=re.I))


def haystack_for(tail):
    """Everything stored for the document the row names: pages and clauses."""
    if not tail:
        return None, None
    with DB.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT DISTINCT source_doc FROM pages
            WHERE session_id = :s AND lower(source_doc) LIKE :t
            LIMIT 1
        """), {"s": SID, "t": "%" + tail[:60].replace(" ", "%") + "%"}).scalar()
        if not row:
            return None, None
        parts = [r[0] for r in c.execute(text("""
            SELECT content FROM pages WHERE session_id = :s AND source_doc = :d
        """), {"s": SID, "d": row}).fetchall()]
        parts += [r[0] for r in c.execute(text("""
            SELECT verbatim_text FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
        """), {"w": WID, "s": SID, "d": row}).fetchall()]
    return row, norm(" ".join(p for p in parts if p))


def haystack_from_question(question):
    """Same content, with the document resolved from the question text."""
    try:
        docs = (W.resolve_scope(question, SID).get("target_docs") or [])[:3]
    except Exception:
        return None, None
    if not docs:
        return None, None
    parts = []
    with DB.get_engine().connect() as c:
        for d in docs:
            parts += [x[0] for x in c.execute(text(
                "SELECT content FROM pages WHERE session_id=:s AND source_doc=:d"),
                {"s": SID, "d": d}).fetchall()]
            parts += [x[0] for x in c.execute(text(
                "SELECT verbatim_text FROM clauses WHERE wiki_id=:w AND session_id=:s "
                "AND source_doc=:d"), {"w": WID, "s": SID, "d": d}).fetchall()]
    if not parts:
        return None, None
    return docs[0], norm(" ".join(p for p in parts if p))


def clause_type_present(doc, subj):
    """Whether a clause of the denied type is recorded for the document.

    Substring matching on the canonical phrase misses paraphrase, which is the
    common case: a lease that reads "the Rent shall escalate by 15% at the end
    of every 1 year(s)" nowhere contains the string "rent escalation". The
    typed clause rows know the type even when the prose does not name it.
    """
    if not doc or not subj:
        return False
    try:
        from services import clause_vocab as _v
        canon = _v.canonical(subj)
        if not canon:
            return False
        return bool(DB.clauses_of_type(WID, SID, doc, canon))
    except Exception:
        return False


def main():
    graded = json.load(io.open(os.path.join(S, "run200_results_graded.json"),
                               encoding="utf-8"))
    out = []
    counts = {}
    for r in graded:
        expected = r.get("expected") or ""
        src = r.get("source_expected") or ""
        doc, hay = haystack_for(doc_tail(src))
        if not hay:
            # Rows asserting a document is silent about something list no source
            # document at all - which is most of them, and exactly the rows worth
            # checking. Resolve the document from the question instead; scope
            # resolution costs no model call.
            doc, hay = haystack_from_question(r.get("question") or "")

        if not hay:
            verdict, detail = "undecidable", "named document not found in the corpus"
        elif _RX_NEGATIVE.search(expected):
            # The expected answer denies something. Check whether the document
            # in fact discusses it — this is the shape of every known-bad row.
            subj = denied_subject(expected)
            if not subj:
                verdict, detail = "undecidable", "denial with no checkable subject"
            elif clause_type_present(doc, subj):
                verdict, detail = ("disagrees-negative",
                                   "expected says the document does not address %r, but "
                                   "a clause of that type is recorded for it" % subj)
            elif subj in hay:
                verdict, detail = ("disagrees-negative",
                                   "expected says the document does not address %r, "
                                   "but that wording is in the document" % subj)
            else:
                verdict, detail = "agrees", "denial holds: %r absent" % subj
        else:
            facts = sorted({norm(m.group(0)) for m in _RX_FACT.finditer(expected)})
            if not facts:
                verdict, detail = "undecidable", "expected answer states no checkable figure"
            else:
                absent = [f for f in facts if f not in hay]
                if not absent:
                    verdict, detail = "agrees", "%d/%d figures present" % (len(facts), len(facts))
                else:
                    verdict, detail = ("disagrees-positive",
                                       "%d of %d figures absent from the document: %s"
                                       % (len(absent), len(facts), "; ".join(absent[:4])))
        counts[verdict] = counts.get(verdict, 0) + 1
        out.append({"n": r["n"], "verdict": verdict, "detail": detail,
                    "question": r["question"], "expected": expected,
                    "source_expected": src, "resolved_doc": doc,
                    "graded_as": r.get("g_verdict")})

    io.open(os.path.join(S, "testset_audit.json"), "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))

    print("audited %d rows" % len(out))
    for k in ("agrees", "disagrees-negative", "disagrees-positive", "undecidable"):
        if counts.get(k):
            print("  %-20s %3d" % (k, counts[k]))
    bad = [o for o in out if o["verdict"].startswith("disagrees")]
    print()
    print("--- %d rows where the test set contradicts the document ---" % len(bad))
    for o in bad:
        print("[%3d] %s   (graded %s)" % (o["n"], o["verdict"], o["graded_as"]))
        print("      Q: %s" % o["question"][:110])
        print("      EXPECTED: %s" % o["expected"][:120])
        print("      %s" % o["detail"][:150])


if __name__ == "__main__":
    main()
