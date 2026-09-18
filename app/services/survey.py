"""Population surveys: questions about every document of one type at once.

"Which of our data processing agreements is the oldest?" and "our standard
requires X; do our data processing agreements meet it?" are not questions
about any one document, and retrieval answers them from whichever handful of
pages it fetched. Measured on the golden set, both shapes came back as "Needs
clarification", and the playbook path declined the second because it compares
one document against the house position, not a population.

Both are computed here from the index instead:

- extremes: the population's effective dates, oldest and most recent.
- standard: for every document in the population, the clause that best
  matches the standard's own wording, the value that clause states (a
  number with a unit, or a place), and whether it meets the standard.

Clause matching is on the clause text, not on clause_type: a clause type is
only as good as the extractor's labelling, and a whole family of
data-location clauses was stored untyped. A document with no matching clause
is reported as such, never as non-compliant - not finding a clause is not
evidence the document lacks one.

"Our" is read through config.HOUSE_PARTIES (the house's own entities). With
none configured, "our" means every document of the type, and the answer says
so.
"""
from __future__ import annotations

import logging
import re

import config
from services import db

logger = logging.getLogger(__name__)

_POPULATION_CAP = 60

_RX_OUR = re.compile(r"\b(?:our|we|us)\b", re.I)
_RX_EXTREME = re.compile(
    r"\b(?:oldest|earliest|most\s+recent|newest|latest|"
    r"first\s+(?:signed|executed|dated)|last\s+(?:signed|executed|dated))\b", re.I)
_RX_POPULATION = re.compile(
    r"\b(?:which|what)\s+(?:of\s+)?(?:our|the|all|these)\b|\b(?:of|among|across)\s+(?:our|all|the|these)\b",
    re.I)
_RX_STANDARD = re.compile(
    r"\b(?:standard|playbook|policy|house\s+position|guideline|benchmark|"
    r"requires?|requirement|must|should)\b", re.I)
_RX_COMPLY = re.compile(
    r"\b(?:do|does|are|is)\s+(?:our|all|the|these|any)\b[^.?]*?\b(?:meet|meets|comply|complies|"
    r"compliant|conform|conforms|satisfy|satisfies|align|aligned|in\s+line)\b", re.I)

_RX_ANOMALY = re.compile(
    r"\b(?:anomal\w*|unusual|outliers?|odd|inconsisten\w*|irregular\w*|deviat\w*|"
    r"stand\s+out|out\s+of\s+line)\b", re.I)

_RX_QTY = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\(\w+\)\s*)?(business\s+days?|calendar\s+days?|working\s+days?|"
    r"hours?|days?|weeks?|months?|years?)\b", re.I)
_RX_MIN_DIRECTION = re.compile(
    r"\b(?:at\s+least|not\s+less\s+than|no\s+(?:less|fewer)\s+than|minimum|or\s+more)\b", re.I)
_RX_STD_PLACE = re.compile(
    r"\b(?:in|within|inside)\s+((?:the\s+)?[A-Z][\w-]*(?:\s+(?:of\s+)?[A-Z][\w-]*)*)")
_RX_CLAUSE_PLACE = re.compile(
    r"\b(?:outside(?:\s+of)?|only\s+(?:in|within)|solely\s+(?:in|within)|located\s+in)\s+"
    r"((?:the\s+)?[A-Z][\w-]*(?:\s+(?:of\s+)?[A-Z][\w-]*)*)")

_STOP = frozenset("""
the a an and or of to in on for with by at from as is are be been our we us
their its this that these those any all each every do does did not no must
should shall will would may can standard playbook policy requires require
requirement requirements position house meet meets comply complies compliant
conform satisfy align within least than more less data processing
agreement agreements document documents contract contracts
""".split())

_UNIT_HOURS = {"hour": 1, "day": 24, "week": 168}


def survey_kind(question: str) -> str:
    """'extremes' | 'standard' | '' for a population question over one type."""
    from services import intent_agent as _ia
    q = question or ""
    label, pats = _ia._doctype_from_question(q)
    if not pats:
        return ""
    if _RX_EXTREME.search(q) and _RX_POPULATION.search(q):
        return "extremes"
    if _RX_STANDARD.search(q) and _RX_COMPLY.search(q) and _standard_value(q):
        return "standard"
    if _RX_ANOMALY.search(q) and (_RX_OUR.search(q) or _RX_POPULATION.search(q)):
        return "anomalies"
    return ""


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def population(wiki_id: str, session_id: str, doc_type_patterns: list,
               question: str) -> tuple[list[dict], str]:
    """(documents, sentence saying which documents were counted)."""
    from sqlalchemy import text
    params = {"w": wiki_id, "s": session_id, "lim": _POPULATION_CAP + 1}
    ors = []
    for i, p in enumerate(doc_type_patterns):
        ors.append(f"{db._PRIMARY_DOC_TYPE_SQL} ILIKE :dt{i}")
        params[f"dt{i}"] = f"%{p}%"
    where = f"d.wiki_id = :w AND d.session_id = :s AND ({' OR '.join(ors)})"
    house = list(config.HOUSE_PARTIES) if _RX_OUR.search(question or "") else []
    if house:
        hs = []
        for i, h in enumerate(house):
            hs.append(f"pp.name ILIKE :h{i}")
            params[f"h{i}"] = f"%{h}%"
        where += (" AND EXISTS (SELECT 1 FROM jsonb_array_elements_text("
                  "COALESCE(d.parties, '[]'::jsonb)) AS pp(name) WHERE "
                  + " OR ".join(hs) + ")")
    with db.get_engine().connect() as conn:
        rows = conn.execute(text(
            f"SELECT d.source_doc, d.effective_date, d.parties FROM documents d "
            f"WHERE {where} ORDER BY d.effective_date NULLS LAST, d.source_doc LIMIT :lim"),
            params).fetchall()
    docs = [{"source_doc": r[0],
             "effective_date": str(r[1]) if r[1] else None,
             "parties": [p for p in (r[2] or []) if isinstance(p, str)]}
            for r in rows[:_POPULATION_CAP]]
    if house:
        rule = (f"Counted over the {len(docs)} {{noun}} in the index that a house "
                f"entity is party to.")
    elif _RX_OUR.search(question or ""):
        rule = (f"No house entities are configured, so \"our\" is read as every "
                f"{{noun}} in the index: {len(docs)} of them.")
    else:
        rule = f"Counted over all {len(docs)} {{noun}} in the index."
    if len(rows) > _POPULATION_CAP:
        rule += f" Only the first {_POPULATION_CAP} by date are shown."
    return docs, rule


def _doc_label(doc: dict) -> str:
    if doc["parties"]:
        return " / ".join(doc["parties"][:2])
    name = doc["source_doc"]
    return re.sub(r"^[0-9a-f-]{36}_", "", name)


# ---------------------------------------------------------------------------
# Extremes
# ---------------------------------------------------------------------------

def answer_extremes(question: str, wiki_id: str, session_id: str) -> str | None:
    from services import intent_agent as _ia
    label, pats = _ia._doctype_from_question(question)
    docs, rule = population(wiki_id, session_id, pats, question)
    dated = [d for d in docs if d["effective_date"]]
    if not dated:
        return None
    noun = f"{label}s" if label else "documents"
    oldest, newest = dated[0], dated[-1]
    lines = [f"**Oldest: {_doc_label(oldest)}, dated {oldest['effective_date']}. "
             f"Most recent: {_doc_label(newest)}, dated {newest['effective_date']}.**",
             "", "| Effective date | Parties | Document |", "| --- | --- | --- |"]
    for d in dated:
        lines.append(f"| {d['effective_date']} | {' / '.join(d['parties'][:2]) or '—'} "
                     f"| {re.sub(r'^[0-9a-f-]{36}_', '', d['source_doc'])} |")
    lines += ["", rule.format(noun=noun)]
    undated = len(docs) - len(dated)
    if undated:
        lines.append(f"{undated} of them record no effective date and are not ranked.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standard
# ---------------------------------------------------------------------------

def _standard_sentence(question: str) -> str:
    parts = re.split(r"(?<=[.?!])\s+", question or "")
    for p in parts:
        if _RX_STANDARD.search(p) and (_RX_QTY.search(p) or _RX_STD_PLACE.search(p)):
            return p
    return ""


def _standard_value(question: str):
    """('qty', number, unit, direction) | ('place', place) | None."""
    s = _standard_sentence(question)
    if not s:
        return None
    m = _RX_QTY.search(s)
    if m:
        return ("qty", float(m.group(1)), _unit(m.group(2)),
                "min" if _RX_MIN_DIRECTION.search(s) else "max")
    m = _RX_STD_PLACE.search(s)
    if m:
        return ("place", re.sub(r"^the\s+", "", m.group(1), flags=re.I))
    return None


def _unit(raw: str) -> str:
    u = raw.lower().split()[-1].rstrip("s")
    if raw.lower().startswith(("business", "working")):
        return "business day"
    return u


def _in_hours(n: float, unit: str):
    return n * _UNIT_HOURS[unit] if unit in _UNIT_HOURS else None


def _tokens(s: str) -> set:
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", (s or "").lower())
            if w not in _STOP}


def _best_clause(clauses: list[str], topic: set, spec) -> str | None:
    best, best_score = None, 0
    for t in clauses:
        score = len(topic & _tokens(t))
        if spec[0] == "qty" and any(_unit(m.group(2)) == spec[2] or
                                    (_in_hours(1, _unit(m.group(2))) and _in_hours(1, spec[2]))
                                    for m in _RX_QTY.finditer(t)):
            score += 2
        if spec[0] == "place" and _RX_CLAUSE_PLACE.search(t):
            score += 2
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 3 else None


def _excerpt(text: str, i: int, j: int, width: int = 170) -> str:
    lo = max(0, i - (width - (j - i)) // 2)
    hi = min(len(text), lo + width)
    if lo:
        sp = text.find(" ", lo, i)
        lo = sp + 1 if sp >= 0 else lo
    if hi < len(text):
        sp = text.rfind(" ", j, hi)
        hi = sp if sp >= 0 else hi
    return ("…" if lo else "") + text[lo:hi].strip() + ("…" if hi < len(text) else "")


def _judge(clause: str, spec):
    """(what the clause says, verdict, excerpt) for one document's clause."""
    if spec[0] == "qty":
        _, std, std_unit, direction = spec
        for m in _RX_QTY.finditer(clause):
            unit = _unit(m.group(2))
            n = float(m.group(1))
            a, b = _in_hours(n, unit), _in_hours(std, std_unit)
            if unit == std_unit:
                a, b = n, std
            if a is None or b is None:
                continue
            ok = a <= b if direction == "max" else a >= b
            said = f"{m.group(1)} {m.group(2)}"
            return said, ("meets" if ok else "does not meet"), _excerpt(clause, m.start(), m.end())
        return None, "unclear", _excerpt(clause, 0, 0)
    place = spec[1]
    m = _RX_CLAUSE_PLACE.search(clause)
    if not m:
        return None, "unclear", _excerpt(clause, 0, 0)
    region = re.sub(r"^the\s+", "", m.group(1), flags=re.I)
    ok = place.lower() in region.lower() or region.lower() in place.lower()
    return region, ("meets" if ok else "does not meet"), _excerpt(clause, m.start(), m.end())


def answer_standard(question: str, wiki_id: str, session_id: str) -> str | None:
    from sqlalchemy import text
    from services import intent_agent as _ia
    spec = _standard_value(question)
    if not spec:
        return None
    label, pats = _ia._doctype_from_question(question)
    docs, rule = population(wiki_id, session_id, pats, question)
    if not docs:
        return None
    noun = f"{label}s" if label else "documents"
    std_sentence = _standard_sentence(question)
    topic = _tokens(std_sentence)
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT source_doc, verbatim_text FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND source_doc = ANY(:d)
              AND COALESCE(review_status, '') <> 'rejected'
        """), {"w": wiki_id, "s": session_id,
               "d": [d["source_doc"] for d in docs]}).fetchall()
    by_doc: dict[str, list[str]] = {}
    for r in rows:
        by_doc.setdefault(r[0], []).append(r[1] or "")

    results = []
    for d in docs:
        clause = _best_clause(by_doc.get(d["source_doc"], []), topic, spec)
        if clause is None:
            results.append((d, None, "no matching clause", None))
            continue
        said, verdict, excerpt = _judge(clause, spec)
        results.append((d, said, verdict, excerpt))

    meets = [r for r in results if r[2] == "meets"]
    fails = [r for r in results if r[2] == "does not meet"]
    other = [r for r in results if r[2] not in ("meets", "does not meet")]
    std_txt = (f"{spec[1]:g} {spec[2]}s" if spec[0] == "qty" else spec[1])
    judged = len(meets) + len(fails)
    if not judged:
        return None
    if not fails:
        head = (f"**Yes, as far as the text shows: all {judged} {noun} with a clause "
                f"on this point meet the standard ({std_txt}).**")
    elif not meets:
        head = (f"**No: none of the {judged} {noun} with a clause on this point meet "
                f"the standard ({std_txt}).**")
    else:
        head = (f"**Not consistently: {len(meets)} of the {judged} {noun} with a clause "
                f"on this point meet the standard ({std_txt}); {len(fails)} do not.**")
    lines = [head, "", "| Agreement | Effective date | States | Meets standard? | Clause |",
             "| --- | --- | --- | --- | --- |"]
    order = {"does not meet": 0, "meets": 1, "unclear": 2, "no matching clause": 3}
    for d, said, verdict, excerpt in sorted(results, key=lambda r: order.get(r[2], 9)):
        lines.append(f"| {_doc_label(d)} | {d['effective_date'] or '—'} | {said or '—'} "
                     f"| {verdict} | {excerpt or '—'} |")
    lines += ["", rule.format(noun=noun)]
    if other:
        lines.append(f"{len(other)} of them have no clause this could read a value from. "
                     f"That is not evidence they lack one; the clause may be worded "
                     f"differently or sit in a schedule the extractor did not cut.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anomalies: one clause template, different values
# ---------------------------------------------------------------------------
# Documents of one type are usually drafted from a few templates. The
# anomalies a reviewer can check mechanically are the places where documents
# share a clause's wording but state a different value in it: a 24-hour
# breach window where the template usually says 48, a data-location
# restriction to one country where the others name a region. Detected by
# masking names, numbers and places out of each clause's operative wording,
# grouping documents on what is left, and reporting any group whose values
# differ.
_RX_OPERATIVE = re.compile(r"\b(?:shall|must|may|agrees?\s+to|undertakes?\s+to)\b", re.I)
_RX_NAME_RUN = re.compile(r"(?:\b[A-Z][\w&.'()-]*)(?:\s+(?:[A-Z][\w&.'()-]*|of|and|&))*")
_ANOMALY_MIN_GROUP = 3


def _template_key(clause: str) -> str | None:
    t = " ".join((clause or "").split())
    m = _RX_OPERATIVE.search(t)
    if not m:
        return None
    t = t[m.start():m.start() + 160]
    t = _RX_CLAUSE_PLACE.sub(lambda x: x.group(0).split()[0] + " <place>", t)
    t = _RX_QTY.sub("<n> <unit>", t)
    t = _RX_NAME_RUN.sub("<name>", t)
    t = re.sub(r"\d+", "<n>", t)
    return t.lower()[:110]


def _clause_value(clause: str) -> str | None:
    m = _RX_QTY.search(clause)
    if m:
        return f"{m.group(1)} {m.group(2).lower()}"
    m = _RX_CLAUSE_PLACE.search(clause)
    if m:
        return re.sub(r"^the\s+", "", m.group(1), flags=re.I)
    return None


def answer_anomalies(question: str, wiki_id: str, session_id: str) -> str | None:
    from sqlalchemy import text
    from services import intent_agent as _ia
    label, pats = _ia._doctype_from_question(question)
    docs, rule = population(wiki_id, session_id, pats, question)
    if len(docs) < _ANOMALY_MIN_GROUP:
        return None
    noun = f"{label}s" if label else "documents"
    by_name = {d["source_doc"]: d for d in docs}
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT source_doc, verbatim_text FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND source_doc = ANY(:d)
              AND COALESCE(review_status, '') <> 'rejected'
        """), {"w": wiki_id, "s": session_id, "d": list(by_name)}).fetchall()
    groups: dict[str, dict] = {}
    for src, clause in rows:
        key, val = _template_key(clause), _clause_value(clause or "")
        if not key or not val:
            continue
        g = groups.setdefault(key, {"example": clause, "values": {}})
        g["values"].setdefault(val, set()).add(src)
    findings = []
    for key, g in groups.items():
        # One value per document: a document restating its own figure in two
        # clauses of the same shape is not a split.
        docs_in = set().union(*g["values"].values())
        if len(docs_in) < _ANOMALY_MIN_GROUP or len(g["values"]) < 2:
            continue
        findings.append((len(docs_in), g))
    if not findings:
        return None
    findings.sort(key=lambda x: -x[0])
    lines = [f"**{len(findings)} clause(s) share one wording across the {noun} but "
             f"state different values in it.**", ""]
    for n, g in findings[:8]:
        ex = " ".join(g["example"].split())
        m = _RX_OPERATIVE.search(ex)
        ex = ex[m.start():m.start() + 140] if m else ex[:140]
        lines.append(f"**“…{ex}…”** ({n} documents)")
        vals = sorted(g["values"].items(), key=lambda kv: -len(kv[1]))
        for i, (v, srcs) in enumerate(vals):
            who = ""
            if i and len(srcs) <= 5:
                who = ": " + "; ".join(_doc_label(by_name[s_]) for s_ in sorted(srcs))
            lines.append(f"- {v}: {len(srcs)} document(s){who}")
        lines.append("")
    lines.append(rule.format(noun=noun))
    lines.append("Only value differences inside a shared clause wording are detected "
                 "here. A clause that is unusual in substance rather than in a figure "
                 "(an obligation running the other way, say) needs a reading, not a count.")
    return "\n".join(lines)


def answer(kind: str, question: str, wiki_id: str, session_id: str) -> str | None:
    if kind == "anomalies":
        return answer_anomalies(question, wiki_id, session_id)
    if kind == "extremes":
        return answer_extremes(question, wiki_id, session_id)
    if kind == "standard":
        return answer_standard(question, wiki_id, session_id)
    return None
