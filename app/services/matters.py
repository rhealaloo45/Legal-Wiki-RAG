"""A party's litigation matters and how each came out, from litigation_facts.

"X appears in more than one matter. What are they, and did X win?" has an
exact answer in the typed litigation records: every case number the party is
recorded against, its role in each, and the disposition of each final order.
Answered by retrieval instead, the same question identified both matters and
then hedged on the outcome of one ("the corpus does not record a final
judgment") while the index held two final orders for it.

The records are reported as they stand. Where two outcome records for one
matter disagree (a draft judgment and the final order, say), both are shown
with their documents rather than one being picked, because which one governs
is a legal question the index cannot settle.
"""
from __future__ import annotations

import logging
import re

from services import db

logger = logging.getLogger(__name__)

_RX_MATTERS = re.compile(
    r"\b(?:matters?|cases?|disputes?|proceedings?|litigations?|suits?|lawsuits?|petitions?)\b",
    re.I)
_RX_MATTERS_ASK = re.compile(
    r"\b(?:appears?\s+in|involved\s+in|part(?:y|ies)\s+to|what\s+are\s+they|which|list|"
    r"how\s+many|did\s+\w+(?:\s+\w+){0,3}\s+(?:win|lose|prevail|succeed)|outcomes?|results?)\b",
    re.I)
_SUFFIX = (r"(?:Private\s+Limited|Pvt\.?\s*Ltd\.?|Pte\.?\s*Ltd\.?|Limited|Ltd\.?|LLP|LLC|"
           r"Inc\.?|Corp(?:oration)?\.?|PLC|plc|GmbH|FZE|FZ-LLC|AG|S\.A\.|N\.V\.)")
_RX_CORP_NAME = re.compile(
    r"((?:[A-Z][\w&.'-]*\s+){1,6}?" + _SUFFIX + r")(?=[\s,.;:?]|$)")

_WON_FOR_PLAINTIFF = re.compile(r"\b(?:allowed|decreed|granted|upheld|succeeds?)\b", re.I)
_PARTLY = re.compile(r"\bpart(?:ly|ially)\b", re.I)
_WON_FOR_DEFENDANT = re.compile(r"\b(?:dismissed|rejected|struck\s+out|quashed|set\s+aside)\b", re.I)


# Capitalised only because they open the sentence or a clause, not name words.
_LEADING_NON_NAME = re.compile(
    r"^(?:(?:Is|Are|Was|Were|Does|Do|Did|Has|Have|Had|Which|What|Where|When|Who|How|"
    r"Can|Could|Should|Would|Will|List|Show|Name|Give|The|In|Under|For|Of|And|Or|"
    r"Between|With|Against|Whether)\s+)+")


def party_in_question(question: str) -> str | None:
    m = _RX_CORP_NAME.search(question or "")
    if not m:
        return None
    name = _LEADING_NON_NAME.sub("", m.group(1).strip())
    return name if len(name.split()) >= 2 else None


def is_matters_query(question: str) -> bool:
    q = question or ""
    return bool(_RX_MATTERS.search(q) and _RX_MATTERS_ASK.search(q) and party_in_question(q))


def _names(v) -> list[str]:
    return [x for x in (v or []) if isinstance(x, str)]


def _outcome_for(role: str | None, disposition: str) -> str:
    d = disposition or ""
    if not role or not d:
        return ""
    if _PARTLY.search(d) and _WON_FOR_PLAINTIFF.search(d):
        return "partly lost" if role == "defendant" else "partly won"
    if _WON_FOR_PLAINTIFF.search(d):
        return "won" if role == "plaintiff" else "lost"
    if _WON_FOR_DEFENDANT.search(d):
        return "lost" if role == "plaintiff" else "won"
    return ""


def party_matters(wiki_id: str, session_id: str, party: str) -> list[dict]:
    from sqlalchemy import text
    like = f"%{party}%"
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT source_doc, case_number, court, plaintiffs, defendants,
                   disposition, relief_granted, procedural_posture
              FROM litigation_facts
             WHERE wiki_id = :w AND session_id = :s
               AND (EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(plaintiffs, '[]'::jsonb)) p(n) WHERE p.n ILIKE :like)
                 OR EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(defendants, '[]'::jsonb)) d(n) WHERE d.n ILIKE :like)
                 OR source_doc ILIKE :like)
        """), {"w": wiki_id, "s": session_id, "like": like}).fetchall()
        keys = {r[1] for r in rows if r[1]}
        if keys:
            # Every record filed under those case numbers, not only the ones
            # whose own parties field names the party: an outcome record
            # often carries the case number and the order but no party list.
            rows = conn.execute(text("""
                SELECT source_doc, case_number, court, plaintiffs, defendants,
                       disposition, relief_granted, procedural_posture
                  FROM litigation_facts
                 WHERE wiki_id = :w AND session_id = :s AND case_number = ANY(:k)
            """), {"w": wiki_id, "s": session_id, "k": sorted(keys)}).fetchall()
    matters: dict[str, dict] = {}
    low = party.lower()
    for src, case, court, pl, df, disp, relief, posture in rows:
        if not case:
            continue
        m = matters.setdefault(case, {"case_number": case, "court": None, "plaintiffs": [],
                                      "defendants": [], "role": None, "outcomes": [],
                                      "documents": set()})
        m["documents"].add(src)
        m["court"] = m["court"] or court
        if _names(pl) and not m["plaintiffs"]:
            m["plaintiffs"] = _names(pl)
        if _names(df) and not m["defendants"]:
            m["defendants"] = _names(df)
        if any(low in n.lower() for n in _names(pl)):
            m["role"] = "plaintiff"
        elif any(low in n.lower() for n in _names(df)):
            m["role"] = m["role"] or "defendant"
        if (disp or "").strip():
            m["outcomes"].append({"source_doc": src, "disposition": disp.strip(),
                                  "relief": (relief or "").strip(),
                                  "posture": (posture or "").strip()})
    return sorted(matters.values(), key=lambda m: m["case_number"])


def answer(question: str, wiki_id: str, session_id: str, display=None) -> dict | None:
    """{"text", "documents"} or None when the index holds no matter for the party."""
    party = party_in_question(question)
    if not party:
        return None
    ms = [m for m in party_matters(wiki_id, session_id, party) if m["role"]]
    if not ms:
        return None
    show = display or (lambda d: re.sub(r"^[0-9a-f-]{36}_", "", d))
    lines = [f"**{party} is recorded as a party in {len(ms)} matter(s).**", ""]
    for m in ms:
        caption = ""
        if m["plaintiffs"] and m["defendants"]:
            caption = f" — {' & '.join(m['plaintiffs'])} v. {' & '.join(m['defendants'])}"
        lines.append(f"**{m['case_number']}**{caption}"
                     + (f" ({m['court']})" if m["court"] else ""))
        lines.append(f"- {party} is the {m['role']}.")
        if not m["outcomes"]:
            lines.append("- No outcome is recorded for this matter in the index.")
        for o in m["outcomes"]:
            v = _outcome_for(m["role"], o["disposition"])
            relief = f" {o['relief']}" if o["relief"] else ""
            lines.append(f"- {o['disposition']}.{relief} "
                         f"({o['posture'] or 'outcome record'}: {show(o['source_doc'])})"
                         + (f" — for {party}: **{v}**" if v else ""))
        if len(m["outcomes"]) > 1 and len({o["disposition"].lower() for o in m["outcomes"]}) > 1:
            lines.append("- These outcome records differ. Both are shown with their "
                         "documents; which one governs (for instance a draft against "
                         "the final order) is not something the index can settle.")
        lines.append("")
    lines.append("Read from the typed litigation records for every case number the "
                 "party is recorded against, not from the pages a search returned.")
    docs = sorted({d for m in ms for d in m["documents"]})
    return {"text": "\n".join(lines), "documents": docs}
