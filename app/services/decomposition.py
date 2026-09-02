"""Query decomposition — a compound question answered in parts, after a decline.

Phase 6, and last on purpose. This sits in front of questions that already
work, including the compound party-pair questions this project spent real
effort getting right. A decomposer that splits *"the liability cap in the
agreement between Apex Meridian and Ironvane"* hands each half to a resolver
that no longer has the party pair to work with, and breaks the exact capability
it was meant to extend.

So it is constrained three ways, and every one of them is load-bearing:

1. **It runs only on the decline path.** When the whole-question route
   resolves, nothing here executes. It can turn "not addressed in the provided
   context" into an answer; it can never turn one answer into a different one.
2. **It never splits a party pair.** "between X and Y", and any "and" sitting
   between two capitalised names, are vetoed outright.
3. **Sub-questions are routed only through the zero-LLM structured paths** —
   counting, structural lookups, corpus analytics, the Calculation Agent. A
   decomposition that fanned out into retrieval would multiply the cost of the
   questions that are already the most expensive ones to get wrong.

Set ``ENABLE_QUERY_DECOMPOSITION=0`` to switch it off entirely.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_MAX_SUBQUESTIONS = 4


def enabled() -> bool:
    return (os.getenv("ENABLE_QUERY_DECOMPOSITION", "1").strip().lower()
            not in ("0", "false", "no", "off"))


# --------------------------------------------------------------------------
# Recognising a decline.
# --------------------------------------------------------------------------

# Deliberately the same shapes the regression harness scores an abstention by,
# so what this module treats as "the pipeline gave up" and what the test suite
# treats as an abstention cannot drift apart.
_RX_DECLINE = re.compile(
    r"\bnot\s+(?:established|addressed|present|stated|specified|mentioned|covered|"
    r"included|found)\b"
    r"|\bdo(?:es)?\s+not[,]?\s+(?:[\w,]+\s+){0,2}?(?:contain|appear|include|address|specify|"
    r"state|impose|label|define|designate|name|identify|provide|require|mention|"
    r"refer|set\s+out|establish)\b"
    r"|\bno\s+(?:such\s+)?(?:clause|provision|restriction|term|fee|cap|information)\b"
    r"|\bcannot\s+be\s+determined\b"
    r"|\bis\s+silent\s+(?:on|as\s+to)\b"
    r"|\bnothing\s+in\s+the\s+(?:document|agreement|excerpts?)\b"
    r"|\bunable\s+to\s+(?:locate|find|answer)\b"
    r"|\bcould\s+not\s+(?:find|locate|identify)\b",
    re.IGNORECASE,
)

_RX_BRACKETED = re.compile(r"\[[^\]]{0,400}\]")


def looks_declined(payload: dict) -> bool:
    """Whether the finished payload is a decline rather than an answer.

    An empty body counts. A corpus that never discusses the subject gives the
    answer model no material to write from and no absent topic to declare, so
    it returns a blank body rather than a refusal — the same failure with none
    of the wording that would identify it.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("not_covered"):
        return True
    body = _RX_BRACKETED.sub("", payload.get("answer") or "").strip()
    if not body:
        return True
    # A long answer that merely contains one of these phrases in passing is an
    # answer. Only a short one is the whole reply being a refusal.
    return bool(_RX_DECLINE.search(body)) and len(body) < 1200


# --------------------------------------------------------------------------
# Splitting.
# --------------------------------------------------------------------------

# "between X and Y" is one party pair, never two sub-questions. Checked against
# the raw question before any split is attempted.
_RX_PARTY_PAIR = re.compile(
    r"\bbetween\s+[A-Z][\w'&.\-]*(?:\s+[\w'&.\-]+)*\s+and\s+[A-Z]", re.IGNORECASE)
# An "and" joining two capitalised names is a name conjunction, not a clause
# conjunction: "Tata Sons and Tata Motors", "Apex Meridian and Ironvane".
_RX_NAME_AND = re.compile(r"[A-Z][\w'&.\-]{2,}\s+and\s+[A-Z][\w'&.\-]{2,}")

# A fragment has to be a question in its own right to be worth routing.
_RX_PREDICATE = re.compile(
    r"\b(?:what|which|who|when|where|how|does|do|did|is|are|was|were|list|show|"
    r"find|count|name|identify|total|sum)\b", re.IGNORECASE)

# The scope a later fragment inherits from the first: "under the X Agreement",
# "in the Y SOW", "of the Z MSA".
_RX_SCOPE_PHRASE = re.compile(
    r"\b(?:under|in|of|for)\s+(?:the\s+)?"
    r"((?:[A-Z][\w'&.\-]*|[A-Z0-9][\w\-]*\d[\w\-]*)(?:\s+[\w'&.\-]+){0,7})",
)

_RX_SPLIT_AND = re.compile(r",\s+and\s+|\s+and\s+also\s+|\s+as\s+well\s+as\s+", re.IGNORECASE)

# A fragment that asks about the corpus is already self-contained, and must not
# inherit the first fragment's document. "…and how many SLAs do we have" became
# "how many SLAs do we have of CND-TOR-SOW-2026-001", which is not a question
# anyone asked and which the counting path would answer as one document.
_RX_CORPUS_SCOPE = re.compile(
    r"^\s*(?:and\s+|also\s+){0,2}(?:how\s+many|number\s+of|count\b|"
    r"which\s+documents?|what\s+documents?|list\s+(?:all|every)|across\b)",
    re.IGNORECASE)

_RX_LEADING_PREP = re.compile(r"^(?:under|in|of|for)\s+", re.IGNORECASE)


def _scope_phrase(question: str) -> str:
    m = _RX_SCOPE_PHRASE.search(question or "")
    if not m:
        return ""
    phrase = m.group(0).strip().rstrip(".,;:?")
    if len(phrase) <= 8:
        return ""
    # Normalised to "under X" regardless of how the first fragment phrased it:
    # appending the original "of the Castellane MSA" to a second fragment reads
    # as a possessive and changes what the fragment appears to ask.
    return "under " + _RX_LEADING_PREP.sub("", phrase, count=1)


def _viable(fragment: str) -> bool:
    f = (fragment or "").strip()
    return len(f) >= 12 and bool(_RX_PREDICATE.search(f))


def split(question: str) -> list[str]:
    """Self-contained sub-questions, or [] when the question is not compound.

    Returns [] rather than [question] for a single question: an empty list is
    the caller's signal that there is nothing to decompose, and a one-element
    list would send the same question back round a second time.
    """
    q = (question or "").strip()
    if not q:
        return []

    parts: list[str] = []

    # 1. Genuinely separate questions, already punctuated as such.
    if q.count("?") > 1:
        parts = [p.strip() + "?" for p in q.split("?") if p.strip()]
    # 2. Semicolons separate independent clauses by definition.
    elif ";" in q:
        parts = [p.strip() for p in q.split(";") if p.strip()]
    # 3. Conjunctions — the risky one, and the vetoes come first.
    elif not _RX_PARTY_PAIR.search(q):
        candidate = _RX_SPLIT_AND.split(q)
        if len(candidate) > 1:
            # Reject the split if it cut through a name conjunction: the join
            # is part of a party's identity, not a boundary between asks.
            rebuilt_ok = not any(_RX_NAME_AND.search(c) is None and
                                 _RX_NAME_AND.search(q) is not None
                                 for c in candidate[:1])
            if rebuilt_ok:
                parts = [p.strip() for p in candidate if p.strip()]

    parts = [p for p in parts if _viable(p)]
    if len(parts) < 2:
        return []

    # Carry the first fragment's document scope into any later fragment that
    # names no document of its own — "…and what is the notice period" cannot be
    # routed without knowing which agreement it belongs to.
    scope = _scope_phrase(parts[0])
    out = [parts[0]]
    for p in parts[1:]:
        if (scope and not _RX_SCOPE_PHRASE.search(p)
                and not _RX_CORPUS_SCOPE.search(p)):
            p = f"{p.rstrip('?').rstrip()} {scope}?"
        out.append(p)
    return out[:_MAX_SUBQUESTIONS]


# --------------------------------------------------------------------------
# Routing one sub-question through the structured paths only.
# --------------------------------------------------------------------------

def _route(sub: str, session_id: str, wiki_id: str,
           docs: list[str] | None) -> tuple[str, str] | None:
    """(answer body, path name) from a zero-LLM path, or None.

    Order matters and mirrors the router's own: the most specific detector
    first. Every path here is SQL or Python; none reaches retrieval, so a
    decomposition that fans out to four sub-questions still costs nothing.
    """
    from services import calculation, intent_agent as ia

    try:
        if calculation.is_calculation_query(sub):
            p = calculation.answer(sub, wiki_id, session_id, docs)
            if p:
                return p["answer"], "calculation"
    except Exception as e:
        logger.warning("[DECOMP] calculation leg failed: %s", e)

    try:
        kind = ia._is_structural_query(sub)
        if kind:
            p = ia._structural_answer(kind, sub, session_id)
            if p:
                return p["answer"], f"structural:{kind}"
    except Exception as e:
        logger.warning("[DECOMP] structural leg failed: %s", e)

    try:
        akind = ia._is_analytics_query(sub)
        if akind:
            p = ia._analytics_answer(akind, sub, session_id, wiki_id)
            if p:
                return p["answer"], f"analytics:{akind}"
    except Exception as e:
        logger.warning("[DECOMP] analytics leg failed: %s", e)

    return None


def rescue(question: str, session_id: str, wiki_id: str,
           docs: list[str] | None = None,
           original_answer: str = "") -> dict | None:
    """Answer the answerable parts of a compound question, or None.

    None means leave the original decline exactly as it is — which is the
    common case, and the right one. A partial answer is only offered when at
    least one sub-question resolved through a structured path; the parts that
    did not resolve are named rather than quietly dropped, because a compound
    question half-answered without saying so is worse than one declined whole.
    """
    if not enabled():
        return None
    subs = split(question)
    if not subs:
        return None

    answered: list[tuple[str, str, str]] = []
    unanswered: list[str] = []
    for sub in subs:
        got = _route(sub, session_id, wiki_id, docs)
        if got:
            answered.append((sub, got[0], got[1]))
        else:
            unanswered.append(sub)

    if not answered:
        return None

    logger.info("[DECOMP] rescued %d of %d parts of a declined question",
                len(answered), len(subs))

    lines = [f"**This question asks {len(subs)} things. "
             f"{len(answered)} of them can be answered from structured records.**", ""]
    for i, (sub, body, path) in enumerate(answered, 1):
        lines.append(f"### {i}. {sub.strip()}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    if unanswered:
        lines.append("---")
        lines.append("")
        lines.append("**Still unanswered:**")
        for sub in unanswered:
            lines.append(f"- {sub.strip()}")
        lines.append("")
        # The original decline is carried through verbatim, not summarised away.
        # A correct abstention explains WHY a provision is absent, and that
        # explanation is often the most useful part of the reply - replacing it
        # with a bare "not found" would lose the reasoning the pipeline already
        # did and paid for.
        body = (original_answer or "").strip()
        if body:
            lines.append("What the documents in scope did say about the question "
                         "as a whole:")
            lines.append("")
            lines.append(body)
        else:
            lines.append("These parts were not found in the documents in scope.")

    from services.intent_agent import _canned_payload
    payload = _canned_payload("\n".join(lines), "Decomposed",
                              "query-decomposition")
    payload["meta_answer"] = False
    payload["decomposed"] = [{"question": s, "path": p} for s, _, p in answered]
    return payload
