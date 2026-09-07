"""Six independent confidence scores, of which the lowest governs.

The system used to report one number: the generating model's own self-report,
which correlates +0.13 with an independent judge — that is, not at all. Beside
it sat a grounding score measuring whether the answer was faithful to the text
it was given. Both were honest and both were useless in the same way, which a
live thread showed exactly: "Confidence: 95% · Grounding: 100%" on an answer
drawn from a different company's agreement. The answer WAS faithful to that
text. Neither number could see that the text was the wrong text.

The fix is not a better single number. It is to score several things that can
independently be wrong, and let the worst one govern — because an answer is
only as good as its weakest link, and averaging hides exactly the link that
broke. Each dimension below is computed from evidence the pipeline already
produces; none of them costs a token.

    evidence      are the answer's quotes actually in the cited document
    retrieval     how confidently was the document set resolved at all
    authority     is the answering document the KIND of thing that was asked about
    completeness  did the answer cover what was asked, and was context dropped
    reasoning     did the deterministic checks flag the answer's own claims
    temporal      does a date the question names match the document answered from

`authority` is the one that would have caught the thread above: the scope
resolved by a party-name content match, which is the weakest resolver there is,
and nothing carried that weakness through to what the reader saw.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# A dimension with nothing to say scores this rather than 100. Absence of
# evidence is not evidence of correctness, and a dimension that cannot see
# anything should not be allowed to certify the answer — but it must not
# govern either, so it sits above the thresholds that matter.
NEUTRAL = 75

# How much each scope resolver is worth. A resolver that pinned the document
# from something the question stated outright is trustworthy; one that fell
# back to searching page CONTENT for a party name is the weakest link in the
# system and is scored like it. Prefix-matched, so the correction suffixes
# resolve_scope appends ("party-pair-family-corrected-doctype") inherit the
# base resolver's score.
_SCOPE_TRUST = (
    ("file", 98),                     # the question named the file
    ("display-name", 96),
    ("named-instrument-single", 94),
    ("effective-date", 92),
    ("date", 90),
    ("party-pair", 90),               # both sides named, one matter
    ("carryover", 82),                # inherited from a turn that resolved
    ("entity", 88),
    ("party-multi-doctype", 74),
    ("party-multi", 68),
    ("party", 60),                    # one party name, possibly shared
    ("family", 58),
    ("collection", 85),
    # The Enquiry Agent's referent, and only where every deterministic resolver
    # returned nothing. It is validated - the description had to resolve to a
    # real, small document set - but it is a model's reading of the
    # conversation, so it is scored below every resolver that read the question
    # itself and above the corpus default it replaces.
    ("enquiry-referent", 66),
    ("corpus", 45),
    ("default", 40),                  # nothing resolved; the corpus answered
    ("error", 30),
)

# Doc-type words a question can name, mapped to what documents.doc_type holds.
# Only the unambiguous ones: the point is to catch an SLA question answered
# from a judgment, not to adjudicate near-synonyms.
_ASKED_TYPE_RE = re.compile(
    r"\b(nda|non-disclosure(?:\s+agreement)?|sla|service\s+level\s+agreement|"
    r"dpa|data\s+processing\s+agreement|msa|master\s+services\s+agreement|"
    r"sow|statement\s+of\s+work|power\s+of\s+attorney|poa|"
    r"joint\s+venture(?:\s+agreement)?|jva|shareholders?\s+agreement|sha|"
    r"lease(?:\s+deed)?|judgment|judgement|legal\s+opinion|"
    r"consultancy\s+agreement|employment\s+agreement|facility\s+agreement|"
    r"tax\s+deed|closing\s+certificate|escrow(?:\s+agreement)?)\b",
    re.IGNORECASE)

_TYPE_CANON = {
    "nda": "nda", "non-disclosure": "nda", "non-disclosure agreement": "nda",
    "sla": "sla", "service level agreement": "sla",
    "dpa": "dpa", "data processing agreement": "dpa",
    "msa": "msa", "master services agreement": "msa",
    "sow": "sow", "statement of work": "sow",
    "poa": "power of attorney", "power of attorney": "power of attorney",
    "jva": "joint venture", "joint venture": "joint venture",
    "joint venture agreement": "joint venture",
    "sha": "shareholders agreement", "shareholder agreement": "shareholders agreement",
    "shareholders agreement": "shareholders agreement",
    "lease": "lease", "lease deed": "lease",
    "judgment": "judgment", "judgement": "judgment",
    "legal opinion": "legal opinion",
    "consultancy agreement": "consultancy", "employment agreement": "employment",
    "facility agreement": "facility", "tax deed": "tax deed",
    "closing certificate": "closing certificate",
    "escrow": "escrow", "escrow agreement": "escrow",
}

# A four-digit year, or a written date, stated in the question.
_QUESTION_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _scope_score(method: str, scope_docs=None, files_used=None) -> tuple[int, str]:
    """How much the document set can be trusted, and whether it then drifted.

    The resolver's own strength is only half of it. An answer can resolve scope
    correctly and still cite documents outside it, because retrieval runs over
    a wider candidate pool than the scope decision pinned — which is exactly
    the shape of the failure this whole module exists for: a conversation
    carried the right Consultancy Agreement forward and the answer came back
    citing that agreement AND an unrelated Court Judgment.
    """
    m = (method or "").strip().lower()
    base = NEUTRAL
    if m:
        for prefix, sc in _SCOPE_TRUST:
            if m.startswith(prefix):
                base = sc
                break
    why = "scope resolved by %s" % (method or "nothing")

    pinned = {str(d) for d in (scope_docs or []) if d}
    cited = {str(d) for d in (files_used or []) if d}
    if pinned and cited:
        outside = cited - pinned
        if outside:
            # Cited beyond what scope pinned. Scaled by how much of the answer
            # rests outside: one stray document among five is a different thing
            # from an answer built entirely elsewhere.
            share = len(outside) / float(len(cited))
            base = min(base, int(70 - 40 * share))
            why = ("%d of %d cited document(s) lie outside the resolved scope"
                   % (len(outside), len(cited)))
    return base, why


def _evidence_score(citation_check: dict, quality_warning) -> tuple[int, str, bool]:
    """Whether the answer's own quotes survive being checked against the source."""
    cc = citation_check or {}
    total = int(cc.get("total") or 0)
    unver = int(cc.get("unverified") or 0)
    misatt = int(cc.get("misattributed") or 0)
    if total == 0:
        # Nothing quoted. Common and fine for a counting or calculation answer,
        # but it means this dimension verified nothing, so it cannot certify.
        return NEUTRAL, "no quoted spans to check", False
    bad = unver + misatt
    if bad == 0:
        score, why = 97, "%d of %d quoted spans verified" % (total, total)
    else:
        ratio = max(0.0, 1.0 - (bad / float(total)))
        score = int(30 + 60 * ratio)
        why = "%d of %d quoted spans did not verify" % (bad, total)
    if quality_warning:
        # The answer rests on a document whose text did not fully extract.
        score = min(score, 60)
        why += "; answered from a partially unreadable document"
    return score, why, True


def _completeness_score(missing_items, not_covered: bool,
                        pages_omitted: int, topics_missing=None) -> tuple[int, str, bool]:
    """Did the answer cover what was asked.

    Two different failures live here and they are not the same size.

    missing_items is the answer saying these documents do not state something.
    That is honest, and on a question with several parts it is often the
    correct outcome, so it is penalised gently.

    topics_missing is the opposite: the retrieved pages DO use the wording the
    question asked about, and the answer did not cite it. Nothing is absent -
    the answer went past it. That is the "correct but beside the point" failure,
    the one no other dimension can see, and it is penalised harder because the
    material was right there.
    """
    n = len(missing_items or [])
    t = len(topics_missing or [])
    if not_covered:
        # A refusal is complete in its own terms: it answered the question by
        # saying the documents do not. It is not an incomplete answer.
        return 90, "declined, which is an answer", True
    score, bits = 95, []
    if t:
        score = min(score, max(35, 95 - 20 * t))
        bits.append("%d topic(s) the pages cover were not addressed" % t)
    if n:
        # Gentle per-item penalty on purpose. The answer cannot tell whether an
        # item is missing because the documents are silent or because retrieval
        # failed, and one honest caveat on an otherwise complete answer is not
        # the same kind of thing as an answer that reached almost nothing asked.
        score = max(45, 95 - 12 * n)
        bits.append("%d requested item(s) not answered" % n)
    if pages_omitted:
        score = min(score, 70)
        bits.append("%d retrieved page(s) dropped at the context budget" % pages_omitted)
    return score, ("; ".join(bits) or "nothing reported missing"), True


def _reasoning_score(context_warning: str, claim_states: dict) -> tuple[int, str, bool]:
    if context_warning:
        return 55, "term-presence check flagged this answer", True
    af = claim_states or {}
    unsupported = int(af.get("unsupported") or 0)
    checked = int(af.get("checked") or af.get("total") or 0)
    if checked and unsupported:
        return max(40, int(95 - 55 * (unsupported / float(checked)))), \
            "%d of %d claims unsupported" % (unsupported, checked), True
    if checked:
        return 95, "%d of %d claims supported" % (checked, checked), True
    return NEUTRAL, "no claim-level check ran", False


def _authority_score(question: str, doc_types) -> tuple[int, str, bool]:
    """Is the answering document the KIND of thing the question asked about?"""
    m = _ASKED_TYPE_RE.search(question or "")
    if not m:
        return NEUTRAL, "question names no document type", False
    asked_raw = re.sub(r"\s+", " ", m.group(0)).strip().lower()
    asked = _TYPE_CANON.get(asked_raw, asked_raw)
    raw = [re.sub(r"\s+", " ", (t or "")).strip().lower() for t in (doc_types or []) if t]
    if not raw:
        return NEUTRAL, "no document type recorded for the answering document(s)", False
    # Canonicalise BOTH sides or the comparison is between different alphabets:
    # the question says "Master Services Agreement" and canonicalises to "msa",
    # while documents.doc_type holds the spelled-out form, and "msa" is not a
    # substring of it. Each recorded type contributes its own canonical form
    # alongside its raw text.
    have = set(raw)
    for t in raw:
        mt = _ASKED_TYPE_RE.search(t)
        if mt:
            have.add(_TYPE_CANON.get(re.sub(r"\s+", " ", mt.group(0)).strip().lower(),
                                     mt.group(0).strip().lower()))
    for t in have:
        if asked == t or asked in t or t in asked:
            return 96, "answered from a %s, as asked" % asked, True
    return 35, "asked about a %s; answered from %s" % (asked, ", ".join(sorted(raw)[:3])), True


def _temporal_score(question: str, effective_dates) -> tuple[int, str, bool]:
    years = {m.group(0) for m in _QUESTION_YEAR_RE.finditer(question or "")}
    if not years:
        return NEUTRAL, "question names no date", False
    have = " ".join(str(d) for d in (effective_dates or []) if d)
    if not have:
        return NEUTRAL, "no effective date recorded for the answering document(s)", False
    if any(y in have for y in years):
        return 96, "the question's year matches the document", True
    return 45, "question names %s; the document(s) are dated %s" % (
        "/".join(sorted(years)), have[:60]), True


def score(question: str, *, scope_method: str = "", citation_check: dict | None = None,
          missing_items=None, not_covered: bool = False, pages_omitted: int = 0,
          context_warning: str = "", claim_states: dict | None = None,
          doc_types=None, effective_dates=None,
          scope_docs=None, files_used=None, topics_missing=None,
          quality_warning=None) -> dict:
    """Six scores, the lowest of which governs the answer.

    Every input is something the pipeline already computed. Nothing here calls
    a model, and nothing here changes the answer text — this reports on an
    answer, it does not edit one.
    """
    ev, ev_why, ev_on = _evidence_score(citation_check, quality_warning)
    comp, comp_why, comp_on = _completeness_score(
        missing_items, not_covered, pages_omitted, topics_missing)
    rea, rea_why, rea_on = _reasoning_score(context_warning, claim_states)
    aut, aut_why, aut_on = _authority_score(question, doc_types)
    tem, tem_why, tem_on = _temporal_score(question, effective_dates)
    ret, ret_why = _scope_score(scope_method, scope_docs, files_used)
    ret_on = bool(scope_method) or bool(scope_docs)

    scores = {"evidence": ev, "retrieval": ret, "authority": aut,
              "completeness": comp, "reasoning": rea, "temporal": tem}
    why = {"evidence": ev_why,
           "retrieval": ret_why,
           "authority": aut_why, "completeness": comp_why,
           "reasoning": rea_why, "temporal": tem_why}
    assessed = {"evidence": ev_on, "retrieval": ret_on, "authority": aut_on,
                "completeness": comp_on, "reasoning": rea_on, "temporal": tem_on}

    # Only a dimension that actually looked at something may govern. A check
    # that could not run says nothing about the answer, and letting it set the
    # verdict would peg most answers at NEUTRAL and make the number as
    # uninformative as the single score this replaces - failing in the opposite
    # direction, but failing.
    live = [k for k in scores if assessed[k]]
    if live:
        governing = min(live, key=lambda k: scores[k])
    else:
        governing = "retrieval"
    return {
        "scores": scores,
        "why": why,
        "assessed": assessed,
        "not_assessed": sorted(k for k in scores if not assessed[k]),
        "governing": governing,
        "value": scores[governing],
        "reason": why[governing],
    }
