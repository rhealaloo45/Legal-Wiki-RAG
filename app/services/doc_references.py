"""
Doc-reference resolution and amendment edges
(target architecture § 01 stage 03, § 01.2 Cross-Document Relations).

A legal document rarely stands alone. It says "as defined in the Master
Services Agreement dated 3 March 2024", or "this Amendment No. 2 amends the
Shareholders' Agreement". Those in-text mentions are edges to other documents,
and today they vanish into prose.

Two rules from the doc shape this:

  * **A reference is never silently dropped.** It resolves against the
    `documents` registry to a real edge, or it is written as
    `references-unresolved` with the raw mention kept verbatim. A dropped
    reference is indistinguishable from a document that referenced nothing,
    and the second is a much stronger claim than the evidence supports.

  * **Amendment edges must exist before contradiction detection runs.**
    A superseded clause and the amendment that replaced it disagree by
    construction. Flagging that as a contradiction is not just noise, it is
    wrong: it is a resolved version chain, and the pipeline knows which
    direction resolved it only once the edge is written.

Matching is fuzzy but conservative. A wrong edge asserts a legal relationship
between two documents that does not exist — "this amends that" is a claim
about which text currently governs, so an unresolved reference a human can
follow up is strictly better than a confident wrong one.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

import config
from services import db

logger = logging.getLogger(__name__)

# The doc's own edge vocabulary (§ 03 "relations extended further").
AMENDS = "amends"
SUPERSEDED_BY = "superseded-by"
ANCILLARY_TO = "ancillary-to"
REFERENCES = "references"
UNRESOLVED = "references-unresolved"

# Edges that mean "one of these two documents replaces part of the other".
# Contradiction detection consults exactly this set.
_AMENDMENT_LABELS = frozenset({AMENDS, SUPERSEDED_BY})

_LABEL_ALIASES = {
    "amends": AMENDS, "amendment": AMENDS, "amending": AMENDS,
    "modifies": AMENDS, "varies": AMENDS,
    "superseded_by": SUPERSEDED_BY, "superseded-by": SUPERSEDED_BY,
    "supersedes": AMENDS,          # A supersedes B == A amends B, from A's side
    "replaced_by": SUPERSEDED_BY, "replaces": AMENDS,
    "ancillary_to": ANCILLARY_TO, "ancillary-to": ANCILLARY_TO,
    "annexure_to": ANCILLARY_TO, "schedule_to": ANCILLARY_TO,
    "subordinate_to": ANCILLARY_TO,
    "references": REFERENCES, "refers_to": REFERENCES, "cites": REFERENCES,
    "defined_in": REFERENCES, "incorporates": REFERENCES,
}

# Below this the mention is left unresolved. Set high deliberately: the cost
# of a wrong amendment edge (asserting the wrong text governs) is far higher
# than the cost of a reference a human has to resolve by hand.
MATCH_THRESHOLD = 0.72
# Amendment edges are held to a stricter bar again, for the same reason.
AMENDMENT_MATCH_THRESHOLD = 0.82

_UUID_PREFIX = re.compile(r"^[0-9a-fA-F-]{36}_")
_NOISE = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "dated", "certain", "that", "this",
    "agreement", "pdf", "docx", "txt", "redacted", "copy", "final", "draft",
    "executed", "signed", "version",
})


def _basename(name: str) -> str:
    """Strip the session-UUID prefix, folder path and extension.

    Stored source_doc values look like
    `<uuid>_Legal AI_NDA (1)_NDA 1_Redacted (1).pdf`. An in-text mention is
    "NDA 1". Scoring against the full stored string buries the two tokens that
    matter under a dozen that carry no identity at all.
    """
    s = _UUID_PREFIX.sub("", name or "")
    s = s.rsplit(".", 1)[0]
    parts = [p for p in re.split(r"[_/\\]", s) if p.strip()]
    if not parts:
        return s
    # Drop trailing segments that carry no identity — real filenames here end
    # in "_Redacted (1)" or "_final", and taking the last segment blindly would
    # make every document's name "Redacted (1)". A segment needs a real word,
    # not just a bare copy number.
    meaningful = [p for p in parts
                  if any(t.isalpha() for t in _tokens(p))]
    return meaningful[-1] if meaningful else parts[-1]


def _normalize(name: str) -> str:
    s = _NOISE.sub(" ", (name or "").lower())
    return _WS.sub(" ", s).strip()


_MONTHS_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)"
)
_DATE_EXPRESSIONS = (
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{_MONTHS_RE}\.?,?\s+\d{{4}}\b", re.I),
    re.compile(rf"\b{_MONTHS_RE}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I),
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
)


def _strip_dates(text: str) -> str:
    """Remove date expressions before matching.

    "the Services Agreement dated 11 September 2025" identifies a document;
    the date qualifies it. Filenames in this corpus never carry dates, so
    leaving those tokens in drags every candidate's score down equally — and
    the day number in particular would collide with the document-number check
    below, making a legitimate reference look like a mismatch.
    """
    out = text or ""
    for pattern in _DATE_EXPRESSIONS:
        out = pattern.sub(" ", out)
    return out


def _stem(token: str) -> str:
    """Crude singular form so "Services Agreement" matches "Service Agreement"."""
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(name: str) -> set[str]:
    return {_stem(t) for t in _normalize(name).split()
            if t and t not in _STOPWORDS}


def _score(mention: str, candidate: str, vocabulary: set[str] | None = None) -> float:
    """Similarity between an in-text mention and a registered document name.

    Containment, not Jaccard. A mention is a *subset* of the stored filename —
    "NDA 1" inside "<uuid>_Legal AI_NDA (1)_NDA 1_Redacted (1).pdf" — so
    scoring set-overlap symmetrically punishes the candidate for every extra
    token it carries and resolves nothing. What matters is how much of the
    *mention* the candidate accounts for.

    A distinctive token must match. Without that, "Employment Agreement" would
    score full marks against every other agreement in the corpus on its generic
    words alone — and a wrong document edge is the failure this whole module is
    built to avoid.
    """
    mention_clean = _strip_dates(mention)
    m_tokens = _tokens(mention_clean)
    if not m_tokens:
        return 0.0

    base = _basename(candidate)
    c_tokens = _tokens(base) | _tokens(candidate)
    if not c_tokens:
        return 0.0

    m_words = {t for t in m_tokens if not t.isdigit()}
    m_nums = {t for t in m_tokens if t.isdigit()}
    c_words = {t for t in c_tokens if not t.isdigit()}
    c_nums = {t for t in c_tokens if t.isdigit()}

    # Number agreement is decisive. "Service Agreement 2" and "Service
    # Agreement 5" share every word; the numeral is the entire difference
    # between them. A mention naming a numbered document that this candidate
    # does not carry is not a weak match — it is the wrong document.
    if m_nums and not (m_nums & c_nums):
        return 0.0

    if not m_words or not (m_words & c_words):
        return 0.0

    # Containment measured both ways, best wins.
    #
    # Forward ("how much of the mention does this document account for")
    # handles a bare "Service Agreement 2". Reverse ("how much of this
    # document's name does the mention contain") handles a mention that wraps
    # the name in referring language — "this Amendment to NDA 1" carries the
    # whole of "NDA 1" plus words that belong to the sentence, not the title.
    #
    # Neither direction alone is enough, and no word is discarded for being
    # unmatched: "the Master Services Agreement" must NOT resolve to
    # "Service Agreement 2", and it only fails to if "master" keeps counting
    # against it.
    base_tokens = _tokens(base)
    base_words = {t for t in base_tokens if not t.isdigit()}
    base_nums = {t for t in base_tokens if t.isdigit()}

    forward = len(m_words & c_words) / len(m_words)

    # Reverse only counts if the document's own identifying number is present
    # in the mention. Without that gate a thin basename like "Service
    # Agreement 2" reduces to the single word {service}, which any mention
    # containing "services" trivially contains in full — so "the Master
    # Services Agreement" would score a perfect reverse match against a
    # document it has nothing to do with.
    reverse = 0.0
    if base_words and (not base_nums or (base_nums & m_nums)):
        reverse = len(base_words & m_words) / len(base_words)

    containment = max(forward, reverse)

    seq = SequenceMatcher(None, _normalize(mention_clean), _normalize(base)).ratio()
    number_bonus = 0.12 if (m_nums and m_nums <= c_nums) else 0.0
    return min(1.0, 0.62 * containment + 0.26 * seq + number_bonus)


def _registry(wiki_id: str, session_id: str) -> list[str]:
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT source_doc FROM documents
            WHERE wiki_id = :w AND session_id = :s
        """), {"w": wiki_id, "s": session_id}).fetchall()
    return [r[0] for r in rows]


def resolve_mention(mention: str, candidates: list[str], exclude: str | None = None,
                    threshold: float = MATCH_THRESHOLD) -> tuple[str | None, float]:
    """Best registry match for an in-text mention, or (None, best_score).

    Returning the score even on failure matters: it is what gets recorded on
    an unresolved edge, so a near-miss is visibly different from a mention
    that matched nothing at all.
    """
    pool = [c for c in candidates if not (exclude and c == exclude)]
    vocabulary: set[str] = set()
    for cand in pool:
        vocabulary |= _tokens(cand) | _tokens(_basename(cand))

    best, best_score = None, 0.0
    for cand in pool:
        s = _score(mention, cand, vocabulary)
        if s > best_score:
            best, best_score = cand, s
    if best_score >= threshold:
        return best, best_score
    return None, best_score


def persist_references(wiki_id: str, session_id: str, from_doc: str,
                       references: list[dict]) -> dict[str, int]:
    """Resolve and write a document's outgoing references.

    Replaces this document's prior edges in the same transaction, matching the
    swap-not-blend rule the typed tables already follow — a re-ingest that
    dropped a reference must not leave the old edge behind asserting a
    relationship the current text no longer claims.
    """
    if not config.USE_DATABASE or not references:
        return {}
    from sqlalchemy import text

    candidates = _registry(wiki_id, session_id)
    counts: dict[str, int] = {}
    with db.get_engine().connect() as conn:
        try:
            conn.execute(text("""
                DELETE FROM document_relations
                WHERE wiki_id = :w AND session_id = :s AND from_doc = :d
            """), {"w": wiki_id, "s": session_id, "d": from_doc})

            seen: set[tuple[str, str]] = set()
            for ref in references:
                if not isinstance(ref, dict):
                    continue
                raw = str(ref.get("referenced_document") or ref.get("reference_text")
                          or "").strip()
                if not raw:
                    continue
                label = _LABEL_ALIASES.get(
                    str(ref.get("relationship") or "").strip().lower().replace(" ", "_"),
                    REFERENCES,
                )
                threshold = (AMENDMENT_MATCH_THRESHOLD
                             if label in _AMENDMENT_LABELS else MATCH_THRESHOLD)
                match, score = resolve_mention(raw, candidates, exclude=from_doc,
                                               threshold=threshold)
                final_label = label if match else UNRESOLVED
                key = (raw.lower(), final_label)
                if key in seen:
                    continue
                seen.add(key)
                conn.execute(text("""
                    INSERT INTO document_relations
                        (wiki_id, session_id, from_doc, to_doc, to_doc_raw, label,
                         resolved, match_score, confidence, evidence_text)
                    VALUES (:w, :s, :f, :t, :raw, :l, :res, :score, :conf, :ev)
                    ON CONFLICT (wiki_id, session_id, from_doc, to_doc_raw, label)
                    DO NOTHING
                """), {
                    "w": wiki_id, "s": session_id, "f": from_doc, "t": match,
                    "raw": raw[:1000], "l": final_label, "res": bool(match),
                    "score": round(score, 3),
                    "conf": ref.get("confidence"),
                    "ev": (str(ref.get("reference_text") or "")[:1000]) or None,
                })
                counts[final_label] = counts.get(final_label, 0) + 1
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Failed writing document references for %s", from_doc)
            raise

    if counts.get(UNRESOLVED):
        logger.info("%s: %d document reference(s) could not be resolved to a "
                    "registered document — kept verbatim, not dropped",
                    from_doc, counts[UNRESOLVED])
    return counts


def amendment_partners(session_id: str, doc_name: str) -> set[str]:
    """Documents in an amendment/supersession chain with `doc_name`, in either
    direction. Both directions matter: whether A amends B or B amends A, a
    disagreement between them is a version chain rather than a contradiction.
    """
    if not config.USE_DATABASE:
        return set()
    from sqlalchemy import text
    labels = tuple(_AMENDMENT_LABELS)
    try:
        with db.get_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT to_doc FROM document_relations
                WHERE session_id = :s AND from_doc = :d AND resolved
                  AND label = ANY(:labels)
                UNION
                SELECT from_doc FROM document_relations
                WHERE session_id = :s AND to_doc = :d AND resolved
                  AND label = ANY(:labels)
            """), {"s": session_id, "d": doc_name, "labels": list(labels)}).fetchall()
        return {r[0] for r in rows if r[0]}
    except Exception as err:
        logger.warning("Could not read amendment edges for %s: %s", doc_name, err)
        return set()


def is_amendment_pair(session_id: str, doc_a: str, doc_b: str) -> bool:
    """Whether these two documents are in a version chain. Used by
    contradiction detection to tell a resolved amendment apart from a genuine
    factual conflict."""
    if not doc_a or not doc_b or doc_a == doc_b:
        return False
    return doc_b in amendment_partners(session_id, doc_a)


def get_document_relations(wiki_id: str, session_id: str,
                           doc_name: str | None = None) -> list[dict]:
    if not config.USE_DATABASE:
        return []
    from sqlalchemy import text
    sql = """
        SELECT from_doc, to_doc, to_doc_raw, label, resolved, match_score, evidence_text
        FROM document_relations
        WHERE wiki_id = :w AND session_id = :s
    """
    params = {"w": wiki_id, "s": session_id}
    if doc_name:
        sql += " AND (from_doc = :d OR to_doc = :d)"
        params["d"] = doc_name
    sql += " ORDER BY resolved DESC, label"
    with db.get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [
        {"from_doc": r[0], "to_doc": r[1], "to_doc_raw": r[2], "label": r[3],
         "resolved": r[4], "match_score": r[5], "evidence_text": r[6]}
        for r in rows
    ]
