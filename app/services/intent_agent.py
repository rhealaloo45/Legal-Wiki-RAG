"""
Intent classifier agent — LangGraph orchestration of the query pipeline.

The lawyer-facing query flow is modelled as a directed graph:

    classify_intent → disambiguation → clarification → retrieve → generate → validate

Conditional edges short-circuit to END when disambiguation or clarification is needed.
Each node emits real-time stage events via LangGraph's custom stream writer so the
frontend can render progress tiles. The heavy lifting still lives in the existing
services (wiki.py, db.py, llm.py) — this module only classifies intent and orchestrates.

Five lawyer intents drive prompt selection in wiki.generate_answer():
    factual          — direct extraction / Q&A          (ANSWER_PROMPT)
    risk_assessment  — go/no-go, red flags, recommend   (ASSESSMENT_PROMPT)
    comparison       — side-by-side across documents     (COMPARISON_PROMPT)
    obligation       — duties, deadlines, compliance     (OBLIGATION_PROMPT)
    drafting         — draft / redline clause language   (DRAFTING_PROMPT)
"""

import re
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover - older langgraph
    get_stream_writer = None

import config
from services import wiki
from services import llm
from services import db
from services import tracing

logger = logging.getLogger(__name__)

VALID_INTENTS = ["factual", "risk_assessment", "comparison", "obligation", "drafting"]

# How many unresolved "which document?" prompts a thread may issue in a row
# before the question is answered from the whole corpus instead. Two, not one:
# the first prompt is the one that legitimately resolves most questions, and a
# single retry covers a user who names the document imprecisely on their first
# try. A third has never resolved one in testing — see check_disambiguation_node.
_MAX_DISAMBIGUATION_ROUNDS = 2

INTENT_LABELS = {
    "factual": "Factual",
    "risk_assessment": "Risk Assessment",
    "comparison": "Comparison",
    "obligation": "Obligation",
    "drafting": "Drafting",
}

# ---------------------------------------------------------------------------
# Regex fast-paths — 0 tokens, instant. Checked before the LLM fallback.
# Order encodes priority: a query matching several buckets resolves to the
# most specific lawyer task (drafting > comparison > risk > obligation).
# ---------------------------------------------------------------------------
_RX_DRAFTING = re.compile(
    r'\b(draft|re-?draft|redline|re-?write|suggest(?:ed)?\s+(?:language|wording|clause)|'
    r'counter[- ]?proposal|alternative\s+(?:wording|language|clause)|'
    r'propose\s+(?:a|an|new|revised)\s+clause|word(?:ing)?\s+for\s+(?:a|an|the))\b',
    re.IGNORECASE,
)
# Drafting a document that does not exist yet, as opposed to redrafting one that
# does. The distinction decides whether "which document do you mean?" is even an
# answerable question (see check_disambiguation_node): "redline the indemnity
# clause" refers to an existing document and SHOULD be disambiguated, whereas
# "draft a new agreement" / "tips for drafting one" refers to nothing in the
# corpus and can never be resolved by any reply.
#
# Deliberately narrow — requires an explicit newness marker ("a new ..."/"from
# scratch"), a how-to framing, or a request for drafting GUIDANCE (steps/tips/
# advice), rather than treating every drafting question as document-free. A bare
# "redraft the termination clause" matches none of these and still disambiguates.
_RX_DRAFTING_NEW_DOC = re.compile(
    r'\b(?:'
    r'draft(?:ing)?\s+(?:a|an)\s+new\b'
    r'|(?:a|an)\s+new\s+(?:agreement|contract|nda|deed|lease|licen[cs]e|mou|'
    r'memorandum|policy|clause)\b'
    r'|from\s+scratch\b'
    r'|how\s+(?:to|do\s+i|should\s+i|would\s+i)\s+(?:go\s+about\s+)?draft'
    r'|(?:steps|tips|advice|guidance|pointers|best\s+practices|things|points|'
    r'keep\s+in\s+mind)\b'
    r'[^.?!]{0,60}\bdraft'
    r'|\bdraft(?:ing)?\b[^.?!]{0,60}\b(?:steps|tips|advice|guidance|pointers|'
    r'best\s+practices|keep\s+in\s+mind)\b'
    # "draft a clause for a software vendor" names the counterparty by its
    # GENERIC ROLE (vendor, supplier, partner, ...), not a proper name or an
    # existing document — there is nothing in the corpus such a reference
    # could resolve to, the same way "a new agreement" has nothing to resolve
    # to. Confirmed live (deployed): "Help me draft an AI governance clause
    # for a software vendor" re-asked "which agreement?" on every turn,
    # including once the user clarified "a Statement of Work with a software
    # integration partner" — still just a generic role, never a real
    # corpus document, so no reply could ever satisfy the prompt.
    #
    # The first cut of this rule required the role word within 2 words of
    # "for a/an/any/our" — too tight for real phrasing. Confirmed live again:
    # "before drafting an indemnity clause for a services agreement with a
    # software vendor" has FIVE words between "for a" and "vendor" (the
    # agreement type itself), so the tight version never matched and the same
    # disambiguation loop recurred in the Ask tab. Dropped the immediate-
    # article requirement and just look for "for ... <role>" within a wider
    # span — a role word this far past "draft ... for" is never a corpus
    # document reference either way.
    r'|\bdraft(?:ing)?\b[^.?!]{0,100}\bfor\b[^.?!]{0,60}\b'
    r'(?:vendor|supplier|partner|counterparty|client|customer|contractor|'
    r'licensor|licensee|distributor|reseller|integrator)\b'
    r')',
    re.IGNORECASE,
)
_RX_COMPARISON = re.compile(
    r'\b(compare|comparison|differ(?:s|ence|ences)?|versus|vs\.?|'
    r'side[- ]by[- ]side|contrast|how\s+do\s+.+\s+differ)\b',
    re.IGNORECASE,
)
_RX_BETWEEN = re.compile(r'\bbetween\b.+\band\b', re.IGNORECASE)
# _RX_BETWEEN alone is too broad: legal writing routinely phrases a SINGLE-
# document pleading/analysis question as "the distinction/difference between
# X and Y" (e.g. "how is the legal distinction between the manufacturer and
# the independent dealership pleaded?") — that's asking how one document
# argues a distinction, not requesting a cross-document comparison. Suppress
# the _RX_BETWEEN-only trigger for that shape; _RX_COMPARISON's own keywords
# (compare/differ/versus/contrast) still fire normally regardless.
_RX_BETWEEN_EXCLUDE = re.compile(
    r'\bhow\s+(?:is|are)\b.{0,20}\b(?:distinction|difference)\b.{0,10}between\b.{0,100}'
    r'\b(?:pleaded|argued|established|asserted|drawn|treated|addressed|framed|structured)\b',
    re.IGNORECASE,
)
# A second, far more common false trigger for the _RX_BETWEEN-only rule: naming a
# SINGLE bilateral instrument by its two parties — "the Joint Venture Agreement
# between Tata Consumer Products and BrewSphere", "the NDA between X and Y". That
# is a factual question about ONE document, not a cross-document comparison, but
# "between … and …" matched and forced a comparison intent, which then rendered a
# two-column "Key Differences / Which Party It Favors" table that fabricated a
# second-document framing absent from the question (confirmed live on the
# BrewSphere JVA1 and SteelLoop JVA3 questions). Suppress the between-only trigger
# when an instrument noun immediately precedes "between … and". A genuine
# cross-document comparison ("compare the NDA between A and B with the one between
# C and D") still carries an explicit comparison keyword that _RX_COMPARISON
# matches one check earlier, so this exclusion never reaches it.
_RX_BETWEEN_PARTIES = re.compile(
    # Litigation filings name their two parties the same way a contract does
    # ("the Notice Invoking Arbitration between Tata Steel and NordForge"), and
    # are just as much a SINGLE document — but only contract-type nouns were
    # listed, so those questions still forced a comparison template.
    r'\b(?:agreement|contract|nda|sha|jva?|jv|msa|mou|deed|lease|licen[cs]e|'
    r'venture|arrangement|memorandum|settlement|'
    r'notice|petition|affidavit|application|award|plaint|summons|'
    r'suit|claim|proceedings?|dispute|arbitration)\b[^.?!;]{0,30}?\bbetween\b'
    # A corporate suffix between the two party names ("... Ltd. and ...",
    # "... Pte. Ltd. and ...") carries its own period, which [^.?!;]+ can never
    # cross — the exclusion silently stopped matching short of "and" and the
    # whole rule went inert for exactly the two-party names it exists to catch
    # (confirmed live: "the NDA between Tata Passenger Electric Mobility Ltd.
    # and Cirrus Battery Intelligence" fell through to a comparison template
    # over a single document). Allow periods, since the pattern is already
    # bounded at the next "and" and a short abbreviation dot is far more
    # likely here than an actual sentence break.
    r'[^?!;]{1,80}?\band\b',
    re.IGNORECASE,
)
# A third false trigger: "board composition structured between Tata Power and
# the founders", "equity split between the two shareholders", "seats divided
# between X and Y" — this describes how ONE document internally apportions
# something (board seats, equity, duties) between two parties/constituencies,
# not a request to compare two documents. The verb immediately before
# "between" signals an internal allocation, not a second document's identity
# (confirmed live: "How is the board composition structured between Tata
# Power Renewable Energy Limited and the founders" on a single-document SHA
# review forced a comparison template that rendered a "Not applicable — no
# second document in corpus" table instead of just answering the question).
_RX_BETWEEN_ALLOCATION = re.compile(
    r'\b(?:structured|split|divided|allocated|apportioned|distributed|shared|balanced)\s+between\b',
    re.IGNORECASE,
)
_RX_OBLIGATION = re.compile(
    r'(?:'
    r'what\s+are\s+(?:the|our)\s+obligations'       # "what are the/our obligations"
    r'|(?:extract|identify|list)\s+(?:all\s+)?(?:the\s+)?(?:key\s+)?obligations'
    r'|our\s+(?:obligations|duties)'
    r'|list\s+(?:the\s+)?(?:obligations|duties|deadlines)'
    r'|deadlines?\s+(?:to|we|under|in)\b'
    r'|comply\s+with|compliance\s+requirements'
    r'|required\s+to|must\s+we'
    r'|what\s+(?:do|must)\s+we\s+(?:do|provide)'
    r'|notice\s+periods?\s+(?:for|under|in)'
    r')',
    re.IGNORECASE,
)
# Risk assessment reuses the legal-recommendation patterns (moved here from wiki.py).
_RX_RISK = re.compile(
    r'(?:go\s*/\s*no[- ]?go|recommend|recommendation|should\s+(?:we|i|tata)\s+sign|'
    r'risk\s+assessment|risk\s+review|advise|advisory|red\s+flag|deal[- ]?breaker|'
    r'approve|approval|sign\s+off|signoff|would\s+you\s+(?:recommend|advise|sign)|'
    r'safe\s+to\s+sign|ready\s+to\s+(?:sign|execute)|negotiation\s+strategy|'
    r'accept\s+or\s+reject|proceed\s+or\s+not)',
    re.IGNORECASE,
)
# "board approval", "shareholder consent", "unanimous / special-majority approval"
# are GOVERNANCE MECHANICS a factual question asks ABOUT (e.g. "which reserved
# matters require unanimous board approval?") — not a request to assess risk or
# recommend signing. But _RX_RISK's bare "approve|approval" token matched them,
# forcing a risk-assessment template onto a plain clause-extraction question
# (confirmed live on the SteelLoop Reserved-Matters question). This guard fires
# only when approval/consent is governance-qualified.
_RX_GOVERNANCE_APPROVAL = re.compile(
    # "written" alone (not just "prior written") is a genuine governance qualifier
    # too — "without Tata's written approval" is a personnel-control clause a
    # factual/risk_assessment question asks ABOUT, not a request to assess risk
    # (confirmed live: SA6 personnel-clause question got forced into a full
    # Accept/Reject/Negotiate essay off this single bare "approval" hit). Kept
    # narrow: this only suppresses risk intent when approval/consent is the ONLY
    # _RX_RISK signal (enforced by _is_governance_approval_only's subset check
    # below) — a question with an actual risk cue alongside it still classifies
    # as risk regardless of this guard.
    r'\b(?:board|shareholders?|members?|majority|unanimous|special\s+majority|'
    r'statutory|regulatory|prior\s+written|written|requisite)\b\s+\w*\s*\b(?:approval|approve|consent)\b'
    r'|\b(?:approval|approve|consent)\s+(?:of|by|from)\s+(?:the\s+)?(?:board|shareholders?|members?)\b',
    re.IGNORECASE,
)


# "advise|advisory" in _RX_RISK is meant to catch an advice-request ("please
# advise", "your advisory on this") — but it also matches a party's own NAME
# when the corpus contains an entity like "Helios Grid Advisory Private
# Limited" (confirmed live: a plain payment-terms/TDS/GST lookup question got
# forced into a full risk-assessment essay purely because the counterparty's
# name contains the word "Advisory"). "Advisory" immediately followed by a
# corporate suffix is a company name, not a verb (same suffix list wiki.py's
# _PARTY_NAME_RE uses for the same purpose — kept local here to avoid a
# cross-module import for one shared string).
_RX_ADVISORY_ENTITY = re.compile(
    r'\badvisory\b\s+(?:Private\s+Limited|Pvt\.?\s*Ltd\.?|Pte\.?\s*Ltd\.?|Limited|Ltd\.?|'
    r'LLP|LLC|Inc\.?|Corp(?:oration)?|PLC|GmbH|N\.?V\.?|S\.?A\.?)\b',
    re.IGNORECASE,
)


def _is_advisory_entity_name_only(q: str) -> bool:
    """True when the ONLY _RX_RISK signal is 'advise'/'advisory' and that word
    is immediately followed by a corporate suffix — i.e. it's part of a party's
    name, not a request for advice. Same subset-guard shape as
    _is_governance_approval_only: a question that ALSO carries a real risk cue
    still classifies as risk regardless of this guard."""
    hits = {m.group(0).lower() for m in _RX_RISK.finditer(q)}
    return hits <= {"advise", "advisory"} and bool(_RX_ADVISORY_ENTITY.search(q))


# A case caption ("the Tata Sons vs Deepak Kumar case", "X vs. Y") names ONE
# matter by its two litigants the same way a contract names its two
# signatories — it is not a request to compare two documents, but bare "vs"/
# "versus" is one of _RX_COMPARISON's own trigger words, so a single-document
# question about a captioned case forced the comparison template every time
# (confirmed live: "What did the defendants agree to transfer ... in the Tata
# Sons vs Deepak Kumar case?" rendered a two-column table with "Not
# Applicable" for the missing second document).
_RX_CASE_CAPTION_VS = re.compile(
    r'\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4}\s+vs\.?\s+'
    r'[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4}\b',
)


def _is_case_caption_vs_only(q: str) -> bool:
    """True when the ONLY _RX_COMPARISON signal is a "vs" naming a case
    caption — so comparison intent should be suppressed. Same subset-guard
    shape as _is_governance_approval_only: a question that ALSO carries a
    real comparison cue (compare, differ, contrast, side-by-side) elsewhere
    still classifies as comparison, because stripping the caption match
    leaves that cue in place for _RX_COMPARISON to find again."""
    m = _RX_CASE_CAPTION_VS.search(q)
    if not m:
        return False
    remainder = q[:m.start()] + q[m.end():]
    return not _RX_COMPARISON.search(remainder)


def _is_governance_approval_only(q: str) -> bool:
    """True when the ONLY _RX_RISK signal in the question is an approval/consent
    word used in a governance sense — so risk intent should be suppressed and the
    question routed to factual/obligation instead. A question that ALSO carries a
    real risk cue (recommend, should we sign, go/no-go, …) still classifies as
    risk, because its risk hits then aren't a subset of {approve, approval}."""
    hits = {m.group(0).lower() for m in _RX_RISK.finditer(q)}
    return hits <= {"approve", "approval"} and bool(_RX_GOVERNANCE_APPROVAL.search(q))

_INTENT_LLM_PROMPT = (
    "You classify a lawyer's question into exactly ONE intent.\n\n"
    "Intents:\n"
    "- factual: extract or explain what a document says (clauses, definitions, parties, "
    "summaries, term length, subject matter, dates) — including a single specific "
    "document's own attributes.\n"
    "- risk_assessment: judge risk, recommend go/no-go, flag red flags, advise whether to sign.\n"
    "- comparison: compare two or more documents/clauses side by side, find differences.\n"
    "- obligation: list duties, obligations, deadlines, or compliance requirements.\n"
    "- drafting: write, redline, or propose new/alternative clause language.\n\n"
    "IMPORTANT: classify as comparison ONLY if the question ITSELF asks to compare, "
    "contrast, or find differences between two or more documents/clauses. A question "
    "asking about ONE specific document's own attribute — e.g. \"What is the term of "
    "NDA 5?\", \"What is NDA 5 about?\", \"When was Service Agreement 2 signed?\" — is "
    "factual, never comparison, even if the same conversation earlier discussed several "
    "other documents.\n\n"
    "Question: {question}\n\n"
    "Respond with JSON only:\n"
    '{{"intent": "one of the five slugs", "confidence": 0.0-1.0}}'
)


def classify_intent(question: str, conversation_context: str = "") -> dict:
    """Classify a question into one of the five lawyer intents.

    Regex fast-path first (0 tokens); fast LLM fallback only for ambiguous queries.
    Always returns a valid intent — defaults to 'factual' (safest) on any failure.
    """
    q = question or ""

    if _RX_DRAFTING.search(q):
        return {"intent": "drafting", "confidence": 0.95, "method": "regex"}
    if _RX_COMPARISON.search(q) and not _is_case_caption_vs_only(q):
        return {"intent": "comparison", "confidence": 0.9, "method": "regex"}
    if _RX_BETWEEN.search(q) and not _RX_BETWEEN_EXCLUDE.search(q) \
            and not _RX_BETWEEN_PARTIES.search(q) \
            and not _RX_BETWEEN_ALLOCATION.search(q):
        return {"intent": "comparison", "confidence": 0.9, "method": "regex"}
    if _RX_RISK.search(q) and not _is_governance_approval_only(q) \
            and not _is_advisory_entity_name_only(q):
        return {"intent": "risk_assessment", "confidence": 0.9, "method": "regex"}
    if _RX_OBLIGATION.search(q):
        return {"intent": "obligation", "confidence": 0.85, "method": "regex"}

    if not config.ENABLE_INTENT_CLASSIFIER:
        return {"intent": "factual", "confidence": 0.5, "method": "disabled"}

    try:
        prompt = _INTENT_LLM_PROMPT.format(question=q)
        raw, _ = llm.ask(prompt, fast=True, max_tokens=config.MAX_TOKENS_INTENT_CLASSIFY)
        parsed = wiki._parse_json_safe(raw)
        if parsed:
            intent = str(parsed.get("intent", "")).strip().lower()
            if intent in VALID_INTENTS:
                try:
                    conf = float(parsed.get("confidence", 0.6))
                except (TypeError, ValueError):
                    conf = 0.6
                return {"intent": intent, "confidence": conf, "method": "llm"}
    except Exception as e:
        logger.error("classify_intent LLM fallback failed: %s", e)

    return {"intent": "factual", "confidence": 0.5, "method": "fallback"}


def get_query_strategy(intent: str) -> dict:
    """Return retrieval hints tuned per intent. Consumed by wiki.get_context()."""
    if intent == "comparison":
        # Comparisons need pages from several documents — widen the net.
        return {"multi_doc": True, "small_wiki_threshold": 30}
    if intent == "obligation":
        return {"keyword_boost": ["shall", "must", "obligation", "duty", "within", "days", "notice"]}
    if intent == "drafting":
        return {"keyword_boost": ["definition", "defined term", "clause", "shall", "means"]}
    return {}


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class QueryState(TypedDict, total=False):
    # Input
    question: str
    session_id: str          # wiki/doc session — where pages & documents live (retrieval)
    chat_session_id: str     # per-thread chat session — where THIS conversation's
                             # messages are stored (carryover + conversation history).
                             # Diverges from session_id whenever a fixed main wiki is
                             # served; falls back to session_id when unset.
    target_doc: str
    is_followup: bool
    exclude_cached_answers: bool
    collection_id: int       # user-pinned collection — a hard retrieval
                             # boundary, see resolve_scope_node
    # Classification
    intent: str
    intent_confidence: float
    intent_method: str
    # Pre-query checks
    needs_disambiguation: bool
    disambiguation_data: dict
    needs_clarification: bool
    clarification_data: dict
    unconfirmed_doc_reference: bool
    scope_decision: dict
    # Retrieval / generation
    conversation_context: str
    wiki_context: str
    selected_titles: list
    retrieval_meta: dict
    clause_lookup: dict      # clause-number -> heading/page mapping result (or None)
    answer_result: dict
    validation: dict


def _emit(event: dict) -> None:
    """Emit a custom stage event into the LangGraph stream (best-effort)."""
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@tracing.traced_node("classify_intent")
def classify_intent_node(state: QueryState) -> dict:
    logger.info("[AGENT] classify_intent_node: question=%r", state["question"][:80])
    _emit({"stage": "classifying", "status": "active", "message": "Classifying intent…"})
    res = classify_intent(state["question"], state.get("conversation_context", ""))
    label = INTENT_LABELS.get(res["intent"], "Factual")
    logger.info("[AGENT] intent=%s conf=%.2f method=%s", res["intent"], res["confidence"], res["method"])
    _emit({
        "stage": "intent_identified", "status": "done",
        "intent": res["intent"], "intent_label": label,
        "intent_confidence": res["confidence"], "intent_method": res["method"],
        "message": f"Intent: {label}",
    })
    return {
        "intent": res["intent"],
        "intent_confidence": res["confidence"],
        "intent_method": res["method"],
    }


@tracing.traced_node("check_disambiguation")
def check_disambiguation_node(state: QueryState) -> dict:
    logger.info("[AGENT] check_disambiguation_node")
    # Deliberately does NOT bypass on is_followup: classify_query() already has
    # deterministic vague-reference detection (_VAGUE_DOC_PATTERN catches "this
    # document"/"the agreement" etc.) that must still run even mid-conversation —
    # skipping it for every follow-up meant a vague "this document" question
    # silently resolved to whatever document conversation history implied,
    # without ever confirming with the user (confirmed live: a "top 10 risks in
    # this document" follow-up silently answered about a document from several
    # questions earlier, no disambiguation prompt shown). Named-document and
    # known-entity mentions still skip via the checks inside classify_query().
    if state.get("target_doc"):
        logger.info("[AGENT] disambiguation skipped (doc named)")
        return {"needs_disambiguation": False}

    # Drafting something NEW has no existing document to resolve to, so asking
    # "which agreement are you asking about?" is unanswerable by construction —
    # the prompt promises to "pull up the right one", but for a counterparty that
    # isn't in the corpus at all no reply can ever satisfy it. Confirmed live: a
    # request to draft a Jaguar Land Rover manufacturing agreement re-asked the
    # same question on every turn, including when the user answered it with the
    # counterparty name, and never escaped.
    #
    # This only suppresses the PROMPT, not document scoping: resolve_scope runs
    # afterwards regardless and still pins any document the question does name,
    # so "how to draft a clause like the one in Service Agreement 2" is still
    # answered against SA2 — it just isn't interrogated first.
    if state.get("intent") == "drafting" and _RX_DRAFTING_NEW_DOC.search(state["question"]):
        logger.info("[AGENT] disambiguation skipped (drafting a new document — "
                    "no existing document to resolve)")
        return {"needs_disambiguation": False}

    if wiki._question_names_a_document(state["question"], []):
        # _question_names_a_document is a pure pattern match ("type + number",
        # e.g. "service agreement 1") — it has no way to check whether such a
        # document actually exists in this corpus. Confirmed live: a nonexistent
        # "service agreement 1" reference was treated as fully resolved every
        # time on this signal alone, so disambiguation never ran; retrieval then
        # fell through to generic semantic search and silently returned a
        # different arbitrary document on every rephrasing of the same question
        # (Test_SA_36, then Test_SA_44, then Test_SA_35 across one session),
        # with the answer-generation model inventing its own "Service Agreement
        # 1 (Test_SA_44)" identity label that retrieval never established.
        # Cross-check against the real corpus before trusting the pattern match.
        try:
            index = wiki._load_index(state["session_id"])
            pages = index.get("pages", {})
            # _question_names_a_document's entity-pattern branch (as opposed to
            # its numbered-pattern branch) captures at most 3 words before the
            # doc-type phrase — for a longer real company name ("DriveConnect
            # Experience Labs Private Limited joint venture agreement") it grabs
            # a truncated fragment ("Labs Private Limited") that _detect_
            # mentioned_files correctly can't find verbatim, even though the
            # full real entity genuinely exists and gets found via the broader
            # fuzzy entity matcher elsewhere in the pipeline. Confirmed live:
            # this exact case was flagged "unconfirmed" despite both target
            # documents being correctly retrieved and cited moments later.
            # Accept that broader signal as confirmation too, not just an exact
            # file-name/number match.
            confirmed = bool(wiki._detect_mentioned_files(state["question"], pages)) \
                or wiki._question_mentions_known_entity(state["question"], pages)
        except Exception as e:
            logger.error("Named-document confirmation check failed: %s", e)
            confirmed = True  # fail open — don't block the pipeline on an internal error
        if confirmed:
            logger.info("[AGENT] disambiguation skipped (doc named and confirmed)")
            return {"needs_disambiguation": False}
        # Confirmation failed. HOW the question named a document decides what an
        # unresolved reference means:
        #  - Numbered reference ("Service Agreement 1", "NDA 3"): names a document
        #    by the identifier the corpus uses as a filename. If none matches it
        #    genuinely does not exist, so flag unconfirmed_doc_reference (the model
        #    then says so plainly instead of silently answering from an arbitrary
        #    document — the original protection this branch was built for).
        #  - Descriptive paraphrase ("wastewater-dosing NDA", "subsea diagnostic
        #    deliverables"): a subject-matter description that is NEVER a literal
        #    corpus name even when the document is real and merely filed under a
        #    bare type+number. A miss here means "couldn't resolve the paraphrase",
        #    not "document absent" — so DON'T assert non-existence. Fall through to
        #    classify_query below, which runs vague-reference disambiguation and,
        #    failing that, ordinary corpus retrieval. Confirmed live: descriptive
        #    NDA/JV questions (e.g. "irrigation-management NDA", "floodgate-
        #    automation NDA") that name a genuinely-present document were being
        #    answered with a flat "No document matching '<paraphrase>' exists in
        #    this corpus" plus a fabricated generic-clause answer.
        if wiki._names_numbered_document(state["question"]):
            logger.warning(
                "Question names a NUMBERED document but no matching document exists "
                "in this session — flagging as unconfirmed instead of silently "
                "answering from an arbitrary document: %r", state["question"][:80],
            )
            return {"needs_disambiguation": False, "unconfirmed_doc_reference": True}
        logger.info(
            "Question names a document by descriptive paraphrase that didn't "
            "resolve — falling through to disambiguation/corpus retrieval instead "
            "of asserting non-existence: %r", state["question"][:80],
        )
        # fall through to classify_query below

    # Phase 2: an ordinary conversational follow-up ("who can terminate it",
    # "why", "does the same apply to the other party") carries no document
    # reference at all, so it used to fall straight through to
    # classify_query() below every single time — and that LLM ambiguity
    # judgment has no memory of the conversation, so it re-asked which
    # document 2-3 times per thread even immediately after the document was
    # unambiguously resolved (confirmed live: a 10-turn thread got stuck
    # asking from turn 3 onward and never recovered for the rest of the
    # conversation). Skip straight to resolve_scope/retrieve — which already
    # inherits the same carryover scope via wiki._carryover_scope() — when
    # BOTH hold:
    #   (a) wiki._carryover_scope() itself agrees a scope can be inherited —
    #       already conservative and battle-tested: refuses to carry over if
    #       the question names any document TYPE, is broad/collective
    #       phrasing, or the prior turn's own scope wasn't cleanly resolved.
    #   (b) the question doesn't match _VAGUE_DOC_PATTERN ("this document",
    #       "the agreement") — this is a SEPARATE, deliberately redundant
    #       check: _carryover_scope's own type-word gate (_CARRYOVER_TYPE_RE)
    #       does not include the word "document", only named types (NDA,
    #       service agreement, ...), so "top 10 risks in THIS DOCUMENT" — the
    #       exact phrase that caused the original documented bug this
    #       function's is_followup comment above describes — would NOT be
    #       blocked by _carryover_scope alone. Re-checking the vague pattern
    #       here directly is what actually reopens that guard; skipping it
    #       would silently reintroduce the original bug instead of just
    #       fixing the new one.
    # _VAGUE_DOC_PATTERN also matches ordinary uses of "the agreement"/"the
    # contract" that are not actually naming/pivoting to a document ("breach
    # THE AGREEMENT", "under THE AGREEMENT") — the same false-positive shape
    # wiki._ORDINARY_TYPE_USAGE_RE already carves out one layer down inside
    # _carryover_scope. But this gate runs BEFORE that function is ever
    # called, so a phrase matching _VAGUE_DOC_PATTERN never reached
    # _carryover_scope for its own exception to apply. Re-check the same
    # exception here directly: an ordinary-usage phrase should still be
    # allowed through to the carryover check below, same as if
    # _VAGUE_DOC_PATTERN hadn't matched at all. Confirmed live: "What if they
    # breach the agreement?", asked mid-thread on an already-pinned document,
    # fell through to classify_query() and re-asked which document.
    # Same reasoning again for a demonstrative backreference to a specific
    # filing type already established ("that petition", "that affidavit") —
    # wiki._DEMONSTRATIVE_BACKREF_RE carves this out one layer down inside
    # _carryover_scope, but "affidavit" (among others in its word list) is
    # ALSO one of _VAGUE_DOC_PATTERN's own type words, so this outer gate
    # blocks the same phrase from ever reaching that exception. Confirmed
    # live: "What records are sought to be preserved under THAT PETITION?",
    # asked right after a Section 9 petition was the established scope, still
    # forced a disambiguation prompt.
    if (not wiki._VAGUE_DOC_PATTERN.search(state["question"])
            or wiki._ORDINARY_TYPE_USAGE_RE.search(state["question"])
            or wiki._DEMONSTRATIVE_BACKREF_RE.search(state["question"])):
        # Carryover must read THIS thread's messages, which live under the CHAT
        # session — not the wiki/doc session, which is shared by every thread
        # served off the same fixed main wiki and therefore holds every answer
        # ever generated against that corpus. resolve_scope already documents
        # this hazard at length and passes chat_session_id for exactly this
        # reason; this call was left on session_id, so in a brand-new thread
        # _carryover_scope found some unrelated older answer and reported a
        # scope "established" by a conversation that never happened — silently
        # skipping disambiguation on genuinely ambiguous questions (confirmed
        # live: a fresh-session "Redraft the indemnity clause" logged
        # "carryover scope established" with no prior turn in that thread).
        _chat_sid = state.get("chat_session_id") or state["session_id"]
        try:
            if wiki._carryover_scope(state["question"], _chat_sid):
                logger.info("[AGENT] disambiguation skipped (carryover scope established)")
                return {"needs_disambiguation": False}
            # Same reasoning for a comparative follow-up after a multi-document
            # turn ("which of those…"): the set is already established by the
            # conversation, so asking which document they mean is noise.
            if wiki._carryover_comparative_set(state["question"], _chat_sid):
                logger.info("[AGENT] disambiguation skipped (comparison set established)")
                return {"needs_disambiguation": False}
        except Exception as e:
            logger.error("Carryover-scope check failed, falling through to classify_query: %s", e)

    # A question that names BOTH sides of a matter ("damages Helios claimed in
    # its counterclaim against Aether") is not ambiguous — it identifies one
    # matter precisely — but classify_query below sees no document NUMBER and
    # no single dominant party, and asks anyway. wiki._resolve_docs_by_party_pair
    # resolves exactly this case deterministically, so consult it first and skip
    # the question when it pins a concrete document set. Same shape as the
    # carryover skips above: a deterministic resolver pre-empting an LLM
    # ambiguity judgment that has no way to see what it saw.
    try:
        if wiki._resolve_docs_by_party_pair(state["question"], state["session_id"]):
            logger.info("[AGENT] disambiguation skipped (party-pair scope resolved)")
            return {"needs_disambiguation": False}
    except Exception as e:
        logger.error("Party-pair disambiguation check failed: %s", e)

    # Loop breaker. Everything above resolves a document from the question; if
    # none of it fired, the prompt below asks the user. But that prompt is only
    # useful the FIRST time — the reply is appended to the question and re-run
    # through this same node, so a reply the matcher can't resolve produces the
    # identical prompt again, and again. Confirmed live: three consecutive
    # prompts on one question ("same document", then "the document I just spoke
    # about in the previous question"), none resolving, the user never answered.
    # Also confirmed on a question whose own subject collides with a corpus
    # entity ("Project Aurora" vs Aurora Cloud Computing Inc.), where even a
    # correct reply naming the agreement could not break the tie.
    #
    # After two unresolved rounds the question is one this matcher cannot pin,
    # and a third identical prompt has no path to an answer. Fall through to
    # ordinary corpus retrieval instead. Not silent: scope stays method="default"
    # with no target_docs, which generate_answer_node already discloses as "this
    # question named no document, so the entire corpus was searched" — the
    # honest description of what then happens.
    _dis_chat_sid = state.get("chat_session_id") or state["session_id"]
    if config.USE_DATABASE:
        try:
            _streak = db.count_trailing_disambiguations(_dis_chat_sid)
            if _streak >= _MAX_DISAMBIGUATION_ROUNDS:
                logger.warning(
                    "Disambiguation asked %d time(s) in a row on this thread without "
                    "resolving — falling through to corpus retrieval rather than "
                    "asking again: %r", _streak, state["question"][:80],
                )
                return {"needs_disambiguation": False}
        except Exception as e:
            logger.error("Disambiguation loop-breaker check failed: %s", e)

    _emit({"stage": "disambiguation", "status": "active", "message": "Checking document scope…"})
    try:
        result = wiki.classify_query(state["question"], state["session_id"])
    except Exception as e:
        logger.error("Disambiguation check failed: %s", e)
        result = {"needs_disambiguation": False}

    if result.get("needs_disambiguation"):
        # Deliberately does NOT name any candidate document here (no picker,
        # no list) — the reader must not learn what's in the corpus from a
        # disambiguation prompt alone. Resolution now depends entirely on the
        # user's own next message naming the document (counterparty, type, or
        # number); that free-text reply gets appended to the original question
        # and re-run through this same node, which resolves it via the
        # existing fuzzy entity/filename matching above — no new matching
        # logic, only a different UI shell around it.
        msg = ("Which agreement are you asking about? You can name the "
               "counterparty, the type of agreement, or a document number "
               "(e.g. \"the NDA with Acme\" or \"Service Agreement 2\") and "
               "I'll pull up the right one.")
        data = {"message": msg, "original_question": state["question"]}
        _emit({"stage": "disambiguation", "status": "done",
               "message": "Needs document selection",
               "type": "disambiguation", "payload": data})
        return {"needs_disambiguation": True, "disambiguation_data": data}

    return {"needs_disambiguation": False}


def _question_precisely_names_a_document(question: str, session_id: str) -> bool:
    """True if the question already pins down a document precisely enough that
    asking "which document?" would be pointless — even though it names it by
    counterparty rather than by the "type + number" pattern
    wiki._question_names_a_document alone recognises.

    check_clarification_node's own skip check used to be that narrower pattern
    only, so a question naming both parties of a document in full ("the NDA
    between Tata Passenger Electric Mobility Ltd. and Cirrus Battery
    Intelligence Pte. Ltd., what is the subject matter?") still reached the
    ambiguity LLM triage, which asked "summary of the entire NDA or specific
    clauses?" despite the question already stating precisely what it wanted —
    exactly the "names a specific document AND states what to do with it"
    case check_ambiguity's OWN prompt says should not need clarification. The
    LLM doesn't reliably apply that rule for a party-named (non-numbered)
    reference, so resolve it deterministically first, the same way
    classify_query's disambiguation gate already does.
    """
    try:
        if wiki._resolve_docs_by_party(question, session_id):
            return True
        if wiki._resolve_docs_by_party_pair(question, session_id):
            return True
        index = wiki._load_index(session_id)
        pages = index.get("pages", {})
        return bool(pages) and wiki._question_names_distinctive_entity(question, pages)
    except Exception as e:
        logger.warning("Precise-document check for clarification gate failed: %s", e)
        return False


@tracing.traced_node("check_clarification")
def check_clarification_node(state: QueryState) -> dict:
    # Must read the per-thread chat session, not the shared wiki/doc session —
    # build_conversation_context(state["session_id"]) pulled "recent conversation"
    # text from EVERY chat thread that has ever queried this wiki (confirmed live:
    # a fresh SHA-GridEdge question was answered about "joint venture deadlock /
    # arbitral institution", content from an unrelated earlier thread's question,
    # because generate_answer_node's own chat_session_id fallback never runs —
    # it only fires when conversation_context is still unset, and this node
    # always sets it first). resolve_scope already gets this right via its own
    # chat_session_id param; mirror that here.
    chat_sid = state.get("chat_session_id") or state["session_id"]
    if (state.get("is_followup") or wiki._question_names_a_document(state["question"], [])
            or not config.ENABLE_CLARIFICATION
            or _question_precisely_names_a_document(state["question"], state["session_id"])):
        conv = wiki.build_conversation_context(chat_sid)
        return {"needs_clarification": False, "conversation_context": conv}

    _emit({"stage": "clarification", "status": "active", "message": "Checking for ambiguity…"})
    conv = wiki.build_conversation_context(chat_sid)
    try:
        amb = wiki.check_ambiguity(state["question"], state["session_id"], conv)
    except Exception as e:
        logger.error("Ambiguity check failed: %s", e)
        amb = {"needs_clarification": False}

    if amb.get("needs_clarification"):
        data = {
            "message": amb.get("question", "Could you clarify your question?"),
            "options": amb.get("options") or [],
            "original_question": state["question"],
        }
        _emit({"stage": "clarification", "status": "done",
               "message": "Needs clarification",
               "type": "clarification", "payload": data})
        return {"needs_clarification": True, "clarification_data": data,
                "conversation_context": conv}

    return {"needs_clarification": False, "conversation_context": conv}


@tracing.traced_node("resolve_scope")
def resolve_scope_node(state: QueryState) -> dict:
    """Resolve where retrieval is allowed to search (Phase 2) — single document,
    a document family, or the whole corpus — in one place, before retrieval.

    Fail-open: any internal error yields a default "corpus" decision so the
    pipeline still answers exactly as it did pre-Phase-2.
    """
    logger.info("[AGENT] resolve_scope_node")
    try:
        decision = wiki.resolve_scope(state["question"], state["session_id"],
                                      chat_session_id=state.get("chat_session_id"))
    except Exception as e:
        logger.error("resolve_scope failed, defaulting to corpus: %s", e)
        decision = {"scope": "corpus", "target_docs": [], "target_family": None,
                    "is_broad": False, "confidence": 0.0, "method": "error"}
    decision = _apply_collection_scope(decision, state.get("collection_id"))
    logger.info("[AGENT] scope=%s family=%s broad=%s method=%s",
                decision.get("scope"), decision.get("target_family"),
                decision.get("is_broad"), decision.get("method"))
    _trace = tracing.get_trace()
    if _trace:
        _trace.log_scope_decision(decision)
    return {"scope_decision": decision}


def _apply_collection_scope(decision: dict, collection_id) -> dict:
    """Confine a resolved scope to a collection the user pinned in the Ask box.

    A pinned collection is a HARD boundary, not a hint: the user has said
    which documents this question is about, and that outranks anything the
    resolver inferred from the wording. So the collection is applied AFTER
    resolution rather than instead of it — the resolver still does the work
    of picking the right instrument, and this only removes what falls
    outside the pin:

      * documents it resolved that are in the collection  -> keep those
      * documents it resolved, none of them in the collection -> the
        resolver was looking outside the pin, so fall back to the whole
        collection rather than answering from documents the user excluded
      * family or corpus scope -> replace with the whole collection

    An empty or unreadable collection returns the decision untouched: a pin
    that resolves to nothing must not silently turn into "no documents",
    which would abstain on every question.
    """
    if not collection_id:
        return decision
    try:
        from services import collections as _collections
        from services import wikis as _wikis
        members = set(_collections.documents_in(_wikis.active_wiki_id(), int(collection_id)))
    except Exception as e:
        logger.error("Collection scope lookup failed for %r: %s", collection_id, e)
        return decision
    if not members:
        logger.warning("Pinned collection %r has no documents — ignoring the pin",
                       collection_id)
        return decision

    resolved = [d for d in (decision.get("target_docs") or []) if d]
    kept = [d for d in resolved if d in members]
    if kept:
        docs, how = kept, "collection-narrowed"
    else:
        docs, how = sorted(members), "collection"
    logger.info("[AGENT] collection %s pinned: %d document(s) in scope (%s)",
                collection_id, len(docs), how)
    return {**decision,
            "scope": "single_doc",
            "target_docs": docs,
            "target_family": None,
            "is_broad": False,
            "collection_id": int(collection_id),
            "method": f"{decision.get('method', 'default')}+{how}"}


@tracing.traced_node("retrieve_context")
def retrieve_context_node(state: QueryState) -> dict:
    logger.info("[AGENT] retrieve_context_node: intent=%s", state.get("intent"))
    _emit({"stage": "retrieving", "status": "active", "message": "Retrieving relevant pages…"})
    conv = state.get("conversation_context") or wiki.build_conversation_context(
        state.get("chat_session_id") or state["session_id"])
    hints = get_query_strategy(state["intent"])
    # Phase 2: forward the resolved scope's family filter + broad flag into
    # retrieval. Only "family" scope narrows the vector search; everything else
    # passes doc_family=None / force_broad as-decided, preserving prior behaviour.
    scope = state.get("scope_decision") or {}
    _fam = scope.get("target_family") if scope.get("scope") == "family" else None
    _force_broad = bool(scope.get("is_broad"))
    # single_doc scope resolved to concrete documents (e.g. a party-name content
    # match) — pin retrieval to them so a topically-similar page from another
    # agreement can't crowd out the document the user actually named.
    _force_docs = scope.get("target_docs") if scope.get("scope") == "single_doc" else None

    # Clause-number resolution. Ingest strips the leading number when it builds
    # page titles ("5. Return, Destruction..." -> "Return, Destruction..."), so
    # "what does clause 5 say" matches nothing even when the source is numbered.
    # clause_map (backfilled from the original PDFs) restores the link: on a hit
    # the retrieval query is augmented with the real heading so BM25/vector land
    # on the right page, and generate_answer discloses the mapping. On a miss the
    # lookup result still flows to validate_response so the note can say
    # "numbered 1-8, no clause 12" instead of something generic.
    clause_lookup = None
    _q_for_retrieval = state["question"]
    _cm = _RX_CLAUSE_NUMBER_Q.search(state["question"])
    if _cm:
        _cnum = next(g for g in _cm.groups() if g)
        _cdocs = (_force_docs or ([state["target_doc"]] if state.get("target_doc") else None)
                  or scope.get("target_docs") or [])
        if _cdocs:
            # Scope can resolve to several documents (real PDF + its Test_
            # stand-in); take the first one the map knows anything about.
            _hits, _nums, _cdoc = [], [], _cdocs[0]
            for _cand in _cdocs:
                try:
                    from services import wikis as _wikis
                    _wid = _wikis.active_wiki_id()
                    _hits = db.lookup_clause(_wid, state["session_id"], _cand, _cnum)
                    _nums = db.doc_clause_numbers(_wid, state["session_id"], _cand)
                except Exception as _cl_err:   # e.g. map not backfilled yet — degrade silently
                    logger.warning("clause_map lookup failed: %s", _cl_err)
                    break
                if _hits or _nums:
                    _cdoc = _cand
                    break
            clause_lookup = {"num": _cnum, "doc": _cdoc, "hits": _hits, "doc_numbers": _nums}
            if _hits:
                _headings = ", ".join(sorted({h["heading"] for h in _hits}))
                _q_for_retrieval = (f'{state["question"]} (in this document clause '
                                    f'{_cnum} is titled "{_headings}")')
                logger.info("[AGENT] clause_map hit: clause %s -> %s", _cnum, _headings)

    res = wiki.get_context(
        _q_for_retrieval, state["session_id"],
        target_doc=state.get("target_doc", ""), retrieval_hints=hints,
        exclude_cached_answers=state.get("exclude_cached_answers", False),
        doc_family=_fam, force_broad=_force_broad, force_docs=_force_docs,
        family_docs=(scope.get("target_docs") if scope.get("scope") == "family" else None),
    )
    titles = res.get("selected_titles", [])
    _emit({"stage": "pages_retrieved", "status": "done",
           "count": len(titles), "message": f"Retrieved {len(titles)} page(s)"})
    return {
        "wiki_context": res.get("context", ""),
        "selected_titles": titles,
        "retrieval_meta": {
            "bm25_count": res.get("bm25_count", 0),
            "page_selection_usage": res.get("page_selection_usage", {}),
        },
        "conversation_context": conv,
        "clause_lookup": clause_lookup,
    }


@tracing.traced_node("generate_answer")
def generate_answer_node(state: QueryState) -> dict:
    logger.info("[AGENT] generate_answer_node: intent=%s pages=%d",
                state.get("intent"), len(state.get("selected_titles", [])))
    intent = state["intent"]
    label = INTENT_LABELS.get(intent, "Factual")
    _emit({"stage": "generating", "status": "active",
           "intent": intent, "prompt_type": intent,
           "message": f"Generating {label.lower()} answer…"})
    meta = state.get("retrieval_meta") or {}
    # The question named a document by pattern ("service agreement 1") that
    # check_disambiguation_node could not confirm exists in this corpus.
    # Previously appended as a note to the question text itself — verified
    # live that placement gets partial compliance at best (the {question}
    # placeholder sits mid-prompt, right before a rigid REQUIRED OUTPUT FORMAT
    # directive every per-intent template ends with, which the model follows
    # very literally and which crowds out a free-form instruction competing
    # for attention at that position). Passed through as its own parameter
    # instead so wiki.generate_answer can inject it via house_rules_block,
    # which every template substitutes at the TOP, before the RULES section.
    # The question named no document and the scope was inherited from the
    # document already under discussion (wiki.resolve_scope → method="carryover").
    # That inference is invisible in the answer text, so disclose it explicitly:
    # the user must be able to see WHY they got an answer about a document they
    # never named, and how to override it.
    _scope = state.get("scope_decision") or {}
    _scope_note = ""
    if _scope.get("method") == "carryover" and _scope.get("target_docs"):
        _carried = wiki._norm_doc_name(_scope["target_docs"][0])
        _scope_note = (
            f"This question named no document, so it was answered using "
            f"\"{_carried}\" — the document already under discussion in this "
            f"conversation. To scope it differently, name a document explicitly, "
            f"or use \"across all …\" to search every document."
        )
    elif _scope.get("method") == "carryover-set" and _scope.get("target_docs"):
        # Same disclosure duty as the single-document carryover above: the
        # comparison was silently limited to the documents the previous turn
        # covered, which the user never named in this question.
        _set_docs = _scope["target_docs"]
        _set_names = ", ".join(f'"{wiki._norm_doc_name(d)}"' for d in _set_docs[:4])
        _more = f" and {len(_set_docs) - 4} more" if len(_set_docs) > 4 else ""
        _scope_note = (
            f"This question named no document, so it was answered by comparing the "
            f"{len(_set_docs)} document(s) already under discussion in this "
            f"conversation — {_set_names}{_more}. To compare a different set, name "
            f"the documents explicitly, or use \"across all …\" to search every document."
        )
    elif (_scope.get("method") == "default" and not _scope.get("target_docs")
            and not _scope.get("unresolved_party")):
        # No document was ever named in this conversation (not carried over from
        # a prior turn, not a named-but-ambiguous party — genuinely never named),
        # so retrieval searched the whole corpus with no document pinned. If the
        # answer below reads as if it's about one specific document, that's an
        # artifact of which pages happened to rank highest, not a confirmed
        # match — disclose it, same duty as the carryover note above. Confirmed
        # live: a 4-turn thread with no document ever named silently answered as
        # if about "Service Agreement 1" (never mentioned anywhere in the
        # conversation) with zero indication a guess had been made.
        _scope_note = (
            "This question named no document, so the entire corpus was searched "
            "with nothing pinned to one specific agreement. If the answer below "
            "reads as if it concerns a single document, verify that against the "
            "References section below — no document was confirmed as the one you "
            "meant. Name a document explicitly to get a scoped answer."
        )

    # The question identified its document by a description (a party's
    # registered-office block) that several documents state identically, so it
    # has as many valid answers as it has matches. resolve_scope pinned them all
    # rather than letting the entity branch collapse them to a top-ranked winner
    # (wiki._detect_description_ambiguity). Three separate channels, each doing a
    # job the others can't: the DIRECTIVE reshapes the answer itself (one value
    # per document, each attributed), the WARNING is a deterministic banner the
    # model cannot omit, and the NOTE explains how to resolve it next turn.
    _ambiguity_directive = ""
    _amb = _scope.get("ambiguous_match") or {}
    if len(_amb.get("docs") or []) > 1:
        _amb_docs = _amb["docs"]
        _amb_desc = _amb.get("description", "")
        _amb_names = ", ".join(f'"{wiki._norm_doc_name(d)}"' for d in _amb_docs)
        _ambiguity_directive = (
            f"The question identifies its document by {_amb_desc} \u2014 which fits "
            f"{len(_amb_docs)} documents in this corpus equally well: "
            f"{_amb_names}. Nothing else in the question distinguishes them, so the "
            f"question has {len(_amb_docs)} equally valid answers, not one. Do NOT pick "
            f"the best-matching document and answer as if it were the only match, and do "
            f"NOT merge their values into one figure. Instead: state in the FIRST "
            f"SENTENCE that the description matches {len(_amb_docs)} documents and name "
            f"them, then answer the question SEPARATELY FOR EACH ONE, with every value "
            f"carrying the name of the document it came from. Close by saying what would "
            f"distinguish them (a document number, the counterparty, or the subject "
            f"matter) so the user can narrow it. If the retrieved excerpts only support "
            f"an answer for some of these documents, say which ones you could answer for "
            f"and which you could not \u2014 never let a missing excerpt turn this back into "
            f"a single-document answer."
        )
        _scope_note = (
            f"This question identified its document by {_amb_desc}, which fits "
            f"{len(_amb_docs)} documents equally \u2014 {_amb_names} \u2014 so all of them "
            f"were searched and each is answered separately above. To pin one, "
            f"name it by number (e.g. \"Service Agreement 2\") or by its subject "
            f"matter."
        )

    # The clause map resolved the asked-for number to a real section of this
    # document. Two channels, deliberately separate: scope_note is DISPLAY-ONLY
    # (wiki appends it under the finished answer), so it gets the short
    # disclosure; the instruction goes via clause_directive, which wiki prepends
    # to the prompt itself — passing the instruction through scope_note was
    # verified live to change nothing, because the model never sees it.
    _cl = state.get("clause_lookup") or {}
    _clause_directive = ""
    if _cl.get("hits"):
        _cl_names = sorted({x["heading"] for x in _cl["hits"]})
        # Two renderings of the same headings, deliberately not shared: the
        # scope_note copy is human-facing display text with quote marks for
        # readability, never re-examined by anything. The directive copy feeds
        # the GENERATION PROMPT, and _clause_directive's first version quoted
        # the heading in its own worked example ('"Clause 5 (\"Heading\")..."')
        # — the model followed the example literally, wrapped the heading in
        # quotation marks in its real answer, and _verify_answer_citations
        # flagged that quote as an unverified excerpt: it checks quoted spans
        # against _known_page_titles(), which stores the FULL title including
        # the " – Party (Doc)" suffix that clause_map's bare heading omits, so
        # the two never matched (confirmed live: real content, correct
        # citation, still flagged as a false-positive citation warning).
        # Plain parentheses read just as naturally and never enter the
        # quote-verification path at all, which is the safer fix — it doesn't
        # touch _known_page_titles' matching logic, which handles several
        # other confirmed edge cases (unicode hyphens, trailing punctuation)
        # that a change here risks disturbing.
        _cl_heads = ", ".join(f'"{h}"' for h in _cl_names)
        _cl_heads_plain = ", ".join(_cl_names)
        _scope_note = ((_scope_note + " ") if _scope_note else "") + (
            f"In this document's own numbering, clause {_cl['num']} is the section "
            f"titled {_cl_heads}."
        )
        _clause_directive = (
            f"The question asks about clause {_cl['num']} by number. In this "
            f"document the clause numbers are not printed in the page text, but "
            f"clause {_cl['num']} IS present: it is the section titled "
            f"{_cl_heads_plain} (this is a section name, not a verbatim quote — do "
            f"not put it in quotation marks). Do NOT answer 'not addressed' — "
            f"answer the question from that section's content, and open the "
            f"answer by naming the mapping without quotation marks, e.g. "
            f"Clause {_cl['num']} ({_cl_heads_plain}) provides..."
        )

    # The question named a specific counterparty the pipeline could not pin to
    # one document (the name spans too many documents to disambiguate), so scope
    # fell through to a broad corpus search. The answer may be sourced from a
    # same-type sibling rather than the exact document the user meant — warn.
    _scope_warning = ""
    # An ambiguous identifying description outranks the warning below: that one
    # says "this MIGHT be the wrong document", whereas here it is established
    # that no single document is the right one. Set first, and the weaker
    # warning defers to it.
    if _ambiguity_directive:
        _scope_warning = (
            f"Your question identifies its document by {_amb_desc}, which fits "
            f"{len(_amb_docs)} documents equally ({_amb_names}), so it does not "
            f"single one out. The answer below is given separately for each; "
            f"there is no single value that answers this question. Name the document by "
            f"number, counterparty, or subject matter to get one answer."
        )
    # Set by the corpus-wide default AND by its family-narrowed variant: limiting
    # the search to the instrument the question named does not establish WHICH
    # document of that type was meant, so the disclosure duty is unchanged.
    _unresolved = _scope.get("unresolved_party") or ""
    if _unresolved and not _scope_warning:
        _scope_warning = (
            f"The question named \"{_unresolved}\" but that party appears in several "
            f"documents, so no single document could be confirmed as the one you meant. "
            f"The answer below was drawn from a broad search and may reflect a "
            f"different document of the same type — check the References section, and "
            f"if it's the wrong one, name the document by its number (e.g. \"NDA 7\") "
            f"or its distinctive counterparty."
        )

    # A numbered document reference matched BOTH the real document and a synthetic
    # Test_* stand-in of the same number — both were pinned into context and the
    # answer may have been drawn from the fictional stand-in rather than the real
    # document (confirmed live: a GridEdge SHA question answered from Test_SHA_01's
    # invented parties). Warn so the reader can verify which document the facts
    # actually came from.
    _collisions = _scope.get("doc_collisions") or []
    if _collisions and not _scope_warning:
        _names = ", ".join(f'"{c}"' for c in _collisions[:3])
        _scope_warning = (
            f"For {_names}, this corpus contains BOTH a real document and a "
            f"synthetic \"Test_\" stand-in of the same number — both were searched, "
            f"so some facts below (party names, figures, clause numbers) may come "
            f"from the fictional stand-in rather than the real document. Verify each "
            f"cited figure against the document named in the References section "
            f"before relying on it."
        )

    try:
        wr = wiki.generate_answer(
            state["question"], state.get("wiki_context", ""),
            state.get("selected_titles", []), state["session_id"],
            meta.get("bm25_count", 0), meta.get("page_selection_usage", {}),
            state.get("conversation_context", ""), intent=intent,
            unconfirmed_doc_reference=state.get("unconfirmed_doc_reference", False),
            scope_note=_scope_note, scope_warning=_scope_warning,
            clause_directive=_clause_directive,
            ambiguity_directive=_ambiguity_directive,
        )
    except Exception as e:
        logger.error("Answer generation failed (%s): %s", type(e).__name__, e)
        wr = {"answer": f"Wiki error: {type(e).__name__}: {e}",
              "scope_method": "", "scope_docs": [],
              "pages_used": [], "files_used": [], "confidence_score": 0}

    wr["intent"] = intent
    wr["intent_label"] = label
    wr["intent_confidence"] = state.get("intent_confidence", 0.0)
    wr["intent_method"] = state.get("intent_method", "")
    # Record HOW this turn's scope was resolved. app.py persists it to
    # chat_messages.metadata so the NEXT turn's wiki._carryover_scope can inherit
    # a genuinely-resolved single-document scope instead of guessing from a file
    # count (see wiki._CARRYOVER_FROM_METHODS).
    wr["scope_method"] = _scope.get("method", "")
    wr["scope_docs"] = _scope.get("target_docs") or []
    # Private key, not part of the public answer shape — app.py pops this off
    # before the SSE payload reaches the frontend. Exists only so the RAG query
    # log (_log_rag_query) can record what was actually retrieved; previously
    # app.py had no access to the raw context at all and hardcoded "" there,
    # silently making the logged "contexts" field meaningless for every query
    # through this streaming path (confirmed: 354/446 logged records showed 0
    # contexts, including substantive multi-page answers where retrieval had
    # genuinely pulled real content).
    wr["_debug_context"] = state.get("wiki_context", "")
    return {"answer_result": wr}


_GROUNDING_PROMPT = """\
You are a legal QA auditor. Compare the ANSWER against the CONTEXT and determine \
how well the answer's FACTUAL CLAIMS are grounded in the provided documents.

INTENT: {intent}

CONTEXT (excerpts from source documents):
{context}

---
QUESTION: {question}

---
ANSWER TO CHECK:
{answer}

---
IMPORTANT DISTINCTION — only flag FACTUAL claims as ungrounded:
- FACTUAL claims: clause numbers, party names, dates, amounts, obligations stated in the document.
- PROFESSIONAL ANALYSIS: risk classifications, market-practice comparisons, gap identification, \
suggested changes, negotiation advice. These are EXPECTED legal judgment, NOT ungrounded claims.
For "risk_assessment" and "drafting" intents, the answer is supposed to go beyond the text with \
professional analysis. Only penalize fabricated facts (wrong clause numbers, invented provisions, \
incorrect party names), not analytical conclusions or recommendations.
- ABSENCE/NEGATIVE FINDINGS ARE NOT UNGROUNDED CLAIMS (CRITICAL): When the answer states that a \
clause, right, or topic is "not covered", "not addressed", "not present", "not stated", or similar in \
the provided excerpts, that is the CORRECT, expected behavior when the context genuinely lacks the \
information — do NOT flag the absence-statement itself as an unverifiable or ungrounded fact. An \
honest "the excerpts do not contain X" is high-grounding behavior (it's not claiming X is true or \
false about the real document — only that X isn't in the retrieved excerpts), never a fabrication. \
Only flag it as ungrounded if the answer goes further and asserts something affirmative not supported \
by context (e.g. inventing a reason FOR the absence, or claiming the document was fully reviewed when \
it wasn't).

Respond with ONLY valid JSON, no other text — no preamble, no step-by-step reasoning, no restating \
the answer or context before the JSON. For a long, many-source answer, do not verify every claim \
narratively in your head one by one before writing output — spot-check the most load-bearing claims \
(specific numbers, dates, party names) and write the JSON directly. List at most 8 ungrounded_claims \
even if more exist (the worst offenders are enough to make the point). Go straight to the JSON block \
below as your first output:
{{
  "grounding_score": <0-100>,
  "ungrounded_claims": ["<fabricated fact not in context>", ...],
  "summary": "<one sentence assessment>"
}}

Scoring guide:
- 90-100: All factual claims traceable to context; analysis is reasonable
- 70-89: Most facts grounded, minor factual extrapolations
- 50-69: Some factual claims lack context support
- 0-49: Significant fabricated facts"""


_CLAIM_PATTERNS = [
    re.compile(r'\$\s?\d[\d,]*(?:\.\d+)?'),                                    # currency
    re.compile(r'\b\d[\d,]*(?:\.\d+)?\s?%'),                                   # percentages
    re.compile(r'\b\d+\s*(?:day|days|month|months|year|years|week|weeks)\b', re.I),  # durations
    re.compile(r'\b(?:January|February|March|April|May|June|July|August|'
               r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', re.I),  # long dates
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),                                # numeric dates
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),                                      # ISO dates
    # doc refs — the identifier MUST start with a digit ("Clause 5.3") or an
    # uppercase roman numeral ("Section IV"), never a bare following word.
    # Without this, "Clause"/"Schedule" followed by ANY word matched ordinary
    # prose continuations like "clause allowing", "schedule of", "clause
    # tailored" as if they were checkable document references (confirmed live:
    # a risk-assessment answer's own sentence "the explicit liberty clause
    # allowing plaintiffs..." produced a fake claim "clause allowing" that
    # then, correctly but meaninglessly, never matched context — one of
    # several junk matches that dragged a well-grounded answer's deterministic
    # score down to 20%). Split into two case-sensitivity-matched patterns
    # rather than one re.I pattern: under re.I, a lowercase "in" or "is" would
    # satisfy a case-insensitive [IVXLC] lookahead (I matches i), letting the
    # exact same junk back in through the roman-numeral branch.
    re.compile(r'\b(?:Section|Clause|Article|Exhibit|Schedule|Appendix)\s+(?=\d)[\dA-Za-z.]+\b', re.I),  # doc refs (numeric)
    re.compile(r'\b(?:Section|Clause|Article|Exhibit|Schedule|Appendix)\s+(?=[IVXLC]+\b)[IVXLC]+\b'),   # doc refs (roman numeral, case-sensitive)
    re.compile(r'\b(?:[A-Z][a-zA-Z&\'-]+(?:\s+[A-Z][a-zA-Z&\'-]+){1,3})\b'),    # multi-word proper nouns
]

# Words that pattern-match as "proper nouns" (capitalized phrases) but are
# generic legal boilerplate, not a fact that could be fabricated — checking
# these against context is noise, not signal.
_CLAIM_STOPWORDS = {
    "the agreement", "this agreement", "the document", "this document",
    "the parties", "the party", "the company", "the effective date",
    "not covered", "not applicable", "not stated", "not specified",
}


# A heading line marking the start of a Recommendations-type section — the
# ASSESSMENT_PROMPT's own rules explicitly exempt this content from grounding
# ("Your analysis may go beyond the text; your facts may not"), but the model's
# own invented sub-headings for its suggested framework ("Dynamic Injunction
# Protocol", "Domain Additions", proposed cadences like "10 business days")
# still pattern-match as checkable claims, since the extractor has no concept
# of "which section of the answer is this in." Confirmed live: a well-grounded
# risk-assessment answer scored 20% deterministically because most of its
# extracted "claims" were the model's own Recommendations-section headings and
# proposed numbers, none of which were ever meant to appear in the source
# context. Cutting claim extraction off at this heading is a narrower,
# lower-risk alternative to trying to classify every sentence — the FACTS
# section (which precedes Recommendations in every ASSESSMENT/OBLIGATION/
# COMPARISON template) still gets fully checked.
_RECOMMENDATIONS_HEADING_RE = re.compile(
    r'(?im)^[ \t]*(?:\d+[.)]\s*|[*\-•]\s*)?\**'
    r'(?:recommendations?|recommended\s+stance|next\s+steps|'
    r'practical\s+(?:steps|caveats|next\s+steps|impact\s+and\s+decision\s+guidance)|'
    r'action\s+items|concrete\s+next\s+steps|what\s+to\s+do\s+next|operational\s+playbook|'
    r'key\s+(?:negotiation\s+points|concrete\s+drafting\s+positions)|'
    r'acceptance\s*/\s*rejection\s*/\s*negotiation\s+stance)\**'
    r'(?:\s*\([^)]{0,60}\))?[ \t]*:?[ \t]*$'
)


def _extract_claims(text: str) -> list[str]:
    """Pull out checkable factual atoms (amounts, dates, durations, doc
    references, proper nouns) from an answer, deduplicated and stripped of
    generic boilerplate phrases that aren't really factual claims.

    Stops scanning at the first Recommendations-type heading — see
    _RECOMMENDATIONS_HEADING_RE — so the model's own suggested framework
    (proposed cadences, invented sub-heading names) never gets checked against
    the source context as if it were a claim about what the context says."""
    cutoff = _RECOMMENDATIONS_HEADING_RE.search(text)
    if cutoff:
        text = text[:cutoff.start()]
    claims: list[str] = []
    seen = set()
    for pattern in _CLAIM_PATTERNS:
        for m in pattern.finditer(text):
            claim = m.group(0).strip()
            key = claim.lower()
            if key in seen or key in _CLAIM_STOPWORDS or len(key) < 3:
                continue
            seen.add(key)
            claims.append(claim)
    return claims


def _claim_grounded(claim: str, context_norm: str) -> bool:
    """Normalize and substring-match a single claim against the flattened
    context. Exact-enough for numbers/dates; good-enough for proper nouns
    (a real fabrication swaps the name/number outright, it doesn't just
    reformat whitespace)."""
    norm = re.sub(r'\s+', ' ', claim.strip().lower())
    norm = re.sub(r'[,.]', '', norm)
    return norm in context_norm


def _deterministic_grounding(context: str, answer: str) -> dict:
    """Zero-cost grounding score: extract factual claims from the answer and
    check each one's literal presence in the retrieved context. Catches the
    most common fabrication shape (wrong number/date/party name) without an
    LLM call. Cannot catch claims that are semantically wrong while reusing
    real numbers/names already in context — that gap is what the LLM
    escalation in _check_grounding_hybrid exists for."""
    claims = _extract_claims(answer)
    if not claims:
        return {"grounding_score": None, "ungrounded_claims": [], "summary":
                 "No checkable factual claims (amounts/dates/names) found in answer.",
                 "total_claims": 0, "method": "deterministic"}

    context_norm = re.sub(r'\s+', ' ', context.lower())
    context_norm = re.sub(r'[,.]', '', context_norm)
    ungrounded = [c for c in claims if not _claim_grounded(c, context_norm)]
    score = round(100 * (len(claims) - len(ungrounded)) / len(claims))
    return {
        "grounding_score": score,
        "ungrounded_claims": ungrounded[:8],
        "summary": f"{len(claims) - len(ungrounded)}/{len(claims)} checkable claims verified against context (deterministic).",
        "total_claims": len(claims),
        "method": "deterministic",
    }


# Escalation bands per intent — comparison/obligation/risk_assessment answers
# synthesize across many sources and are more prone to claims that are
# factually-real-but-logically-wrong (right number, wrong clause it's
# attached to), which the deterministic pass can't detect. Bias those
# intents toward escalating to the LLM judge more readily: lower min-claims
# floor and a wider "ambiguous" band around the score.
#
# escalate_high was 95 for those three, so nearly EVERY answer escalated (real
# answers rarely score above 95 deterministically) — the "deterministic first
# to save cost" barely saved anything for them, and each escalation is a full
# (sometimes reasoning-truncated, retried) LLM call for a DECORATIVE score that
# gates nothing. Lowered to 85: an answer that already verifies ≥85% of its
# checkable claims is well-grounded enough to trust without paying for the
# judge.
#
# escalate_low was previously 35 with the STATED intent that "a low
# deterministic score on these paraphrase-heavy intents is often just
# reworded-but-real quotes, so surfacing it raw would mislead — those still
# escalate to get a truer number." But the escalation condition is
# `escalate_low < score < escalate_high` — a score AT OR BELOW escalate_low
# never satisfies that and is trusted raw, the opposite of the stated intent.
# Confirmed live: a well-grounded risk-assessment answer scored 20%
# deterministically (mostly claim-extraction noise, see _extract_claims) and
# was never escalated because 20 < 35; once escalated by hand to the LLM judge
# it scored 92%. Raised to 15 — only a near-zero deterministic score (which
# realistically means almost every extracted claim failed, not just noisy
# extraction) is now trusted without a second look; anything above that gets
# the LLM judge's nuance. Costs more escalations for these three intents, by
# design — the whole point is these are the intents where a low raw score is
# least likely to mean what it says.
_GROUNDING_THRESHOLDS = {
    "default":         {"min_claims": 3, "escalate_low": 50, "escalate_high": 85},
    "comparison":      {"min_claims": 2, "escalate_low": 15, "escalate_high": 85},
    "obligation":      {"min_claims": 2, "escalate_low": 15, "escalate_high": 85},
    "risk_assessment": {"min_claims": 2, "escalate_low": 15, "escalate_high": 85},
}


def _check_grounding_hybrid(question: str, context: str, answer: str, intent: str = "factual") -> dict:
    """Deterministic grounding check first (free); escalate to the LLM judge
    only when the deterministic signal is too thin or too ambiguous to trust."""
    if not context or not answer or len(answer) < 20:
        return {"grounding_score": None, "ungrounded_claims": [], "summary": "Skipped — insufficient content."}

    # Drafting answers are NEW clause language written to order, not facts
    # retrieved from the context — so claim-by-claim matching against the
    # source is structurally the wrong metric and always scores low
    # (confirmed live: good drafted clauses scored 25–46% grounding, which
    # reads as a quality warning when it is nothing of the sort). The part of
    # a drafting answer that CAN be verified — its "Source Clauses" verbatim
    # quotes — is already checked by _verify_answer_citations in
    # wiki.generate_answer, so skip the grounding score here rather than
    # surface a misleading number.
    if intent == "drafting":
        return {"grounding_score": None, "ungrounded_claims": [],
                "summary": "Not scored — drafted clause language is newly authored text, not retrieved facts (source-clause citations are verified separately).",
                "method": "skipped-drafting"}

    det = _deterministic_grounding(context, answer)
    thresholds = _GROUNDING_THRESHOLDS.get(intent, _GROUNDING_THRESHOLDS["default"])
    score = det["grounding_score"]

    needs_escalation = (
        det["total_claims"] < thresholds["min_claims"]
        or (score is not None and thresholds["escalate_low"] < score < thresholds["escalate_high"])
    )

    if needs_escalation:
        logger.info(
            "[GROUNDING] Deterministic pass inconclusive (claims=%d, score=%s) — escalating to LLM judge",
            det["total_claims"], score,
        )
        llm_result = _check_grounding(question, context, answer, intent=intent)
        if llm_result.get("grounding_score") is not None:
            llm_result["method"] = "llm"
            return llm_result
        # LLM check failed outright — fall back to the deterministic read
        # rather than surfacing nothing.
        return det

    logger.info("[GROUNDING] Deterministic pass sufficient (claims=%d, score=%s) — skipped LLM call",
                det["total_claims"], score)
    return det


def _check_grounding(question: str, context: str, answer: str, intent: str = "factual") -> dict:
    """LLM-based grounding check — verifies answer claims against context."""
    if not context or not answer or len(answer) < 20:
        return {"grounding_score": None, "ungrounded_claims": [], "summary": "Skipped — insufficient content."}

    ctx = context

    # NOTE: previously truncated to answer[:2000] — long comparison/obligation
    # answers put their References/quotes section (the exact content this check
    # exists to catch fabrication in) near the END of the answer, past 2000
    # chars, so the checker never even saw it. Check the full answer.
    prompt = _GROUNDING_PROMPT.format(
        context=ctx,
        question=question,
        answer=answer,
        intent=intent,
    )
    try:
        raw, _usage = llm.ask(prompt, fast=False, max_tokens=config.MAX_TOKENS_GROUNDING_CHECK)
        # Same truncation failure mode as generate_answer's answer-generation
        # pass (services/wiki.py): a large context/answer (e.g. a 35-row
        # cross-document obligation table) can make the judge's own reasoning +
        # JSON output exceed the token budget. Confirmed live in three
        # different shapes — sometimes the whole budget goes to hidden
        # reasoning and raw comes back completely empty, sometimes it writes
        # ~900 chars of real JSON but still gets cut off mid-string before the
        # closing brace, and for genuinely large broad-synthesis answers
        # (15+ source documents, 14k+ char answer) even a single doubling
        # still spends the entire retry budget on hidden reasoning with zero
        # visible output — confirmed live that a 2x retry (1800 tokens) still
        # truncated empty, but 4x (3600) succeeded cleanly with ~2400
        # completion tokens used. A truncated response can never be trusted
        # to be valid JSON either way, so keep escalating the budget (doubling
        # each time, capped at 4 attempts total) whenever finish_reason ==
        # "length", regardless of whether the partial output looks empty or
        # substantial. Confirmed live on a 63-page/43.9k-char-context answer:
        # 900/1800/3600 all came back raw_len=0 with completion_tokens exactly
        # equal to the budget (the model spent 100% of it on hidden reasoning
        # tokens with zero visible output) — only 7200 succeeded (finish_reason
        # stop, 4226 completion tokens, 443 visible chars). One more doubling
        # step than before is needed to reach that budget.
        attempt_budget = config.MAX_TOKENS_GROUNDING_CHECK
        attempts = 0
        while _usage.get("finish_reason") == "length" and attempts < 3:
            attempt_budget *= 2
            attempts += 1
            logger.warning("Grounding check truncated (finish_reason=length) — retrying with budget=%d (attempt %d)",
                            attempt_budget, attempts)
            raw, _usage = llm.ask(prompt, fast=False, max_tokens=attempt_budget)
        import json as _json
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = _json.loads(raw[start:end])
            score = result.get("grounding_score")
            if isinstance(score, (int, float)):
                result["grounding_score"] = max(0, min(100, int(score)))
            return result
    except Exception as e:
        logger.warning("Grounding check failed: %s", e)
    return {"grounding_score": None, "ungrounded_claims": [], "summary": "Check failed."}


# ---------------------------------------------------------------------------
# Deterministic term-presence check
#
# Exists because the LLM grounding check above cannot be trusted as a gate: it
# scored a confirmed fabrication (a comparison answer that asserted an
# indemnification clause the retrieved pages never mention — 0 occurrences in
# context, 8 in the answer) at 90%, and its summary explicitly called the
# invented clause accurate. The judge hallucinated alongside the generator.
#
# A string count cannot be talked into anything, which is exactly why it catches
# what the judge missed. It only ever ATTACHES A WARNING — it never rewrites or
# suppresses an answer, because the corpus already has a real "wrongly said the
# document is silent" failure mode and a suppressing gate would worsen it.
# ---------------------------------------------------------------------------

# Curated, not extracted from the question. Picking "the topic" out of arbitrary
# phrasing is a guess; matching against a fixed legal vocabulary is not. Each
# entry lists the alternates real drafting uses instead of the obvious word —
# without these, an agreement that says "hold harmless" and never "indemnify"
# would be scored as silent on indemnity.
# Alternates are kept as a LIST, not one alternation, so each can be counted
# separately. That is what makes the under-read signal precise: the useful thing
# to tell a reader is not "this topic exists somewhere" but "the pages say
# ARBITRATION and your answer never mentions it".
# Every alternate carries its own plain-English label. Paired rather than derived,
# because the under-read message quotes the wording back to the reader and a raw
# pattern ("applicable\\s+law") in the UI is worse than no message at all.
_TOPIC_TERMS: dict[str, list[tuple[str, str]]] = {
    "indemnity": [
        (r"indemnif\w*", "indemnification"), (r"indemnit\w*", "indemnity"),
        (r"hold\s+harmless", "hold harmless")],
    "payment": [
        (r"payment\w*", "payment"), (r"\bpay\b", "pay"), (r"\bfees?\b", "fees"),
        (r"invoice\w*", "invoicing"), (r"consideration", "consideration"),
        (r"remunerat\w*", "remuneration")],
    "termination": [
        (r"terminat\w*", "termination"), (r"expir\w*", "expiry"),
        (r"exit\s+(?:right|mechanic|trigger)\w*", "exit mechanics"),
        (r"(?:call|put)\s+option", "call/put options"),
        (r"wind[-\s]?(?:up|down)", "wind-up")],
    "confidentiality": [
        (r"confidential\w*", "confidentiality"),
        (r"non[-\s]?disclosure", "non-disclosure"), (r"secrec\w*", "secrecy")],
    "governing law": [
        (r"governing\s+law", "governing law"), (r"governed\s+by", "governed by"),
        (r"applicable\s+law", "applicable law"),
        (r"choice\s+of\s+law", "choice of law")],
    "dispute resolution": [
        (r"dispute\w*", "disputes"), (r"arbitrat\w*", "arbitration"),
        (r"mediat\w*", "mediation"), (r"litigat\w*", "litigation"),
        (r"jurisdiction", "jurisdiction"), (r"\bforum\b", "forum"),
        (r"escalat\w*", "escalation"), (r"deadlock", "deadlock")],
    "liability": [
        (r"liabilit\w*", "liability"), (r"liable", "liable"),
        (r"\bcap(?:ped|s)?\b", "liability cap")],
    "warranty": [
        (r"warrant\w*", "warranties"), (r"represent\w*", "representations")],
    "force majeure": [
        (r"force\s+majeure", "force majeure"), (r"act\s+of\s+god", "act of God")],
    "intellectual property": [
        (r"intellectual\s+property", "intellectual property"), (r"\bIP\b", "IP"),
        (r"copyright", "copyright"), (r"trademark", "trade marks"),
        (r"patent", "patents")],
    "non-compete": [
        (r"non[-\s]?compet\w*", "non-compete"),
        (r"restraint\s+of\s+trade", "restraint of trade"),
        (r"exclusivit\w*", "exclusivity")],
    "assignment": [
        (r"assign\w*", "assignment"), (r"novat\w*", "novation"),
        (r"transfer\s+of\s+rights", "transfer of rights")],
    "notice": [
        (r"notice\s+period", "notice period"), (r"written\s+notice", "written notice"),
        (r"notif\w*", "notification")],
}

# An answer conceding it found nothing. Used in both directions: it suppresses a
# fabrication flag (conceding absence is not asserting a clause) and it is the
# trigger for the under-read flag.
_RX_ABSENCE_CLAIM = re.compile(
    r"not\s+(?:addressed|covered|specified|stated|mentioned|provided|present|included|found)"
    # "do not contain" as well as "does not" — the plural is what real answers use
    # when they talk about "the documents", and missing it made the check silent
    # on exactly those phrasings.
    r"|(?:does|do|did)\s+not\s+(?:address|cover|specify|state|mention|contain|include|provide)"
    r"|(?:is|are)\s+silent\s+on"
    r"|\bno\s+(?:such\s+|explicit\s+|express\s+|defined\s+)*"
    r"(?:clause|provision|section|mechanism|reference|requirement|term)s?\b"
    r"|not\s+in\s+the\s+(?:retrieved|provided)\s+(?:context|document|pages)",
    re.IGNORECASE,
)

# A positive assertion about what a document says, as opposed to a mention of the
# topic in passing. Operative legal verbs are the tell.
_RX_OPERATIVE = re.compile(
    r"\bshall\b|\bmust\b|\bwill\b|\bagrees?\s+to\b|\bis\s+required\b|\bundertakes?\b"
    r"|\bentitled\s+to\b|\bobligated\b|\bcovenants?\b|[\"“][^\"”]{20,}[\"”]",
    re.IGNORECASE,
)

_RX_CLAUSE_NUMBER_Q = re.compile(
    r"\bclause\s+(\d+(?:\.\d+)*)|\bsection\s+(\d+(?:\.\d+)*)|\barticle\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)


def _check_term_presence(question: str, context: str, answer: str,
                         clause_lookup: dict | None = None) -> tuple[str, str, dict]:
    """Compare the question's legal topics against context and answer by string count.

    Returns (alert, note, facts). `facts` carries topics_asked / topics_found —
    of the legal topics the question raised, how many appear anywhere in the
    retrieved pages. It is a count, not a judgement, which is the point: every
    predicted score in this pipeline correlates ~0 with an independent judge, so
    the UI needs at least one number that is true by construction.

    alert and note are two severities, because they mean different things to a
    reader and merging them made the milder one overclaim:

    - alert (fabrication): the topic appears in no retrieved page, yet the answer
      makes an operative statement about it. Part of the answer may not come from
      any document. Serious.
    - note (under-read): the pages use wording the answer never cites, alongside
      an absence claim. Phrased as the OBSERVATION only — "the pages also use X"
      — never as "the answer reports nothing on X". An answer can discuss a topic
      at length and still correctly say one sub-mechanism is missing; the earlier
      wording called such answers empty, which was simply false.
    - note (clause number): numbering is discarded at ingest, so a clause-number
      question can never match. Say so, rather than let "not addressed" read as
      "the clause does not exist".
    """
    q, ctx, ans = question or "", context or "", answer or ""
    if not ctx or not ans:
        return "", "", {}

    alerts: list[str] = []
    notes: list[str] = []
    asked: list[str] = []
    found: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n", ans)

    for label, alternates in _TOPIC_TERMS.items():
        rxs = [(name, re.compile(pat, re.IGNORECASE)) for pat, name in alternates]
        if not any(rx.search(q) for _, rx in rxs):
            continue
        in_ctx = sum(len(rx.findall(ctx)) for _, rx in rxs)
        in_ans = sum(len(rx.findall(ans)) for _, rx in rxs)
        asked.append(label)
        if in_ctx:
            found.append(label)

        if in_ctx == 0 and in_ans >= 2:
            # Sentence-level so a passing mention doesn't trip it — only an
            # operative statement about a topic the pages never raise.
            asserted = [s for s in sentences
                        if any(rx.search(s) for _, rx in rxs)
                        and _RX_OPERATIVE.search(s)
                        and not _RX_ABSENCE_CLAIM.search(s)]
            if asserted:
                alerts.append(
                    f"**{label}** is not mentioned anywhere in the pages retrieved for "
                    f"this answer, yet the answer states terms for it. Treat every "
                    f"{label} detail below as unverified."
                )
            continue

        # Under-read: the answer declares the document silent on this topic while
        # the pages use a phrasing the answer never picked up. Naming that phrasing
        # is the whole point — "it mentions arbitration" is actionable, "the topic
        # appears somewhere" is not.
        if in_ctx >= 1 and any(_RX_ABSENCE_CLAIM.search(s)
                               and any(rx.search(s) for _, rx in rxs) for s in sentences):
            missed = [name for name, rx in rxs if rx.search(ctx) and not rx.search(ans)]
            if missed:
                # Each note carries its own lead-in. The frontend used to hardcode
                # one heading for every note, which read as a non-sequitur on the
                # clause-number case below.
                notes.append(
                    f"**Also in these pages.** On **{label}**, the retrieved pages also "
                    f"use the wording **{', '.join(missed[:3])}**, which this answer does "
                    f"not cite. If your question turns on that, re-ask naming it against "
                    f"a single named document."
                )

    def _done() -> tuple[str, str, dict]:
        return " ".join(alerts), " ".join(notes), {
            "topics_asked": len(asked),
            "topics_found": len(found),
            "topics_missing": [t for t in asked if t not in found],
        }

    m = _RX_CLAUSE_NUMBER_Q.search(q)
    if m:
        num = next(g for g in m.groups() if g)
        n = re.escape(num)
        cl = clause_lookup or {}

        # Out-of-range is a FACT from clause_map, independent of what the model
        # said — deliberately NOT gated on _RX_ABSENCE_CLAIM. Confirmed live
        # (clause 12 of a document mapped 1-8): scope resolves this document
        # alongside its Test_ synthetic stand-in, retrieval pulled a page from
        # the stand-in, and the model answered from it confidently — "Clause 12
        # — Term, Survival, and Governing Law (NDA-GreenSteel — NDA)" — with no
        # absence language anywhere, so a gated check never runs at all. Since
        # the map already proves the number doesn't belong to the real
        # document, waiting for the model to agree was the bug, not a
        # legitimate gate.
        if cl.get("doc_numbers") and num not in cl["doc_numbers"]:
            nums = cl["doc_numbers"]
            hedged = _RX_ABSENCE_CLAIM.search(ans)
            msg = (
                f"**No clause {num} in this document.** Its source is numbered "
                f"**{nums[0]}–{nums[-1]}**, and clause {num} is not among those "
                f"sections."
            )
            if hedged:
                notes.append(msg)
            else:
                # The model answered as if clause N were real, with no hedge at
                # all — a stronger signal than a topic gap, since it means
                # content was very likely drawn from a different document
                # sharing this one's scope (the same real/Test_ collision the
                # SCOPE WARNING already names, here pinned to a specific number).
                alerts.append(
                    msg + " The answer above did not flag this, so treat its "
                    "content as unverified against this document."
                )
            return _done()

        if not _RX_ABSENCE_CLAIM.search(ans):
            return _done()   # answered without hedging and the number checks out (or is unmapped) — nothing to add

        if cl.get("hits"):
            # The map resolved the number but the answer still claimed absence —
            # tell the reader exactly where the clause lives.
            heads = ", ".join(sorted({h["heading"] for h in cl["hits"]}))
            notes.append(
                f"**Clause {num} does exist in this document** — it is the section "
                f"titled **{heads}**. If the answer above missed it, re-ask naming "
                f"that section."
            )
            return _done()
        # Must look like a CLAUSE reference, not any stray digit. The first
        # version tested `\b5[\.\s)]`, which "within 5 business days" satisfies —
        # so the note never fired on a real corpus. Accept "Clause 5", "5.2",
        # or a "5." / "5)" heading at the start of a line, nothing else.
        looks_numbered = re.search(
            rf"(?:clause|section|article|art\.|cl\.)\s*{n}\b"
            rf"|^\s*{n}[.)]\s"
            rf"|\b{n}\.\d",
            ctx, re.IGNORECASE | re.MULTILINE,
        )
        if not looks_numbered:
            # Phrased as how to search, not as what the system lacks. The
            # information is the same either way, but "numbering is not retained"
            # reads as a defect report to a client, and the note exists to stop a
            # reader concluding the clause does not exist — not to apologise.
            notes.append(
                f"**Try asking by subject.** Documents here are indexed by topic "
                f"rather than by clause number, so **clause {num}** has nothing to "
                f"match against — this is not a finding that the clause is missing. "
                f"Name the subject instead (e.g. \"the termination clause\" or "
                f"\"the indemnity clause\") and it will be found if it is there."
            )

    return _done()


@tracing.traced_node("validate_response")
def validate_response_node(state: QueryState) -> dict:
    logger.info("[AGENT] validate_response_node: intent=%s", state.get("intent"))
    wr = state.get("answer_result") or {}
    answer = wr.get("answer", "") or ""
    intent = state["intent"]
    valid, warning = True, None

    if intent == "comparison" and "|" not in answer:
        valid, warning = False, "Comparison intent but no table detected in answer."
    elif intent == "obligation" and "|" not in answer and not re.search(r'^\s*[-*\d]', answer, re.M):
        valid, warning = False, "Obligation intent but no list/table detected in answer."
    elif intent == "drafting" and not re.search(r'```|^\s*>|aggressive|balanced|conservative', answer, re.I | re.M):
        valid, warning = False, "Drafting intent but no clause formulations detected in answer."

    if warning:
        logger.info("Intent validation warning: %s", warning)

    # LLM grounding check — verifies factual claims against context
    grounding = {"grounding_score": None, "ungrounded_claims": [], "summary": "Disabled."}
    if config.ENABLE_ANSWER_VALIDATION:
        _emit({"stage": "validating", "status": "active", "message": "Checking answer grounding…"})
        grounding = _check_grounding_hybrid(
            state["question"],
            state.get("wiki_context", ""),
            answer,
            intent=intent,
        )
        logger.info("[AGENT] Grounding score: %s | %s", grounding.get("grounding_score"), grounding.get("summary"))
        _emit({"stage": "validating", "status": "done",
               "grounding_score": grounding.get("grounding_score"),
               "message": f"Grounding: {grounding.get('grounding_score', '?')}%"})

    wr["validation"] = {"valid": valid, "warning": warning, "grounding": grounding}

    # Deliberately outside the ENABLE_ANSWER_VALIDATION block: this costs no LLM
    # call, and it is the only check that caught the confirmed fabrication.
    if config.ENABLE_TERM_CHECK:
        ctx_alert, ctx_note, term_facts = _check_term_presence(
            state["question"], state.get("wiki_context", ""), answer,
            clause_lookup=state.get("clause_lookup"))
        if ctx_alert or ctx_note:
            logger.info("[AGENT] term-presence alert=%r note=%r",
                        ctx_alert[:110], ctx_note[:110])
        wr["context_warning"] = ctx_alert   # red — may not come from any document
        wr["context_note"] = ctx_note       # amber — worth a second look

        # Counts the reader can act on, shown beside the confidence percentage
        # rather than replacing it. Every one is derived by string match or by
        # counting what retrieval returned, so none of them is a prediction that
        # could be wrong the way confidence_score routinely is.
        cite = wr.get("citation_check") or {}
        wr["answer_facts"] = {
            "pages": len(wr.get("selected_titles") or wr.get("pages_used") or []),
            "documents": len(wr.get("files_used") or []),
            "topics_asked": term_facts.get("topics_asked", 0),
            "topics_found": term_facts.get("topics_found", 0),
            "topics_missing": term_facts.get("topics_missing", []),
            "quotes_total": cite.get("total", 0),
            "quotes_unverified": cite.get("unverified", 0) + cite.get("misattributed", 0),
        }

    _trace = tracing.get_trace()
    if _trace:
        _trace.log_validation({"valid": valid, "warning": warning, "grounding": grounding})

    _emit({"stage": "complete", "status": "done", "type": "answer",
           "payload": wr, "message": "Done"})
    return {"answer_result": wr, "validation": {"valid": valid, "warning": warning, "grounding": grounding}}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def _route_after_disambiguation(state: QueryState) -> str:
    return "stop" if state.get("needs_disambiguation") else "continue"


def _route_after_clarification(state: QueryState) -> str:
    return "stop" if state.get("needs_clarification") else "continue"


def build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("disambiguation", check_disambiguation_node)
    g.add_node("clarification", check_clarification_node)
    g.add_node("resolve_scope", resolve_scope_node)
    g.add_node("retrieve", retrieve_context_node)
    g.add_node("generate", generate_answer_node)
    g.add_node("validate", validate_response_node)

    g.add_edge(START, "classify_intent")
    g.add_edge("classify_intent", "disambiguation")
    g.add_conditional_edges("disambiguation", _route_after_disambiguation,
                            {"stop": END, "continue": "clarification"})
    g.add_conditional_edges("clarification", _route_after_clarification,
                            {"stop": END, "continue": "resolve_scope"})
    g.add_edge("resolve_scope", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_edge("validate", END)
    return g.compile()


_QUERY_GRAPH = None


def get_query_graph():
    """Lazily compile and cache the query graph (compile once, reuse)."""
    global _QUERY_GRAPH
    if _QUERY_GRAPH is None:
        _QUERY_GRAPH = build_query_graph()
    return _QUERY_GRAPH


# ---------------------------------------------------------------------------
# Meta / capability questions
# ---------------------------------------------------------------------------
# A question ABOUT the assistant is not a question about a document. Every one
# of the five intents assumes "some page in the corpus answers this", so a meta
# question fell through to `factual`, ran full retrieval against a legal corpus
# that contains nothing about itself, and produced the worst possible first
# impression: "The retrieved context does not contain information regarding the
# types of questions that could be asked", 0% confidence, three unrelated
# documents listed as References, ~8.2k tokens burned. Confirmed live — it was
# literally the first thing a pilot tester typed.
#
# Anchored at ^ and kept deliberately narrow: these must never swallow a real
# document question. "What can you tell me about the indemnity clause" does NOT
# match (the `can you` branch requires do/help/answer/handle, not `tell`), and
# nothing here matches mid-sentence.
_RX_META_QUERY = re.compile(
    r'^\s*(?:'
    r'help[\s!.?]*$|\?+\s*$'
    # Up to 3 modifier words allowed between "of" and "questions" — "what kind
    # of NDA related questions can I ask" is still a meta question (asking
    # about capability, scoped to a topic), not a document question; the
    # _RX_META_DOC_HINT veto below still disqualifies it if a legal term shows
    # up AFTER this whole opener instead of as a modifier inside it.
    r'|what\s+(?:kind|kinds|type|types|sort|sorts)\s+of\s+(?:\w+[\s-]+){0,3}?(?:questions?|things?|queries)\b'
    r'|what\s+(?:questions?|things?)\s+(?:can|could|should)\s+i\s+ask\b'
    r'|what\s+can\s+(?:you|this|it|lexwiki)\s+(?:do|help|answer|handle)\b'
    r'|what\s+can\s+i\s+ask\b'
    r'|what\s+are\s+(?:you|your\s+capabilities)\b'
    r'|who\s+are\s+you\b'
    r'|how\s+(?:do|can)\s+i\s+use\s+(?:you|this|it|lexwiki)\b'
    r'|how\s+does\s+(?:this|it|lexwiki)\s+work\b'
    r'|(?:give|show)\s+me\s+(?:some\s+)?(?:example|sample)\s+questions?\b'
    r'|what\s+(?:documents?|files?|data)\s+(?:do\s+you\s+have|are\s+(?:there|loaded|available))\b'
    r')',
    re.IGNORECASE,
)

# Disqualifier: the opener above may be followed by harmless filler ("...could I
# ask TO YOU?"), but if the rest of the question names a document, clause or
# legal concept it is a real retrieval question wearing a meta-sounding opener
# ("what documents do you have to deliver under clause 5") and must go to the
# graph. Anything matching this vetoes the fast path.
_RX_META_DOC_HINT = re.compile(
    r'\b(?:clause|section|article|agreement|contract|nda|sha|jva?|deed|schedule|annex|'
    r'party|parties|obligation|termination|indemnit|liabilit|confidential|warrant|'
    r'breach|dispute|governing\s+law|judgment|opinion|deliver|pay|notice|about)\b',
    re.IGNORECASE,
)


def _is_meta_query(question: str) -> bool:
    """True for questions about the assistant itself rather than the documents."""
    q = question or ""
    m = _RX_META_QUERY.match(q)
    if not m:
        return False
    return not _RX_META_DOC_HINT.search(q[m.end():])


# LLM fallback for meta questions the regex misses ("what should I be asking
# you about?" — no fixed opener the regex above anticipated). Gated hard
# before it ever calls the model:
#   1. Must clear _RX_META_DOC_HINT with ZERO legal vocabulary anywhere in the
#      message (not just after an opener) — the same veto the regex path
#      uses, just applied to the whole message instead of a tail. A real
#      document question always contains at least one of these words, so it
#      can never reach the LLM call; this is what makes the fallback provably
#      unable to turn a real question into a canned reply.
#   2. Must look like a question (starts with a wh-word/aux verb, or ends in
#      "?") — filters out ordinary short statements that reach this point.
#   3. Must be short — genuine meta questions are short; a long message this
#      deep into the checks is never one.
# Failing any gate skips the LLM call entirely and falls through to the
# normal graph, identical to today's regex-only behaviour — so the worst
# case this fallback can produce is unchanged from the status quo.
_META_LLM_MAX_LEN = 120

_RX_QUESTION_LIKE = re.compile(
    r'^\s*(?:what|how|who|why|can|could|do|does|is|are)\b|\?\s*$', re.IGNORECASE,
)

# Regulatory/compliance framing is never a question about the assistant, but it
# clears every gate above: no meta opener, none of _RX_META_DOC_HINT's legal
# vocabulary ("GDPR" and "compliant" are in neither list), question-like, short.
# So it reached the LLM tiebreak — which is a coin flip on it. Measured on "are
# we GDPR compliant": 3/8 runs answered "meta" and returned the capabilities
# blurb, a total non-sequitur to someone asking about compliance.
#
# Applied ONLY in _is_meta_query_extended, gating the LLM fallback. Deliberately
# not added to _RX_META_DOC_HINT, which is also applied to the tail of a matched
# meta opener — a word like "law" or "policy" there would start vetoing genuine
# capability questions ("what can you do with legal documents").
_RX_META_COMPLIANCE = re.compile(
    r'\b(?:compl(?:y|ies|iant|iance)|non[\s-]?compliance'
    r'|gdpr|hipaa|ccpa|dpdp|sox|pci|ferpa'
    r'|regulat(?:ion|ions|ory|ed)|statutor(?:y|ily)'
    r'|audit(?:s|ed|ing)?|penalt(?:y|ies)|liable|exposure'
    r')\b',
    re.IGNORECASE,
)

_META_LLM_PROMPT = (
    "Decide whether this message is asking about the ASSISTANT itself — its "
    "capabilities, how to use it, or what kinds of questions it can answer — "
    "rather than asking about the content of any legal document.\n\n"
    'Message: "{question}"\n\n'
    'Respond with ONLY one word: "meta" or "document".'
)


def _is_meta_query_llm(question: str) -> bool:
    """Cheap last-resort LLM check for a meta question the regex missed.

    Only ever called after the message has already cleared the zero-legal-
    vocabulary gate in _is_meta_query_extended, so a real document question
    can't reach this call in the first place — it can only fail to catch a
    genuine meta question (same as today's regex-only behaviour), never
    mistake a real one for meta.
    """
    try:
        raw, _ = llm.ask(
            _META_LLM_PROMPT.format(question=question),
            fast=True, max_tokens=config.MAX_TOKENS_META_CLASSIFY,
        )
        return raw.strip().lower().startswith("meta")
    except Exception as e:
        logger.warning("Meta LLM fallback failed: %s", e)
        return False


def _is_meta_query_extended(question: str) -> bool:
    """_is_meta_query, plus a gated LLM fallback for phrasing the regex misses."""
    if _is_meta_query(question):
        return True
    if not config.ENABLE_META_LLM_FALLBACK:
        return False
    q = (question or "").strip()
    if not q or len(q) > _META_LLM_MAX_LEN:
        return False
    if _RX_META_DOC_HINT.search(q):
        return False
    if _RX_META_COMPLIANCE.search(q):
        return False
    if not _RX_QUESTION_LIKE.search(q):
        return False
    return _is_meta_query_llm(q)


def _corpus_summary(session_id: str, max_families: int = 0) -> str:
    """One line describing what's actually loaded, or '' if it can't be read.

    Reads the real corpus rather than hardcoding a document list, so the answer
    stays true for any session — including a client's own upload of categories
    this codebase's type regexes have never seen.

    max_families caps how many types are named (0 = all). A real corpus can have
    a dozen, which is orientation in a help answer but a wall of text in a
    one-line greeting.
    """
    try:
        from services import db as _db, wikis as _wikis
        _wid = _wikis.active_wiki_id()
        families = sorted(f for f in (_db.list_doc_families(_wid, session_id) or []) if f)
        n_docs = len(_db.get_source_docs(_wid, session_id) or [])
    except Exception as e:
        logger.warning("Meta answer: corpus summary unavailable: %s", e)
        return ""

    if not n_docs:
        return ""
    parts = [f"**{n_docs} documents** are currently loaded"]
    if families:
        shown, rest = families, 0
        if max_families and len(families) > max_families:
            shown, rest = families[:max_families], len(families) - max_families
        listed = ", ".join(shown)
        parts.append(f"covering {listed} and {rest} other types" if rest
                     else f"covering {listed}")
    return " ".join(parts) + "."


def _canned_payload(answer: str, label: str, method: str) -> dict:
    """Shape a zero-token canned reply like a normal `answer_result`.

    Same key set app.py and the frontend already read, so nothing downstream
    needs to special-case a canned turn beyond the `meta_answer` render flag.
    """
    return {
        "answer": answer,
        "meta_answer": True,       # frontend: render without confidence/grounding/refs
        "intent": "factual",
        "intent_label": label,
        "intent_confidence": 1.0,
        "intent_method": method,
        "confidence_score": 100,
        "pages_used": [],
        "files_used": [],
        "selected_titles": [],
        "token_total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "validation": {"valid": True, "warning": None, "grounding": {}},
        # Deliberately blank: a canned turn resolved no document scope, so it must
        # not become the scope the NEXT turn inherits via wiki._carryover_scope.
        "scope_method": "",
        "scope_docs": [],
        "_debug_context": "",
    }


def _meta_answer(session_id: str) -> dict:
    """Canned capability answer — no LLM call, no retrieval, no tokens."""
    corpus = _corpus_summary(session_id)
    corpus_line = f"{corpus}\n\n" if corpus else ""

    answer = (
        "I'm here to help you work through the documents loaded into this "
        f"workspace. {corpus_line}"
        "Ask me to pull a specific clause, definition, party, date, or figure out of "
        "a document; list out obligations and deadlines; put a few agreements "
        "side by side and show where they differ; flag unusual or one-sided terms; "
        "or draft clause language, letters, and trackers over in the Draft tab.\n\n"
        "A few examples of what that looks like:\n\n"
        "- *What are the termination rights in Service Agreement 2?*\n"
        "- *List every notice obligation and its deadline under NDA 3.*\n"
        "- *Compare the confidentiality clauses in NDA 1 and NDA 4.*\n"
        "- *Are there any red flags in the SunBridge joint venture agreement?*\n\n"
        "One thing that helps: if you already know which document you mean, just say "
        "so — it gets you a faster, more precise answer. You can browse everything "
        "that's loaded under the **Files** tab, and every answer comes with a "
        "citation back to the clause it's based on, so you can double-check it "
        "yourself.\n\n"
        "One honest caveat: I can tell you what the documents say, but I'm not a "
        "lawyer, and this isn't legal advice."
    )

    return _canned_payload(answer, "Help", "meta-regex")


# ---------------------------------------------------------------------------
# Greetings, thanks, sign-offs
# ---------------------------------------------------------------------------
# Same failure shape as the meta questions above and cheaper to trigger: "hi"
# has no answer anywhere in a legal corpus, so it retrieved unrelated contracts
# and reported 0% confidence at full token cost.
#
# Every pattern here is a FULL match on the whole message (each ends in `$`),
# unlike the prefix-anchored meta patterns — so "hello, what does clause 5 say?"
# does not match and goes to the graph as a normal question. `[\s\W]*` at both
# ends absorbs punctuation and emoji ("hi!!", "thanks :)") but never letters.
_RX_GREETING = re.compile(
    r'^[\s\W]*(?:hi+|hey+|hell?o+|hiya|heya|greetings|namaste|'
    r'good\s+(?:morning|afternoon|evening|day))'
    r'(?:\s+(?:there|team|all|folks|everyone))?[\s\W]*$',
    re.IGNORECASE,
)

# Checked before _RX_ACK so "ok thanks" reads as thanks, not a bare ack.
_RX_THANKS = re.compile(
    r'^[\s\W]*'
    r'(?:(?:ok(?:ay)?|alright|cool|nice|great|perfect|awesome|excellent)\W+)?'
    r'(?:thanks|thank\s+you|thankyou|thx|ty|cheers|much\s+appreciated|appreciate\s+it)'
    r'(?:\s+(?:a\s+lot|so\s+much|very\s+much|again))?[\s\W]*$',
    re.IGNORECASE,
)

_RX_FAREWELL = re.compile(
    r'^[\s\W]*'
    r'(?:bye|goodbye|good\s?bye|see\s+(?:you|ya)(?:\s+later)?|'
    r'that(?:\'|’)?s\s+all(?:\s+for\s+now)?|that\s+is\s+all|'
    r'no\s+more\s+questions|nothing\s+else)'
    r'[\s\W]*$',
    re.IGNORECASE,
)

# Deliberately excludes "yes"/"no"/"sure" — those are plausible replies to a
# clarification prompt, where swallowing them would break the flow.
_RX_ACK = re.compile(
    r'^[\s\W]*'
    r'(?:ok(?:ay)?|cool|nice|great|perfect|awesome|excellent|alright|'
    r'got\s+it|understood|noted|makes\s+sense|sounds\s+good|fair\s+enough)'
    r'[\s\W]*$',
    re.IGNORECASE,
)


_RX_CITES = re.compile(
    r"\b(?:which|what)\s+(?:documents?|agreements?|contracts?|judgments?)\b[^?]*"
    r"\b(?:cite|cites|citing|rely\s+on|relies\s+on|reference|references|invoke)\b",
    re.IGNORECASE,
)
_RX_CITED_BY = re.compile(
    r"\b(?:what|which)\s+(?:authorities|statutes?|acts?|rules?|laws?|cases?)\b[^?]*"
    r"\b(?:cite[sd]?|cited|relied|rely|invoked?|reference[sd]?)\b",
    re.IGNORECASE,
)
_RX_AMENDS = re.compile(
    r"\b(?:what|which)\b[^?]*\b(?:amends?|amended|amendment\s+to|supersedes?|"
    r"replaces?|varies)\b",
    re.IGNORECASE,
)
# The authority a "which documents cite X" question is asking about.
_RX_AUTHORITY = re.compile(
    # Lowercase connectors ("and", "of", "the") sit inside real statute names —
    # "Arbitration and Conciliation Act 1996" — so a run of capitalised words
    # alone clips the name at the connector and loses its distinctive head.
    r"\b((?:[A-Z][\w'&.-]*\s+(?:(?:and|of|the|for|on)\s+)?){0,6}"
    r"(?:Act|Rules?|Code|Regulations?|Convention|Guidelines?|Standard)"
    # Names like "Code of Civil Procedure 1908" put the terminator FIRST, so
    # the trailing phrase is part of the name, not the sentence around it.
    r"(?:\s+(?:of|on|for)\s+(?:[A-Z][\w'&.-]*\s*){1,4})?"
    r"(?:[,\s]+\d{4})?)",
)


# Counting over document metadata (§ Phase 3.5b). Deliberately narrow: the
# noun being counted must be a DOCUMENT word. "How many days' notice is
# required" and "how many parties signed" are ordinary factual questions that
# the retrieval pipeline answers well today, and a greedy counting detector
# stealing them would be a regression, not a feature — which is the one real
# risk this branch carries.
# Words that, appearing between "how many" and a document noun, mean the
# question is counting something OTHER than documents — "how many PARTIES
# signed the agreement", "how many DAYS notice does the contract require".
# Without this the filler happily spans a different head noun and the branch
# steals an ordinary factual question, which is the one regression this
# feature can cause. Found by testing, not by inspection.
_COUNT_BLOCKERS = (
    r"parties|part(?:y|ies)|days?|years?|months?|weeks?|hours?|people|persons?|"
    r"signator(?:y|ies)|pages?|copies|times?|clauses?|sections?|schedules?|"
    r"annexures?|milestones?|employees?|shares?|installments?|instalments?|"
    r"signed|is|are|was|were|do|does|did|have|has|had|must|shall|require[sd]?|"
    r"needs?|takes?|remain|survive[sd]?"
)
_RX_COUNT = re.compile(
    r"\b(?:how\s+many|number\s+of|count\s+(?:of|the))\s+"
    rf"(?:(?!(?:{_COUNT_BLOCKERS})\b)[a-z]+\s+){{0,3}}?"
    r"(?:contracts?|agreements?|documents?|ndas?|msas?|deeds?|leases?|"
    r"licen[cs]es?|amendments?|policies)\b",
    re.IGNORECASE,
)
# A counting question must also name who or what to count, otherwise "how many
# contracts do we have" is a whole-corpus count — which is valid, and handled,
# but the party form is the one that needs extraction.
_RX_COUNT_PARTY = re.compile(
    r"\b(?:with|for|involving|between|from|against)\s+"
    r"((?:[A-Z][\w'&.\-]*)(?:\s+(?:[A-Z][\w'&.\-]*|and|&|of|the))*)",
)
_RX_COUNT_DOCTYPE = re.compile(
    r"\b(?:how\s+many|number\s+of|count\s+(?:of|the))\s+"
    rf"((?:(?!(?:{_COUNT_BLOCKERS})\b)[a-z]+\s+){{0,3}}?"
    r"(?:contracts?|agreements?|documents?|ndas?|msas?|deeds?|leases?|"
    r"licen[cs]es?|amendments?|policies))\b",
    re.IGNORECASE,
)


# --- Phase 4 analytics detectors ------------------------------------------
# All three are narrow by construction. These branches answer from SQL with no
# retrieval at all, so a false positive does not merely answer oddly — it
# answers a document question with a corpus statistic. Each therefore requires
# BOTH an operation word and a metric word, and the metric list is closed.
_RX_AGG_METRIC = re.compile(
    r"\b(?:liability\s+caps?|caps?\b|contract\s+values?|contract\s+prices?|"
    r"total\s+values?|deal\s+values?)", re.IGNORECASE)
_RX_AGG_OP = re.compile(
    r"\b(?:total|sum|average|avg|mean|median|typical|highest|lowest|largest|"
    r"smallest|aggregate|combined|across\s+all|how\s+much\s+in\s+total)\b",
    re.IGNORECASE)
# "what is the liability cap in X" is a document lookup, not an aggregate. An
# operation word plus a singular document reference is still a lookup, so a
# preposition naming one instrument vetoes the branch.
_RX_AGG_VETO = re.compile(
    r"\b(?:in|of|under|for)\s+(?:the|this|that|our)\s+"
    r"(?:agreement|contract|msa|nda|document|deed|lease)\b"
    r"|\bdated\b|\bclause\s+\d", re.IGNORECASE)
# A question also asking WHAT KIND of liability a cap excludes ("carve-outs",
# "what is excluded/excepted from the cap") wants clause text, not a number —
# the aggregate branch answers with bare statistics and has no clause text to
# return. Confirmed live: "the aggregate liability cap for X and Y, and what
# types of liability are excluded from that cap" tripped only the numeric
# half and silently dropped the carve-out half of its own question.
_RX_AGG_CARVEOUT_VETO = re.compile(
    r"\bcarve[-\s]?outs?\b|\bexclu(?:ded|ding|sions?)\s+from\b|"
    r"\bexcept(?:ed|ions?)\s+from\b", re.IGNORECASE)

_RX_GAP = re.compile(
    r"\b(?:which|what|list|show|find|how\s+many)\b[^?]{0,80}?"
    r"\b(?:do\s+not|don't|doesn't|does\s+not|lack|lacks|lacking|missing|"
    r"without|no|absent|fail\s+to)\b",
    re.IGNORECASE)
_RX_GAP_FIELD = re.compile(
    r"\b(?:liability\s+caps?|caps?\b|governing\s+law|termination(?:\s+(?:clause|provision))?)\b",
    re.IGNORECASE)

_RX_TREND = re.compile(
    r"\b(?:over\s+time|over\s+the\s+(?:years|last|past)|trend|trending|"
    r"year[- ]on[- ]year|by\s+year|changed?\s+since|historically|"
    r"getting\s+(?:longer|shorter|higher|lower|bigger|smaller))\b",
    re.IGNORECASE)


def _is_analytics_query(question: str) -> str:
    """'aggregate' | 'gap' | 'trend' | '' — questions the normalised columns answer.

    Checked before the document-oriented structural branches because these are
    corpus-shaped questions: "what's the average liability cap across our
    contracts" is not asking about any one document, and retrieval answering it
    from whichever handful of pages it fetched would produce a confident
    average over an arbitrary sample.
    """
    q = question or ""
    if _RX_TREND.search(q) and _RX_AGG_METRIC.search(q):
        return "trend"
    if _RX_GAP.search(q) and _RX_GAP_FIELD.search(q):
        return "gap"
    if (_RX_AGG_OP.search(q) and _RX_AGG_METRIC.search(q)
            and not _RX_AGG_VETO.search(q)
            and not _RX_AGG_CARVEOUT_VETO.search(q)
            # A question naming a specific party (a corporate suffix the
            # question spells out) is asking about THAT party's document(s),
            # not a corpus-wide statistic — the closed operation-word list
            # above includes "aggregate"/"total", words a lawyer also uses
            # for "the aggregate cap FOR [named party]" meaning that party's
            # own single cap, not a sum across the corpus. Confirmed live:
            # "the aggregate liability cap for Apex Suvarna Telecommunications
            # Private Limited and Nidra Bhandari" returned a corpus statistic
            # over one arbitrarily-retrieved document and dropped the
            # question's own carve-out half entirely.
            and not wiki._PARTY_NAME_RE.search(q)):
        return "aggregate"
    return ""


# Version-chain questions — answered by a bounded 2-hop traversal rather than
# the single-hop amends lookup, since a chain is commonly original → amendment
# → further amendment and one hop shows only the middle of it.
_RX_CHAIN = re.compile(
    r"\b(?:amendment\s+(?:chain|history|trail)|version\s+(?:chain|history)|"
    r"full\s+(?:chain|history)\s+of|chain\s+of\s+amendments?|"
    r"all\s+(?:the\s+)?amendments?\s+(?:to|of)|"
    r"(?:complete|entire)\s+(?:amendment|version))\b",
    re.IGNORECASE,
)


def _is_structural_query(question: str) -> str:
    """'cites' | 'cited_by' | 'amends' | 'chain' | 'count' | '' — typed-table questions.

    Citation and amendment links are single lines inside documents that are
    otherwise about unrelated subjects, so embedding the question ranks the
    wrong pages: "which documents cite the Arbitration Act" retrieves documents
    ABOUT arbitration, not the ones carrying the citation. The citations and
    document_relations tables already hold these as structured rows, and a SQL
    join answers exactly and with no LLM call.
    """
    q = question or ""
    if _RX_CITES.search(q):
        return "cites"
    if _RX_CITED_BY.search(q):
        return "cited_by"
    # Chain before the single-hop amends branch: "the full amendment history"
    # wants the whole version chain, and answering it one hop deep would show
    # the first amendment while omitting the one that superseded it.
    if _RX_CHAIN.search(q):
        return "chain"
    if _RX_AMENDS.search(q):
        return "amends"
    # Checked last: an amendment question phrased as "how many documents amend
    # X" is better served by the amends branch, which names them rather than
    # only counting them.
    if _RX_COUNT.search(q):
        return "count"
    return ""


def _fmt_money(v, currency: str | None = None) -> str:
    if v is None:
        return "—"
    cur = "" if not currency or currency == "unspecified" else f"{currency} "
    return f"{cur}{v:,.0f}"


def _analytics_answer(kind: str, question: str, session_id: str,
                      wiki_id: str) -> dict | None:
    """Answer an aggregate / gap / trend question from the normalised columns.

    Every answer carries the coverage note the analytics layer produced. That
    is not decoration: an average over the caps this corpus could parse is a
    statement about those contracts, not about the corpus, and a reader shown
    a bare figure will reasonably assume the latter.
    """
    from services import analytics, wiki as _wiki
    parties = []
    m = _RX_COUNT_PARTY.search(question or "")
    if m:
        raw = m.group(1).strip().rstrip(".,;:?")
        parties = [p.strip() for p in re.split(r"\s+(?:and|&)\s+", raw) if len(p.strip()) > 2]

    try:
        if kind == "aggregate":
            metric = ("contract_value"
                      if re.search(r"\b(?:contract|deal|total)\s+(?:value|price)", question or "", re.I)
                      else "liability_cap")
            data = (analytics.aggregate_contract_values(wiki_id, session_id, parties)
                    if metric == "contract_value"
                    else analytics.aggregate_liability_caps(wiki_id, session_id, parties))
            if data.get("error") or not data.get("by_currency"):
                return None
            label = "contract value" if metric == "contract_value" else "liability cap"
            scope_txt = f" for {' and '.join(parties)}" if parties else ""
            lines = [f"**{label.title()} across the corpus{scope_txt}**", ""]
            for c in data["by_currency"]:
                n = c.get("contracts") or c.get("documents")
                lines.append(f"- **{c['currency']}** — {n} document(s)")
                lines.append(f"  - Total: {_fmt_money(c['sum'], c['currency'])}")
                if c.get("median") is not None:
                    lines.append(f"  - Median: {_fmt_money(c['median'], c['currency'])}")
                lines.append(f"  - Mean: {_fmt_money(c['mean'], c['currency'])}")
                lines.append(f"  - Range: {_fmt_money(c['min'], c['currency'])} to "
                             f"{_fmt_money(c['max'], c['currency'])}")
            if data.get("mixed_currency"):
                lines += ["", "Reported per currency and deliberately not converted or "
                              "combined — a single total across currencies would be wrong "
                              "in each of them."]
            lines += ["", f"*{data['coverage']}*"]
            payload = _canned_payload("\n".join(lines), "Aggregate", "structured-analytics")

        elif kind == "gap":
            field = ("governing_law" if re.search(r"governing\s+law", question or "", re.I)
                     else "termination" if re.search(r"terminat", question or "", re.I)
                     else "liability_cap")
            data = analytics.find_gaps(wiki_id, session_id, field, parties)
            if data.get("error"):
                return None
            lines = [f"**{data['missing']} document(s) state no {data['label']}.**", ""]
            for d in data["documents"][:20]:
                date = f" — {d['effective_date']}" if d.get("effective_date") else ""
                lines.append(f"- {_wiki._norm_doc_name(d['source_doc'])}{date}")
            if data["truncated"]:
                lines.append(f"- …and {data['missing'] - len(data['documents'])} more")
            lines += ["", f"*{data['note']}*"]
            if data["indeterminate"]:
                lines.append("")
                lines.append("The indeterminate documents are excluded from the list above "
                             "on purpose: reporting a contract as uncapped when its cap is "
                             "recorded in a schedule would be a worse error than omitting it.")
            payload = _canned_payload("\n".join(lines), "Gap analysis", "structured-analytics")

        elif kind == "trend":
            metric = ("contract_value"
                      if re.search(r"\b(?:contract|deal|total)\s+(?:value|price)", question or "", re.I)
                      else "liability_cap")
            data = analytics.trend_over_time(wiki_id, session_id, metric, parties)
            if data.get("error") or not data.get("buckets"):
                return None
            label = "contract value" if metric == "contract_value" else "liability cap"
            head = (f"**{label.title()} by year — {data['direction']}**"
                    if data.get("direction") else f"**{label.title()} by year**")
            lines = [head, "", "| Year | Documents | With a readable value | Median |",
                     "| --- | --- | --- | --- |"]
            for b in data["buckets"]:
                med = _fmt_money(b["median"]) if b["median"] is not None else "—"
                lines.append(f"| {b['year']} | {b['documents']} | {b['with_value']} | {med} |")
            lines += ["", f"*{data['note']}*"]
            payload = _canned_payload("\n".join(lines), "Trend", "structured-analytics")
        else:
            return None
    except Exception as e:
        logger.error("[AGENT] analytics fast-path (%s) failed: %s", kind, e)
        return None

    payload["meta_answer"] = False
    payload["files_used"] = []
    return payload


def _count_answer(question: str, session_id: str, wiki_id: str) -> dict | None:
    """Answer "how many contracts do we have with X" by counting, not retrieving.

    Returns None on an empty count rather than reporting zero. A zero here has
    two very different causes — the party genuinely has no documents, or the
    name was not extracted the way the question spells it — and they are not
    distinguishable from this side. Falling through to retrieval lets the
    normal pipeline try, which is the safer of the two failure modes: a wrong
    "you have none" is far worse than a slow answer.
    """
    from services import db as _db, wiki
    parties: list[str] = []
    m = _RX_COUNT_PARTY.search(question or "")
    if m:
        raw = m.group(1).strip().rstrip(".,;:?")
        # "X and Y" is two parties; "Tata Sons and Company Limited" is one.
        # Split only when both sides survive as plausible names, and let the
        # AND-semantics of the query do the rest — a bad split narrows the
        # count rather than inflating it, which fails safe.
        parts = re.split(r"\s+(?:and|&)\s+", raw)
        parties = [p.strip() for p in parts if len(p.strip()) > 2]

    doc_type = None
    dm = _RX_COUNT_DOCTYPE.search(question or "")
    if dm:
        phrase = dm.group(1).strip().lower()
        # Only a qualified type ("supply agreements") narrows the query; the
        # bare noun ("contracts", "documents") is the generic ask and must not
        # be used as a doc_type filter or it matches almost nothing.
        generic = {"contract", "contracts", "agreement", "agreements",
                   "document", "documents"}
        if phrase not in generic:
            words = [w for w in phrase.split() if w not in ("the", "our", "all", "total")]
            if words and words[0] not in generic:
                doc_type = " ".join(words)

    if not parties and not doc_type:
        return None

    try:
        result = _db.count_documents_by_party(wiki_id, session_id, parties, doc_type)
    except Exception as e:
        logger.error("[AGENT] count fast-path failed: %s", e)
        return None
    if not result["total"]:
        return None

    subject = " and ".join(parties) if parties else (doc_type or "the corpus")
    noun = doc_type or "document"
    lines = [f"**{result['total']} {noun}(s) matching {subject}.**", ""]
    if len(result["by_type"]) > 1:
        lines.append("By document type:")
        for t in result["by_type"]:
            lines.append(f"- {t['doc_type']}: {t['count']}")
        lines.append("")
    shown = result["documents"]
    if shown:
        lines.append(f"{'Most recent, by effective date' if len(shown) > 1 else 'Document'}:")
        for d in shown:
            date = f" — {d['effective_date']}" if d.get("effective_date") else ""
            lines.append(f"- {wiki._norm_doc_name(d['source_doc'])}{date}")
        if result["truncated"]:
            lines.append(f"- …and {result['total'] - len(shown)} more")
    lines.append("")
    lines.append("Counted directly from the document index rather than from a text "
                 "search, so this is the complete total, not the closest matches.")

    payload = _canned_payload("\n".join(lines), "Count", "document-index")
    payload["files_used"] = [d["source_doc"] for d in shown]
    payload["meta_answer"] = False
    return payload


def _structural_answer(kind: str, question: str, session_id: str) -> dict | None:
    """Answer a citation / amendment question from the typed tables.

    Returns None whenever the lookup is empty or the subject can't be pinned,
    so the normal pipeline still gets its chance — an empty structured answer
    is not evidence that the answer does not exist.
    """
    from services import db as _db, wikis as _wikis
    try:
        wiki_id = _wikis.active_wiki_id()

        if kind == "cites":
            m = _RX_AUTHORITY.search(question)
            authority = (m.group(1).strip() if m else "").strip(" ,")
            if not authority:
                return None
            hits = _db.find_documents_citing(wiki_id, session_id, authority)
            if not hits:
                return None
            lines = [f"**{len(hits)} document(s) cite {authority}:**", ""]
            for h in hits[:25]:
                pages = (f" (p. {', '.join(map(str, h['pages']))})"
                         if h.get("pages") else "")
                lines.append(f"- {wiki._norm_doc_name(h['source_doc'])}{pages}")
            if len(hits) > 25:
                lines.append(f"- …and {len(hits) - 25} more")
            lines.append("")
            lines.append("Read directly from the citation index, so this is every "
                         "recorded occurrence rather than the closest matches.")
            payload = _canned_payload("\n".join(lines), "Citations", "citation-index")
            payload["files_used"] = [h["source_doc"] for h in hits[:25]]
            payload["meta_answer"] = False
            return payload

        if kind == "count":
            return _count_answer(question, session_id, wiki_id)

        # cited_by / amends / chain all need the document the question is about.
        anchor = _resolve_anchor_doc(question, session_id, wiki_id)
        if not anchor:
            return None

        if kind == "chain":
            from services import relations as _rel
            chain = _rel.find_amendment_chain(wiki_id, session_id, anchor)
            if chain.get("error") or not chain.get("documents"):
                return None
            lines = [f"**Version chain for {wiki._norm_doc_name(anchor)}**", ""]
            for d in chain["documents"]:
                hop = "directly" if d["hops"] == 1 else f"{d['hops']} steps away"
                lines.append(f"- {d['name']} — linked {hop}")
            if chain["edges"]:
                lines += ["", "Links:"]
                for e in chain["edges"][:12]:
                    lines.append(f"- {wiki._norm_doc_name(e['from'])} "
                                 f"*{e['label']}* {wiki._norm_doc_name(e['to'])}")
            lines += ["", f"*{chain['note']} Read from the recorded document "
                          f"relationships, not from a text search.*"]
            payload = _canned_payload("\n".join(lines), "Version chain", "relation-graph")
            payload["files_used"] = [anchor] + [d["source_doc"] for d in chain["documents"]]
            payload["meta_answer"] = False
            return payload

        if kind == "cited_by":
            auths = _db.get_authorities_cited(wiki_id, session_id, anchor)
            if not auths:
                return None
            lines = [f"**{wiki._norm_doc_name(anchor)}** cites "
                     f"{len(auths)} authorit{'y' if len(auths) == 1 else 'ies'}:", ""]
            for a in auths[:30]:
                page = f" (p. {a['page_num']})" if a.get("page_num") else ""
                kindlbl = f" — *{a['authority_type']}*" if a.get("authority_type") else ""
                lines.append(f"- {a['authority']}{kindlbl}{page}")
            payload = _canned_payload("\n".join(lines), "Citations", "citation-index")
            payload["files_used"] = [anchor]
            payload["meta_answer"] = False
            return payload

        rel = _db.get_document_relations(wiki_id, session_id, anchor)
        if not any((rel["outgoing"], rel["incoming"], rel["unresolved"])):
            return None
        name = wiki._norm_doc_name(anchor)
        lines = [f"**Document links recorded for {name}:**", ""]
        for r in rel["outgoing"]:
            lines.append(f"- {name} **{r['label']}** {wiki._norm_doc_name(r['doc'])}")
        for r in rel["incoming"]:
            lines.append(f"- {wiki._norm_doc_name(r['doc'])} **{r['label']}** {name}")
        for r in rel["unresolved"]:
            lines.append(f"- {name} **{r['label'].replace('-unresolved', '')}** "
                         f"“{(r['raw'] or 'an unnamed document').strip()}” "
                         f"— not held in this corpus")
        lines.append("")
        lines.append("Read from the document-relation index built at ingest.")
        payload = _canned_payload("\n".join(lines), "Relations", "relation-index")
        payload["files_used"] = [anchor] + [r["doc"] for r in rel["outgoing"] if r["doc"]]
        payload["meta_answer"] = False
        return payload
    except Exception as e:
        logger.warning("[AGENT] structural path failed, falling through: %s", e)
        return None


def _resolve_anchor_doc(question: str, session_id: str, wiki_id: str) -> str | None:
    """The document a question is about, by page-vote over its own embedding."""
    from services import db as _db, embedder as _embedder
    vec = _embedder.embed(question, is_query=True)
    titles = _db.search_similar_pages(wiki_id, session_id, vec, limit=25,
                                      exclude_cached=True)
    counts: dict[str, int] = {}
    for t in titles:
        pg = _db.get_page(wiki_id, session_id, t) or {}
        sd = pg.get("source_doc")
        if sd:
            counts[sd] = counts.get(sd, 0) + 1
    if not counts:
        return None
    q = (question or "").lower()
    try:
        type_of = _db.get_document_types(wiki_id, session_id) or {}
    except Exception:
        type_of = {}

    def score(sd: str) -> tuple:
        dt = (type_of.get(sd) or "").lower()
        stem = re.sub(r"[^a-z0-9]+", " ",
                      sd.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower())
        named = sum(1 for t in stem.split() if len(t) > 4 and t in q)
        return (1 if dt and dt in q else 0, named, counts[sd])

    return max(counts, key=score)


_RX_PRECEDENT = re.compile(
    r"\b(?:closest\s+precedents?|nearest\s+precedents?|which\s+other\s+documents?"
    r"|what\s+other\s+documents?|similar\s+(?:documents?|agreements?|contracts?)"
    r"|most\s+similar|comparable\s+(?:documents?|agreements?))\b",
    re.IGNORECASE,
)


# Clause-level precedent, asked conversationally (§ Phase 3.5b). The Precedent
# layer's clause search has existed since Phase 2 but was reachable only from
# Draft Mode and an admin route — a lawyer asking "have we agreed to this
# before" in Ask got ordinary retrieval, which ranks documents ABOUT the topic
# rather than the clauses that actually match. Same retrieval, new entry point,
# no drafting step.
_RX_CLAUSE_PRECEDENT = re.compile(
    r"\b(?:"
    r"have\s+we\s+(?:ever\s+)?(?:agreed|accepted|signed\s+up|conceded|given)"
    r"|has\s+(?:the\s+)?(?:company|firm|business)\s+(?:ever\s+)?agreed"
    r"|(?:what|which)\s+(?:did|have)\s+we\s+(?:do|done|agree[d]?)\s+(?:before|previously|in\s+the\s+past)"
    r"|do\s+we\s+have\s+precedent"
    r"|(?:any|show\s+me)\s+precedent\s+for"
    r"|where\s+else\s+have\s+we\s+(?:agreed|used|accepted)"
    r")\b",
    re.IGNORECASE,
)


# Playbook compliance, asked conversationally (Phase 3.5c). Playbooks have only
# ever run as a batch job from Admin over a whole collection, producing a
# deviation dashboard. The question a lawyer actually asks - "is this NDA in
# satisfaction of our company rules" - reached ordinary retrieval instead,
# which reads the document back to them without ever consulting the house
# position it is supposed to be measured against.
#
# Needs BOTH halves to fire: a compliance verb AND a reference to the house
# standard. "Does this comply with the MSA" is a question about the MSA, not
# about a playbook, and must not be intercepted.
_RX_COMPLIANCE_VERB = re.compile(
    r"\b(?:comply|complies|compliant|compliance|conform(?:s|ing)?|"
    r"in\s+satisfaction\s+of|satisf(?:y|ies|ying)|adhere(?:s|nce)?|"
    r"meet(?:s|ing)?|align(?:s|ed|ment)?|consistent\s+with|in\s+line\s+with|"
    r"deviat(?:e|es|ion|ions)|acceptable\s+(?:under|per)|"
    r"(?:review|check|assess|vet)\s+(?:this|it|the)?\s*\w*\s*against)\b",
    re.IGNORECASE,
)
_RX_HOUSE_STANDARD = re.compile(
    r"\b(?:playbook|house\s+(?:position|standard|style|rules?)|"
    r"(?:our|company|firm|internal|standard|approved)\s+"
    r"(?:rules?|standards?|positions?|policy|policies|guidelines?|templates?|"
    r"requirements?|terms?|precedent)|"
    r"fallback\s+positions?|standard\s+terms?)\b",
    re.IGNORECASE,
)


def _is_compliance_query(question: str) -> bool:
    """"Does this document meet our house position?" - a playbook question.

    Deliberately conjunctive. The compliance verbs alone are far too common in
    ordinary contract questions ("does the supplier comply with Applicable
    Law", "which milestones did they meet"), and intercepting one of those
    would replace a real answer with a playbook report about a rule the
    question never mentioned.
    """
    q = question or ""
    return bool(_RX_COMPLIANCE_VERB.search(q) and _RX_HOUSE_STANDARD.search(q))


def _pick_playbook(question: str, wiki_id: str):
    """Resolve which playbook a compliance question means.

    Named outright wins; a single playbook in the wiki is unambiguous on its
    own; anything else is genuinely ambiguous and returns the candidate list
    so the caller can ask rather than guess - running the wrong house position
    produces a deviation report that reads as authoritative and is not.
    """
    from services import playbooks as _pb
    try:
        books = _pb.list_all(wiki_id)
    except Exception as e:
        logger.error("Playbook lookup failed: %s", e)
        return None, []
    if not books:
        return None, []
    q = (question or "").lower()
    named = [b for b in books if b.get("name") and b["name"].lower() in q]
    if len(named) == 1:
        return _pb.get(wiki_id, named[0]["id"]), books
    if len(books) == 1:
        return _pb.get(wiki_id, books[0]["id"]), books
    return None, books


_VERDICT_LABEL = {
    "standard": "Meets the house position",
    "fallback": "Falls back - acceptable but not preferred",
    "unacceptable": "Outside the house position",
    "missing": "Clause absent from the document",
    "unclear": "Could not be assessed",
}
_VERDICT_ORDER = ["unacceptable", "missing", "fallback", "unclear", "standard"]


def _compliance_answer(question: str, session_id: str, wiki_id: str,
                       docs: list, ask=None) -> dict | None:
    """Measure the in-scope document(s) against a playbook, clause by clause.

    Returns None (so ordinary retrieval still runs) whenever the question
    cannot be answered honestly: no playbook, no document in scope, or a
    playbook that produced no findings at all. A compliance report is acted
    on, so an empty or half-resolved one is worse than no report.
    """
    from services import playbooks as _pb
    book, candidates = _pick_playbook(question, wiki_id)
    if not book and not candidates:
        # No house position exists in this wiki at all. Ordinary retrieval is
        # genuinely the best available answer, so fall through to it.
        return None
    if not docs:
        # A playbook exists but scope could not pin an instrument — commonly a
        # whole family ("the NDAs", 148 documents). Assessing an arbitrary
        # handful of them would read as a verdict on the document the user
        # meant, and falling through answers a compliance question from
        # ordinary retrieval, which has no access to the house position at
        # all. Ask instead.
        return _canned_payload(
            "I can measure a document against the house position, but this "
            "question does not narrow to one.\n\nName the document (or pin a "
            "collection above the message box) and ask again.",
            "Compliance", "playbook-needs-document")
    if not book:
        if len(candidates) > 1:
            names = ", ".join('"%s"' % b["name"] for b in candidates[:8])
            return _canned_payload(
                "This wiki has more than one playbook, and the question does not "
                "say which house position to measure against: " + names + ".\n\n"
                "Name the playbook in the question and I will run it over "
                + wiki._norm_doc_name(docs[0]) + ".",
                "Compliance", "playbook-ambiguous")
        return None
    if not book.get("rules"):
        return None

    # Capped deliberately. A run costs one fast classification per clause per
    # rule, and a compliance question asked in chat is about the document in
    # front of the lawyer - a whole collection belongs in the Admin batch run,
    # which reports progress and stores its results.
    docs = list(docs)[:3]
    try:
        run_id = _pb.run(wiki_id, session_id, book["id"], docs, ask=ask)
        result = _pb.get_run(wiki_id, run_id, with_findings=True)
    except Exception as e:
        logger.error("Compliance run failed: %s", e)
        return None
    findings = (result or {}).get("findings") or []
    if not findings:
        return None

    by_verdict = {}
    for f in findings:
        by_verdict.setdefault(f.get("verdict") or "unclear", []).append(f)
    breaches = len(by_verdict.get("unacceptable", [])) + len(by_verdict.get("missing", []))

    names = ", ".join(wiki._norm_doc_name(d) for d in docs)
    lines = ['**%s measured against the "%s" playbook**' % (names, book["name"]), ""]
    counts = ", ".join("%d %s" % (len(by_verdict[v]), v)
                       for v in _VERDICT_ORDER if by_verdict.get(v))
    lines.append("%d rule check(s) across %d document(s): %s."
                 % (len(findings), len(docs), counts))
    lines.append("")
    lines.append("**No** - this does not fully meet the house position."
                 if breaches else
                 "**Yes** - nothing falls outside the house position.")
    lines.append("")

    for verdict in _VERDICT_ORDER:
        rows = by_verdict.get(verdict)
        if not rows:
            continue
        lines.append("### " + _VERDICT_LABEL.get(verdict, verdict.title()))
        for f in rows:
            doc = wiki._norm_doc_name(f.get("source_doc") or "")
            lines.append("- **%s** - %s" % (f.get("clause_type"), doc))
            if f.get("rationale"):
                lines.append("  - " + str(f["rationale"]))
            if f.get("redline"):
                lines.append("  - Suggested redline: " + str(f["redline"]))
            # A verdict reached against text that is not really in the stored
            # clause is what the dashboard's `grounded` flag exists to
            # separate, so it is surfaced here too rather than read as fact.
            if f.get("grounded") is False:
                lines.append("  - [warning] assessed text could not be matched to "
                             "the stored clause - treat this verdict as unverified")
        lines.append("")

    lines.append('Rules come from the "%s" playbook, not from the document. '
                 "Full results are in Admin > Deviation Dashboard (run %s)."
                 % (book["name"], run_id))

    payload = _canned_payload("\n".join(lines), "Compliance", "playbook-compliance")
    payload["meta_answer"] = False
    payload["files_used"] = docs
    # Scope IS resolved here, so the next turn may inherit it - unlike a true
    # canned reply, this answer really is about these documents.
    payload["scope_method"] = "playbook-compliance"
    payload["scope_docs"] = docs
    return payload


def _is_clause_precedent_query(question: str) -> bool:
    """"Have we agreed to this kind of term before, and what did we do?"

    Distinct from _is_precedent_query, which finds documents similar to a named
    document. This one has no anchor document at all — the subject is a KIND OF
    TERM, and the answer is the clauses themselves.
    """
    return bool(_RX_CLAUSE_PRECEDENT.search(question or ""))


def _clause_precedent_answer(question: str, session_id: str) -> dict | None:
    """Rank precedent clauses matching the term the question describes.

    Costs one embedding call (the question), no completion call — the answer is
    assembled from the retrieved clauses rather than written by a model, so
    nothing here can invent a term the corpus does not contain.

    Returns None on no hits rather than reporting an absence: clause embeddings
    do not cover the whole corpus (see precedent.coverage), so "nothing found"
    here genuinely means "not found in the embedded subset", which is not a
    statement worth making to a lawyer.
    """
    from services import precedent as _prec, wikis as _wikis, wiki
    try:
        wiki_id = _wikis.active_wiki_id()
        hits = _prec.search_clauses(wiki_id, session_id, question, limit=8)
    except Exception as e:
        logger.error("[AGENT] clause-precedent fast-path failed: %s", e)
        return None
    if not hits:
        return None

    lines = [f"**{len(hits)} precedent clause(s) matching that term:**", ""]
    for h in hits:
        doc = wiki._norm_doc_name(h.get("source_doc") or "")
        ctype = h.get("clause_type") or "Clause"
        text_ = (h.get("verbatim_text") or h.get("text") or "").strip()
        if len(text_) > 600:
            text_ = text_[:600].rsplit(" ", 1)[0] + "…"
        lines.append(f"**{ctype}** — {doc}")
        lines.append(f"> {text_}")
        lines.append("")
    lines.append("Ranked from the precedent clause index by similarity to your "
                 "question, and quoted verbatim — these are clauses already agreed "
                 "in the documents named, not drafting suggestions.")

    payload = _canned_payload("\n".join(lines), "Precedent", "clause-precedent")
    payload["files_used"] = list({h["source_doc"] for h in hits if h.get("source_doc")})
    payload["meta_answer"] = False
    return payload


def _is_precedent_query(question: str) -> bool:
    """"Which other documents are the closest precedents to X?" and friends.

    These cannot be served by ordinary retrieval: it embeds the QUESTION, and
    every page that comes back belongs to the one document the question names,
    so the model correctly reports it has nothing to compare against. The
    answer needs a document-to-document search instead — see
    db.find_similar_documents.
    """
    return bool(_RX_PRECEDENT.search(question or ""))


def _precedent_answer(question: str, session_id: str) -> dict | None:
    """Resolve the document the question is about, then rank its neighbours.

    Returns None (fall through to the normal graph) whenever the anchor
    document can't be identified or has no neighbours — a wrong precedent list
    is worse than letting the usual pipeline answer.
    """
    from services import db as _db, wikis as _wikis, embedder as _embedder
    try:
        wiki_id = _wikis.active_wiki_id()
        # Anchor: the document the question is about. Use the question's own
        # embedding, then take the source_doc that owns the most top pages.
        vec = _embedder.embed(question, is_query=True)
        titles = _db.search_similar_pages(wiki_id, session_id, vec, limit=25,
                                          exclude_cached=True)
        if not titles:
            return None
        counts: dict[str, int] = {}
        for t in titles:
            pg = _db.get_page(wiki_id, session_id, t) or {}
            sd = pg.get("source_doc")
            if sd:
                counts[sd] = counts.get(sd, 0) + 1
        if not counts:
            return None

        # Page-count alone picks the wrong anchor when a same-worded but
        # different-instrument document dominates the neighbourhood (a
        # "Software License Agreement" question resolving to a Legal Opinion).
        # The question almost always names the instrument, so a candidate whose
        # recorded doc_type appears in the question outranks raw page count.
        _q = (question or "").lower()
        try:
            type_of = _db.get_document_types(wiki_id, session_id) or {}
        except Exception:
            type_of = {}

        def _anchor_score(sd: str) -> tuple:
            dt = (type_of.get(sd) or "").lower()
            typed = 1 if dt and dt in _q else 0
            # also credit a doc whose filename words show up in the question
            stem = re.sub(r"[^a-z0-9]+", " ", sd.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower())
            toks = [t for t in stem.split() if len(t) > 4]
            named = sum(1 for t in toks if t in _q)
            return (typed, named, counts[sd])

        anchor = max(counts, key=_anchor_score)

        similar = _db.find_similar_documents(wiki_id, session_id, anchor, limit=5)
        if not similar:
            similar = _db.find_similar_documents(wiki_id, session_id, anchor,
                                                 limit=5, same_type_only=False)
        if not similar:
            return None
    except Exception as e:
        logger.warning("[AGENT] precedent path failed, falling through: %s", e)
        return None

    def _clean(name: str) -> str:
        return wiki._norm_doc_name(name) if hasattr(wiki, "_norm_doc_name") else name

    lines = [f"The closest precedents to **{_clean(anchor)}** in this corpus are:", ""]
    for i, s in enumerate(similar, 1):
        pct = round(s["best_score"] * 100)
        lines.append(f"{i}. **{_clean(s['source_doc'])}** — {pct}% similar"
                     f" across {s['pages_matched']} matching section(s)")
    lines.append("")
    lines.append("Ranked by similarity between this document's content and every "
                 "other document in the corpus, not by title or date.")
    if similar and similar[0].get("doc_type"):
        lines.append(f"Restricted to other documents of the same type "
                     f"(*{similar[0]['doc_type']}*).")

    payload = _canned_payload("\n".join(lines), "Precedents", "precedent-similarity")
    payload["files_used"] = [s["source_doc"] for s in similar]
    payload["meta_answer"] = False
    return payload


def _social_query_kind(question: str) -> str:
    """'greeting' | 'thanks' | 'farewell' | 'ack' | '' — whole-message match only."""
    q = (question or "").strip()
    # A real question is never this short; the cap is a second guard on top of
    # the `$` anchors so no long message can reach the patterns at all.
    if not q or len(q) > 60:
        return ""
    if _RX_GREETING.match(q):
        return "greeting"
    if _RX_THANKS.match(q):
        return "thanks"
    if _RX_FAREWELL.match(q):
        return "farewell"
    if _RX_ACK.match(q):
        return "ack"
    return ""


def _social_answer(kind: str, session_id: str) -> dict:
    """Canned reply to a greeting / thanks / sign-off — no LLM call, no tokens."""
    if kind == "greeting":
        corpus = _corpus_summary(session_id, max_families=4)
        answer = (
            "Hello — I answer questions about the legal documents in this workspace."
            + (f" {corpus}" if corpus else "")
            + "\n\nAsk me about any of them, for example *\"What are the termination "
              "rights in Service Agreement 2?\"* — or type **help** for the full list "
              "of what I can do."
        )
    elif kind == "thanks":
        answer = ("You're welcome. Ask me anything else about the documents in this "
                  "workspace whenever you're ready.")
    elif kind == "farewell":
        answer = "Goodbye — come back whenever you need something from these documents."
    else:
        answer = ("Ready when you are. Ask another question about the documents, or "
                  "type **help** to see what I can do.")

    return _canned_payload(answer, "Help", f"social-{kind}")


# ---------------------------------------------------------------------------
# Legal-advice framing
# ---------------------------------------------------------------------------
# "Should I sign this?" is usually answerable from the documents — the
# termination clause, the liability cap — so this must NOT short-circuit or
# refuse. What it changes is the framing: the user asked for a decision and gets
# back a document summary, and nothing on screen said which of the two it is.
#
# Detection is regex-only (0 tokens) and its only effect is an `advice_notice`
# string on the payload that the frontend renders under the answer, so a false
# positive costs one banner and cannot alter the answer itself.
_RX_ADVICE_SEEKING = re.compile(
    r'\b(?:'
    r'(?:should|shall|must|ought\s+to|can|could|may|do|am|are|will|would)\s+'
    r'(?:i|we|my\s+client|our\s+client)\b'
    r'|what\s+(?:should|would|do)\s+(?:i|we|you)\s+(?:do|recommend|advise|suggest)'
    r'|what\s+are\s+(?:my|our)\s+(?:options|rights|chances)'
    r'|(?:my|our)\s+(?:legal\s+)?(?:rights|options|exposure|liability|risk)\b'
    r'|do\s+(?:i|we)\s+have\s+(?:a|any)\s+(?:case|claim|grounds|defence|defense)'
    r'|(?:advise|advice)\s+(?:me|us)\b'
    r'|what\s+would\s+you\s+(?:do|recommend|advise)'
    # Allows a noun between the pronoun and the adjective ("is this CLAUSE
    # enforceable"). `valid until/from/for` is excluded — that is a question
    # about the term length, not about legal validity.
    r'|(?:is|are|was|would|will)\s+(?:this|it|that|these|those|the)\s+(?:[\w\'’-]+\s+){0,3}'
    r'(?:legally\s+)?(?:enforceable|binding|lawful|valid\b(?!\s+(?:until|from|till|through|for)\b))'
    r'|is\s+(?:this|it|that)\s+legal\b'
    r'|(?:hold|stand)\s+up\s+in\s+court'
    r'|(?:can|could|would)\s+(?:they|he|she|the\s+other\s+party|the\s+counterparty)\s+sue'
    r'|is\s+(?:this|it)\s+a\s+good\s+(?:deal|idea)'
    r')\b',
    re.IGNORECASE,
)

_ADVICE_NOTICE = (
    "This answers what the documents say, not what you should do. It is not legal "
    "advice, does not account for facts, jurisdiction or case law outside this "
    "workspace, and should be reviewed by a qualified lawyer before you act on it."
)

# The pipeline's own bracketed disclosures — "[SCOPE NOTE: …]", "[SCOPE WARNING: …]",
# "[CITATION NOTE: …]". Stripped before asking whether an answer actually said
# anything: they are appended regardless of outcome, so a payload holding nothing
# but disclosures is an empty answer wearing a full-looking body.
_RX_BRACKET_NOTE = re.compile(r'\[[A-Z][A-Z \-]{2,30}:.*?\]', re.DOTALL)


def _is_advice_seeking(question: str) -> bool:
    """True when the question asks for a decision rather than a document fact."""
    return bool(_RX_ADVICE_SEEKING.search(question or ""))


# ---------------------------------------------------------------------------
# Personal legal predicaments
# ---------------------------------------------------------------------------
# The subject-match rule in the answer prompts is a judgement the model re-makes
# on every request, and it fails on phrasings where the mismatch is subtle.
# Measured: "…my ITR legal issue…my personal income tax return filing" refuses
# 3/3, but the shorter "can u advice me ragarding my ITR legal issue i have been
# facing this year" leaked 3/3 — each time opening "the context DOES address the
# question's subject" and then serving corporate joint-venture tax-structuring
# advice (retain a Big Four firm, review your 704(b) allocations) to someone
# asking about their own tax return. Every sentence true of those opinions; the
# advice still wrong, because a company's tax structuring is not a person's tax
# filing and "tax" was the only thing connecting them.
#
# A third prompt revision would only move which phrasings fail. This is a
# deterministic gate instead, and it sits with the other pre-checks rather than
# inside the pipeline: a question about the user's OWN personal legal
# predicament is not a question about the documents at all, so there is nothing
# for retrieval to do and no answer that could be grounded.
#
# Deliberately NOT built on the term-presence check above — that mechanism is
# documented as warning-only on purpose, because the corpus has a real "wrongly
# said the document is silent" failure and a suppressing gate would worsen it.
#
# All three conditions must hold, which is what keeps it off real questions:
#   1. the matter is a personal-life legal domain this corpus does not hold,
#   2. it is framed as the USER'S OWN (not "my client's", not a document's),
#   3. the question names no document.

# Legal domains belonging to an individual's private life. A corpus of
# commercial agreements, judgments and opinions holds none of them.
_RX_PERSONAL_DOMAIN = re.compile(
    r'\b(?:'
    r'itr\b|income[\s-]?tax[\s-]?return|tax[\s-]?return|tax[\s-]?filing|form\s*16'
    r'|divorce|alimony|custody|maintenance\s+petition|matrimonial'
    r'|landlord|tenant|rent\s+agreement|eviction'
    r'|visa|immigration|passport|citizenship|green\s+card'
    r'|\bwill\b\s+(?:and|&)\s+testament|probate|inheritance|succession\s+certificate'
    r'|personal\s+injury|accident\s+claim|insurance\s+claim'
    r'|criminal\s+case|\bfir\b|bail|police\s+complaint'
    r'|consumer\s+(?:court|complaint|forum)'
    r'|provident\s+fund|gratuity|pension'
    r'|my\s+(?:salary|employer|boss|landlord|marriage|property|flat|house)'
    r')\b',
    re.IGNORECASE,
)

# The matter must be the USER'S OWN. "My client's divorce" is professional use
# and stays in the normal pipeline; so does any third-party framing.
_RX_PERSONAL_OWNERSHIP = re.compile(
    r'\b(?:my|mine|i|me|myself|we|our|us)\b',
    re.IGNORECASE,
)

# Professional framing that disqualifies condition 2 — a lawyer asking on behalf
# of someone else is doing exactly what this tool is for.
_RX_ON_BEHALF = re.compile(
    r'\b(?:my|our)\s+client|\bthe\s+client\b|\bon\s+behalf\s+of\b',
    re.IGNORECASE,
)

# Any document reference sends it back to the pipeline — "what does my tax
# indemnity say in SA 2" is a document question wearing personal phrasing.
_RX_PERSONAL_DOC_REF = re.compile(
    r'\b(?:clause|section|article|schedule|annexures?|annexes?|exhibit|appendix'
    # "files" plural only — bare "file" is usually the VERB here ("file an FIR",
    # "file my return"), and vetoing on it sent personal matters back into the
    # pipeline, which is the exact failure this gate exists to stop.
    r'|agreements?|contracts?|nda|sha|jva|msa|sow|deed|documents?|files'
    r'|workspace|corpus|judgment|opinion|pleading)\b'
    r'|\b[a-z]{2,}\s*[-_]?\s*\d{1,3}\b',
    re.IGNORECASE,
)


def _is_personal_matter(question: str) -> bool:
    """True when the question is about the user's own private legal predicament."""
    q = (question or "").strip()
    if not q:
        return False
    if not _RX_PERSONAL_DOMAIN.search(q):
        return False
    if _RX_ON_BEHALF.search(q):
        return False
    if not _RX_PERSONAL_OWNERSHIP.search(q):
        return False
    if _RX_PERSONAL_DOC_REF.search(q):
        return False
    return True


def _personal_matter_answer(session_id: str) -> dict:
    """Canned scope reply — no LLM call, no retrieval, no tokens.

    Says plainly that this is outside what the workspace can answer, rather than
    reaching for the nearest corporate document and reasoning from it.
    """
    corpus = _corpus_summary(session_id, max_families=4)
    answer = (
        "That sounds like a matter about your own situation, and it isn't something "
        "I can help with here.\n\n"
        "I can only tell you what the documents loaded into this workspace say"
        + (f" — {corpus[0].lower() + corpus[1:]}" if corpus else ".")
        + "\n\nThose are commercial legal documents, so they contain nothing about a "
        "personal matter like this, and answering from them would mean applying "
        "terms written for an entirely different situation to yours. For something "
        "affecting you personally, please speak to a qualified lawyer or adviser "
        "who can look at your actual facts.\n\n"
        "If you did mean a document in this workspace, name it and I'll pull it up."
    )
    return _canned_payload(answer, "Out of scope", "personal-matter")


# ---------------------------------------------------------------------------
# General legal knowledge
# ---------------------------------------------------------------------------
# "What is arbitration?" has no answer in a corpus of executed contracts — they
# USE the concept, they do not define it — so it retrieved whatever mentioned
# the word and reported "not covered" at full token cost. Same shape as the meta
# and social fast-paths above: a question the pipeline structurally cannot
# answer, recognised before the pipeline runs.
#
# Unlike those two, this path costs an LLM call, and its answer is the ONLY
# output in this system with no document behind it — nothing for
# _verify_answer_citations to check, no grounding score that would mean
# anything. That asymmetry drives every design decision here:
#
#   * the gates are deterministic regex, in a fixed order, and the LLM tiebreak
#     runs LAST and only on questions that have already cleared every one of
#     them — so it can never pull a document question onto this path,
#   * failing any gate means falling through to the normal pipeline, which is
#     exactly today's behaviour, so a miss costs nothing new,
#   * the answer itself is capped and labelled, structurally (a render flag the
#     frontend reads) rather than trusting the model to say so in prose.
#
# It runs BEFORE resolve_scope and carryover deliberately. Placed later, a
# genuinely general question asked mid-thread ("by the way, what is novation?")
# would inherit the scope of the document under discussion and be answered as
# though that document defined the term.

# Definitional / explanatory openers. Anchored at ^ — nothing here matches
# mid-sentence, so a real question that happens to contain "what is" further in
# ("tell me the notice period and what is the cure window") never matches.
_RX_GK_OPENER = re.compile(
    r'^\s*'
    r'(?:(?:so|and|but|also|ok(?:ay)?|btw|by\s+the\s+way|just|hey|hi)\b[\s,]*){0,2}'
    r'(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?'
    r'(?:'
    r"what(?:'s|’s|\s+is|\s+are|\s+does|\s+do)\b"
    r'|what\s+exactly\s+(?:is|are|does|do)\b'
    r'|what\s+do(?:es)?\s+(?:the\s+)?(?:term|word|phrase)\b'
    r'|explain\b|define\b|describe\b'
    r'|tell\s+me\s+(?:what|about|how)\b'
    r'|how\s+(?:does|do|is|are)\b'
    r'|(?:the\s+)?(?:meaning|definition|concept|purpose)\s+of\b'
    r')',
    re.IGNORECASE,
)

# Abstract framing consumed off the front of the subject so the definite-article
# veto below doesn't reject "what is THE doctrine of frustration" — "the" there
# points at a concept, not at a document.
_RX_GK_SUBJECT_STRIP = re.compile(
    r'^\s*(?:the\s+)?'
    # An adjective may sit between the article and the framing noun — "the LEGAL
    # doctrine of unclean hands" is the same shape as "the doctrine of
    # frustration", but without this the article survives and the veto below
    # rejects a plainly general question (confirmed live on "unclean hands").
    r'(?:(?:legal|equitable|common[\s-]law|general|basic|underlying)\s+)?'
    r'(?:doctrine|concept|principle|term|word|phrase|meaning|'
    r'definition|purpose|idea|notion|rule|test|standard)\s+of\s+',
    re.IGNORECASE,
)

# Named legal authorities and tests keep their definite article as part of the
# name — "THE Delaware Uniform Trade Secrets Act", "THE Alice test". The veto
# below reads that article as pointing at a workspace document, so without this
# exception a textbook question about a named statute is answered from whatever
# agreement happened to be under discussion. Requires a capitalised name (or a
# quoted one) in front of the category noun, which no bare document reference in
# this corpus has — those carry a number and are caught by _RX_GK_DOC_REF.
_GK_AUTHORITY_NOUN = (r'(?i:acts?|doctrines?|rules?|tests?|standards?|conventions?|'
                      r'codes?|treaties|treaty|principles?)')
_RX_GK_NAMED_AUTHORITY = re.compile(
    r'^\s*the\s+(?:'
    # Quoted name carrying the category noun inside the quotes — the "Alice test".
    rf'["“][^"”]*\b{_GK_AUTHORITY_NOUN}["”]'
    # Capitalised name followed by the category noun — the Delaware Uniform
    # Trade Secrets Act. Case-sensitive on the name so a lowercase phrase such
    # as "the termination clause test" cannot qualify.
    rf'|(?:[A-Z][\w.\-]*\s+){{0,5}}{_GK_AUTHORITY_NOUN}\b'
    r')',
)

# The subject must be a bare concept. A definite article or a demonstrative
# means the user is pointing at something specific — "what is THE governing
# law", "what does THIS mean" — which is a document question in a workspace
# full of documents, and belongs in the normal pipeline.
_RX_GK_SUBJECT_VETO = re.compile(
    r'^\s*(?:the|this|that|these|those|it|its|such|said|above|'
    r'our|my|your|their|his|her)\b',
    re.IGNORECASE,
)

# Anywhere in the question: a reference to the workspace's own material. Wider
# than strictly necessary — "what is a force majeure clause" is vetoed by
# `clause` even though it reads general — because the safe direction is the
# document pipeline, which already answers that well.
_RX_GK_DOC_REF = re.compile(
    r'\b(?:'
    r'clause|section|article|schedule|annexures?|annexes?|annex|exhibit|appendix|recital'
    r'|documents?|files?|pages?|workspace|corpus|uploaded|attached'
    r'|nda|sha|jva|msa|sow|deed'
    r'|part(?:y|ies)|counterpart(?:y|ies)'
    r'|here|above|earlier|previous(?:ly)?|mentioned|aforesaid'
    r')\b'
    r'|\b[a-z]{2,}\s*[-_]?\s*\d{1,3}\b'     # "agreement 2", "sa 01", "nda3"
    r'|\b\d+(?:\.\d+)+\b'                    # clause numbers — 5.2.1
    # A demonstrative/possessive next to "agreement"/"contract" points at a
    # specific document ("in THIS contract", "OUR agreement") even though
    # neither bare word is a reliable veto on its own — "what is a service
    # agreement" is a genuine general question. Confirmed live: "what does
    # force majeure mean in this contract" cleared every other gate and was
    # answered as a textbook definition, never touching retrieval, silently
    # discarding the one word in the question that pointed at an actual file.
    r'|\b(?:this|that|these|those|our|my|the)\s+(?:agreements?|contracts?)\b',
    re.IGNORECASE,
)

# Hard block, checked before anything else can let a question through. This is
# the gate that keeps the carve-out from becoming a legal-advice channel, so it
# is deliberately over-broad: it blocks "what is legal advice" along with the
# cases it exists for, and blocking those costs only a fall-through to the
# normal pipeline.
#
# First-person framing is the signal. A genuine general-knowledge question
# needs no "I" or "my" — "explain to me what novation is" survives (the block
# is on first-person SUBJECTS and possessives, not the object of "tell/explain
# me"), while "what does novation mean for my contract" does not.
#
# A former line here also bare-matched compl(y|ies|iant|iance) — no
# self-reference required — reasoning that a "fall-through" was a harmless
# cost. Confirmed live it was not: "what is compliance?" fell through, named no
# document, disambiguated three times, and finally answered from an unrelated
# document instead of the one the conversation had just established. Removed;
# every self-referential compliance phrasing this was meant to catch ("are we
# compliant", "is our data GDPR compliant") is already covered by the pronoun
# lines above, since it always pairs "compliant/comply" with my/our/we/are-we —
# the bare word alone was doing no real work, only causing this failure.
_RX_GK_ADVICE_BLOCK = re.compile(
    r'\b(?:my|mine|our|ours|myself|ourselves)\b'
    r"|\b(?:i|we)\s*(?:'m|'re|'ve|'ll|'d)\b"
    r'|\b(?:i|we)\s+(?:am|are|was|were|have|had|has|need|want|think|face|faced|'
    r"facing|got|get|do|don'?t|did|didn'?t|should|shall|can|could|may|might|must|"
    r'will|would|filed?|signed?|received?|paid|owe|run|own|work|live)\b'
    r'|\b(?:should|shall|can|could|may|must|do|does|did|will|would|am|are|is)\s+'
    r'(?:i|we|one)\b'
    r'|\bin\s+(?:my|our|this|that)\s+(?:case|situation|matter|position)\b'
    r'|\bwhat\s+should\b'
    r'|\badvi[cs]e\b|\brecommend\w*\b'
    r'|\bsue\b|\bsuing\b|\blawsuit\b'
    r'|\bapplies?\s+to\s+(?:me|us)\b',
    re.IGNORECASE,
)

# Legal vocabulary the subject must contain for the regex path to fire. Not
# exhaustive by design and never will be — the LLM tiebreak below exists
# precisely to cover the terms this list misses ("what is laches"). Kept to
# concepts, doctrines and procedures; deliberately excludes bare "law"/"legal",
# which appear in far too many document questions to be a reliable signal.
_RX_GK_LEGAL_TERM = re.compile(
    r'\b(?:'
    # Dispute resolution and procedure
    r'arbitrat\w*|mediat\w*|conciliat\w*|litigat\w*|adjudicat\w*|tribunals?|'
    r'injunct\w*|subpoenas?|depositions?|discovery|pleadings?|affidavits?|'
    r'plaintiffs?|defendants?|appellants?|respondents?|claimants?|'
    r'jurisdictions?|appeals?|writs?|decrees?|summons|'
    r'class\s+action|due\s+process|burden\s+of\s+proof|statutes?\s+of\s+limitations?|'
    r'limitation\s+period|res\s+judicata|locus\s+standi|prima\s+facie|'
    # Contract law
    r'contracts?|consideration|privity|novation|assignments?|'
    r'rescission|rescind\w*|repudiat\w*|frustration|force\s+majeure|'
    r'conditions?\s+precedent|indemnit\w*|indemnif\w*|warrant\w*|covenants?|'
    r'guarantees?|suret\w*|liquidated\s+damages|specific\s+performance|'
    r'severab\w*|waivers?|estoppel|breach\w*|boilerplate|'
    r'non[\s-]?compete|non[\s-]?disclosure|confidentialit\w*|'
    # Liability, tort, remedies
    r'liabilit\w*|neglig\w*|torts?|damages|remed\w*|restitution|'
    r'vicarious|strict\s+liability|duty\s+of\s+care|causation|mitigat\w*|'
    r'unjust\s+enrichment|nuisance|defamation|misrepresentation|'
    # Corporate and commercial
    r'fiduciar\w*|due\s+diligence|escrow|liens?|mortgages?|pledges?|'
    r'shareholders?|incorporat\w*|winding\s+up|insolvenc\w*|'
    r'bankrupt\w*|liquidation|mergers?|acquisitions?|joint\s+ventures?|'
    r'partnerships?|limited\s+liability|piercing\s+the\s+(?:corporate\s+)?veil|'
    # Intellectual property
    r'copyrights?|trade\s?marks?|patents?|trade\s+secrets?|'
    r'intellectual\s+propert\w*|licens\w*|royalt\w*|infring\w*|moral\s+rights|'
    # Property and employment
    r'leases?|tenanc\w*|landlords?|easements?|freehold|leasehold|conveyanc\w*|'
    r'redundanc\w*|wrongful\s+dismissal|'
    # Jurisprudence
    r'justice|equit\w*|jurisprudence|common\s+law|civil\s+law|'
    r'legislation|precedents?|stare\s+decisis|rule\s+of\s+law|natural\s+justice|'
    r'good\s+faith|bona\s+fide|ultra\s+vires|mens\s+rea|actus\s+reus'
    r')\b',
    re.IGNORECASE,
)

# A statute citation names public law, never a workspace document, but its
# shape trips _RX_GK_DOC_REF's generic "word followed by a number" alternation —
# "under 35" in "35 U.S.C. § 101" reads exactly like "agreement 2". Removed from
# the question before that check rather than loosening the alternation itself,
# which would let real document references through.
_RX_GK_STATUTE_CITE = re.compile(
    r'\b\d+\s*U\.?\s?S\.?\s?C\.?(?:\s*§+)?(?:\s*[\d]+[\w.()\-]*)?'
    r'|§+\s*[\w.()\-]+',
    re.IGNORECASE,
)

# Same cap as the meta fallback, for the same reason: a genuine definitional
# question is short, and a long message this deep into the gates is not one.
_GK_LLM_MAX_LEN = 120

_GK_LLM_PROMPT = (
    "A user asked a definition-style question. Decide whether its subject is a "
    "LEGAL term, doctrine, or procedure — something you would find in a law "
    "dictionary or a legal textbook — or something else entirely.\n\n"
    'Question: "{question}"\n\n'
    'Respond with ONLY one word: "legal" or "other".'
)


def _gk_subject(question: str) -> str | None:
    """The concept a definitional question is about, or None if it isn't one.

    Returns the text after the opener, with abstract framing ("the doctrine
    of…") removed. None means the question never looked definitional in the
    first place, or its subject points at something specific rather than at a
    concept.
    """
    m = _RX_GK_OPENER.match(question or "")
    if not m:
        return None
    subject = _RX_GK_SUBJECT_STRIP.sub("", question[m.end():]).strip()
    # Drop the trailing verb/punctuation a definitional question ends on, so
    # "arbitration work?" reduces to the concept itself.
    subject = re.sub(r'\s*(?:mean(?:s|ing)?|work(?:s)?|entail|involve(?:s)?)?\s*[?.!]*\s*$',
                     '', subject, flags=re.IGNORECASE).strip()
    if not subject:
        return None
    if _RX_GK_SUBJECT_VETO.match(subject) and not _RX_GK_NAMED_AUTHORITY.match(subject):
        return None
    return subject


def _question_names_corpus_entity(question: str, session_id: str) -> bool:
    """True if the question names a party/entity this corpus holds documents about.

    The general-knowledge gates recognise a document reference only by KEYWORD —
    clause, section, NDA, MSA, SOW, party, "this agreement" (_RX_GK_DOC_REF). A
    question that names a real COUNTERPARTY instead carries none of those words,
    so it clears every gate and is answered from textbook knowledge while the
    corpus holds the clause verbatim. Confirmed live, repeatedly: "What are Hyden
    Tech's obligations regarding AI bias and discrimination?" took the standalone
    path on three separate runs — "Hyden Tech" is not a gate keyword, and
    "discrimination" then matched the legal-vocabulary regex — while DPA Clause
    4.6 and SOW Clause 4.7 answer it directly. Same shape for "how is ownership
    handled for AI-generated deliverables", answered generically against MSA
    Clause 5.4, which addresses exactly that.

    Uses the proper-noun-aware check, not the looser token match: a lowercase
    clause word that leaked into the entity set ("termination", "liability")
    must not divert a genuine general question into retrieval.

    Delegates to wiki._question_names_distinctive_entity, which itself checks
    both page titles and source_doc filenames — see that function for why both
    are needed ("Hyden" lives only in filenames, not titles).
    """
    try:
        index = wiki._load_index(session_id)
        pages = index.get("pages", {})
        return bool(pages) and wiki._question_names_distinctive_entity(question, pages)
    except Exception as e:
        logger.warning("Corpus-entity check for general-knowledge gate failed: %s", e)
        return False


def _is_general_knowledge_llm(question: str) -> bool:
    """Cheap tiebreak for a legal term the vocabulary regex doesn't list.

    Only ever reached after every deterministic gate has passed — definitional
    phrasing, no document reference, no advice framing, short. So the worst it
    can do is decline to answer a general question (identical to today), never
    divert a document question onto the general path.
    """
    try:
        raw, _ = llm.ask(
            _GK_LLM_PROMPT.format(question=question),
            fast=True, max_tokens=config.MAX_TOKENS_META_CLASSIFY,
        )
        return raw.strip().lower().startswith("legal")
    except Exception as e:
        logger.warning("General-knowledge LLM tiebreak failed: %s", e)
        return False


def _general_knowledge_kind(question: str) -> str:
    """'gk-regex' | 'gk-llm' | 'gk-named' | '' — which gate, if any, claims this.

    Every gate must pass. Order matters only for cost: the advice block runs
    before the vocabulary check so an advice-seeking question is rejected
    without ever being scored as legal.

    'gk-named' is deliberately weaker than the other two. A bare concept
    ("what is novation") is textbook material a commercial corpus is unlikely to
    define, so answering it without retrieval is right. A NAMED authority — the
    Delaware Uniform Trade Secrets Act, the Alice test — is exactly what legal
    opinions and pleadings discuss at length, and a corpus that holds that
    discussion should answer from it. So the caller runs retrieval first for
    these and keeps the general answer as a fallback (see run_query_stream);
    skipping retrieval here would replace a document's own analysis with a
    dictionary definition.
    """
    if not config.ENABLE_GENERAL_KNOWLEDGE:
        return ""
    q = (question or "").strip()
    if not q:
        return ""
    if _RX_GK_DOC_REF.search(_RX_GK_STATUTE_CITE.sub(" ", q)):
        return ""
    if _RX_GK_ADVICE_BLOCK.search(q) or _is_advice_seeking(q):
        return ""
    subject = _gk_subject(q)
    if subject is None:
        return ""
    if _RX_GK_NAMED_AUTHORITY.match(subject):
        return "gk-named"
    if _RX_GK_LEGAL_TERM.search(subject):
        return "gk-regex"
    if not config.ENABLE_GK_LLM_FALLBACK or len(q) > _GK_LLM_MAX_LEN:
        return ""
    return "gk-llm" if _is_general_knowledge_llm(q) else ""


# Appended by code, not asked of the model. A label the model has to remember
# to write is a label it will eventually omit, and this one carries the whole
# distinction between a checkable answer and an unchecked one.
_GK_FOOTER = (
    "\n\n---\n\n*This is general legal information, not drawn from the documents "
    "in this workspace and not checked against them. It is not legal advice — for "
    "anything touching your own situation, consult a qualified lawyer.*"
)

# Belt-and-braces on the prompt's "no References section, no confidence score"
# rules: the model has seen those sections in every other prompt in this file
# and occasionally emits them out of habit. They are meaningless here — there
# is nothing to reference — so they are stripped rather than rendered.
_RX_GK_STRIP_TAIL = re.compile(
    r'\n+\s*(?:#{1,4}\s*)?(?:\*\*)?(?:References?|Sources?|CONFIDENCE_SCORE|'
    r'CONFIDENCE_REASON)\b.*$',
    re.IGNORECASE | re.DOTALL,
)


def _general_knowledge_aside(question: str) -> str:
    """Short general definition to sit BESIDE a document-grounded answer.

    Returns "" on any failure — the aside is an addition, so losing it must
    never cost the user the grounded answer it accompanies.
    """
    from services.prompts import GENERAL_KNOWLEDGE_ASIDE_PROMPT

    try:
        raw, _ = llm.ask(
            GENERAL_KNOWLEDGE_ASIDE_PROMPT.format(question=question),
            max_tokens=config.MAX_TOKENS_GENERAL_KNOWLEDGE,
        )
    except Exception as e:
        logger.error("General-knowledge aside failed: %s", e)
        return ""
    text = _RX_GK_STRIP_TAIL.sub("", (raw or "").strip()).strip()
    if not text:
        return ""
    # A citation marker here would be a lie — nothing was retrieved for this
    # text. Strip rather than render one next to genuinely cited prose.
    text = re.sub(r'\[\d+\]', '', text).strip()
    if len(text) > _GK_ASIDE_MAX_CHARS:
        text = text[:_GK_ASIDE_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return text


# The aside is a margin note beside a real answer. A long one starts competing
# with the cited content for attention, which is exactly the confusion the
# separate field exists to prevent.
_GK_ASIDE_MAX_CHARS = 900


def _general_knowledge_answer(question: str, method: str) -> dict | None:
    """Answer a general legal-knowledge question — no retrieval, no citations.

    Returns None if the call fails or comes back empty, in which case the caller
    falls through to the normal pipeline rather than showing an error.
    """
    from services.prompts import GENERAL_KNOWLEDGE_PROMPT

    try:
        raw, usage = llm.ask(
            GENERAL_KNOWLEDGE_PROMPT.format(question=question),
            max_tokens=config.MAX_TOKENS_GENERAL_KNOWLEDGE,
        )
    except Exception as e:
        logger.error("General-knowledge generation failed: %s", e)
        return None

    text = _RX_GK_STRIP_TAIL.sub("", (raw or "").strip()).strip()
    if not text:
        logger.warning("General-knowledge generation returned nothing — falling through")
        return None

    payload = _canned_payload(text + _GK_FOOTER, "General", method)
    # Reuses the meta render path (no confidence, no grounding, no References —
    # none of which exist here), and adds the flag the frontend keys the
    # general-knowledge banner and tag off.
    payload["general_knowledge"] = True
    payload["confidence_score"] = 0        # not 100 — there is nothing to be confident against
    payload["intent_confidence"] = 1.0
    # llm.ask returns prompt/completion counts only — no total — so it is summed
    # here rather than read with a .get that would silently report every
    # general-knowledge answer as costing zero tokens.
    _prompt_tok = (usage or {}).get("prompt_tokens", 0) or 0
    _completion_tok = (usage or {}).get("completion_tokens", 0) or 0
    payload["token_total"] = {
        "prompt_tokens": _prompt_tok,
        "completion_tokens": _completion_tok,
        "total_tokens": _prompt_tok + _completion_tok,
    }
    return payload


def run_query_stream(question: str, session_id: str, target_doc: str = "",
                     is_followup: bool = False, exclude_cached_answers: bool = False,
                     chat_session_id: str = "", collection_id=None):
    """Run the query graph and yield stage event dicts in real time.

    Each yielded dict is a custom stage event emitted by a node. The terminal
    'complete' / 'disambiguation' / 'clarification' event carries the payload
    the frontend renders. app.py wraps each dict as a Server-Sent Event.

    exclude_cached_answers: QA/testing option — see wiki.get_context().
    collection_id: when set, confines retrieval to that collection's member
        documents — see _apply_collection_scope. The corpus-wide fast paths
        below are skipped while a collection is pinned, since each answers
        over the whole wiki and would silently escape the boundary.
    """
    # Greetings and sign-offs short-circuit before the graph. Unlike the meta
    # path below this one DOES run on follow-ups — "thanks" after an answer is
    # exactly when it happens — and it is safe there because every pattern is a
    # whole-message match.
    social = _social_query_kind(question)
    if social:
        logger.info("[AGENT] social fast-path (%s, 0 tokens): %r", social, (question or "")[:40])
        yield {"stage": "complete", "status": "done", "type": "answer",
               "payload": _social_answer(social, session_id), "message": "Done"}
        return

    # "Which other documents are the closest precedents to X?" needs a
    # document-to-document search, which the retrieval graph cannot express —
    # it embeds the question, so every page it finds belongs to the single
    # document the question names. Falls through to the graph if the anchor
    # document or its neighbours can't be resolved.
    # Citation and amendment questions are answered by a SQL join over the
    # typed tables rather than by retrieval, which ranks documents ABOUT the
    # subject above the ones actually carrying the citation.
    # Corpus-shaped analytics (aggregate / gap / trend) — checked before the
    # document-oriented structural paths, since "the average liability cap
    # across our contracts" is about the corpus, not about any one document,
    # and retrieval would answer it from whatever sample it happened to fetch.
    if not is_followup and not collection_id:
        _akind = _is_analytics_query(question)
        if _akind:
            from services import wikis as _wikis_a
            try:
                _a = _analytics_answer(_akind, question, session_id, _wikis_a.active_wiki_id())
            except Exception as _a_err:
                logger.error("[AGENT] analytics fast-path failed: %s", _a_err)
                _a = None
            if _a:
                logger.info("[AGENT] analytics fast-path (%s): %r", _akind, (question or "")[:70])
                yield {"stage": "complete", "status": "done", "type": "answer",
                       "payload": _a, "message": "Done"}
                return

    if not is_followup and not collection_id:
        _kind = _is_structural_query(question)
        if _kind:
            _structural = _structural_answer(_kind, question, session_id)
            if _structural:
                logger.info("[AGENT] structural fast-path (%s): %r",
                            _kind, (question or "")[:70])
                yield {"stage": "complete", "status": "done", "type": "answer",
                       "payload": _structural, "message": "Done"}
                return

    if not is_followup and not collection_id and _is_precedent_query(question):
        _prec = _precedent_answer(question, session_id)
        if _prec:
            logger.info("[AGENT] precedent fast-path: %r", (question or "")[:70])
            yield {"stage": "complete", "status": "done", "type": "answer",
                   "payload": _prec, "message": "Done"}
            return

    # Clause-level precedent — "have we agreed to this before". Checked after
    # the document-level path above, which is the more specific question when
    # both could match.
    if not is_followup and not collection_id and _is_clause_precedent_query(question):
        _cprec = _clause_precedent_answer(question, session_id)
        if _cprec:
            logger.info("[AGENT] clause-precedent fast-path: %r", (question or "")[:70])
            yield {"stage": "complete", "status": "done", "type": "answer",
                   "payload": _cprec, "message": "Done"}
            return

    # Playbook compliance. Unlike every other fast path this one needs the
    # document scope first — "is this NDA in satisfaction of our rules" is
    # about a specific instrument — so it resolves scope itself rather than
    # keying off the question alone. resolve_scope costs no LLM call, and the
    # fallthrough path re-resolves it inside the graph, so a miss here is
    # cheap and leaves ordinary retrieval completely unchanged.
    if not is_followup and _is_compliance_query(question):
        from services import wikis as _wikis_c
        try:
            _wid_c = _wikis_c.active_wiki_id()
            _scope_c = wiki.resolve_scope(question, session_id,
                                          chat_session_id=chat_session_id or session_id)
            _scope_c = _apply_collection_scope(_scope_c, collection_id)
            _docs_c = (_scope_c.get("target_docs") or []
                       if _scope_c.get("scope") == "single_doc" else [])
            if target_doc:
                _docs_c = [target_doc]
            _comp = _compliance_answer(question, session_id, _wid_c, _docs_c)
        except Exception as _c_err:
            logger.error("[AGENT] compliance fast-path failed: %s", _c_err)
            _comp = None
        if _comp:
            logger.info("[AGENT] playbook-compliance fast-path: %r", (question or "")[:70])
            yield {"stage": "complete", "status": "done", "type": "answer",
                   "payload": _comp, "message": "Done"}
            return

    # Meta questions short-circuit before the graph — no classify, no retrieve,
    # no generate, no validate. Skipped for follow-ups, where a bare "help" is
    # far more likely to be an answer to a previous prompt than a fresh request
    # for capabilities.
    if not is_followup and _is_meta_query_extended(question):
        logger.info("[AGENT] meta query fast-path: %r", (question or "")[:80])
        yield {"stage": "complete", "status": "done", "type": "answer",
               "payload": _meta_answer(session_id), "message": "Done"}
        return

    # General legal-knowledge questions ("what is arbitration") answer from a
    # separate path with no retrieval — see _general_knowledge_kind. Placed here
    # for a reason: everything below this line resolves document scope, and a
    # general question that reaches resolve_scope inherits whatever document the
    # thread was last about and gets answered as though that document defined
    # the term.
    #
    # Unlike the meta path above, this DOES run on follow-ups — "by the way,
    # what is novation?" mid-thread is exactly when these get asked. That is
    # safe because the gates require a named legal concept as the subject and
    # veto every demonstrative, so a follow-up that depends on the previous turn
    # ("what does that mean?") can never match.
    # A question about the user's own personal legal predicament has no answer
    # in a corpus of commercial agreements, and letting it reach retrieval is
    # how corporate tax-structuring advice ended up answering a personal tax
    # question. Runs on follow-ups too — that is exactly how it was reported.
    if _is_personal_matter(question):
        logger.info("[AGENT] personal-matter fast-path (0 tokens): %r", (question or "")[:80])
        yield {"stage": "complete", "status": "done", "type": "answer",
               "payload": _personal_matter_answer(session_id), "message": "Done"}
        return

    gk_method = _general_knowledge_kind(question)

    # "What does indemnify mean?" asked cold is a general question. Asked three
    # turns into a thread about one agreement, the user wants BOTH: how their
    # document uses the term, and what it means generally. Taking the standalone
    # path there answers only half and silently drops the document they were
    # discussing.
    #
    # So when the thread already has a document, the question goes to the normal
    # pipeline — grounded and cited exactly as before, with the prompts
    # untouched — and the general meaning is attached afterwards as a separate
    # field. Keeping it out of the grounded prompt is deliberate: a labelled
    # aside woven INTO cited prose is how a general sentence ends up wearing a
    # citation number it has no right to.
    gk_aside_wanted = False
    gk_deferred_method = ""
    # A named authority never takes the standalone path — the corpus may well
    # discuss it, and a document's own analysis beats a definition. Held back as
    # a fallback for the case where retrieval genuinely finds nothing.
    if gk_method == "gk-named":
        gk_deferred_method, gk_method = gk_method, ""
    # Same treatment, same reason, for a question naming a real counterparty:
    # the corpus probably answers it, and a document's own clause beats a
    # textbook definition. Deferred rather than vetoed outright — if retrieval
    # genuinely finds nothing, the general answer is still the right reply, so
    # this only reorders the two, never removes the fallback.
    if gk_method and _question_names_corpus_entity(question, session_id):
        logger.info("[AGENT] general-knowledge deferred — question names a corpus "
                    "entity, trying retrieval first: %r", (question or "")[:80])
        gk_deferred_method, gk_method = gk_method, ""
    if gk_method:
        _gk_chat_sid = chat_session_id or session_id
        try:
            if wiki.has_established_document_scope(_gk_chat_sid):
                gk_aside_wanted = True
                gk_deferred_method = gk_method
                gk_method = ""
        except Exception as e:
            logger.warning("Could not check document scope for GK aside: %s", e)

    if gk_method:
        logger.info("[AGENT] general-knowledge fast-path (%s): %r",
                    gk_method, (question or "")[:80])
        yield {"stage": "generating", "status": "active",
               "message": "Answering from general legal knowledge…"}
        gk_payload = _general_knowledge_answer(question, gk_method)
        if gk_payload:
            yield {"stage": "complete", "status": "done", "type": "answer",
                   "payload": gk_payload, "message": "Done"}
            return
        # Generation failed or came back empty — fall through to the normal
        # pipeline rather than surfacing an error.
        logger.info("[AGENT] general-knowledge path yielded nothing — using normal pipeline")

    # Decided once, up front: the notice is attached to the terminal answer event
    # below rather than inside a node, so no graph node's behaviour changes.
    advice_seeking = _is_advice_seeking(question)
    if advice_seeking:
        logger.info("[AGENT] advice-seeking phrasing — attaching disclaimer")

    graph = get_query_graph()
    state: QueryState = {
        "question": question,
        "session_id": session_id,
        "chat_session_id": chat_session_id or session_id,
        "target_doc": target_doc or "",
        "is_followup": bool(is_followup),
        "exclude_cached_answers": bool(exclude_cached_answers),
        "collection_id": int(collection_id) if collection_id else None,
        "conversation_context": "",
    }
    for chunk in graph.stream(state, stream_mode="custom"):
        if chunk.get("type") == "answer" and isinstance(chunk.get("payload"), dict):
            # The aside path above assumes the document has something to say
            # about the term. When it does not, the user is left with a refusal
            # ("not addressed in the provided context") beside a margin note
            # holding the actual answer — for a question the general path was
            # willing to answer outright. Promoting it here costs nothing a
            # grounded answer would have provided, because it only ever fires
            # once that grounded answer has already come back empty.
            #
            # "Nothing to say" also arrives as an EMPTY answer, not only as an
            # explicit refusal. A corpus that never discusses the named authority
            # gives the answer LLM no material to write from and no absent-topic
            # to declare, so it returns a blank body that not_covered does not
            # flag. Measured on the 46-document production-representative corpus:
            # "What is the Delaware Uniform Trade Secrets Act (DUTSA)…" produced
            # a payload whose entire content was the scope disclosure — no answer
            # at all — while the general path was ready to answer it outright.
            # Bracketed disclosures are discounted before the emptiness test
            # precisely because they are what a blank answer is left holding.
            _body = _RX_BRACKET_NOTE.sub("", chunk["payload"].get("answer") or "").strip()
            if gk_deferred_method and (chunk["payload"].get("not_covered") or not _body):
                promoted = _general_knowledge_answer(question, gk_deferred_method)
                if promoted:
                    logger.info("[AGENT] document scope had nothing (%s) — promoting "
                                "general-knowledge answer (%s)",
                                "refused" if chunk["payload"].get("not_covered") else "empty",
                                gk_deferred_method)
                    chunk["payload"] = promoted
                    gk_aside_wanted = False
            if advice_seeking:
                chunk["payload"]["advice_notice"] = _ADVICE_NOTICE
            # Answer handoff (§ Phase 3.5b) — names the shipped surface that
            # owns this answer's depth, when one has something to show.
            # Attached to the finished payload, never merged into the answer
            # text, for the same reason the general-knowledge aside is not.
            from services import handoffs as _handoffs
            _ho = _handoffs.suggest(question, chunk["payload"], session_id)
            if _ho:
                chunk["payload"]["handoff"] = _ho
            # Attached to the finished payload, never merged into the answer
            # text — the separation between cited and general content is
            # structural, not a formatting convention the model has to honour.
            if gk_aside_wanted:
                _aside = _general_knowledge_aside(question)
                if _aside:
                    chunk["payload"]["general_knowledge_note"] = _aside
                    logger.info("[AGENT] attached general-knowledge aside (%d chars)",
                                len(_aside))
        yield chunk
