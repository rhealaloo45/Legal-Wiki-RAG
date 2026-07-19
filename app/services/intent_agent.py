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

logger = logging.getLogger(__name__)

VALID_INTENTS = ["factual", "risk_assessment", "comparison", "obligation", "drafting"]

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
    r'[^.?!;]+\band\b',
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
    r'\b(?:board|shareholders?|members?|majority|unanimous|special\s+majority|'
    r'statutory|regulatory|prior\s+written|requisite)\b\s+\w*\s*\b(?:approval|approve|consent)\b'
    r'|\b(?:approval|approve|consent)\s+(?:of|by|from)\s+(?:the\s+)?(?:board|shareholders?|members?)\b',
    re.IGNORECASE,
)


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
    "- factual: extract or explain what a document says (clauses, definitions, parties, summaries).\n"
    "- risk_assessment: judge risk, recommend go/no-go, flag red flags, advise whether to sign.\n"
    "- comparison: compare two or more documents/clauses side by side, find differences.\n"
    "- obligation: list duties, obligations, deadlines, or compliance requirements.\n"
    "- drafting: write, redline, or propose new/alternative clause language.\n\n"
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
    if _RX_COMPARISON.search(q):
        return {"intent": "comparison", "confidence": 0.9, "method": "regex"}
    if _RX_BETWEEN.search(q) and not _RX_BETWEEN_EXCLUDE.search(q) \
            and not _RX_BETWEEN_PARTIES.search(q) \
            and not _RX_BETWEEN_ALLOCATION.search(q):
        return {"intent": "comparison", "confidence": 0.9, "method": "regex"}
    if _RX_RISK.search(q) and not _is_governance_approval_only(q):
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

    _emit({"stage": "disambiguation", "status": "active", "message": "Checking document scope…"})
    try:
        result = wiki.classify_query(state["question"], state["session_id"])
    except Exception as e:
        logger.error("Disambiguation check failed: %s", e)
        result = {"needs_disambiguation": False}

    if result.get("needs_disambiguation"):
        docs = result.get("documents", [])
        clean = [re.sub(r'^[a-f0-9-]{36}_', '', d) for d in docs]
        msg = ("I'd like to help, but could you specify which document you're referring to? "
               "Please select one below, or upload a new document.")
        data = {"message": msg, "documents": clean, "raw_documents": docs}
        _emit({"stage": "disambiguation", "status": "done",
               "message": "Needs document selection",
               "type": "disambiguation", "payload": data})
        return {"needs_disambiguation": True, "disambiguation_data": data}

    return {"needs_disambiguation": False}


def check_clarification_node(state: QueryState) -> dict:
    if state.get("is_followup") or wiki._question_names_a_document(state["question"], []) \
            or not config.ENABLE_CLARIFICATION:
        conv = wiki.build_conversation_context(state["session_id"])
        return {"needs_clarification": False, "conversation_context": conv}

    _emit({"stage": "clarification", "status": "active", "message": "Checking for ambiguity…"})
    conv = wiki.build_conversation_context(state["session_id"])
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
    logger.info("[AGENT] scope=%s family=%s broad=%s method=%s",
                decision.get("scope"), decision.get("target_family"),
                decision.get("is_broad"), decision.get("method"))
    return {"scope_decision": decision}


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
    res = wiki.get_context(
        state["question"], state["session_id"],
        target_doc=state.get("target_doc", ""), retrieval_hints=hints,
        exclude_cached_answers=state.get("exclude_cached_answers", False),
        doc_family=_fam, force_broad=_force_broad, force_docs=_force_docs,
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
    }


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

    # The question named a specific counterparty the pipeline could not pin to
    # one document (the name spans too many documents to disambiguate), so scope
    # fell through to a broad corpus search. The answer may be sourced from a
    # same-type sibling rather than the exact document the user meant — warn.
    _scope_warning = ""
    _unresolved = _scope.get("unresolved_party") if _scope.get("method") == "default" else ""
    if _unresolved:
        _scope_warning = (
            f"The question named \"{_unresolved}\" but that party appears in several "
            f"documents, so no single document could be confirmed as the one you meant. "
            f"The answer below was drawn from a broad search and may reflect a "
            f"different document of the same type — check the References section, and "
            f"if it's the wrong one, name the document by its number (e.g. \"NDA 7\") "
            f"or its distinctive counterparty."
        )

    try:
        wr = wiki.generate_answer(
            state["question"], state.get("wiki_context", ""),
            state.get("selected_titles", []), state["session_id"],
            meta.get("bm25_count", 0), meta.get("page_selection_usage", {}),
            state.get("conversation_context", ""), intent=intent,
            unconfirmed_doc_reference=state.get("unconfirmed_doc_reference", False),
            scope_note=_scope_note, scope_warning=_scope_warning,
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
    re.compile(r'\b(?:Section|Clause|Article|Exhibit|Schedule|Appendix)\s+[\dA-Za-z.]+\b', re.I),  # doc refs
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


def _extract_claims(text: str) -> list[str]:
    """Pull out checkable factual atoms (amounts, dates, durations, doc
    references, proper nouns) from an answer, deduplicated and stripped of
    generic boilerplate phrases that aren't really factual claims."""
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
# judge. escalate_low stays low ON PURPOSE — a low deterministic score on these
# paraphrase-heavy intents is often just reworded-but-real quotes, so surfacing
# it raw would mislead (same trap as drafting grounding); those still escalate
# to get a truer number.
_GROUNDING_THRESHOLDS = {
    "default":         {"min_claims": 3, "escalate_low": 50, "escalate_high": 85},
    "comparison":      {"min_claims": 2, "escalate_low": 35, "escalate_high": 85},
    "obligation":      {"min_claims": 2, "escalate_low": 35, "escalate_high": 85},
    "risk_assessment": {"min_claims": 2, "escalate_low": 35, "escalate_high": 85},
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


def run_query_stream(question: str, session_id: str, target_doc: str = "",
                     is_followup: bool = False, exclude_cached_answers: bool = False,
                     chat_session_id: str = ""):
    """Run the query graph and yield stage event dicts in real time.

    Each yielded dict is a custom stage event emitted by a node. The terminal
    'complete' / 'disambiguation' / 'clarification' event carries the payload
    the frontend renders. app.py wraps each dict as a Server-Sent Event.

    exclude_cached_answers: QA/testing option — see wiki.get_context().
    """
    graph = get_query_graph()
    state: QueryState = {
        "question": question,
        "session_id": session_id,
        "chat_session_id": chat_session_id or session_id,
        "target_doc": target_doc or "",
        "is_followup": bool(is_followup),
        "exclude_cached_answers": bool(exclude_cached_answers),
        "conversation_context": "",
    }
    for chunk in graph.stream(state, stream_mode="custom"):
        yield chunk
