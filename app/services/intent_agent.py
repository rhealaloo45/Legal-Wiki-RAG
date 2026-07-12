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
    if _RX_BETWEEN.search(q) and not _RX_BETWEEN_EXCLUDE.search(q):
        return {"intent": "comparison", "confidence": 0.9, "method": "regex"}
    if _RX_RISK.search(q):
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
    session_id: str
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
    if state.get("target_doc") or wiki._question_names_a_document(state["question"], []):
        logger.info("[AGENT] disambiguation skipped (doc named)")
        return {"needs_disambiguation": False}

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


def retrieve_context_node(state: QueryState) -> dict:
    logger.info("[AGENT] retrieve_context_node: intent=%s", state.get("intent"))
    _emit({"stage": "retrieving", "status": "active", "message": "Retrieving relevant pages…"})
    conv = state.get("conversation_context") or wiki.build_conversation_context(state["session_id"])
    hints = get_query_strategy(state["intent"])
    res = wiki.get_context(
        state["question"], state["session_id"],
        target_doc=state.get("target_doc", ""), retrieval_hints=hints,
        exclude_cached_answers=state.get("exclude_cached_answers", False),
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
    try:
        wr = wiki.generate_answer(
            state["question"], state.get("wiki_context", ""),
            state.get("selected_titles", []), state["session_id"],
            meta.get("bm25_count", 0), meta.get("page_selection_usage", {}),
            state.get("conversation_context", ""), intent=intent,
        )
    except Exception as e:
        logger.error("Answer generation failed (%s): %s", type(e).__name__, e)
        wr = {"answer": f"Wiki error: {type(e).__name__}: {e}",
              "pages_used": [], "files_used": [], "confidence_score": 0}

    wr["intent"] = intent
    wr["intent_label"] = label
    wr["intent_confidence"] = state.get("intent_confidence", 0.0)
    wr["intent_method"] = state.get("intent_method", "")
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

Respond with ONLY valid JSON, no other text:
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
        grounding = _check_grounding(
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
    g.add_node("retrieve", retrieve_context_node)
    g.add_node("generate", generate_answer_node)
    g.add_node("validate", validate_response_node)

    g.add_edge(START, "classify_intent")
    g.add_edge("classify_intent", "disambiguation")
    g.add_conditional_edges("disambiguation", _route_after_disambiguation,
                            {"stop": END, "continue": "clarification"})
    g.add_conditional_edges("clarification", _route_after_clarification,
                            {"stop": END, "continue": "retrieve"})
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
                     is_followup: bool = False, exclude_cached_answers: bool = False):
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
        "target_doc": target_doc or "",
        "is_followup": bool(is_followup),
        "exclude_cached_answers": bool(exclude_cached_answers),
        "conversation_context": "",
    }
    for chunk in graph.stream(state, stream_mode="custom"):
        yield chunk
