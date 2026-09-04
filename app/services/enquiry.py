"""The Enquiry Agent: what is being asked, and which document it is about.

Everything else in this system moved the other way — out of the prompt and into
code — and that was right for output discipline: voice, quoting, identifiers.
Those are rules, and a rule expressed as a regex cannot be talked out of.

Resolving a REFERENT is not that kind of problem. "In this agreement", "the
second one", "what about termination" — deciding what those point at is
language understanding, and the regex stack that does it today has been
patched three times in one session for three unrelated reasons. There will be a
fourth. This is the one place where the model has the advantage.

The discipline that makes it safe is that the model never returns a document.
It returns a DESCRIPTION of one, in the user's own terms, and the existing
deterministic resolvers decide whether that description names anything real. A
referent that does not resolve is discarded and the pipeline behaves exactly as
it did before. The model contributes understanding; code contributes truth.

Cost is neutral by construction. classify_intent already makes a fast-model
call when its regexes do not fire; this widens what that one call returns
rather than adding a second.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

ENQUIRY_PROMPT = """\
You read one turn of a conversation between a lawyer and a document assistant, \
and report what is being asked and which document it concerns. You do not answer \
the question and you do not see the documents.

CONVERSATION SO FAR (may be empty):
{conversation}

THIS TURN: {question}

Return ONLY a JSON object:
{{
  "intent": one of factual | comparison | risk_assessment | obligation | drafting,
  "confidence": 0.0-1.0,
  "is_followup": true if this turn depends on the conversation to be understood,
  "referent": "" or a short description of the document this turn is about,
  "referent_basis": "stated" | "carried" | "none"
}}

Rules for `referent`, which matter more than the intent:

- If THIS TURN names its own document, describe it as the turn does: "the Master \
Services Agreement dated 12 August 2020", "the NDA with Amberline".
- If this turn points BACK ("in this agreement", "the second one", "what about \
termination", "and if they don't cure") then the referent is whatever document \
the conversation was already about. Describe THAT document, using the words the \
earlier turns used. Set referent_basis to "carried".
- If you cannot tell which document is meant, return "" and basis "none". An \
empty referent is a correct answer and costs nothing; a guessed one sends the \
reader to the wrong contract.
- Never invent a party name, a date or a document type that appears neither in \
this turn nor in the conversation above. Describe, do not embellish.
- The referent is a description, not a filename. Do not guess at file names."""

VALID_INTENTS = ("factual", "comparison", "risk_assessment", "obligation", "drafting")
VALID_BASIS = ("stated", "carried", "none")

# A referent long enough to be a sentence is the model narrating rather than
# naming, and a referent of one word cannot identify a document.
_REFERENT_MIN = 4
_REFERENT_MAX = 160

# Phrases that mean the model described the conversation instead of a document.
_RX_NON_REFERENT = re.compile(
    r"^(?:the (?:previous|last|same|above) (?:document|answer|turn)|"
    r"unknown|unclear|not (?:specified|stated|clear)|n/?a|none)\b",
    re.IGNORECASE)


def parse(raw: str) -> dict:
    """The model's reply as a validated enquiry, or an empty one.

    Every field is checked rather than trusted. An unparseable or malformed
    reply returns the same shape with nothing in it, which the caller treats as
    "the agent said nothing" — the pipeline then behaves exactly as it did
    before this existed.
    """
    from services import wiki as _wiki
    out = {"intent": "", "confidence": 0.0, "is_followup": False,
           "referent": "", "referent_basis": "none"}
    parsed = None
    try:
        parsed = _wiki._parse_json_safe(raw)
    except Exception as e:
        logger.error("enquiry: could not parse reply: %s", e)
    if not isinstance(parsed, dict):
        return out

    intent = str(parsed.get("intent", "")).strip().lower()
    if intent in VALID_INTENTS:
        out["intent"] = intent
    try:
        out["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.6))))
    except (TypeError, ValueError):
        out["confidence"] = 0.6
    out["is_followup"] = bool(parsed.get("is_followup"))

    basis = str(parsed.get("referent_basis", "")).strip().lower()
    out["referent_basis"] = basis if basis in VALID_BASIS else "none"

    ref = re.sub(r"\s+", " ", str(parsed.get("referent", "") or "")).strip(' "\'')
    if (_REFERENT_MIN <= len(ref) <= _REFERENT_MAX
            and not _RX_NON_REFERENT.match(ref)):
        out["referent"] = ref
    else:
        out["referent_basis"] = "none"
    return out


# The intent classifier runs on 150 completion tokens because its output is two
# fields. This prompt asks the model to read a conversation and decide what a
# turn points at, and on the Azure GPT-5.x deployments the budget also covers
# hidden reasoning - measured: at 150 the call came back finish_reason=length
# with 150 completion tokens spent and an EMPTY string, so the agent silently
# never fired. Sized for the reasoning, not for the two lines of JSON.
_ENQUIRY_MAX_TOKENS = 900


def ask(question: str, conversation: str = "") -> dict:
    """One fast-model call. Returns a validated enquiry, never raises."""
    from services import llm
    try:
        prompt = ENQUIRY_PROMPT.format(
            question=(question or "")[:1200],
            conversation=(conversation or "(no earlier turns)")[:3000])
        raw, usage = llm.ask(prompt, fast=True, max_tokens=_ENQUIRY_MAX_TOKENS)
        if usage.get("finish_reason") == "length" and not (raw or "").strip():
            logger.warning("enquiry: reply truncated with no visible output "
                           "(%s completion tokens) - treating as no answer",
                           usage.get("completion_tokens"))
            return parse("")
    except Exception as e:
        logger.error("enquiry call failed: %s", e)
        return parse("")
    return parse(raw)


def resolve_referent(referent: str, session_id: str, chat_session_id: str = "") -> list:
    """The documents a referent description actually names, or nothing.

    This is the half that keeps the agent honest. The description goes through
    the same deterministic resolver every other path uses, and only a
    description that pins a small, real set is accepted. A referent that
    resolves to the whole corpus, to nothing, or to a broad family has told us
    nothing we did not already know, and is discarded rather than acted on.
    """
    from services import wiki as _wiki
    if not referent:
        return []
    try:
        sc = _wiki.resolve_scope(referent, session_id,
                                 chat_session_id=chat_session_id or session_id)
    except Exception as e:
        logger.error("enquiry: referent resolution failed for %r: %s", referent, e)
        return []
    method = (sc.get("method") or "").lower()
    docs = [str(d) for d in (sc.get("target_docs") or []) if d]
    if not docs or len(docs) > 4:
        return []
    # A referent that only resolved by inheriting the conversation adds nothing:
    # carryover already ran, and accepting it here would let the agent launder
    # the previous turn's scope as if it were a fresh finding.
    if method.startswith(("carryover", "default", "corpus", "error")):
        return []
    logger.info("Enquiry referent %r resolved to %d document(s) by %s",
                referent[:60], len(docs), method)
    return docs
