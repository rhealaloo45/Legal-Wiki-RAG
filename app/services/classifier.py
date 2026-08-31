"""
Ingestion stage 02 — document-type + jurisdiction classification
(target architecture § 01 stage 02, § 00.1 Document Families).

Runs upstream of the length fork, so the family is known before stage 03's
synthesis call is built. Which family applies decides which schema gets
extracted from the whole document — so a wrong call here is not one wrong
field, it is the wrong set of fields entirely. That is why a low-confidence
classification is arguably the highest-stakes Review Queue item there is,
and why it is treated as one.

The mechanism the doc specifies, and the reason for it:

  **Content-primary, folder-assisted.** One code path, not two. Content
  classification runs on every document whether or not a folder exists —
  a bulk library upload and an ad-hoc chat upload go through the same call,
  because "we have a folder" must never become a reason to skip reading the
  document.

  **Folder raises confidence on agreement; it never lowers scrutiny on
  disagreement.** A document sitting in the wrong folder — misfiled, legacy
  structure, plain human error — routes to the Review Queue as a mismatch.
  Folder is never ground truth over what the content actually says. That
  asymmetry is the whole point: the case the boost is worth having is
  cheap, and the case it would paper over is the one that matters.
"""
from __future__ import annotations

import logging

import config
from services import extraction_validation as ev
from services import llm, schema_registry

logger = logging.getLogger(__name__)

# How much of the document the classifier reads. A document's type is almost
# always evident from its opening — title, parties block, recitals — and
# reading the whole thing to decide would multiply the cost of every ingest
# for no accuracy gain. The tail sample catches execution/signature blocks,
# which is where a term sheet's binding language often actually lives.
_HEAD_CHARS = 5000
_TAIL_CHARS = 1500
_MAX_TOKENS = 700

# Agreement with the folder adds this much confidence. The ceiling applies to
# every classification, not just boosted ones: a model reporting 1.0 is making
# a claim about its own calibration that the doc's own hardening notes say is
# not reliable, so nothing downstream should read "certain" here. Capping only
# the boosted path would leave the model's own self-reported 1.0 as the single
# way to reach certainty, which is precisely backwards.
_FOLDER_AGREEMENT_BOOST = 0.1
_MAX_CONFIDENCE = 0.98

# At or below this, the document routes to the Review Queue regardless of
# folder agreement.
REVIEW_THRESHOLD = 0.7


def _excerpt(text: str) -> str:
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text
    return (text[:_HEAD_CHARS]
            + "\n\n[... middle of document omitted ...]\n\n"
            + text[-_TAIL_CHARS:])


def _build_prompt(text: str) -> str:
    families = schema_registry.all_families()
    lines = []
    for fam in families:
        examples = ", ".join(fam.doc_types) if fam.doc_types else "anything else"
        lines.append(f'- "{fam.key}" — {fam.label}. {fam.description} Examples: {examples}')
    family_block = "\n".join(lines)

    return f"""You are classifying a legal document so the correct extraction schema can be applied to it.

Choose exactly ONE family from this list:
{family_block}

Rules that matter:
- Decide from the CONTENT ONLY. Do not guess from a filename or a folder.
- A document that reads like a court filing is "litigation" even if it is a
  draft or a training copy — what matters is the document's form, not whether
  it was ever filed.
- A document proposing terms that are expressly not yet binding is
  "term_sheet", not "contract", even when it looks like an agreement.
- If the document does not clearly fit any of the five, answer "generic".
  A confident wrong family causes the wrong fields to be extracted from the
  entire document, so "generic" with honest uncertainty is strictly better
  than a forced fit.

Also identify:
- doc_type: the specific type in the document's own words (e.g. "Non-Disclosure
  Agreement", "Plaint with Application for Interim Relief"). Null if unclear.
- jurisdiction: the governing/forum jurisdiction if the document states one
  (e.g. "India", "England and Wales", "Delaware"). Null if not stated — do NOT
  infer it from language, currency or party names.
- confidence: 0.0-1.0, how certain you are of the FAMILY.
    1.0 = the document names its own type unambiguously
    0.8 = clearly this family from its structure and content
    0.5 = plausible but genuinely ambiguous
    0.0 = no basis to decide
- reasoning: one short sentence, citing what in the text decided it.

Return ONLY this JSON object, no prose around it:
{{"family": "...", "doc_type": "...", "jurisdiction": "...", "confidence": 0.0, "reasoning": "..."}}

DOCUMENT:
---
{_excerpt(text)}
---"""


def classify_document(text: str, source_doc: str | None = None,
                      allow_llm: bool = True) -> dict:
    """Classify one document. Returns a dict ready for the documents registry.

    Never raises. A classifier that can abort an ingest is worse than one
    that returns low confidence — the fallback is the generic family, which
    extracts less but extracts nothing wrong, and the Review Queue flag makes
    the degradation visible rather than silent.
    """
    folder_family, folder_hint = schema_registry.classify_by_folder(source_doc)

    result = {
        "doc_family": schema_registry.GENERIC_FAMILY,
        "doc_type": None,
        "jurisdiction": None,
        "family_confidence": 0.0,
        "family_method": "none",
        "folder_hint": folder_hint,
        "folder_family": folder_family,
        "folder_agreement": None,
        "flagged": True,
        "flag_reason": "not classified",
        "reasoning": None,
    }

    if not (text or "").strip():
        result["flag_reason"] = "document has no extractable text"
        return result

    if not allow_llm:
        # Folder-only mode exists for cost-free dry runs and tests. It is
        # deliberately capped low and always flagged: a folder name is a
        # filing convention, not evidence about content, and must never
        # present itself as a real classification.
        if folder_family:
            result.update({
                "doc_family": folder_family, "family_confidence": 0.4,
                "family_method": "folder_only", "folder_agreement": None,
                "flagged": True,
                "flag_reason": "classified from folder name only, content not read",
            })
        return result

    try:
        raw, usage = llm.ask(_build_prompt(text), pipeline="classify",
                             max_tokens=_MAX_TOKENS, fast=True)
    except Exception as err:
        logger.warning("Doc-type classification call failed for %s: %s",
                       source_doc, err)
        result["flag_reason"] = f"classification call failed: {err}"
        return result

    from services.wiki import _parse_json_safe
    parsed = _parse_json_safe(raw)
    if not isinstance(parsed, dict):
        logger.warning("Doc-type classifier returned unparseable output for %s",
                       source_doc)
        result["flag_reason"] = "classifier returned unparseable output"
        return result

    valid_keys = tuple(schema_registry.family_keys())
    report = ev.validate_payload(parsed, {
        "family": valid_keys,
        "doc_type": "text",
        "jurisdiction": "text",
        "reasoning": "text",
    }, base_confidence=1.0)

    family = report.values.get("family")
    confidence = min(_MAX_CONFIDENCE, ev.coerce_confidence(parsed.get("confidence")))

    if not family:
        # The model named a family that isn't in the registry. Falling back
        # to generic rather than to its raw string keeps an unknown value out
        # of doc_family, which downstream code treats as a real key.
        logger.info("Classifier gave unknown family %r for %s — using generic",
                    parsed.get("family"), source_doc)
        result.update({
            "family_confidence": 0.0, "family_method": "llm",
            "flag_reason": f"unrecognised family {parsed.get('family')!r}",
            "reasoning": report.values.get("reasoning"),
        })
        return result

    method = "content"
    agreement = None
    if folder_family:
        agreement = (folder_family == family)
        if agreement:
            confidence = min(_MAX_CONFIDENCE,
                             confidence + _FOLDER_AGREEMENT_BOOST)
            method = "content+folder"
        else:
            method = "content_folder_mismatch"

    flagged = False
    reason = None
    if agreement is False:
        # Not a confidence penalty — the content classification stands on its
        # own merits. But a document whose content and filing disagree is
        # exactly the misfiled case worth a human glance, so it is flagged
        # regardless of how confident the content call was.
        flagged = True
        reason = (f"content says '{family}' but folder suggests "
                  f"'{folder_family}' (hint: {folder_hint!r})")
    elif confidence <= REVIEW_THRESHOLD:
        flagged = True
        reason = f"low classification confidence ({confidence:.2f})"
    elif family == schema_registry.GENERIC_FAMILY:
        flagged = True
        reason = "fell through to the generic fallback — no family matched"

    result.update({
        "doc_family": family,
        "doc_type": report.values.get("doc_type"),
        "jurisdiction": report.values.get("jurisdiction"),
        "family_confidence": round(confidence, 3),
        "family_method": method,
        "folder_agreement": agreement,
        "flagged": flagged,
        "flag_reason": reason,
        "reasoning": report.values.get("reasoning"),
        "usage": usage,
    })
    logger.info(
        "Classified %s -> %s (%s, conf=%.2f%s)",
        source_doc, family, method, confidence,
        f", FLAGGED: {reason}" if flagged else "",
    )
    return result
