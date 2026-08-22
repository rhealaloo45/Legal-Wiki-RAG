"""
Stage 03 — registry-driven synthesis supplement
(target architecture § 01 stage 03 "Adaptive synthesis call, extended").

The synthesis call keeps its existing shape: one call, structured output
alongside the wiki prose, no extra round trips. What changes is that the
schema it asks for is pulled from the schema registry by family rather than
being a fixed contract schema, and that it now also returns citations,
structural anchors, hypothetical questions, tables and figures.

This module builds only the *supplement* — the family-specific block that
gets appended to the existing prompt templates. It is deliberately additive:
the long, carefully-tuned page-title and quote-verification instructions in
those templates are load-bearing and were not rewritten to accommodate this.

One decision worth stating: clause extraction is requested only for families
whose registry entry sets `clause_extraction=True` (Family 1 and 2). Asking
a judgment or a power of attorney for "clauses" does not return an empty
list — it returns confident fabrications, because the model will find
something to call a clause. Not asking is the only reliable way to not get
them.
"""
from __future__ import annotations

import json

from services import schema_registry
from services.schema_registry import Family


def _fields_block(family: Family) -> str:
    lines = [f'    "{name}": "{desc}"' for name, desc in family.extraction_fields.items()]
    return ",\n".join(lines)


def build_supplement(family_key: str | None, segment_mode: bool = False) -> str:
    """The family-specific instruction + output-schema block.

    `segment_mode` tunes the wording for the per-segment detail pass, where
    the model sees only part of the document and must not invent a
    document-level answer from a fragment of it.
    """
    fam = schema_registry.get(family_key)
    lo, hi = fam.page_target
    scope = "this segment" if segment_mode else "this document"

    parts: list[str] = []

    parts.append(f"""
DOCUMENT FAMILY: {fam.label} ({fam.key})
{fam.description}

EXTRACTION EMPHASIS FOR THIS FAMILY: {fam.emphasis}

Extract {lo}-{hi} pages. That range is set for this family specifically — do \
not pad to reach it, and do not compress past it. Fewer, accurate pages beat \
more pages built from inference.""")

    parts.append(f"""
FAMILY METADATA — additionally return a "family_metadata" object holding \
these fields for {scope}. Use null for anything the document does not state. \
Never infer a value from context, party names, or what is typical for this \
kind of document — a null is a correct answer and an inferred value is not:
{{
{_fields_block(fam)}
}}""")

    if fam.clause_extraction:
        parts.append("""
CLAUSES: keep returning the "clauses" array described above — this family is \
clause-bearing.""")
    else:
        parts.append("""
CLAUSES: return "clauses": [] for this document. This family is NOT \
clause-bearing — it does not contain contractual clauses, and labelling its \
provisions as clauses would misrepresent what they are. Do not populate this \
array even if some passage superficially resembles a clause.""")

    parts.append("""
CITATIONS — return a "citations" array of every statute, rule, regulation or \
case referenced in the text:
[
  {
    "citation_text": "Exact reference as written, e.g. 'Section 27 of the Trade Marks Act, 1999'",
    "authority_type": "statute | case | rule | regulation | other",
    "normalized_form": "Canonical short form, e.g. 'Trade Marks Act 1999 s.27'",
    "page": 3,
    "confidence": 1.0
  }
]
Include only references actually present in the text. Do not add the \
well-known citation you would expect a document like this to contain.""")

    parts.append("""
STRUCTURAL ANCHORS — return a "structural_anchors" array naming the section \
or paragraph numbering this content sits under, so a later question about \
"paragraph 14" can be answered:
[
  {"anchor_label": "14", "anchor_kind": "paragraph | section | article | clause | schedule | recital", "heading_text": "Heading as written", "page": 3}
]
Only report numbering the document actually shows. Never renumber, never \
invent a label for an unnumbered section.""")

    parts.append("""
HYPOTHETICAL QUESTIONS — for each page you produce, return 2-4 questions a \
lawyer could answer using ONLY that page's content, in a "hypothetical_questions" \
object keyed by page title:
{"Page Title": ["Question 1?", "Question 2?"]}
Write them as a lawyer would actually ask, in their words rather than the \
document's. A question the page cannot answer is worse than no question — it \
will surface this page for a query it has nothing to say about.""")

    parts.append("""
DOCUMENT REFERENCES — return a "document_references" array for every OTHER \
document this text refers to ("as defined in the Master Services Agreement \
dated 3 March 2024", "this Amendment No. 2 to the Shareholders' Agreement"):
[
  {
    "reference_text": "The exact sentence or phrase making the reference, verbatim",
    "referenced_document": "The other document's name/title as this text gives it",
    "relationship": "amends | superseded_by | ancillary_to | references",
    "confidence": 1.0
  }
]
Use "amends" only where this document expressly changes the other one's \
terms, and "superseded_by" only where this document says it has been replaced. \
Those two assert which text currently governs, so state them only from \
explicit language — never infer an amendment from two documents merely \
covering the same subject. Anything weaker is "references". Return an empty \
array if the text refers to no other document.""")

    parts.append("""
TABLES AND FIGURES — if the text contains a table or describes a chart, \
diagram or image, return it in "tables" / "figures" rather than flattening it \
into page prose:
"tables":  [{"caption": "...", "columns": ["Col A", "Col B"], "rows": [["a1","b1"]], "page": 3, "confidence": 1.0}]
"figures": [{"figure_kind": "chart | diagram | image | signature | seal", "description": "What it shows", "page": 3, "confidence": 1.0}]
Return empty arrays if there are none. A table reconstructed from a guess at \
its layout is worse than no table — if the row/column structure is not \
recoverable from the text, describe it in "figures" instead.""")

    return "\n".join(parts)


def family_output_keys() -> tuple[str, ...]:
    """The keys stage 03 adds to the synthesis response, so callers can strip
    them out before the payload reaches the existing page-merge logic (which
    ignores unknown keys, but is easier to reason about when they're gone)."""
    return ("family_metadata", "citations", "structural_anchors",
            "hypothetical_questions", "tables", "figures",
            "document_references")


def metadata_spec(family_key: str | None) -> dict[str, str]:
    """Validation spec for a family's metadata block.

    Field *types* are inferred from field names rather than declared in the
    registry: the registry describes what to extract in the model's terms,
    and duplicating a type vocabulary there would mean two places to update
    and one of them silently going stale.
    """
    fam = schema_registry.get(family_key)
    spec: dict[str, str] = {}
    for name in fam.extraction_fields:
        if name.endswith("_date") or name in ("decided_date", "opinion_date"):
            spec[name] = "date"
        elif name in ("liability_cap",):
            spec[name] = "currency"
        elif name in ("notice_period", "term_length"):
            spec[name] = "duration"
        elif name in ("parties", "plaintiffs", "defendants"):
            # Party fields get the redaction-aware list coercer — a
            # description-only entry is reduced to the document's own defined
            # term rather than shown as an unreadable clause fragment.
            spec[name] = "party_list"
        elif name in ("matters_opined", "assumptions", "qualifications",
                      "statutes_cited", "conditions_precedent", "key_terms"):
            spec[name] = "list"
        elif name == "binding_status":
            spec[name] = "text"
        else:
            spec[name] = "text"
    return spec
