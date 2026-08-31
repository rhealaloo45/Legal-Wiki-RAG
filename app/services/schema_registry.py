"""
Schema registry — five document families plus a generic fallback
(target architecture § 00.1 "Document Families & Schema Registry").

The problem this fixes: the extraction schema the ingest call asks for today
(clauses / obligations / entities / relationships) is a *contract* schema
wearing a general-purpose name. A judgment has no liability cap; it has a
court, a case number and a disposition. Forcing every family through one
schema either extracts the wrong fields or extracts nothing.

So the schema is looked up per document type rather than hardcoded. That
lookup is what lives here. Everything downstream — stage 03's synthesis
prompt, the per-family metadata list, which typed table a row lands in, the
page-count target — reads from these definitions rather than carrying its
own copy.

Config-driven, two layers:
  1. The built-in definitions below, derived from this org's own folder list.
  2. An optional org-override JSON file (SCHEMA_REGISTRY_OVERRIDES, default
     app/data/schema_overrides.json), merged in at load. This is the seam the
     doc names for "a client's taxonomy quirks" — a second organization can
     add folder hints or extra fields without editing code.

The family set is explicitly a first cut. It was derived from 24 folders in
one organization; whether it survives a trust deed or a regulatory consent
order is untested. The generic fallback is not a dumping ground — what lands
in it is the live signal for whether a sixth family is warranted.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace

logger = logging.getLogger(__name__)

GENERIC_FAMILY = "generic"


@dataclass(frozen=True)
class Family:
    """One family's extraction contract.

    `extraction_fields` maps field name -> the instruction the synthesis call
    sees for it. Keeping the wording here rather than in the prompt template
    means adding a family never means editing a prompt string.
    """
    key: str
    label: str
    description: str
    # Folder-name fragments that suggest this family. A hint is a *boost* on
    # agreement with content classification, never a substitute for it and
    # never a reason to lower scrutiny on disagreement (§ Document Families).
    folder_hints: tuple[str, ...]
    # Where this family's typed row lands. None for generic — a document we
    # couldn't classify has no typed home, and inventing one would be worse
    # than leaving the row in `documents` alone.
    typed_table: str | None
    doc_types: tuple[str, ...]
    extraction_fields: dict[str, str]
    metadata_fields: tuple[str, ...]
    # Metadata fields that require individual Review Queue sign-off rather
    # than bulk accept. Same stakes split already built for clauses.
    high_stakes_metadata: frozenset[str]
    page_target: tuple[int, int]
    emphasis: str
    clause_extraction: bool = True
    extra: dict = field(default_factory=dict)


_SHARED_METADATA = ("parties", "jurisdiction", "doc_type", "doc_family")


FAMILY_1_CONTRACT = Family(
    key="contract",
    label="Bilateral commercial contracts",
    description=(
        "Two parties, mutual promises. Well served by the existing "
        "clauses/obligations schema — this is the family that schema was "
        "quietly written for all along."
    ),
    folder_hints=(
        "consulting", "dpa", "data processing", "employment", "escrow",
        "ip assignment", "joint venture", "jva", "lease", "license",
        "licence", "loan", "nda", "non-disclosure", "service agreement",
        "sla", "spa", "share purchase", "shareholder", "sha", "subscription",
        "supply", "agreement",
    ),
    typed_table="contracts",
    doc_types=(
        "NDA", "Service Agreement", "Joint Venture Agreement",
        "Shareholder Agreement", "Employment Agreement", "Lease",
        "Licence Agreement", "Loan Agreement", "Supply Agreement",
        "Share Purchase Agreement", "Data Processing Agreement",
    ),
    extraction_fields={
        "parties": "Every contracting party, verbatim as named in the document",
        "governing_law": "Jurisdiction whose law governs (e.g. 'English law'), or null",
        "effective_date": "Date the agreement takes effect, or null",
        "expiry_date": "Date the agreement expires or terminates, or null",
        "term_length": "Stated duration of the agreement, or null",
        "renewal_terms": "Auto-renewal or extension mechanics, or null",
        "termination": "How and on what notice either party may terminate, or null",
        "liability_cap": "Any monetary or formula cap on liability, or null",
        "ip_ownership": "Who owns IP created or licensed under the agreement, or null",
        "payment_terms": "Fees, rates, invoicing and payment timing, or null",
        "notice_period": "Contractual notice period(s), or null",
        "confidentiality": "Confidentiality obligation and its duration, or null",
    },
    metadata_fields=_SHARED_METADATA + (
        "governing_law", "effective_date", "termination_notice", "liability_cap",
        "ip_ownership", "auto_renewal", "notice_period", "payment_terms",
    ),
    high_stakes_metadata=frozenset({
        "governing_law", "liability_cap", "termination_notice", "ip_ownership",
    }),
    page_target=(10, 30),
    emphasis=(
        "Cover the operative clauses thoroughly — obligations, liability, "
        "termination, IP and confidentiality each deserve their own page."
    ),
)


FAMILY_2_TERM_SHEET = Family(
    key="term_sheet",
    label="Deal-stage / pre-binding",
    description=(
        "Proposed terms, not obligations. Carries binding_status precisely so "
        "indicative terms are never rendered as enforceable ones — the single "
        "most consequential distinction in this family."
    ),
    folder_hints=("term sheet", "termsheet", "loi", "letter of intent", "mou",
                  "memorandum of understanding", "heads of terms"),
    typed_table="contracts",
    doc_types=("Term Sheet", "Letter of Intent", "Memorandum of Understanding"),
    extraction_fields={
        "parties": "Parties to the proposed transaction",
        "binding_status": (
            "Which provisions are binding vs. indicative. Quote the document's "
            "own binding/non-binding language — never infer it"
        ),
        "exclusivity": "Exclusivity or no-shop window and its duration, or null",
        "conditions_precedent": "Conditions that must be met before signing, or null",
        "proposed_terms": "The commercial terms as proposed (price, structure, timing)",
        "expiry_of_offer": "When the proposal lapses, or null",
        "governing_law": "Stated governing law, or null",
    },
    metadata_fields=_SHARED_METADATA + (
        "governing_law", "effective_date", "binding_status", "exclusivity",
    ),
    high_stakes_metadata=frozenset({"binding_status", "exclusivity", "governing_law"}),
    page_target=(4, 12),
    emphasis=(
        "State on every page whether the term described is binding or "
        "indicative. Do not phrase a proposed term as an existing obligation."
    ),
)


FAMILY_3_LITIGATION = Family(
    key="litigation",
    label="Litigation / dispute",
    description=(
        "Case citations, procedural posture, holding, relief. Parties are "
        "plaintiff/defendant, not counterparties. Nothing to do with the "
        "clause/obligation schema."
    ),
    folder_hints=(
        "court case", "ccd", "judgment", "judgement", "pleading", "plaint",
        "brand judgement", "brand judgment", "litigation", "petition",
        "writ", "suit", "order",
    ),
    typed_table="litigation_facts",
    doc_types=("Court Judgment", "Pleading", "Plaint", "Petition", "Court Order"),
    extraction_fields={
        "court": "Name of the court, or null if not stated in the document",
        "case_number": "Case/docket number, or null if not stated",
        "plaintiffs": "Party or parties bringing the action",
        "defendants": "Party or parties defending",
        "procedural_posture": (
            "Stage of proceedings this document represents (plaint, interim "
            "application, final judgment, appeal)"
        ),
        "holding": (
            "What the court actually held. Null if this document is a pleading "
            "rather than a decision — a plaint records no holding"
        ),
        "relief_granted": (
            "Relief actually granted by the court. Null if none is recorded. "
            "Relief *sought* is not relief granted — never conflate them"
        ),
        "relief_sought": "Relief the filing party asks for, or null",
        "disposition": "Final outcome (allowed/dismissed/settled), or null if none recorded",
        "decided_date": "Date of decision, or null",
        "statutes_cited": "Statutes and rules invoked, verbatim",
    },
    metadata_fields=_SHARED_METADATA + (
        "court", "case_number", "disposition", "decided_date",
        "procedural_posture", "matter_reference",
    ),
    high_stakes_metadata=frozenset({
        "holding", "relief_granted", "disposition", "court", "case_number",
    }),
    page_target=(12, 40),
    emphasis=(
        "Litigation documents are citation-dense and paragraph-numbered — "
        "prefer finer granularity, one page per issue or per numbered "
        "paragraph group. Distinguish relief sought from relief granted "
        "explicitly on every page that touches either."
    ),
    clause_extraction=False,
)


FAMILY_4_AUTHORIZATION = Family(
    key="authorization",
    label="Corporate / authorization instruments",
    description=(
        "Grants of authority, not agreements. Nobody is obligated by a power "
        "of attorney — somebody is empowered by it."
    ),
    folder_hints=("board resolution", "resolution", "power of attorney", "poa",
                  "authorisation", "authorization", "delegation"),
    typed_table="authorizations",
    doc_types=("Board Resolution", "Power of Attorney", "Delegation of Authority"),
    extraction_fields={
        "grantor": "Who confers the authority",
        "grantee": "Who receives it",
        "scope_of_authority": "Precisely what the grantee is empowered to do",
        "limitations": "Express limits, monetary thresholds or carve-outs, or null",
        "resolving_body": "Board, committee or officer that passed the instrument, or null",
        "effective_date": "When the authority takes effect, or null",
        "expiry_date": "When the authority lapses or is revoked, or null",
    },
    metadata_fields=_SHARED_METADATA + (
        "grantor", "grantee", "effective_date", "expiry_date", "resolving_body",
    ),
    high_stakes_metadata=frozenset({
        "scope_of_authority", "limitations", "grantee", "expiry_date",
    }),
    page_target=(3, 10),
    emphasis=(
        "Scope and limitations are the whole substance — record them verbatim. "
        "Never describe a grant of authority as an obligation."
    ),
    clause_extraction=False,
)


FAMILY_5_OPINION = Family(
    key="opinion",
    label="Advisory",
    description=(
        "An opinion advises; it doesn't obligate anyone. Its assumptions and "
        "reliance limitation are as load-bearing as its conclusion."
    ),
    folder_hints=("legal opinion", "opinion", "advice", "memo of advice",
                  "counsel opinion"),
    typed_table="opinions",
    doc_types=("Legal Opinion", "Counsel Advice"),
    extraction_fields={
        "addressee": "Who the opinion is addressed to",
        "matters_opined": "The specific questions the opinion answers",
        "assumptions": "Assumptions the opinion rests on, verbatim",
        "qualifications": "Qualifications and carve-outs, verbatim",
        "conclusion": "The opinion's actual conclusion",
        "reliance_limitation": "Who may rely on it and on what terms, or null",
        "opinion_date": "Date of the opinion, or null",
        "governing_law": "Law the opinion is given under, or null",
    },
    metadata_fields=_SHARED_METADATA + (
        "addressee", "opinion_date", "governing_law", "matter_reference",
    ),
    high_stakes_metadata=frozenset({
        "conclusion", "assumptions", "qualifications", "reliance_limitation",
    }),
    page_target=(4, 15),
    emphasis=(
        "Never state the conclusion without the assumptions and qualifications "
        "it is conditioned on — an unqualified restatement of a qualified "
        "opinion is a misstatement of it."
    ),
    clause_extraction=False,
)


FAMILY_GENERIC = Family(
    key=GENERIC_FAMILY,
    label="Ungrouped / generic fallback",
    description=(
        "Whatever doesn't match Families 1-5. Not a dumping ground — what "
        "accumulates here is the signal for whether a sixth family is needed."
    ),
    folder_hints=(),
    typed_table=None,
    doc_types=(),
    extraction_fields={
        "parties": "Any named parties or entities, or null",
        "subject_matter": "What this document is about, in one sentence",
        "effective_date": "Any operative date stated, or null",
        "jurisdiction": "Any jurisdiction stated, or null",
        "key_terms": "The document's most significant stated terms",
    },
    metadata_fields=_SHARED_METADATA + ("effective_date", "matter_reference"),
    high_stakes_metadata=frozenset(),
    page_target=(6, 20),
    emphasis=(
        "Document type is uncertain — describe what the document says without "
        "assuming it is an agreement, a decision, or a grant of authority."
    ),
)


_BUILTIN: dict[str, Family] = {
    f.key: f for f in (
        FAMILY_1_CONTRACT,
        FAMILY_2_TERM_SHEET,
        FAMILY_3_LITIGATION,
        FAMILY_4_AUTHORIZATION,
        FAMILY_5_OPINION,
        FAMILY_GENERIC,
    )
}

_registry: dict[str, Family] | None = None


def _overrides_path() -> str:
    explicit = os.getenv("SCHEMA_REGISTRY_OVERRIDES")
    if explicit:
        return explicit
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "schema_overrides.json")


def _apply_overrides(base: dict[str, Family]) -> dict[str, Family]:
    """Merge an org-override file over the built-ins.

    Deliberately additive-only for the dict/tuple fields: an override adds
    folder hints and extraction fields, it doesn't delete built-in ones. An
    org that needs a field *gone* is describing a new family, not an override,
    and should say so explicitly by defining one.
    """
    path = _overrides_path()
    if not os.path.exists(path):
        return base
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as err:
        logger.error("Schema override file %s unreadable, ignoring it: %s", path, err)
        return base

    out = dict(base)
    for key, patch in (raw.get("families") or {}).items():
        if not isinstance(patch, dict):
            continue
        current = out.get(key)
        if current is None:
            # A genuinely new family. Everything not supplied falls back to
            # the generic shape rather than to Family 1's contract fields —
            # inheriting contract fields by default is how a schema registry
            # quietly becomes a contract schema again.
            current = replace(
                FAMILY_GENERIC,
                key=key,
                label=patch.get("label", key),
                description=patch.get("description", ""),
                typed_table=patch.get("typed_table"),
            )
        merged = {
            "folder_hints": tuple(dict.fromkeys(
                current.folder_hints + tuple(
                    h.lower() for h in patch.get("folder_hints", [])
                )
            )),
            "extraction_fields": {
                **current.extraction_fields,
                **(patch.get("extraction_fields") or {}),
            },
            "metadata_fields": tuple(dict.fromkeys(
                current.metadata_fields + tuple(patch.get("metadata_fields", []))
            )),
            "high_stakes_metadata": frozenset(
                current.high_stakes_metadata
                | set(patch.get("high_stakes_metadata", []))
            ),
        }
        if "page_target" in patch and isinstance(patch["page_target"], list) \
                and len(patch["page_target"]) == 2:
            merged["page_target"] = (int(patch["page_target"][0]),
                                     int(patch["page_target"][1]))
        for scalar in ("label", "description", "emphasis", "typed_table"):
            if scalar in patch:
                merged[scalar] = patch[scalar]
        if "clause_extraction" in patch:
            merged["clause_extraction"] = bool(patch["clause_extraction"])
        out[key] = replace(current, **merged)
        logger.info("Schema registry: applied org override for family '%s'", key)
    return out


def registry() -> dict[str, Family]:
    global _registry
    if _registry is None:
        _registry = _apply_overrides(_BUILTIN)
    return _registry


def reload_registry() -> None:
    """Drop the cached registry so the next call re-reads the override file."""
    global _registry
    _registry = None


def get(family_key: str | None) -> Family:
    """Look up a family, falling back to generic. Never raises — an unknown
    family key at extraction time should degrade to the fallback schema, not
    abort the ingest of a document that is otherwise fine."""
    reg = registry()
    if family_key and family_key in reg:
        return reg[family_key]
    if family_key:
        logger.warning("Unknown family '%s', using generic fallback", family_key)
    return reg[GENERIC_FAMILY]


def all_families() -> list[Family]:
    return [f for k, f in registry().items() if k != GENERIC_FAMILY] + [
        registry()[GENERIC_FAMILY]
    ]


def family_keys() -> list[str]:
    return list(registry().keys())


def classify_by_folder(source_doc: str | None) -> tuple[str | None, str | None]:
    """Folder-derived family hint, or (None, None) when no folder context
    exists — which is the normal case for an ad-hoc chat upload, not an error.

    Returns (family_key, matched_hint). This is *only ever* a hint: the caller
    boosts confidence when content classification agrees with it and routes to
    the Review Queue when it disagrees. It is never used as ground truth on
    its own, because a misfiled document is exactly the case this has to catch
    rather than rubber-stamp.
    """
    if not source_doc:
        return None, None
    haystack = source_doc.lower().replace("_", " ").replace("/", " ").replace("\\", " ")
    best: tuple[str, str] | None = None
    for key, fam in registry().items():
        for hint in fam.folder_hints:
            if hint in haystack:
                # Longest hint wins: "joint venture" should beat the generic
                # "agreement" substring that also matches it.
                if best is None or len(hint) > len(best[1]):
                    best = (key, hint)
    return best if best else (None, None)


def metadata_field_list(family_key: str | None) -> tuple[str, ...]:
    return get(family_key).metadata_fields


def is_high_stakes_metadata(family_key: str | None, field_name: str) -> bool:
    return field_name in get(family_key).high_stakes_metadata


def typed_table_for(family_key: str | None) -> str | None:
    return get(family_key).typed_table
