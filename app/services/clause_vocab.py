"""
Canonical clause vocabulary (target architecture § Phase 3.5c).

`clauses.clause_type` is whatever the extracting model chose to call a clause:
5,956 distinct values across 31,457 rows on this corpus, one new type per five
clauses. "Definitions", "Definition", "Definition - Affiliate"; "Liability
Cap", "Liability cap", "Aggregate Liability Cap", "Exclusions from Liability
Cap" — all separate types. Nothing downstream can key off that reliably, which
is why Playbooks matches on keywords instead, and why a "Liability Cap" rule
reduces to the single word "liability" and then assesses carve-out clauses
against a cap standard.

This module maps that free text onto a controlled set. Three properties matter
more than coverage:

  The raw label is never discarded. It is often more specific than the canon
  ("Change of Control Termination" vs `termination`), it is what a reviewer
  recognises, and it is the only record of what the model actually said.
  `clause_type_canon` is added beside `clause_type`, never over it.

  An unmatched label maps to NULL, not to a nearest guess. A wrong canonical
  type is worse than an absent one: absent means a downstream query skips the
  row, wrong means it confidently includes it in the wrong bucket — the exact
  failure being fixed here.

  Carve-outs are their own type, not a flavour of what they carve out of.
  `liability_cap_exclusion` exists precisely so it stops being assessed as a
  `liability_cap`. This distinction is the point of the whole exercise.

Deterministic string work — no LLM call, no embedding, so the backfill over
the whole corpus is free and re-runnable.
"""

import logging
import re

logger = logging.getLogger(__name__)

# The controlled set. Grouped by what a Playbook rule or a typed query would
# actually want to select, not by legal taxonomy for its own sake.
CANON = (
    # liability / risk
    "liability_cap", "liability_cap_exclusion", "indemnity", "indemnity_exclusion",
    "insurance", "warranty", "representation", "force_majeure",
    # lifecycle
    "term", "termination_cause", "termination_convenience", "termination_change_of_control",
    "renewal", "survival", "exit", "notice_period",
    # commercial
    "payment_terms", "fees", "fee_escalation", "contract_value", "audit_rights",
    "service_levels", "change_control", "escrow",
    # information
    "confidentiality", "data_protection", "security_incident", "ip_ownership",
    "publicity", "records_retention",
    # governance / boilerplate
    "governing_law", "dispute_resolution", "assignment", "subcontracting",
    "compliance", "anti_bribery", "notices", "entire_agreement", "severability",
    "waiver", "counterparts", "third_party_rights", "relationship_of_parties",
    "further_assurance", "amendment_clause",
    # corporate / transactional
    "board_composition", "reserved_matters", "information_rights", "deadlock",
    "conditions_precedent", "closing", "restrictive_covenant", "transfer_restriction",
    # obligations and scope
    "effective_date", "obligations", "purpose", "permitted_use",
    "minimum_commitment", "interim_relief",
    # structure, not substance — kept so these stop polluting substantive queries
    "definition", "structural",
)

# Exact matches after normalisation, checked first. Cheapest and least
# ambiguous path; the pattern rules below only see what these miss.
_EXACT: dict[str, str] = {
    "governing law": "governing_law",
    "applicable law": "governing_law",
    "confidentiality": "confidentiality",
    "confidentiality obligations": "confidentiality",
    "indemnity": "indemnity",
    "indemnification": "indemnity",
    "indemnity scope": "indemnity",
    "survival": "survival",
    "entire agreement": "entire_agreement",
    "severability": "severability",
    "waiver": "waiver",
    "counterparts": "counterparts",
    "notices": "notices",
    "notice": "notices",
    "assignment": "assignment",
    "insurance": "insurance",
    "term": "term",
    "audit rights": "audit_rights",
    "audit right": "audit_rights",
    "payment terms": "payment_terms",
    "publicity": "publicity",
    "records retention": "records_retention",
    "further assurance": "further_assurance",
    "board composition": "board_composition",
    "reserved matters": "reserved_matters",
    "reserved matter": "reserved_matters",
    "information rights": "information_rights",
    "ip ownership": "ip_ownership",
    "background ip ownership": "ip_ownership",
    "change control": "change_control",
    "anti-bribery": "anti_bribery",
    "anti bribery": "anti_bribery",
    "compliance with laws": "compliance",
    "compliance obligation": "compliance",
    "dispute resolution": "dispute_resolution",
    "relationship of parties": "relationship_of_parties",
    "no third party rights": "third_party_rights",
    "third party rights": "third_party_rights",
    "escrow deposit": "escrow",
    "exit triggers": "exit",
    "deadlock escalation": "deadlock",
    "force majeure": "force_majeure",
    "conditions precedent": "conditions_precedent",
}

# Ordered rules. FIRST MATCH WINS, so the carve-out and the more specific
# lifecycle patterns are listed above the general ones they would otherwise be
# swallowed by — the ordering here is load-bearing, not cosmetic.
_RULES: list[tuple[re.Pattern, str]] = [
    # --- definitions and structure first ------------------------------------
    # A definition is not the operative clause it defines a term for, and
    # letting "Definition - Effective Date" land on `effective_date` (or
    # "Definition - Confidential Information" on `confidentiality`) is the same
    # error as the liability carve-out below: a Playbook rule would assess a
    # definition against a standard written for the clause that uses it. The
    # "Definition -" prefix is unambiguous, so it is settled before anything
    # else gets a look. Caught by the mapping test after an earlier version
    # ordered these last.
    (re.compile(r"^definitions?\b|^definition\s*[-–:]", re.I), "definition"),
    (re.compile(r"^(?:schedule|annexure|annex|appendix|exhibit)\b|"
                r"\b(?:document\s+header|section\s+heading|signature\s+block|"
                r"execution\s+block|preamble|recitals?|table\s+of\s+contents)\b|"
                r"^(?:execution|parties)$", re.I), "structural"),

    # --- carve-outs next: these must never fall through to their parent ---
    # Plurals are spelled out deliberately. An earlier version ended these
    # alternations with \b after the singular, so "Liability Cap ExclusionS"
    # failed to match and mapped to `liability_cap` — silently reintroducing
    # the precise bug this module exists to remove. Caught by the mapping test,
    # not by reading the regex.
    (re.compile(r"\b(?:exclusions?|exceptions?|carve[- ]?outs?|not\s+subject|uncapped)"
                r"[^|]*\bliab", re.I), "liability_cap_exclusion"),
    (re.compile(r"\bliab[^|]*\b(?:exclusions?|exceptions?|carve[- ]?outs?|uncapped)", re.I),
     "liability_cap_exclusion"),
    (re.compile(r"\bindemn[^|]*\b(?:exclusions?|exceptions?|carve[- ]?outs?)", re.I),
     "indemnity_exclusion"),
    (re.compile(r"\b(?:exclusions?|exceptions?|carve[- ]?outs?)[^|]*\bindemn", re.I),
     "indemnity_exclusion"),
    (re.compile(r"\bnon[- ]?indemnifiable\b", re.I), "indemnity_exclusion"),
    # "Liability Distinctions" / "Liability Differentiation" — the corpus's own
    # labels for a clause whose text carves fraud, wilful misconduct and
    # confidentiality breaches out of the ordinary cap. Without this they map
    # to NULL, which drops them from a cap rule for the right outcome but the
    # wrong reason: unmapped also makes them invisible to any future query for
    # carve-outs. Verified against the clause text, not inferred from the label.
    (re.compile(r"\bliab[^|]*\b(?:distinction|differentiat|distinguish)", re.I),
     "liability_cap_exclusion"),

    # --- liability ---
    (re.compile(r"\b(?:liability\s+cap|cap\s+on\s+liability|limitation\s+of\s+liability|"
                r"aggregate\s+liability|liability\s+caps?|unlimited\s+liability)\b", re.I),
     "liability_cap"),
    # "Data Breach Cap and Unlimited Liability" — a cap clause whose label
    # leads with its subject rather than with the word pair. Co-occurrence is
    # enough here because every carve-out rule has already been tried above,
    # and because the failure this guards against is the dangerous direction:
    # a cap clause classified as something else is a cap a Playbook rule will
    # never assess.
    (re.compile(r"\bcaps?\b[^|]*\bliab|\bliab[^|]*\bcaps?\b", re.I), "liability_cap"),
    (re.compile(r"\bindemn", re.I), "indemnity"),

    # --- termination: specific forms before the generic one ---
    (re.compile(r"\bchange\s+of\s+control\b[^|]*\bterminat|"
                r"\bterminat[^|]*\bchange\s+of\s+control\b", re.I),
     "termination_change_of_control"),
    (re.compile(r"\bterminat[^|]*\b(?:convenience|without\s+cause)\b", re.I),
     "termination_convenience"),
    (re.compile(r"\bterminat[^|]*\b(?:cause|breach|default)\b|"
                r"\b(?:cause|breach|default)[^|]*\bterminat", re.I), "termination_cause"),
    (re.compile(r"\bforce\s+majeure\b", re.I), "force_majeure"),
    (re.compile(r"\bterminat", re.I), "termination_cause"),

    # --- lifecycle ---
    (re.compile(r"\b(?:auto[- ]?renewal|renewal|extension\s+of\s+term)\b", re.I), "renewal"),
    (re.compile(r"\bnotice\s+period|period\s+of\s+notice\b", re.I), "notice_period"),
    (re.compile(r"\bsurviv", re.I), "survival"),
    (re.compile(r"\bexit\b", re.I), "exit"),

    # --- commercial ---
    # "Total Vendor Value", "Total Annual Value" — the corpus puts a qualifier
    # between "total" and "value", so an adjacent-words pattern misses them.
    (re.compile(r"\b(?:total|aggregate|contract)\b(?:\s+[a-z]+){0,2}\s+value\b|"
                r"\bconsideration\b|\bcontract\s+price\b", re.I), "contract_value"),
    (re.compile(r"\bfee\s+escalation|escalation\b", re.I), "fee_escalation"),
    (re.compile(r"\b(?:payment|invoic|billing)\b", re.I), "payment_terms"),
    (re.compile(r"\bfees?\b|\bcharges\b|\bpricing\b", re.I), "fees"),
    (re.compile(r"\baudit\b", re.I), "audit_rights"),
    (re.compile(r"\b(?:service\s+level|sla\b|kpi\b|performance\s+standard)", re.I),
     "service_levels"),
    (re.compile(r"\bchange\s+control\b", re.I), "change_control"),
    (re.compile(r"\bescrow\b", re.I), "escrow"),

    # --- information ---
    (re.compile(r"\b(?:data\s+protection|gdpr|dpdp|personal\s+data|privacy)\b", re.I),
     "data_protection"),
    (re.compile(r"\bsecurity\s+incident|data\s+breach|breach\s+notification\b", re.I),
     "security_incident"),
    (re.compile(r"\bconfidential", re.I), "confidentiality"),
    (re.compile(r"\b(?:intellectual\s+property|ip\s+ownership|ipr\b|ownership\s+of\s+ip)\b", re.I),
     "ip_ownership"),
    (re.compile(r"\b(?:publicity|press\s+release|announcement)\b", re.I), "publicity"),
    (re.compile(r"\brecords?\s+retention|retention\s+of\s+records\b", re.I), "records_retention"),

    # --- governance ---
    (re.compile(r"\b(?:governing\s+law|applicable\s+law|choice\s+of\s+law)\b", re.I),
     "governing_law"),
    (re.compile(r"\b(?:dispute|arbitrat|mediation|jurisdiction)\b", re.I), "dispute_resolution"),
    (re.compile(r"\bsub[- ]?contract", re.I), "subcontracting"),
    (re.compile(r"\bassign", re.I), "assignment"),
    (re.compile(r"\b(?:anti[- ]?bribery|anti[- ]?corruption|fcpa)\b", re.I), "anti_bribery"),
    (re.compile(r"\bcomplian|\bregulatory\b", re.I), "compliance"),
    (re.compile(r"\b(?:non[- ]?compete|non[- ]?solicit|restrictive\s+covenant)\b", re.I),
     "restrictive_covenant"),
    (re.compile(r"\bwarrant", re.I), "warranty"),
    (re.compile(r"\brepresentation", re.I), "representation"),
    (re.compile(r"\binsurance\b", re.I), "insurance"),
    (re.compile(r"\bamendment|variation\b", re.I), "amendment_clause"),

    # --- corporate ---
    (re.compile(r"\bboard\b", re.I), "board_composition"),
    (re.compile(r"\breserved\s+matter", re.I), "reserved_matters"),
    (re.compile(r"\bdeadlock\b", re.I), "deadlock"),
    (re.compile(r"\binformation\s+rights\b", re.I), "information_rights"),
    (re.compile(r"\bconditions?\s+precedent\b", re.I), "conditions_precedent"),
    (re.compile(r"\bclosing\b|\bcompletion\b", re.I), "closing"),

    # --- high-volume corpus-specific types -----------------------------------
    (re.compile(r"\beffective\s+date|commencement\s+date\b", re.I), "effective_date"),
    (re.compile(r"\b(?:inspection\s+rights?|right\s+of\s+inspection)\b", re.I), "audit_rights"),
    (re.compile(r"\btransfer\s+restrict|restrictions?\s+on\s+transfer|"
                r"\b(?:lock[- ]?in|right\s+of\s+first\s+refusal|tag[- ]?along|"
                r"drag[- ]?along|pre[- ]?emption)\b", re.I), "transfer_restriction"),
    (re.compile(r"\bminimum\s+(?:purchase|volume|order)\s+commitment|"
                r"\bminimum\s+commitment\b", re.I), "minimum_commitment"),
    (re.compile(r"\binterim\s+relief|injunctive\s+relief|specific\s+performance\b", re.I),
     "interim_relief"),
    (re.compile(r"\bpermitted\s+use|permitted\s+purpose\b", re.I), "permitted_use"),
    (re.compile(r"^purpose$|^purpose\s+of\b", re.I), "purpose"),
    (re.compile(r"\bobligations?\b|\bduties\b|\bresponsibilities\b|\bundertakings?\b", re.I),
     "obligations"),

]


def normalise(label: str) -> str:
    """Lowercase, collapse whitespace, drop trailing clause numbers and noise.

    "Indemnity (Clause 10)", "Indemnity (10)" and "Indemnity" are the same
    clause type wearing three labels; this is what makes them agree before
    either the exact map or the rules get a look.
    """
    s = (label or "").strip().lower()
    s = re.sub(r"\s*\((?:clause\s*)?\d+[a-z]?\)\s*$", "", s)   # trailing "(10)" / "(Clause 10)"
    s = re.sub(r"\s*[-–—]\s*(?:clause\s*)?\d+[a-z]?\s*$", "", s)
    s = re.sub(r"[‘’“”]", "'", s)
    s = re.sub(r"\s+", " ", s).strip(" .;:,-")
    return s


def canonical(label: str) -> str | None:
    """The canonical type for one raw clause label, or None if unmatched.

    None is a real answer, not a failure to try: see the module docstring on
    why a nearest guess is worse than an absent value here.
    """
    norm = normalise(label)
    if not norm:
        return None
    hit = _EXACT.get(norm)
    if hit:
        return hit
    # Singular/plural is the single most common spelling difference in the
    # corpus ("Audit Right"/"Audit Rights", "Liability Cap"/"Liability Caps").
    if norm.endswith("s") and _EXACT.get(norm[:-1]):
        return _EXACT[norm[:-1]]
    if _EXACT.get(norm + "s"):
        return _EXACT[norm + "s"]
    for pattern, canon in _RULES:
        if pattern.search(norm):
            return canon
    return None


def classify_all(labels: list[str]) -> dict[str, str | None]:
    """Map many labels at once, de-duplicated — the backfill's inner loop."""
    return {label: canonical(label) for label in set(labels)}


def coverage_report(counts: list[tuple[str, int]]) -> dict:
    """How much of the corpus the vocabulary actually maps, by clause volume.

    Takes (label, row_count) pairs so coverage is weighted by how many clauses
    carry each label — mapping 90% of distinct labels means little if the
    unmapped 10% are the common ones.
    """
    mapped_rows = unmapped_rows = 0
    mapped_labels = unmapped_labels = 0
    per_canon: dict[str, int] = {}
    unmapped_examples: list[tuple[str, int]] = []
    for label, n in counts:
        canon = canonical(label)
        if canon:
            mapped_rows += n
            mapped_labels += 1
            per_canon[canon] = per_canon.get(canon, 0) + n
        else:
            unmapped_rows += n
            unmapped_labels += 1
            unmapped_examples.append((label, n))
    unmapped_examples.sort(key=lambda x: -x[1])
    total = mapped_rows + unmapped_rows
    return {
        "rows_total": total,
        "rows_mapped": mapped_rows,
        "rows_unmapped": unmapped_rows,
        "row_coverage": round(mapped_rows / total, 4) if total else 0.0,
        "labels_mapped": mapped_labels,
        "labels_unmapped": unmapped_labels,
        "by_canon": sorted(per_canon.items(), key=lambda x: -x[1]),
        "top_unmapped": unmapped_examples[:40],
    }
