"""
Mapping tests for the canonical clause vocabulary (§ Phase 3.5c).

These are not illustrative examples — every label below is a real
`clauses.clause_type` value from the live corpus, and the carve-out cases
encode the specific defect this vocabulary exists to remove: a "Liability Cap"
Playbook rule assessing "Exclusions from Liability Cap" against a cap standard.

Run: python -m tests.test_clause_vocab   (from the app/ directory)

Zero database, zero network, zero model calls.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import clause_vocab as V  # noqa: E402


# (raw label from the corpus, expected canonical type)
CASES: list[tuple[str, str | None]] = [
    # --- the carve-out split: the whole point of the module -----------------
    ("Liability Cap", "liability_cap"),
    ("Liability cap", "liability_cap"),
    ("Liability Caps", "liability_cap"),
    ("Aggregate Liability Cap", "liability_cap"),
    ("General Liability Cap", "liability_cap"),
    ("General Aggregate Liability Cap", "liability_cap"),
    ("Limitation of Liability", "liability_cap"),
    ("Limitation Of Liability", "liability_cap"),
    ("Limitation of Liability - Cap", "liability_cap"),
    ("Data Breach Cap and Unlimited Liability", "liability_cap"),
    ("Liability Cap Exclusions", "liability_cap_exclusion"),
    ("Liability cap exclusions", "liability_cap_exclusion"),
    ("Liability Cap Exceptions", "liability_cap_exclusion"),
    ("Liability Cap Carve-Outs", "liability_cap_exclusion"),
    ("Liability Cap Carve-outs", "liability_cap_exclusion"),
    ("Liability Cap and Carve-Outs", "liability_cap_exclusion"),
    ("Exclusions from Liability Cap", "liability_cap_exclusion"),
    ("Limitation of Liability - Exclusions", "liability_cap_exclusion"),
    ("Limitation of Liability - Exclusion", "liability_cap_exclusion"),
    ("General Liability Exclusion", "liability_cap_exclusion"),

    # --- indemnity and its carve-outs ---------------------------------------
    ("Indemnity", "indemnity"),
    ("Indemnification", "indemnity"),
    ("Indemnity Scope", "indemnity"),
    ("Indemnity (10)", "indemnity"),
    ("Indemnity (Clause 10)", "indemnity"),
    ("Indemnity Clause", "indemnity"),
    ("IP Indemnity", "indemnity"),
    ("Indemnity Exclusion", "indemnity_exclusion"),
    ("Non-Indemnifiable Underperformance", "indemnity_exclusion"),
    ("Non-indemnifiable Underperformance", "indemnity_exclusion"),

    # --- termination: the specific forms must not collapse ------------------
    ("Termination for Cause", "termination_cause"),
    ("Termination For Cause", "termination_cause"),
    ("Termination for Material Breach", "termination_cause"),
    ("Termination for Breach", "termination_cause"),
    ("Termination for breach", "termination_cause"),
    ("Termination", "termination_cause"),
    ("Termination for Convenience", "termination_convenience"),
    ("Termination for convenience", "termination_convenience"),
    ("Change of Control Termination", "termination_change_of_control"),
    ("Force Majeure Termination", "force_majeure"),

    # --- spelling and plural variants ---------------------------------------
    ("Audit Right", "audit_rights"),
    ("Audit Rights", "audit_rights"),
    ("Reserved Matter", "reserved_matters"),
    ("Reserved Matters", "reserved_matters"),

    # --- commercial ----------------------------------------------------------
    ("Total Contract Value", "contract_value"),
    ("TOTAL Value", "contract_value"),
    ("Total Aggregate Value", "contract_value"),
    ("Total Vendor Value", "contract_value"),
    ("Fee Escalation", "fee_escalation"),
    ("Fees Escalation", "fee_escalation"),
    ("Payment Terms", "payment_terms"),

    # --- definitions and structure stay out of substantive buckets ----------
    ("Definitions", "definition"),
    ("Definition", "definition"),
    ("Definition - Affiliate", "definition"),
    ("Definition - Effective Date", "definition"),
    ("Schedule I", "structural"),
    ("Schedule IV", "structural"),
    ("Document Header", "structural"),
    ("Section Heading", "structural"),
    ("Signature Block", "structural"),
    ("Execution Block", "structural"),
    ("Parties", "structural"),

    # --- governance ----------------------------------------------------------
    ("Governing Law", "governing_law"),
    ("Applicable Law", "governing_law"),
    ("Dispute Resolution", "dispute_resolution"),
    ("Confidentiality", "confidentiality"),
    ("Security Incident Notification", "security_incident"),
    ("Anti-Bribery", "anti_bribery"),
    ("Entire Agreement", "entire_agreement"),
    ("Survival", "survival"),
]

# Labels that must map to NOTHING. A wrong canonical type is worse than an
# absent one, so the vocabulary is expected to decline on these rather than
# reach for the nearest bucket.
MUST_NOT_MAP = [
    "",
    "   ",
    "Miscellaneous",
    "General",
    "Other",
]


def run() -> int:
    failures = []

    for raw, expected in CASES:
        got = V.canonical(raw)
        if got != expected:
            failures.append(f"canonical({raw!r}) = {got!r}, expected {expected!r}")

    for raw in MUST_NOT_MAP:
        got = V.canonical(raw)
        if got is not None:
            failures.append(f"canonical({raw!r}) = {got!r}, expected None")

    # Every canonical value the rules can emit must be declared in CANON —
    # a typo in a rule's target would otherwise create a silent extra type
    # that nothing downstream knows to query.
    emitted = {c for _, c in CASES if c}
    undeclared = emitted - set(V.CANON)
    if undeclared:
        failures.append(f"rules emit types not declared in CANON: {sorted(undeclared)}")

    total = len(CASES) + len(MUST_NOT_MAP)
    if failures:
        print(f"FAIL — {len(failures)} of {total} checks failed:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok — {total} clause-vocabulary mappings correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
