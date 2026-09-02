"""
Seed case set for the accuracy regression suite (§ Phase 3.5a).

Every case here was verified by hand against the actual documents in the
corpus — read out of the database or the source PDF — not taken from a model's
own answer. That distinction is the whole value of the file: a suite seeded
from what the system currently says would lock in today's mistakes as the
expected result.

Two groups:

  Scope cases pin which documents a question must resolve to. They cost
  nothing to run and cover the bug class this corpus keeps producing —
  compound party pairs, documents named only by nickname, a document named by
  type alone opposite an amendment. Each one of these corresponds to a real
  retrieval failure that was found and fixed; they exist so it stays fixed.

  Answer cases carry a verified expected answer for the graded tier, including
  the abstention cases where the correct behaviour is to decline. Those are
  the most valuable cases in the set: a system that answers everything is easy
  to build and useless, and abstention is the first thing to regress when
  retrieval is loosened.

Loaded with `python -m services.regression_seed` or via the admin route.
Idempotent — cases upsert on (wiki, session, name).
"""

import logging

logger = logging.getLogger(__name__)

# Document-name fragments, not full source_doc values — see
# regression._doc_matches for why.
_IOA = "Apex Meridian Software-IOA"
_IOA_AMDT = "Apex Meridian Software Amdt"
# Sept 2026 grading pass — see the block at the end of SEED_CASES.
_CONS = "Consultancy Agreement - 2024-11-17"
_SPA = "Share Purchase Agreement_2024-05-26"
_NDA_ZEPHYRA = "NDA_APEX ZEPHYRA TRADING"

SEED_CASES: list[dict] = [
    # ---------------------------------------------------------------- scope
    {
        "name": "scope-compound-party-pair",
        "archetype": "abstention",
        "question": "What is the liability cap in the agreement between Apex Meridian Software and Ironvane Data Centers?",
        # Only the document a correct answer actually needs to cite. resolve_scope
        # itself reaches the IT Outsourcing Agreement too (checked separately, by
        # hand, against wiki.resolve_scope directly) — but files_used reflects what
        # the ANSWER cites, and a correct abstention here is fully supported by the
        # Amendment's own text without needing to quote the IOA. Requiring both
        # fragments here would fail a correct answer for not citing a document it
        # had no need to.
        "expect_docs": [_IOA_AMDT],
        "expect_abstain": True,
        "expect_answer": "No numeric liability cap is stated. The Amendment explicitly notes it "
                         "does not set or modify a cap and only preserves the original agreement's "
                         "limitation-of-liability terms; the original IT Outsourcing Agreement itself "
                         "contains no liability-cap clause.",
        "notes": "CORRECTED after the pipeline tier's first real run flagged this case as failing, "
                 "which led to checking the actual document text rather than trusting the seed's "
                 "original assumption. Two things were wrong, not the product:"
                 " (1) this was seeded as a 'find the cap' case, but neither the IOA nor the "
                 "Amendment states one — verified directly against both documents' extracted text. "
                 "It is an abstention case."
                 " (2) the case's own docstring called it 'baseline for _resolve_docs_by_party_pair', "
                 "but that resolver returns empty here: its title-based intersection needs both party "
                 "names in a shared page TITLE, and this document family titles pages by matter code "
                 "('MAT-2026-4342'), not by party. Scope still reaches both correct documents, via the "
                 "weaker party-multi single-party fallback — which also pulls in one unrelated document "
                 "(a Consulting Agreement between Meridian and a third party) as noise. Not fixed here: "
                 "content-based pair intersection was tried as a fix and produced its own false positive "
                 "(a Loan Agreement page coincidentally containing both 'Ironvane' and 'Meridian' from two "
                 "unrelated entities), matching the exact failure class _resolve_one_party_pair's own "
                 "docstring already warns content-intersection causes. A real fix belongs on "
                 "documents.parties (a clean JSONB array, not text matching) — flagged as a follow-up, "
                 "not attempted under this fix.",
    },
    {
        "name": "scope-original-of-amendment",
        "archetype": "scope",
        "question": "What did the original IT Outsourcing Agreement say about payment terms, and how does the Apex Meridian amendment change it?",
        "expect_scope_method": "party-pair-compound",
        "expect_docs": [_IOA, _IOA_AMDT],
        "notes": "A document named by TYPE alone, opposite an amendment named by a "
                 "bare party with no corporate suffix. Fell through to unscoped "
                 "corpus-wide search before _resolve_original_of_amendment existed; "
                 "the amendment's document_relations edge is still unresolved, so "
                 "this resolves by party-array overlap rather than by the edge.",
    },
    {
        "name": "scope-named-instrument-list",
        "archetype": "scope",
        "question": "Compare the confidentiality clauses of the Amberline NDA and the Apex Cobalt NDA.",
        # "Cobalt" alone also matches an unrelated Cobalt Capital SPA, so the
        # case could pass while scope resolved the wrong document. Narrowed to
        # the instrument the question actually names.
        "expect_docs": ["Amberline", "Apex Cobalt - NDA"],
        "notes": "Several documents named by bare nickname with no party suffix and "
                 "no 'between'. Covers _resolve_docs_by_named_instruments.",
    },
    {
        "name": "scope-single-doc-by-date",
        "archetype": "scope",
        "question": "What is the liability cap in the Master Services Agreement dated 12 August 2020?",
        "expect_docs": ["2020-08-12"],
        "notes": "Simplest shape — one document pinned by an explicit date. Exists to "
                 "catch a scope refactor breaking the easy case while fixing a hard one.",
    },

    # --------------------------------------------------------------- answers
    {
        "name": "answer-liability-cap-msa-2020",
        "archetype": "factual",
        "question": "What is the liability cap in the Master Services Agreement dated 12 August 2020?",
        "expect_docs": ["2020-08-12"],
        "must_contain": ["110,661,324"],
        "expect_answer": "The aggregate liability cap is Rs. 110,661,324.",
        "notes": "Precise figure lookup. Verified against the document's own liability "
                 "clause.",
    },
    {
        "name": "answer-liability-cap-carveouts",
        "archetype": "factual",
        "question": "What are the exceptions to the liability cap in the Master Services Agreement dated 12 August 2020?",
        "expect_docs": ["2020-08-12"],
        "expect_answer": "The cap does not apply to liability arising from fraud, "
                         "gross negligence, wilful misconduct, or breach of "
                         "confidentiality obligations.",
        "notes": "Clause plus carve-out list. The carve-outs are the part a "
                 "cap-focused retrieval most often drops.",
    },
    {
        "name": "answer-governing-law-family",
        "archetype": "factual",
        "question": "What is the governing law of our Master Services Agreements?",
        "expect_answer": "The Republic of India.",
        "notes": "Family-scoped question, no single document named.",
    },
    {
        "name": "abstain-noncompete-msa",
        "archetype": "abstention",
        "question": "Does the Master Services Agreement dated 12 August 2020 contain a non-compete restriction?",
        "expect_abstain": True,
        "expect_docs": ["2020-08-12"],
        "expect_answer": "No non-compete restriction is established in that agreement. "
                         "The document does not contain such a clause.",
        "notes": "ABSENCE case. The correct answer is a clean refusal. A system that "
                 "invents a non-compete here is doing the single worst thing this "
                 "product can do.",
    },
    {
        "name": "abstain-termination-fee-msa",
        "archetype": "abstention",
        "question": "What is the termination fee payable under the Master Services Agreement dated 12 August 2020?",
        "expect_abstain": True,
        "expect_docs": ["2020-08-12"],
        "expect_answer": "No termination fee is established in that agreement — the "
                         "document provides for termination without specifying any "
                         "such fee.",
        "notes": "Presupposition trap: the question assumes a fee exists. Correct "
                 "behaviour is to decline rather than to produce a plausible number.",
    },
    # ------------------------------------------------- Sept 2026 grading pass
    # Fifteen lawyer-grade questions run against the live system and graded by
    # hand. Every figure below was then read back out of `clauses` verbatim
    # before being written here — none is transcribed from the system's own
    # answer, which is the standard the rest of this file is held to.
    #
    # Two documents carry these cases:
    #   CONS  a Consultancy Agreement, Apex Suvarna Telecommunications / Nidra
    #         Bhandari, MAT-2018-3636. Nidra Bhandari is a natural person, so
    #         the name carries no corporate suffix — which is exactly what the
    #         party-pair resolver used to choke on.
    #   SPA   a Share Purchase Agreement, Apex Zephyra Trading (Buyer) /
    #         Nimbus Capital (Seller), MAT-2021-7750. One of SEVENTEEN real
    #         documents in this corpus between those same two parties.
    {
        "name": "answer-termination-convenience-cons",
        "archetype": "factual",
        "question": "What notice is required to terminate the consultancy agreement between "
                    "Apex Suvarna Telecommunications Private Limited and Nidra Bhandari for convenience?",
        "expect_docs": [_CONS],
        "must_contain": ["60"],
        "expect_answer": "Either Party may terminate for convenience on not less than 60 days' "
                         "prior written notice, without liability save for accrued rights and "
                         "obligations as at the date of termination.",
        "notes": "Scored 10/10. Baseline single-document factual lookup — the archetype the "
                 "system is strongest at. Here to catch a regression in the easy case.",
    },
    {
        "name": "answer-insurance-cover-cons",
        "archetype": "factual",
        "question": "What insurance is Nidra Bhandari required to maintain under the consultancy agreement?",
        "expect_docs": [_CONS],
        "must_contain": ["30,618,959"],
        "expect_answer": "Comprehensive general liability and professional indemnity insurance of "
                         "not less than Rs. 30,618,959, maintained at Nidra Bhandari's own cost for "
                         "the duration of the Term.",
        "notes": "Scored 10/10. Precise figure plus the two named cover types.",
    },
    {
        "name": "answer-customer-data-location-cons",
        "archetype": "factual",
        "question": "What restrictions apply to where Customer Data may be stored under the "
                    "consultancy agreement with Nidra Bhandari?",
        "expect_docs": [_CONS],
        "must_contain": ["Singapore"],
        "expect_answer": "Customer Data may not be stored or processed outside Singapore without "
                         "Apex Suvarna's prior written consent, and Nidra Bhandari must keep a "
                         "current record of every location at which Customer Data is processed.",
        "notes": "Scored 10/10. The record-keeping duty is the half a summary most often drops.",
    },
    {
        "name": "answer-security-incident-24h-cons",
        "archetype": "factual",
        "question": "How quickly must Nidra Bhandari notify Apex Suvarna Telecommunications "
                    "Private Limited of a security incident?",
        "expect_docs": [_CONS],
        "must_contain": ["24 hours"],
        "expect_answer": "Without undue delay and in any event within 24 hours of becoming aware "
                         "of any actual or suspected security incident affecting Customer Data.",
        "notes": "Scored 10/10. Paired deliberately with the 48-hour SPA-side case below — the "
                 "two together are what the cross-document comparison case needs.",
    },
    {
        "name": "answer-audit-rights-cons",
        "archetype": "factual",
        "question": "What audit rights does Apex Suvarna Telecommunications Private Limited have "
                    "over Nidra Bhandari?",
        "expect_docs": [_CONS],
        "must_contain": ["15", "5%"],
        "expect_answer": "Apex Suvarna may audit Nidra Bhandari's relevant books and records on not "
                         "less than 15 days' written notice, no more than once in any twelve-month "
                         "period, during normal business hours and at Apex Suvarna's cost — except "
                         "where a material discrepancy of more than 5% is identified.",
        "notes": "Scored 10/10. Four constraints in one clause; the cost-shifting exception is "
                 "the one most often lost.",
    },
    {
        "name": "answer-milestone-payment-45d-cons",
        "archetype": "factual",
        "question": "When do milestone payments fall due under the consultancy agreement with Nidra Bhandari?",
        "expect_docs": [_CONS],
        "must_contain": ["45"],
        "expect_answer": "Within 45 days of Apex Suvarna's written acceptance of the corresponding "
                         "Milestone deliverable, subject to the retention provision.",
        "notes": "Scored 10/10. The retention carve-out is part of a correct answer.",
    },
    {
        "name": "answer-take-or-pay-cons",
        "archetype": "factual",
        "question": "What is the take-or-pay minimum quantity under the agreement with Nidra Bhandari?",
        "expect_docs": [_CONS],
        "must_contain": ["487,520"],
        "expect_answer": "487,520 units per annum, payable whether or not Apex Suvarna takes "
                         "delivery, save where non-delivery arises from a Force Majeure Event or "
                         "Nidra Bhandari's own default.",
        "notes": "Scored 10/10. The two exceptions matter — without them the obligation reads as "
                 "absolute when it is not.",
    },
    {
        "name": "router-liability-cap-with-carveouts-cons",
        "archetype": "factual",
        "question": "What is the aggregate liability cap for Apex Suvarna Telecommunications "
                    "Private Limited and Nidra Bhandari, and what types of liability are excluded "
                    "from that cap?",
        "expect_docs": [_CONS],
        "must_contain": ["366,841,583", "fraud", "gross negligence",
                         "wilful misconduct", "confidentiality"],
        "expect_answer": "The aggregate liability cap is Rs. 366,841,583. It does not apply to "
                         "liability arising from fraud, gross negligence, wilful misconduct, or "
                         "breach of confidentiality obligations.",
        "notes": "ROUTER REGRESSION CASE. Scored 6/10 originally: the words 'aggregate' and "
                 "'liability cap' sent this to the corpus-analytics fast path, which answered "
                 "with Total/Median/Mean over ONE document and silently dropped the carve-out "
                 "half of the question entirely. Fixed by vetoing the aggregate branch when a "
                 "named party or a carve-out request is present. must_contain deliberately lists "
                 "all four carve-outs — a statistic cannot satisfy them.",
    },
    {
        "name": "scope-buyer-seller-role-disambiguation",
        "archetype": "scope",
        "question": "Now look at the agreement where Apex Zephyra Trading Company LLC is the buyer "
                    "and Nimbus Capital Private Limited is the seller. What is the minimum annual "
                    "off-take commitment of Apex Zephyra, and what happens if it falls short?",
        "expect_docs": [_SPA],
        "must_contain": ["206,696"],
        "expect_answer": "Apex Zephyra must purchase not less than 206,696 units of the Products "
                         "each Contract Year (the Minimum Purchase Commitment). On a shortfall it "
                         "pays Nimbus Capital a shortfall fee equal to 5% of the value of the "
                         "shortfall.",
        "notes": "Seventeen real documents in this corpus name these same two parties. Only the "
                 "Share Purchase Agreement labels them 'Buyer' and 'Seller' in its own Parties "
                 "clause — verified directly — so the question's own wording resolves it uniquely. "
                 "A near-identical Minimum Off-take of 46,133 units sits in the NDA-titled "
                 "document and is the wrong answer here; that is the trap this case exists to "
                 "catch.",
    },
    {
        "name": "scope-cross-document-notification-compare",
        "archetype": "comparison",
        "question": "Compare the security incident notification obligations of Nidra Bhandari to "
                    "Apex Suvarna Telecommunications Private Limited with those of Nimbus Capital "
                    "Private Limited to Apex Zephyra Trading Company LLC. Which counterparty has "
                    "the shorter contractual notification period?",
        "expect_docs": [_CONS, _NDA_ZEPHYRA],
        "must_contain": ["24", "48"],
        "expect_answer": "Nidra Bhandari must notify within 24 hours; Nimbus Capital must notify "
                         "within 48 hours. Nidra Bhandari therefore has the shorter notification "
                         "period.",
        "notes": "THE HARDEST CASE IN THE SET, and the one that found a real bug. Scored 5/10: it "
                 "retrieved the 24-hour obligation and then reported the 48-hour one as 'not "
                 "present in the supplied materials' — it is present, verbatim. Two independent "
                 "causes: (1) 'Nidra Bhandari' carries no corporate suffix so only THREE of the "
                 "four party names were extracted, leaving an odd token count that made "
                 "combinatorial pairing decline outright; (2) the document holding the 48-hour "
                 "clause is titled 'NDA-Apex Zephyra' on every page and never says 'Nimbus', so "
                 "title-token search could not reach it at any threshold. Both fixed. Requires "
                 "BOTH documents, which is the point.",
    },
    {
        "name": "answer-milestone-failure-protection-spa",
        "archetype": "factual",
        "question": "From Apex Zephyra Trading Company LLC's perspective, what protections does it "
                    "have if Nimbus Capital Private Limited fails to achieve a contractual "
                    "milestone on time?",
        "expect_docs": [_SPA],
        "must_contain": ["0.5%"],
        "expect_answer": "Liquidated damages at 0.5% of the Contract Price per week of delay, "
                         "subject to the stated cap.",
        "notes": "Scored 9/10. Worth keeping because the identical liquidated-damages wording "
                 "also appears in the NDA-titled sibling document, so a wrong-document answer "
                 "still produces the right number here — the case tests scope by expect_docs, "
                 "not by the figure.",
    },
    {
        "name": "abstain-consultant-role-label-cons",
        "archetype": "abstention",
        "question": "Is Nidra Bhandari identified as the Consultant in the parties clause of the "
                    "consultancy agreement?",
        "expect_abstain": True,
        "expect_docs": [_CONS],
        "expect_answer": "The extracted parties text does not assign Nidra Bhandari the role label "
                         "'Consultant'. The party is named directly, without a defined role term.",
        "notes": "ABSENCE case, and a genuine disagreement with the human answer key: the key "
                 "asserted the 'Consultant' label, the system said it was absent, and checking "
                 "the document showed the system was right. Kept because the failure mode it "
                 "guards against — inventing a role label that reads plausibly — is exactly the "
                 "kind of small fabrication that survives review.",
    },

]


def load(wiki_id: str, session_id: str) -> dict:
    """Upsert every seed case. Idempotent — safe to re-run after edits."""
    from services import db
    loaded = []
    for case in SEED_CASES:
        fields = {k: v for k, v in case.items() if k not in ("name", "question")}
        db.upsert_regression_case(wiki_id, session_id, case["name"],
                                  case["question"], **fields)
        loaded.append(case["name"])
    logger.info("regression seed: loaded %d cases into wiki=%s session=%s",
                len(loaded), wiki_id, session_id)
    return {"loaded": len(loaded), "names": loaded}


if __name__ == "__main__":
    import os
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    wiki_id = os.environ.get("REGRESSION_WIKI_ID")
    session_id = os.environ.get("REGRESSION_SESSION_ID")
    if len(sys.argv) >= 3:
        wiki_id, session_id = sys.argv[1], sys.argv[2]
    if not wiki_id or not session_id:
        print("usage: python -m services.regression_seed <wiki_id> <session_id>")
        raise SystemExit(2)
    print(load(wiki_id, session_id))
