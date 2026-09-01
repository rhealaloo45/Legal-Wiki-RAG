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
        "expect_docs": ["Amberline", "Cobalt"],
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
