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

# The cases themselves name real documents, parties and figures from the
# corpus, so they live in services/regression_cases_local.py, which is not
# committed. regression_cases_local.example.py shows the format with
# fictional content. With no local file there is nothing to seed.
try:
    from services.regression_cases_local import SEED_CASES
except ImportError:
    SEED_CASES: list[dict] = []


def load(wiki_id: str, session_id: str) -> dict:
    """Upsert every seed case. Idempotent — safe to re-run after edits."""
    from services import db
    loaded = []
    if not SEED_CASES:
        logger.warning("regression seed: no services/regression_cases_local.py; "
                       "nothing to load (see regression_cases_local.example.py)")
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
