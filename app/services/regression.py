"""
Accuracy regression suite (target architecture § Phase 3.5a).

The roadmap's own premise is that bad answers should become rare. Nothing in
the codebase measured that before this module: every phase so far was verified
by building the feature and checking it works, which proves the feature runs —
not that answers improved, and not that anything else stopped working.

Three tiers, separated by what they cost rather than by what they check:

  scope     Asserts resolve_scope()'s decision only. No pipeline, no LLM, no
            embedding call, no cost. Covers the failure class this corpus
            actually keeps hitting — which documents a question resolves to —
            so it is the tier that can run on every commit.

  pipeline  Runs the real query graph and asserts structural facts about the
            result: did it abstain when it should have, did it avoid asserting
            something it must not, which documents did it actually read.
            Costs one real query per case.

  graded    Pipeline plus an LLM judge scoring the answer against a stored,
            corpus-verified expected answer on a fixed rubric. Costs the query
            plus the judge call per case.

Nothing here runs automatically. Every entry point is invoked explicitly, and
`estimate_run` reports the call count before a paid tier is started, because a
full run over a large case set is a real spend that should be a decision rather
than a side effect.
"""

import logging
import re
import subprocess
import time

import config
from services import db

logger = logging.getLogger(__name__)

TIERS = ("scope", "pipeline", "graded")

# Phrases the answer layer uses when it declines to assert something. An
# abstention case passes when the answer carries one of these AND does not
# then go on to assert a specific figure — checked separately, since a model
# can hedge in one sentence and assert in the next.
# A regex rather than a substring list: the answer layer writes "does not
# contain" and "do not contain" interchangeably depending on whether the
# subject is one document or several, and an earlier substring list carrying
# only the singular reported a correct abstention as an asserted answer. A
# false failure in an abstention case is the worst kind this suite can
# produce — it argues for loosening exactly the behaviour that should stay
# strict.
_RX_ABSTAIN = re.compile(
    r"\bnot\s+(?:established|addressed|present|stated|specified|mentioned|covered|"
    r"included|found)\b"
    # An adverb routinely sits between the negation and its verb — "does not
    # EXPRESSLY label", "does not SPECIFICALLY provide". A correct abstention
    # was being scored as an assertion for exactly that reason, on an answer
    # the grader itself marked 10/10/10. The verb list is wider for the same
    # reason: a document can fail to label, define, designate or name a thing
    # just as readily as it can fail to contain one.
    r"|\bdo(?:es)?\s+not\s+(?:\w+ly\s+)?(?:contain|appear|include|address|specify|"
    r"state|impose|label|define|designate|name|identify|provide|require|mention|"
    r"refer|set\s+out|establish)\b"
    r"|\bno\s+(?:such\s+)?(?:clause|provision|restriction|term|fee|cap|information)\b"
    r"|\bcannot\s+be\s+determined\b"
    r"|\bis\s+silent\s+(?:on|as\s+to)\b"
    r"|\bnothing\s+in\s+the\s+(?:document|agreement|excerpts?)\b",
    re.IGNORECASE,
)


def _enabled() -> bool:
    return bool(getattr(config, "USE_DATABASE", False))


def _git_sha() -> str | None:
    """Current commit, so a stored run can be tied to the code that produced it."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _doc_matches(actual_docs: list[str], expected_fragment: str) -> bool:
    """Expected docs are stored as fragments, not full filenames.

    Full source_doc values carry a session-id prefix and an arbitrary original
    filename, so pinning a case to one would make the whole suite break on
    re-ingest under a new session — which is a corpus change, not a regression.
    A fragment ("Apex Meridian Software-IOA") survives that.
    """
    frag = _norm(expected_fragment)
    return any(frag in _norm(d) for d in (actual_docs or []))


def _looks_abstained(answer: str) -> bool:
    return bool(_RX_ABSTAIN.search(answer or ""))


# ---------------------------------------------------------------------------
# tier: scope — zero cost
# ---------------------------------------------------------------------------

def _check_scope_case(case: dict, session_id: str) -> tuple[bool, list[str], dict]:
    """Resolve one case's scope and compare against its expectations.

    Deliberately tolerant about *how* the right documents were reached when the
    case doesn't pin a method: several resolvers can legitimately produce the
    same correct document set, and a case that over-specifies the route fails
    on a refactor that changed nothing a user would notice.
    """
    from services import wiki
    failures: list[str] = []
    if case.get("scope_resolved_by"):
        # Some paths reach their documents without resolve_scope at all: the
        # counting path counts from the document index, and the Calculation
        # Agent falls back to an identifier lookup when scope returns nothing.
        # Asserting resolve_scope for those reports a working feature as a
        # scope regression. The pipeline tier still checks the documents, from
        # files_used, which is where they actually appear.
        return True, [], {"skipped": f"scope tier n/a: {case['scope_resolved_by']}"}
    try:
        scope = wiki.resolve_scope(case["question"], session_id) or {}
    except Exception as e:
        return False, [f"resolve_scope raised {type(e).__name__}: {e}"], {}

    method = scope.get("method") or ""
    docs = sorted(scope.get("target_docs") or [])

    expected_method = case.get("expect_scope_method")
    if expected_method and method != expected_method:
        failures.append(f"scope_method: expected {expected_method!r}, got {method!r}")

    for frag in (case.get("expect_docs") or []):
        if not _doc_matches(docs, frag):
            failures.append(f"expected document not in scope: {frag!r}")

    return (not failures), failures, {"scope_method": method, "docs": docs}


# ---------------------------------------------------------------------------
# tier: pipeline — one real query per case
# ---------------------------------------------------------------------------

def _run_pipeline(question: str, session_id: str) -> dict:
    """Drive the real query graph and return its terminal payload.

    Uses the same generator app.py's /query route uses, so a case exercises the
    production path rather than a test-only reimplementation of it.

    Terminal detection is by PAYLOAD, not by stage name. The graph emits
    progress chunks that share their stage name with a terminal state —
    `{"stage": "disambiguation", "status": "active", "message": "Checking
    document scope…"}` is emitted on the way to a perfectly good answer, not
    instead of one. An earlier version of this function broke on the stage name
    alone and reported 8 of 9 cases as failing at disambiguation while the
    pipeline was in fact answering all of them correctly. A stream event is
    terminal when it carries the result, so that is what this waits for.
    """
    from services import intent_agent
    payload: dict = {}
    kind = ""
    for chunk in intent_agent.run_query_stream(
        question, session_id, exclude_cached_answers=True
    ):
        if not isinstance(chunk.get("payload"), dict):
            continue
        payload = chunk["payload"]
        kind = chunk.get("type") or chunk.get("stage") or ""
        break
    payload = dict(payload)
    payload["_terminal_kind"] = kind
    return payload


def _check_pipeline_case(case: dict, session_id: str) -> tuple[bool, list[str], dict]:
    failures: list[str] = []
    t0 = time.time()
    try:
        payload = _run_pipeline(case["question"], session_id)
    except Exception as e:
        logger.error("regression: pipeline raised on %r: %s", case.get("name"), e)
        return False, [f"pipeline raised {type(e).__name__}: {e}"], {}
    elapsed_ms = int((time.time() - t0) * 1000)

    answer = payload.get("answer") or ""
    docs = sorted(payload.get("files_used") or payload.get("scope_docs") or [])
    method = payload.get("scope_method") or ""

    # A case that expects an answer but got a disambiguation prompt has failed
    # in a way worth naming explicitly — the pipeline didn't err, it declined
    # to proceed, and that reads very differently in a results table. Keyed on
    # the terminal payload's own type, never on a stage name (see _run_pipeline).
    if payload.get("_terminal_kind") in ("disambiguation", "clarification"):
        failures.append(f"pipeline stopped at {payload['_terminal_kind']} instead of answering")
    elif not payload:
        failures.append("pipeline produced no terminal payload")

    if case.get("expect_abstain"):
        if not _looks_abstained(answer):
            failures.append("expected an abstention, answer asserted instead")
    else:
        if not answer.strip():
            failures.append("empty answer")

    for frag in (case.get("expect_docs") or []):
        if not _doc_matches(docs, frag):
            failures.append(f"expected document not read: {frag!r}")

    for needle in (case.get("must_contain") or []):
        if _norm(needle) not in _norm(answer):
            failures.append(f"missing required text: {needle!r}")

    for needle in (case.get("must_not_contain") or []):
        if _norm(needle) in _norm(answer):
            failures.append(f"contains forbidden text: {needle!r}")

    actual = {
        "scope_method": method,
        "docs": docs,
        "answer": answer,
        "total_ms": elapsed_ms,
    }
    return (not failures), failures, actual


# ---------------------------------------------------------------------------
# tier: graded — pipeline + LLM judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are grading one answer produced by a legal document \
retrieval system, against a reference answer that has already been verified \
against the source documents by hand.

QUESTION:
{question}

REFERENCE ANSWER (verified correct):
{expected}

SYSTEM ANSWER (grade this):
{actual}

The reference is a MINIMUM, not an exhaustive account. It is one or two
sentences written by hand; the system reads whole documents and legitimately
returns more — clause references, party names, dates, the filenames it read.
Detail beyond the reference that is consistent with it is CORRECT BEHAVIOUR
and must not be penalised on any axis. Penalise only what CONTRADICTS the
reference or could not have come from the documents at all.

Score three axes from 0 to 10:

- accuracy: does the system answer state the facts the reference states?
  Extra correct detail does not reduce this. A correct refusal, where the
  reference itself says the information is not established, scores 10.
  Contradicting the reference scores 0.
- relevance: does it answer the question that was asked, about the right
  document or the right corpus? Supporting quotes, breakdowns and lists of the
  documents relied on are not padding. Drifting to a different document or a
  different clause is.
- hallucination: 10 means nothing in the answer contradicts the reference and
  no specific looks invented. A figure, date, clause number or party name that
  the reference simply does not mention is NOT hallucination. Deduct only for
  a claim the reference contradicts, or a specific that could not plausibly
  come from the documents named. Vagueness is not hallucination; a wrong
  number is.

Reply with ONLY a JSON object, no other text:
{{"accuracy": <0-10>, "relevance": <0-10>, "hallucination": <0-10>, "note": "<one short sentence>"}}"""

# Below this on any axis, the case fails. Set where a lawyer would still call
# the answer usable: an 8 tolerates wording differences and missing nuance,
# a 7 usually means a real fact is wrong or missing.
GRADE_PASS_FLOOR = 8


# The judge's visible output is a JSON object of about sixty tokens, but on an
# Azure GPT-5.x deployment max_completion_tokens covers HIDDEN REASONING TOO.
# At 300 the reasoning routinely consumed the whole budget and the call came
# back with no visible content at all, which the parser could only report as
# "judge returned no JSON" — five of twenty-one graded cases in one run, none
# of them a product fault. Same failure, same fix as the grounding check, which
# was moved 900 -> 1500 for exactly this reason (see config.MAX_TOKENS_GROUNDING_CHECK).
_JUDGE_MAX_TOKENS = 1500


def _judge(question: str, expected: str, actual: str) -> dict:
    """One LLM call scoring an answer against a verified reference."""
    from services import llm
    import json as _json
    prompt = _JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)
    raw, _usage = llm.ask(prompt, pipeline="regression_judge",
                          max_tokens=_JUDGE_MAX_TOKENS)
    text = (raw or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"error": "judge returned no JSON", "raw": text[:400]}
    try:
        parsed = _json.loads(m.group(0))
    except Exception as e:
        return {"error": f"judge JSON unparseable: {e}", "raw": text[:400]}
    out = {}
    for axis in ("accuracy", "relevance", "hallucination"):
        try:
            out[axis] = float(parsed.get(axis))
        except (TypeError, ValueError):
            out[axis] = None
    out["note"] = str(parsed.get("note") or "")[:400]
    return out


def _check_graded_case(case: dict, session_id: str) -> tuple[bool, list[str], dict]:
    passed, failures, actual = _check_pipeline_case(case, session_id)
    expected = (case.get("expect_answer") or "").strip()
    if not expected:
        # A scope case asserts which documents were reached, and deliberately
        # carries no reference answer — there is nothing for a grader to score
        # it against. Failing it here reported three harness gaps as three
        # product regressions in the same list as real ones, which is the
        # fastest way to make a suite stop being read. It still ran the
        # pipeline tier above; that verdict stands.
        return passed, failures, actual

    scores = _judge(case["question"], expected, actual.get("answer") or "")
    actual["scores"] = scores
    if scores.get("error"):
        failures.append(f"judge failed: {scores['error']}")
        return False, failures, actual

    for axis in ("accuracy", "relevance", "hallucination"):
        val = scores.get(axis)
        if val is None:
            failures.append(f"judge returned no {axis} score")
        elif val < GRADE_PASS_FLOOR:
            failures.append(f"{axis} {val:.0f}/10 below floor {GRADE_PASS_FLOOR}")
    return (not failures), failures, actual


# ---------------------------------------------------------------------------
# run orchestration
# ---------------------------------------------------------------------------

_CHECKERS = {
    "scope": _check_scope_case,
    "pipeline": _check_pipeline_case,
    "graded": _check_graded_case,
}


def estimate_run(wiki_id: str, session_id: str, tier: str,
                 archetype: str | None = None) -> dict:
    """What a run would cost, before starting one.

    Call counts are per case and approximate — the query graph's own call count
    varies with which fast path a question takes (measured p50 is ~2.5 LLM calls
    per query on this corpus). Reported so a paid run is an explicit decision.
    """
    cases = db.get_regression_cases(wiki_id, session_id, archetype=archetype)
    n = len(cases)
    per_case = {"scope": 0.0, "pipeline": 2.5, "graded": 3.5}.get(tier, 0.0)
    return {
        "tier": tier,
        "cases": n,
        "llm_calls_per_case": per_case,
        "estimated_llm_calls": round(n * per_case),
        "free": per_case == 0.0,
    }


def run(wiki_id: str, session_id: str, tier: str = "scope",
        archetype: str | None = None, label: str | None = None,
        case_names: list[str] | None = None) -> dict:
    """Execute one regression run and store it.

    Never raises for a single case's failure: a case that blows up is recorded
    as a failing case and the run continues, because a suite that stops at the
    first error tells you about one problem instead of all of them.
    """
    if not _enabled():
        raise RuntimeError("Regression suite needs USE_DATABASE")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}, expected one of {TIERS}")

    cases = db.get_regression_cases(wiki_id, session_id, archetype=archetype)
    if case_names:
        wanted = {n.strip() for n in case_names if n.strip()}
        cases = [c for c in cases if c["name"] in wanted]
    if not cases:
        raise ValueError("no active regression cases matched")

    checker = _CHECKERS[tier]
    run_id = db.start_regression_run(wiki_id, session_id, tier, label=label,
                                     git_sha=_git_sha(), cases_total=len(cases))
    passed = failed = 0
    logger.info("regression: run %d starting — tier=%s cases=%d", run_id, tier, len(cases))
    try:
        for case in cases:
            try:
                ok, failures, actual = checker(case, session_id)
            except Exception as e:
                logger.error("regression: case %r raised: %s", case.get("name"), e)
                ok, failures, actual = False, [f"{type(e).__name__}: {e}"], {}
            db.record_regression_result(run_id, case, ok, failures, **actual)
            if ok:
                passed += 1
            else:
                failed += 1
                logger.info("regression: FAIL %s — %s", case["name"], "; ".join(failures))
        db.finish_regression_run(run_id, passed, failed)
    except Exception as e:
        db.finish_regression_run(run_id, passed, failed, status="error", error=str(e))
        raise

    logger.info("regression: run %d complete — %d passed, %d failed", run_id, passed, failed)
    return {"run_id": run_id, "tier": tier, "total": len(cases),
            "passed": passed, "failed": failed,
            "results": db.get_regression_results(run_id)}
