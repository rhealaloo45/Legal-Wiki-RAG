"""Generate multi-turn threads from single-document questions, and check that
the conversation stays on the document it started on.

The failure this exists for needed a thread to reproduce. A four-turn thread on
one Consultancy Agreement resolved correctly on turn 1 and then answered from
three different documents belonging to two other companies - three independent
bugs, every one of them silent on turn 1, and every one invisible to a
200-question set of independent questions. No amount of single-question testing
would have found them.

The assertion is deliberately narrow. It does not judge the answer: it checks
which documents the turn resolved to. That is a field the pipeline already
records, so checking costs no model call, and it is the thing that actually
broke. An answer can be poor for many reasons; a follow-up that silently
changes document is wrong for one reason, and that reason is fixable.

    python tools/thread_suite.py            run every generated thread
    python tools/thread_suite.py --list     show the threads without running
"""
import io
import json
import os
import re
import sys
import time
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "app"))

SID = os.environ.get("WIKI_SESSION_ID", "57983304-3d63-40dc-bbd9-c9ff55a75232")
OUT = os.path.join(_HERE, "thread_suite_results.json")

from services import db as DB
from services import intent_agent as IA
import services.wiki as W

# Follow-up shapes, each aimed at one resolver that has failed in production.
#
#   demonstrative  "in this agreement" - the type word reads as a pivot to a new
#                  document type unless something recognises the demonstrative
#   ordinal        "the second one" - carries no subject of its own; the rewrite
#                  that gives it one must not also become the retrieval query
#   conjunction    "And what happens if..." - a capitalised sentence-opening
#                  word is not a proper noun, though a Title-Case regex says so
#   bare           "What about termination?" - no document reference at all
FOLLOW_UPS = [
    ("demonstrative", "What are the biggest risks in this agreement?", False),
    ("ordinal", "Tell me more about the second one.", True),
    ("conjunction", "And what happens if they do not cure the breach in time?", False),
    ("bare", "What about termination?", False),
]

# An ordinal follow-up only means something if the previous answer enumerated
# things to point at.
_RX_ENUMERATED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+\S", re.M)


def seed_questions(limit=None):
    """Regression cases that name exactly one document — the only usable seeds.

    A thread has to start somewhere unambiguous: if turn 1 does not pin one
    document, a later turn changing document proves nothing.
    """
    cases = DB.get_regression_cases(W._active_wiki_id(), SID)
    seeds, seen = [], set()
    for c in cases:
        q = (c.get("question") or "").strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        # A corpus-wide question is not a thread seed. It resolves to one
        # document only by accident - "how many SLAs do we have" came back
        # pinned to a single file through carryover - and a follow-up that then
        # changes document proves nothing about conversation handling.
        if W._BROAD_SCOPE_RE.search(q) or W._PLURAL_FAMILY_HINT_RE.search(q):
            continue
        try:
            sc = W.resolve_scope(q, SID)
        except Exception:
            continue
        docs = sc.get("target_docs") or []
        method = sc.get("method") or ""
        # A seed must resolve from its OWN text. A question that only pins a
        # document by inheriting the previous turn's scope is not a starting
        # point - "how many SLAs do we have in our corpus" resolves that way and
        # is a corpus question, so a later turn changing document would prove
        # nothing about how the conversation is handled.
        if method.startswith("carryover"):
            continue
        if len(docs) == 1 and method:
            seeds.append({"question": q, "doc": docs[0], "method": method,
                          "name": c.get("name") or ""})
        if limit and len(seeds) >= limit:
            break
    return seeds


def run_thread(seed, record=True):
    """One thread. Returns a turn-by-turn record with the scope assertion."""
    chat = "threadsuite-" + str(uuid.uuid4())[:8]
    turns, prev_answer = [], ""
    pinned = seed["doc"]

    def ask(q):
        p = None
        try:
            for ev in IA.run_query_stream(q, SID, exclude_cached_answers=True,
                                          chat_session_id=chat):
                if isinstance(ev.get("payload"), dict) and ev.get("type") in (
                        "answer", "disambiguation", "clarification"):
                    p = ev["payload"]
                    break
        except Exception as e:
            p = {"answer": "[ERROR] %s: %s" % (type(e).__name__, e)}
        return p or {}

    plan = [("seed", seed["question"], False)] + list(FOLLOW_UPS)
    for shape, q, needs_list in plan:
        if needs_list and not _RX_ENUMERATED.search(prev_answer or ""):
            turns.append({"shape": shape, "question": q, "skipped":
                          "previous answer enumerated nothing to point at"})
            continue
        p = ask(q)
        answer = p.get("answer") or p.get("message") or ""
        docs = p.get("scope_docs") or p.get("files_used") or []
        # A turn that resolved NOTHING is not a pass. It did not drift, but it
        # also did not stay - and counting it as held lets a turn that quietly
        # answered from the whole corpus look like a turn that held its
        # document. Reported as its own state so it can neither fail the suite
        # falsely nor hide in it.
        if not docs:
            state = "unresolved"
        elif pinned in docs:
            state = "held"
        else:
            state = "drifted"
        turns.append({
            "shape": shape, "question": q,
            "scope_method": p.get("scope_method", ""),
            "docs": [str(d) for d in docs][:4],
            "state": state,
            "held": state != "drifted",
            "confidence": (p.get("confidence_six") or {}).get("value"),
            "chars": len(answer),
        })
        if record:
            DB.insert_message(chat, "user", q, "text")
            DB.insert_message(chat, "assistant", answer, "answer", {
                "files_used": p.get("files_used", []),
                "intent": p.get("intent", "factual"),
                "scope_method": p.get("scope_method", ""),
                "scope_docs": p.get("scope_docs", [])})
        prev_answer = answer
    return {"seed": seed, "chat_session": chat, "turns": turns}


def main():
    if "--list" in sys.argv:
        for s in seed_questions():
            print("%-26s %s" % (s["method"], s["question"][:96]))
        return
    seeds = seed_questions()
    print("%d single-document seeds -> up to %d threaded assertions"
          % (len(seeds), len(seeds) * (1 + len(FOLLOW_UPS))), flush=True)
    results, failed, checked, unres = [], 0, 0, 0
    t0 = time.time()
    for i, s in enumerate(seeds, 1):
        r = run_thread(s)
        results.append(r)
        io.open(OUT, "w", encoding="utf-8").write(
            json.dumps(results, ensure_ascii=False, indent=1))
        bad = [t for t in r["turns"] if t.get("state") == "drifted"]
        unres += sum(1 for t in r["turns"] if t.get("state") == "unresolved")
        checked += sum(1 for t in r["turns"] if not t.get("skipped"))
        failed += len(bad)
        flag = "DRIFT" if bad else "ok"
        print("[%2d/%2d] %-5s %s" % (i, len(seeds), flag, s["question"][:78]), flush=True)
        for t in bad:
            print("        %-14s %-22s -> %s" % (
                t["shape"], t["scope_method"] or "-",
                ", ".join(d.split("_")[-1][:40] for d in t["docs"]) or "nothing"), flush=True)
    print()
    print("%d turns checked, %d drifted, %d resolved nothing, %.1f min"
          % (checked, failed, unres, (time.time() - t0) / 60))
    print("RESULT: %s" % ("PASS" if failed == 0 else "%d DRIFTING TURNS" % failed))


if __name__ == "__main__":
    main()
