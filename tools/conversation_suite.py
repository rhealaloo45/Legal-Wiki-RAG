"""Scripted multi-turn conversations, each turn asserting what it must do.

The companion to thread_suite.py, which asks a different question. That suite
generates threads from single-document seeds and checks one thing: does a
follow-up stay on the document the thread pinned. It found drift, which is a
real failure and not the only one.

This suite exists because three bugs reached a user's session that a
48-question benchmark had scored 9.1 on, and every one of them was invisible to
that benchmark for the same structural reason: it starts a fresh thread per
question, so nothing it runs is ever a second turn.

  * A corpus-wide question asked third in a thread skipped the expiry index
    entirely — every deterministic branch was gated on the turn not being a
    follow-up — was answered by retrieval over the twelve documents the
    conversation happened to be holding, and opened by repeating the PREVIOUS
    answer. 49,210 tokens to get wrong what the index answers exactly, free.
  * A question naming a new subject inherited the pinned document and answered
    about a confidentiality agreement having nothing to do with the subject.
  * A count over a text predicate answered "fifteen" where the corpus holds 701.

None of those is drift. The document did not change under a follow-up; the
wrong thing happened for reasons a drift check cannot see — a branch not
reached, a scope wrongly kept, an answer bleeding into the next one.

So the assertions here are per turn and explicit: which path answered it,
whether it cost anything, what the text must and must not say, and which
document it must still be about. A scenario is a script, not a generator,
because the bugs were sequence-dependent and a generator cannot state what
turn three is supposed to do.

Turns marked free=True must cost zero tokens. That is not a performance note:
every one of them is a question the index answers exactly, and a turn that
starts costing tokens has stopped reaching the index and is about to start
being wrong.

    python tools/conversation_suite.py              run everything
    python tools/conversation_suite.py --list       show the scenarios
    python tools/conversation_suite.py --free-only  run only zero-cost turns
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

os.environ.pop("ENABLE_ENQUIRY_AGENT", None)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_HERE), "app", ".env"))
except Exception:
    pass

import logging
logging.basicConfig(level=logging.ERROR)

SID = os.environ.get("WIKI_SESSION_ID", "57983304-3d63-40dc-bbd9-c9ff55a75232")
OUT = os.path.join(_HERE, "conversation_suite_results.json")

from services import db as DB                    # noqa: E402
from services import intent_agent as IA          # noqa: E402
from services import tracing                     # noqa: E402


# --- the scenarios -----------------------------------------------------------
#
# Per turn:
#   q         the question
#   path      substring the answering method must contain ("" = don't care)
#   free      True  -> must cost zero tokens (it is an index lookup)
#             False -> may cost tokens
#             None  -> don't care
#   must      substrings the answer must contain (case-insensitive)
#   must_not  substrings the answer must NOT contain
#   holds     substring that must appear among the documents the turn used
#
# Every must_not below is a defect that actually shipped, not a hypothetical.

# The scenarios name real documents and parties, so they live in
# tools/conversation_scenarios_local.py, which is not committed. See
# conversation_scenarios_local.example.py for the format.
try:
    sys.path.insert(0, _HERE)
    from conversation_scenarios_local import SCENARIOS
except ImportError:
    SCENARIOS = []
# The first sentence of one answer turning up at the head of the next is the
# contamination signature, and it is checked on every turn rather than declared
# per scenario — it was not predicted anywhere, it was just noticed.
def _first_sentence(text):
    s = re.split(r"(?<=[.!?])\s", (text or "").strip(), maxsplit=1)
    return re.sub(r"\s+", " ", (s[0] if s else "")).strip().lower()


def ask(question, chat, sid=SID):
    """One turn, persisted, so the NEXT turn has a conversation to inherit.

    Recording is not bookkeeping — it is the thing under test. Carryover reads
    the thread out of chat_messages, so a harness that does not write there
    runs every turn as though it were the first. The first version of this file
    did exactly that: the follow-ups came back "Needs document selection", and
    the scenario asserting that a new subject DROPS the inherited document
    passed while inheriting nothing. A green suite that never built a
    conversation is worse than no suite, because thread_suite.py already
    records for this reason and the omission looked deliberate.
    """
    trace, token = tracing.start_trace(question, chat, sid)
    payload, t0 = {}, time.time()
    try:
        for ev in IA.run_query_stream(question, sid, "", False, True,
                                      chat_session_id=chat):
            if ev.get("type") in ("answer", "disambiguation", "clarification"):
                payload = ev.get("payload") or {}
                if not payload.get("answer"):
                    payload["answer"] = ev.get("message", "")
    except Exception as e:
        payload = {"answer": "[ERROR] %s: %s" % (type(e).__name__, e)}
    finally:
        d = trace.to_dict()
        try:
            tracing._current.reset(token)
        except Exception:
            pass

    answer = payload.get("answer") or ""
    try:
        DB.insert_message(chat, "user", question, "text")
        DB.insert_message(chat, "assistant", answer, "answer", {
            "files_used": payload.get("files_used", []),
            "intent": payload.get("intent", "factual"),
            "scope_method": payload.get("scope_method", ""),
            "scope_docs": payload.get("scope_docs", [])})
    except Exception as e:
        print("        ! could not persist turn: %s" % e, flush=True)

    return {
        "answer": answer,
        "method": payload.get("intent_method") or payload.get("scope_method") or "",
        "docs": [str(x) for x in (payload.get("scope_docs")
                                  or payload.get("files_used") or [])],
        "tokens": d.get("total_prompt_tokens", 0) + d.get("total_completion_tokens", 0),
        "elapsed_s": round(time.time() - t0, 1),
    }


def check(turn, got, prev_answer):
    """Every assertion this turn makes. Returns a list of failure strings."""
    bad = []
    ans, low = got["answer"], got["answer"].lower()

    if turn.get("path") and turn["path"].lower() not in (got["method"] or "").lower():
        bad.append("path %r, wanted %r" % (got["method"] or "-", turn["path"]))
    if turn.get("path_not") and turn["path_not"].lower() in (got["method"] or "").lower():
        bad.append("path %r must not be %r" % (got["method"], turn["path_not"]))

    if turn.get("free") is True and got["tokens"] != 0:
        bad.append("cost %d tokens, must be free" % got["tokens"])
    if turn.get("free") is False and got["tokens"] == 0:
        bad.append("cost nothing — expected a model call, so it likely took a "
                   "fast path it should not have")

    for s in turn.get("must", []):
        if s.lower() not in low:
            bad.append("missing %r" % s)
    for s in turn.get("must_not", []):
        if s.lower() in low:
            bad.append("contains %r" % s)

    if turn.get("holds"):
        if not any(turn["holds"].lower() in d.lower() for d in got["docs"]):
            bad.append("lost its document: wanted %r among %s"
                       % (turn["holds"],
                          [d.rsplit("_", 1)[-1][:34] for d in got["docs"][:3]] or "nothing"))

    # Unpredicted, and the one that actually shipped.
    fs = _first_sentence(ans)
    if fs and len(fs) > 40 and fs == _first_sentence(prev_answer):
        bad.append("repeats the previous answer's opening sentence")
    return bad


def main():
    if "--list" in sys.argv:
        for s in SCENARIOS:
            print("\n%s\n  (%s)" % (s["name"], s["why"]))
            for i, t in enumerate(s["turns"], 1):
                print("   %d. %s" % (i, t["q"][:88]))
        return
    free_only = "--free-only" in sys.argv

    results, failures, checked, t0 = [], 0, 0, time.time()
    for s in SCENARIOS:
        chat = "convsuite-" + str(uuid.uuid4())[:8]
        print("\n%s" % s["name"], flush=True)
        rec, prev = {"name": s["name"], "turns": []}, ""
        for i, turn in enumerate(s["turns"], 1):
            if free_only and turn.get("free") is not True:
                # Still has to be ASKED — the turn under test is a follow-up,
                # and skipping the turns before it removes the thing being
                # tested. Only the assertion is skipped.
                got = ask(turn["q"], chat)
                prev = got["answer"]
                print("   %d. (context) %s" % (i, turn["q"][:66]), flush=True)
                continue
            got = ask(turn["q"], chat)
            bad = check(turn, got, prev)
            prev = got["answer"]
            checked += 1
            failures += len(bad)
            rec["turns"].append({"q": turn["q"], "method": got["method"],
                                 "tokens": got["tokens"],
                                 "elapsed_s": got["elapsed_s"],
                                 "failures": bad,
                                 "answer": got["answer"][:600]})
            print("   %d. %-4s %7s tok %5.1fs  %s" % (
                i, "OK" if not bad else "FAIL", format(got["tokens"], ","),
                got["elapsed_s"],
                turn["q"][:56]), flush=True)
            for b in bad:
                print("        - %s" % b, flush=True)
        results.append(rec)
        io.open(OUT, "w", encoding="utf-8").write(
            json.dumps(results, ensure_ascii=False, indent=1))

    print("\n%d turns asserted, %d failures, %.1f min"
          % (checked, failures, (time.time() - t0) / 60))
    print("RESULT: %s" % ("PASS" if not failures else "%d FAILED" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
