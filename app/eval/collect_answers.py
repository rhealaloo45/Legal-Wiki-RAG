"""Assemble one answer per ground-truth question, for the full-set report.

Re-tested questions take their answer from this round's harness runs. Every other
question's answer comes from the original UI run, recovered out of chat_messages
by matching the stored user turn back to the ground-truth question text.

Matching is on a normalised character ratio rather than equality: the UI run
often appended a document name to break a disambiguation prompt ("... (NDA 5)"),
so the stored turn is a superset of the question as written in the audit set.
"""
import difflib
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sqlalchemy import text

from services import db

TEMP = os.environ.get("TEMP", ".")

# The v1 audit's own sessions — one per 20-question batch, run 2026-08-10
# 11:31–15:01. Restricting to these is not tidiness: chat_messages also holds the
# previous audit and older broken runs, and searching all of them matched a
# GridEdge question to an OmniRetail answer and another to a stored
# "LLM unavailable: 404" at ratio 1.00.
AUDIT_SESSION_WINDOW = ("2026-08-10 11:00", "2026-08-10 15:30")

# High floor for the same reason. A question that cannot be matched confidently
# is reported as unrecovered; showing a near-miss answer under the wrong question
# is worse than showing none.
MATCH_FLOOR = 0.85

DISAMBIG_PROMPT = "Which agreement are you asking about?"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())[:400].strip()


def main() -> None:
    gt = json.load(open(os.path.join(TEMP, "gt_full.json"), encoding="utf-8"))
    qids = sorted([q for q in gt if q.startswith("Q")], key=lambda s: int(s[1:]))[:105]

    answers = {}
    for fname, tag in (("sub5_out.json", "mixed"), ("clean_out.json", "clean")):
        path = os.path.join(TEMP, fname)
        if not os.path.exists(path):
            continue
        for r in json.load(open(path, encoding="utf-8")):
            answers.setdefault(r["id"], {})[tag] = r

    engine = db.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT session_id, role, content, created_at
            FROM chat_messages
            WHERE created_at >= :lo AND created_at <= :hi
            ORDER BY session_id, created_at, id
        """), {"lo": AUDIT_SESSION_WINDOW[0], "hi": AUDIT_SESSION_WINDOW[1]}).fetchall()

    # Pair each user turn with the assistant turn that followed it. When that
    # reply is the "which agreement?" prompt, the real answer is the one after
    # the user names the document — recording the prompt would score the run as
    # having refused a question it went on to answer.
    pairs = []
    for i, r in enumerate(rows):
        if r.role != "user":
            continue
        answer = ""
        for nxt in rows[i + 1:i + 6]:
            if nxt.session_id != r.session_id:
                break
            if nxt.role != "assistant":
                continue
            if DISAMBIG_PROMPT in (nxt.content or ""):
                continue  # keep walking to the resolved reply
            answer = nxt.content
            break
        if answer:
            pairs.append((norm(r.content), r.content, answer))

    recovered, missing = 0, []
    out = {}
    for qid in qids:
        entry = {"question": gt[qid]["q"], "gt": gt[qid]["gt"], "src": gt[qid]["src"]}
        if qid in answers:
            pick = answers[qid].get("clean") or answers[qid].get("mixed")
            entry["answer"] = pick.get("answer", "")
            entry["files"] = (pick.get("meta") or {}).get("files_used") or []
            entry["origin"] = "retest"
        else:
            target = norm(gt[qid]["q"])
            best, score = None, 0.0
            for nq, _raw, ans in pairs:
                s = difflib.SequenceMatcher(None, target, nq).ratio()
                if s > score:
                    best, score = ans, s
            if best and score >= MATCH_FLOOR:
                entry["answer"] = best
                entry["files"] = []
                entry["origin"] = f"original run (match {score:.2f})"
                recovered += 1
            else:
                entry["answer"] = ""
                entry["files"] = []
                entry["origin"] = "not recovered"
                missing.append(qid)
        out[qid] = entry

    json.dump(out, open(os.path.join(TEMP, "full_report_data.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"re-tested answers : {len(answers)}")
    print(f"recovered from DB : {recovered}")
    print(f"NOT recovered     : {len(missing)} -> {', '.join(missing) if missing else '(none)'}")


if __name__ == "__main__":
    main()
