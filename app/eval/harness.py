"""Scripted query harness — runs questions through the pipeline without the browser.

Mirrors what POST /query does (see app.query_route): same run_query_stream call,
same exclude_cached_answers flag the UI's "Testing mode" checkbox sets.

Usage:
    python app/eval/harness.py chains.json out.json
"""
import io
import json
import logging
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# app.py configures this at INFO when the server runs; the harness never
# imports app.py, so without this the root logger sits at its default WARNING
# and every logger.info() diagnostic in the pipeline — refusal-recheck firing,
# scope decisions, retry outcomes — is silently dropped from harness runs.
# Confirmed: grepped every harness log from an eval session for "recheck" and
# found nothing, which looked like the mechanism never firing; it was firing,
# the log just never carried it.
logging.basicConfig(level=logging.INFO, format="%(message)s")

WIKI_SESSION = os.getenv("HARNESS_WIKI_SESSION", "3a66b0ab-a9cc-48f0-a3f3-b0ab863936fe")


def ask(question: str, chat_session_id: str) -> dict:
    from services import intent_agent

    result = {"question": question, "type": None, "answer": "", "meta": {}}
    t0 = time.time()
    for ev in intent_agent.run_query_stream(
        question, WIKI_SESSION, "", False, True, chat_session_id=chat_session_id
    ):
        etype = ev.get("type")
        if etype in ("answer", "disambiguation", "clarification"):
            payload = ev.get("payload", {}) or {}
            result["type"] = etype
            result["answer"] = payload.get("answer") or ev.get("message", "")
            result["meta"] = {
                k: payload.get(k)
                for k in (
                    "scope_docs", "scope_method", "files_used", "not_covered",
                    "confidence_score", "general_knowledge", "general_knowledge_note",
                    "context_warning", "validation",
                )
            }
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def run_chain(chain: dict) -> list[dict]:
    """A chain is {id, setup: [...], target: "question", doc_hint: "NDA 5"}.

    Setup questions run first in the same chat session so scope-carryover state
    matches the original UI run; only the target's result is reported.

    A disambiguation prompt is answered by naming the document, exactly as the
    original UI run did — otherwise a question that merely ASKS which document
    would score against a run where it was told, and the two are not comparable.
    The reply is recorded so a resolved answer is never mistaken for a direct one.
    """
    sid = f"harness-{uuid.uuid4()}"
    for q in chain.get("setup", []):
        ask(q, sid)
    out = ask(chain["target"], sid)
    if out["type"] == "disambiguation" and chain.get("doc_hint"):
        out = ask(f'{chain["target"]} ({chain["doc_hint"]})', sid)
        out["needed_disambiguation"] = True
    out["id"] = chain["id"]
    return [out]


def main() -> None:
    chains = json.load(open(sys.argv[1], encoding="utf-8"))
    results = []
    for chain in chains:
        print(f"[{chain['id']}] {chain['target'][:70]}...", flush=True)
        try:
            results.extend(run_chain(chain))
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append({"id": chain["id"], "error": str(e)})
        json.dump(results, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nWrote {len(results)} results to {sys.argv[2]}")


if __name__ == "__main__":
    main()
