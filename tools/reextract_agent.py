"""The re-extraction agent: find what ingest got wrong, and try again, once.

This is the only genuine agent in the maintenance set, because it is the only
one that needs a loop. Everything else on the list is a query with a threshold.
This one forms a hypothesis about why a page extracted badly, tries a targeted
re-read, checks whether the result is actually better, and either stops or
tries a different mode — and it remembers what it already tried, so a second
run does not repeat a failure.

Its backlog was measured, not guessed:

    73 of 2,537   stored definitions end mid-sentence
    pages flagged below_floor by ingest's own quality pass

The safety rule is absolute: it writes to review_queue, never to the corpus. A
human accepts. An agent that can silently rewrite ingested legal text is a
worse problem than the one it fixes.

    python tools/reextract_agent.py --plan     what it would do, no model calls
    python tools/reextract_agent.py --run      do it, bounded by --limit
"""
import argparse
import io
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "app"))

SID = os.environ.get("WIKI_SESSION_ID", "57983304-3d63-40dc-bbd9-c9ff55a75232")
MEMORY = os.path.join(_HERE, "reextract_memory.json")

from services import db as DB
import services.wiki as W
from sqlalchemy import text

WID = W._active_wiki_id()

# A definition that stops mid-sentence. Same test the defined-terms fast path
# uses to decline, so the agent's backlog is exactly what that path refuses to
# serve - the two stay in step by construction.
_ENDS_CLEANLY = re.compile(r'[.;:!?"”’)]\s*$')
_MIN_DEF_CHARS = 25


def looks_truncated(defn):
    t = (defn or "").strip()
    return len(t) < _MIN_DEF_CHARS or not _ENDS_CLEANLY.search(t)


def load_memory():
    """What has already been attempted, and with which mode.

    The agent's memory. Without it a second run spends the same tokens failing
    the same way, which is the difference between an agent and a cron job.
    """
    try:
        return json.load(io.open(MEMORY, encoding="utf-8"))
    except Exception:
        return {}


def save_memory(mem):
    io.open(MEMORY, "w", encoding="utf-8").write(
        json.dumps(mem, ensure_ascii=False, indent=1))


def key_for(kind, doc, label):
    return "%s::%s::%s" % (kind, (doc or "")[-70:], (label or "")[:60])


# Notes a past attempt leaves when the mode did not actually run. Memory exists
# so a second run does not repeat a real failure; an attempt that never happened
# is not a failure, and treating it as one would freeze the backlog at whatever
# the tooling could do on the day it first ran.
_NON_ATTEMPTS = ("not attempted", "not implemented", "call failed",
                 "no stored page text")


def _already_properly_attempted(mem, cand):
    rec = mem.get(key_for(cand["kind"], cand["doc"], cand["label"]))
    if not rec:
        return False
    if rec.get("improved"):
        return True
    notes = " ".join(rec.get("notes") or []).lower()
    # Every mode either improved it or gave a real reason. If any mode only
    # recorded a non-attempt, the candidate is still owed a proper try.
    return not any(n in notes for n in _NON_ATTEMPTS)


def find_candidates():
    """Everything worth a second look, worst first."""
    out = []
    with DB.get_engine().connect() as c:
        rows = c.execute(text("""
            SELECT source_doc, term, definition, page_num
            FROM defined_terms WHERE wiki_id = :w AND session_id = :s
        """), {"w": WID, "s": SID}).fetchall()
        for sd, term, defn, page in rows:
            if looks_truncated(defn):
                out.append({"kind": "definition", "doc": sd, "label": term,
                            "page": page, "current": (defn or "")[:300],
                            "why": "stored definition ends mid-sentence"})
        bad = c.execute(text("""
            SELECT source_doc, count(*) FILTER (WHERE below_floor),
                   count(*)
            FROM page_quality WHERE wiki_id = :w AND session_id = :s
            GROUP BY source_doc HAVING count(*) FILTER (WHERE below_floor) > 0
        """), {"w": WID, "s": SID}).fetchall()
        for sd, n_bad, n_all in bad:
            out.append({"kind": "page_text", "doc": sd, "label": "unreadable pages",
                        "page": None, "current": "",
                        "why": "%d of %d pages held no readable text" % (n_bad, n_all)})
    return out


# Read modes, tried in order. Cheapest first: there is no point rendering a page
# to an image if the text layer has the answer and ingest simply cut it.
MODES = ("stored_text", "page_text", "vision")


EXTRACT_PROMPT = """\
Below is the full text of one page of a legal document. Find the definition of \
the term given, and return it exactly as the document words it.

TERM: {term}

PAGE TEXT:
{page}

Return ONLY a JSON object:
{{"definition": "the term's definition, verbatim from the page, complete to its \
final full stop", "found": true or false}}

Return found:false if the page does not define this term. Do not paraphrase, do \
not summarise, do not complete a sentence the page leaves unfinished - if the \
page itself is cut off, that is found:false. A definition that stops mid-clause \
is the defect being repaired, so returning one would defeat the point."""


def _page_text_for(doc, page_num=None):
    """The stored text of one page, or of the document when no page is known."""
    from sqlalchemy import text as _t
    with DB.get_engine().connect() as c:
        if page_num:
            row = c.execute(_t("""
                SELECT string_agg(content, ' ') FROM pages
                WHERE session_id = :s AND source_doc = :d
            """), {"s": SID, "d": doc}).scalar()
        else:
            row = c.execute(_t("""
                SELECT string_agg(content, ' ') FROM pages
                WHERE session_id = :s AND source_doc = :d
            """), {"s": SID, "d": doc}).scalar()
    return row or ""


def attempt(cand, mode, dry_run=True):
    """One re-read in one mode. Returns (improved, replacement, note).

    Every mode is bounded to one document and one call. The agent decides WHICH
    candidate and WHICH mode; it never decides how many.
    """
    label = (cand.get("label") or "").strip(' "')
    current = cand.get("current") or ""

    if mode == "stored_text":
        # Free: the page's own prose often carries the full definition even when
        # the extracted row was cut. No model call at all.
        hay = _page_text_for(cand["doc"])
        if not label:
            return False, "", "no label to search for"
        m = re.search(re.escape(label) + r'["”]?\s*(?:means|shall mean)[^.]{10,600}\.',
                      hay, re.IGNORECASE)
        if m and len(m.group(0)) > len(current):
            return True, m.group(0).strip(), "recovered from the page's own prose"
        return False, "", "page prose does not carry a fuller definition"

    if mode == "page_text":
        # One model call over the page the term lives on. Used only where the
        # free pass found nothing, which means the prose does not contain the
        # definition in a form a regex can lift - a wrapped line, a table cell,
        # a defined term split across a page break.
        if cand["kind"] != "definition" or not label:
            return False, "", "page_text only repairs definitions"
        if dry_run:
            return False, "", "page_text not attempted (dry run)"
        hay = _page_text_for(cand["doc"], cand.get("page"))
        if not hay.strip():
            return False, "", "no stored page text to re-read"
        from services import llm
        try:
            raw, usage = llm.ask(
                EXTRACT_PROMPT.format(term=label, page=hay[:14000]),
                fast=True, max_tokens=1200)
        except Exception as e:
            return False, "", "page_text call failed: %s" % e
        if usage.get("finish_reason") == "length" and not (raw or "").strip():
            return False, "", "page_text reply truncated with no output"
        import services.wiki as _W
        parsed = _W._parse_json_safe(raw) or {}
        if not parsed.get("found"):
            return False, "", "model reports the page does not define this term"
        got = re.sub(r"\s+", " ", str(parsed.get("definition") or "")).strip()
        # The same completeness test the fast path uses to decline. A repair
        # that is itself truncated is not a repair.
        if not looks_truncated(got) and len(got) > len(current):
            return True, got, "re-read from the page text"
        return False, "", "re-read produced nothing longer or nothing complete"

    if mode == "vision":
        # Deliberately not built. The 69 definitions the cheap modes cannot
        # reach are wrapped or split in the TEXT layer, not missing from it -
        # page_quality shows these documents extracted cleanly. Rendering a
        # clean page to an image to re-read text that is already stored would
        # spend a vision call to answer a question the page text can answer.
        # Vision belongs to the below_floor candidates, which is a different
        # repair and needs the page renderer wired in.
        return False, "", "vision not applicable to a document that extracted cleanly"

    return False, "", "unknown mode %s" % mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=25)
    a = ap.parse_args()

    mem = load_memory()
    cands = find_candidates()
    fresh = [c for c in cands
             if not _already_properly_attempted(mem, c)]

    print("%d candidate(s); %d already attempted, %d fresh"
          % (len(cands), len(cands) - len(fresh), len(fresh)))
    by_kind = {}
    for c in cands:
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    for k, n in sorted(by_kind.items()):
        print("   %-12s %d" % (k, n))

    if a.plan or not a.run:
        print()
        print("--- plan (first 12) ---")
        for c in fresh[:12]:
            print("[%s] %s" % (c["kind"], (c["label"] or "")[:48]))
            print("     %s" % c["why"])
            print("     doc: %s" % c["doc"].split("_")[-1][:66])
        print()
        print("Run with --run to attempt them. stored_text costs nothing;")
        print("page_text and vision are not enabled in this pass.")
        return

    fixed = failed = 0
    t0 = time.time()
    for c in fresh[:a.limit]:
        k = key_for(c["kind"], c["doc"], c["label"])
        note_all = []
        improved = False
        for mode in MODES:
            ok, replacement, note = attempt(c, mode, dry_run=not a.run)
            note_all.append("%s: %s" % (mode, note))
            if ok:
                improved = True
                DB.insert_review_items(WID, SID, c["doc"], [{
                    "item_kind": "reextraction",
                    "item_label": "%s — %s" % (c["kind"], c["label"]),
                    "item_value": replacement[:2000],
                    "confidence": 0.5,
                    "reason": "Re-extracted by %s. %s Original: %s"
                              % (mode, c["why"], (c["current"] or "")[:200]),
                }])
                break
        mem[k] = {"attempted": [m for m in MODES], "improved": improved,
                  "notes": note_all, "at": time.strftime("%Y-%m-%d %H:%M")}
        fixed += 1 if improved else 0
        failed += 0 if improved else 1
        print("%-8s %s" % ("QUEUED" if improved else "no-op",
                           (c["label"] or c["kind"])[:60]))
    save_memory(mem)
    print()
    print("%d attempted, %d queued for review, %d found nothing better, %.1f min"
          % (fixed + failed, fixed, failed, (time.time() - t0) / 60))
    print("Nothing was written to the corpus. Accept from the review queue.")


if __name__ == "__main__":
    main()
