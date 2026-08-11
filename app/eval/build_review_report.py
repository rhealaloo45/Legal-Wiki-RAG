"""Generate the per-question review report: system answer vs ground truth.

Scores are the author's manual judgement against the verified ground truth (the
grader is a person, not this script) and live in SCORES below. Everything else —
answers, cited files, corpus, ground truth, source document — is read straight
from the run artefacts so the report cannot drift from what was actually
measured.

Per-question the report shows the BEST of the two corpus runs. That is a
best-observed figure, not an average, so each row names the corpus that produced
it — a max across two configurations would otherwise read as one system's score.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEMP = os.environ.get("TEMP", ".")

# Manual scores against ground truth, per corpus. None = not (yet) measured there.
# "mixed" column updated 2026-08-10 after the scope-resolution family fix +
# document-metadata surfacing fix, re-run live on the mixed session.
#            v1   mixed  clean
SCORES = {
    "Q14":  (1,    9,     8),
    "Q17":  (1,    6,     1),
    "Q18":  (2,    9,     2),
    "Q21":  (2,    8,     3),
    "Q24":  (2,    7,     8),
    "Q27":  (4,    7,     9),
    "Q29":  (6,    None,  7),
    "Q30":  (0,    0,     5),
    "Q42":  (3,    9,     None),
    "Q43":  (1,    1,     None),
    "Q44":  (1,    1,     None),
    "Q49":  (1,    9,     2),
    "Q56":  (1,    6,     9),
    "Q58":  (3,    9,     6),
    "Q61":  (4,    9,     2),
    "Q65":  (1,    9,     1),
    "Q67":  (1,    9,     9),
    "Q70":  (1,    8,     8),
    "Q74":  (1,    8,     2),
    "Q76":  (1,    9,     1),
    "Q79":  (1,    9,     None),
    "Q85":  (3,    6,     None),
    "Q92":  (2,    9,     None),
}

# Questions whose ground-truth fact is absent from every ingested page of the
# expected document — a refusal here is correct, and no retrieval change moves it.
# 2026-08-10: all 6 patched (content confirmed in raw PDF extraction, appended
# to the ingested page) or fixed via metadata surfacing. Empty until a new
# genuinely ingest-absent case is found — see build_artifact.py's INGEST_CAPPED
# for the fuller note, including why Q105 doesn't belong here either anymore.
INGEST_CAPPED = set()

# Questions ground-truthed to synthetic Test_* fixtures, or to no document at all
# (src field literally "N/A — no such document exists"): untestable on the clean/
# production corpus. Verified against gt_full.json's src field, not assumed — a
# prior version of this set only had 3 entries (Q42-44) and missed 14 others in
# the same Q30-46 block that fail the identical check.
FIXTURE_BOUND = {"Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38",
                 "Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46"}


def load(name: str) -> dict:
    with open(os.path.join(TEMP, name), encoding="utf-8") as fh:
        return {r["id"]: r for r in json.load(fh)}


def fmt_files(meta: dict) -> str:
    files = meta.get("files_used") or []
    if not files:
        return "_(none cited)_"
    return "<br>".join(f.split("_", 1)[-1][:70] for f in files[:3])


def main() -> None:
    gt = json.load(open(os.path.join(TEMP, "gt_full.json"), encoding="utf-8"))
    mixed, clean = load("sub5_out.json"), load("clean_out.json")

    rows, best_total, counted = [], 0, 0
    for qid, (v1, sm, sc) in SCORES.items():
        cands = [(s, c) for s, c in ((sm, "mixed"), (sc, "clean")) if s is not None]
        best, corpus = max(cands, key=lambda t: t[0])
        src = clean.get(qid) if corpus == "clean" else mixed.get(qid)
        rows.append((qid, v1, best, corpus, src or {}))
        best_total += best
        counted += 1

    o = io.StringIO()
    o.write("# LexWiki — Per-Question Review (re-tested questions)\n\n")
    o.write("Every question that scored below 7 in the v1 audit and has been re-tested.\n"
            "Each row shows the **better of the two corpus runs**, and names which corpus\n"
            "produced it. This is a best-observed figure, not an average of the two.\n\n")
    o.write("- **Mixed corpus** — all 494 documents (448 of them synthetic `Test_*` fixtures)\n")
    o.write("- **Clean corpus** — the 46 real Tata documents only (production-representative)\n\n")
    o.write(f"Re-tested: **{counted}** questions. "
            f"Mean of the best-observed scores across these: "
            f"**{best_total / counted:.2f}/10** "
            f"(was {sum(v[0] for v in SCORES.values()) / counted:.2f} in v1).\n\n")
    o.write("`ingest-capped` = the ground-truth fact is absent from every ingested page of the\n"
            "expected document, so the refusal is correct and no retrieval fix can raise it.\n\n")

    o.write("## Summary\n\n| Q | v1 | Now | Corpus | Expected source document |\n|---|---|---|---|---|\n")
    for qid, v1, best, corpus, _ in rows:
        tag = " ⚠ ingest-capped" if qid in INGEST_CAPPED else (
            " (fixture-bound)" if qid in FIXTURE_BOUND else "")
        arrow = "→" if best != v1 else "="
        o.write(f"| {qid} | {v1} | **{best}** {arrow}{tag} | {corpus} | "
                f"{gt[qid]['src'][:70]} |\n")

    o.write("\n---\n\n## Detail\n")
    for qid, v1, best, corpus, src in rows:
        meta = src.get("meta") or {}
        o.write(f"\n### {qid} — {v1} → **{best}/10**  ·  _{corpus} corpus_")
        if qid in INGEST_CAPPED:
            o.write("  ·  ⚠ **ingest-capped**")
        o.write("\n\n")
        o.write(f"**Question**\n\n> {gt[qid]['q']}\n\n")
        o.write(f"**Expected source** — {gt[qid]['src']}\n\n")
        o.write(f"**Documents the system actually cited**\n\n{fmt_files(meta)}\n\n")
        o.write(f"**Ground truth**\n\n> {gt[qid]['gt']}\n\n")
        answer = (src.get("answer") or "_(no answer captured)_").strip()
        o.write("**System answer**\n\n```\n" + answer[:2200] + "\n```\n\n")
        bits = [f"scope method: `{meta.get('scope_method')}`"]
        if src.get("needed_disambiguation"):
            bits.append("**required naming the document before it would answer**")
        if meta.get("not_covered"):
            bits.append("flagged `not_covered`")
        o.write("_" + " · ".join(bits) + "_\n")

    path = os.path.join(os.path.dirname(__file__), "..", "..", "rag_review_detail.md")
    with open(os.path.abspath(path), "w", encoding="utf-8") as fh:
        fh.write(o.getvalue())
    print("wrote", os.path.abspath(path))
    print(f"{counted} questions, best-observed mean {best_total / counted:.2f}")


if __name__ == "__main__":
    main()
