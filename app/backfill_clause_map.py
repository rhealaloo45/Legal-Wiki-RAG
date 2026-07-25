"""Backfill the clause_map table from the original source PDFs.

Ingest builds wiki page titles from each clause's heading but strips the
leading number — "5. Return, Destruction, and Record Retention" is stored as
"Return, Destruction, and Record Retention – Tata-NordForge (NDA)" — which
leaves ask-by-clause-number questions unanswerable even though the source is
numbered and the content is in the DB. This script restores the link without
re-ingesting anything:

  1. For every real (non-Test_) source_doc in a session, open the original PDF
     from data/uploads and extract its numbered headings.
  2. Match each heading to the session's stored page titles for that document —
     first by title prefix (the title IS the heading minus the number for a
     clean ingest), then by token overlap for retitled pages, then by locating
     the clause's body text inside a page's content (ingest splits compound
     clauses like "Remedies, Term, and Governing Law" into several topic pages).
  3. Write (session, source_doc, clause_num, heading, page_title) rows.

Deterministic end to end — no LLM anywhere. Idempotent: reruns replace the
session's rows. Standalone CLI, never shipped in deploy.zip (backfill_* is on
the exclusion list); run it locally against whichever DATABASE_URL should gain
the map (local Docker, or the Azure DB for the deployed pilot).

Usage:
    python backfill_clause_map.py --session <id>       # one session
    python backfill_clause_map.py --all                # every session with pages
    python backfill_clause_map.py --session <id> --dry # report only, no writes
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import fitz  # pymupdf — already a core dependency (reader.py)
from sqlalchemy import text

import config
from services import db

UPLOADS = os.path.join(config.DATA_DIR if hasattr(config, "DATA_DIR") else
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                       "uploads")
if not os.path.isdir(UPLOADS):
    UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")

# A numbered heading line: "5. Heading", "5) Heading", "5.2 Heading". Requires a
# capitalised heading of sane length on the same line, which filters out list
# items and prose ("within 5 business days" has no leading anchor and fails).
RX_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]\s+([A-Z][^\n]{2,88})\s*$", re.M)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _tokens(s: str) -> set[str]:
    STOP = {"and", "of", "the", "to", "a", "in", "on", "for", "no"}
    return {t for t in _norm(s).split() if t not in STOP}


def extract_headings(pdf_path: str) -> list[tuple[str, str, int]]:
    """(clause_num, heading, char_offset) for every numbered heading in the PDF."""
    doc = fitz.open(pdf_path)
    txt = "\n".join(p.get_text() for p in doc)
    doc.close()
    out = []
    for m in RX_HEADING.finditer(txt):
        heading = m.group(2).strip().rstrip(".")
        # Headings are short noun phrases; a line ending mid-sentence is prose.
        if len(heading.split()) <= 12:
            out.append((m.group(1), heading, m.end()))
    return out, txt


def reverse_match(titles: list[str], fulltext: str) -> dict[str, list[str]]:
    """{clause_num: [page_title, ...]} found by locating each stored TITLE in the
    source and reading the number in front of it.

    Title-driven rather than line-driven, so it works even when PDF extraction
    glues a heading to the body text on one long line (every Legal Opinion here
    does this, which makes forward heading extraction blind to them). It also
    cannot invent a mapping: a title that never appears after a number — every
    topic-titled judgment page — simply finds nothing.
    """
    out: dict[str, list[str]] = {}
    for t in titles:
        words = [re.escape(w) for w in t.split(" – ")[0].split()[:6] if w.strip(",&-")]
        if len(words) < 2:
            continue
        pat = r"(\d+(?:\.\d+)*)[.)]\s*" + r"[\s,&-]{1,4}".join(words)
        m = re.search(pat, fulltext, re.IGNORECASE)
        if m:
            out.setdefault(m.group(1), []).append(t)
    return out


def forward_match(headings: list, fulltext: str, titles: list[str]) -> dict[str, list[str]]:
    """{clause_num: [page_title, ...]} from clean-line headings, gated on quality.

    Judgments number their PARAGRAPHS ("14. This Court notes that…"), which are
    prose, not headings — fuzzy-matching those against topic titles produced
    garbage rows in the first dry run. Gate: only trust a document's forward
    matches if at least half of its extracted headings match a title exactly,
    which no paragraph-numbered document can satisfy.
    """
    exact_hits = 0
    per_heading: list[tuple[str, list[str]]] = []
    for num, heading, off in headings:
        h_norm, h_tok = _norm(heading), _tokens(heading)
        exact = [t for t in titles if _norm(t.split(" – ")[0]) == h_norm]
        if exact:
            exact_hits += 1
            per_heading.append((num, exact))
            continue
        scored = []
        for t in titles:
            p_tok = _tokens(t.split(" – ")[0])
            if p_tok:
                overlap = len(h_tok & p_tok) / min(len(h_tok), len(p_tok))
                if overlap >= 0.6:
                    scored.append((overlap, t))
        per_heading.append((num, [t for _, t in sorted(scored, reverse=True)]))

    if exact_hits * 2 < len(headings):
        return {}
    out: dict[str, list[str]] = {}
    for num, pages in per_heading:
        if pages:
            out.setdefault(num, []).extend(pages)
    return out


def backfill_session(session_id: str, dry: bool) -> None:
    engine = db.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT title, content, source_doc FROM pages WHERE session_id=:s"),
            {"s": session_id}).fetchall()
    by_doc: dict[str, dict[str, str]] = {}
    for title, content, src in rows:
        by_doc.setdefault(src, {})[title] = content or ""

    # Synthetic stand-ins are files like "Test_NDA_06.txt" / "Test_JVA_03.txt".
    # Match that shape, not the substring "Test_" — in some sessions the real
    # PDFs live in folders named "Legal AI - Test_NDA (1)", so a substring test
    # would throw away every real document in the session.
    real_docs = [d for d in by_doc
                 if d.strip() and not re.search(r"Test_[A-Z]{2,5}_\d", d)]
    print(f"\n=== session {session_id[:16]}…  ({len(real_docs)} real docs) ===")

    inserts, unmatched, unnumbered, nopdf = [], [], [], []
    for src in sorted(real_docs):
        path = os.path.join(UPLOADS, src)
        if not os.path.isfile(path) or not src.lower().endswith(".pdf"):
            nopdf.append(src)
            continue
        try:
            headings, fulltext = extract_headings(path)
        except Exception as e:
            print(f"  !! {src[-50:]}: {e}")
            continue
        titles = list(by_doc[src].keys())

        # Reverse (title-driven) wins on precision; forward fills the compound
        # clauses that ingest split across pages, but only for docs that pass
        # its exact-match quality gate.
        mapping = reverse_match(titles, fulltext)
        for num, pages in forward_match(headings, fulltext, titles).items():
            got = set(mapping.get(num, []))
            mapping.setdefault(num, []).extend(p for p in pages if p not in got)

        if not mapping:
            (unnumbered if not headings else unmatched).append(src)
            continue
        doc_rows = 0
        head_by_num = {num: h for num, h, _ in headings}
        for num, pages in mapping.items():
            for pt in pages:
                inserts.append({"sid": session_id, "src": src, "num": num,
                                "head": head_by_num.get(num, pt.split(" – ")[0]), "pt": pt})
                doc_rows += 1
        disp = "_".join(src.split("_")[-2:])[-52:]
        print(f"  {disp:54} clause_nums={len(mapping):>3} rows={doc_rows}")

    print(f"\n  docs with no numbering found : {len(unnumbered)}")
    print(f"  docs numbered but unmappable : {len(unmatched)}  (paragraph-numbered, e.g. judgments)")
    for src in unmatched[:8]:
        print(f"     ? {'_'.join(src.split('_')[-2:])[-52:]}")
    print(f"  docs without a PDF on disk   : {len(nopdf)}")
    print(f"  rows to insert               : {len(inserts)}")

    if dry or not inserts:
        print("  (dry run — nothing written)" if dry else "  (nothing to write)")
        return
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM clause_map WHERE session_id=:s"), {"s": session_id})
        conn.execute(text(
            "INSERT INTO clause_map (session_id, source_doc, clause_num, heading, page_title) "
            "VALUES (:sid, :src, :num, :head, :pt) ON CONFLICT DO NOTHING"), inserts)
    print(f"  WROTE {len(inserts)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if not args.session and not args.all:
        ap.error("--session <id> or --all required")

    if args.all:
        engine = db.get_engine()
        with engine.connect() as conn:
            sids = [r[0] for r in conn.execute(text(
                "SELECT DISTINCT session_id FROM pages"))]
    else:
        sids = [args.session]
    for sid in sids:
        backfill_session(sid, args.dry)


if __name__ == "__main__":
    main()
