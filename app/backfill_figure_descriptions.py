"""
backfill_figure_descriptions.py — describe a figure from its image, not its caption

Every figure row this corpus holds was written by the ingest pass with
extraction_method='synthesis': the model inferred the figure from the page's
TEXT — its caption and the sentence introducing it — and never looked at the
image. That produces rows like "Figure 1. Floor Plan — the document states 'The
diagram below illustrates the floor plan referred to in this Agreement.'", which
is a description of the caption, not of the diagram.

The diagrams do carry real content. The floor plan on page 2 of the Infiniti
Retail commercial lease is a labelled site plan (Cafeteria, Parking, Utility
Room, Loading Bay, Warehouse, Office Block B) rendered as an embedded image, so
none of its labels appear in the PDF's text layer and nothing in the pipeline
could reach them. Asked what the diagram shows, the system could only report
that a diagram exists.

This renders the figure's page, asks the vision deployment about that ONE figure,
and appends what the model can read off the image to the document's own caption,
marking the row extraction_method='vision'.

COSTS MONEY: one vision call per figure, every run. Rows already marked
'vision' are skipped, so a second run over the same scope costs nothing;
--only scopes a run to named documents, --kinds to figure kinds, and --redo
re-describes rows already marked 'vision' (recovering the original caption
first). Always dry-run first — it prints the call count without making any.

By default it describes only the kinds whose caption genuinely withholds the
content: image, diagram, figure. The remaining kinds this corpus records are
signature blocks, seals and tables, where the caption already says what the
visual is ("Execution block with signatories: David Sastry ...") and a table's
real content is in the `tables` store — describing those would spend a call per
page to restate what is already known.

Usage:
    cd app
    python3 backfill_figure_descriptions.py <session_id> --dry-run
    python3 backfill_figure_descriptions.py <session_id> --only "Infiniti Retail_CLD"
    python3 backfill_figure_descriptions.py <session_id> --kinds all
"""

from __future__ import annotations

import logging
import re
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("backfill_figure_descriptions")

# A prompt about ONE figure, rather than reader.py's page-level "describe" mode.
# That mode transcribes the whole page before describing its visuals, which is
# right for ingest (the transcription is the page's text) and wrong here: the
# page's text already reached the wiki, and on a page carrying both a rent table
# and a floor plan the transcription consumed the entire length budget before
# reaching the diagram. Confirmed live — the first pass stored 1,500 characters
# of rent-table rows for the Infiniti Retail floor plan and not one room label.
_FIGURE_PROMPT = (
    "This page of a legal document contains a figure captioned:\n"
    "  {caption}\n"
    "Describe ONLY that figure. Do not transcribe the page's body text, and do "
    "not describe any other table or visual on the page.\n"
    "State: what type of visual it is; what it depicts; and EVERY piece of text "
    "readable inside it — node labels, room or area names, axis labels, legend "
    "entries, data labels, connector labels — verbatim, in the arrangement they "
    "appear in (say which node connects to which, which area adjoins which).\n"
    "Report only what is legible. If a label cannot be read with confidence, say "
    "so explicitly rather than guessing — an invented label is far worse than an "
    "acknowledged gap in a legal record. If the page contains no such figure, "
    "reply with exactly: NO FIGURE"
)

# Descriptions are read back into the answer context, which truncates them; a
# figure whose description runs longer than this is padding, not detail. Larger
# than the page-level cap it replaces because every character now describes the
# figure itself.
_MAX_DESCRIPTION_CHARS = 2000

# Figure kinds whose caption cannot stand in for the visual — see the module
# docstring for why signatures, seals and tables are left out by default.
_VISUAL_KINDS = {"image", "diagram", "figure"}


# Separates the document's own caption from what the model read off the image.
# Also how --redo recovers the caption from a row this script already rewrote.
_READ_PREFIX = "Read from the page image: "


# The synthesis pass ended many captions with its own disclaimer — "(diagram not
# reproduced in extract)". Once the image HAS been read that sentence is false,
# and left in place the answer repeats it: the first live check reported the
# floor plan's content as unavailable while the room labels sat directly below.
_STALE_DISCLAIMER_RE = re.compile(
    r"\s*\((?:[^()]*\b(?:not reproduced|not available|not included|not present)\b[^()]*)\)\.?",
    re.IGNORECASE,
)


def _original_caption(description: str | None) -> str:
    """The document's caption, with any previously-added description stripped."""
    text = (description or "").strip()
    idx = text.find(_READ_PREFIX)
    if idx >= 0:
        text = text[:idx].strip()
    return _STALE_DISCLAIMER_RE.sub("", text).strip()


def _describe_figure(page, caption: str, kind: str) -> str:
    """Ask the vision deployment about one figure on a rendered page."""
    import base64
    import fitz
    from services import llm

    label = caption.strip() or f"an unlabelled {kind or 'figure'}"
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    text, _usage = llm.ask_vision(image_b64, _FIGURE_PROMPT.format(caption=label),
                                  max_tokens=4096, fast=True)
    text = (text or "").strip()
    return "" if text.upper().startswith("NO FIGURE") else text


def backfill(target_session: str, dry_run: bool = False, only: str | None = None,
             all_kinds: bool = False, redo: bool = False) -> None:
    import config

    if not config.USE_DATABASE:
        logger.error("DATABASE_URL is not set — backfill only applies to PostgreSQL mode.")
        sys.exit(1)
    if config.OCR_ENGINE != "azure_vision":
        logger.error("OCR_ENGINE is %s — describing a diagram needs the vision "
                     "deployment (set OCR_ENGINE=azure_vision).", config.OCR_ENGINE)
        sys.exit(1)

    import fitz
    from services import db as _db
    from backfill_sections import _build_file_index, _find_file
    from sqlalchemy import text as _sql

    file_index = _build_file_index(config.UPLOAD_PATH)
    engine = _db.get_engine()
    already = "" if redo else "AND (extraction_method IS DISTINCT FROM 'vision')"

    with engine.connect() as conn:
        rows = conn.execute(
            _sql(f"""
                SELECT id, source_doc, page_num, figure_kind, description
                FROM figures
                WHERE session_id = :sid
                  {already}
                  AND page_num IS NOT NULL
                ORDER BY source_doc, page_num
            """),
            {"sid": target_session},
        ).fetchall()

    todo = [
        r for r in rows
        if (not only or only.lower() in (r.source_doc or "").lower())
        and (all_kinds or (r.figure_kind or "").strip().lower() in _VISUAL_KINDS)
    ]
    if not todo:
        logger.info("Nothing to do — every figure in scope already has a vision description.")
        return

    logger.warning("%d figure(s) in scope. This makes ONE vision call each%s.",
                   len(todo), " — dry run, no calls will be made" if dry_run else "")

    described = skipped = failed = 0
    for row in todo:
        caption = _original_caption(row.description)
        path = _find_file(file_index, row.source_doc)
        if not path:
            logger.warning("  [%s p%s] no file on disk — skipping", row.source_doc, row.page_num)
            skipped += 1
            continue
        if dry_run:
            logger.info("  [%s p%s] would describe (%s): %.70s",
                        row.source_doc, row.page_num, row.figure_kind, row.description or "")
            described += 1
            continue

        try:
            doc = fitz.open(path)
            try:
                if row.page_num < 1 or row.page_num > len(doc):
                    logger.warning("  [%s] page %s outside a %d-page document — skipping",
                                   row.source_doc, row.page_num, len(doc))
                    skipped += 1
                    continue
                visual = _describe_figure(doc[row.page_num - 1], caption, row.figure_kind)
            finally:
                doc.close()
        except Exception as e:
            logger.error("  [%s p%s] vision call failed: %s", row.source_doc, row.page_num, e)
            failed += 1
            continue

        if not visual:
            logger.warning("  [%s p%s] vision returned nothing usable — leaving the row alone",
                           row.source_doc, row.page_num)
            failed += 1
            continue

        # The caption is kept: it is the document's OWN name for the figure and
        # the thing a question is most likely to quote back ("the floor plan
        # diagram"). The visual description is added to it, never instead of it.
        merged = (f"{caption}\n\n{_READ_PREFIX}{visual}"
                  if caption else visual)[:_MAX_DESCRIPTION_CHARS]

        try:
            with engine.begin() as conn:
                conn.execute(
                    _sql("""
                        UPDATE figures
                        SET description = :d, extraction_method = 'vision', confidence = 0.8
                        WHERE id = :id
                    """),
                    {"d": merged, "id": row.id},
                )
        except Exception as e:
            logger.error("  [%s p%s] update failed: %s", row.source_doc, row.page_num, e)
            failed += 1
            continue

        described += 1
        logger.info("  [%s p%s] described (+%d chars)",
                    row.source_doc, row.page_num, len(visual))

    logger.info("%s — %d figure(s) described, %d skipped, %d failed.",
                "Dry run complete" if dry_run else "Backfill complete",
                described, skipped, failed)


if __name__ == "__main__":
    argv = sys.argv[1:]
    only = kinds = None
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = argv[i + 1]
        elif a == "--kinds" and i + 1 < len(argv):
            kinds = argv[i + 1]
    consumed = {v for v in (only, kinds) if v}
    positional = [a for a in argv if not a.startswith("--") and a not in consumed]
    if not positional:
        print(__doc__)
        sys.exit(1)
    backfill(target_session=positional[0], dry_run="--dry-run" in argv,
             only=only, all_kinds=(kinds or "").lower() == "all",
             redo="--redo" in argv)
