"""
Ingestion stage 01 — page classification beyond text/scanned
(target architecture § 01 stage 01, § 01.1 Tables, Charts & Images).

Today a page is classified one of two ways: it extracted enough characters
(text) or it didn't (scanned, send to OCR). That misses two cases the doc
calls out, and both fail *silently*, which is what makes them worth fixing:

  * **A table-bearing page extracts "successfully."** The character count is
    fine, so nothing flags it — but pypdf flattens rows and columns into
    loose text, so the structure that carried the meaning is gone. A fee
    schedule becomes a list of numbers with no way to tell which column they
    came from.

  * **A chart or diagram on a page with surrounding text never triggers OCR
    at all.** The >50-chars-per-page test passes on the caption and body
    text, so the visual content is simply never looked at. Zero extraction,
    no warning.

This module decides which pages are worth the extra look, so the expensive
vision path only sees pages that actually need it. Detection is heuristic and
deliberately errs toward *not* escalating: a false positive costs a real
vision call on every ingest, and this runs over every page of every document.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

TEXT = "text"
SCANNED = "scanned"
TABLE_BEARING = "table_bearing"
FIGURE_BEARING = "figure_bearing"

# Below this, the page is scanned — matches reader.MIN_CHARS_PER_PAGE's
# existing contract, restated here so this module can be reasoned about and
# tested on its own.
MIN_CHARS_PER_PAGE = 50

# --- table signals ---------------------------------------------------------
# Runs of whitespace that look like column gutters, and the pipe/box-drawing
# characters a PDF table sometimes survives as.
_COLUMN_GUTTER = re.compile(r"\S {3,}\S")
_PIPE_ROW = re.compile(r"\|.*\|")
_BOX_DRAWING = re.compile(r"[┌┬┐├┼┤└┴┘│─╔╦╗╠╬╣╚╩╝║═]")
_NUMERIC_CELL = re.compile(r"(?<!\w)[\d,]+(?:\.\d+)?%?(?!\w)")
_TABLE_WORDS = re.compile(
    r"\b(table\s+\d+|schedule\s+[A-Z0-9]|fee schedule|rate card|"
    r"price list|annexure\s+[A-Z0-9]|milestone|sl\.?\s*no\.?|s\.?\s*no\.?)\b",
    re.IGNORECASE,
)

# --- figure signals --------------------------------------------------------
_FIGURE_WORDS = re.compile(
    r"\b(figure\s+\d+|fig\.\s*\d+|chart|graph|diagram|flowchart|"
    r"illustrated below|shown below|as depicted|screenshot|exhibit\s+[A-Z0-9])\b",
    re.IGNORECASE,
)


def _table_score(text: str) -> float:
    """0.0-1.0 confidence that this page contains tabular structure."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return 0.0

    gutter_lines = sum(1 for l in lines if _COLUMN_GUTTER.search(l))
    pipe_lines = sum(1 for l in lines if _PIPE_ROW.search(l))
    box_lines = sum(1 for l in lines if _BOX_DRAWING.search(l))
    numeric_lines = sum(1 for l in lines if len(_NUMERIC_CELL.findall(l)) >= 2)

    total = len(lines)
    score = 0.0
    # Aligned columns across a run of lines is the strongest available signal
    # from flattened text.
    score += min(0.5, (gutter_lines / total) * 1.2)
    score += min(0.4, (pipe_lines + box_lines) / total * 2.0)
    score += min(0.3, (numeric_lines / total) * 0.9)
    if _TABLE_WORDS.search(text):
        score += 0.15
    return min(1.0, score)


def _figure_score(text: str, image_count: int, image_area_ratio: float) -> float:
    """0.0-1.0 confidence that this page carries meaningful visual content.

    Image area matters more than image count: a page with a letterhead logo
    and a page with a full-width chart both have images, and only one of them
    is worth a vision call. Small images are assumed decorative, because on
    this corpus they overwhelmingly are.
    """
    score = 0.0
    if image_area_ratio >= 0.5:
        score += 0.6
    elif image_area_ratio >= 0.2:
        score += 0.4
    elif image_area_ratio >= 0.08:
        score += 0.2
    if image_count >= 1 and image_area_ratio >= 0.08:
        score += 0.1
    if _FIGURE_WORDS.search(text):
        score += 0.3
    return min(1.0, score)


# Thresholds are set high on purpose. This runs per page across every
# document; a false positive is a vision call that costs money and time on
# a page that didn't need one, so the bar to escalate is deliberately above
# "might be".
TABLE_THRESHOLD = 0.55
FIGURE_THRESHOLD = 0.5


def classify_page(text: str, image_count: int = 0,
                  image_area_ratio: float = 0.0) -> dict:
    """Classify one page. Returns kind plus the scores behind the call.

    Order matters: a page with too little text is scanned regardless of what
    else is on it, because until OCR runs there is no text to judge the other
    signals from.
    """
    text = text or ""
    stripped = text.strip()

    if len(stripped) < MIN_CHARS_PER_PAGE:
        return {"kind": SCANNED, "table_score": 0.0,
                "figure_score": _figure_score("", image_count, image_area_ratio),
                "needs_vision": True, "vision_mode": "transcribe",
                "reason": f"only {len(stripped)} chars extracted"}

    t_score = _table_score(stripped)
    f_score = _figure_score(stripped, image_count, image_area_ratio)

    # A page can hold both. Whichever signal is stronger decides the vision
    # prompt mode, since one call has to pick a mode — but both scores are
    # returned so the caller can record what else was suspected.
    if t_score >= TABLE_THRESHOLD and t_score >= f_score:
        return {"kind": TABLE_BEARING, "table_score": round(t_score, 3),
                "figure_score": round(f_score, 3), "needs_vision": True,
                "vision_mode": "table",
                "reason": f"tabular structure detected (score {t_score:.2f})"}

    if f_score >= FIGURE_THRESHOLD:
        return {"kind": FIGURE_BEARING, "table_score": round(t_score, 3),
                "figure_score": round(f_score, 3), "needs_vision": True,
                "vision_mode": "describe",
                "reason": f"visual content detected (score {f_score:.2f})"}

    return {"kind": TEXT, "table_score": round(t_score, 3),
            "figure_score": round(f_score, 3), "needs_vision": False,
            "vision_mode": None, "reason": "plain text page"}


def page_image_stats(page) -> tuple[int, float]:
    """Image count and the fraction of the page they cover, from a PyMuPDF
    page. Returns (0, 0.0) if anything goes wrong — an unreadable image list
    should downgrade this page to plain text, not fail the read."""
    try:
        rect = page.rect
        page_area = float(rect.width * rect.height) or 1.0
        images = page.get_images(full=True)
        if not images:
            return 0, 0.0
        covered = 0.0
        for img in images:
            try:
                for bbox in page.get_image_rects(img[0]):
                    covered += float(bbox.width * bbox.height)
            except Exception:
                continue
        return len(images), min(1.0, covered / page_area)
    except Exception as err:
        logger.debug("Could not read image stats for page: %s", err)
        return 0, 0.0
