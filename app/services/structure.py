"""
Structural anchors + structure-aware segmentation
(target architecture § 01 "Segmentation — mechanism change, not just schema growth").

Today's segmentation is a blind 40,000-character cut with 500 characters of
overlap: arithmetic, with no regard for where the document's own boundaries
are. That was tolerable while only prose pages were being merged. It stops
being tolerable once structured rows are persisted per segment, because a
blind cut lands mid-clause and mid-citation, producing duplicated or garbled
DB rows that the overlap papers over rather than prevents.

So: parse the document's section/¶ numbering first (a cheap regex pass, zero
LLM calls), then pack whole sections up to the same character budget instead
of cutting through them. The anchors are worth persisting in their own right
— they answer "what does ¶14 say" — so the same parse feeds both the
`structural_anchors` table and the segmentation decision.

What this deliberately does *not* do: guess at structure that isn't marked.
A document with no numbering at all falls back to paragraph-boundary packing,
and if even that fails, to the original character cut. A wrong anchor is
worse than no anchor — it would put a confident, checkable-looking "¶ 14"
pointer on text that isn't paragraph 14.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Same budget as the existing blind cut — this changes *where* segments are
# cut, not how big they are, so the cost model is unchanged.
DEFAULT_SEGMENT_BUDGET = 40_000
# Only used by the last-resort character fallback, where a cut really can
# land mid-sentence and overlap is the only mitigation available.
FALLBACK_OVERLAP = 500

# A section shorter than this is almost always a heading that got matched
# without its body (a table-of-contents line, a running header). Packing it
# as its own unit is harmless; treating it as a real section boundary for
# anchor purposes is noise, so it's recorded but not given a heading role.
_MIN_MEANINGFUL_SECTION = 40


@dataclass(frozen=True)
class Anchor:
    """One structural marker found in the document."""
    label: str          # "14", "3.2", "Article IV", "(a)"
    kind: str           # paragraph | section | article | clause | schedule | recital
    heading_text: str   # the heading line itself, trimmed
    char_start: int
    ordinal: int        # order of appearance, 0-based

    def as_row(self) -> dict:
        return {
            "anchor_label": self.label,
            "anchor_kind": self.kind,
            "heading_text": self.heading_text[:500],
            "char_start": self.char_start,
            "ordinal": self.ordinal,
        }


# Ordered by specificity — the first pattern that matches a line wins, so
# "Article IV" is not also read as a bare numbered section. Every pattern is
# anchored at line start: a "section 12" mentioned mid-sentence is a
# cross-reference to a section, not the start of one, and matching it would
# fabricate boundaries inside running prose.
_ANCHOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("article", re.compile(
        r"^\s{0,8}(?:ARTICLE|Article)\s+([IVXLCDM]+|\d+(?:\.\d+)*)\s*[.:\-–—]?\s*(.*)$"
    )),
    ("schedule", re.compile(
        r"^\s{0,8}(?:SCHEDULE|Schedule|ANNEXURE|Annexure|APPENDIX|Appendix|EXHIBIT|Exhibit)"
        r"\s+([A-Z0-9]+(?:\.\d+)*)\s*[.:\-–—]?\s*(.*)$"
    )),
    ("section", re.compile(
        r"^\s{0,8}(?:SECTION|Section|CLAUSE|Clause)\s+(\d+(?:\.\d+)*)\s*[.:\-–—]?\s*(.*)$"
    )),
    ("recital", re.compile(
        r"^\s{0,8}(?:WHEREAS|RECITAL)\s*[.:\-–—]?\s*(.{0,200})$"
    )),
    # "14." / "3.2.1" / "14 " at line start followed by capitalised text —
    # the most common numbering in both contracts and judgments.
    ("section", re.compile(
        r"^\s{0,8}(\d+(?:\.\d+){0,3})[.)]?\s+([A-Z\"'(].{0,200})$"
    )),
    # Indian/UK judgment paragraph numbering: "14." on its own line, or
    # bracketed "[14]" as used in neutral citations.
    ("paragraph", re.compile(r"^\s{0,8}\[(\d{1,4})\]\s*(.{0,200})$")),
    ("clause", re.compile(r"^\s{0,8}\(([a-z]{1,3}|[ivxlcdm]{1,6}|\d{1,3})\)\s+(.{0,200})$")),
)

_ROMAN = re.compile(r"^[IVXLCDM]+$")

# Unnumbered headings ("Material Facts", "GOVERNING LAW"). Real legal
# documents in this corpus frequently have section structure with no
# numbering at all, and refusing to see it would mean segmenting those
# documents blind purely because their drafter didn't number the sections.
_MAX_HEADING_LEN = 80
_HEADING_STOPWORDS = re.compile(
    r"\b(shall|hereby|means|includes|pursuant|whereas|agrees?|the\s+parties)\b",
    re.IGNORECASE,
)
# "Date: 06 July 2025", "Client: Tata Sons Private Limited" — a metadata
# label/value pair, not a section that has a body under it. Shaped exactly
# like a Title Case heading otherwise, so it needs its own exclusion.
_METADATA_LINE = re.compile(r"^[A-Z][A-Za-z /]{1,24}:\s*\S")


def _is_heading_line(line: str, next_line: str | None) -> bool:
    """Conservative unnumbered-heading test.

    Every condition here exists to keep a line of ordinary prose from being
    promoted to a structural boundary — a false heading puts a segment cut in
    the middle of a sentence, which is precisely the failure this module was
    written to remove. When unsure, this returns False and the document falls
    through to paragraph packing, which is merely coarser, not wrong.
    """
    s = line.strip()
    if not s or len(s) > _MAX_HEADING_LEN:
        return False
    if s[-1] in ".,;:":          # headings don't end in sentence punctuation
        return False
    words = s.split()
    if not (2 <= len(words) <= 10):
        return False
    if not s[0].isupper():
        return False
    if _HEADING_STOPWORDS.search(s):   # reads as a sentence, not a label
        return False
    if _METADATA_LINE.match(s):        # "Date: ...", "Client: ..."
        return False
    if any(ch.isdigit() for ch in s[:3]):  # numbered — an earlier pattern owns it
        return False
    # A heading heads something: the following line must exist and be longer,
    # i.e. body text rather than another fragment of a list or address block.
    if next_line is None:
        return False
    nxt = next_line.strip()
    if not nxt or len(nxt) <= len(s):
        return False
    # Title Case or ALL CAPS only. Sentence case is how prose starts.
    alpha_words = [w for w in words if w[:1].isalpha()]
    if not alpha_words:
        return False
    if s.isupper():
        return True
    capitalised = sum(1 for w in alpha_words if w[0].isupper())
    return capitalised >= max(2, int(len(alpha_words) * 0.6))


def parse_anchors(text: str, max_anchors: int = 5000) -> list[Anchor]:
    """Find the document's own structural markers. Pure regex, no LLM.

    `max_anchors` guards a pathological input (an OCR'd table where every
    cell begins with a digit) from producing a hundred thousand rows — past
    the cap the document is treated as unstructured, which is the honest
    reading of a document with that many "sections".
    """
    if not text:
        return []

    anchors: list[Anchor] = []
    offset = 0
    ordinal = 0
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            matched = False
            for kind, pattern in _ANCHOR_PATTERNS:
                m = pattern.match(line)
                if not m:
                    continue
                if kind == "recital":
                    label, heading = "WHEREAS", m.group(1).strip()
                else:
                    label = m.group(1).strip()
                    heading = (m.group(2) or "").strip() if m.lastindex and m.lastindex >= 2 else ""
                anchors.append(Anchor(
                    label=label, kind=kind,
                    heading_text=(heading or stripped)[:500],
                    char_start=offset, ordinal=ordinal,
                ))
                ordinal += 1
                matched = True
                break
            if not matched:
                nxt = lines[idx + 1] if idx + 1 < len(lines) else None
                if _is_heading_line(line, nxt):
                    anchors.append(Anchor(
                        label=stripped[:60], kind="heading",
                        heading_text=stripped[:500],
                        char_start=offset, ordinal=ordinal,
                    ))
                    ordinal += 1
        offset += len(line)
        if len(anchors) >= max_anchors:
            logger.warning(
                "Anchor cap (%d) hit — treating document as unstructured. "
                "Usually means OCR noise or a table matched as numbering.",
                max_anchors,
            )
            return []

    return _drop_spurious(anchors, text)


def _drop_spurious(anchors: list[Anchor], text: str) -> list[Anchor]:
    """Remove anchors that clearly aren't real boundaries.

    Two cases matter. A dense run of one-line "sections" is a table of
    contents, not the body — keeping it would put every segment boundary in
    the first page. And a numbering sequence that never advances (fifty
    consecutive "1.") is list formatting, not structure.
    """
    if len(anchors) < 3:
        return anchors

    kept: list[Anchor] = []
    for i, a in enumerate(anchors):
        nxt = anchors[i + 1].char_start if i + 1 < len(anchors) else len(text)
        body_len = nxt - a.char_start
        if body_len < _MIN_MEANINGFUL_SECTION and a.kind in ("section", "paragraph"):
            continue  # heading with no body under it — TOC line or running header
        kept.append(a)

    if not kept:
        return []

    numeric = [a for a in kept if a.label.replace(".", "").isdigit()]
    if len(numeric) >= 5:
        firsts = {a.label.split(".")[0] for a in numeric}
        if len(firsts) == 1:
            logger.info("Numbering never advances (all '%s') — treating as list "
                        "formatting, not structure", next(iter(firsts)))
            return [a for a in kept if a not in numeric]
    return kept


def structure_ratio(text: str, anchors: list[Anchor]) -> float:
    """Roughly how structured the document is: anchors per 10k characters.
    Used only for logging and the Review Queue's own confidence signal — the
    segmentation decision keys off whether packing actually succeeds, not off
    a threshold on this number."""
    if not text:
        return 0.0
    return len(anchors) / max(1.0, len(text) / 10_000)


@dataclass(frozen=True)
class Segment:
    """One unit handed to a synthesis call."""
    text: str
    char_start: int
    char_end: int
    anchor_labels: tuple[str, ...]
    method: str  # anchors | paragraphs | characters

    @property
    def first_anchor(self) -> str | None:
        return self.anchor_labels[0] if self.anchor_labels else None


def split_segments(text: str, budget: int = DEFAULT_SEGMENT_BUDGET,
                   anchors: list[Anchor] | None = None) -> list[Segment]:
    """Segment along document structure, packing whole units up to `budget`.

    Three tiers, degrading honestly:
      1. Structural anchors, when the document has usable ones.
      2. Blank-line paragraph boundaries, when it doesn't.
      3. The original blind character cut with overlap — only when a single
         unit is itself larger than the budget, which means there is no
         boundary inside it to respect.

    Overlap exists only in tier 3. In tiers 1 and 2 the cut lands on a real
    boundary, so overlap would duplicate content across segments for no
    benefit — and duplicated content is exactly what produces the duplicate
    structured rows this change exists to prevent.
    """
    if not text:
        return []
    if len(text) <= budget:
        labels = tuple(a.label for a in (anchors or []))
        return [Segment(text, 0, len(text), labels, "anchors" if labels else "characters")]

    if anchors is None:
        anchors = parse_anchors(text)

    units = _units_from_anchors(text, anchors) if anchors else []
    method = "anchors"
    if not units:
        units = _units_from_paragraphs(text)
        method = "paragraphs"
    if not units:
        return _character_split(text, budget)

    return _pack(units, text, budget, method)


def _units_from_anchors(text: str, anchors: list[Anchor]) -> list[tuple[int, int, str]]:
    """(start, end, label) per anchored section, including any preamble before
    the first anchor — a contract's parties block sits above section 1 and
    losing it would drop exactly the fields extraction most needs."""
    if not anchors:
        return []
    units: list[tuple[int, int, str]] = []
    if anchors[0].char_start > 0:
        units.append((0, anchors[0].char_start, "preamble"))
    for i, a in enumerate(anchors):
        end = anchors[i + 1].char_start if i + 1 < len(anchors) else len(text)
        if end > a.char_start:
            units.append((a.char_start, end, a.label))
    return units


def _units_from_paragraphs(text: str) -> list[tuple[int, int, str]]:
    units: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"\n\s*\n", text):
        end = m.end()
        if end > start:
            units.append((start, end, ""))
            start = end
    if start < len(text):
        units.append((start, len(text), ""))
    return units if len(units) > 1 else []


def _pack(units: list[tuple[int, int, str]], text: str, budget: int,
          method: str) -> list[Segment]:
    segments: list[Segment] = []
    cur_start: int | None = None
    cur_end = 0
    cur_labels: list[str] = []

    def flush():
        nonlocal cur_start, cur_end, cur_labels
        if cur_start is None:
            return
        segments.append(Segment(
            text[cur_start:cur_end], cur_start, cur_end,
            tuple(l for l in cur_labels if l), method,
        ))
        cur_start, cur_end, cur_labels = None, 0, []

    for u_start, u_end, label in units:
        u_len = u_end - u_start
        if u_len > budget:
            # One section bigger than the whole budget. There is no boundary
            # inside it to honour, so this is the one place the blind cut is
            # still the right answer — applied to this unit alone, not to the
            # document.
            flush()
            for sub in _character_split(text[u_start:u_end], budget):
                segments.append(Segment(
                    sub.text, u_start + sub.char_start, u_start + sub.char_end,
                    (label,) if label else (), "characters",
                ))
            continue
        if cur_start is not None and (cur_end - cur_start) + u_len > budget:
            flush()
        if cur_start is None:
            cur_start = u_start
        cur_end = u_end
        if label:
            cur_labels.append(label)

    flush()
    return segments or _character_split(text, budget)


def _character_split(text: str, budget: int) -> list[Segment]:
    """The original blind cut, kept verbatim in behaviour (same budget, same
    500-char overlap) as the last-resort tier."""
    out: list[Segment] = []
    start = 0
    step = max(1, budget - FALLBACK_OVERLAP)
    while start < len(text):
        end = min(start + budget, len(text))
        out.append(Segment(text[start:end], start, end, (), "characters"))
        if end >= len(text):
            break
        start += step
    return out or [Segment(text, 0, len(text), (), "characters")]


def anchors_for_segment(anchors: list[Anchor], seg: Segment) -> list[Anchor]:
    return [a for a in anchors if seg.char_start <= a.char_start < seg.char_end]
