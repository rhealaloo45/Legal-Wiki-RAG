"""
Value normalisation (target architecture § Phase 3.5c).

Phase 4's aggregation, gap detection and trend queries, and both Phase 5
agents, are all written as if the numbers and dates they need are already
queryable. They are extracted, which is not the same thing. On this corpus:

  contracts.liability_cap  mixes structured JSON, whole prose sentences, and
                           "the cap agreed in Schedule IV" in one column
  obligations.deadline     is prose — "not less than 90 days' prior written
                           notice", "within 72 hours", "prior to commencing work"
  clauses.typed_value      is 27% populated with per-row invented key shapes

This module turns those strings into comparable values, deterministically —
no LLM call, so a full-corpus backfill is free and re-runnable.

THE RULE THAT MATTERS MOST: a value that cannot be parsed is recorded as
explicitly unparsed, never as NULL. Gap detection asks "which contracts have
no liability cap"; if a cap we failed to read is stored as NULL, that query
answers "this contract has no cap" about a contract that plainly has one. The
distinction between "absent" and "unreadable" is the difference between a
useful answer and a confidently wrong one, and it has to survive in the data.

Raw text is never replaced. The normalised value sits beside it, and the raw
string stays the citable source — a lawyer is shown what the document says,
not what the parser made of it.
"""

import logging
import re
from datetime import date

logger = logging.getLogger(__name__)

# Parse outcomes. Stored alongside the value so a consumer can tell the three
# states apart without guessing from a NULL.
OK = "parsed"
UNPARSED = "unparsed"        # a value is present but this parser could not read it
ABSENT = "absent"            # the source field was genuinely empty
REFERENCE = "reference"      # the value points elsewhere ("as set out in Schedule IV")

_CURRENCY_WORDS = {
    "rs": "INR", "rs.": "INR", "inr": "INR", "rupees": "INR", "₹": "INR",
    "usd": "USD", "$": "USD", "us$": "USD", "dollars": "USD",
    "eur": "EUR", "€": "EUR", "euros": "EUR",
    "gbp": "GBP", "£": "GBP", "pounds": "GBP",
}

# "the cap agreed in Schedule IV", "as per Annexure B" — a real value that
# lives somewhere else. Distinct from unparsed: nothing is wrong, the figure
# simply is not in this field, and no amount of better parsing will find it.
_RX_REFERENCE = re.compile(
    # Two shapes, both common in this corpus and both meaning the same thing:
    # a pointer phrase followed by a document part, or a value described as
    # living in one ("the financial cap specified in the applicable commercial
    # schedule or Statement of Work" — 15% of liability_cap values, previously
    # misclassified as unparsed, which reads as a parser limitation rather
    # than as "the figure is in another document").
    r"(?:\b(?:as\s+)?(?:set\s+out|agreed|specified|provided|stated|negotiated|"
    r"documented|contained)\s+in\b|\bas\s+per\b|\bin\s+accordance\s+with\b|"
    r"\brefer\s+to\b|\bsee\b|\bsubject\s+to\b)"
    r"[^.]{0,80}?"
    r"\b(?:schedule|annexure|annex|appendix|exhibit|sow|statement\s+of\s+work|"
    r"commercial\s+schedule|order\s+form)\b",
    re.IGNORECASE,
)

_RX_AMOUNT = re.compile(
    r"(?P<cur>₹|\$|£|€|Rs\.?|INR|USD|EUR|GBP|US\$)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<scale>crores?|lakhs?|lacs?|millions?|billions?|mn|bn)?",
    re.IGNORECASE,
)

_SCALE = {
    "crore": 10_000_000, "crores": 10_000_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "million": 1_000_000, "millions": 1_000_000, "mn": 1_000_000,
    "billion": 1_000_000_000, "billions": 1_000_000_000, "bn": 1_000_000_000,
}

_RX_DURATION = re.compile(
    r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|"
    r"fifteen|twenty|thirty|sixty|ninety)\s*"
    r"(?:\(\s*\d+\s*\)\s*)?"
    r"(?P<unit>day|days|business\s+day|business\s+days|working\s+day|working\s+days|"
    r"week|weeks|month|months|year|years|hour|hours)\b",
    re.IGNORECASE,
)

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "sixty": 60, "ninety": 90,
}

_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7,
              "month": 30, "months": 30, "year": 365, "years": 365}


def parse_money(raw) -> dict:
    """A currency amount from a string, a dict, or a number.

    Handles what the corpus actually holds: already-structured
    {"amount": ..., "currency": ...}, prose containing a figure, Indian
    scale words (crore/lakh), and cross-references to a schedule.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {"status": ABSENT, "amount": None, "currency": None, "raw": raw}

    # Already structured — the ingest sometimes produces this shape directly.
    if isinstance(raw, dict):
        amt = raw.get("amount")
        if isinstance(amt, (int, float)):
            return {"status": OK, "amount": float(amt),
                    "currency": raw.get("currency") or "INR",
                    "raw": raw.get("raw") or str(raw)}
        return {"status": UNPARSED, "amount": None, "currency": None, "raw": str(raw)}

    if isinstance(raw, (int, float)):
        return {"status": OK, "amount": float(raw), "currency": None, "raw": str(raw)}

    s = str(raw).strip()
    if _RX_REFERENCE.search(s):
        return {"status": REFERENCE, "amount": None, "currency": None, "raw": s}

    m = _RX_AMOUNT.search(s)
    if not m:
        return {"status": UNPARSED, "amount": None, "currency": None, "raw": s}
    try:
        num = float(m.group("num").replace(",", ""))
    except ValueError:
        return {"status": UNPARSED, "amount": None, "currency": None, "raw": s}
    scale = (m.group("scale") or "").lower().strip()
    if scale:
        num *= _SCALE.get(scale, 1)
    cur_raw = (m.group("cur") or "").lower().strip()
    currency = _CURRENCY_WORDS.get(cur_raw)
    if not currency:
        low = s.lower()
        for word, code in _CURRENCY_WORDS.items():
            if word in low:
                currency = code
                break
    return {"status": OK, "amount": num, "currency": currency, "raw": s}


def parse_duration(raw) -> dict:
    """A duration in days from prose like "not less than 90 days' notice".

    `business_days` is preserved rather than silently converted: a notice
    period of 10 business days is not 10 calendar days, and flattening the
    distinction here would make the Date/Deadline Agent wrong in a way nobody
    could see downstream.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {"status": ABSENT, "days": None, "business_days": False, "raw": raw}
    s = str(raw).strip()
    if _RX_REFERENCE.search(s):
        return {"status": REFERENCE, "days": None, "business_days": False, "raw": s}

    m = _RX_DURATION.search(s)
    if not m:
        return {"status": UNPARSED, "days": None, "business_days": False, "raw": s}
    token = m.group("num").lower()
    n = _WORD_NUM.get(token)
    if n is None:
        try:
            n = int(token)
        except ValueError:
            return {"status": UNPARSED, "days": None, "business_days": False, "raw": s}
    unit = re.sub(r"\s+", " ", m.group("unit").lower()).strip()
    business = "business" in unit or "working" in unit
    base = unit.replace("business ", "").replace("working ", "")
    if base in ("hour", "hours"):
        # Sub-day windows are real in this corpus ("within 72 hours") and must
        # not round to zero days.
        return {"status": OK, "days": round(n / 24, 3), "business_days": False,
                "hours": n, "raw": s}
    mult = _UNIT_DAYS.get(base)
    if mult is None:
        return {"status": UNPARSED, "days": None, "business_days": False, "raw": s}
    return {"status": OK, "days": n * mult, "business_days": business, "raw": s}


_RX_DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_RX_DATE_DMY = re.compile(
    r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{4})\b",
    re.IGNORECASE)
_RX_DATE_MDY = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def parse_date(raw) -> dict:
    """An ISO date from the formats this corpus actually uses."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {"status": ABSENT, "date": None, "raw": raw}
    if isinstance(raw, date):
        return {"status": OK, "date": raw.isoformat(), "raw": str(raw)}
    s = str(raw).strip()
    for rx, order in ((_RX_DATE_ISO, "ymd"), (_RX_DATE_DMY, "dmy"), (_RX_DATE_MDY, "mdy")):
        m = rx.search(s)
        if not m:
            continue
        try:
            if order == "ymd":
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif order == "dmy":
                d, mo, y = int(m.group(1)), _MONTHS[m.group(2).lower()[:3]], int(m.group(3))
            else:
                mo, d, y = _MONTHS[m.group(1).lower()[:3]], int(m.group(2)), int(m.group(3))
            return {"status": OK, "date": date(y, mo, d).isoformat(), "raw": s}
        except (ValueError, KeyError):
            continue
    return {"status": UNPARSED, "date": None, "raw": s}
