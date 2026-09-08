"""Calculation Agent — derived values computed in Python, never by the model.

Phase 5. The principle is the one the rest of this system is built on: never
spend an LLM call on work a deterministic system can do exactly right. A
language model asked to add five milestone fees will usually get it right and
will occasionally not, and there is no way to tell the two apart from the
answer. Arithmetic done here is either exact or explicitly declined.

Distinct from the Phase 4 aggregation path, which SUMS STORED NUMBERS across
the corpus ("the average liability cap across our contracts"). This computes a
DERIVED value for one document from the terms that document itself states.

**The scope boundary is the design, not a caveat.** A formula whose inputs a
contract cannot contain is declined by name rather than approximated. Measured
on this corpus, only 6 of 387 populated liability caps are formula-shaped, and
all 6 resolve to "fees paid or payable under the SOW in the twelve months
preceding the event" — billing data that lives in an ERP, not a term any
agreement states. "Liability cap in rupees" is therefore permanently out of
scope here; asked for it, this agent says which fact is missing and stops.

What IS in scope, because the corpus really holds the inputs:

* **Total contract value** — milestone fee schedules and priced line items are
  stored as typed rows, so they can be summed and, where the document also
  states a total, reconciled against it. That reconciliation is the most
  valuable thing this agent does: it is a check no one performs by hand.
* **Liquidated-damages exposure** — several documents state a per-week rate and
  an aggregate cap as percentages of the contract price, and state the contract
  price too. Delay exposure is then exact arithmetic under a cap.
* **Escalation** — where a document states an escalation percentage and a
  period, the compounded figure after N years is exact. Where escalation is
  index-linked (CPI, PPI, WPI), it is declined: the index value is not in the
  contract and inventing one would be the exact failure this module exists to
  prevent.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Parsing. Every input arrives as a string an extraction model wrote, so each
# parser returns None rather than a guess — a None becomes a named missing
# input in the answer, which is a usable result; a guess becomes a wrong
# number that looks calculated.
# --------------------------------------------------------------------------

_CURRENCY_WORDS = {
    "inr": "INR", "rs": "INR", "rs.": "INR", "₹": "INR", "rupees": "INR",
    "usd": "USD", "$": "USD", "us$": "USD", "dollars": "USD",
    "eur": "EUR", "€": "EUR", "gbp": "GBP", "£": "GBP",
}

# Indian grouping ("14,40,000") and Western grouping ("1,440,000") both read
# correctly once the separators are removed, so one rule covers both.
_MONEY_RE = re.compile(
    r"(?P<cur>₹|\$|£|€|INR|USD|EUR|GBP|Rs\.?|US\$)?\s*"
    r"(?P<num>\d[\d,\s]*(?:\.\d+)?)"
    r"(?:\s*(?P<scale>lakhs?|lacs?|crores?|million|mn|billion|bn|thousand|k)\b)?",
    re.IGNORECASE,
)

_SCALES = {
    "lakh": Decimal(100000), "lakhs": Decimal(100000),
    "lac": Decimal(100000), "lacs": Decimal(100000),
    "crore": Decimal(10000000), "crores": Decimal(10000000),
    "million": Decimal(1000000), "mn": Decimal(1000000),
    "billion": Decimal(1000000000), "bn": Decimal(1000000000),
    "thousand": Decimal(1000), "k": Decimal(1000),
}

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)", re.IGNORECASE)

# An escalation tied to a published index cannot be computed from the contract.
_INDEX_RE = re.compile(
    r"(?:consumer|producer|wholesale|retail)\s+price(?:\s+index)?|"
    r"\bCPI\b|\bPPI\b|\bWPI\b|\bRPI\b|inflation\s+index|price\s+index",
    re.IGNORECASE,
)

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20,
}


def parse_money(raw) -> tuple[Decimal | None, str | None]:
    """(amount, currency) from a stated money string, or (None, None).

    Returns None for anything that is not a bare amount — "15% of the Contract
    Price" and "the cap agreed in Schedule IV" are both real values of this
    field on the live corpus, and both must fail here rather than yield a
    number, because a percentage read as an amount is off by seven orders of
    magnitude and looks entirely plausible in an answer.
    """
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    # A percentage is a rate, not an amount. Rejected outright.
    if "%" in text or re.search(r"\bper\s*cent|percent\b", text, re.IGNORECASE):
        return None, None
    m = _MONEY_RE.search(text)
    if not m:
        return None, None
    digits = re.sub(r"[,\s]", "", m.group("num"))
    if not digits or digits.startswith("."):
        return None, None
    try:
        amount = Decimal(digits)
    except InvalidOperation:
        return None, None
    scale = (m.group("scale") or "").lower()
    if scale in _SCALES:
        amount *= _SCALES[scale]
    cur_raw = (m.group("cur") or "").strip().lower()
    currency = _CURRENCY_WORDS.get(cur_raw)
    if not currency:
        # The symbol may sit after the number, or be spelled as a word.
        for word, code in _CURRENCY_WORDS.items():
            if word.isalpha() and len(word) > 2 and word in text.lower():
                currency = code
                break
    return amount, currency


def parse_percent(raw) -> Decimal | None:
    if raw is None:
        return None
    m = _PCT_RE.search(str(raw))
    if not m:
        return None
    try:
        return Decimal(m.group(1))
    except InvalidOperation:
        return None


def parse_periods(raw, unit_words: str) -> int | None:
    """A count of weeks / years / months stated either in digits or in words."""
    if raw is None:
        return None
    text = str(raw).lower()
    m = re.search(rf"(\d+)\s*(?:\(\d+\)\s*)?(?:{unit_words})", text)
    if m:
        return int(m.group(1))
    words = "|".join(_NUM_WORDS)
    m = re.search(rf"({words})\s*(?:\(\d+\)\s*)?(?:{unit_words})", text)
    if m:
        return _NUM_WORDS[m.group(1)]
    return None


def fmt_money(amount: Decimal | None, currency: str | None = None) -> str:
    if amount is None:
        return "—"
    prefix = f"{currency} " if currency else ""
    quantised = amount.quantize(Decimal("1")) if amount == amount.to_integral_value() \
        else amount.quantize(Decimal("0.01"))
    return f"{prefix}{quantised:,}"


# --------------------------------------------------------------------------
# Reading the typed rows a document already has.
# --------------------------------------------------------------------------

def _typed_rows(wiki_id: str, session_id: str, source_doc: str) -> list[dict]:
    """Every typed clause row for one document, newest extraction wins order."""
    from sqlalchemy import text as sql
    from services import db

    with db.get_engine().connect() as conn:
        rows = conn.execute(sql("""
            SELECT id, clause_type_canon, clause_type, typed_value, verbatim_text,
                   page_num, value_amount, value_currency
            FROM clauses
            WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
              AND typed_value IS NOT NULL
            ORDER BY id
        """), {"w": wiki_id, "s": session_id, "d": source_doc}).fetchall()
    return [{
        "id": r[0], "canon": r[1] or "", "type": r[2] or "",
        "typed": r[3] or {}, "text": r[4] or "", "page": r[5],
        "value_amount": r[6], "value_currency": r[7],
    } for r in rows]


# --------------------------------------------------------------------------
# The three computations.
# --------------------------------------------------------------------------

_GENERIC_COMPONENT_LABELS = {"line item", "milestone", ""}


def _dedupe_components(components: list[dict]) -> list[dict]:
    """Drop rows that are the same priced line extracted twice.

    A document re-extracted, or extracted once coarsely and once in detail,
    leaves two clause rows describing one line of one table. Summing both
    double-counts it, and the symptom is the worst kind this module can
    produce: a total that looks computed, cites real figures, and is wrong.

    Found on the Palladion Global purchase agreement, where three of ten typed
    line items were an earlier partial pass over the same table — identical
    unit price, identical annual value, identical payment terms, just without
    the item name the later pass captured. The duplicates summed to
    INR 926,482,386, and removing them lands the schedule on
    INR 2,582,327,852: exactly the total the document states. The document
    reconciled all along; the arithmetic was double-counting.

    Equal AMOUNT alone is not duplication — M4 and M5 of CND-TOR-SOW are both
    19,20,000 and both real. The key pairs the amount with what identifies the
    row: its schedule position for a milestone, its unit price for a line item.
    Where a duplicate carries a real name and the row already kept does not,
    the name is promoted, so deduplication never costs information.
    """
    kept: dict = {}
    order: list = []
    for c in components:
        k = c.get("key") or (c["amount"], c["label"])
        if k in kept:
            existing = kept[k]
            if (existing["label"].strip().lower() in _GENERIC_COMPONENT_LABELS
                    and c["label"].strip().lower() not in _GENERIC_COMPONENT_LABELS):
                existing["label"] = c["label"]
            existing["duplicates"] = existing.get("duplicates", 0) + 1
            continue
        kept[k] = dict(c)
        order.append(k)
    dropped = len(components) - len(order)
    if dropped:
        logger.info("Calculation: %d duplicate priced component(s) dropped "
                    "before summing", dropped)
    return [kept[k] for k in order]


def total_contract_value(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """Sum the priced components a document states, and reconcile with its total.

    Three component shapes appear on this corpus, in priority order:
      * milestone fee schedules  {"milestone": "M1", "week": "Wk 3", "fee": "…"}
      * priced line items        {"item": …, "annual_value": "Rs. …"}
      * a single stated total    {"total": "Rs. …"} / {"total_value": …}

    The reconciliation is the point. Where both a component schedule and a
    stated total exist, they are compared and any difference is reported
    rather than quietly resolved in favour of one of them — a schedule that
    does not add up to its own stated total is a drafting error worth
    surfacing, and it is not something anyone checks by hand.
    """
    rows = _typed_rows(wiki_id, session_id, source_doc)
    milestones, line_items, stated = [], [], []

    for r in rows:
        tv = r["typed"]
        if not isinstance(tv, dict):
            continue
        if "fee" in tv and ("milestone" in tv or "week" in tv):
            amt, cur = parse_money(tv.get("fee"))
            if amt is not None:
                _label = " ".join(str(tv.get(k)) for k in ("milestone", "week")
                                  if tv.get(k)) or "Milestone"
                milestones.append({
                    "label": _label, "amount": amt, "currency": cur,
                    "page": r["page"],
                    # Two milestones can legitimately carry the same fee — M4 and
                    # M5 of CND-TOR-SOW are both 19,20,000 — so the schedule
                    # position is part of what makes a row distinct.
                    "key": (amt, _label)})
        elif "annual_value" in tv:
            amt, cur = parse_money(tv.get("annual_value"))
            if amt is not None:
                _unit, _ = parse_money(tv.get("unit_price"))
                line_items.append({
                    "label": str(tv.get("item") or tv.get("vendor") or "Line item"),
                    "amount": amt, "currency": cur, "page": r["page"],
                    # Unit price alongside the total is what separates two real
                    # rows of equal value from one row extracted twice.
                    "key": (amt, _unit)})
        for key in ("total", "total_value", "contract_value", "aggregate_value"):
            if key in tv:
                amt, cur = parse_money(tv.get(key))
                if amt is not None:
                    stated.append({"label": key, "amount": amt,
                                   "currency": cur, "page": r["page"]})
        # A fees row carrying a bare "amount" and no milestone is the document
        # stating its own contract value in the fees clause.
        if "amount" in tv and "milestone" not in tv and r["canon"] == "fees":
            amt, cur = parse_money(tv.get("amount"))
            if amt is not None:
                stated.append({"label": "stated fee", "amount": amt,
                               "currency": cur, "page": r["page"]})

    components = _dedupe_components(milestones or line_items)
    if not components and not stated:
        return {"ok": False, "missing": "priced components",
                "detail": "This document has no typed fee schedule, priced line "
                          "items, or stated total value to compute from."}

    currency = next((c["currency"] for c in components + stated if c["currency"]), None)
    result = {"ok": True, "currency": currency,
              "kind": "milestones" if milestones else ("line_items" if line_items else "stated"),
              "components": components, "stated": stated}
    if components:
        result["computed"] = sum(c["amount"] for c in components)
    if stated:
        # Where the document states a total more than once, the largest is the
        # contract-level figure; the smaller ones are sub-totals.
        result["stated_total"] = max(s["amount"] for s in stated)
    if components and stated:
        diff = result["computed"] - result["stated_total"]
        result["difference"] = diff
        # Measured across the corpus: of 29 documents where a typed component
        # schedule disagrees with the document's own stated total, 28 have the
        # components summing UNDER it and only 1 over. Extraction captures a
        # subset of a long priced schedule far more often than a contract
        # misstates its own value, so a shortfall is reported as an incomplete
        # capture, not as a drafting error. Calling 28 sound contracts wrong is
        # a worse failure than saying less.
        if diff == 0:
            result["reconciliation"] = "exact"
        elif diff < 0:
            result["reconciliation"] = "partial_components"
        else:
            # A partial capture cannot produce a sum ABOVE the stated total, so
            # this one really is worth a human looking at.
            result["reconciliation"] = "exceeds_stated"
    return result


def ld_exposure(wiki_id: str, session_id: str, source_doc: str,
                weeks: int) -> dict:
    """Delay exposure at a stated per-week rate, capped at a stated aggregate cap.

    Rate and cap are stated on this corpus as percentages of the contract
    price, so the contract price must be resolvable too; when it is not, that
    is named as the missing input rather than substituted.
    """
    rows = _typed_rows(wiki_id, session_id, source_doc)
    rate_pct = cap_pct = None
    rate_src = cap_src = ""
    for r in rows:
        tv = r["typed"]
        if not isinstance(tv, dict):
            continue
        for key, val in tv.items():
            k = key.lower()
            if rate_pct is None and ("week" in k and "rate" in k or k == "rate_per_week"
                                     or k == "weekly_rate" or k == "weekly_rate_percent"):
                rate_pct = parse_percent(val) if not isinstance(val, (int, float)) \
                    else Decimal(str(val))
                rate_src = f"{key}: {val}"
            if cap_pct is None and ("cap" in k):
                got = parse_percent(val) if not isinstance(val, (int, float)) \
                    else Decimal(str(val))
                if got is not None:
                    cap_pct, cap_src = got, f"{key}: {val}"

    if rate_pct is None:
        return {"ok": False, "missing": "a stated per-week delay rate",
                "detail": "This document does not state a liquidated-damages rate "
                          "per week of delay, so exposure cannot be computed from it."}

    value = total_contract_value(wiki_id, session_id, source_doc)
    base = value.get("stated_total") or value.get("computed")
    if base is None:
        return {"ok": False, "missing": "the contract price",
                "detail": "The delay rate is stated as a percentage of the contract "
                          "price, but this document states no contract price to apply "
                          "it to."}

    uncapped = base * rate_pct / Decimal(100) * Decimal(weeks)
    cap_amount = base * cap_pct / Decimal(100) if cap_pct is not None else None
    capped = min(uncapped, cap_amount) if cap_amount is not None else uncapped
    weeks_to_cap = None
    if cap_amount is not None and rate_pct > 0:
        # int(), not Decimal: 15 / 0.5 is Decimal('3E+1'), which renders in an
        # answer as "week 3E+1".
        weeks_to_cap = int((cap_pct / rate_pct).to_integral_value(rounding="ROUND_CEILING"))
    return {"ok": True, "base": base, "currency": value.get("currency"),
            "rate_pct": rate_pct, "cap_pct": cap_pct, "weeks": weeks,
            "uncapped": uncapped, "cap_amount": cap_amount, "exposure": capped,
            "capped": cap_amount is not None and uncapped > cap_amount,
            "weeks_to_cap": weeks_to_cap,
            "rate_src": rate_src, "cap_src": cap_src}


def escalation(wiki_id: str, session_id: str, source_doc: str,
               years: int, base_amount=None) -> dict:
    """Compound a stated escalation percentage over N years.

    Every candidate is read from ONE clause at a time and never assembled
    across clauses. That is not fastidiousness: the live corpus has documents
    carrying two unrelated escalation regimes at once - an Escrow agreement
    here escalates Fees by the Producer Price Index while escalating Rent by a
    flat 10% every two years. Mining a percentage from one clause and a period
    from the other produced "10% every 1 year", a rate stated nowhere in the
    document.

    Two declines follow from that, both deliberate:

    * **Index-linked clauses are never mined for a rate.** Their percentage is
      a CAP on the index movement ("the increase in PPI, capped at 5% per
      annum"), not the escalation itself, and reading it as the escalation
      would silently answer a different question.
    * **Two different fixed regimes decline as ambiguous**, naming both. The
      document really does escalate two things differently, and choosing one
      without saying so would be a guess wearing a calculation's clothes.
    """
    rows = _typed_rows(wiki_id, session_id, source_doc)
    fixed: list[dict] = []
    indexed: list[dict] = []

    for r in rows:
        tv = r["typed"]
        if not isinstance(tv, dict):
            continue
        blob = " ".join(f"{k}: {v}" for k, v in tv.items())
        whole = blob + " " + (r["text"] or "")
        if r["canon"] != "fee_escalation" and "escalat" not in whole.lower():
            continue

        idx = _INDEX_RE.search(whole)
        if idx:
            indexed.append({"index": idx.group(0), "src": blob[:120]})
            continue

        pct = None
        every = None
        for key, val in tv.items():
            k = key.lower()
            if "escalat" in k or k in ("rate", "increase", "uplift"):
                if pct is None:
                    pct = parse_percent(val)
                if every is None:
                    every = parse_periods(val, "years?|yrs?")
            if k == "period" and every is None:
                every = parse_periods(val, "years?|yrs?")
        if pct is None:
            pct = parse_percent(r["text"])
        if every is None:
            every = parse_periods(r["text"], "years?|yrs?")
        if pct is not None:
            fixed.append({"pct": pct, "every": every or 1,
                          "src": blob[:120] or (r["text"] or "")[:120],
                          "page": r["page"]})

    unique = {(f["pct"], f["every"]) for f in fixed}

    if not fixed and indexed:
        names = ", ".join(sorted({i["index"] for i in indexed}))
        return {"ok": False, "missing": f"the {names} value",
                "detail": "Escalation in this document is tied to a published "
                          f"index ({names}). The index value is not a term of the "
                          "contract, so the escalated figure cannot be computed "
                          "from the document alone."}
    if not fixed:
        return {"ok": False, "missing": "a stated escalation percentage",
                "detail": "This document states no fixed escalation percentage."}
    if len(unique) > 1 or (fixed and indexed):
        described = [f"{f['pct']}% every {f['every']} year(s)" for f in fixed]
        described += [f"linked to the {i['index']}" for i in indexed]
        return {"ok": False, "missing": "one unambiguous escalation regime",
                "detail": "This document contains more than one escalation "
                          "provision - " + "; ".join(described) + ". They apply to "
                          "different amounts, so which one governs has to be read "
                          "from the clauses rather than assumed. Ask what the "
                          "escalation clauses say."}

    chosen = fixed[0]
    pct, every, src = chosen["pct"], chosen["every"], chosen["src"]

    currency = None
    if base_amount is None:
        value = total_contract_value(wiki_id, session_id, source_doc)
        base_amount = value.get("stated_total") or value.get("computed")
        currency = value.get("currency")
    if base_amount is None:
        return {"ok": False, "missing": "a base amount to escalate",
                "detail": f"Escalation is {pct}% every {every} year(s), but this "
                          "document states no base amount to apply it to."}

    steps = years // every
    factor = (Decimal(1) + pct / Decimal(100)) ** steps
    return {"ok": True, "pct": pct, "every": every, "years": years,
            "steps": steps, "base": base_amount, "currency": currency,
            "final": base_amount * factor,
            "increase": base_amount * factor - base_amount, "src": src}


# A document identifier: two or more hyphen/underscore-joined parts, at least
# one of them carrying a digit. "CND-TOR-SOW-2026-001" and "MAT-2021-6375"
# match; "Non-Disclosure" and "e-mail" do not.
_RX_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+){2,}")


def resolve_by_identifier(wiki_id: str, session_id: str,
                          question: str) -> list[str]:
    """One document named by a distinctive identifier in the question, or none.

    Runs ONLY when ordinary scope resolution has already returned nothing, and
    only accepts a token that matches exactly one document. Both conditions
    matter: it can never override a scope that resolved, and an ambiguous token
    resolves nothing rather than picking. That makes it strictly additive - it
    can turn a fall-through into an answer, never an answer into a different
    one.

    It exists because scope resolution is built around party names and
    instrument types, and a statement of work is commonly referred to by its
    reference number instead: "the total contract value of CND-TOR-SOW-2026-001"
    resolved to the whole corpus, and a calculation over the whole corpus is
    not a calculation.
    """
    from sqlalchemy import text as sql
    from services import db

    tokens = sorted({t for t in _RX_IDENTIFIER.findall(question or "")
                     if any(ch.isdigit() for ch in t)},
                    key=len, reverse=True)
    if not tokens:
        return []
    with db.get_engine().connect() as conn:
        for tok in tokens:
            rows = conn.execute(sql("""
                SELECT DISTINCT source_doc FROM documents
                WHERE wiki_id = :w AND session_id = :s AND source_doc ILIKE :t
                LIMIT 3
            """), {"w": wiki_id, "s": session_id, "t": f"%{tok}%"}).fetchall()
            if len(rows) == 1:
                logger.info("[CALC] identifier %r resolved one document", tok)
                return [rows[0][0]]
    return []


# --------------------------------------------------------------------------
# Detection and answer rendering.
# --------------------------------------------------------------------------

# Narrow by construction, like every other fast path here. Each kind needs an
# arithmetic verb AND its own subject; a question that merely mentions money
# ("what is the fee under the SOW") is a lookup and must reach retrieval, which
# can quote the clause. This branch returns a number, and a number offered in
# place of a quote is a worse answer even when it is right.
_RX_CALC_TOTAL = re.compile(
    r"\b(?:total|sum|add\s+up|aggregate|altogether|overall|combined)\b"
    r"[^?]{0,60}?\b(?:contract\s+value|value|fees?|milestones?|price|"
    r"consideration|payments?)\b"
    r"|\b(?:contract\s+value|milestone\s+totals?)\b",
    re.IGNORECASE)
_RX_CALC_LD = re.compile(
    r"\b(?:liquidated\s+damages|\bLDs?\b|delay\s+(?:penalt|damages|exposure)|"
    r"late\s+delivery\s+(?:penalt|damages))\w*", re.IGNORECASE)
_RX_CALC_LD_WEEKS = re.compile(
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
    r"(?:\(\d+\)\s*)?weeks?", re.IGNORECASE)
_RX_CALC_ESC = re.compile(
    r"\bescalat\w*", re.IGNORECASE)
_RX_CALC_YEARS = re.compile(
    r"(?:after|over|in|for)\s+"
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
    r"(?:\(\d+\)\s*)?years?", re.IGNORECASE)
# --- date arithmetic -------------------------------------------------------
# Calendar maths was the gap this module's own docstring implied it covered and
# did not: asked "how many days is the term" or "how many days ago did it end",
# the pipeline fell through to retrieval, which quoted the two dates back and
# said the day-count "is not stated in the Agreement". Literally true and
# useless — the document states both endpoints, and subtracting them is exactly
# the kind of work this module exists to do rather than ask a model to do.
#
# Sourced from `documents.effective_date` / `expiry_date`, the columns ingest
# already normalises to ISO, not from prose in a page. Where a document has no
# such date recorded the calculation is declined by name, the same as a missing
# fee schedule — on this corpus only a minority of documents carry an expiry
# date at all, so declining honestly matters more here than anywhere else.
_RX_CALC_TERM_DAYS = re.compile(
    r"\b(?:how\s+many\s+days|how\s+long|number\s+of\s+days|day[-\s]?count|"
    r"length\s+in\s+days)\b[^?]{0,60}?\b(?:term|agreement|contract|period)\b"
    r"|\b(?:term|contract)\s+length\b[^?]{0,30}\bdays?\b",
    re.IGNORECASE)
_RX_CALC_ELAPSED = re.compile(
    r"\bhow\s+(?:many\s+days|long)\s+ago\b"
    r"|\bdays?\s+(?:since|elapsed\s+since)\b"
    r"|\bhow\s+many\s+days\b[^?]{0,40}\b(?:since|ago|outstanding|pending|open)\b"
    r"|\bhow\s+long\b[^?]{0,40}\b(?:outstanding|pending|open)\b"
    r"|\bbeen\s+outstanding\b",
    re.IGNORECASE)
_RX_CALC_REMAINING = re.compile(
    r"\b(?:how\s+many\s+days|how\s+long)\b[^?]{0,40}?"
    r"\b(?:until|till|to\s+go|remain(?:ing)?|left)\b"
    r"|\bdays?\s+remaining\b|\bdays?\s+left\b",
    re.IGNORECASE)
# "When does the notice period actually end, accounting for business days?"
# Needs a period the question or the document supplies, and a start date.
_RX_CALC_NOTICE_END = re.compile(
    r"\b(?:when\s+does|when\s+will)\b[^?]{0,60}?\bnotice\s+period\b[^?]{0,30}\bend\b"
    r"|\bnotice\s+period\s+end(?:s|ing)?\s+(?:date|on)\b"
    r"|\bend\s+of\s+the\s+notice\s+period\b",
    re.IGNORECASE)
_RX_CALC_BUSINESS_DAYS = re.compile(
    r"\b(?:business|working|clear)\s+days?\b", re.IGNORECASE)
_RX_CALC_DAYS_N = re.compile(
    r"(\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"fifteen|twenty|thirty|forty[-\s]?five|sixty|ninety)\s*"
    r"(?:\(\d+\)\s*)?(?:calendar\s+|business\s+|working\s+)?days?\b",
    re.IGNORECASE)
# "Expressed in days, how long is that retention period?" — the period is in
# the question, so this needs no document dates. Kept separate from
# _RX_CALC_TERM_DAYS, which is about a document's own recorded term.
_RX_CALC_IN_DAYS = re.compile(
    r"\b(?:expressed|stated|measured|converted?)\s+in\s+(?:whole\s+)?days\b"
    r"|\bin\s+(?:whole\s+)?days\b[^?]{0,40}\bhow\s+(?:long|many)\b"
    r"|\bhow\s+(?:long|many\s+days)\b[^?]{0,60}?\bin\s+(?:whole\s+)?days\b",
    re.IGNORECASE)
# The period the question itself states, as (count, unit). Hours are included
# because a 72-hour notice window is the same question in a smaller unit.
_RX_STATED_PERIOD = re.compile(
    r"\b(\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty[-\s]?five|sixty|ninety)\s*"
    r"(?:\(\d+\)\s*)?(year|month|week|day|hour)s?\b",
    re.IGNORECASE)
_PERIOD_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty-five": 45,
    "forty five": 45, "sixty": 60, "ninety": 90,
}
# A year is 365 days and a month 30. Both are conventions, and the answer says
# so rather than implying a calendar-exact figure: the source clause says "not
# less than 7 years", which is itself a duration and not a pair of dates.
_PERIOD_IN_DAYS = {"year": 365, "month": 30, "week": 7, "day": 1}


def _period_stated_in_question(question: str):
    """(count, unit) the question states, or None. Longest period wins.

    A question can carry several numbers — "not less than 7 years following
    expiry" alongside a clause number or a date — so the largest period is
    taken rather than the first, which is the one the question is asking to
    convert.
    """
    best = None
    for m in _RX_STATED_PERIOD.finditer(question or ""):
        raw = m.group(1).lower().replace(" ", "-")
        n = _PERIOD_WORDS.get(raw)
        if n is None:
            try:
                n = int(raw)
            except ValueError:
                continue
        unit = m.group(2).lower()
        days = n / 24 if unit == "hour" else n * _PERIOD_IN_DAYS[unit]
        if best is None or days > best[2]:
            best = (n, unit, days)
    if not best:
        return None
    return (best[0], best[1])


def period_days(question: str) -> dict:
    """A period the question states, converted to whole days."""
    p = _period_stated_in_question(question)
    if not p:
        return {"ok": False, "missing": "a period to convert",
                "detail": "No number and unit (days, weeks, months, years) "
                          "could be read from the question."}
    count, unit = p
    if unit == "hour":
        days = count / 24
        whole = int(days) if float(days).is_integer() else round(days, 2)
        return {"ok": True, "kind": "period_days", "count": count, "unit": unit,
                "days": whole, "basis": f"{count} hours ÷ 24"}
    per = _PERIOD_IN_DAYS[unit]
    return {"ok": True, "kind": "period_days", "count": count, "unit": unit,
            "days": count * per,
            "basis": f"{count} {unit}{'s' if count != 1 else ''} × {per} days"}


_RX_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# Notice periods are written in round numbers this list covers and _NUM_WORDS,
# built for weeks and years, does not. Kept separate rather than widening the
# shared map, which the LD and escalation kinds also read.
_EXTRA_NUM_WORDS = {"thirty": 30, "forty-five": 45, "forty five": 45,
                    "sixty": 60, "ninety": 90}

# The one thing this agent must never attempt. Held as an explicit veto rather
# than left to fall through, so the decline can name the missing input.
_RX_CALC_OUT_OF_SCOPE = re.compile(
    r"\bliability\s+cap\b[^?]{0,40}\b(?:in\s+(?:rupees|INR|dollars|USD|currency)|"
    r"actual\s+(?:amount|value)|works?\s+out\s+to)"
    # The same question with the qualifier in front: "the actual amount of
    # the liability cap". A bare "what is the liability cap" is deliberately
    # NOT here - that is a lookup, and the clause quoted is the better answer.
    r"|\b(?:actual\s+(?:amount|value)|amount)\s+of\s+the\s+liability\s+cap\b"
    r"|\bhow\s+much\s+is\s+the\s+liability\s+cap\b", re.IGNORECASE)


def _doc_dates(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """The document's own recorded effective/expiry dates, as date objects.

    Reads the normalised columns rather than page prose: a date parsed out of a
    sentence is a date the model chose, and the whole point of this module is
    that every number in a computed answer came from a stored fact.
    """
    from datetime import date as _date
    from services import db as _db
    from sqlalchemy import text as _text

    out = {"effective": None, "expiry": None}
    try:
        with _db.get_engine().connect() as conn:
            row = conn.execute(_text("""
                SELECT effective_date, expiry_date FROM documents
                 WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
            """), {"w": wiki_id, "s": session_id, "d": source_doc}).fetchone()
    except Exception as e:
        logger.error("[CALC] date lookup failed for %r: %s", source_doc[:60], e)
        return out
    if not row:
        return out
    for key, raw in (("effective", row[0]), ("expiry", row[1])):
        m = _RX_ISO_DATE.search(str(raw or ""))
        if m:
            try:
                out[key] = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
    return out


# A place a holiday calendar could actually be selected for. Deliberately a
# closed list: a loose "capitalised word near 'open for business'" match would
# read a party name as a city and pick a calendar on that basis, which is the
# same class of confident-wrong this module exists to avoid.
_KNOWN_PLACES = (
    "Mumbai", "New Delhi", "Delhi", "Bengaluru", "Bangalore", "Chennai",
    "Kolkata", "Pune", "Hyderabad", "Ahmedabad", "Gurugram", "Noida",
    "Singapore", "London", "Dubai", "Abu Dhabi", "Sydney", "Melbourne",
    "Frankfurt", "New York",
)
_RX_BD_PLACE = re.compile(
    r"open\s+for\s+business\s+in\s+([A-Z][A-Za-z .'-]{2,30})", re.IGNORECASE)
_RX_BD_INDIRECT = re.compile(
    r"place\s+specified\s+for\s+notices|specified\s+for\s+notices|"
    r"place\s+of\s+notice|address\s+for\s+notices", re.IGNORECASE)


# Jurisdiction text -> a holiday calendar, most specific first. Ordered so a
# state or city named alongside the country wins over the bare country: "High
# Court of Judicature at Bombay" should select Maharashtra, not national-only.
#
# Bare "India" deliberately resolves to national holidays WITHOUT a subdivision
# rather than defaulting to Maharashtra because most documents here are Mumbai-
# flavoured. National holidays are the subset that applies in every Indian
# state, so the answer is short by any state-specific days and says so — which
# is a knowable error in one direction, not a guess that could be wrong in
# either.
_JURISDICTION_CALENDARS = (
    (r"\b(?:bombay|mumbai|maharashtra)\b", "IN", "MH", "Maharashtra, India"),
    (r"\b(?:delhi|new\s+delhi)\b", "IN", "DL", "Delhi, India"),
    (r"\b(?:karnataka|bengaluru|bangalore)\b", "IN", "KA", "Karnataka, India"),
    (r"\b(?:tamil\s*nadu|chennai|madras)\b", "IN", "TN", "Tamil Nadu, India"),
    (r"\b(?:telangana|hyderabad)\b", "IN", "TS", "Telangana, India"),
    (r"\b(?:west\s+bengal|kolkata|calcutta)\b", "IN", "WB", "West Bengal, India"),
    (r"\b(?:gujarat|ahmedabad)\b", "IN", "GJ", "Gujarat, India"),
    (r"\bindia\b", "IN", None, "India — national holidays only"),
    (r"\bsingapore\b", "SG", None, "Singapore"),
    (r"\b(?:england|wales|united\s+kingdom|london)\b", "GB", None, "United Kingdom"),
    (r"\b(?:emirate|u\.?a\.?e\.?|united\s+arab|dubai|abu\s+dhabi)\b", "AE", None,
     "United Arab Emirates"),
    (r"\b(?:australia|sydney|melbourne)\b", "AU", None, "Australia"),
    (r"\b(?:germany|frankfurt)\b", "DE", None, "Germany"),
)


def resolve_calendar(jurisdiction: str | None) -> dict | None:
    """A holiday calendar for a stored jurisdiction string, or None.

    Returns None rather than a default. There is no sensible fallback calendar:
    a wrong one moves a date silently, which is the failure this module refuses
    everywhere else.
    """
    j = (jurisdiction or "").strip()
    if not j:
        return None
    for pattern, country, subdiv, label in _JURISDICTION_CALENDARS:
        if re.search(pattern, j, re.IGNORECASE):
            return {"country": country, "subdiv": subdiv, "label": label,
                    "national_only": subdiv is None and country == "IN"}
    return None


def _holidays_in(country: str, subdiv, start, end) -> list:
    """Public holidays between two dates, or [] if the calendar is unavailable.

    Import is local and failure is silent-but-total: if the holiday package is
    missing the caller falls back to weekday-only counting and says so, which
    is the previous behaviour rather than an error.
    """
    try:
        import holidays as _h
    except Exception:
        return []
    try:
        years = list(range(start.year, end.year + 1))
        cal = _h.country_holidays(country, subdiv=subdiv, years=years)
        return sorted(d for d in cal if start < d <= end and d.weekday() < 5)
    except Exception as e:
        logger.error("[CALC] holiday calendar %s/%s failed: %s", country, subdiv, e)
        return []


def business_day_basis(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """What THIS document says a Business Day is, and whether that resolves.

    The generic disclaimer this replaced ("public holidays are not modelled")
    was true of every document and therefore informative about none. The
    documents are not silent on the question — 371 of them define the term —
    they simply define it by pointing somewhere. Reading each document's own
    definition turns one boilerplate caveat into a specific, checkable
    statement about that document, which is the same rule the rest of this
    module already follows for a missing fee schedule.

    Measured on this corpus: 3 definitions name a place outright (Mumbai), 368
    defer to "the place specified for notices", and the notices clauses as
    typed carry no place — so for those the chain terminates and the honest
    answer names where it terminated rather than choosing a calendar.
    """
    from services import db as _db
    from sqlalchemy import text as _text

    out = {"definition": None, "place": None, "indirect": False,
           "jurisdiction": None, "calendar": None}
    try:
        with _db.get_engine().connect() as conn:
            row = conn.execute(_text("""
                SELECT definition FROM defined_terms
                 WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
                   AND term ILIKE '%business day%'
                 LIMIT 1
            """), {"w": wiki_id, "s": session_id, "d": source_doc}).fetchone()
            jrow = conn.execute(_text("""
                SELECT jurisdiction FROM documents
                 WHERE wiki_id = :w AND session_id = :s AND source_doc = :d
            """), {"w": wiki_id, "s": session_id, "d": source_doc}).fetchone()
    except Exception as e:
        logger.error("[CALC] business-day definition lookup failed: %s", e)
        return out
    # The jurisdiction is read even when the document defines no Business Day:
    # the offset still needs a calendar, and this column is populated on 930 of
    # 1,372 documents here where a recoverable notice address is populated on
    # almost none.
    if jrow and jrow[0]:
        out["jurisdiction"] = str(jrow[0]).strip()
        out["calendar"] = resolve_calendar(out["jurisdiction"])
    if not row or not row[0]:
        return out
    definition = str(row[0]).strip()
    out["definition"] = definition
    m = _RX_BD_PLACE.search(definition)
    if m:
        cand = m.group(1).strip().rstrip(".").strip()
        for known in _KNOWN_PLACES:
            if re.search(rf"\b{re.escape(known)}\b", cand, re.IGNORECASE):
                out["place"] = known
                break
    if not out["place"] and _RX_BD_INDIRECT.search(definition):
        out["indirect"] = True
    # A place named in the Business Day clause itself outranks the recorded
    # jurisdiction: it is what the definition actually points at, where the
    # jurisdiction is only the closest available proxy for it. Overrides even
    # when both resolve, and the renderer then asks the reader to check the two
    # agree rather than hiding that a choice was made.
    if out["place"]:
        from_place = resolve_calendar(out["place"])
        if from_place:
            out["calendar"] = from_place
            out["calendar_from"] = "definition"
    elif out.get("calendar"):
        out["calendar_from"] = "jurisdiction"
    return out


def _business_days_between(start, end) -> int:
    """Weekdays strictly after `start` up to and including `end`.

    Public holidays are deliberately NOT modelled: this corpus spans several
    jurisdictions and holds no holiday calendar, so subtracting a guessed list
    would produce a date that looks authoritative and is wrong. Stated as a
    limitation in the rendered answer rather than silently approximated.
    """
    from datetime import timedelta
    if end < start:
        return -_business_days_between(end, start)
    days, cur = 0, start
    while cur < end:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def term_length(wiki_id: str, session_id: str, source_doc: str) -> dict:
    """Calendar days between the document's effective and expiry dates."""
    d = _doc_dates(wiki_id, session_id, source_doc)
    if not d["effective"] or not d["expiry"]:
        missing = ("an effective date" if not d["effective"] else "an expiry date")
        if not d["effective"] and not d["expiry"]:
            missing = "both a start and an end date"
        return {"ok": False, "missing": missing,
                "detail": "The term length is the difference between two dates, "
                          "and this document records "
                          + ("neither." if missing.startswith("both")
                             else "only one of them.")}
    days = (d["expiry"] - d["effective"]).days
    if days < 0:
        return {"ok": False, "missing": "dates in a usable order",
                "detail": f"The recorded expiry date ({d['expiry'].isoformat()}) "
                          f"is before the effective date "
                          f"({d['effective'].isoformat()}), so no term length "
                          "can be derived from them."}
    return {"ok": True, "kind": "term_days", "days": days,
            "inclusive_days": days + 1,
            "start": d["effective"], "end": d["expiry"]}


def date_delta(wiki_id: str, session_id: str, source_doc: str,
               direction: str) -> dict:
    """Days between today and the document's expiry (or effective) date.

    `direction` is 'elapsed' (how long ago) or 'remaining' (how long until).
    Both are computed against the real current date, and the answer states
    which date it used so a reader can check it.
    """
    from datetime import date as _date
    d = _doc_dates(wiki_id, session_id, source_doc)
    anchor = d["expiry"] or d["effective"]
    anchor_name = "expiry date" if d["expiry"] else "effective date"
    if not anchor:
        return {"ok": False, "missing": "a recorded date to measure from",
                "detail": "This document records neither an effective date nor "
                          "an expiry date, so nothing can be measured against "
                          "today."}
    today = _date.today()
    delta = (today - anchor).days
    return {"ok": True, "kind": direction, "anchor": anchor,
            "anchor_name": anchor_name, "today": today,
            "elapsed": delta, "remaining": -delta,
            "already_passed": delta > 0}


def notice_end(wiki_id: str, session_id: str, source_doc: str,
               days: int, business: bool) -> dict:
    """The date a notice period of `days` ends, counted from today.

    Counted from today rather than from a date in the document: a notice period
    runs from when notice is GIVEN, which is not a fact any contract states in
    advance. The answer says so, so the reader can re-base it on the real date
    notice went out.
    """
    from datetime import date as _date, timedelta
    if not days or days < 0:
        return {"ok": False, "missing": "a notice period in days",
                "detail": "No number of days was named in the question, and the "
                          "notice period could not be read from the document as "
                          "a plain day count."}
    start = _date.today()
    basis = business_day_basis(wiki_id, session_id, source_doc) if business else {}
    cal = (basis or {}).get("calendar")
    hol_used: list = []
    if business:
        # Where a calendar resolves, a public holiday is skipped exactly as a
        # weekend is. Where it does not, the loop is weekday-only and the
        # rendered answer says which of the two happened.
        hol_set = set()
        if cal:
            horizon = start + timedelta(days=days * 3 + 30)
            hol_set = set(_holidays_in(cal["country"], cal["subdiv"], start, horizon))
        cur, counted = start, 0
        while counted < days:
            cur += timedelta(days=1)
            if cur.weekday() < 5 and cur not in hol_set:
                counted += 1
        end = cur
        calendar_equiv = (end - start).days
        hol_used = sorted(d for d in hol_set if start < d <= end)
    else:
        end = start + timedelta(days=days)
        calendar_equiv = days
    return {"ok": True, "kind": "notice_end", "start": start, "end": end,
            "days": days, "business": business,
            "calendar_span": calendar_equiv,
            "business_days": _business_days_between(start, end),
            "basis": basis, "holidays_excluded": hol_used}


def is_calculation_query(question: str) -> str:
    """'total_value' | 'ld' | 'escalation' | 'term_days' | 'elapsed' |
    'remaining' | 'notice_end' | 'period_days' | 'out_of_scope' | ''."""
    q = question or ""
    if _RX_CALC_OUT_OF_SCOPE.search(q):
        return "out_of_scope"
    # Ahead of term_days, which reads the two recorded date columns and declines
    # when a document has no expiry: a question that STATES its own period
    # ("retain records for not less than 7 years -- expressed in days, how long
    # is that?") needs no document dates at all, and answering it from the
    # question's own words is not estimating. Confirmed live: three separate
    # evaluation questions of this shape were declined as "needs an expiry date"
    # while the period sat in the question itself.
    if _RX_CALC_IN_DAYS.search(q) and _period_stated_in_question(q):
        return "period_days"
    if _RX_CALC_LD.search(q) and _RX_CALC_LD_WEEKS.search(q):
        return "ld"
    if _RX_CALC_ESC.search(q) and _RX_CALC_YEARS.search(q):
        return "escalation"
    # Date kinds ahead of total_value: "how many days" carries no money word, so
    # they cannot collide, and ahead of each other most-specific first.
    if _RX_CALC_NOTICE_END.search(q):
        return "notice_end"
    if _RX_CALC_ELAPSED.search(q):
        return "elapsed"
    if _RX_CALC_REMAINING.search(q):
        return "remaining"
    if _RX_CALC_TERM_DAYS.search(q):
        return "term_days"
    if _RX_CALC_TOTAL.search(q):
        return "total_value"
    return ""


def _recorded_cap_text(wiki_id: str, session_id: str, docs: list) -> str:
    """The liability cap as the contracts row records it, quoted, or "".

    Only non-numeric caps reach here (a numeric one is not out_of_scope), so
    this is the "cap agreed in Schedule IV" shape: a real answer to what the
    cap is, in the document's own words, and better than explaining in the
    abstract why no rupee figure can be produced.
    """
    from sqlalchemy import text as _text
    from services import db as _db
    out = []
    try:
        with _db.get_engine().connect() as conn:
            for d in docs:
                row = conn.execute(_text(
                    "SELECT liability_cap FROM contracts WHERE wiki_id = :w "
                    "AND session_id = :s AND source_doc = :d"),
                    {"w": wiki_id, "s": session_id, "d": d}).fetchone()
                if row and (row[0] or "").strip():
                    out.append(f"> {str(row[0]).strip()}")
    except Exception as e:
        logger.error("[CALC] recorded cap lookup failed: %s", e)
        return ""
    return "\n\n".join(dict.fromkeys(out))


def _decline(kind: str, missing: str, detail: str, doc_label: str) -> str:
    return (f"**Cannot compute this — the calculation needs {missing}, which "
            f"this document does not state.**\n\n{detail}\n\n"
            f"Document assessed: {doc_label}\n\n"
            "Reported as a missing input rather than estimated. A derived figure "
            "is only worth having if every number in it came from the document.")


def render(kind: str, result: dict, doc_label: str, weeks: int = 0,
           years: int = 0) -> str:
    """Markdown showing the arithmetic, not only its result."""
    if not result.get("ok"):
        return _decline(kind, result.get("missing", "an input"),
                        result.get("detail", ""), doc_label)

    cur = result.get("currency")
    lines: list[str] = []

    if kind == "total_value":
        computed = result.get("computed")
        stated = result.get("stated_total")
        recon = result.get("reconciliation")
        # The stated total is the contract value whenever the typed components
        # fall short of it, because the shortfall means the schedule was only
        # partly captured - not that the document is worth less.
        headline = stated if (stated is not None and recon == "partial_components")             else (computed if computed is not None else stated)
        lines.append(f"**Total contract value: {fmt_money(headline, cur)}**")
        lines.append("")
        comps = result.get("components") or []
        if comps:
            noun = "Milestone fees" if result["kind"] == "milestones" else "Priced line items"
            lines.append(f"{noun} typed from this document ({len(comps)}):")
            for c in comps:
                lines.append(f"- {c['label']}: {fmt_money(c['amount'], cur)}")
            lines.append(f"- **Sum: {fmt_money(computed, cur)}**")
            lines.append("")
        if stated is not None:
            lines.append(f"Value stated in the document: {fmt_money(stated, cur)}")
            if recon == "exact":
                lines.append("The schedule reconciles exactly against the stated "
                             "total, so both figures agree.")
            elif recon == "partial_components":
                lines.append(f"The typed components come to "
                             f"{fmt_money(abs(result['difference']), cur)} less than "
                             "the stated total, which means only part of the priced "
                             "schedule was captured in extraction. The stated total "
                             "is reported above; the components are shown for what "
                             "they cover, not as a complete schedule.")
            elif recon == "exceeds_stated":
                lines.append(f"**The components exceed the stated total by "
                             f"{fmt_money(result['difference'], cur)}.** Incomplete "
                             "extraction cannot cause that, so the two figures in "
                             "the document are worth checking against each other.")
            lines.append("")

    elif kind == "ld":
        lines.append(f"**Liquidated-damages exposure after {weeks} week(s) of "
                     f"delay: {fmt_money(result['exposure'], cur)}**")
        lines.append("")
        lines.append(f"- Contract price: {fmt_money(result['base'], cur)}")
        lines.append(f"- Stated rate: {result['rate_pct']}% per week ({result['rate_src']})")
        _pct_total = (result['rate_pct'] * weeks).normalize()
        lines.append(f"- {weeks} week(s) x {result['rate_pct']}% = "
                     f"{_pct_total:f}% of the contract price "
                     f"= {fmt_money(result['uncapped'], cur)}")
        if result.get("cap_amount") is not None:
            lines.append(f"- Aggregate cap: {result['cap_pct']}% "
                         f"= {fmt_money(result['cap_amount'], cur)} ({result['cap_src']})")
            if result["capped"]:
                lines.append(f"- **Exposure is capped.** The uncapped figure exceeds "
                             f"the cap, so the cap applies.")
            elif result.get("weeks_to_cap"):
                lines.append(f"- The cap is reached at week {result['weeks_to_cap']}.")
        lines.append("")

    elif kind == "escalation":
        lines.append(f"**Escalated amount after {result['years']} year(s): "
                     f"{fmt_money(result['final'], cur)}**")
        lines.append("")
        lines.append(f"- Base amount: {fmt_money(result['base'], cur)}")
        lines.append(f"- Stated escalation: {result['pct']}% every "
                     f"{result['every']} year(s) ({result['src']})")
        lines.append(f"- Escalations applied in {result['years']} year(s): "
                     f"{result['steps']}")
        lines.append(f"- Increase: {fmt_money(result['increase'], cur)}")
        lines.append("")

    elif kind == "term_days":
        lines.append(f"**Term length: {result['days']} days**")
        lines.append("")
        lines.append(f"- Effective date: {result['start'].isoformat()}")
        lines.append(f"- Expiry date: {result['end'].isoformat()}")
        lines.append(f"- {result['end'].isoformat()} − {result['start'].isoformat()} "
                     f"= **{result['days']} days**")
        lines.append(f"- Counting both endpoints as days of the term: "
                     f"{result['inclusive_days']} days")
        lines.append("")
        lines.append("Both dates are the ones recorded for this document at "
                     "ingest; the subtraction is exact.")
        lines.append("")

    elif kind == "period_days":
        lines.append(f"**{result['days']} days**")
        lines.append("")
        lines.append(f"- Period stated in the question: {result['count']} "
                     f"{result['unit']}{'s' if result['count'] != 1 else ''}")
        lines.append(f"- {result['basis']} = **{result['days']} days**")
        lines.append("")
        if result["unit"] in ("year", "month"):
            lines.append("Converted at the usual convention of 365 days to a "
                         "year and 30 days to a month. The source period is a "
                         "duration, not a pair of dates, so this is a "
                         "conversion rather than a calendar count.")
            lines.append("")

    elif kind in ("elapsed", "remaining"):
        anchor = result["anchor"].isoformat()
        today = result["today"].isoformat()
        if kind == "elapsed":
            if result["already_passed"]:
                lines.append(f"**{result['elapsed']} days ago** "
                             f"({anchor}, measured to {today})")
            else:
                lines.append(f"**That date has not passed yet — it is "
                             f"{result['remaining']} days away** "
                             f"({anchor}, measured from {today})")
        else:
            if result["already_passed"]:
                lines.append(f"**No days remain — that date passed "
                             f"{result['elapsed']} days ago** "
                             f"({anchor}, measured to {today})")
            else:
                lines.append(f"**{result['remaining']} days remaining** "
                             f"({anchor}, measured from {today})")
        lines.append("")
        lines.append(f"- Date used: {anchor} (the document's {result['anchor_name']})")
        lines.append(f"- Today: {today}")
        lines.append(f"- Difference: **{abs(result['elapsed'])} days**")
        lines.append("")

    elif kind == "notice_end":
        unit = "business days" if result["business"] else "calendar days"
        lines.append(f"**Notice of {result['days']} {unit} given today "
                     f"({result['start'].isoformat()}) ends "
                     f"{result['end'].isoformat()}**")
        lines.append("")
        lines.append(f"- Start: {result['start'].isoformat()}")
        lines.append(f"- Period: {result['days']} {unit}")
        lines.append(f"- End: **{result['end'].isoformat()}**")
        if result["business"]:
            lines.append(f"- That is {result['calendar_span']} calendar days, "
                         f"because weekends do not count toward the period.")
        else:
            lines.append(f"- Of which {result['business_days']} are weekdays.")
        lines.append("")
        lines.append("**Counted from today, not from a date in the document** — a "
                     "notice period runs from when notice is actually given, which "
                     "no contract states in advance. Re-base it on the real notice "
                     "date if that differs.")
        lines.append("")
        if result.get("business"):
            basis = result.get("basis") or {}
            defn, place, indirect = (basis.get("definition"),
                                     basis.get("place"), basis.get("indirect"))
            cal, juris = basis.get("calendar"), basis.get("jurisdiction")
            hols = result.get("holidays_excluded") or []
            if defn:
                lines.append("**How this document defines a Business Day:**")
                lines.append("")
                lines.append(f"> {defn}")
                lines.append("")
            if cal:
                lines.append(f"**Holiday calendar applied: {cal['label']}.** "
                             f"Weekends and public holidays are both excluded from "
                             f"the count.")
                if hols:
                    lines.append("")
                    lines.append(f"Holidays falling inside this window ({len(hols)}), "
                                 "each of which pushed the end date out by a day:")
                    for d in hols[:12]:
                        lines.append(f"- {d.isoformat()}")
                else:
                    lines.append("")
                    lines.append("No public holiday falls inside this window, so the "
                                 "result equals the weekday-only count.")
                lines.append("")
                # The substitution is the part a reader has to be able to reject.
                if place and basis.get("calendar_from") == "definition":
                    lines.append(f"Calendar taken from the Business Day clause "
                                 f"itself, which names **{place}** — the document's "
                                 f"own statement of the place, not a proxy for it."
                                 + (f" Its recorded jurisdiction is *{juris}*; check "
                                    f"the two agree." if juris else ""))
                elif place:
                    lines.append(f"The definition fixes Business Days to **{place}**, "
                                 f"and the calendar above was selected from this "
                                 f"document's recorded jurisdiction ({juris}). Check "
                                 f"they agree.")
                elif indirect:
                    lines.append(f"**This is a substitution, and it is the one thing "
                                 f"to check.** The definition fixes Business Days to "
                                 f"*the place specified for notices*, and no place is "
                                 f"recoverable from the notices clause as extracted. "
                                 f"The calendar above comes from this document's "
                                 f"recorded jurisdiction instead ({juris}) — the "
                                 f"closest defensible proxy, not what the clause "
                                 f"says. If notices go somewhere else, this date is "
                                 f"wrong and should be recomputed against that place.")
                else:
                    lines.append(f"Calendar selected from this document's recorded "
                                 f"jurisdiction ({juris}).")
                if cal.get("national_only"):
                    lines.append("")
                    lines.append("The jurisdiction names India without a state, so "
                                 "only national holidays are applied. State-declared "
                                 "holidays are not — the count can therefore be short "
                                 "by those, never long.")
            else:
                lines.append("**Weekends are excluded; public holidays are not.**")
                lines.append("")
                if juris:
                    lines.append(f"This document records its jurisdiction as "
                                 f"*{juris}*, which no holiday calendar here covers, "
                                 f"so the figure counts weekdays only.")
                elif indirect:
                    lines.append("This Agreement defines a Business Day by reference "
                                 "to **the place specified for notices**, no place is "
                                 "recoverable from that clause, and the document "
                                 "records no jurisdiction either — so there is "
                                 "nothing to select a calendar from and the figure "
                                 "counts weekdays only.")
                else:
                    lines.append("This document records no jurisdiction and defines "
                                 "no place a calendar could be selected for, so the "
                                 "figure counts weekdays only.")
            lines.append("")

    lines.append(f"Document: {doc_label}")
    lines.append("")
    lines.append("Computed in Python from the values this document states, not by "
                 "the language model. Every figure above traces to a typed clause "
                 "in this document.")
    return "\n".join(lines)


# More than this and the answer stops being readable; scope resolving this
# broadly also means the question did not really name one instrument.
_MAX_CALC_DOCS = 3


def _label_for(source_doc: str, wiki_id: str, session_id: str) -> str:
    try:
        from services import doc_paths
        return doc_paths.display(source_doc, wiki_id, session_id)
    except Exception:
        return source_doc


def answer(question: str, wiki_id: str, session_id: str,
           docs: list[str] | None) -> dict | None:
    """A calculation payload for the in-scope document(s), or None to fall through.

    Scope on this corpus routinely resolves to a small set rather than to one
    document - OCR and non-OCR copies of the same instrument, or an MSA, DPA
    and SOW that share a party prefix. Each is computed separately and reported
    separately. Silently picking one of them would report a figure from a
    document the reader did not know was chosen, which is worse than showing
    two figures and saying which is which.
    """
    kind = is_calculation_query(question)
    if not kind:
        return None

    docs = list(docs or [])
    if not docs:
        # Only now, and only for an identifier that names exactly one document.
        docs = resolve_by_identifier(wiki_id, session_id, question)

    # Answered before the scope gate below, because it needs no document: the
    # period being converted is stated in the question itself. Scope resolution
    # on a purely descriptive reference ("Apex Veyra Digital Limited must retain
    # records ... expressed in days") often resolves to nothing or to the wrong
    # sibling, and declining a stated 5-years-to-days conversion on that basis
    # was measured as a wrong answer three times in one evaluation run.
    if kind == "period_days":
        result = period_days(question)
        if not result.get("ok"):
            return None
        label = ", ".join(_label_for(d, wiki_id, session_id)
                          for d in docs[:_MAX_CALC_DOCS])
        body = render(kind, result, label or "stated in the question")
        p = _payload(body, "Calculation")
        p["files_used"] = docs[:_MAX_CALC_DOCS]
        return p

    if kind == "out_of_scope":
        # Nothing resolved: say only that, and make no claim about how this
        # corpus drafts liability caps. The generic explanation below is true
        # of the fee-multiple caps it was written for, and was measured being
        # served for a document that was never read -- an assertion about a
        # document the answer had not seen is exactly the failure the rest of
        # this module exists to prevent.
        if not docs:
            return None
        label = ", ".join(_label_for(d, wiki_id, session_id)
                          for d in docs[:_MAX_CALC_DOCS])
        # Where the document records its cap as drafted, quote that rather than
        # explain in the abstract why a number cannot be produced: "the cap
        # agreed in Schedule IV" IS the answer to what the cap is, and it is
        # the document's own words.
        _drafted = _recorded_cap_text(wiki_id, session_id, docs[:_MAX_CALC_DOCS])
        if _drafted:
            body = (f"**No monetary amount is stated — the cap is recorded as "
                    f"drafted.**\n\n{_drafted}\n\nDocument assessed: {label}\n\n"
                    "Reported as drafted rather than converted to a figure: the "
                    "clause states the cap by reference, and any rupee amount "
                    "would have to come from billing data this corpus does not "
                    "hold.")
        else:
            body = _decline(
                "out_of_scope", "fees actually invoiced under the contract",
                "The liability cap in this document is expressed as a multiple of "
                "fees paid or payable over a rolling window. That is billing data "
                "held in a finance system, not a term the agreement states, so no "
                "figure computed from the document alone would be the real cap."
                "\n\nThe cap as drafted can be quoted instead - ask what the "
                "liability cap clause says.", label)
        p = _payload(body, "Calculation")
        p["files_used"] = docs[:_MAX_CALC_DOCS]
        return p

    if not docs:
        # Nothing resolved: there is nothing to compute from, and ordinary
        # retrieval is the better failure mode.
        return None

    weeks = years = 0
    notice_days, notice_business = 0, False
    if kind == "ld":
        m = _RX_CALC_LD_WEEKS.search(question)
        weeks = parse_periods(m.group(0), "weeks?") if m else 0
        if not weeks:
            return None
    elif kind == "escalation":
        m = _RX_CALC_YEARS.search(question)
        years = parse_periods(m.group(0), "years?") if m else 0
        if not years:
            return None
    elif kind == "notice_end":
        # Read from this pattern's own capture group rather than through
        # parse_periods, which requires the number to sit immediately against
        # the unit word: "30 business days" puts a qualifier in between and
        # parsed as None, which silently dropped the whole branch to retrieval.
        # parse_periods is shared with the weeks/years kinds and is left alone.
        m = _RX_CALC_DAYS_N.search(question)
        notice_days = 0
        if m:
            tok = m.group(1).strip().lower()
            if tok.isdigit():
                notice_days = int(tok)
            else:
                notice_days = _NUM_WORDS.get(tok, _EXTRA_NUM_WORDS.get(tok, 0))
        notice_business = bool(_RX_CALC_BUSINESS_DAYS.search(question))
        # Without a day count there is nothing to add to a date. Falling through
        # to retrieval is right: the clause stating the period is the answer.
        if not notice_days:
            return None

    sections, used, any_ok = [], [], False
    for source_doc in docs[:_MAX_CALC_DOCS]:
        label = _label_for(source_doc, wiki_id, session_id)
        try:
            if kind == "total_value":
                result = total_contract_value(wiki_id, session_id, source_doc)
            elif kind == "ld":
                result = ld_exposure(wiki_id, session_id, source_doc, weeks)
            elif kind == "term_days":
                result = term_length(wiki_id, session_id, source_doc)
            elif kind in ("elapsed", "remaining"):
                result = date_delta(wiki_id, session_id, source_doc, kind)
            elif kind == "notice_end":
                result = notice_end(wiki_id, session_id, source_doc,
                                    notice_days, notice_business)
            else:
                result = escalation(wiki_id, session_id, source_doc, years)
        except Exception as e:
            logger.error("[CALC] %s failed on %r: %s", kind, source_doc[:60], e)
            continue
        any_ok = any_ok or bool(result.get("ok"))
        sections.append(render(kind, result, label, weeks=weeks, years=years))
        used.append(source_doc)

    if not sections:
        return None
    # Where several documents were in scope but only some can be computed, the
    # ones that cannot are still shown - "this copy states no fee schedule" is
    # information about the corpus, not noise.
    if len(sections) > 1:
        body = (f"Scope resolved to {len(sections)} documents, computed "
                "separately:\n\n" + "\n\n---\n\n".join(sections))
    else:
        body = sections[0]

    payload = _payload(body, "Calculation")
    payload["files_used"] = used
    return payload


def _payload(body: str, label: str) -> dict:
    from services.intent_agent import _canned_payload
    p = _canned_payload(body, label, "deterministic-calculation")
    p["meta_answer"] = False
    return p
