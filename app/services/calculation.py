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


def is_calculation_query(question: str) -> str:
    """'total_value' | 'ld' | 'escalation' | 'out_of_scope' | ''."""
    q = question or ""
    if _RX_CALC_OUT_OF_SCOPE.search(q):
        return "out_of_scope"
    if _RX_CALC_LD.search(q) and _RX_CALC_LD_WEEKS.search(q):
        return "ld"
    if _RX_CALC_ESC.search(q) and _RX_CALC_YEARS.search(q):
        return "escalation"
    if _RX_CALC_TOTAL.search(q):
        return "total_value"
    return ""


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

    if kind == "out_of_scope":
        label = ", ".join(_label_for(d, wiki_id, session_id)
                          for d in docs[:_MAX_CALC_DOCS]) or "not resolved"
        body = _decline(
            "out_of_scope", "fees actually invoiced under the contract",
            "The liability cap in this corpus is expressed as a multiple of fees "
            "paid or payable over a rolling window. That is billing data held in "
            "a finance system, not a term any agreement states, so no figure "
            "computed from the document alone would be the real cap.\n\nThe cap as "
            "drafted can be quoted instead - ask what the liability cap clause "
            "says.", label)
        p = _payload(body, "Calculation")
        p["files_used"] = docs[:_MAX_CALC_DOCS]
        return p

    if not docs:
        # Nothing resolved: there is nothing to compute from, and ordinary
        # retrieval is the better failure mode.
        return None

    weeks = years = 0
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

    sections, used, any_ok = [], [], False
    for source_doc in docs[:_MAX_CALC_DOCS]:
        label = _label_for(source_doc, wiki_id, session_id)
        try:
            if kind == "total_value":
                result = total_contract_value(wiki_id, session_id, source_doc)
            elif kind == "ld":
                result = ld_exposure(wiki_id, session_id, source_doc, weeks)
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
