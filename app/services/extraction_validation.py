"""
LLM output is untrusted input (target architecture § 01.6 Hardening).

Nothing in the pipeline today rejects or coerces a malformed model response.
A model returning `"about 90 days"` where a date is expected, or `{...}` where
a list is expected, either fails silently or lands raw in a typed column. This
module sits between every extraction result and every DB write.

Two rules shape the whole design:

  1. **A failed field is never dropped and never trusted.** It is coerced if
     it can be coerced unambiguously, and otherwise kept verbatim with its
     confidence knocked down and a flag set so it routes to the Review Queue.
     Dropping it loses information a reviewer needs; trusting it corrupts a
     typed column. Neither is acceptable, so it goes to a human.

  2. **Coercion never invents.** `"90 days"` becomes a normalized duration
     because that reading is unambiguous. `"end of next quarter"` does not
     become a date, because picking one would be fabrication wearing the
     costume of data cleaning.

Confidence, note, is self-reported by the same model that did the extraction,
and models are poorly calibrated — often most confident on exactly what they
got wrong. This layer therefore *lowers* confidence on validation failure but
never raises it on success: a value that validates cleanly is still only as
trustworthy as the model that produced it. The doc's calibration spot-check
remains an open item, and nothing here substitutes for it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# How far confidence drops when a field had to be coerced, vs. when it failed
# validation outright. Coercion is a smaller penalty than failure because the
# value was recoverable; both are penalties because neither was clean.
COERCION_PENALTY = 0.15
FAILURE_PENALTY = 0.4

# Confidence at or below this routes to the Review Queue regardless of what
# the model claimed. Matches the bulk-accept floor already used for clauses.
REVIEW_THRESHOLD = 0.6

_NULL_STRINGS = {
    "", "null", "none", "n/a", "na", "not stated", "not specified",
    "not applicable", "unknown", "unspecified", "not found", "-", "--",
}

_TRUE_STRINGS = {"true", "yes", "y", "1", "affirmative"}
_FALSE_STRINGS = {"false", "no", "n", "0", "negative"}

# Deliberately strict. Anything a human would have to guess at is left alone
# for a human to actually look at.
_DATE_PATTERNS = (
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"), ("y", "m", "d")),
    (re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"), ("d", "m", "y")),
    (re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$"), ("d", "m", "y")),
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DAY_MONTH_YEAR = re.compile(
    r"^(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?([A-Za-z]+),?\s+(\d{4})$"
)
_MONTH_DAY_YEAR = re.compile(r"^([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$")

_CURRENCY_SYMBOLS = {
    "$": "USD", "£": "GBP", "€": "EUR", "₹": "INR", "¥": "JPY",
}
_CURRENCY_CODES = {
    "usd", "gbp", "eur", "inr", "jpy", "aud", "cad", "chf", "sgd", "aed",
    "rs", "rs.", "inr.", "rupees", "dollars", "pounds", "euros",
}
_CURRENCY_RE = re.compile(
    r"^\s*(?P<sym>[$£€₹¥])?\s*(?P<code1>[A-Za-z.]{2,8})?\s*"
    r"(?P<num>[\d,]+(?:\.\d+)?)\s*(?P<mult>lakh|lakhs|crore|crores|million|mn|billion|bn|k|thousand)?"
    r"\s*(?P<code2>[A-Za-z.]{2,8})?\s*$"
)
_MULTIPLIERS = {
    "k": 1_000, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
    "million": 1_000_000, "mn": 1_000_000, "crore": 10_000_000,
    "crores": 10_000_000, "billion": 1_000_000_000, "bn": 1_000_000_000,
}

_DURATION_RE = re.compile(
    r"^\s*(\d+)\s*(day|days|week|weeks|month|months|year|years|"
    r"business day|business days|working day|working days)\s*$",
    re.IGNORECASE,
)


@dataclass
class FieldResult:
    """One validated field. `flagged` is the only thing callers must not
    ignore — it means a human has to look at this before it's trusted."""
    name: str
    value: Any
    raw: Any
    confidence: float
    flagged: bool = False
    reason: str | None = None
    coerced: bool = False

    def as_review_note(self) -> str | None:
        if not self.flagged:
            return None
        return f"{self.name}: {self.reason} (raw: {_truncate(self.raw)})"


@dataclass
class ValidationReport:
    """Result of validating a whole extraction payload."""
    values: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, FieldResult] = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def ok(self) -> bool:
        return not self.flagged

    def notes(self) -> list[str]:
        out = []
        for name in self.flagged:
            note = self.fields[name].as_review_note()
            if note:
                out.append(note)
        return out


def _truncate(v: Any, limit: int = 80) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def is_nullish(value: Any) -> bool:
    """The many ways a model says "not present". Treated as a real null rather
    than as the literal string "N/A" sitting in a governing_law column."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NULL_STRINGS
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


# ---------------------------------------------------------------------------
# Per-type coercion. Each returns (value, coerced, error) — error non-None
# means it could not be coerced without guessing, which is a Review Queue
# case, not a discard case.
# ---------------------------------------------------------------------------

def coerce_text(value: Any) -> tuple[Any, bool, str | None]:
    if is_nullish(value):
        return None, False, None
    if isinstance(value, str):
        return value.strip(), False, None
    # A model returning a list or dict where prose was asked for is recoverable
    # — flatten it rather than storing "['a', 'b']" with the brackets.
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if not is_nullish(v)]
        return "; ".join(parts), True, None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False), True, None
    return str(value), True, None


def coerce_date(value: Any) -> tuple[Any, bool, str | None]:
    """ISO date, or the raw string flagged for review. Never a guess.

    Ambiguity note: a bare `03/04/2024` is read day-first. That is a real
    assumption, not a neutral one — but this corpus is Indian and UK legal
    material where day-first is the convention, and picking silently either
    way is unavoidable. It is called out here so the choice is visible rather
    than buried, and such values keep a coercion penalty so they stay
    reviewable.
    """
    if is_nullish(value):
        return None, False, None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d"), False, None
    if not isinstance(value, str):
        return value, False, f"expected a date, got {type(value).__name__}"

    s = value.strip().rstrip(".")
    for pattern, order in _DATE_PATTERNS:
        m = pattern.match(s)
        if not m:
            continue
        parts = dict(zip(order, m.groups()))
        try:
            d = date(int(parts["y"]), int(parts["m"]), int(parts["d"]))
        except ValueError:
            return value, False, f"not a valid calendar date: {_truncate(value)}"
        return d.isoformat(), order != ("y", "m", "d"), None

    m = _DAY_MONTH_YEAR.match(s)
    if m and m.group(2).lower() in _MONTHS:
        try:
            d = date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
            return d.isoformat(), True, None
        except ValueError:
            return value, False, f"not a valid calendar date: {_truncate(value)}"

    m = _MONTH_DAY_YEAR.match(s)
    if m and m.group(1).lower() in _MONTHS:
        try:
            d = date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
            return d.isoformat(), True, None
        except ValueError:
            return value, False, f"not a valid calendar date: {_truncate(value)}"

    # "on or about March 2024", "end of the next quarter" — a human can read
    # these; a coercion routine that picks a day is fabricating.
    return value, False, f"not an unambiguous date: {_truncate(value)}"


def coerce_currency(value: Any) -> tuple[Any, bool, str | None]:
    """Normalized to {"amount": float, "currency": str|None, "raw": str}.

    Kept as a structured value rather than a bare float: a liability cap of
    5,000,000 means nothing without knowing whether it is rupees or dollars,
    and a float column silently loses that.
    """
    if is_nullish(value):
        return None, False, None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"amount": float(value), "currency": None, "raw": str(value)}, False, None
    if isinstance(value, dict) and "amount" in value:
        try:
            return ({"amount": float(value["amount"]),
                     "currency": value.get("currency"),
                     "raw": json.dumps(value)}, False, None)
        except (TypeError, ValueError):
            return value, False, "currency object has a non-numeric amount"
    if not isinstance(value, str):
        return value, False, f"expected a currency amount, got {type(value).__name__}"

    m = _CURRENCY_RE.match(value)
    if not m:
        # "capped at fees paid in the preceding 12 months" is a real and common
        # liability cap. It is not a number and must not be forced into one.
        return value, False, f"not a parseable amount: {_truncate(value)}"

    try:
        amount = float(m.group("num").replace(",", ""))
    except ValueError:
        return value, False, f"not a parseable amount: {_truncate(value)}"

    mult = (m.group("mult") or "").lower()
    if mult:
        amount *= _MULTIPLIERS[mult]

    currency = None
    if m.group("sym"):
        currency = _CURRENCY_SYMBOLS[m.group("sym")]
    else:
        for grp in ("code1", "code2"):
            code = (m.group(grp) or "").strip().lower().rstrip(".")
            if code in _CURRENCY_CODES:
                currency = {"rs": "INR", "rupees": "INR", "dollars": "USD",
                            "pounds": "GBP", "euros": "EUR"}.get(code, code.upper())
                break
        else:
            # A stray word that isn't a currency means this wasn't really a
            # currency string — don't quietly keep the number out of it.
            for grp in ("code1", "code2"):
                if (m.group(grp) or "").strip():
                    return value, False, f"unrecognized currency unit in: {_truncate(value)}"

    return ({"amount": amount, "currency": currency, "raw": value.strip()},
            True, None)


def coerce_number(value: Any) -> tuple[Any, bool, str | None]:
    if is_nullish(value):
        return None, False, None
    if isinstance(value, bool):
        return value, False, "expected a number, got a boolean"
    if isinstance(value, (int, float)):
        return value, False, None
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            return (float(s) if "." in s else int(s)), True, None
        except ValueError:
            return value, False, f"not a number: {_truncate(value)}"
    return value, False, f"expected a number, got {type(value).__name__}"


def coerce_boolean(value: Any) -> tuple[Any, bool, str | None]:
    if value is None:
        return None, False, None
    if isinstance(value, bool):
        return value, False, None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _NULL_STRINGS:
            return None, False, None
        if s in _TRUE_STRINGS:
            return True, True, None
        if s in _FALSE_STRINGS:
            return False, True, None
        return value, False, f"not a yes/no value: {_truncate(value)}"
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value), True, None
    return value, False, f"expected a boolean, got {type(value).__name__}"


def coerce_duration(value: Any) -> tuple[Any, bool, str | None]:
    """Normalized to {"count": int, "unit": str, "raw": str}. Business days
    stay distinct from calendar days — collapsing them changes deadlines."""
    if is_nullish(value):
        return None, False, None
    if not isinstance(value, str):
        return value, False, f"expected a duration, got {type(value).__name__}"
    m = _DURATION_RE.match(value)
    if not m:
        return value, False, f"not a simple duration: {_truncate(value)}"
    unit = m.group(2).lower().rstrip("s")
    if unit.endswith(" day"):
        unit = unit.replace(" day", "_day")
    return ({"count": int(m.group(1)), "unit": unit, "raw": value.strip()},
            True, None)


def _split_list_string(value: str) -> list[str]:
    """Split a delimited string into items without fracturing legal names.

    Semicolons and newlines are unambiguous separators. Commas are not: a
    party in a legal document is routinely written as
    `Acme Corp, a company incorporated under the laws of India`, where the
    comma introduces a *description of the same party*, not a second one.
    Splitting there produces a phantom party — and worse, one that reads as a
    real name to everything downstream.

    So a comma only separates when what follows starts like a new name
    (capital letter, digit, or a bracketed redaction marker). A fragment
    starting lowercase is an apposition and stays attached.
    """
    # Semicolons and newlines first — always real separators.
    coarse = [p for p in re.split(r"\s*[;\n]\s*", value) if p.strip()]
    out: list[str] = []
    for chunk in coarse:
        buf = ""
        # Keep parenthesised commas intact ("Smith, Jones and Co (India, Ltd)").
        for piece in re.split(r",(?![^(]*\))", chunk):
            piece_stripped = piece.strip()
            if not piece_stripped:
                continue
            starts_new = bool(re.match(r"^[A-Z0-9\[\"']", piece_stripped))
            if buf and starts_new:
                out.append(buf.strip())
                buf = piece_stripped
            elif buf:
                buf = f"{buf}, {piece_stripped}"
            else:
                buf = piece_stripped
        if buf.strip():
            out.append(buf.strip())
    return [p for p in out if p and not is_nullish(p)]


def coerce_list(value: Any) -> tuple[Any, bool, str | None]:
    """A list of non-empty strings. A model given a list field routinely
    returns a delimited string instead; that is recoverable."""
    if is_nullish(value):
        return [], False, None
    if isinstance(value, list):
        out = [str(v).strip() for v in value if not is_nullish(v)]
        return out, len(out) != len(value), None
    if isinstance(value, str):
        return _split_list_string(value), True, None
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()], True, None
    return [str(value)], True, None


# Tokens that invert meaning. A candidate match that differs from the input
# on any of these is not a near-miss to be rescued — it is the opposite claim.
_NEGATIONS = frozenset({"non", "not", "no", "never", "without", "excluding", "un"})

_ENUM_NORMALIZE = re.compile(r"[^a-z0-9]+")


def _enum_key(s: str) -> str:
    """Punctuation-insensitive form, so 'non-binding', 'non binding' and
    'Non Binding' are one value rather than three."""
    return _ENUM_NORMALIZE.sub(" ", s.strip().lower()).strip()


def _negation_tokens(s: str) -> frozenset[str]:
    return frozenset(_enum_key(s).split()) & _NEGATIONS


# A party entry that is only a description of a company, with the name
# redacted out of the source. Extremely common in this corpus.
_PARTY_DESCRIPTOR = re.compile(
    r"^\s*(a|an|the)\s+.*\b(company|corporation|partnership|limited liability|"
    r"incorporated|organized|organised|registered|existing|duly)\b",
    re.IGNORECASE,
)
# The defined term a contract assigns such a party: (hereinafter "Participant"),
# (the "Buyer"), referred to as 'Tata'.
_DEFINED_TERM = re.compile(
    r"""(?:hereinafter\s+(?:referred\s+to\s+as\s+)?|referred\s+to\s+as\s+|the\s+)?
        [“"']([A-Z][A-Za-z .&-]{1,40})[”"']""",
    re.VERBOSE,
)


def coerce_party_list(value: Any) -> tuple[Any, bool, str | None]:
    """A party list, with redacted parties rendered by their defined term.

    Legal drafting names a party then describes it: `Tata Sons Private
    Limited, a company incorporated under the laws of India`. When the name
    is redacted only the description survives, and showing a reviewer
    "a limited liability company duly incorporated under the laws of  having
    its registered address at" is technically faithful and practically
    useless — they cannot tell which party it is.

    So a description-only entry is reduced to the defined term the document
    itself assigns it ("Participant"), marked as redacted. The original text
    is kept as the raw value, so nothing is lost — the reviewer sees a usable
    label and can still check what it came from.
    """
    items, coerced, err = coerce_list(value)
    if err:
        return items, coerced, err

    out: list[str] = []
    changed = coerced
    for item in items:
        text = str(item).strip()
        if not _PARTY_DESCRIPTOR.match(text):
            out.append(text)
            continue
        m = _DEFINED_TERM.search(text)
        if m:
            out.append(f"{m.group(1)} (name redacted)")
            changed = True
        else:
            # No defined term either — keep it, but say plainly that this is a
            # description rather than letting it masquerade as a party name.
            out.append(f"[unnamed party — {text[:70].rstrip()}…]")
            changed = True
    return out, changed, None


def coerce_enum(value: Any, allowed: tuple[str, ...]) -> tuple[Any, bool, str | None]:
    """Map free text onto an allowed value, or flag it.

    The rescue path here is deliberately narrow because of one specific way it
    can fail catastrophically: `"non binding"` naively substring-matches
    `"binding"`, and a term sheet's indicative terms get recorded as
    enforceable ones. Any candidate whose negation words differ from the
    input's is rejected outright rather than rescued — a value that means the
    opposite is not a formatting variant, and a Review Queue card is a far
    cheaper outcome than a silently inverted binding status.
    """
    if is_nullish(value):
        return None, False, None
    if not isinstance(value, str):
        return value, False, f"expected one of {allowed}, got {type(value).__name__}"

    raw = value.strip()
    key = _enum_key(raw)
    lookup = {_enum_key(a): a for a in allowed}

    if key in lookup:
        return lookup[key], lookup[key] != raw, None

    input_negations = _negation_tokens(raw)
    hits = [
        orig for cand_key, orig in lookup.items()
        if (cand_key in key or key in cand_key)
        and _negation_tokens(orig) == input_negations
    ]
    # Longest candidate wins so a more specific option beats one contained in
    # it; still refuses when two equally-long candidates both match.
    if hits:
        hits.sort(key=lambda h: len(_enum_key(h)), reverse=True)
        if len(hits) == 1 or len(_enum_key(hits[0])) > len(_enum_key(hits[1])):
            return hits[0], True, None
    return value, False, f"not one of {allowed}: {_truncate(value)}"


def coerce_confidence(value: Any) -> float:
    """A confidence that isn't a number in [0,1] is treated as maximally
    uncertain, not as 1.0. A model that garbles this field has given no
    evidence its extraction was any better."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    return max(0.0, min(1.0, c))


_COERCERS = {
    "party_list": coerce_party_list,
    "text": coerce_text,
    "string": coerce_text,
    "date": coerce_date,
    "currency": coerce_currency,
    "number": coerce_number,
    "int": coerce_number,
    "float": coerce_number,
    "boolean": coerce_boolean,
    "bool": coerce_boolean,
    "duration": coerce_duration,
    "list": coerce_list,
    "array": coerce_list,
}


def validate_field(name: str, value: Any, field_type: str = "text",
                   base_confidence: float = 1.0,
                   allowed: tuple[str, ...] | None = None) -> FieldResult:
    """Validate and coerce one field. Never raises — a validation layer that
    can throw just moves the crash rather than preventing it."""
    conf = coerce_confidence(base_confidence)
    try:
        if field_type == "enum":
            coerced_value, was_coerced, err = coerce_enum(value, allowed or ())
        else:
            coercer = _COERCERS.get(field_type, coerce_text)
            coerced_value, was_coerced, err = coercer(value)
    except Exception as exc:  # a coercer bug must not abort an ingest
        logger.exception("Coercion of field %r raised", name)
        return FieldResult(name=name, value=value, raw=value,
                           confidence=max(0.0, conf - FAILURE_PENALTY),
                           flagged=True, reason=f"coercion error: {exc}")

    if err:
        return FieldResult(name=name, value=value, raw=value,
                           confidence=max(0.0, conf - FAILURE_PENALTY),
                           flagged=True, reason=err)

    if was_coerced:
        conf = max(0.0, conf - COERCION_PENALTY)

    return FieldResult(name=name, value=coerced_value, raw=value,
                       confidence=conf, coerced=was_coerced,
                       flagged=conf <= REVIEW_THRESHOLD and not is_nullish(coerced_value),
                       reason=("low confidence after coercion"
                               if (was_coerced and conf <= REVIEW_THRESHOLD) else None))


def validate_payload(payload: dict, spec: dict[str, str],
                     base_confidence: float = 1.0) -> ValidationReport:
    """Validate a whole extraction dict against {field_name: field_type}.

    A field the model omitted entirely is validated as None rather than being
    skipped — "the model didn't answer" and "the model said not-present" are
    the same fact downstream, and both belong in the report.
    """
    report = ValidationReport()
    if not isinstance(payload, dict):
        logger.warning("Extraction payload was %s, not a dict — rejecting",
                       type(payload).__name__)
        report.confidence = 0.0
        report.flagged.append("__payload__")
        report.fields["__payload__"] = FieldResult(
            name="__payload__", value=None, raw=payload, confidence=0.0,
            flagged=True, reason=f"expected an object, got {type(payload).__name__}",
        )
        return report

    for name, field_type in spec.items():
        allowed = None
        if isinstance(field_type, (list, tuple)):
            allowed, field_type = tuple(field_type), "enum"
        result = validate_field(name, payload.get(name), field_type,
                                base_confidence, allowed)
        report.fields[name] = result
        report.values[name] = result.value
        if result.flagged:
            report.flagged.append(name)

    scored = [f.confidence for f in report.fields.values()
              if not is_nullish(f.value)]
    report.confidence = min(scored) if scored else base_confidence
    return report


def sanitize_rows(rows: Any, spec: dict[str, str], required: tuple[str, ...] = (),
                  base_confidence: float = 1.0) -> tuple[list[dict], list[str]]:
    """Validate a list of extracted rows (clauses, citations, obligations...).

    Returns (clean_rows, problems). A row missing a required field is dropped
    with its reason recorded — unlike a bad *field*, a row with no verbatim
    text at all has nothing for a reviewer to review, so keeping it would put
    an empty card in the queue rather than useful work.
    """
    problems: list[str] = []
    if rows is None:
        return [], problems
    if not isinstance(rows, list):
        problems.append(f"expected a list of rows, got {type(rows).__name__}")
        return [], problems

    clean: list[dict] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"row {idx}: expected an object, got {type(row).__name__}")
            continue
        report = validate_payload(row, spec, base_confidence)
        missing = [r for r in required if is_nullish(report.values.get(r))]
        if missing:
            problems.append(f"row {idx}: missing required {', '.join(missing)} — dropped")
            continue
        out = dict(report.values)
        out["_confidence"] = report.confidence
        out["_flagged"] = bool(report.flagged)
        out["_validation_notes"] = report.notes()
        clean.append(out)
        if report.flagged:
            problems.extend(f"row {idx}: {n}" for n in report.notes())
    return clean, problems
