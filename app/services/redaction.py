"""
Basic PII redaction — masks a small set of high-precision personal-data
patterns in generated answer text before it reaches chat history, logs, or
the frontend.

Deliberately narrow: every pattern here has a syntactic shape that's very
unlikely to collide with the numeric IDs legal documents are full of (clause
numbers, dates, contract/invoice numbers, monetary figures). Bare digit runs
with no distinguishing punctuation or checksum are never touched, so a
contract reference number never gets mistaken for an account number.
"""

import logging
import re

import config

logger = logging.getLogger(__name__)

_RX_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RX_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_RX_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,6}[ ]?[A-Z0-9]{1,4}\b")

# 13-16 digits, optionally grouped in 4s with spaces or dashes. Luhn-checked
# below before redaction — the checksum is what keeps this from firing on
# ordinary long document/contract numbers.
_RX_CARD = re.compile(r"\b(?:\d[ -]?){12,15}\d\b")

# Requires phone-like punctuation (parens or multiple dashes/spaces around an
# optional country code) so a bare digit run — indistinguishable from a
# clause/contract number — is never matched.
_RX_PHONE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)\d{3,4}[ -]?\d{3,4}\b"
    r"|(?<!\w)(?:\+\d{1,3}[ -])?\d{2,4}-\d{3,4}-\d{3,4}\b"
)

# "account number", "a/c no", "routing number", "sort code" etc. followed
# within a short window by a run of digits — context-anchored so bare digit
# runs elsewhere (clause numbers, contract IDs) are left alone.
_RX_BANK_LABEL = re.compile(
    r"(?:bank\s+)?account\s*(?:no\.?|number)?|a/c\s*no\.?|routing\s*number|sort\s*code",
    re.I,
)
_RX_BANK_DIGITS = re.compile(r"\d[\d -]{6,20}\d")


def _luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits]
    checksum = 0
    for i, digit in enumerate(reversed(d)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _redact_cards(text: str) -> str:
    def _sub(m: re.Match) -> str:
        digits = re.sub(r"[ -]", "", m.group(0))
        if 13 <= len(digits) <= 16 and _luhn_ok(digits):
            return "[REDACTED-CARD]"
        return m.group(0)
    return _RX_CARD.sub(_sub, text)


def _redact_bank_accounts(text: str) -> str:
    # Redact only the digit run that follows a bank-label match within a
    # short window, not the label text itself.
    out, pos = [], 0
    for lm in _RX_BANK_LABEL.finditer(text):
        window_start = lm.end()
        window = text[window_start:window_start + 40]
        dm = _RX_BANK_DIGITS.search(window)
        if not dm:
            continue
        abs_start = window_start + dm.start()
        abs_end = window_start + dm.end()
        out.append(text[pos:abs_start])
        out.append("[REDACTED-BANK-ACCOUNT]")
        pos = abs_end
    out.append(text[pos:])
    return "".join(out)


def redact_pii(text: str) -> str:
    """Mask email, SSN, IBAN, bank-account, credit-card, and phone patterns.

    Fails open: any internal error returns the original text unchanged
    rather than raising, since this runs on the response path.
    """
    if not text or not getattr(config, "PII_REDACTION_ENABLED", True):
        return text
    try:
        out = _RX_EMAIL.sub("[REDACTED-EMAIL]", text)
        out = _RX_SSN.sub("[REDACTED-SSN]", out)
        out = _RX_IBAN.sub("[REDACTED-IBAN]", out)
        out = _redact_bank_accounts(out)
        out = _redact_cards(out)
        out = _RX_PHONE.sub("[REDACTED-PHONE]", out)
        return out
    except Exception as e:
        logger.warning("redact_pii failed, returning original text: %s", e)
        return text
