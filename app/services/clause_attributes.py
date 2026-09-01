"""
Clause sub-attributes (target architecture § Phase 4).

A clause is typed at the row level — one canonical type, one verbatim text,
one typed value. For the clause types Playbooks, Review, Compare and Risk all
care about, that single value flattens what is really a small structured
object. A limitation-of-liability provision is not "text plus one number": it
is whether a cap exists, what kind of cap, on what basis, and which liabilities
escape it.

Extracting that shape once means those four features read the same object
instead of each re-interpreting the same clause text independently.

Built by derivation, not by a new ingest prompt — the same choice made for
defined terms, for the same reason: a prompt change reaches only documents
ingested from now on, while the inputs for this already exist on all 1,372.
Every field below comes from data Phase 3.5c produced: `liability_cap_status`
and `liability_cap_amount` on `contracts`, and the `liability_cap_exclusion`
clauses the canonical vocabulary now separates from real caps.

ONE DISTINCTION THIS MODULE INSISTS ON, because conflating it is a real legal
error rather than a tidiness problem:

    An UNCAPPED EXCEPTION means liability survives the cap — fraud, wilful
    misconduct, breach of confidentiality. Exposure is unlimited.

    An EXCLUDED LOSS means the liability does not arise at all — indirect,
    consequential, loss of profit. Exposure is nil.

Those are opposite outcomes. Both are drafted as "the cap shall not apply to
…"-shaped prose and both landed on the same canonical clause type, so the
vocabulary alone cannot tell them apart. This module does, and reports them as
separate fields.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Liabilities that ESCAPE a cap — exposure is unlimited for these. Patterns are
# drawn from the corpus's own carve-out clause text, not from a textbook list.
UNCAPPED_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("fraud", re.compile(r"\bfraud(?:ulent)?\b", re.I)),
    ("wilful misconduct", re.compile(r"\b(?:wilful|willful)\s+misconduct\b", re.I)),
    ("gross negligence", re.compile(r"\bgross\s+negligence\b", re.I)),
    ("breach of confidentiality", re.compile(
        r"\bconfidentiality\s+breach|breach\s+of\s+confidential", re.I)),
    ("IP infringement", re.compile(
        r"\b(?:ip|intellectual\s+property)\s+infringement\b", re.I)),
    ("data misuse", re.compile(
        r"\bdata\s+misuse\b|\bunauthoris?zed\s+use\s+of\s+data\b|\bdata\s+breach\b", re.I)),
    ("anti-corruption failure", re.compile(
        r"\banti[- ]?(?:corruption|bribery)\b|\bcorrupt\s+practices\b", re.I)),
    ("regulatory misconduct", re.compile(r"\bregulatory\s+misconduct\b", re.I)),
    ("death or personal injury", re.compile(
        r"\bdeath\s+or\s+personal\s+injury\b|\bpersonal\s+injury\b", re.I)),
    ("liabilities that cannot lawfully be limited", re.compile(
        r"\bcannot\s+(?:lawfully\s+)?be\s+(?:limited|excluded)\b", re.I)),
]

# Losses a contract says are NOT RECOVERABLE at all — the opposite outcome.
EXCLUDED_LOSS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("indirect loss", re.compile(r"\bindirect\b", re.I)),
    ("consequential loss", re.compile(r"\bconsequential\b", re.I)),
    ("incidental loss", re.compile(r"\bincidental\b", re.I)),
    ("special damages", re.compile(r"\bspecial\s+(?:loss|damages)\b", re.I)),
    ("loss of profit", re.compile(r"\bloss\s+of\s+profits?\b", re.I)),
    ("loss of revenue", re.compile(r"\bloss\s+of\s+revenue\b|\brevenue\b", re.I)),
    ("loss of goodwill", re.compile(r"\bgoodwill\b", re.I)),
    ("anticipated savings", re.compile(r"\banticipated\s+savings\b", re.I)),
]

# A clause is an exclusion-of-liability rather than a carve-out when it says
# nobody is liable, rather than that the cap does not apply.
_RX_EXCLUSION_SHAPE = re.compile(
    r"\b(?:neither\s+party\s+is\s+liable|not\s+be\s+liable|no\s+liability\s+for|"
    r"excludes?\s+liability|shall\s+not\s+be\s+liable)\b", re.I)
_RX_CARVEOUT_SHAPE = re.compile(
    r"\b(?:cap\s+shall\s+not\s+apply|not\s+apply\s+to|except|carve[- ]?out|"
    r"uncapped|unlimited|distinguish)\b", re.I)


def _enabled() -> bool:
    import config
    return bool(getattr(config, "USE_DATABASE", False))


def classify_exclusion_clause(text_: str) -> str:
    """'uncapped_exception' | 'excluded_loss' | 'unknown' for one clause.

    Shape first, vocabulary second. "Neither Party is liable for indirect …"
    and "The cap shall not apply to fraud …" are structurally different
    sentences, and the structure is the more reliable signal — a carve-out
    clause routinely MENTIONS the losses it does not cover.
    """
    t = text_ or ""
    if _RX_EXCLUSION_SHAPE.search(t) and not _RX_CARVEOUT_SHAPE.search(t):
        return "excluded_loss"
    if _RX_CARVEOUT_SHAPE.search(t):
        return "uncapped_exception"
    # No structural signal — fall back to which vocabulary dominates.
    unc = sum(1 for _, rx in UNCAPPED_PATTERNS if rx.search(t))
    exc = sum(1 for _, rx in EXCLUDED_LOSS_PATTERNS if rx.search(t))
    if unc > exc:
        return "uncapped_exception"
    if exc > unc:
        return "excluded_loss"
    return "unknown"


def liability_cap_attributes(wiki_id: str, session_id: str,
                             source_doc: str) -> dict:
    """The structured object behind a document's limitation-of-liability terms.

    Returns cap_present as a three-state value, not a boolean: `True`,
    `False`, and `None` for "recorded elsewhere or unreadable". A boolean here
    would force the same collapse the Phase 3.5c status columns exist to
    prevent — a cap recorded in a schedule reported as `cap_present: False`
    is a contract wrongly described as uncapped.
    """
    if not _enabled():
        return {"error": "database not configured"}
    from sqlalchemy import text
    from services import db
    from services.normalize import OK, ABSENT, REFERENCE, FORMULA, UNPARSED

    with db.get_engine().connect() as conn:
        row = conn.execute(text("""
            SELECT liability_cap, liability_cap_amount, liability_cap_currency,
                   liability_cap_status
            FROM contracts
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = :d
            LIMIT 1
        """), {"w": wiki_id, "sid": session_id, "d": source_doc}).fetchone()
        exclusion_rows = [r[0] or "" for r in conn.execute(text("""
            SELECT verbatim_text FROM clauses
            WHERE wiki_id = :w AND session_id = :sid AND source_doc = :d
              AND clause_type_canon IN ('liability_cap_exclusion', 'liability_cap')
        """), {"w": wiki_id, "sid": session_id, "d": source_doc})]

    status = row[3] if row else None
    # A cross-reference is EVIDENCE OF A CAP, not absence of one: "the cap
    # agreed in Schedule IV" says a cap exists and tells you where the figure
    # lives. Only `unparsed` (and a missing row) leave it genuinely unknown.
    # Getting this backwards would describe a capped contract as indeterminate
    # and, worse, make the three-state value useless — everything uncertain
    # would collapse into one bucket regardless of what is actually known.
    if status in (OK, FORMULA, REFERENCE):
        cap_present = True
    elif status == ABSENT:
        cap_present = False
    else:
        cap_present = None

    cap_type = None
    if status == OK:
        cap_type = "fixed_amount"
    elif status == FORMULA:
        cap_type = "formula"
    elif status == REFERENCE:
        cap_type = "stated_elsewhere"
    elif status == UNPARSED:
        cap_type = "unreadable"

    uncapped: list[str] = []
    excluded: list[str] = []
    for t in exclusion_rows:
        kind = classify_exclusion_clause(t)
        if kind == "uncapped_exception":
            for label, rx in UNCAPPED_PATTERNS:
                if rx.search(t) and label not in uncapped:
                    uncapped.append(label)
        elif kind == "excluded_loss":
            for label, rx in EXCLUDED_LOSS_PATTERNS:
                if rx.search(t) and label not in excluded:
                    excluded.append(label)

    return {
        "source_doc": source_doc,
        "cap_present": cap_present,
        "cap_type": cap_type,
        "cap_status": status,
        "cap_amount": float(row[1]) if row and row[1] is not None else None,
        "cap_currency": row[2] if row else None,
        "cap_raw": (row[0] or "")[:400] if row else None,
        "uncapped_exceptions": uncapped,
        "excluded_losses": excluded,
        "clauses_examined": len(exclusion_rows),
        "note": _attribute_note(cap_present, cap_type, uncapped, excluded),
    }


def _attribute_note(cap_present, cap_type, uncapped, excluded) -> str:
    if cap_present is None:
        base = ("Whether this contract caps liability could not be established — the "
                "provision could not be read. This is deliberately not reported as "
                "'no cap'.")
    elif cap_present is False:
        base = "No liability cap is stated in this contract."
    elif cap_type == "formula":
        base = "Liability is capped, with the cap stated as a formula rather than a figure."
    elif cap_type == "stated_elsewhere":
        base = "Liability is capped, but the figure lives in a schedule or statement of work."
    else:
        base = "Liability is capped at a stated amount."
    if uncapped:
        base += (f" {len(uncapped)} categor{'y' if len(uncapped) == 1 else 'ies'} of "
                 f"liability escape the cap entirely.")
    if excluded:
        base += (f" A further {len(excluded)} loss type(s) are excluded from recovery "
                 f"altogether — a different outcome from being uncapped, and reported "
                 f"separately for that reason.")
    return base


def corpus_cap_profile(wiki_id: str, session_id: str, limit: int = 200) -> dict:
    """How the corpus distributes across cap shapes, and which carve-outs recur.

    The aggregate view Playbooks and Risk want: not "what does this contract
    say" but "what is normal here, and which contracts depart from it".
    """
    if not _enabled():
        return {"error": "database not configured"}
    from sqlalchemy import text
    from services import db

    with db.get_engine().connect() as conn:
        docs = [r[0] for r in conn.execute(text("""
            SELECT DISTINCT source_doc FROM clauses
            WHERE wiki_id = :w AND session_id = :sid
              AND clause_type_canon IN ('liability_cap', 'liability_cap_exclusion')
            LIMIT :lim
        """), {"w": wiki_id, "sid": session_id, "lim": limit})]

    shapes: dict[str, int] = {}
    uncapped_freq: dict[str, int] = {}
    excluded_freq: dict[str, int] = {}
    for d in docs:
        a = liability_cap_attributes(wiki_id, session_id, d)
        key = a.get("cap_type") or "no_cap"
        shapes[key] = shapes.get(key, 0) + 1
        for u in a["uncapped_exceptions"]:
            uncapped_freq[u] = uncapped_freq.get(u, 0) + 1
        for e in a["excluded_losses"]:
            excluded_freq[e] = excluded_freq.get(e, 0) + 1

    return {
        "documents_examined": len(docs),
        "cap_shapes": sorted(shapes.items(), key=lambda kv: -kv[1]),
        "common_uncapped_exceptions": sorted(uncapped_freq.items(), key=lambda kv: -kv[1]),
        "common_excluded_losses": sorted(excluded_freq.items(), key=lambda kv: -kv[1]),
        "note": ("Uncapped exceptions and excluded losses are counted separately "
                 "because they are opposite outcomes: the first means liability "
                 "survives the cap without limit, the second means it does not arise "
                 "at all."),
    }
