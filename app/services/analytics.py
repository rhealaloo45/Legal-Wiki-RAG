"""
Structured analytics over the normalised layer (target architecture § Phase 4).

Three query shapes the retrieval pipeline cannot answer, all pure SQL over
columns Phase 3.5c produced — no LLM call, no embedding:

  aggregate   SUM / AVG / MIN / MAX / median over liability caps and contract
              values, optionally scoped to a party or document type.
  gaps        Which documents LACK something — the mirror of exhaustive set.
  trend       The same aggregates bucketed by year, over effective_date.

ONE RULE GOVERNS ALL THREE, and it is the difference between these being
useful and being dangerous:

    Every result reports its own coverage, and no result is ever a bare number.

An average over the 143 liability caps this corpus could parse is a statement
about 143 contracts, not about the 940 that exist. Reporting "the average
liability cap is Rs 208 million" without saying what it was computed over
invites a reader to treat it as a corpus fact. Every function here returns the
denominator alongside the number, broken down by why each excluded row was
excluded — absent, recorded as a cross-reference to a schedule, stated as a
formula, or unparseable.

The gap functions are the sharpest case. "Which contracts have no liability
cap" must never be answered from `amount IS NULL`, because on this corpus that
returns 780 contracts when only 560 genuinely lack one — the other 220 have a
cap recorded in a schedule or a cap the parser could not read. Those are
reported separately, as "not established here" rather than as absence.
"""

import logging

logger = logging.getLogger(__name__)

# Status values from services/normalize. Imported by name rather than value so
# the two cannot drift apart silently.
from services.normalize import OK, UNPARSED, ABSENT, REFERENCE, FORMULA  # noqa: E402


def _enabled() -> bool:
    import config
    return bool(getattr(config, "USE_DATABASE", False))


def _party_clause(parties: list[str] | None, params: dict) -> str:
    """SQL fragment restricting to documents naming every listed party."""
    if not parties:
        return ""
    frags = []
    for i, p in enumerate(parties):
        key = f"agg_party{i}"
        params[key] = f"%{p.strip()}%"
        frags.append(f"""EXISTS (
            SELECT 1 FROM documents dd
            CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(dd.parties,'[]'::jsonb)) AS pp(name)
            WHERE dd.wiki_id = c.wiki_id AND dd.session_id = c.session_id
              AND dd.source_doc = c.source_doc AND pp.name ILIKE :{key}
        )""")
    return " AND " + " AND ".join(frags)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate_liability_caps(wiki_id: str, session_id: str,
                             parties: list[str] | None = None,
                             doc_type: str | None = None) -> dict:
    """SUM/AVG/median over parsed liability caps, with full coverage reporting.

    Currency is reported but NOT converted — mixing INR and USD into one sum
    would produce a number that is wrong in every currency. When more than one
    appears, the aggregate is computed per currency and the caller is told.
    """
    if not _enabled():
        return {"error": "database not configured"}
    from sqlalchemy import text
    from services import db

    params: dict = {"w": wiki_id, "sid": session_id}
    where = "c.wiki_id = :w AND c.session_id = :sid"
    where += _party_clause(parties, params)
    if doc_type:
        params["dt"] = f"%{doc_type}%"
        where += """ AND EXISTS (SELECT 1 FROM documents d2 WHERE d2.wiki_id = c.wiki_id
                     AND d2.session_id = c.session_id AND d2.source_doc = c.source_doc
                     AND d2.doc_type ILIKE :dt)"""

    with db.get_engine().connect() as conn:
        coverage = {r[0] or "unknown": int(r[1]) for r in conn.execute(text(
            f"SELECT c.liability_cap_status, count(*) FROM contracts c "
            f"WHERE {where} GROUP BY 1"), params)}
        rows = conn.execute(text(f"""
            SELECT COALESCE(c.liability_cap_currency, 'unspecified') AS cur,
                   count(*), sum(c.liability_cap_amount), avg(c.liability_cap_amount),
                   min(c.liability_cap_amount), max(c.liability_cap_amount),
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY c.liability_cap_amount)
            FROM contracts c
            WHERE {where} AND c.liability_cap_amount IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """), params).fetchall()

    total = sum(coverage.values())
    by_currency = [{
        "currency": r[0], "contracts": int(r[1]),
        "sum": float(r[2]) if r[2] is not None else None,
        "mean": float(r[3]) if r[3] is not None else None,
        "min": float(r[4]) if r[4] is not None else None,
        "max": float(r[5]) if r[5] is not None else None,
        "median": float(r[6]) if r[6] is not None else None,
    } for r in rows]
    computed = sum(c["contracts"] for c in by_currency)
    return {
        "metric": "liability_cap",
        "by_currency": by_currency,
        "mixed_currency": len(by_currency) > 1,
        "computed_over": computed,
        "in_scope": total,
        "coverage": _coverage_note(coverage, computed, total, "liability cap"),
        "excluded": {k: v for k, v in coverage.items() if k != OK},
        "parties": parties or [], "doc_type": doc_type,
    }


def aggregate_contract_values(wiki_id: str, session_id: str,
                              parties: list[str] | None = None,
                              doc_type: str | None = None) -> dict:
    """Same shape, over clause-level contract-value figures."""
    if not _enabled():
        return {"error": "database not configured"}
    from sqlalchemy import text
    from services import db

    params: dict = {"w": wiki_id, "sid": session_id}
    where = ("c.wiki_id = :w AND c.session_id = :sid "
             "AND c.clause_type_canon = 'contract_value'")
    where += _party_clause(parties, params)
    if doc_type:
        params["dt"] = f"%{doc_type}%"
        where += """ AND EXISTS (SELECT 1 FROM documents d2 WHERE d2.wiki_id = c.wiki_id
                     AND d2.session_id = c.session_id AND d2.source_doc = c.source_doc
                     AND d2.doc_type ILIKE :dt)"""

    with db.get_engine().connect() as conn:
        coverage = {r[0] or "unknown": int(r[1]) for r in conn.execute(text(
            f"SELECT c.value_status, count(*) FROM clauses c WHERE {where} GROUP BY 1"),
            params)}
        rows = conn.execute(text(f"""
            SELECT COALESCE(c.value_currency, 'unspecified'),
                   count(DISTINCT c.source_doc), sum(c.value_amount),
                   avg(c.value_amount), min(c.value_amount), max(c.value_amount)
            FROM clauses c WHERE {where} AND c.value_amount IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        """), params).fetchall()

    total = sum(coverage.values())
    by_currency = [{
        "currency": r[0], "documents": int(r[1]),
        "sum": float(r[2]) if r[2] is not None else None,
        "mean": float(r[3]) if r[3] is not None else None,
        "min": float(r[4]) if r[4] is not None else None,
        "max": float(r[5]) if r[5] is not None else None,
    } for r in rows]
    computed = sum(c["documents"] for c in by_currency)
    return {
        "metric": "contract_value",
        "by_currency": by_currency,
        "mixed_currency": len(by_currency) > 1,
        "computed_over": computed, "in_scope": total,
        "coverage": _coverage_note(coverage, computed, total, "contract value"),
        "excluded": {k: v for k, v in coverage.items() if k != OK},
        "parties": parties or [], "doc_type": doc_type,
    }


def _coverage_note(coverage: dict, computed: int, total: int, label: str) -> str:
    """One plain sentence saying what the number was and was not computed over.

    Written here rather than in the UI so every consumer — API, chat answer,
    admin panel — carries the same caveat and none of them can drop it.
    """
    if not total:
        return f"No {label} data in scope."
    parts = []
    if coverage.get(REFERENCE):
        parts.append(f"{coverage[REFERENCE]} record the figure in a schedule or "
                     f"statement of work rather than inline")
    if coverage.get(FORMULA):
        parts.append(f"{coverage[FORMULA]} state it as a formula rather than an amount")
    if coverage.get(ABSENT):
        parts.append(f"{coverage[ABSENT]} state none at all")
    if coverage.get(UNPARSED):
        parts.append(f"{coverage[UNPARSED]} could not be read")
    tail = ("; " + ", ".join(parts)) if parts else ""
    return (f"Computed over {computed} of {total} document(s) in scope that state a "
            f"readable {label}{tail}.")


# ---------------------------------------------------------------------------
# gap / absence detection
# ---------------------------------------------------------------------------

# What each gap check looks for, and how it distinguishes real absence from
# "we could not read it". Keyed by the phrase a question is likely to use.
GAP_FIELDS = {
    "liability_cap": {
        "label": "liability cap",
        "table": "contracts", "status_col": "liability_cap_status",
    },
    "governing_law": {
        "label": "governing law",
        "table": "contracts", "plain_col": "governing_law",
    },
    "termination": {
        "label": "termination provision",
        "table": "contracts", "plain_col": "termination",
    },
}


def find_gaps(wiki_id: str, session_id: str, field: str,
              parties: list[str] | None = None, doc_type: str | None = None,
              limit: int = 50) -> dict:
    """Documents that genuinely LACK `field`, separated from ones we can't read.

    The separation is the entire point. A naive `IS NULL` gap query on this
    corpus reports 780 contracts with no liability cap when 560 genuinely have
    none — the other 220 record it in a schedule or could not be parsed. Telling
    a lawyer 220 contracts are uncapped when they are not is precisely the class
    of confident wrong answer this roadmap exists to remove.
    """
    if not _enabled():
        return {"error": "database not configured"}
    spec = GAP_FIELDS.get(field)
    if not spec:
        return {"error": f"unknown gap field {field!r}",
                "available": sorted(GAP_FIELDS)}
    from sqlalchemy import text
    from services import db

    params: dict = {"w": wiki_id, "sid": session_id, "lim": limit}
    where = "c.wiki_id = :w AND c.session_id = :sid"
    where += _party_clause(parties, params)
    if doc_type:
        params["dt"] = f"%{doc_type}%"
        where += """ AND EXISTS (SELECT 1 FROM documents d2 WHERE d2.wiki_id = c.wiki_id
                     AND d2.session_id = c.session_id AND d2.source_doc = c.source_doc
                     AND d2.doc_type ILIKE :dt)"""

    if spec.get("status_col"):
        col = spec["status_col"]
        missing_sql = f"c.{col} = '{ABSENT}'"
        # Everything that is neither present nor genuinely absent: reported
        # separately so it is never counted as a gap.
        unknown_sql = f"c.{col} IN ('{REFERENCE}', '{FORMULA}', '{UNPARSED}')"
    else:
        col = spec["plain_col"]
        missing_sql = f"(c.{col} IS NULL OR btrim(c.{col}) = '')"
        unknown_sql = "FALSE"

    with db.get_engine().connect() as conn:
        total = conn.execute(text(
            f"SELECT count(*) FROM contracts c WHERE {where}"), params).scalar() or 0
        missing = conn.execute(text(
            f"SELECT count(*) FROM contracts c WHERE {where} AND {missing_sql}"),
            params).scalar() or 0
        unknown = conn.execute(text(
            f"SELECT count(*) FROM contracts c WHERE {where} AND {unknown_sql}"),
            params).scalar() or 0
        rows = conn.execute(text(f"""
            SELECT c.source_doc, d2.doc_type, d2.effective_date
            FROM contracts c
            LEFT JOIN documents d2 ON d2.wiki_id = c.wiki_id
                 AND d2.session_id = c.session_id AND d2.source_doc = c.source_doc
            WHERE {where} AND {missing_sql}
            ORDER BY d2.effective_date DESC NULLS LAST
            LIMIT :lim
        """), params).fetchall()

    return {
        "field": field, "label": spec["label"],
        "in_scope": int(total),
        "missing": int(missing),
        "indeterminate": int(unknown),
        "present": int(total) - int(missing) - int(unknown),
        "documents": [{"source_doc": r[0], "doc_type": r[1],
                       "effective_date": str(r[2]) if r[2] else None} for r in rows],
        "truncated": int(missing) > len(rows),
        "note": (f"{missing} document(s) state no {spec['label']}. "
                 + (f"A further {unknown} could not be confirmed either way — the value is "
                    f"recorded elsewhere (a schedule or statement of work), stated as a "
                    f"formula, or could not be read — and are deliberately NOT counted as "
                    f"gaps." if unknown else "")),
        "parties": parties or [], "doc_type": doc_type,
    }


# ---------------------------------------------------------------------------
# trend over time
# ---------------------------------------------------------------------------

def trend_over_time(wiki_id: str, session_id: str, metric: str = "liability_cap",
                    parties: list[str] | None = None,
                    doc_type: str | None = None) -> dict:
    """Year-bucketed aggregate over documents.effective_date.

    Buckets with too few documents to mean anything are still returned, with
    their count, rather than smoothed away — a "trend" drawn over two contracts
    a year is not a trend, and the reader needs the n to see that.
    """
    if not _enabled():
        return {"error": "database not configured"}
    if metric not in ("liability_cap", "contract_value"):
        return {"error": f"unknown metric {metric!r}"}
    from sqlalchemy import text
    from services import db

    params: dict = {"w": wiki_id, "sid": session_id}
    if metric == "liability_cap":
        src = ("contracts c JOIN documents d2 ON d2.wiki_id = c.wiki_id "
               "AND d2.session_id = c.session_id AND d2.source_doc = c.source_doc")
        val = "c.liability_cap_amount"
    else:
        src = ("clauses c JOIN documents d2 ON d2.wiki_id = c.wiki_id "
               "AND d2.session_id = c.session_id AND d2.source_doc = c.source_doc")
        val = "c.value_amount"

    where = "c.wiki_id = :w AND c.session_id = :sid AND d2.effective_date IS NOT NULL"
    if metric == "contract_value":
        where += " AND c.clause_type_canon = 'contract_value'"
    where += _party_clause(parties, params)
    if doc_type:
        params["dt"] = f"%{doc_type}%"
        where += " AND d2.doc_type ILIKE :dt"

    # documents.effective_date is TEXT, and 76 values on this corpus are not in
    # ISO form — a date cast throws on them and takes the whole query down.
    # Pull the first 19xx/20xx year out by regex instead: it tolerates
    # "12 August 2020" and "2020-08-12" alike, and yields NULL rather than an
    # error for anything with no recognisable year, which is then reported as
    # undated rather than silently dropped.
    # NB: no '(?:...)' group here. SQLAlchemy's text() reads ':19' inside a
    # non-capturing group as a bind parameter named "19" and the statement
    # fails before it reaches Postgres. '[12][0-9]{3}' needs no colon.
    year_expr = "substring(d2.effective_date from '[12][0-9]{3}')::int"

    with db.get_engine().connect() as conn:
        rows = conn.execute(text(f"""
            SELECT {year_expr} AS yr,
                   count(*) AS docs,
                   count({val}) AS with_value,
                   avg({val}) AS mean,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY {val}) AS median
            FROM {src}
            WHERE {where} AND {year_expr} IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """), params).fetchall()
        undated = conn.execute(text(
            f"SELECT count(*) FROM {src} WHERE {where} AND {year_expr} IS NULL"),
            params).scalar() or 0

    buckets = [{"year": int(r[0]), "documents": int(r[1]), "with_value": int(r[2]),
                "mean": float(r[3]) if r[3] is not None else None,
                "median": float(r[4]) if r[4] is not None else None} for r in rows]
    usable = [b for b in buckets if b["with_value"] >= 3]
    direction = None
    if len(usable) >= 2 and usable[0]["median"] and usable[-1]["median"]:
        first, last = usable[0]["median"], usable[-1]["median"]
        change = (last - first) / first if first else 0
        if abs(change) >= 0.10:
            direction = "increasing" if change > 0 else "decreasing"
        else:
            direction = "broadly flat"
    return {
        "metric": metric, "buckets": buckets,
        "years_with_enough_data": len(usable),
        "direction": direction,
        "undated": int(undated),
        "note": (("Direction is read from year buckets holding at least 3 documents with "
                  "a readable value; buckets below that are shown with their count but "
                  "not used, because a trend drawn over one or two contracts is not one."
                  if usable else
                  "Not enough documents with both an effective date and a readable value "
                  "to describe a trend.")
                 + (f" {undated} document(s) in scope carry an effective date this could "
                    f"not read a year from and are excluded." if undated else "")),
        "parties": parties or [], "doc_type": doc_type,
    }
