"""Prompt library — reusable, wiki-scoped prompt templates ("Also fixing").

Distinct from services/rules.py's House Rules, which are global answer-style
instructions silently appended to every prompt. This is a library a person
picks FROM for one specific drafting or query request — templates carry
{{placeholder}} variables filled in at use time, and live per wiki like
Collections and Playbooks, not in one shared global file.

Kept deliberately small: this item sits under "Also fixing" in the target
architecture, not among the seven locked-in features, and nothing in the spec
asks for prompt versioning, sharing across wikis, or LLM-assisted authoring.
"""
from __future__ import annotations

import re

from services import db

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptLibraryError(Exception):
    """Conflicts a caller should report rather than swallow."""


def _text():
    from sqlalchemy import text
    return text


def _enabled() -> bool:
    import config
    return bool(config.USE_DATABASE)


def variables(body: str) -> list[str]:
    """Distinct {{placeholder}} names in a template body, in first-seen order."""
    seen: list[str] = []
    for m in _VAR_RE.finditer(body or ""):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def render(body: str, values: dict | None = None) -> dict:
    """Fill in {{placeholders}}. A variable with no supplied value is left as
    the literal placeholder rather than blanked or erroring — a half-filled
    template that still shows what's missing is safer to send than one with
    silent gaps where a name should be."""
    values = values or {}
    missing = [v for v in variables(body) if v not in values or values[v] in (None, "")]

    def _sub(m):
        name = m.group(1)
        v = values.get(name)
        return str(v) if v not in (None, "") else m.group(0)

    return {"text": _VAR_RE.sub(_sub, body or ""), "missing": missing}


def create(wiki_id: str, session_id: str, name: str, body: str,
           category: str | None = None) -> dict:
    name = (name or "").strip()
    body = (body or "").strip()
    if not name:
        raise PromptLibraryError("Template name cannot be empty")
    if not body:
        raise PromptLibraryError("Template body cannot be empty")
    if not _enabled():
        raise PromptLibraryError("Database not configured")
    text = _text()
    with db.get_engine().connect() as c:
        if c.execute(text("SELECT id FROM prompt_templates WHERE wiki_id=:w AND name=:n"),
                     {"w": wiki_id, "n": name}).scalar():
            raise PromptLibraryError(f"A template named {name!r} already exists in this wiki")
        row = c.execute(text("""
            INSERT INTO prompt_templates (wiki_id, session_id, name, category, body)
            VALUES (:w, :s, :n, :cat, :b) RETURNING id, created_at
        """), {"w": wiki_id, "s": session_id, "n": name,
               "cat": (category or "").strip() or None, "b": body}).fetchone()
        c.commit()
    return {"id": int(row[0]), "name": name, "category": category, "body": body,
            "variables": variables(body),
            "created_at": row[1].isoformat() if row[1] else None}


def list_all(wiki_id: str, category: str | None = None) -> list[dict]:
    if not _enabled():
        return []
    text = _text()
    clause = "AND category = :cat" if category else ""
    params = {"w": wiki_id}
    if category:
        params["cat"] = category
    with db.get_engine().connect() as c:
        rows = c.execute(text(f"""
            SELECT id, name, category, body, updated_at FROM prompt_templates
            WHERE wiki_id = :w {clause} ORDER BY category NULLS FIRST, name
        """), params).fetchall()
    return [{"id": int(r[0]), "name": r[1], "category": r[2],
             "body": r[3], "variables": variables(r[3]),
             "updated_at": r[4].isoformat() if r[4] else None} for r in rows]


def get(wiki_id: str, template_id: int) -> dict | None:
    if not _enabled():
        return None
    text = _text()
    with db.get_engine().connect() as c:
        row = c.execute(text("""
            SELECT id, name, category, body, created_at, updated_at
            FROM prompt_templates WHERE wiki_id=:w AND id=:i
        """), {"w": wiki_id, "i": template_id}).fetchone()
    if not row:
        return None
    return {"id": int(row[0]), "name": row[1], "category": row[2], "body": row[3],
            "variables": variables(row[3]),
            "created_at": row[4].isoformat() if row[4] else None,
            "updated_at": row[5].isoformat() if row[5] else None}


def update(wiki_id: str, template_id: int, name: str | None = None,
           body: str | None = None, category: str | None = None) -> bool:
    if not _enabled():
        return False
    sets, params = [], {"w": wiki_id, "i": template_id}
    if name is not None:
        n = name.strip()
        if not n:
            raise PromptLibraryError("Template name cannot be empty")
        sets.append("name = :n")
        params["n"] = n
    if body is not None:
        b = body.strip()
        if not b:
            raise PromptLibraryError("Template body cannot be empty")
        sets.append("body = :b")
        params["b"] = b
    if category is not None:
        sets.append("category = :cat")
        params["cat"] = category.strip() or None
    if not sets:
        return False
    sets.append("updated_at = now()")
    text = _text()
    with db.get_engine().connect() as c:
        if name is not None:
            clash = c.execute(text("""
                SELECT id FROM prompt_templates WHERE wiki_id=:w AND name=:n AND id<>:i
            """), params).scalar()
            if clash:
                raise PromptLibraryError(f"A template named {name!r} already exists")
        res = c.execute(text(
            f"UPDATE prompt_templates SET {', '.join(sets)} WHERE wiki_id=:w AND id=:i"),
            params)
        c.commit()
    return (res.rowcount or 0) > 0


def delete(wiki_id: str, template_id: int) -> bool:
    if not _enabled():
        return False
    text = _text()
    with db.get_engine().connect() as c:
        res = c.execute(text("DELETE FROM prompt_templates WHERE wiki_id=:w AND id=:i"),
                        {"w": wiki_id, "i": template_id})
        c.commit()
    return (res.rowcount or 0) > 0


def resolve(wiki_id: str, ref) -> int | None:
    if ref is None:
        return None
    text = _text()
    with db.get_engine().connect() as c:
        if isinstance(ref, int) or (isinstance(ref, str) and str(ref).isdigit()):
            return c.execute(text(
                "SELECT id FROM prompt_templates WHERE wiki_id=:w AND id=:i"),
                {"w": wiki_id, "i": int(ref)}).scalar()
        return c.execute(text(
            "SELECT id FROM prompt_templates WHERE wiki_id=:w AND name=:n"),
            {"w": wiki_id, "n": str(ref).strip()}).scalar()


def categories(wiki_id: str) -> list[str]:
    text = _text()
    with db.get_engine().connect() as c:
        return [r[0] for r in c.execute(text("""
            SELECT DISTINCT category FROM prompt_templates
            WHERE wiki_id=:w AND category IS NOT NULL ORDER BY 1
        """), {"w": wiki_id})]
