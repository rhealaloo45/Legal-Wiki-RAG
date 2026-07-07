"""
Global answer-style rules — a user-editable list of house-style instructions
("use a formal tone", "say clause not section", ...) that get appended to
every answer-generation prompt, on top of the fixed per-intent RULES already
in prompts.py.

Stored as a single JSON file (global, not per-session) since these are meant
to be shared house-style preferences across the whole app, not per-document-set
settings.
"""

import json
import logging
import os
import uuid

import config

logger = logging.getLogger(__name__)

RULES_PATH = os.path.join(config._APP_DIR, "data", "rules.json")


def _default_rules() -> list[dict]:
    defaults = [
        "Maintain a formal legal tone.",
        "Minimise jargon — define terms on first use.",
        "Use \"clauses/sub-clauses\" rather than \"paragraphs/sections\".",
        "Use actual party names, not generic labels.",
        "Reproduce monetary figures exactly — no rounding.",
        "Use active voice for obligations.",
        "Distinguish \"shall\" vs \"may\" vs \"provided that\".",
        "Flag ambiguity, gaps, and inconsistencies.",
        "Don't offer legal advice unless explicitly asked.",
    ]
    return [
        {"id": str(uuid.uuid4()), "text": text, "enabled": True, "predefined": True}
        for text in defaults
    ]


def load_rules() -> list[dict]:
    """Return the current global rules list, creating the defaults file if missing."""
    if not os.path.exists(RULES_PATH):
        rules = _default_rules()
        save_rules(rules)
        return rules
    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            rules = json.load(f)
        if not isinstance(rules, list):
            raise ValueError("rules.json did not contain a list")
        return rules
    except Exception as e:
        logger.warning("Failed to load rules.json (%s) — falling back to defaults", e)
        return _default_rules()


def save_rules(rules: list[dict]) -> None:
    os.makedirs(os.path.dirname(RULES_PATH), exist_ok=True)
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)


def add_rule(text: str) -> dict:
    text = (text or "").strip()
    rules = load_rules()
    rule = {"id": str(uuid.uuid4()), "text": text, "enabled": True, "predefined": False}
    rules.append(rule)
    save_rules(rules)
    return rule


def update_rule(rule_id: str, text: str | None = None, enabled: bool | None = None) -> dict | None:
    rules = load_rules()
    for r in rules:
        if r.get("id") == rule_id:
            if text is not None:
                r["text"] = text.strip()
            if enabled is not None:
                r["enabled"] = bool(enabled)
            save_rules(rules)
            return r
    return None


def delete_rule(rule_id: str) -> bool:
    rules = load_rules()
    new_rules = [r for r in rules if r.get("id") != rule_id]
    if len(new_rules) == len(rules):
        return False
    save_rules(new_rules)
    return True


def reorder_rules(ordered_ids: list[str]) -> list[dict]:
    """Rewrite the rules list in the given id order (drag-to-reorder).

    Any existing rule whose id isn't in ordered_ids is appended at the end,
    preserving it rather than silently dropping it if the client sent a stale list.
    """
    rules = load_rules()
    by_id = {r["id"]: r for r in rules}
    new_order = [by_id[i] for i in ordered_ids if i in by_id]
    missing = [r for r in rules if r["id"] not in ordered_ids]
    new_order.extend(missing)
    save_rules(new_order)
    return new_order


def reset_rules() -> list[dict]:
    """Discard all rules (including any custom ones) and restore the defaults."""
    rules = _default_rules()
    save_rules(rules)
    return rules


def enabled_rules_block() -> str:
    """Return a formatted prompt block of enabled rule texts, or "" if none are enabled."""
    rules = [r for r in load_rules() if r.get("enabled") and r.get("text")]
    if not rules:
        return ""
    lines = "\n".join(f"- {r['text']}" for r in rules)
    return f"\nADDITIONAL HOUSE-STYLE RULES (apply on top of the rules above):\n{lines}\n"
