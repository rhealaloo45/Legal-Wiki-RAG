"""Identify which chat sessions hold the v1 audit run.

chat_messages spans every session this project has ever run, including older and
failed ones. Matching a question against all of them pulls answers from the wrong
run — confirmed: a GridEdge question matched an OmniRetail answer, and one
question matched a stored "LLM unavailable: 404" error at ratio 1.00. So the
candidate pool has to be restricted to the audit's own sessions first.
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sqlalchemy import text

from services import db


def main() -> None:
    engine = db.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT session_id,
                   count(*)                        AS msgs,
                   min(created_at)                 AS first_at,
                   max(created_at)                 AS last_at
            FROM chat_messages
            GROUP BY session_id
            HAVING count(*) >= 6
            ORDER BY min(created_at) DESC
            LIMIT 25
        """)).fetchall()
    for r in rows:
        print(f"{r.session_id[:40]:42} msgs={r.msgs:<5} {r.first_at:%Y-%m-%d %H:%M} -> {r.last_at:%H:%M}")


if __name__ == "__main__":
    main()
