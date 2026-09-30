"""Fingerprint every clause and page quote line already in a wiki session.

    python backfill_clause_templates.py <wiki_id> <session_id>

Makes no model calls: it reads clause and page text already stored and writes
one hash row per unit to clause_templates (see services/templates.py). Safe to
re-run; each run replaces the session's rows in a single transaction. New
documents are fingerprinted when they finish ingest, so this is only needed for
documents ingested before the table existed.
"""
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from services import templates  # noqa: E402  (after load_dotenv: config reads the env)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    print(templates.backfill(sys.argv[1], sys.argv[2]))
