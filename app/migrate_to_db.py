"""
One-time migration: copy all existing index.json files into PostgreSQL.

Usage:
    python migrate_to_db.py

Requires DATABASE_URL to be set in the environment (or in .env).
Safe to run multiple times — ON CONFLICT DO NOTHING skips existing rows.
After a successful migration, each index.json is renamed to index.json.migrated.
"""
import os
import sys
import logging

# Ensure the app directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    if not config.USE_DATABASE:
        logger.error("DATABASE_URL is not set. Nothing to migrate to.")
        sys.exit(1)

    from services import db as _db

    wiki_root = config.WIKI_PATH
    if not os.path.isdir(wiki_root):
        logger.info("No wiki directory found at %s — nothing to migrate.", wiki_root)
        return

    sessions_migrated = 0
    sessions_skipped = 0

    for session_id in os.listdir(wiki_root):
        session_dir = os.path.join(wiki_root, session_id)
        if not os.path.isdir(session_dir):
            continue

        json_path = os.path.join(session_dir, "index.json")
        if not os.path.exists(json_path):
            continue

        existing = _db.count_pages(_db.DEFAULT_WIKI_ID, session_id)
        if existing > 0:
            logger.info("Session %s already has %d pages in DB — skipping.", session_id, existing)
            sessions_skipped += 1
            continue

        logger.info("Migrating session %s ...", session_id)
        try:
            _db.migrate_from_json(_db.DEFAULT_WIKI_ID, session_id, json_path)
            os.rename(json_path, json_path + ".migrated")
            sessions_migrated += 1
        except Exception as e:
            logger.error("Failed to migrate session %s: %s", session_id, e)

    logger.info(
        "Migration complete: %d sessions migrated, %d skipped (already in DB).",
        sessions_migrated, sessions_skipped,
    )


if __name__ == "__main__":
    main()
