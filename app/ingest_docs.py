"""
ingest_docs.py — ingest specific documents into an existing wiki session, standalone

Runs as its own process, completely independent of the Flask dev server's
ThreadPoolExecutor — safe to run alongside another ingest already in flight
(e.g. a large batch upload via the UI) without competing for those worker
slots. It also doesn't touch app.py's "last-completed-ingest becomes the
main session" logic (that only lives in the /upload route's _check_completion,
not in wiki.ingest() itself), so it can't flip which session is "main"
mid-run the way a second /upload call could.

It does still call the same Azure OpenAI deployment as whatever else is
running, so heavy concurrent use elsewhere can slow this down (or trigger
429s) — it just can't corrupt or interfere with the other job's data, since
each session's rows are independent.

Usage:
    cd app
    python3 ingest_docs.py <session_id> <file1> [file2] [file3] ...

Example:
    python3 ingest_docs.py 3a66b1... "data/uploads/New Doc 1.pdf" "data/uploads/New Doc 2.pdf"
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("ingest_docs")

import config
from services import wiki


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    session_id = sys.argv[1]
    file_paths = sys.argv[2:]

    missing = [f for f in file_paths if not os.path.isfile(f)]
    if missing:
        for f in missing:
            print(f"File not found: {f}")
        sys.exit(1)

    if not config.USE_DATABASE:
        print("USE_DATABASE is not enabled — this script requires the Postgres-backed wiki.")
        sys.exit(1)

    print(f"Ingesting {len(file_paths)} document(s) into session '{session_id}' "
          f"(merges into whatever wiki content already exists there)")

    for fp in file_paths:
        doc_name = os.path.basename(fp)
        print(f"  -> {doc_name} ...")
        try:
            result = wiki.ingest(fp, session_id)
            pages = (result or {}).get("pages_updated", 0)
            print(f"     done: {pages} pages updated")
        except Exception as e:
            logger.error("Failed to ingest %s: %s", doc_name, e)
            print(f"     FAILED: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
