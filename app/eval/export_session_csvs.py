"""One-shot export of one wiki session's rows to CSV, for moving a local DB
snapshot to another machine (see diff_sync_session.py for the compare/push
side once you're there).

Usage:
    python eval/export_session_csvs.py <session_id> <output_dir>
"""
import os
import sys

import psycopg2

TABLES = ["pages", "relations", "page_embeddings_azure", "page_metadata",
          "contradictions", "source_positions", "clause_map"]


def main():
    session_id, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    conn = psycopg2.connect(os.environ["LOCAL_DB_URL"])
    with conn.cursor() as cur:
        for table in TABLES:
            path = os.path.join(out_dir, f"{table}.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                cur.copy_expert(
                    f"COPY (SELECT * FROM {table} WHERE session_id = "
                    f"{psycopg2.extensions.QuotedString(session_id)}) "
                    f"TO STDOUT WITH CSV",
                    f,
                )
            print(f"{table}: wrote {path}")
    conn.close()


if __name__ == "__main__":
    main()
