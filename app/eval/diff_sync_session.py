"""Diff one wiki session between two Postgres databases and push only the delta.

Compares LOCAL_DB_URL against AZURE_DB_URL for one session_id, across every
table that carries wiki content, and reports (or applies) the minimal set of
INSERT / UPDATE / delete-candidate changes needed to make Azure match local
for that session — instead of a full delete+reload.

Usage:
    LOCAL_DB_URL=postgresql://postgres:pw@localhost:5433/legal_wiki \
    AZURE_DB_URL=postgresql://...  \
    python eval/diff_sync_session.py <session_id> [--apply]

Without --apply, this is a dry run: it prints counts and a sample of what
would change, and touches neither database.

Deliberately does NOT auto-delete rows that exist on Azure but not locally —
those are reported as "azure-only" so a human decides whether they're a
legitimate Azure-side addition or genuine drift to clean up by hand.
"""
import os
import sys

import psycopg2
import psycopg2.extras

# (table, natural-key columns, insertable columns i.e. everything except
# autoincrement id and the generated content_tsv column)
TABLES = [
    ("pages", ["title"],
     ["title", "content", "summary", "source_doc", "contradiction_flagged",
      "variants", "append_count", "char_count", "last_modified"]),
    ("relations", ["from_title", "to_title", "label"],
     ["from_title", "to_title", "label"]),
    ("page_embeddings_azure", ["title"],
     ["title", "embedding", "doc_family"]),
    ("page_metadata", ["title"],
     ["title", "governing_law", "jurisdiction", "effective_date",
      "termination_notice", "liability_cap", "ip_ownership", "parties",
      "auto_renewal", "notice_period", "payment_terms", "matter_reference",
      "doc_type", "doc_family"]),
    ("contradictions", ["page_title", "claim", "value_a", "value_b"],
     ["page_title", "claim", "value_a", "source_a", "value_b", "source_b",
      "detected_at"]),
    ("source_positions", ["source_doc", "page_num"],
     ["source_doc", "page_num", "char_start", "char_end"]),
    ("clause_map", ["source_doc", "clause_num"],
     ["source_doc", "clause_num", "heading", "page_title"]),
]


def fetch_rows(conn, table: str, cols: list[str], session_id: str) -> dict:
    """{natural_key_tuple: row_dict} for every row of this session."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        col_list = ", ".join(cols)
        # embedding is pgvector's own type — cast to text so it compares and
        # serializes like every other column instead of needing the pgvector
        # python extension registered on both connections.
        col_list = col_list.replace("embedding", "embedding::text")
        cur.execute(
            f"SELECT {col_list} FROM {table} WHERE session_id = %s",
            (session_id,),
        )
        rows = cur.fetchall()
    key_cols = TABLES_KEY_MAP[table]
    out = {}
    for r in rows:
        key = tuple(r[k] for k in key_cols)
        out[key] = dict(r)
    return out


def diff_table(local_rows: dict, azure_rows: dict):
    local_keys = set(local_rows)
    azure_keys = set(azure_rows)
    to_insert = local_keys - azure_keys
    azure_only = azure_keys - local_keys
    common = local_keys & azure_keys
    to_update = {k for k in common if local_rows[k] != azure_rows[k]}
    return to_insert, to_update, azure_only


def apply_changes(azure_conn, table: str, cols: list[str], key_cols: list[str],
                   local_rows: dict, to_insert: set, to_update: set):
    with azure_conn.cursor() as cur:
        col_list = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        where_clause = " AND ".join(f"{k} = %s" for k in key_cols)
        delete_sql = f"DELETE FROM {table} WHERE {where_clause}"

        for key in to_update:
            cur.execute(delete_sql, key)
        for key in to_update | to_insert:
            row = local_rows[key]
            values = [row[c] for c in cols]
            cur.execute(insert_sql, values)
    azure_conn.commit()


TABLES_KEY_MAP = {t: k for t, k, _ in TABLES}


def main():
    if len(sys.argv) < 2:
        print("Usage: python diff_sync_session.py <session_id> [--apply]")
        sys.exit(1)
    session_id = sys.argv[1]
    apply = "--apply" in sys.argv[2:]

    local_url = os.environ["LOCAL_DB_URL"]
    azure_url = os.environ["AZURE_DB_URL"]

    local_conn = psycopg2.connect(local_url)
    azure_conn = psycopg2.connect(azure_url)

    print(f"Diffing session {session_id!r} — local vs azure "
          f"({'APPLY' if apply else 'DRY RUN'})\n")

    grand_insert = grand_update = grand_azure_only = 0

    for table, key_cols, cols in TABLES:
        local_rows = fetch_rows(local_conn, table, cols, session_id)
        azure_rows = fetch_rows(azure_conn, table, cols, session_id)
        to_insert, to_update, azure_only = diff_table(local_rows, azure_rows)

        print(f"{table}: local={len(local_rows)} azure={len(azure_rows)} "
              f"-> insert={len(to_insert)} update={len(to_update)} "
              f"azure_only={len(azure_only)}")
        if azure_only:
            sample = list(azure_only)[:5]
            print(f"    azure-only keys (not auto-deleted, review by hand): {sample}")

        grand_insert += len(to_insert)
        grand_update += len(to_update)
        grand_azure_only += len(azure_only)

        if apply and (to_insert or to_update):
            apply_changes(azure_conn, table, cols, key_cols, local_rows,
                          to_insert, to_update)
            print(f"    applied {len(to_insert)} insert(s), {len(to_update)} update(s)")

    print(f"\nTotals: {grand_insert} to insert, {grand_update} to update, "
          f"{grand_azure_only} azure-only (not touched).")
    if not apply and (grand_insert or grand_update):
        print("Dry run only — re-run with --apply to push these changes.")

    local_conn.close()
    azure_conn.close()


if __name__ == "__main__":
    main()
