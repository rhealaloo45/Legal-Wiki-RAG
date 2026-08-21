"""
Account management CLI for the single-user login.

The only way to set a password. Deliberately interactive — the password is
read via getpass rather than taken as an argument, so it never lands in shell
history, and there is no seed-from-env path because an env var holding the
login password would sit in .env in plaintext indefinitely.

Usage (from the repo root, with the venv active):

    python app/manage_user.py set-password admin
    python app/manage_user.py list
    python app/manage_user.py prune-attempts 90
    python app/manage_user.py backfill-chat-owner admin

Runs against whatever DATABASE_URL the current .env points at — check with
`app/switch-env.ps1` first if you keep per-branch env files.
"""

import os
import sys
import getpass

sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: E402
from services import auth  # noqa: E402


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def cmd_set_password(username: str) -> None:
    pw = getpass.getpass(f"New password for {username!r}: ")
    if not pw:
        _fail("Password cannot be empty.")
    if pw != getpass.getpass("Confirm password: "):
        _fail("Passwords do not match.")

    try:
        auth.set_password(username, pw)
    except ValueError as e:
        _fail(str(e))

    print(f"Password set for {username!r}.")


def cmd_list() -> None:
    from sqlalchemy import text
    from services import db

    with db.get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT username, role, created_at, last_login_at
            FROM users ORDER BY username
        """)).mappings().all()

    if not rows:
        print("No accounts yet. Create one with:  python app/manage_user.py set-password admin")
        return

    print(f"{'USERNAME':<24} {'ROLE':<10} {'CREATED':<22} LAST LOGIN")
    for r in rows:
        created = r["created_at"].strftime("%Y-%m-%d %H:%M") if r["created_at"] else "-"
        last = r["last_login_at"].strftime("%Y-%m-%d %H:%M") if r["last_login_at"] else "never"
        print(f"{r['username']:<24} {r['role']:<10} {created:<22} {last}")


def cmd_backfill_chat_owner(username: str) -> None:
    """Assign every unattributed chat_messages row to one account.

    Only correct while exactly one account exists — then every historical
    message provably belongs to it. Refuses to run otherwise rather than
    silently mis-attributing another user's history. See the user_id column
    comment in services/db.py.
    """
    from sqlalchemy import text
    from services import db

    total = auth.user_count()
    if total != 1:
        _fail(
            f"{total} accounts exist. Backfill is only unambiguous with exactly one — "
            "with more, historical messages can't be attributed automatically."
        )

    user = auth.get_user(username)
    if not user:
        _fail(f"No such account: {username!r}")

    with db.get_engine().connect() as conn:
        updated = conn.execute(
            text("UPDATE chat_messages SET user_id = :uid WHERE user_id IS NULL"),
            {"uid": user["id"]},
        ).rowcount
        conn.commit()

    print(f"Attributed {updated} previously unowned chat message(s) to {username!r}.")


def cmd_prune_attempts(days: int) -> None:
    deleted = auth.prune_old_attempts(days)
    print(f"Deleted {deleted} login attempt record(s) older than {days} days.")


def main() -> None:
    if not config.USE_DATABASE:
        _fail("DATABASE_URL is not set — accounts live in Postgres. Check your .env.")

    args = sys.argv[1:]
    if not args:
        print(__doc__.strip())
        sys.exit(2)

    cmd = args[0]
    if cmd == "set-password":
        if len(args) < 2:
            _fail("Usage: manage_user.py set-password <username>")
        cmd_set_password(args[1])
    elif cmd == "list":
        cmd_list()
    elif cmd == "prune-attempts":
        cmd_prune_attempts(int(args[1]) if len(args) > 1 else 90)
    elif cmd == "backfill-chat-owner":
        if len(args) < 2:
            _fail("Usage: manage_user.py backfill-chat-owner <username>")
        cmd_backfill_chat_owner(args[1])
    else:
        _fail(
            f"Unknown command {cmd!r}. Expected: "
            "set-password | list | prune-attempts | backfill-chat-owner"
        )


if __name__ == "__main__":
    main()
