"""
Authentication — single-user login.

Scope is deliberately narrow, per the target architecture doc § 01.4 (Access &
Admin Document Lifecycle): ONE password-gated account with full access to
everything — ingestion, Review Queue, document management, chat, every query
mode. No role split, no per-user chat isolation, no ACL. Those are explicitly
deferred, not forgotten.

The one thing borrowed early from the deferred design is the `users.role`
column. The doc notes the users table and role column are "cheap to add later"
— which is true of the column but only if the table exists to add it to. It's
inert this pass (nothing reads it) and costs one column now instead of a
migration later. Per-user chat isolation is the part that genuinely can't be
retrofitted cheaply (it needs a user_id FK on conversation records from the
start), and that is NOT being built here — see the doc before adding it.

Hardening below is the floor-level set § 01.6 lists as non-deferrable even at
single-user scope:
  - password hashing       → scrypt (memory-hard, salted, one-way)
  - login rate limiting    → DB-backed, holds across gunicorn workers
  - HTTPS-only cookies     → set in app.py from config.SESSION_COOKIE_SECURE

On scrypt vs. the bcrypt/argon2 the doc names: werkzeug 3.x ships scrypt
(32768:8:1) as its default, and it's the same class of memory-hard, salted,
non-reversible KDF. It costs zero new dependencies — bcrypt and argon2 both
pull native builds that are a known friction point on Windows dev machines.
If that tradeoff ever changes, _HASH_METHOD below is the only line to touch;
existing hashes keep verifying either way, since the method is encoded in the
stored hash string itself.
"""

import logging

import config

logger = logging.getLogger(__name__)

# generate_password_hash() is called without an explicit `method` on purpose:
# werkzeug's default is scrypt as of 3.x, and following the library default
# means this gets stronger on upgrade instead of being pinned to whatever was
# current the day it was written. The method is encoded in each stored hash,
# so old hashes keep verifying after a default change.

# Verifying a password against a real hash takes measurable time; returning
# instantly for an unknown username would leak which usernames exist via
# response timing. A throwaway hash of a dummy password gives the
# no-such-user path the same cost profile as a wrong-password one.
_DUMMY_HASH = None


def _get_dummy_hash() -> str:
    """Lazily built (hashing is deliberately slow — don't pay for it at import)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        from werkzeug.security import generate_password_hash
        _DUMMY_HASH = generate_password_hash("dummy-password-never-matches")
    return _DUMMY_HASH


def _require_db():
    """Auth stores credentials in Postgres — refuse to half-work without it."""
    if not config.USE_DATABASE:
        raise RuntimeError(
            "AUTH_ENABLED is true but DATABASE_URL is not set — the users table "
            "lives in Postgres. Set DATABASE_URL, or set AUTH_ENABLED=false to "
            "deliberately run without a login gate."
        )
    from services import db
    return db


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

def user_count() -> int:
    """How many accounts exist. Used at boot to warn when none are set up."""
    db = _require_db()
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        return conn.execute(text("SELECT count(*) FROM users")).scalar() or 0


def get_user(username: str) -> dict | None:
    db = _require_db()
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT id, username, password_hash, role FROM users WHERE username = :u"),
            {"u": username},
        ).mappings().first()
    return dict(row) if row else None


def set_password(username: str, password: str) -> None:
    """Create the account, or reset its password if it already exists.

    Deliberately the only way a password enters the system — there is no
    seed-from-env path. An env var holding the login password would sit in
    .env in plaintext indefinitely, which is a worse steady state than a
    one-time CLI invocation. See manage_user.py.
    """
    if not username or not username.strip():
        raise ValueError("Username cannot be empty")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    from werkzeug.security import generate_password_hash
    db = _require_db()
    from sqlalchemy import text

    pw_hash = generate_password_hash(password)
    with db.get_engine().connect() as conn:
        conn.execute(
            text("""
                INSERT INTO users (username, password_hash)
                VALUES (:u, :h)
                ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash
            """),
            {"u": username.strip(), "h": pw_hash},
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _count_recent_failures(conn, text, column: str, value: str) -> int:
    if not value:
        return 0
    return conn.execute(
        text(f"""
            SELECT count(*) FROM login_attempts
            WHERE {column} = :v
              AND success = false
              AND created_at > now() - make_interval(mins => :mins)
        """),
        {"v": value, "mins": config.LOGIN_RATE_WINDOW_MINUTES},
    ).scalar() or 0


def _seconds_until_unlock(conn, text, column: str, value: str, limit: int) -> int:
    """How long until enough failures age out of the window to allow a retry.

    Reads the timestamp of the Nth-most-recent failure (N = limit); once that
    one leaves the rolling window, the caller is back under the threshold.
    """
    row = conn.execute(
        text(f"""
            SELECT EXTRACT(EPOCH FROM (
                       created_at + make_interval(mins => :mins) - now()
                   ))::int AS wait_s
            FROM login_attempts
            WHERE {column} = :v
              AND success = false
              AND created_at > now() - make_interval(mins => :mins)
            ORDER BY created_at DESC
            OFFSET :off LIMIT 1
        """),
        {"v": value, "mins": config.LOGIN_RATE_WINDOW_MINUTES, "off": limit - 1},
    ).first()
    return max(int(row[0]), 1) if row and row[0] is not None else 60


def check_rate_limit(username: str, ip: str) -> int:
    """Return seconds to wait if locked out, or 0 if the attempt may proceed.

    Two independent counters. The per-username one is the real control at
    single-user scope; the per-IP one is looser and catches an attacker
    spraying different usernames from one source.
    """
    db = _require_db()
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        if username:
            fails = _count_recent_failures(conn, text, "username", username)
            if fails >= config.LOGIN_MAX_FAILURES_PER_USER:
                return _seconds_until_unlock(
                    conn, text, "username", username, config.LOGIN_MAX_FAILURES_PER_USER
                )
        if ip:
            fails = _count_recent_failures(conn, text, "ip", ip)
            if fails >= config.LOGIN_MAX_FAILURES_PER_IP:
                return _seconds_until_unlock(
                    conn, text, "ip", ip, config.LOGIN_MAX_FAILURES_PER_IP
                )
    return 0


def record_attempt(username: str, ip: str, success: bool) -> None:
    """Log an evaluated attempt. Never called for rate-limit rejections — an
    attempt that was refused before the password was even checked shouldn't
    extend its own lockout (that turns the limiter into a self-sustaining DoS
    against the only real user)."""
    db = _require_db()
    from sqlalchemy import text
    try:
        with db.get_engine().connect() as conn:
            conn.execute(
                text("INSERT INTO login_attempts (username, ip, success) VALUES (:u, :i, :s)"),
                {"u": (username or "")[:200], "i": (ip or "")[:100], "s": success},
            )
            conn.commit()
    except Exception as e:
        # Logging an attempt must never be what breaks a valid login.
        logger.error("Failed to record login attempt: %s", e)


def prune_old_attempts(days: int = 90) -> int:
    """Trim the attempt log. Nothing calls this automatically yet — the table
    grows slowly at single-user scale. Exposed for manage_user.py."""
    db = _require_db()
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        result = conn.execute(
            text("DELETE FROM login_attempts WHERE created_at < now() - make_interval(days => :d)"),
            {"d": days},
        )
        conn.commit()
        return result.rowcount or 0


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def verify_login(username: str, password: str, ip: str) -> tuple[dict | None, str, int]:
    """Authenticate a login attempt.

    Returns (user, error_message, retry_after_seconds). On success the user
    dict is returned and error is empty; on failure user is None and error
    carries a message safe to show the client. retry_after is non-zero only
    when the caller is rate-limited.
    """
    from werkzeug.security import check_password_hash

    username = (username or "").strip()

    retry_after = check_rate_limit(username, ip)
    if retry_after:
        logger.warning("Login rate-limited for user=%r ip=%r (%ss)", username, ip, retry_after)
        return None, "Too many failed attempts. Try again later.", retry_after

    user = get_user(username) if username else None

    if user is None:
        # Equalize timing against the wrong-password path (see _DUMMY_HASH).
        check_password_hash(_get_dummy_hash(), password or "")
        record_attempt(username, ip, False)
        return None, "Incorrect username or password.", 0

    if not check_password_hash(user["password_hash"], password or ""):
        record_attempt(username, ip, False)
        return None, "Incorrect username or password.", 0

    record_attempt(username, ip, True)
    _touch_last_login(user["id"])
    return user, "", 0


def _touch_last_login(user_id: int) -> None:
    db = _require_db()
    from sqlalchemy import text
    try:
        with db.get_engine().connect() as conn:
            conn.execute(
                text("UPDATE users SET last_login_at = now() WHERE id = :i"),
                {"i": user_id},
            )
            conn.commit()
    except Exception as e:
        logger.error("Failed to update last_login_at: %s", e)


def client_ip(request) -> str:
    """Best-effort client IP.

    Behind Azure App Service (and any reverse proxy) remote_addr is the proxy,
    so the leftmost X-Forwarded-For hop is the real client. That header is
    caller-controlled and therefore spoofable, which means the per-IP limit
    can be evaded by an attacker rotating it — the per-username limit is the
    one that actually holds, and it doesn't depend on this value at all.
    """
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or ""
