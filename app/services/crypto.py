"""
Encryption at rest (target architecture § 01.6 Hardening).

Real client documents are privileged, confidential legal material. The doc's
position is that this should be on from day one rather than bolted on once
client data is already flowing — so this ships before the first real corpus
lands, not after.

WHAT THIS COVERS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
Covered by this module, at the application layer:

  * Uploaded source files on disk. They are read once at ingest and never
    searched in place, so encrypting them costs nothing but a decrypt on read.
  * Extracted structured text: clause verbatim text, typed values, party
    lists, queued review values. None of it is searched by SQL predicate.

NOT covered, and this is a deliberate engineering limit rather than an
oversight: `pages.content`. Postgres builds `content_tsv` from it as a
`GENERATED ALWAYS ... STORED` tsvector with a GIN index, and hybrid retrieval
searches that index. Encrypting the column would have Postgres index
ciphertext — keyword search would silently return nothing, which is worse
than an honest gap because it looks like the corpus simply has no match.

The correct control for that column is storage-level encryption — an
encrypted volume, or Postgres TDE — which protects it at rest without the
database losing the ability to index it. That is a deployment requirement,
recorded here and in the architecture doc rather than papered over with
application code that would break search to look thorough.

KEY HANDLING
------------
`ENCRYPTION_KEY` holds a Fernet key. `ENCRYPTION_PASSPHRASE` is accepted as
an alternative and stretched with PBKDF2-SHA256; that path exists for
convenience and is weaker than a generated key, so it warns.

Ciphertext carries a `lwenc:v1:` prefix. Every decrypt accepts unprefixed
input and returns it unchanged, which is what makes this safe to switch on
over an existing corpus: rows written before the key existed keep working,
and encryption applies going forward instead of demanding a migration before
anything can be read.

With no key configured the module is inert — encrypt() is identity — so the
system runs exactly as before rather than failing closed on a dev machine.
That is a deliberate default for local work; the architecture doc records
that a deployment holding real client documents must set a key.
"""
from __future__ import annotations

import base64
import logging
import os

logger = logging.getLogger(__name__)

PREFIX = "lwenc:v1:"
_FILE_MAGIC = b"LWENC1\x00"

_fernet = None
_initialized = False
_warned_no_key = False


def _load() -> None:
    global _fernet, _initialized, _warned_no_key
    if _initialized:
        return
    _initialized = True

    key = (os.getenv("ENCRYPTION_KEY") or "").strip()
    passphrase = (os.getenv("ENCRYPTION_PASSPHRASE") or "").strip()
    if not key and not passphrase:
        if not _warned_no_key:
            logger.info(
                "Encryption at rest is OFF (no ENCRYPTION_KEY set). Fine for "
                "local work; a deployment holding real client documents must "
                "set one — see services/crypto.py."
            )
            _warned_no_key = True
        return

    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.error("ENCRYPTION_KEY is set but the `cryptography` package is "
                     "not installed — refusing to run with encryption silently "
                     "disabled. Install cryptography or unset the key.")
        raise

    if not key:
        # Passphrase path. A fixed salt is a real weakness — it makes the
        # derived key attackable by precomputation — but a random per-process
        # salt would derive a different key each restart and make every
        # existing ciphertext unreadable. Named here so the tradeoff is
        # visible; a generated ENCRYPTION_KEY avoids it entirely.
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        logger.warning(
            "Deriving the encryption key from ENCRYPTION_PASSPHRASE. This is "
            "weaker than a generated ENCRYPTION_KEY — prefer "
            "Fernet.generate_key() and store the result."
        )
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                         salt=b"legal-wiki-rag/v1", iterations=480_000)
        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))

    try:
        _fernet = Fernet(key if isinstance(key, bytes) else key.encode("utf-8"))
    except Exception as err:
        logger.error("ENCRYPTION_KEY is not a valid Fernet key: %s", err)
        raise
    logger.info("Encryption at rest is ON (AES-128-CBC + HMAC via Fernet)")


def is_enabled() -> bool:
    _load()
    return _fernet is not None


def generate_key() -> str:
    """Mint a key for an operator to put in .env. Never called automatically —
    a key generated at runtime would differ per process and make yesterday's
    ciphertext unreadable."""
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode("ascii")


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------

def encrypt(value):
    """Encrypt a string. Returns the input unchanged when no key is set, and
    passes through anything already encrypted so a double call is harmless."""
    if value is None or value == "":
        return value
    if not isinstance(value, str):
        return value
    _load()
    if _fernet is None or value.startswith(PREFIX):
        return value
    try:
        return PREFIX + _fernet.encrypt(value.encode("utf-8")).decode("ascii")
    except Exception as err:
        logger.error("Encryption failed, storing plaintext rather than losing "
                     "the value: %s", err)
        return value


def decrypt(value):
    """Decrypt a string written by encrypt().

    Unprefixed input is returned unchanged — that is what lets encryption be
    switched on over a corpus that already exists, instead of requiring every
    historical row to be migrated before anything is readable.
    """
    if not isinstance(value, str) or not value.startswith(PREFIX):
        return value
    _load()
    if _fernet is None:
        # Encrypted data with no key is not something to paper over: returning
        # the ciphertext would put `lwenc:v1:gAAAA...` in front of a lawyer as
        # if it were clause text.
        raise RuntimeError(
            "Encrypted value found but no ENCRYPTION_KEY is configured. The "
            "key that wrote this data must be restored to read it."
        )
    from cryptography.fernet import InvalidToken
    try:
        return _fernet.decrypt(value[len(PREFIX):].encode("ascii")).decode("utf-8")
    except InvalidToken as err:
        raise RuntimeError(
            "Could not decrypt a stored value — the configured ENCRYPTION_KEY "
            "does not match the one that wrote it."
        ) from err


def decrypt_safe(value, default=None):
    """decrypt(), but returns `default` instead of raising. For read paths that
    render many rows, where one unreadable row should not blank the page."""
    try:
        return decrypt(value)
    except Exception as err:
        logger.error("Could not decrypt a stored value: %s", err)
        return default


def encrypt_json(value):
    """Encrypt a dict/list for a JSONB column, as {"enc": "<ciphertext>"}.

    Kept as valid JSON rather than a bare string so the column stays a legal
    JSONB value and nothing that reads it generically breaks on the type.
    """
    if value is None:
        return None
    _load()
    if _fernet is None:
        return value
    import json
    if isinstance(value, dict) and set(value.keys()) == {"enc"}:
        return value
    return {"enc": encrypt(json.dumps(value, ensure_ascii=False, default=str))}


def decrypt_json(value):
    if not isinstance(value, dict) or set(value.keys()) != {"enc"}:
        return value
    import json
    raw = decrypt_safe(value["enc"])
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def encrypt_file(path: str) -> bool:
    """Encrypt an uploaded file in place. Returns True if it was encrypted.

    Written to a temp file and swapped with os.replace, which is atomic on
    both POSIX and Windows — a crash mid-write must never leave a source
    document truncated, since at that point the plaintext is already gone.
    """
    _load()
    if _fernet is None:
        return False
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if data.startswith(_FILE_MAGIC):
            return True
        token = _fernet.encrypt(data)
        tmp = path + ".enc.tmp"
        with open(tmp, "wb") as fh:
            fh.write(_FILE_MAGIC)
            fh.write(token)
        os.replace(tmp, path)
        return True
    except Exception as err:
        logger.error("Could not encrypt %s, leaving it as-is: %s", path, err)
        try:
            if os.path.exists(path + ".enc.tmp"):
                os.remove(path + ".enc.tmp")
        except OSError:
            pass
        return False


def is_encrypted_file(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(_FILE_MAGIC)) == _FILE_MAGIC
    except OSError:
        return False


def decrypt_file_to_temp(path: str) -> str | None:
    """Decrypt an encrypted upload to a temp file, returning its path.

    A temp file rather than bytes in memory because the readers (pypdf,
    PyMuPDF, python-docx) all take a path, and rewriting them to take handles
    would be a much larger change than encryption warrants. The caller must
    delete it — see reader._decrypted_source().
    """
    if not is_encrypted_file(path):
        return None
    _load()
    if _fernet is None:
        raise RuntimeError(
            f"{os.path.basename(path)} is encrypted but no ENCRYPTION_KEY is "
            "configured. The key that wrote it must be restored to read it."
        )
    import tempfile
    from cryptography.fernet import InvalidToken
    with open(path, "rb") as fh:
        fh.read(len(_FILE_MAGIC))
        token = fh.read()
    try:
        plain = _fernet.decrypt(token)
    except InvalidToken as err:
        raise RuntimeError(
            f"Could not decrypt {os.path.basename(path)} — the configured "
            "ENCRYPTION_KEY does not match the one that encrypted it."
        ) from err
    suffix = os.path.splitext(path)[1] or ".bin"
    fd, tmp_path = tempfile.mkstemp(prefix="lwdec_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(plain)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path
