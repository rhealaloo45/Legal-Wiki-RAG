"""
File-upload validation gate — target architecture § Phase 0-parallel,
"File-upload validation" (malformed/malicious PDF handling, before parsing
starts).

Runs synchronously inside /upload, right after a file is saved to disk and
before it's queued for background ingest via the executor. Rejects anything
that would otherwise reach pypdf/python-docx/Tesseract un-vetted:
decompression-bomb PDFs, corrupted structures, oversized files, and files
whose actual bytes don't match their claimed extension.

Structural PDF probing (opening the file, counting pages, touching the first
page's content stream) can genuinely hang on a hostile file — pypdf has no
built-in timeout, unlike pytesseract's `timeout=` kwarg used elsewhere in
reader.py. _run_with_timeout bounds that probe from a worker thread; a
timeout is treated as a validation failure. The probing thread itself can't
be safely force-killed (Python has no safe thread-kill primitive), so a
genuine hang leaks one background thread rather than freezing the /upload
request — an accepted tradeoff over blocking indefinitely.
"""

import concurrent.futures
import logging
import os

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = {
    ".pdf": int(os.getenv("MAX_UPLOAD_PDF_MB", "50")),
    ".docx": int(os.getenv("MAX_UPLOAD_DOCX_MB", "25")),
    ".txt": int(os.getenv("MAX_UPLOAD_TXT_MB", "10")),
}
MAX_PDF_PAGES = int(os.getenv("MAX_UPLOAD_PDF_PAGES", "2000"))
PROBE_TIMEOUT_SECS = int(os.getenv("UPLOAD_PROBE_TIMEOUT_SECS", "20"))

# Extension → the byte prefix a genuine file of that type must start with.
# .txt has no reliable magic bytes so it's intentionally absent here.
_MAGIC_BYTES = {
    ".pdf": b"%PDF-",
    ".docx": b"PK\x03\x04",  # .docx is a zip archive
}

# Small, dedicated pool so a hung probe can never starve the app's main
# request-handling capacity — mirrors the isolation OCR_PAGE_TIMEOUT_SECS
# gives Tesseract calls in reader.py.
_probe_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="upload-probe"
)


def _run_with_timeout(fn, timeout_secs):
    future = _probe_executor.submit(fn)
    return future.result(timeout=timeout_secs)


def _probe_pdf(file_path: str) -> None:
    """Raise if this PDF is unsafe or too costly to ingest.

    Deliberately touches page 0's content stream, not just PdfReader() and
    len(reader.pages) — a decompression bomb or corrupt object stream
    typically only blows up once something actually decodes the stream, not
    at the point the page tree is parsed.
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    if reader.is_encrypted:
        raise ValueError("PDF is password-protected/encrypted")
    num_pages = len(reader.pages)
    if num_pages == 0:
        raise ValueError("PDF has no pages")
    if num_pages > MAX_PDF_PAGES:
        raise ValueError(f"PDF has {num_pages} pages, exceeds cap of {MAX_PDF_PAGES}")
    reader.pages[0].extract_text()


def _probe_docx(file_path: str) -> None:
    """Raise if this isn't actually a valid .docx (corrupt/non-docx zip)."""
    from docx import Document

    Document(file_path)


def validate_upload(file_path: str, original_filename: str) -> tuple[bool, str]:
    """Validate a just-saved upload before it's queued for ingest.

    Returns (True, "") if safe to ingest, else (False, reason). Only
    inspects file_path — the caller owns deleting it and excluding it from
    the session on a False result.
    """
    ext = os.path.splitext(original_filename)[1].lower()

    try:
        size = os.path.getsize(file_path)
    except OSError as e:
        return False, f"could not read saved file: {e}"

    if size == 0:
        return False, "file is empty"

    cap_mb = MAX_FILE_SIZE_MB.get(ext)
    if cap_mb is not None and size > cap_mb * 1024 * 1024:
        return False, f"file is {size / 1024 / 1024:.1f} MB, exceeds the {cap_mb} MB cap for {ext}"

    magic = _MAGIC_BYTES.get(ext)
    if magic:
        try:
            with open(file_path, "rb") as f:
                head = f.read(len(magic))
        except OSError as e:
            return False, f"could not read saved file: {e}"
        if head != magic:
            return False, f"file content doesn't match its {ext} extension"

    try:
        if ext == ".pdf":
            _run_with_timeout(lambda: _probe_pdf(file_path), PROBE_TIMEOUT_SECS)
        elif ext == ".docx":
            _run_with_timeout(lambda: _probe_docx(file_path), PROBE_TIMEOUT_SECS)
        # .txt needs no structural probe — it's read as raw bytes, nothing to parse
    except concurrent.futures.TimeoutError:
        logger.warning(
            "Upload probe timed out after %ds for %r (possible decompression bomb)",
            PROBE_TIMEOUT_SECS, original_filename,
        )
        return False, f"file took too long to open (>{PROBE_TIMEOUT_SECS}s) — likely corrupt or a decompression bomb"
    except Exception as e:
        logger.warning("Upload probe rejected %r: %s", original_filename, e)
        return False, f"file failed validation: {e}"

    return True, ""
