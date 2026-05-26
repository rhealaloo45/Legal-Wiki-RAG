"""
Shared file reading module with OCR fallback for scanned/image-based PDFs.

Extraction strategy per PDF page:
  1. Try pypdf text extraction (fast, works for digital/text-layer PDFs).
  2. If a page yields < MIN_CHARS_PER_PAGE characters, render it as an image
     via pymupdf and OCR it with pytesseract (accurate, slower).
  3. Combine all pages into a single text string.

Graceful degradation: if pytesseract or pymupdf aren't installed,
falls back to pypdf-only extraction (original behavior) and logs a warning
so the user knows scanned pages were skipped.
"""

import io
import os
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# If pypdf extracts fewer characters than this from a single page,
# treat it as a scanned/image page and attempt OCR.
MIN_CHARS_PER_PAGE = 50

# ---------------------------------------------------------------------------
# OCR dependency check — done once at import time
# ---------------------------------------------------------------------------
_ocr_available = False
try:
    import pytesseract
    import fitz  # pymupdf
    from PIL import Image

    _ocr_available = True
    logger.info("OCR support enabled (pytesseract + pymupdf + Pillow)")
except ImportError as e:
    logger.warning(
        "OCR dependencies not fully available (%s). Scanned PDFs will have "
        "limited text extraction. Install with: pip install pymupdf pytesseract Pillow  "
        "and ensure Tesseract is installed on your system.",
        e,
    )


def configure_tesseract(cmd_path: str | None = None):
    """Set the Tesseract executable path (useful on Windows).

    Call this once at startup if Tesseract is not on PATH, e.g.:
        configure_tesseract(r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    """
    if cmd_path and _ocr_available:
        import pytesseract as _pt
        _pt.pytesseract.tesseract_cmd = cmd_path
        logger.info("Tesseract path set to: %s", cmd_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def read_file(file_path: str) -> str:
    """Read text from a .txt or .pdf file, using OCR for scanned pages.

    Returns the full document text as a single string.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = _read_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    # Collapse excessive whitespace (shared cleanup)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PDF reading with per-page OCR fallback
# ---------------------------------------------------------------------------
def _read_pdf(file_path: str) -> str:
    """Read PDF, falling back to OCR for pages where pypdf extracts little text."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    fname = os.path.basename(file_path)

    # --- First pass: pypdf text extraction ---
    pypdf_texts: list[str] = []
    needs_ocr: list[int] = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        pypdf_texts.append(text)
        if len(text) < MIN_CHARS_PER_PAGE:
            needs_ocr.append(i)

    # If every page had enough text, we're done
    if not needs_ocr:
        return "\n".join(pypdf_texts)

    # If OCR isn't available, warn and return what we have
    if not _ocr_available:
        logger.warning(
            "%d/%d pages in %s appear to be scanned but OCR is not available. "
            "Install pymupdf, pytesseract, and Pillow for OCR support.",
            len(needs_ocr), num_pages, fname,
        )
        return "\n".join(pypdf_texts)

    # --- Second pass: OCR only the pages that need it ---
    logger.info(
        "OCR: processing %d/%d scanned pages in %s",
        len(needs_ocr), num_pages, fname,
    )

    try:
        doc = fitz.open(file_path)
        for i in needs_ocr:
            try:
                ocr_text = _ocr_page(doc[i])
                if ocr_text.strip():
                    pypdf_texts[i] = ocr_text
                    logger.debug(
                        "  OCR page %d/%d: extracted %d chars",
                        i + 1, num_pages, len(ocr_text),
                    )
            except Exception as e:
                logger.warning("  OCR failed for page %d of %s: %s", i + 1, fname, e)
        doc.close()
    except Exception as e:
        logger.error("Failed to open %s with pymupdf for OCR: %s", fname, e)

    ocr_count = sum(
        1 for i in needs_ocr if len(pypdf_texts[i]) >= MIN_CHARS_PER_PAGE
    )
    logger.info("OCR complete for %s: %d/%d pages successfully OCR'd", fname, ocr_count, len(needs_ocr))

    return "\n".join(pypdf_texts)


def _ocr_page(page) -> str:
    """Render a single pymupdf page to an image and OCR it with Tesseract.

    Uses 300 DPI for high accuracy on legal documents, and psm 6
    (assume a single uniform block of text).
    """
    # Render at 300 DPI (default PDF is 72 DPI)
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert pymupdf pixmap → PIL Image → Tesseract
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    text = pytesseract.image_to_string(
        img,
        lang="eng",
        config="--psm 6",  # uniform block of text — best for full-page docs
    )
    return text
