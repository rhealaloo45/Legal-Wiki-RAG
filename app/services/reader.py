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

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# If pypdf extracts fewer characters than this from a single page,
# treat it as a scanned/image page and attempt OCR.
MIN_CHARS_PER_PAGE = 50

# OCR hang protection: Tesseract can hang indefinitely on a malformed/complex
# image, which — with no timeout — freezes the whole ingest worker thread and
# is a primary cause of ingestion "getting stuck". Each Tesseract invocation is
# capped at OCR_PAGE_TIMEOUT_SECS and retried up to OCR_MAX_ATTEMPTS times; if it
# still fails, that OCR attempt is skipped and ingestion moves on to the next
# PSM mode / page / document rather than blocking forever.
OCR_PAGE_TIMEOUT_SECS = int(os.getenv("OCR_PAGE_TIMEOUT_SECS", "30"))
OCR_MAX_ATTEMPTS = int(os.getenv("OCR_MAX_ATTEMPTS", "2"))

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
        # Set on both the top-level module and the submodule to ensure
        # the path is found regardless of which attribute the library
        # version checks internally.
        pytesseract.tesseract_cmd = cmd_path
        pytesseract.pytesseract.tesseract_cmd = cmd_path
        logger.info("Tesseract path set to: %s", cmd_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def read_file_with_positions(file_path: str) -> dict:
    """Read a file and return text with page-level character positions.

    Returns: {text: str, page_map: [{page_num, char_start, char_end}, ...]}
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _read_pdf_with_positions(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    return {
        "text": text,
        "page_map": [{"page_num": 1, "char_start": 0, "char_end": len(text)}],
    }


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
    text = text.strip()

    # Warn if extraction yielded very little text (likely a scanned/image PDF
    # where OCR failed or wasn't available)
    fname = os.path.basename(file_path)
    if len(text) < 100:
        logger.warning(
            "EXTRACTION WARNING: '%s' yielded only %d chars. "
            "This file may be a scanned/image PDF that needs better OCR, "
            "or the file may be empty/corrupted.",
            fname, len(text),
        )
    elif len(text) < 500:
        logger.warning(
            "EXTRACTION WARNING: '%s' yielded only %d chars (low). "
            "Some pages may not have been extracted correctly.",
            fname, len(text),
        )

    return text


# ---------------------------------------------------------------------------
# PDF reading with per-page OCR fallback
# ---------------------------------------------------------------------------
def _read_pdf_with_positions(file_path: str) -> dict:
    """Read PDF and return text with per-page character offsets."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    page_texts: list[str] = []

    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        page_texts.append(text)

    if _ocr_available:
        needs_ocr = [i for i, t in enumerate(page_texts) if len(t) < MIN_CHARS_PER_PAGE]
        if needs_ocr:
            try:
                doc = fitz.open(file_path)
                for i in needs_ocr:
                    try:
                        ocr_text = _ocr_page(doc[i])
                        if ocr_text.strip():
                            page_texts[i] = ocr_text
                    except Exception:
                        pass
                doc.close()
            except Exception:
                pass

    full_text = "\n".join(page_texts)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]{2,}", " ", full_text)
    full_text = full_text.strip()

    page_map = []
    offset = 0
    for i, pt in enumerate(page_texts):
        cleaned = re.sub(r"\n{3,}", "\n\n", pt)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        start = full_text.find(cleaned, max(0, offset - 50)) if cleaned else offset
        if start == -1:
            start = offset
        end = start + len(cleaned)
        page_map.append({"page_num": i + 1, "char_start": start, "char_end": end})
        offset = end

    return {"text": full_text, "page_map": page_map}


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


def _preprocess_for_ocr(img: "Image.Image") -> "Image.Image":
    """Apply image preprocessing to improve OCR accuracy on scanned/redacted PDFs.

    Steps:
      1. Convert to grayscale
      2. Increase contrast (optional, via histogram stretching)
      3. Apply Otsu binarization for clean black/white text
      4. Remove noise via median filter

    Falls back to the original image if any step fails (e.g. if numpy is missing).
    """
    try:
        from PIL import ImageFilter, ImageOps

        # 1. Grayscale
        gray = img.convert("L")

        # 2. Auto-contrast — stretches histogram to use full 0-255 range
        contrasted = ImageOps.autocontrast(gray, cutoff=1)

        # 3. Otsu-like binarization via simple threshold
        #    (Use Pillow's built-in point operation; no numpy needed)
        threshold = _otsu_threshold(contrasted)
        binary = contrasted.point(lambda px: 255 if px > threshold else 0, mode="1")

        # 4. Light noise removal — median filter
        cleaned = binary.convert("L").filter(ImageFilter.MedianFilter(size=3))

        return cleaned
    except Exception as e:
        logger.debug("Image preprocessing failed, using raw image: %s", e)
        return img


def _otsu_threshold(gray_img: "Image.Image") -> int:
    """Compute an Otsu-like threshold from a grayscale PIL Image.

    This avoids a numpy dependency — builds a 256-bin histogram from Pillow,
    then sweeps for the threshold that minimises intra-class variance.
    """
    hist = gray_img.histogram()  # 256 entries
    total_pixels = sum(hist)
    if total_pixels == 0:
        return 128

    sum_total = sum(i * hist[i] for i in range(256))
    sum_bg = 0.0
    weight_bg = 0
    best_thresh = 128
    best_var = 0.0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t

    return best_thresh


def _ocr_with_retry(img, psm: str) -> str:
    """Run one Tesseract OCR pass with a hard timeout and bounded retries.

    Tesseract with no timeout can hang forever on a bad image and freeze the
    ingest worker. Each invocation is time-boxed to OCR_PAGE_TIMEOUT_SECS
    (pytesseract raises RuntimeError on timeout) and retried up to
    OCR_MAX_ATTEMPTS times; if every attempt times out or errors, returns ""
    so the caller skips this pass and continues rather than blocking.
    """
    for attempt in range(1, OCR_MAX_ATTEMPTS + 1):
        try:
            return pytesseract.image_to_string(
                img,
                lang="eng",
                config=f"--psm {psm}",
                timeout=OCR_PAGE_TIMEOUT_SECS,
            )
        except Exception as e:
            logger.warning(
                "OCR attempt %d/%d (psm=%s) failed/timed out after %ds: %s",
                attempt, OCR_MAX_ATTEMPTS, psm, OCR_PAGE_TIMEOUT_SECS, e,
            )
    logger.warning("OCR skipped for one page (psm=%s) after %d attempts", psm, OCR_MAX_ATTEMPTS)
    return ""


_VISION_OCR_PROMPT = (
    "Transcribe every word of visible text from this scanned legal document page, "
    "exactly as it appears, in reading order. Include headings, numbered clauses, "
    "table contents (row by row), and signature-block labels. Do not summarise, "
    "translate, correct spelling/grammar, or add commentary — output the raw "
    "transcription only, no preamble."
)


def _ocr_page_azure_vision(page) -> str:
    """Render a page and OCR it via the Azure OpenAI vision-capable deployment.

    Used instead of Tesseract when OCR_ENGINE=azure_vision — helpful for scans
    Tesseract garbles (skew, low DPI originals, redaction artefacts) since the
    model reads the rendered image directly rather than running local OCR.
    """
    import base64
    from services import llm

    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    image_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")

    try:
        text, _usage = llm.ask_vision(image_b64, _VISION_OCR_PROMPT, max_tokens=4096, fast=True)
        return text.strip()
    except Exception as e:
        logger.warning("Azure vision OCR failed for a page: %s", e)
        return ""


def _ocr_page(page) -> str:
    """Render a single pymupdf page to an image and OCR it.

    Uses 300 DPI for high accuracy on legal documents. Routes to the Azure
    vision deployment when OCR_ENGINE=azure_vision; otherwise runs local
    Tesseract, trying multiple PSM modes and keeping the result with the most
    extracted text. Every Tesseract call is time-boxed and retried (see
    _ocr_with_retry) so a page that Tesseract hangs on is skipped instead of
    freezing the whole ingest.
    """
    if config.OCR_ENGINE == "azure_vision":
        return _ocr_page_azure_vision(page)

    # Render at 300 DPI (default PDF is 72 DPI)
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert pymupdf pixmap → PIL Image
    raw_img = Image.open(io.BytesIO(pix.tobytes("png")))

    # Preprocess for better OCR
    processed_img = _preprocess_for_ocr(raw_img)

    # Try multiple PSM modes and keep the best result
    best_text = ""
    for psm in ("6", "3", "4"):  # 6=uniform block, 3=auto, 4=single column
        text = _ocr_with_retry(processed_img, psm)
        if len(text.strip()) > len(best_text.strip()):
            best_text = text

    # If preprocessed image gave poor results, try raw image as fallback
    if len(best_text.strip()) < MIN_CHARS_PER_PAGE:
        raw_text = _ocr_with_retry(raw_img, "6")
        if len(raw_text.strip()) > len(best_text.strip()):
            best_text = raw_text

    return best_text
