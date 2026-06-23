# ── Legal Wiki RAG — Application Image ────────────────────────────────────────
# Build context: repo root  (docker build -t legal-wiki .)
# ------------------------------------------------------------------------------

FROM python:3.11-slim

# ── OS-level dependencies ──────────────────────────────────────────────────────
# • libgl1 / libglib2.0-0  → PyMuPDF / OpenCV shared libs
# • tesseract-ocr           → pytesseract OCR backend
# • gcc / libpq-dev         → psycopg2-binary native build (fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY app/ .

# Persistent data directory (mounted as a volume in docker-compose)
RUN mkdir -p data/uploads data/wiki

# ── Runtime ────────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    PORT=5001

EXPOSE 5001

# Use gunicorn in production; falls back to Flask dev server if gunicorn absent
CMD ["sh", "-c", \
     "gunicorn -w 2 -b 0.0.0.0:${PORT} app:app 2>/dev/null \
      || python app.py"]
