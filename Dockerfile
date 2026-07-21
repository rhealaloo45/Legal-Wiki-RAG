# ── Legal Wiki RAG — Application Image ────────────────────────────────────────
# Build context: repo root  (docker build -t legal-wiki .)
# ------------------------------------------------------------------------------

FROM python:3.11-slim

# ── OS-level dependencies ──────────────────────────────────────────────────────
# All Python deps use pre-compiled binary wheels — no compiler or native headers needed.
# libglib2.0-0 is a lightweight shim some glib-linked wheels dlopen at runtime.
RUN find /etc/apt -name "*.sources" -o -name "*.list" \
        | xargs sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' \
    && apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
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
