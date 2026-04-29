# syntax=docker/dockerfile:1
# ---- Stage 1: Builder ----
# Has build tools needed to compile native extensions (psycopg2, etc.).
# Nothing from this stage leaks into the final image.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first to maximise layer-cache reuse.
COPY pyproject.toml uv.lock ./

# Install production deps into .venv, using a BuildKit cache mount so
# the uv wheel cache persists across builds (never lands in the image layer).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- Stage 2: Runtime ----
# Clean slim base — no compilers, no build artefacts.
FROM python:3.12-slim AS runtime

# Only runtime system libs needed:
#   libpq-dev: required by psycopg2 at runtime
#   curl: used by healthcheck
#   ca-certificates: required for HTTPS to HuggingFace Hub
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the compiled venv from the builder — no compilers follow.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download the HuggingFace embedding model into the image so cold
# container starts don't incur a 30-60s download delay.
# The BuildKit cache mount keeps the downloaded weights across rebuilds
# so the model is only re-fetched when the sentence-transformers version
# or model name changes.
ENV HF_HOME=/app/.cache/huggingface
RUN --mount=type=cache,target=/tmp/hf-cache \
    HF_HOME=/tmp/hf-cache python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading BAAI/bge-small-en-v1.5 ...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Model cached.')" \
    && mkdir -p /app/.cache/huggingface \
    && cp -r /tmp/hf-cache/. /app/.cache/huggingface/

# Copy application source.
# chroma_db/ is included as a baseline — overlaid by bind-mount at runtime.
# .dockerignore excludes: .env, .venv, __pycache__, datasets/, experiments/, etc.
COPY . .

EXPOSE 8000

# Non-root user for security. UID 1000 matches typical host user UID,
# which avoids permission issues on the ./chroma_db bind-mount.
RUN useradd --uid 1000 --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# start_period=60s accounts for PyTorch + model weights loading into memory.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
