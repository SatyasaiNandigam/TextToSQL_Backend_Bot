FROM python:3.12-slim

# System deps:
#   build-essential + libpq-dev: required for psycopg2-binary native extension
#   curl: used by healthcheck
#   ca-certificates: needed for HTTPS to HuggingFace Hub during model download
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first — maximizes Docker layer cache reuse.
# Only these two files are needed for uv sync.
COPY pyproject.toml uv.lock ./

# Install all project dependencies into the system Python.
# UV_SYSTEM_PYTHON=1: install into system Python (not a venv) — replaces removed --system flag (uv >= 0.11)
# --frozen: use exact locked versions, do not update the lock file
# --no-cache: avoids uv's own cache dir bloating the layer
ENV UV_SYSTEM_PYTHON=1
RUN uv sync --frozen --no-cache

# Pre-download the HuggingFace embedding model at build time.
# This avoids a 30-60s download delay on every cold container start.
# We import SentenceTransformer directly to avoid importing schema_rag_pipeline.py,
# which calls create_engine() at module level and requires a live DATABASE_URL.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading BAAI/bge-small-en-v1.5 ...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Model cached.')"

# Copy full application source.
# chroma_db/ is included as a baseline — overlaid by bind-mount at runtime.
# .dockerignore excludes: .env, .venv, __pycache__, datasets/, experiments/, etc.
COPY . .

EXPOSE 8000

# Non-root user for security. UID 1000 matches typical host user UID,
# which avoids permission issues on the ./chroma_db bind-mount.
RUN useradd --uid 1000 --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# start_period=60s accounts for PyTorch + model weights loading into memory
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
