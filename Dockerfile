# ═══════════════════════════════════════════════════════════════════════════════
# ICSR Multi-Agent HITL Server — Dockerfile
# ═══════════════════════════════════════════════════════════════════════════════
#
# Multi-stage build:
#   Stage 1 (builder): Install Python dependencies in an isolated layer.
#   Stage 2 (runtime): Minimal runtime image — no build tools.
#
# Build:
#   docker build -t icsr-hitl:latest .
#
# Run (with .env file):
#   docker run --env-file .env -p 8000:8000 -v $(pwd)/audit:/app/audit icsr-hitl:latest
#
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System libraries needed to compile/link:
#   - libgomp1: OpenMP runtime (required by faiss-cpu)
#   - libxml2-dev, libxslt-dev: required by lxml
#   - build-essential: C compiler for packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libxml2-dev \
        libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency manifest first (better layer caching)
COPY pyproject.toml ./

# Install all project dependencies into /install prefix
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir \
        "fastapi>=0.111" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.7" \
        "pydantic-settings>=2.3" \
        "langgraph>=0.2" \
        "langchain-community>=0.2" \
        "langgraph-checkpoint-sqlite>=3.1" \
        "bcrypt>=4.0" \
        "presidio-analyzer>=2.2" \
        "presidio-anonymizer>=2.2" \
        "cryptography>=42.0" \
        "ollama>=0.3" \
        "faiss-cpu>=1.8" \
        "sentence-transformers>=3.0" \
        "owlready2>=0.46" \
        "openpyxl>=3.1" \
        "requests>=2.31" \
        "pandas>=2.2" \
        "pyarrow>=16.0" \
        "lxml>=5.2" \
        "numpy>=1.26" \
        "networkx>=3.3" \
        "python-multipart>=0.0.9" \
        "httpx>=0.27"

# Download the spaCy NLP model (needed by presidio-analyzer)
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -m spacy download en_core_web_lg --target /install/lib/python3.11/site-packages


# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Only the runtime shared libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for least-privilege execution
RUN groupadd --gid 1001 icsr \
 && useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false icsr

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=icsr:icsr . .

# Create writable data directories with correct ownership
RUN mkdir -p audit checkpoints data/pii_maps data/oae.faiss data/ctcae.faiss \
 && chown -R icsr:icsr audit checkpoints data

USER icsr

# Expose HITL API port
EXPOSE 8000

# Health check — Docker/K8s will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Single Uvicorn worker (Semaphore(1) on OllamaClient requires single process)
CMD ["uvicorn", "hitl.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-config", "/dev/null"]
