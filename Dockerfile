# Dockerfile
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install UV for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy project metadata first (layer caching — only reinstalls deps if
# pyproject.toml changes, not on every source code change)
COPY pyproject.toml ./
COPY .env.example .env

# Install dependencies
RUN uv pip install --system -e ".[dev]" && \
    uv pip install --system pydantic-settings

# Pre-download embedding and reranker models during build so they are
# cached in the image layer. Avoids downloading on every container start.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy source code
COPY src/ src/
COPY ui/ ui/
COPY scripts/ scripts/

# Copy only the eval questions — not the full data/ directory.
# Raw HTML, ChromaDB, and processed data are mounted as volumes at runtime
# (see docker-compose.yml). Baking large data into the image is an
# anti-pattern: it bloats the image and prevents data updates without rebuild.
COPY data/eval/ data/eval/

EXPOSE 8000 8501