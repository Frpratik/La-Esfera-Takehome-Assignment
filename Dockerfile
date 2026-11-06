# API image: FastAPI + FAISS index baked in at build time.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /srv

# Install CPU-only torch first. The default wheel bundles CUDA libraries (~2 GB) that a
# CPU-only service never uses; sentence-transformers then reuses this torch.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Dependencies before source code so this layer is cached across code edits.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model into the image so the container works without internet.
# If the download fails (offline build) we continue: the embedder falls back to hashing.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" \
    || echo "WARNING: model download failed; index will use the hash embedder"

COPY app ./app
COPY ingest ./ingest
COPY data/policies ./data/policies

# Bake the index into the image: a fresh container answers immediately, with no ingestion
# step at startup. The entrypoint rebuilds it only if it is missing (e.g. a wiped volume).
RUN python -m ingest.build_index

COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Run as a non-root user; it only needs to read the app and write the index/SQLite files.
RUN useradd --create-home --uid 1000 app && chown -R app:app /srv
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

ENTRYPOINT ["/entrypoint.sh"]
