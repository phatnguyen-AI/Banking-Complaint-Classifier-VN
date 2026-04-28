# Production Container - Banking Complaint Classifier
# Target: Render Free Plan (0.1 CPU, 512MB RAM)

# slim-bullseye over alpine: glibc compatibility with ONNX Runtime wheels
FROM python:3.9-slim-bullseye

# Single-threaded execution across all numerical libraries.
# PYTHONOPTIMIZE=2 strips docstrings/asserts from bytecode to save RAM.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    PYTHONOPTIMIZE=2 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# libgomp1: required by ONNX Runtime even in single-threaded mode
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 \
    && apt-get autoremove -y && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first for Docker layer caching (code changes more often)
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Model artifacts baked into image to avoid runtime downloads
RUN mkdir -p models/production backend
COPY models/production/ ./models/production/
COPY backend/api.py ./backend/api.py

# Non-root user for container security hardening
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Render uses this to detect unhealthy containers
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# Single worker: each fork duplicates the ~129MB model in memory.
# With 512MB RAM ceiling, only one worker fits safely.
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
