# syntax=docker/dockerfile:1
# =============================================================================
# Agentic GameDev Pipeline — Dockerfile
# =============================================================================
# Multi-stage build:
#   1. base    — Python 3.12 + Playwright system deps
#   2. install — pip deps + Playwright Chromium
#   3. runtime — entrypoint
# =============================================================================

FROM python:3.12-slim-bookworm AS base

LABEL description="Agentic GameDev Pipeline — Scout, Dev, QA"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -- Stage 2: install --------------------------------------------------------
FROM base AS install

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install chromium --with-deps 2>/dev/null || true

# -- Stage 3: runtime --------------------------------------------------------
FROM install AS runtime

ENV LLM_ENDPOINT="http://host.docker.internal:11434/v1" \
    LLM_MODEL="qwen2.5:7b" \
    LLM_MAX_TOKENS="8192" \
    DRY_RUN="false" \
    LOG_LEVEL="INFO"

COPY . .

ENTRYPOINT ["python", "orchestrator.py"]
