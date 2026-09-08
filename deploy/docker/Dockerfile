# Containerizes the existing CLI pipeline (ingest / build_index / ask /
# agent_ask / run_eval). The Streamlit demo runs on Streamlit Community
# Cloud directly from the repo (no container involved there) -- this
# Dockerfile exists to containerize the CLI tooling, not to front the demo.
#
# Build from the REPO ROOT (not this directory) so the COPY commands
# below can see pyproject.toml/uv.lock/src at their real paths:
#   docker build -f deploy/docker/Dockerfile -t findocqa .
# A fresh container has no local search index (data/processed/ is
# gitignored) -- download the one already cached in MongoDB (fast) rather
# than rebuilding from scratch, into a named volume so it persists across
# runs:
#   docker run --rm --env-file .env -v findocqa-data:/app/data findocqa \
#     uv run python -c "from findocqa.retrieval import index_cache; index_cache.download_index('table_aware')"
# Then run any script against that same volume:
#   docker run --rm --env-file .env -v findocqa-data:/app/data findocqa uv run scripts/ask.py "..."
#
# Verified for real (not just built): `uv run pytest` (48 passed) and a
# live end-to-end question both run correctly inside this image --
# see TECHNICAL_REPORT.md for the actual verification run.

FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY . .
RUN uv sync --no-dev

CMD ["python"]
