# Containerizes the existing CLI pipeline (ingest / build_index / ask /
# agent_ask / run_eval) -- not a long-running service yet, since the
# FastAPI/Streamlit apps this would eventually front don't exist as part
# of this project yet.
#
# Build from the REPO ROOT (not this directory) so the COPY commands
# below can see pyproject.toml/uv.lock/src at their real paths:
#   docker build -f deploy/docker/Dockerfile -t findocqa .
# Run any script with:
#   docker run --env-file .env findocqa uv run scripts/ask.py "..."
#
# NOTE: written and reviewed carefully, but not build-tested locally --
# Docker isn't installed in this project's development environment (see
# deploy/aws/DEPLOYMENT.md). CI can build it (see .github/workflows) if
# that verification is wanted; documented as unverified rather than
# claimed working.

FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY . .
RUN uv sync --no-dev

CMD ["python"]
