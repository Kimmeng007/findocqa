# FinDocQA

Agentic RAG system for SEC financial filings (10-K/10-Q), evaluated against
the public [FinanceBench](https://github.com/patronus-ai/financebench)
150-question benchmark. See [findocqa-project-spec.md](findocqa-project-spec.md)
for the full project goals, scope, and week-by-week plan, and
[PROGRESS.md](PROGRESS.md) / [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) /
[DATASET.md](DATASET.md) / [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for detailed
write-ups of what's been built, real bugs found and fixed, and how the
system actually works end-to-end.

**Status:** Weeks 1–5 of the 6-week plan are built:
- **Week 1** — EDGAR ingestion, MongoDB storage, naive baseline RAG
- **Week 2** — table-aware chunking, hybrid (dense+BM25) retrieval, reranking
- **Week 3** — RAGAS evaluation harness + MLflow logging, real measured numbers
- **Week 4** — LangGraph agent (query decomposition, self-correction, calculator tool)
- **Week 5** — CI (GitHub Actions), Dockerfile, deployment docs (this repo)

Week 6 (final polish/write-up) is still ahead.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in GOOGLE_API_KEY, MONGODB_URI, SEC_EDGAR_USER_AGENT
```

You need:
- A [Google AI Studio API key](https://aistudio.google.com/apikey) (Gemini
  free tier — no card required). Note: the free tier caps at 500
  requests/day per model — see `TECHNICAL_REPORT.md` for how this project
  hit that limit twice during development and designed around it.
- A [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) free-tier
  cluster connection string

## Pipeline

```bash
# 1. Fetch, parse, and store FinanceBench-referenced filings in MongoDB
uv run scripts/ingest.py --limit 5

# 2. Chunk + embed + build the retrieval indexes (both variants)
uv run scripts/build_index.py --variant naive
uv run scripts/build_index.py --variant table_aware

# 3. Ask a question against the pipeline
uv run scripts/ask.py "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"

# 4. Ask a multi-step question via the LangGraph agent
uv run scripts/agent_ask.py "Compare 3M and Amazon's revenue growth over the last two fiscal years"

# 5. Run the RAGAS evaluation harness (naive vs. hybrid+rerank vs. hybrid-no-rerank)
uv run scripts/run_eval.py --n-questions 20
```

## Tests

```bash
uv run pytest
```

Runs automatically on every push/PR via GitHub Actions
(`.github/workflows/test.yml`) — no API keys needed, all unit tests.

## CI/CD

- **`test.yml`** — the full unit test suite, on every push/PR, free (no
  API calls).
- **`eval.yml`** — the real RAGAS eval, manually triggered
  (`workflow_dispatch`) rather than on every push, since Gemini's
  free-tier daily quota can't sustain a real eval run on every commit
  (see `TECHNICAL_REPORT.md` for the incident that drove this design).
  Needs `GOOGLE_API_KEY` and `MONGODB_URI` added as repo secrets under
  Settings → Secrets and variables → Actions to run.

## Docker

```bash
docker build -f deploy/docker/Dockerfile -t findocqa .
docker run --env-file .env findocqa uv run scripts/ask.py "..."
```

Containerizes the CLI pipeline above. Not build-tested locally (Docker
isn't installed in this project's dev environment) — see
`deploy/aws/DEPLOYMENT.md` for the honest status and how this would map
to an actual AWS deployment (not done — real ongoing cost, not worth it
for a portfolio project without a FastAPI/Streamlit app to actually serve
yet).

## Project layout

- `src/findocqa/ingestion/` — EDGAR/FinanceBench fetching, HTML parsing
- `src/findocqa/storage/` — MongoDB (raw parsed filings + metadata)
- `src/findocqa/retrieval/` — chunking, embeddings, FAISS + BM25, hybrid fusion, reranking
- `src/findocqa/generation/` — prompt + Gemini call for QA, shared rate limiter
- `src/findocqa/agent/` — LangGraph agent (decomposition, tools, self-correction)
- `src/findocqa/eval/` — RAGAS harness, custom numerical-accuracy metric, eval subset selection
- `src/findocqa/api/`, `src/findocqa/ui/` — FastAPI + Streamlit (not built yet)
- `deploy/docker/` — Dockerfile
- `deploy/aws/` — deployment documentation
- `.github/workflows/` — CI (tests) + manually-triggered eval workflow
