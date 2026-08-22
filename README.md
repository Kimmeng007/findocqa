# FinDocQA

Agentic RAG system for SEC financial filings (10-K/10-Q), evaluated against
the public [FinanceBench](https://github.com/patronus-ai/financebench)
150-question benchmark. See [findocqa-project-spec.md](findocqa-project-spec.md)
for the full project goals, scope, and week-by-week plan.

**Status:** Week 1 — data ingestion + naive baseline RAG.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, MONGODB_URI, SEC_EDGAR_USER_AGENT
```

You need:
- An [Anthropic API key](https://console.anthropic.com/)
- A [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) free-tier
  cluster connection string

## Week 1 pipeline

```bash
# 1. Fetch, parse, and store FinanceBench-referenced filings in MongoDB
uv run scripts/ingest.py --limit 5

# 2. Chunk + embed + build the baseline FAISS index
uv run scripts/build_index.py

# 3. Ask a question against the baseline pipeline
uv run scripts/ask.py "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
```

## Tests

```bash
uv run pytest
```

## Project layout

- `src/findocqa/ingestion/` — EDGAR/FinanceBench fetching, PDF parsing
- `src/findocqa/storage/` — MongoDB (raw parsed filings + metadata)
- `src/findocqa/retrieval/` — chunking, embeddings, FAISS vector store
- `src/findocqa/generation/` — prompt + Claude call for QA
- `src/findocqa/agent/` — LangGraph agent (Week 4)
- `src/findocqa/eval/` — RAGAS + agent eval harness (Week 3)
- `src/findocqa/api/`, `src/findocqa/ui/` — FastAPI + Streamlit (Week 5)
- `deploy/` — Docker + AWS deployment (Week 5)
