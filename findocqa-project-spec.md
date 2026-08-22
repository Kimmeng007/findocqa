# FinDocQA — Agentic RAG System for SEC Financial Filings

## Goal

Build a portfolio-grade AI engineering project demonstrating production-level
RAG + agentic AI + MLOps skills, for CV/job/PhD applications. The project
should be defensible in technical interviews — every design choice needs a
reason, and results need to be measured, not just demoed.

**Target narrative:** "Built and evaluated a production-style agentic RAG
system for financial filings, with automated evaluation, experiment tracking,
and CI/CD-gated deployment."

**Domain:** SEC 10-K / 10-Q filings (via SEC EDGAR API — free, public, no
licensing issues). Financial filings mix long narrative text with tables
(balance sheets, income statements), which forces real engineering decisions
around chunking and retrieval instead of naive text splitting.

**Benchmark:** Evaluate against the public FinanceBench evaluation subset
(150 questions) so results are comparable to published numbers, not just
"works on my documents." Reported baselines to beat/reference:
- Shared vector-store RAG baseline (FinanceBench paper): ~19% accuracy
- Long-context baseline: ~79% accuracy
- Best known open comparable (FinanceBench_RAG repo): ~76% accuracy

## Non-goals / scope boundaries

- Not a general-purpose chatbot — stay scoped to financial filing QA.
- Not a data engineering showcase — no PySpark/Hadoop/Power BI here (separate
  project for that; data volume here doesn't justify distributed processing).
- Don't oversell as "production" — frame as "production-grade practices" in
  any writeup. Honest, non-overclaiming framing throughout.

## Tech stack

- **Ingestion:** SEC EDGAR API, table-aware parsing (don't flatten tables into
  plain text — handle them as structured data)
- **Retrieval:** Hybrid (dense embeddings + BM25) + reranking (cross-encoder
  or similar)
- **Agentic layer:** LangGraph — query decomposition for multi-step questions,
  tool use (calculator for ratios/deltas, EDGAR fetch tool), self-correction
  loop (reformulate/retry when retrieved context coverage is low)
- **Evaluation:** RAGAS (faithfulness, context precision/recall, answer
  relevancy) for RAG-level; agent-level eval for tool-call correctness and
  task completion (Ragas also supports AgentGoalAccuracy / ToolCallAccuracy)
- **Experiment tracking:** MLflow (log configs, prompts, retrieval settings,
  eval scores per run)
- **Storage:** MongoDB for raw parsed filings + metadata (company, fiscal
  year, filing type) ahead of chunking/indexing; vector DB (FAISS or Qdrant)
  for embeddings
- **Deployment:** Docker (docker-compose: vector DB + API + agent
  orchestrator + UI), deploy to AWS (S3 for raw filings, ECS/Fargate or EC2
  for containers, CloudWatch for logging) instead of local-only
- **CI/CD:** GitHub Actions — auto-run the eval suite (RAG + agent) on every
  prompt/retriever/agent change; this is the key differentiator vs. most
  similar open-source repos, which demo but don't gate on eval
- **Observability:** Langfuse or Arize Phoenix — trace agent steps, tool
  calls, latency, cost
- **API/UI:** FastAPI backend, Streamlit demo frontend

## Step-by-step plan (~6 weeks part-time, ~3 weeks full-time)

### Week 1 — Data + baseline RAG
- Pull 50–100 10-K filings via SEC EDGAR API
- Parse text + tables (expect this to take longer than the code itself)
- Store raw parsed filings + metadata in MongoDB
- Baseline pipeline: naive chunking → FAISS → basic LLM answer generation

### Week 2 — Retrieval quality
- Table-aware chunking (separate handling for tables vs. narrative text)
- Hybrid retrieval (dense + BM25) + reranking
- Manual spot-check vs. baseline on ~10 questions before automating eval

### Week 3 — Evaluation (RAG-level)
- Pull FinanceBench public eval subset, build eval harness (RAGAS:
  faithfulness, context precision/recall, numerical accuracy)
- Run baseline vs. hybrid+rerank → get concrete "improved X→Y" numbers
- Log configs/scores in MLflow

### Week 4 — Agentic layer
- Build agent with LangGraph: query decomposition for multi-step questions
  (e.g. "compare operating margins of Company A and B over 3 years")
- Add tools: calculator for ratios/deltas, EDGAR live-fetch tool
- Self-correction loop: agent reformulates/retries on low-coverage retrieval
- Agent-level eval: tool-call correctness, task completion, failure handling

### Week 5 — MLOps + deployment
- Dockerize full stack (vector DB + MongoDB + API + agent orchestrator + UI)
- Deploy to AWS (S3, ECS/Fargate or EC2, CloudWatch)
- GitHub Actions: auto-run eval suite on prompt/retriever/agent changes
- FastAPI backend + Streamlit demo showing multi-step agentic queries live
- Observability (Langfuse or Phoenix): trace agent steps + tool calls

### Week 6 (buffer) — Polish + narrative
- README with architecture diagram (RAG pipeline + agent graph), eval
  numbers vs. published FinanceBench baselines, tool-use examples
- Write the "why" for each design choice (why hybrid retrieval, why
  LangGraph, why this eval framework) — this is what gets asked about in
  interviews
- Error analysis section: where retrieval fails, where agent reasoning fails
  — this level of honesty is what separates strong repos from demo-only ones

## Known risk points (where time usually slips)

- Table parsing in 10-Ks (week 1) — messy, don't underestimate
- Agent-level eval design (week 4) — requires judgment calls, not just code
- Docker/CI/AWS deployment (week 5) is comparatively fast to implement

## Prior art (don't blindly duplicate — know what exists)

Several similar public projects already combine RAG + agentic + FinanceBench
+ eval/observability (e.g. LangGraph-based financial agentic RAG repos with
reported FinanceBench scores in the 70%+ range, some with RBAC/HITL and
self-hosted observability). This is a validated, recognizable project
pattern — not a reason to switch topics — but differentiation must come from:
1. A genuine measured eval score with real error analysis (not just a demo)
2. CI/CD-gated evaluation (most existing repos don't do this)
3. Clear, defensible reasoning for every architectural choice
