# FinDocQA — Project Status Recap

## What this project is
An AI engineering portfolio project — a RAG (retrieval-augmented generation)
system that answers questions about SEC financial filings (10-Ks, 10-Qs),
evaluated against a public benchmark called **FinanceBench**. The goal is a
real, *measured* project to show in job/PhD interviews, not just a demo.

## Key decisions along the way
- **LLM**: Started with Anthropic Claude, switched to **Google Gemini**
  (free tier, no cost) after cost discussion — specifically
  `gemini-3.5-flash-lite` (updated once mid-project after Google deprecated
  the earlier model name).
- **Database**: MongoDB Atlas (free tier) stores the parsed filings. Hit a
  real snag — Atlas didn't have the current IP allowlisted — just fixed.
- **Embeddings**: Free, local model (`sentence-transformers`) — no API cost.
- **Data source**: SEC EDGAR (the government's public filing database),
  targeted at the exact companies/filings FinanceBench's 150 test questions
  reference, so the corpus lines up with the benchmark.

## Week 1 — Basic pipeline (done)
Built the foundation: fetch a filing from SEC EDGAR → parse it into
text/tables → store in MongoDB → chunk it → search it → ask Gemini to
answer a question using the retrieved chunks. Verified end-to-end on a
real question (3M's 2018 capital expenditures).

## Week 2 — Better retrieval (done)
Week 1's chunking was naive — fixed-size pieces with no regard for table
boundaries, which mangled real answers (a table's header ended up in a
different chunk than its data, so the number was retrieved but the model
couldn't tell what it meant). Built:
- **Table-aware chunking** — tables kept whole, or split with headers repeated
- **Hybrid search** — keyword search (BM25) + semantic search (embeddings)
- **Reranking** — a second-pass model re-scoring results for relevance

Testing on 11 real questions found and fixed **two real bugs** (a
data-truncation bug and a search-ranking bug) — and surfaced one
inconclusive finding: the reranker didn't clearly help on this small
sample, and sometimes hurt. Flagged as unresolved rather than papered over.

## Week 3 — Real measurement (in progress)
Rather than guessing whether Week 2's changes helped, this runs a proper
evaluation: **40 real FinanceBench questions across 31 companies**, tested
against **3 configurations**:
1. Week 1's naive baseline
2. Week 2's hybrid + reranker
3. Week 2's hybrid *without* reranker (to settle the open question)

Each answer is scored on 4 metrics: faithfulness (hallucination check),
context precision/recall (did retrieval find the right material?), and a
custom check for whether the correct number actually appears in the
answer. Results are logged to MLflow for comparison.

**This is the long background job currently running** (~3-4 hours total)
— it just cleared a real connectivity issue (MongoDB IP allowlist) and is
now actually processing. Results reported once it completes.

**Everything is committed to git** after each week, with the reasoning
behind decisions in each commit message.
