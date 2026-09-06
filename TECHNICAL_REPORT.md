# FinDocQA — Full Technical Report

Companion to `PROGRESS.md` (the short recap). This is the detailed version:
real code, real bugs with before/after data, real command output. ~1,340
lines of Python across 4 commits so far.

**What this is NOT**: this covers Weeks 1-3 only (data + retrieval + eval).
Weeks 4-5 (agent, CI/CD, Docker/AWS deployment) are **not built yet** — see
"What's not done yet" at the bottom. Nothing below claims otherwise.

---

## 1. Project layout (actual, as of now)

```
FinDocQA/
├── findocqa-project-spec.md      # the original 6-week plan
├── PROGRESS.md                   # short recap
├── pyproject.toml                # deps, managed with uv
├── .env                          # secrets (gitignored) — Gemini key, Mongo URI
├── scripts/
│   ├── ingest.py                 # fetch+parse+store filings from EDGAR
│   ├── build_index.py            # chunk+embed+index a corpus variant
│   ├── ask.py                    # CLI: ask one question, pick pipeline config
│   ├── compare_retrieval.py      # Week 2 manual spot-check (naive vs hybrid)
│   └── run_eval.py               # Week 3 orchestrator: RAGAS + MLflow
├── src/findocqa/
│   ├── config.py                 # pydantic-settings, reads .env
│   ├── ingestion/
│   │   ├── edgar_client.py       # SEC EDGAR HTTP client, CIK lookup
│   │   ├── financebench_docs.py  # pulls the 150-question public benchmark
│   │   ├── parser.py             # filing HTML -> ordered text/table blocks
│   │   └── pipeline.py           # shared single-filing ingest logic
│   ├── storage/mongo.py          # MongoDB client + filings collection
│   ├── retrieval/
│   │   ├── chunking.py           # naive (Wk1) + table-aware (Wk2) chunkers
│   │   ├── embeddings.py         # local sentence-transformers wrapper
│   │   ├── vector_store.py       # FAISS index build/search, per variant
│   │   ├── bm25_index.py         # keyword search index
│   │   ├── hybrid.py             # dense+BM25 fusion (RRF) + rerank
│   │   ├── reranker.py           # cross-encoder wrapper
│   │   └── build.py              # shared index-build logic
│   ├── generation/answer.py      # retrieve -> prompt -> Gemini call
│   ├── eval/
│   │   ├── dataset.py            # stratified eval-question sampler
│   │   ├── numerical_accuracy.py # custom "did it get the number right" check
│   │   └── ragas_harness.py      # RAGAS wired to our Gemini client
│   ├── agent/     (placeholder — Week 4, not built)
│   ├── api/       (placeholder — Week 5, not built)
│   └── ui/        (placeholder — Week 5, not built)
└── tests/unit/    (17 tests, all passing — see §5)
```

---

## 2. Week 1 — the base pipeline

### 2.1 How a filing gets in

FinanceBench (the benchmark) publishes its own list of which company
filings its 150 questions depend on, but the links it gives point at
IR-vendor-hosted PDFs that turned out to be **hotlink-protected** (return
HTTP 403/blocked) and whose embedded "accession numbers" aren't real SEC
IDs. So `ingestion/edgar_client.py` resolves the real filing straight from
SEC EDGAR instead:

```python
def find_filing(cik: str, form_type: str, fiscal_year: int) -> dict | None:
    """Finds the filing of `form_type` (e.g. "10-K") whose reported fiscal
    period falls in `fiscal_year`."""
    for page in _iter_submission_pages(cik):
        for i, form in enumerate(page["form"]):
            if form != form_type:
                continue
            report_date = page["reportDate"][i]
            if not report_date.startswith(str(fiscal_year)):
                continue
            accession_number = page["accessionNumber"][i]
            primary_document = page["primaryDocument"][i]
            ...
            return {"accession_number": accession_number, "url": ..., ...}
    return None
```

### 2.2 Parsing

`ingestion/parser.py` walks the filing's HTML and splits it into an
ordered list of typed blocks — `{"type": "text", "content": ...}` or
`{"type": "table", "rows": [[...], ...]}` — using placeholder substitution
so tables and surrounding narrative stay in true document order:

```python
for i, table in enumerate(soup.find_all("table")):
    rows = _extract_table_rows(table)
    if rows:
        tables_by_id[str(i)] = rows
        table.replace_with(f"\x00TABLE_{i}\x00")
    else:
        table.decompose()

full_text = soup.get_text(separator="\n")
# ... split on the placeholder tokens to rebuild ordered text/table blocks
```

### 2.3 First real test (this actually ran)

```
$ uv run scripts/ask.py "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"

Retrieved context:
  [0.814] 3M_2020_10K chunk 252
  [0.802] 3M_2018_10K chunk 113
  [0.801] 3M_2019_10K chunk 241
  [0.795] 3M_2015_10K chunk 118
  [0.792] 3M_2017_10K chunk 125

Answer: Based on the provided context, there is no mention of the FY2018
capital expenditure amount for 3M in the cash flow statement details. The
context mentions capital spending percentages for 2018 [...] but does not
provide the specific 2018 capital expenditure dollar amount.
```

**Expected answer: `$1577.00` million.** The model failed — but I dug into
*why* rather than just noting the failure. The correct figure ($1,577M)
**was** in the retrieved context (chunk 113, rank #2), sitting in a row
like:

```
Total Company | 93,516 | 91,536 | 91,584 | $ 1,577 | $ 1,373 | $ 1,420 | $ 8,738 | $ 8,866 |
```

— a wall of unlabeled numbers, because the naive chunker sliced the
document at a fixed 1,500-character window with no regard for table
boundaries, so the table's header row landed in a *different* chunk than
its data. The model correctly declined to guess rather than hallucinate a
number it couldn't attribute. This became the concrete motivation for
Week 2.

---

## 3. Week 2 — table-aware chunking, hybrid retrieval, reranking

### 3.1 The chunking fix

`retrieval/chunking.py` now has two chunkers side by side (the naive one
is kept, not deleted, so Week 3 can measure baseline-vs-improved rather
than just overwrite the old numbers):

```python
def chunk_filing_table_aware(filing: dict) -> list[Chunk]:
    """Tables are never split across a chunk boundary (or, if too large
    for one chunk, split with the header row repeated); text blocks keep
    Week 1's sliding window but scoped to a single block so it can't
    bleed into an adjacent table."""
```

Tables are serialized as Markdown instead of flat `" | "`-joined cells,
with the preceding paragraph's last ~200 characters prepended for context.

### 3.2 Bug #1 found via testing: silent data loss

While spot-checking against real questions, an Amazon revenue question
failed even though nothing looked obviously wrong. Investigation:

```
$ python -c "... check raw parsed blocks for Amazon 2017 10-K ..."
block 113 rows: 20
['', '', 'Year Ended December 31,']
['', '', '2013', '', '2014', '', '2015', '', '2016', '', '2017 (1)']
['Net sales', '', '$', '74,452', '', '', '$', '88,988', '', '', '$', '107,006', '', '', '$', '135,987', '', '', '$', '177,866', '']
```

The data was there in the raw parse. But the *chunker's* Markdown
formatter computed column width from the **first row only** — and this
table's first row is a 3-cell caption (`"Year Ended December 31,"`), while
the actual data rows have 11+ cells (currency symbols and blank spacer
cells each get their own column in SEC HTML). Every data row was getting
silently truncated to 3 cells, destroying the dollar figures entirely.

**Fix**: size the table by its widest row, not its first:

```python
# before (bug): width = len(header)   <- truncates wider rows
# after (fix):
width = max(len(r) for r in rows)
```

A second identical bug was found the same way in 3M's debt-securities
table (a 1-cell caption row truncated 5-cell data rows, destroying the
`MMM26`/`MMM30`/`MMM31` trading symbols). Same fix resolved both.

### 3.3 Hybrid retrieval

`retrieval/hybrid.py` combines dense (FAISS) + sparse (BM25) results via
Reciprocal Rank Fusion, then reranks with a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`):

```python
def hybrid_search(query, top_k=5, variant="table_aware", use_reranker=True):
    dense = vector_store.search(query, top_k=30, variant=variant)
    sparse = bm25_index.search_bm25(query, top_k=30, variant=variant)
    fused = _reciprocal_rank_fusion(dense, sparse)
    if not use_reranker:
        return fused[:top_k]
    return rerank(query, fused, top_k=top_k)   # cross-encoder scores ALL of fused
```

### 3.4 Bug #2 found via testing: fusion cutting off the right answer

Original code truncated to the top-30 *fused* candidates **before**
reranking. RRF mathematically penalizes a candidate only one retriever
found — a strong dense-only match (numeric tables have little literal
keyword overlap for BM25) could get pushed past rank 30 and never reach
the (much more accurate) reranker at all, even though it was sitting right
there in the dense results:

```
target chunk in dense top30:  [('AMAZON_2017_10K', 211)]   <- found, strong match
target chunk in bm25 top30:   []                            <- BM25 missed it entirely
target chunk fused rank:      44  of 60                      <- RRF pushed it past the top-30 cutoff
```

**Fix**: rerank the full fused pool, only truncate the *reranker's* output:

```python
fused = _reciprocal_rank_fusion(dense, sparse)   # no truncation here anymore
return rerank(query, fused, top_k=top_k)          # cross-encoder judges everything
```

After the fix, the same Amazon question succeeded end-to-end:

```
Q: What is Amazon's year-over-year change in revenue from FY2016 to FY2017?
Expected: 30.8%

[table_aware hybrid+rerank]
Total net sales for FY2016: $135,987 million
Total net sales for FY2017: $177,866 million
(177,866 - 135,987) / 135,987 ≈ 30.8%
Source: [AMAZON_2017_10K | chunk 239]
```

### 3.5 The honest open finding

Across an 11-question manual spot-check, naive dense-only retrieval still
edged out hybrid+rerank overall (2 clean wins vs. 1). Root cause: the
generic cross-encoder reranker — trained on web search Q&A pairs, not
finance — sometimes scored a chunk full of narrative prose *higher* than
the chunk containing the actual correct number. This was flagged as an
open question for Week 3 to answer with real numbers rather than
eyeballing, instead of quietly "fixing" it by hand-tuning on 11 examples.

---

## 4. Week 3 — RAGAS evaluation (currently running)

### 4.1 What changed structurally

`generation/answer.py`'s retrieval selection became a 3-way mode instead
of a 2-way boolean, specifically to let Week 3 test the reranker question
directly:

```python
_MODES = ("dense", "hybrid", "hybrid_no_rerank")
```

Three pipeline configs get compared:

| Config | Chunking | Retrieval |
|---|---|---|
| `naive_dense` | Week 1 naive | dense only |
| `table_aware_hybrid_rerank` | Week 2 table-aware | hybrid + reranker |
| `table_aware_hybrid_no_rerank` | Week 2 table-aware | hybrid, no reranker |

### 4.2 Metrics

- **Faithfulness, Context Precision, Context Recall** — via
  [RAGAS](https://github.com/explodinggradients/ragas), judged by our own
  Gemini model (not a separate paid judge).
- **Numerical accuracy** (custom, no LLM call) — does the actual expected
  number appear in the answer, within 1% tolerance? RAGAS has no built-in
  for this.

### 4.3 Bug #3 found via testing: false-positive number matching

Smoke-testing on 3 real questions surfaced this:

```json
{
  "ground_truth": "Yes, there was a decline of ~42% between FY2023 and Q2 of FY2024.",
  "answer": "...cash and cash equivalents for October 28, 2023 and November 2, 2024...",
  "numerical_accuracy": true   // <- WRONG. The 42% was never checked.
}
```

The regex-based number extractor was swallowing trailing sentence
punctuation (`"FY2024,"` / `"FY2024."`) into its match, which made a bare
calendar year look like it had a thousands/decimal marker and skip the
year-collision filter. Two unrelated numbers "matched" only because both
texts happened to mention the year 2024. **Fixed** by stripping trailing
`.`/`,` that isn't followed by more digits before classifying a number as
a real financial figure vs. a bare year:

```python
def _is_probable_year(raw: str, value: float) -> bool:
    has_marker = any(c in raw for c in "$%.,")
    return not has_marker and value == int(value) and int(value) in _YEAR_RANGE
```

Covered by 4 regression tests in `tests/unit/test_numerical_accuracy.py`.

### 4.4 RAGAS + Gemini wiring (verified before building on it)

RAGAS's own docs describe a newer `ragas.metrics.collections` API — but
testing directly against the installed version (0.4.3) showed
`evaluate()` doesn't accept those yet (`TypeError: All metrics must be
initialised metric objects`). Fell back to the classic (deprecated but
functional) `ragas.metrics` classes instead of trusting the docs blindly.
Also discovered RAGAS's "google" provider works directly with our
existing `google-genai` client (no separate legacy SDK needed, despite
what online examples showed) via `instructor.from_genai()`.

### 4.5 Scope decision

The full 150-question benchmark × 3 configs would need 2,000+ API calls —
more than Gemini's free-tier daily cap (1,000/model/day) and, at measured
RAGAS latency (~91 seconds per question for 3 metrics), several hours
longer than planned. Evaluating a **stratified 40-question subset across
31 companies** instead — picked with a fixed random seed so it's
reproducible from code alone (`eval/dataset.py`), not cherry-picked.

### 4.6 Live status (as of this report)

```
Selected 40 questions across 31 companies.
Ingesting 32 missing filing(s)...
  [1/32] PEPSICO_2021_10K            ok
  [2/32] MGMRESORTS_2023Q2_10Q       ok
  [3/32] AES_2022_10K                FAILED: could not resolve CIK for 'AES Corporation'
  ...
  [22/32] PFIZER_2021_10K            ok
  ...
  [32/32] ACTIVISIONBLIZZARD_2019_10K FAILED: could not resolve CIK for 'Activision Blizzard'
Building naive index over full corpus...
```

29 of 32 missing filings ingested successfully (3 failed on
company-name-to-CIK lookup mismatches — a known limitation of the
substring-matching lookup, non-fatal, script continues). Now building
indexes, then will run generation + RAGAS scoring across all 3 configs.
Expected total runtime ~3-4 hours from start; final scores will be
reported when it completes, plus logged in MLflow (`mlruns/`) for
side-by-side comparison.

---

## 5. Tests (all passing, run just now)

```
$ uv run pytest -q
.................                                                        [100%]
17 passed in 41.75s
```

Covers: naive chunker (splitting, table flattening), table-aware chunker
(no data loss on small tables, header repeated on large split tables),
RRF fusion (union not truncation, both-lists-ranked-higher), numerical
accuracy (currency/percent/negative extraction, tolerance matching, the
year-collision regression).

---

## 6. What's not done yet

Per the original 6-week spec, still ahead:

- **Week 4 — Agentic layer**: LangGraph agent for multi-step questions
  (query decomposition, calculator tool, EDGAR live-fetch tool,
  self-correction loop). `src/findocqa/agent/` currently has only a
  placeholder `__init__.py`.
- **Week 5 — MLOps + deployment**: Docker, AWS (S3/ECS/CloudWatch),
  **GitHub Actions CI/CD** (auto-run eval on every change — this is the
  project's stated key differentiator vs. similar public repos, and it
  does not exist yet), FastAPI backend, Streamlit UI. `.github/workflows/`
  is currently an empty placeholder directory.
- **Week 6 — polish**: architecture diagram, final written eval report,
  error-analysis writeup.

Nothing above should be read as "CI/CD is set up" or "it's deployed" —
those are explicitly future work.

---

## 7. Post-deployment fix: retrieval-disambiguation via metadata filtering

After Weeks 1–5 shipped (agent, CI/CD, Streamlit demo), a previously-passing
question regressed as the corpus grew to 44 filings: "What is the FY2018
capital expenditure amount (in USD millions) for 3M?" started coming back
"not enough information."

### 7.1 Diagnosis

Traced directly: the correct chunk (`3M_2018_10K`, chunk 385 — the
Consolidated Statement of Cash Flows table containing
`Purchases of property, plant and equipment (PP&E) | (1,577) | ...`)
scored 0.714 on dense cosine similarity, but never made the candidate pool
because **same-topic capex chunks from 3M's *other* fiscal years scored
0.77–0.79** — a same-company, wrong-year chunk beat the right one purely
on topical similarity. BM25 was worse: with 44 documents across dozens of
companies, generic financial vocabulary ("capital", "expenditure",
"million") let chunks from CVS Health, Corning, and MGM Resorts outrank
the correct 3M chunk entirely.

Root cause: **nothing in retrieval ever used the `company` field already
present on every chunk, or the fiscal year named in the question.**
Dense/BM25 similarity was being asked to disambiguate company *and* year
*and* topical relevance all at once, with no structural help — a task
that gets strictly harder as the corpus grows, which is exactly why this
regressed only after ingesting more filings.

### 7.2 Fix

Added `src/findocqa/retrieval/company_filter.py`:
`detect_company(query)` matches the query against the distinct company
names actually in MongoDB (longest name first, so "American Express"
matches before a shorter overlapping name would) — but **only returns a
company if exactly one is named**. A comparison question ("Compare 3M
and Amazon's revenue growth...") naming two companies must not be
narrowed to just one of them; that would silently drop the other
company's chunks entirely, which is worse than not filtering at all.
This was caught by testing the fix against the project's own example
comparison question, not assumed to be safe.

`detect_fiscal_year(query)` extracts a year from `FY2018` / `fiscal year
2018` / a bare `2018`. `doc_fiscal_year(doc_name)` extracts the embedded
year from a doc name like `3M_2018_10K` — done by regex against the
existing doc name rather than adding a `fiscal_year` field to chunk
metadata, to avoid a full corpus reindex for this fix.

Both `vector_store.search()` and `bm25_index.search_bm25()` now accept
optional `company`/`fiscal_year` filters, applied before truncating to
`top_k` (both FAISS and BM25 already score the whole corpus internally
regardless of `top_k`, so filtering costs nothing extra). `hybrid_search()`
and `generation/answer.py`'s `_retrieve()` — the shared chokepoint used
by both the simple pipeline and the agent — thread the detected
company/year through automatically.

### 7.3 Verification

```
>>> detect_company("What was 3M revenue in FY2019?")
'3M'
>>> _retrieve(...) # -> exclusively {'3M_2019_10K'}, was previously mixed

>>> detect_company("What was Amazon net sales in FY2017?")
'Amazon'
>>> _retrieve(...) # -> exclusively {'AMAZON_2017_10K'}

>>> detect_company("Compare 3M and Amazon's revenue growth")
None   # correctly backs off — does not drop either company
```

Test suite grew from 33 to 41 passing tests (`tests/unit/test_company_filter.py`),
covering: longest-match preference, no-match, the multi-company back-off
case, both year-phrasing patterns, and doc-name year extraction.

### 7.4 Honest open finding: this does not fully fix the original 3M capex case

Re-running the diagnosis with company+year filtering active narrows the
search from 44 documents to the single correct filing — but the target
chunk still doesn't reach the top of the reranked results (it lands
around rank 52 of ~300 candidates even with the pool widened to 200).
The reason is a *second*, different problem layered on top of the first:

- Chunk 385 is one large chunk holding the **entire** cash-flow
  statement (~20 line items). The specific line the question needs is
  diluted inside a lot of unrelated numeric text.
- The question says "capital expenditure"; the filing says "Purchases
  of property, plant and equipment (PP&E)" — a real vocabulary gap that
  neither the bi-encoder nor the MS-MARCO-trained cross-encoder reranker
  reliably bridges.

Tried and **rejected**: literally expanding the query with the synonym
("...capital expenditure (purchases of property, plant and equipment,
PP&E)..."). Verified empirically this backfires — it pulled in *other*
chunks that mention "PP&E" more literally but are actually irrelevant
(quarterly-data notes mentioning net PP&E book value), ranking them
above the real answer. Not shipped, because it measurably made this
case worse rather than better.

This is now a known, documented limitation rather than a silently
accepted regression: fixing it properly needs finer-grained table
chunking (e.g., one retrievable unit per line item instead of per
table) so a single relevant figure isn't buried inside a large chunk.
Recorded here rather than papered over, per this project's own standard
of only claiming a fix once it's been checked against the case that
motivated it.

### 7.5 Follow-up: finer table chunking helps, but doesn't fully close the gap

Tried the fix section 7.4 named: lowered `MAX_TABLE_CHUNK_CHARS` from
2000 to 500 in `chunking.py`, so a ~20-line-item cash flow statement now
splits into several small, header-repeating chunks instead of one large
one. Rebuilt the `table_aware` index (43,754 chunks, up from a much
smaller count with the old threshold) and re-ran the same diagnosis.

**Measured effect**: the chunk containing the exact PP&E/capex line
moved from dense-similarity rank ~584th (of ~600 candidates, old large
chunk) to rank ~106th (of ~1,400 candidates, new small chunk) within
the correctly-filtered single filing — a real, large improvement from
isolating the line from the other ~19 unrelated ones. It's now small
enough to plausibly reach a wider candidate pool.

**Still not fully fixed**: even after widening the pool to 250
candidates per retriever (so the correct chunk is actually present in
the fused pool this time), the cross-encoder reranker still doesn't
rank it in the top 5 — it separately, directly scored the correct
chunk *lower* than several less-relevant ones. Manually confirmed via
`reranker.rerank()` directly: `cross-encoder/ms-marco-MiniLM-L-6-v2` was
trained on natural-language web passages, and a markdown table row like
`| Purchases of property, plant and equipment (PP&E) | (1,577) |` is a
different input distribution than what it saw in training — it doesn't
reliably score this as more relevant to "capital expenditure" than
ordinary prose that happens to mention "capital" near the start.

**Kept anyway**: the finer chunking shipped despite not fully closing
this one case, because it's independently verified as a net
improvement (isolates line items generally, real rank improvement
measured above) and causes no regression on any other verified case
(company/year disambiguation, comparison back-off — all re-verified
after this change; full test suite still green). The remaining gap is
now specifically attributable to the reranker/embedding model itself
lacking finance-domain tuning, not to chunking or filtering -- a
different, larger piece of future work (a fine-tuned or finance-specific
reranker) than anything reasonable to bolt on here.

### 7.6 Deployment fix: the finer chunking broke cold-start on Streamlit Cloud

Shrinking `MAX_TABLE_CHUNK_CHARS` (7.5) grew the corpus to ~43,754
chunks. `data/processed/` (the built FAISS/BM25 index files) is
gitignored, so **every fresh Streamlit Cloud container has MongoDB's
filing data but no local index** — it has to build one. Measured local
embedding throughput (~101 texts/sec) put a from-scratch rebuild at
7+ minutes, likely worse on Streamlit Cloud's free-tier CPU. That's not
survivable as a cold-start UX for a demo link a recruiter clicks once.

**Fix**: `retrieval/index_cache.py` persists built index files
(`index.faiss`, `chunks.jsonl`, `bm25.pkl`) in MongoDB GridFS —
already-provisioned, already-free infrastructure, no new service.
`build_variant_index()` now uploads after every build; the Streamlit
UI's `_ensure_index()` tries a GridFS download before falling back to a
full rebuild (which would then upload for the next container). Verified
by simulating a fresh container (moved local index files aside,
confirmed `index_exists()` was `False`) and timing the recovery path:
**download restored both variants in ~19 seconds total**, vs. 7+
minutes to rebuild, and retrieval was re-verified correct against the
downloaded index (not just "files exist").

**Real constraint surfaced by this**: MongoDB Atlas's free tier (M0)
caps storage at 512MB. The two variants' index files together use
~297MB of data size (~58% of the cap) — workable today, but a real
ceiling on how much further the corpus can grow before this specific
caching approach needs revisiting (e.g., dropping the rarely-used
`naive` baseline variant from the cache, or an external object store).

### 7.7 Live verification found a real bug: reformulation silently dropped company names

Everything above was verified by calling internal functions directly
(`_retrieve`, `hybrid_search`, etc.) rather than through a real Gemini
call, specifically to conserve the shared quota. The agent's
comparison-question path had never actually been exercised end-to-end
after the company/fiscal-year filtering work, so it was run for real
once: `"Compare 3M and Amazon's revenue growth over the last two fiscal
years"`.

Decomposition worked as designed — it split into four sub-questions,
one per company per year ("What is 3M's revenue growth for the last
fiscal year?", etc.). But two of the four sub-answers came back visibly
degraded, one literally saying *"the question does not specify which
company it refers to"* for a sub-question that began with "What is
Amazon's revenue growth...".

Root cause: `answer_subquestion`'s retry loop calls `_reformulate()`
when an answer looks insufficient, and its system prompt only said
"use different keywords and phrasing" with no instruction to preserve
the company name. Confirmed directly: reformulating `"What is Amazon's
revenue growth for the fiscal year prior to the last?"` three times
produced rephrasings that dropped "Amazon" from the text entirely on
some runs — silently defeating `detect_company()`'s substring match for
that retry, so the retried retrieval searched the *whole* corpus again
instead of just Amazon's filing, exactly the disambiguation failure
section 7 exists to fix, reintroduced through a code path filtering by
company/year never accounted for.

**Fix**: `_reformulate`'s system prompt now explicitly requires keeping
any company name and fiscal year/date exactly as written, varying only
financial terminology. Re-verified directly against the exact failing
input across 3 calls: the company name survived every rephrasing after
the fix, where it hadn't reliably before. Not re-run through the full
multi-call agent graph again to avoid spending further shared quota on
what direct verification already confirmed at the root cause.
