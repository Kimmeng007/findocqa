"""About page: architecture, measured eval results, known limitations, and
contact -- split out from app.py so the main page stays focused on the
interactive Q&A demo instead of a long scroll of write-up.
"""

import streamlit as st

st.set_page_config(page_title="About — FinDocQA", page_icon="📊")

st.title("About FinDocQA")
st.markdown(
    "An AI engineering portfolio project: a retrieval-augmented QA system "
    "over SEC financial filings (10-K/10-Q), evaluated against the public "
    "**FinanceBench** benchmark. Built to show real, measured engineering "
    "work -- bugs found and fixed, results measured, limitations disclosed "
    "-- not just a working demo."
)

st.header("⚙️ How it works")
st.markdown(
    "1. **Ingest**: filings are pulled from SEC EDGAR and parsed into "
    "ordered text/table blocks.\n"
    "2. **Chunk**: tables are kept intact (never split mid-table) so a "
    "figure and its header stay together -- a real bug found and fixed "
    "in Week 2.\n"
    "3. **Retrieve**: hybrid search combines dense embeddings (FAISS) "
    "with keyword search (BM25), fused via Reciprocal Rank Fusion, then "
    "re-scored by a cross-encoder reranker. The company/fiscal year "
    "named in the question narrows the search first, so retrieval isn't "
    "disambiguating company, year, *and* topic all at once.\n"
    "4. **Answer**: Gemini generates an answer grounded only in the "
    "retrieved excerpts, citing the source filing.\n"
    "5. **Agent mode** (optional): a LangGraph agent decomposes "
    "multi-part or comparison questions into sub-questions, retries "
    "retrieval when context looks insufficient, and can invoke a "
    "calculator tool for derived figures.\n\n"
    "Full write-up with real bugs found and fixed at each stage: "
    "`TECHNICAL_REPORT.md` in the repo."
)

st.header("📈 Measured results")
st.caption("Week 3 RAGAS eval, 20-question subset of FinanceBench")
st.markdown(
    "Three retrieval configs, scored on the same 20 real FinanceBench "
    "questions -- not a claim, an actual run logged to MLflow "
    "(`data/processed/eval_runs/`, `mlruns/`):"
)
st.table(
    {
        "Config": ["naive + dense only", "hybrid + BM25 (no rerank)", "hybrid + BM25 + reranker"],
        "Faithfulness": ["0.787", "0.805", "0.912"],
        "Context precision": ["0.087", "0.117", "0.092"],
        "Context recall": ["0.150", "0.100", "0.100"],
        "Numerical accuracy": ["0.211", "0.211", "0.211"],
    }
)
st.caption(
    "Faithfulness (is the answer actually supported by the retrieved "
    "text) improves clearly with hybrid retrieval + reranking. Context "
    "precision/recall are low across the board -- an honest, measured "
    "result, not hidden -- and are exactly what the company/fiscal-year "
    "filtering fix below targets. Full methodology in "
    "`TECHNICAL_REPORT.md` section 4."
)

st.header("⚠️ Known limitations")
st.markdown(
    "- **Retrieval disambiguation**: fixed a regression where growing "
    "the corpus caused same-topic chunks from the wrong company or "
    "fiscal year to outrank the right one -- retrieval now filters by "
    "company/year detected in the question.\n"
    "- **Still open**: a specific-figure question can fail when the "
    "right number sits inside a large table chunk with many unrelated "
    "line items, or when the question's wording (\"capital "
    "expenditure\") differs from the filing's exact phrasing "
    "(\"purchases of property, plant and equipment\"). Diagnosed, not "
    "yet fully fixed -- documented rather than hidden.\n"
    "- **Shared free-tier quota**: this demo shares one Gemini API key "
    "across every visitor (500 requests/day). Heavy use may exhaust it "
    "for the day.\n\n"
    "Full detail: `TECHNICAL_REPORT.md`."
)

st.header("Contact")
st.markdown(
    "Built by **Kimmeng Hong**.\n\n"
    "📧 [hongkimmeng17@gmail.com](mailto:hongkimmeng17@gmail.com)\n\n"
    "💻 [GitHub repo](https://github.com/Kimmeng007/findocqa)\n\n"
    "🔗 [LinkedIn](https://www.linkedin.com/in/kimmeng-hong-09109222a/)\n\n"
    "🌐 [Portfolio site](https://kimmeng007.github.io/)"
)
