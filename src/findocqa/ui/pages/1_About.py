"""About page: architecture, measured eval results, known limitations, and
contact -- split out from app.py so the main page stays focused on the
interactive Q&A demo instead of a long scroll of write-up.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="About — FinDocQA", page_icon="📊")

PHOTO_PATH = Path(__file__).parent.parent / "assets" / "profile.jpg"

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
st.caption(
    "Same 20-question FinanceBench subset, same config (hybrid + BM25 + "
    "reranker) — before vs. after this session's retrieval fixes "
    "(company/fiscal-year filtering with graceful fallback, finer table "
    "chunking, wider candidate pool when filtered):"
)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Faithfulness", "0.850", "-0.062 vs. before")
col2.metric("Context precision", "0.122", "+0.030 vs. before")
col3.metric("Context recall", "0.167", "+0.067 vs. before")
col4.metric("Numerical accuracy", "0.263", "+0.052 vs. before")
st.caption(
    "Precision, recall, and numerical accuracy — the metrics these fixes "
    "targeted — all improved. Faithfulness dropped somewhat: broader "
    "retrieval surfaces more content, and not all of it is as tightly "
    "quote-supported by the generated answer. A real, reported trade-off, "
    "not hidden. Full root-cause analysis in `TECHNICAL_REPORT.md` "
    "section 7."
)

with st.expander("Full breakdown, all configs and both eval runs"):
    st.markdown(
        "Not a claim -- an actual run logged to MLflow "
        "(`data/processed/eval_runs/`, `mlruns/`):"
    )
    st.table(
        {
            "Config": [
                "naive + dense only",
                "hybrid + BM25 (no rerank)",
                "hybrid + BM25 + reranker (before fixes)",
                "hybrid + BM25 + reranker (after fixes)",
            ],
            "Faithfulness": ["0.787", "0.805", "0.912", "0.850"],
            "Context precision": ["0.087", "0.117", "0.092", "0.122"],
            "Context recall": ["0.150", "0.100", "0.100", "0.167"],
            "Numerical accuracy": ["0.211", "0.211", "0.211", "0.263"],
        }
    )
    st.caption(
        "Context precision/recall remain low in absolute terms even "
        "after these fixes -- an honest, measured result, not hidden. "
        "Tested whether swapping in a finance-tuned embedding model and "
        "two stronger rerankers would close the gap (TECHNICAL_REPORT.md "
        "7.11) -- none of the 5 models tried fixed the underlying issue, "
        "which turned out not to be a model-quality problem at all: a "
        "structural fix (extracting table rows into labeled facts rather "
        "than matching passages by similarity) is the real next step."
    )

st.header("⚠️ Known limitations")
st.markdown(
    "- **Retrieval disambiguation**: fixed a regression where growing "
    "the corpus caused same-topic chunks from the wrong company or "
    "fiscal year to outrank the right one -- retrieval now filters by "
    "company/year detected in the question, and falls back to a looser "
    "filter rather than returning nothing if the named year isn't the "
    "one actually in the corpus for that company.\n"
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
with st.container(border=True):
    col_photo, col_info = st.columns([1, 3])
    with col_photo:
        if PHOTO_PATH.exists():
            st.image(str(PHOTO_PATH), width=150)
        else:
            st.markdown("### 🧑‍💻")
    with col_info:
        st.markdown("### Kimmeng Hong")
        st.caption("Builder of FinDocQA")
        st.markdown(
            "📧 [hongkimmeng17@gmail.com](mailto:hongkimmeng17@gmail.com)  \n"
            "💻 [GitHub repo](https://github.com/Kimmeng007/findocqa)  \n"
            "🔗 [LinkedIn](https://www.linkedin.com/in/kimmeng-hong-09109222a/)  \n"
            "🌐 [Portfolio site](https://kimmeng007.github.io/)"
        )
