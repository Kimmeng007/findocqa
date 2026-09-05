"""Streamlit demo UI.

A thin wrapper around the pipeline that already exists and is already
tested (generation.answer.answer_question, agent.graph.run_agent) -- no
new retrieval/generation logic here, just a way to actually click and see
it work instead of running CLI scripts.

Run with: uv run streamlit run src/findocqa/ui/app.py
"""

from collections import defaultdict

import streamlit as st

from findocqa.generation.answer import answer_question
from findocqa.agent.graph import run_agent
from findocqa.storage.mongo import get_ingested_filings_summary

st.set_page_config(page_title="FinDocQA", page_icon="📊", layout="wide")

st.title("📊 FinDocQA")
st.markdown(
    "**What this is:** ask a plain-English question about a company's SEC "
    "financial filing (10-K annual report or 10-Q quarterly report) and get "
    "an answer grounded in the actual filing text -- not the model's general "
    "knowledge. Every answer shows exactly which filing excerpt it came "
    "from, so you can verify it yourself instead of taking it on faith.\n\n"
    "**Who this is for:** a research prototype for anyone who'd otherwise "
    "have to manually read a 100+ page filing to find one figure -- "
    "analysts, investors, or researchers doing financial-filing lookups. "
    "See `PROGRESS.md` for the full story."
)


@st.cache_data(ttl=300)
def _corpus_summary() -> list[dict]:
    return get_ingested_filings_summary()


try:
    filings = _corpus_summary()
except Exception as exc:
    st.error(
        f"Could not connect to MongoDB ({exc}). Check `.env` and Atlas "
        "Network Access (see TECHNICAL_REPORT.md if this is a recurring issue)."
    )
    st.stop()

by_company: dict[str, list[str]] = defaultdict(list)
for f in filings:
    by_company[f["company"]].append(f"{f['filing_type'].upper()} FY{f['fiscal_year']}")

st.markdown(
    f"**What you can ask about:** only companies/years actually loaded below "
    f"({len(filings)} filings across {len(by_company)} companies) -- a "
    "question about a company or year not listed here will honestly come "
    "back \"not enough information\" rather than a guess."
)
with st.expander("See exactly which filings are loaded"):
    for company in sorted(by_company):
        st.markdown(f"- **{company}**: {', '.join(by_company[company])}")


EXAMPLE_QUESTIONS = [
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
    "Does 3M maintain a stable trend of dividend distribution?",
    "Compare 3M and Amazon's revenue growth over the last two fiscal years",
]

with st.sidebar:
    st.header("Try an example")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q

    st.divider()
    st.header("Mode")
    mode_choice = st.radio(
        "How should this question be answered?",
        ["Simple (single retrieval)", "Agent (multi-step, can decompose)"],
        help=(
            "Simple: one retrieval + one Gemini call, best for direct "
            "factual lookups. Agent: can break a comparison/multi-part "
            "question into sub-questions and retry when retrieval comes "
            "up short -- costs more API calls per question."
        ),
    )

    if mode_choice.startswith("Simple"):
        st.divider()
        st.header("Retrieval config")
        variant = st.selectbox(
            "Chunking variant",
            ["table_aware", "naive"],
            help="table_aware (Week 2) fixes real bugs found in naive (Week 1) chunking -- see TECHNICAL_REPORT.md.",
        )
        pipeline_mode = st.selectbox(
            "Retrieval mode",
            ["hybrid", "dense", "hybrid_no_rerank"],
            help="hybrid = dense+BM25+reranker. See PROGRESS.md for the measured Week 3 comparison of these.",
        )

question = st.text_area(
    "Ask a question about a filing already in the corpus:",
    value=st.session_state.get("question", ""),
    height=80,
)

run_clicked = st.button("Ask", type="primary")

if run_clicked and question.strip():
    if mode_choice.startswith("Simple"):
        with st.spinner("Retrieving and generating..."):
            try:
                result = answer_question(question, variant=variant, mode=pipeline_mode)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander(f"Retrieved sources ({len(result['retrieved_chunks'])})"):
            for c in result["retrieved_chunks"]:
                st.markdown(f"**{c['doc_name']}** · chunk {c['chunk_index']} · score {c['score']:.3f}")
                st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
                st.divider()
    else:
        with st.spinner("Decomposing, retrieving, and synthesizing (this makes several Gemini calls)..."):
            try:
                result = run_agent(question)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        st.subheader("Final answer")
        st.write(result["final_answer"])

        with st.expander(f"Sub-questions ({len(result['sub_answers'])})"):
            for sa in result["sub_answers"]:
                retry_note = f" _(after {sa['retry_count']} retry)_" if sa["retry_count"] else ""
                st.markdown(f"**{sa['sub_question']}**{retry_note}")
                st.write(sa["answer"])
                st.divider()

elif run_clicked:
    st.warning("Type a question first.")

st.divider()
st.caption(
    "This is a portfolio research project, not a production financial "
    "advisory tool -- see PROGRESS.md / TECHNICAL_REPORT.md for measured "
    "accuracy numbers and known limitations. Gemini's free tier caps at "
    "500 requests/day; heavy use may hit that limit."
)
