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
    "##### Ask questions about SEC financial filings, answered from the filing text itself."
)
st.caption(
    "Not the model's general knowledge — every answer cites the exact "
    "filing excerpt it came from, so you can verify it yourself. "
    "Evaluated against the public FinanceBench benchmark. See the "
    "**About** page (sidebar) for architecture, measured results, and "
    "known limitations."
)


@st.cache_data(ttl=300)
def _corpus_summary() -> list[dict]:
    return get_ingested_filings_summary()


def _friendly_error(exc: Exception) -> str:
    """Public demos share one Gemini free-tier quota across every visitor
    -- this project hit that exact 500-requests/day cap twice during its
    own development (see TECHNICAL_REPORT.md), so a stranger hitting it
    on a public deployment is a real, expected scenario, not a rare edge
    case worth a raw traceback."""
    message = str(exc)
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return (
            "⚠️ This demo shares one free-tier Gemini API quota "
            "(500 requests/day) across everyone who tries it, and it's "
            "been used up for today. Please try again later -- see "
            "TECHNICAL_REPORT.md for why this limit exists."
        )
    return f"Request failed: {exc}"


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

st.info(
    f"**What you can ask about:** only companies/years actually loaded below "
    f"({len(filings)} filings across {len(by_company)} companies) -- a "
    "question about a company or year not listed here will honestly come "
    "back \"not enough information\" rather than a guess.",
    icon="📁",
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

    st.divider()
    st.caption(
        "Built by **Kimmeng Hong** — "
        "[Email](mailto:hongkimmeng17@gmail.com) · "
        "[GitHub](https://github.com/Kimmeng007/findocqa) · "
        "[LinkedIn](https://www.linkedin.com/in/kimmeng-hong-09109222a/) · "
        "[Portfolio](https://kimmeng007.github.io/)\n\n"
        "See the **About** page for the full write-up."
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
                st.error(_friendly_error(exc))
                st.stop()

        with st.container(border=True):
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
                st.error(_friendly_error(exc))
                st.stop()

        with st.container(border=True):
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
st.caption("Portfolio project by Kimmeng Hong — not a production financial advisory tool. See the **About** page for details.")
