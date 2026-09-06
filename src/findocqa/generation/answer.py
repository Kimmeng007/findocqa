from google import genai
from google.genai import types

from findocqa.config import settings
from findocqa.generation.rate_limit import throttle
from findocqa.retrieval import vector_store
from findocqa.retrieval.company_filter import detect_company, detect_fiscal_year
from findocqa.retrieval.hybrid import hybrid_search

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about SEC "
    "filings using only the provided context. If the context does not "
    "contain enough information to answer, say so explicitly instead of "
    "guessing. Cite the source document name when you use a figure."
)

_MODES = ("dense", "hybrid", "hybrid_no_rerank")


def _retrieve_with(
    question: str, top_k: int, variant: str, mode: str, company: str | None, fiscal_year: int | None
) -> list[dict]:
    if mode == "dense":
        return vector_store.search(
            question, top_k=top_k, variant=variant, company=company, fiscal_year=fiscal_year
        )
    if mode == "hybrid":
        return hybrid_search(
            question,
            top_k=top_k,
            variant=variant,
            use_reranker=True,
            company=company,
            fiscal_year=fiscal_year,
        )
    if mode == "hybrid_no_rerank":
        return hybrid_search(
            question,
            top_k=top_k,
            variant=variant,
            use_reranker=False,
            company=company,
            fiscal_year=fiscal_year,
        )
    raise ValueError(f"Unknown mode {mode!r}, expected one of {_MODES}")


def _retrieve(question: str, top_k: int, variant: str, mode: str) -> list[dict]:
    # Narrowing to the named company and fiscal year first turns a
    # multi-company, multi-year corpus-wide search into a small, specific
    # one -- neither dense similarity nor BM25 reliably disambiguate either
    # dimension on their own once a company has many same-shaped tables
    # across years. See company_filter.py.
    company = detect_company(question)
    fiscal_year = detect_fiscal_year(question)

    chunks = _retrieve_with(question, top_k, variant, mode, company, fiscal_year)
    if chunks:
        return chunks

    # A multi-year question ("FY2018-2020 average", "FY2016 to FY2017
    # change") names a year that might not be the one actually ingested
    # for that company (found via a real eval run: MGM Resorts/Amazon/
    # Walmart questions naming a year outside what's in the corpus came
    # back with *zero* chunks, even though a different year of the same
    # company's filing existed and could have partially answered). A
    # hard filter should never do worse than no filter at all, so relax
    # it step by step instead of returning nothing.
    if fiscal_year is not None:
        chunks = _retrieve_with(question, top_k, variant, mode, company, None)
        if chunks:
            return chunks
    if company is not None:
        chunks = _retrieve_with(question, top_k, variant, mode, None, None)
    return chunks


def answer_question(
    question: str,
    top_k: int = 5,
    variant: str = "table_aware",
    mode: str = "hybrid",
) -> dict:
    chunks = _retrieve(question, top_k, variant, mode)

    context = "\n\n---\n\n".join(
        f"[{c['doc_name']} | chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
    )

    client = genai.Client(api_key=settings.google_api_key)
    throttle()
    response = client.models.generate_content(
        model=settings.generation_model,
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
    )

    return {
        "question": question,
        "answer": response.text,
        "retrieved_chunks": chunks,
    }
