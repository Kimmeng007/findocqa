from google import genai
from google.genai import types

from findocqa.config import settings
from findocqa.retrieval import vector_store
from findocqa.retrieval.hybrid import hybrid_search

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about SEC "
    "filings using only the provided context. If the context does not "
    "contain enough information to answer, say so explicitly instead of "
    "guessing. Cite the source document name when you use a figure."
)

_MODES = ("dense", "hybrid", "hybrid_no_rerank")


def _retrieve(question: str, top_k: int, variant: str, mode: str) -> list[dict]:
    if mode == "dense":
        return vector_store.search(question, top_k=top_k, variant=variant)
    if mode == "hybrid":
        return hybrid_search(question, top_k=top_k, variant=variant, use_reranker=True)
    if mode == "hybrid_no_rerank":
        return hybrid_search(question, top_k=top_k, variant=variant, use_reranker=False)
    raise ValueError(f"Unknown mode {mode!r}, expected one of {_MODES}")


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
