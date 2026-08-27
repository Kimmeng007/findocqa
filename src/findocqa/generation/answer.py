from google import genai
from google.genai import types

from findocqa.config import settings
from findocqa.retrieval.vector_store import search

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about SEC "
    "filings using only the provided context. If the context does not "
    "contain enough information to answer, say so explicitly instead of "
    "guessing. Cite the source document name when you use a figure."
)


def answer_question(question: str, top_k: int = 5) -> dict:
    chunks = search(question, top_k=top_k)
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
