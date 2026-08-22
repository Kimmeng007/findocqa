import anthropic

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

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.generation_model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    return {
        "question": question,
        "answer": response.content[0].text,
        "retrieved_chunks": chunks,
    }
