"""LangGraph node functions. Each node takes the shared AgentState and
returns a partial-state dict update (LangGraph merges it in).

Every real Gemini call funnels through the same `throttle()` used
everywhere else in this project (generation/answer.py, eval/ragas_harness.py)
-- there is exactly one place in this codebase that talks to Gemini
without it, and that's a bug, not a design choice, per today's Week 3
rate-limiting incident.
"""

from google import genai
from google.genai import types
from pydantic import BaseModel

from findocqa.agent.state import AgentState, SubAnswer
from findocqa.config import settings
from findocqa.generation.answer import _retrieve
from findocqa.generation.rate_limit import throttle

MAX_RETRIES_PER_SUBQUESTION = 2

# The model's own honest refusal phrasing, already observed constantly in
# this project's real output (Weeks 1-3) -- a free, already-validated
# signal that retrieval came up short, rather than a new heuristic.
_INSUFFICIENT_CONTEXT_MARKERS = (
    "not enough information",
    "no information",  # catches "there is no information about..." too
    "does not contain",
    "there is no mention",
    "cannot answer",
    "does not provide",
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Given a question about SEC "
    "filings, decide whether it needs to be broken into multiple simpler "
    "sub-questions to answer completely (e.g. a comparison across "
    "companies or years needs one sub-question per data point). If the "
    "question is already a single, simple lookup, return it unchanged as "
    "the only sub-question."
)

_SYNTHESIZE_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. You are given a original "
    "question and a set of sub-answers already gathered from SEC filings. "
    "Combine them into one clear, final answer to the original question, "
    "citing source documents. If a sub-answer says information was "
    "missing, acknowledge that gap rather than guessing."
)


class Decomposition(BaseModel):
    sub_questions: list[str]


def _client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


def decompose(state: AgentState) -> dict:
    throttle()
    client = _client()  # kept alive for the whole call, including any SDK-internal retries
    response = client.models.generate_content(
        model=settings.generation_model,
        contents=state["question"],
        config=types.GenerateContentConfig(
            system_instruction=_DECOMPOSE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Decomposition,
        ),
    )
    result: Decomposition = response.parsed
    sub_questions = result.sub_questions or [state["question"]]
    return {"sub_questions": sub_questions, "sub_answers": [], "current_index": 0}


def _looks_insufficient(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _INSUFFICIENT_CONTEXT_MARKERS)


def _generate_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['doc_name']} | chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
    )
    throttle()
    client = _client()
    response = client.models.generate_content(
        model=settings.generation_model,
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            system_instruction=(
                "Answer using only the provided context. If it's not "
                "enough, say so explicitly instead of guessing."
            )
        ),
    )
    return response.text


def answer_subquestion(state: AgentState) -> dict:
    """Answers state['sub_questions'][current_index], with a bounded
    self-correction retry loop: if the answer looks like a genuine
    information gap, reformulate the sub-question and retry retrieval
    before giving up -- capped, since unbounded retries against a
    rate-limited API make things worse, not better (see today's incident)."""
    index = state["current_index"]
    question = state["sub_questions"][index]

    retry_count = 0
    chunks = _retrieve(question, top_k=5, variant="table_aware", mode="hybrid")
    answer = _generate_answer(question, chunks)

    while _looks_insufficient(answer) and retry_count < MAX_RETRIES_PER_SUBQUESTION:
        retry_count += 1
        reformulated = _reformulate(question)
        chunks = _retrieve(reformulated, top_k=5, variant="table_aware", mode="hybrid")
        answer = _generate_answer(reformulated, chunks)

    sub_answer: SubAnswer = {
        "sub_question": question,
        "answer": answer,
        "retrieved_chunks": chunks,
        "retry_count": retry_count,
    }
    return {
        "sub_answers": state["sub_answers"] + [sub_answer],
        "current_index": index + 1,
    }


def _reformulate(question: str) -> str:
    throttle()
    client = _client()
    response = client.models.generate_content(
        model=settings.generation_model,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=(
                "Rephrase this financial-filing question to use different "
                "keywords and phrasing that might match the source "
                "document's actual wording more closely, while asking for "
                "exactly the same information. You MUST keep any company "
                "name and any specific fiscal year or date mentioned "
                "exactly as they appear in the original -- retrieval uses "
                "them to narrow the search, so dropping or paraphrasing "
                "them (e.g. into a pronoun) breaks that. Only vary the "
                "financial terminology and general phrasing. Return only "
                "the rephrased question, nothing else."
            )
        ),
    )
    return response.text.strip()


def has_more_subquestions(state: AgentState) -> str:
    if state["current_index"] < len(state["sub_questions"]):
        return "continue"
    return "synthesize"


def synthesize(state: AgentState) -> dict:
    if len(state["sub_answers"]) == 1:
        return {"final_answer": state["sub_answers"][0]["answer"]}

    sub_answers_text = "\n\n".join(
        f"Sub-question: {sa['sub_question']}\nAnswer: {sa['answer']}"
        for sa in state["sub_answers"]
    )
    throttle()
    client = _client()
    response = client.models.generate_content(
        model=settings.generation_model,
        contents=(
            f"Original question: {state['question']}\n\n"
            f"Gathered sub-answers:\n{sub_answers_text}"
        ),
        config=types.GenerateContentConfig(system_instruction=_SYNTHESIZE_SYSTEM_PROMPT),
    )
    return {"final_answer": response.text}
