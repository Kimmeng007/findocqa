from typing import TypedDict


class SubAnswer(TypedDict):
    sub_question: str
    answer: str
    retrieved_chunks: list[dict]
    retry_count: int
    tool_calls: list[dict]


class AgentState(TypedDict):
    question: str
    sub_questions: list[str]
    sub_answers: list[SubAnswer]
    current_index: int
    final_answer: str
