from langgraph.graph import END, START, StateGraph

from findocqa.agent.nodes import answer_subquestion, decompose, has_more_subquestions, synthesize
from findocqa.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("decompose", decompose)
    graph.add_node("answer_subquestion", answer_subquestion)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "answer_subquestion")
    graph.add_conditional_edges(
        "answer_subquestion",
        has_more_subquestions,
        {"continue": "answer_subquestion", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_agent(question: str) -> dict:
    compiled = build_graph()
    result = compiled.invoke(
        {
            "question": question,
            "sub_questions": [],
            "sub_answers": [],
            "current_index": 0,
            "final_answer": "",
            "synthesis_tool_calls": [],
        }
    )
    return result
