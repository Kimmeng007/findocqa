from langgraph.graph import END, START, StateGraph

from findocqa.agent.nodes import (
    MAX_RETRIES_PER_SUBQUESTION,
    _looks_insufficient,
    has_more_subquestions,
)
from findocqa.agent.state import AgentState


def test_looks_insufficient_detects_the_models_own_refusal_phrasing():
    # These are the model's actual real refusal phrasings observed
    # constantly in this project's Week 1-3 output, not invented strings.
    assert _looks_insufficient("There is no mention of the figure in the context.")
    assert _looks_insufficient("The context does not contain enough information.")
    assert _looks_insufficient("Therefore, I cannot answer the question.")


def test_looks_insufficient_regression_missed_phrasing_from_first_live_agent_run():
    # The first real live agent run (Week 4) produced exactly this
    # phrasing and the trigger silently missed it -- self-correction
    # never fired when it should have. Locking this in as a regression
    # test now that the marker list has been broadened to catch it.
    assert _looks_insufficient(
        "Based on the provided context, there is no information about "
        "3M's revenue growth for the last two fiscal years."
    )


def test_looks_insufficient_false_for_a_real_answer():
    assert not _looks_insufficient(
        "Based on the provided context, 3M's FY2018 capital expenditure was $1,577 million."
    )


def test_has_more_subquestions_continues_when_index_below_length():
    state = {"sub_questions": ["a", "b", "c"], "current_index": 1}
    assert has_more_subquestions(state) == "continue"


def test_has_more_subquestions_synthesizes_when_all_answered():
    state = {"sub_questions": ["a", "b"], "current_index": 2}
    assert has_more_subquestions(state) == "synthesize"


def test_has_more_subquestions_synthesizes_for_single_question():
    state = {"sub_questions": ["only one"], "current_index": 1}
    assert has_more_subquestions(state) == "synthesize"


def test_retry_cap_constant_is_bounded():
    # Guards against silently removing the retry cap in a future edit --
    # today's rate-limiting incident is exactly why this must stay small.
    assert 0 < MAX_RETRIES_PER_SUBQUESTION <= 3


def test_graph_shape_loops_over_subquestions_then_synthesizes():
    """Exercises the real StateGraph wiring -- same node names and same
    conditional-edge routing as agent/graph.py -- with stub nodes standing
    in for the real Gemini-calling ones, so the loop-then-terminate
    behavior is verified without spending API quota."""
    calls = {"answer_subquestion": 0, "synthesize": 0}

    def stub_decompose(state: AgentState) -> dict:
        return {"sub_questions": ["q1", "q2", "q3"], "sub_answers": [], "current_index": 0}

    def stub_answer_subquestion(state: AgentState) -> dict:
        calls["answer_subquestion"] += 1
        index = state["current_index"]
        sub_answer = {
            "sub_question": state["sub_questions"][index],
            "answer": f"answer for {state['sub_questions'][index]}",
            "retrieved_chunks": [],
            "retry_count": 0,
        }
        return {"sub_answers": state["sub_answers"] + [sub_answer], "current_index": index + 1}

    def stub_synthesize(state: AgentState) -> dict:
        calls["synthesize"] += 1
        return {"final_answer": "combined answer"}

    graph = StateGraph(AgentState)
    graph.add_node("decompose", stub_decompose)
    graph.add_node("answer_subquestion", stub_answer_subquestion)
    graph.add_node("synthesize", stub_synthesize)
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "answer_subquestion")
    graph.add_conditional_edges(
        "answer_subquestion",
        has_more_subquestions,
        {"continue": "answer_subquestion", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    compiled = graph.compile()

    result = compiled.invoke(
        {
            "question": "compare three things",
            "sub_questions": [],
            "sub_answers": [],
            "current_index": 0,
            "final_answer": "",
        }
    )

    assert calls["answer_subquestion"] == 3  # looped exactly once per sub-question
    assert calls["synthesize"] == 1  # terminated correctly, no infinite loop
    assert len(result["sub_answers"]) == 3
    assert result["final_answer"] == "combined answer"
