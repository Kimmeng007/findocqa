"""Ask a (possibly multi-step) question against the LangGraph agent.

Unlike scripts/ask.py (single retrieve->prompt->Gemini call), this can
decompose a comparison/multi-part question into sub-questions, retrieve
and answer each separately, and synthesize a final combined answer.

Usage:
    uv run scripts/agent_ask.py "Compare 3M and Amazon's revenue growth over the last two fiscal years"
"""

import argparse

from findocqa.agent.graph import run_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    args = parser.parse_args()

    result = run_agent(args.question)

    print(f"Question: {args.question}")
    print(f"\nDecomposed into {len(result['sub_questions'])} sub-question(s):")
    for sa in result["sub_answers"]:
        retry_note = f" (after {sa['retry_count']} retry)" if sa["retry_count"] else ""
        print(f"\n  Sub-question: {sa['sub_question']}{retry_note}")
        print(f"  Answer: {sa['answer']}")

    print(f"\nFinal answer:\n{result['final_answer']}")


if __name__ == "__main__":
    main()
