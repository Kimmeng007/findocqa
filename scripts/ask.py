"""Smoke-test the baseline RAG pipeline with a single question.

Usage:
    uv run scripts/ask.py "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
"""

import argparse

from findocqa.generation.answer import answer_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = answer_question(args.question, top_k=args.top_k)

    print("Retrieved context:")
    for chunk in result["retrieved_chunks"]:
        print(f"  [{chunk['score']:.3f}] {chunk['doc_name']} chunk {chunk['chunk_index']}")

    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
