"""Ask a question against the RAG pipeline.

Usage:
    uv run scripts/ask.py "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
    uv run scripts/ask.py "..." --variant naive --mode dense
"""

import argparse

from findocqa.generation.answer import answer_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variant", choices=["naive", "table_aware"], default="table_aware")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    args = parser.parse_args()

    result = answer_question(
        args.question,
        top_k=args.top_k,
        variant=args.variant,
        use_hybrid=(args.mode == "hybrid"),
    )

    print(f"Retrieved context ({args.variant} / {args.mode}):")
    for chunk in result["retrieved_chunks"]:
        print(f"  [{chunk['score']:.3f}] {chunk['doc_name']} chunk {chunk['chunk_index']}")

    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
