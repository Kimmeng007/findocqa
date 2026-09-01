"""Builds the FAISS + BM25 indexes from every filing currently in MongoDB.

Usage:
    uv run scripts/build_index.py --variant table_aware
    uv run scripts/build_index.py --variant naive
"""

import argparse

from findocqa.retrieval.build import CHUNKERS, build_variant_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=CHUNKERS.keys(), default="table_aware")
    args = parser.parse_args()

    print(f"[{args.variant}] Chunking + embedding + indexing...")
    n_chunks = build_variant_index(args.variant)

    if n_chunks == 0:
        print("No filings found in MongoDB — run scripts/ingest.py first.")
        return
    print(f"[{args.variant}] Indexed {n_chunks} chunks.")


if __name__ == "__main__":
    main()
