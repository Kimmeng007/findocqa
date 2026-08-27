"""Builds the FAISS + BM25 indexes from every filing currently in MongoDB.

Usage:
    uv run scripts/build_index.py --variant table_aware
    uv run scripts/build_index.py --variant naive
"""

import argparse

from findocqa.retrieval.bm25_index import build_bm25_index
from findocqa.retrieval.chunking import chunk_filing_naive, chunk_filing_table_aware
from findocqa.retrieval.vector_store import build_index
from findocqa.storage.mongo import iter_filings

_CHUNKERS = {
    "naive": chunk_filing_naive,
    "table_aware": chunk_filing_table_aware,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=_CHUNKERS.keys(), default="table_aware")
    args = parser.parse_args()

    chunk_fn = _CHUNKERS[args.variant]

    chunks = []
    n_filings = 0
    for filing in iter_filings():
        n_filings += 1
        chunks.extend(chunk_fn(filing))

    if not chunks:
        print("No filings found in MongoDB — run scripts/ingest.py first.")
        return

    print(
        f"[{args.variant}] Chunked {n_filings} filing(s) into {len(chunks)} chunks. "
        "Embedding + indexing..."
    )
    build_index(chunks, variant=args.variant)
    build_bm25_index(chunks, variant=args.variant)
    print(f"[{args.variant}] FAISS + BM25 indexes built.")


if __name__ == "__main__":
    main()
