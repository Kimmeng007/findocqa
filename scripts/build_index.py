"""Builds the baseline FAISS index from every filing currently in MongoDB.

Usage:
    uv run scripts/build_index.py
"""

from findocqa.retrieval.chunking import chunk_filing
from findocqa.retrieval.vector_store import build_index
from findocqa.storage.mongo import iter_filings


def main() -> None:
    chunks = []
    n_filings = 0
    for filing in iter_filings():
        n_filings += 1
        chunks.extend(chunk_filing(filing))

    if not chunks:
        print("No filings found in MongoDB — run scripts/ingest.py first.")
        return

    print(f"Chunked {n_filings} filing(s) into {len(chunks)} chunks. Embedding + indexing...")
    build_index(chunks)
    print("Index built.")


if __name__ == "__main__":
    main()
