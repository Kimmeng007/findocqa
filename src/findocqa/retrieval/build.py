"""Shared index-build logic used by scripts/build_index.py and
scripts/run_eval.py (which needs to rebuild both variants' indexes after
auto-ingesting whatever filings an eval subset references)."""

from findocqa.retrieval import index_cache
from findocqa.retrieval.bm25_index import build_bm25_index
from findocqa.retrieval.chunking import chunk_filing_naive, chunk_filing_table_aware
from findocqa.retrieval.vector_store import build_index
from findocqa.storage.mongo import iter_filings

CHUNKERS = {
    "naive": chunk_filing_naive,
    "table_aware": chunk_filing_table_aware,
}


def build_variant_index(variant: str) -> int:
    """Chunks every filing currently in MongoDB with the given variant's
    chunker and (re)builds its FAISS + BM25 indexes. Returns the chunk
    count, or 0 if there were no filings to index."""
    chunk_fn = CHUNKERS[variant]

    chunks = []
    for filing in iter_filings():
        chunks.extend(chunk_fn(filing))

    if not chunks:
        return 0

    build_index(chunks, variant=variant)
    build_bm25_index(chunks, variant=variant)
    # Every rebuild refreshes MongoDB's cached copy too, so the next
    # container (a fresh Streamlit Cloud deploy, most importantly) can
    # download instead of re-embedding the whole corpus from scratch.
    index_cache.upload_index(variant)
    return len(chunks)
