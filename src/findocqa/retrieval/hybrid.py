"""Hybrid retrieval: dense (FAISS) + sparse (BM25) candidates, deduplicated
via Reciprocal Rank Fusion for ordering, then reranked with a cross-encoder.

RRF is used to combine the two ranked lists into one deduplicated candidate
pool (its rank-only inputs avoid tuning a weight between incomparable
cosine-similarity and BM25 score scales) — but RRF itself is *not* trusted
to make the final relevance call. RRF structurally penalizes a candidate
that only one retriever found: a financial figure sitting in a numbers-heavy
table can be a very strong dense match (little literal keyword overlap for
BM25 to grab) yet still get outranked by mediocre candidates both retrievers
found. So every fused candidate is forwarded to the cross-encoder, which
reads the actual query+chunk pair — only the reranker's output is truncated
to top_k, never RRF's.
"""

from findocqa.retrieval import bm25_index, vector_store
from findocqa.retrieval.reranker import rerank

RRF_K = 60
CANDIDATES_PER_RETRIEVER = 30


def _chunk_key(chunk: dict) -> tuple:
    return (chunk["doc_name"], chunk["chunk_index"])


def _reciprocal_rank_fusion(*ranked_lists: list[dict]) -> list[dict]:
    scores: dict[tuple, float] = {}
    chunks_by_key: dict[tuple, dict] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = _chunk_key(chunk)
            chunks_by_key[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

    ordered_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [{**chunks_by_key[k], "rrf_score": scores[k]} for k in ordered_keys]


def hybrid_search(query: str, top_k: int = 5, variant: str = "table_aware") -> list[dict]:
    dense = vector_store.search(query, top_k=CANDIDATES_PER_RETRIEVER, variant=variant)
    sparse = bm25_index.search_bm25(query, top_k=CANDIDATES_PER_RETRIEVER, variant=variant)

    fused = _reciprocal_rank_fusion(dense, sparse)
    return rerank(query, fused, top_k=top_k)
