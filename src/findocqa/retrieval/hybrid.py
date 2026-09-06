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

# When company/fiscal-year filtering has already narrowed the search space,
# a wider candidate pool gives the reranker more chances to see the right
# chunk for exactly the cases this filtering targets. FAISS/BM25 scoring
# itself is nearly free either way (both already score everything
# internally regardless of top_k), but cross-encoder reranking cost scales
# with pool size -- measured ~3.6s at 30/retriever vs. ~8.3s at 100/retriever
# with warm models. 60 was picked empirically as a middle ground: a real
# 50% wider pool for ~1s added latency, not a 2x wait for marginal gains
# past that (see TECHNICAL_REPORT.md).
CANDIDATES_WHEN_FILTERED = 60


def _candidate_pool_size(company: str | None, fiscal_year: int | None) -> int:
    if company or fiscal_year:
        return CANDIDATES_WHEN_FILTERED
    return CANDIDATES_PER_RETRIEVER


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


def hybrid_search(
    query: str,
    top_k: int = 5,
    variant: str = "table_aware",
    use_reranker: bool = True,
    company: str | None = None,
    fiscal_year: int | None = None,
) -> list[dict]:
    pool_size = _candidate_pool_size(company, fiscal_year)
    dense = vector_store.search(
        query, top_k=pool_size, variant=variant, company=company, fiscal_year=fiscal_year
    )
    sparse = bm25_index.search_bm25(
        query, top_k=pool_size, variant=variant, company=company, fiscal_year=fiscal_year
    )

    fused = _reciprocal_rank_fusion(dense, sparse)
    if not use_reranker:
        results = []
        for c in fused[:top_k]:
            c = dict(c)
            c["score"] = c.pop("rrf_score")
            results.append(c)
        return results
    return rerank(query, fused, top_k=top_k)
