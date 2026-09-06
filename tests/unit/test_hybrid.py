from findocqa.retrieval.hybrid import (
    CANDIDATES_PER_RETRIEVER,
    CANDIDATES_WHEN_FILTERED,
    _candidate_pool_size,
    _reciprocal_rank_fusion,
)


def _fake_chunk(doc_name: str, chunk_index: int) -> dict:
    return {"doc_name": doc_name, "chunk_index": chunk_index, "text": f"chunk {chunk_index}"}


def test_fusion_keeps_full_union_not_just_top_of_either_list():
    """Regression guard: a candidate a dense-only search ranked #1 must
    survive fusion even if BM25 never found it at all — RRF should combine
    the two lists, not silently drop whichever one is disjoint from the
    other. (This is what let a strong dense-only match get truncated away
    before the reranker ever saw it.)"""
    dense = [_fake_chunk("DOC", i) for i in range(30)]
    sparse = [_fake_chunk("DOC", i) for i in range(100, 130)]

    fused = _reciprocal_rank_fusion(dense, sparse)

    assert len(fused) == 60
    keys = {(c["doc_name"], c["chunk_index"]) for c in fused}
    assert ("DOC", 0) in keys
    assert ("DOC", 100) in keys


def test_fusion_ranks_items_found_by_both_lists_higher():
    dense = [_fake_chunk("DOC", 1), _fake_chunk("DOC", 2)]
    sparse = [_fake_chunk("DOC", 2), _fake_chunk("DOC", 3)]

    fused = _reciprocal_rank_fusion(dense, sparse)

    assert fused[0]["chunk_index"] == 2


def test_candidate_pool_widens_when_company_or_year_filters_the_search():
    """Once company/fiscal-year filtering has already shrunk the search
    space to a handful of filings, a wider reranker pool is nearly free
    (FAISS/BM25 already score everything internally) -- only the
    unfiltered, whole-corpus case needs the smaller default to bound
    reranker cost."""
    assert _candidate_pool_size(company=None, fiscal_year=None) == CANDIDATES_PER_RETRIEVER
    assert _candidate_pool_size(company="3M", fiscal_year=None) == CANDIDATES_WHEN_FILTERED
    assert _candidate_pool_size(company=None, fiscal_year=2018) == CANDIDATES_WHEN_FILTERED
    assert _candidate_pool_size(company="3M", fiscal_year=2018) == CANDIDATES_WHEN_FILTERED
