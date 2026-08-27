from functools import lru_cache

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(_MODEL_NAME)


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = get_reranker().predict(pairs)

    ranked = sorted(
        zip(candidates, scores), key=lambda pair: pair[1], reverse=True
    )[:top_k]
    return [{**c, "score": float(s)} for c, s in ranked]
