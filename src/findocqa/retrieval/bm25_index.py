import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from findocqa.config import DATA_DIR
from findocqa.retrieval.chunking import Chunk
from findocqa.retrieval.company_filter import doc_fiscal_year

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_path(variant: str) -> Path:
    return DATA_DIR / "processed" / variant / "bm25.pkl"


def build_bm25_index(chunks: list[Chunk], variant: str) -> None:
    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    path = _bm25_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": [c.__dict__ for c in chunks]}, f)


def search_bm25(
    query: str,
    top_k: int,
    variant: str,
    company: str | None = None,
    fiscal_year: int | None = None,
) -> list[dict]:
    path = _bm25_path(variant)
    if not path.exists():
        raise FileNotFoundError(
            f"No BM25 index at {path} — run scripts/build_index.py --variant {variant} first."
        )
    with path.open("rb") as f:
        data = pickle.load(f)

    bm25: BM25Okapi = data["bm25"]
    chunks: list[dict] = data["chunks"]
    scores = bm25.get_scores(_tokenize(query))

    candidate_idx = range(len(chunks))
    if company:
        candidate_idx = [i for i in candidate_idx if chunks[i]["company"].lower() == company.lower()]
    if fiscal_year:
        candidate_idx = [i for i in candidate_idx if doc_fiscal_year(chunks[i]["doc_name"]) == fiscal_year]

    ranked = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in ranked]
