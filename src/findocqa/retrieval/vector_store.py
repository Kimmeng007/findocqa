import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from findocqa.config import DATA_DIR
from findocqa.retrieval.chunking import Chunk
from findocqa.retrieval.company_filter import doc_fiscal_year
from findocqa.retrieval.embeddings import embed_texts


def _variant_dir(variant: str) -> Path:
    return DATA_DIR / "processed" / variant


def _index_path(variant: str) -> Path:
    return _variant_dir(variant) / "faiss" / "index.faiss"


def _metadata_path(variant: str) -> Path:
    return _variant_dir(variant) / "faiss" / "chunks.jsonl"


def index_files(variant: str) -> tuple[Path, Path]:
    """(index.faiss path, chunks.jsonl path) -- public accessor so
    index_cache can sync these files to/from MongoDB."""
    return _index_path(variant), _metadata_path(variant)


def index_exists(variant: str) -> bool:
    """Lets a caller (e.g. the Streamlit UI) check before searching,
    since a fresh deploy has MongoDB data but no local index files --
    those live under gitignored data/processed/ and must be built once
    per container rather than assumed to already be there."""
    return _index_path(variant).exists()


def build_index(chunks: list[Chunk], variant: str) -> None:
    index_path = _index_path(variant)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = embed_texts([c.text for c in chunks]).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    with _metadata_path(variant).open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")


def load_metadata(variant: str) -> list[dict]:
    with _metadata_path(variant).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def search(
    query: str,
    top_k: int = 5,
    variant: str = "table_aware",
    company: str | None = None,
    fiscal_year: int | None = None,
) -> list[dict]:
    index_path = _index_path(variant)
    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index at {index_path} — run "
            f"scripts/build_index.py --variant {variant} first."
        )
    index = faiss.read_index(str(index_path))
    metadata = load_metadata(variant)

    query_vec = embed_texts([query]).astype("float32")
    # When a company/year is known, score against the whole index (cheap —
    # an in-memory dot product) but only keep matching results, so a
    # same-topic chunk from the wrong company or wrong fiscal year can
    # never crowd out the right one before disambiguation gets a chance.
    search_k = len(metadata) if (company or fiscal_year) else top_k
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        if company and meta["company"].lower() != company.lower():
            continue
        if fiscal_year and doc_fiscal_year(meta["doc_name"]) != fiscal_year:
            continue
        results.append({**meta, "score": float(score)})
        if len(results) >= top_k:
            break
    return results
