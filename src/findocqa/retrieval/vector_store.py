import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from findocqa.config import DATA_DIR
from findocqa.retrieval.chunking import Chunk
from findocqa.retrieval.embeddings import embed_texts

INDEX_DIR = DATA_DIR / "processed" / "faiss"
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "chunks.jsonl"


def build_index(chunks: list[Chunk]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectors = embed_texts([c.text for c in chunks]).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")


def _load_metadata() -> list[dict]:
    with METADATA_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def search(query: str, top_k: int = 5) -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index at {INDEX_PATH} — run scripts/build_index.py first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    metadata = _load_metadata()

    query_vec = embed_texts([query]).astype("float32")
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**metadata[idx], "score": float(score)})
    return results
