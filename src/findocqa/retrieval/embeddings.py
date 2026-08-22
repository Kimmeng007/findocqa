from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from findocqa.config import settings


@lru_cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
