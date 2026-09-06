"""Persists built FAISS/BM25 index files in MongoDB GridFS.

Why this exists: data/processed/ (the built indexes) is gitignored, so
a fresh container -- which is what every Streamlit Cloud redeploy is --
has MongoDB's filing data but no local index files. Rebuilding from
scratch means re-embedding the whole corpus (~40k+ chunks), measured at
7+ minutes even on a dev machine -- not a workable cold start for a
public demo. MongoDB Atlas is already the project's persistence layer
and has ample free-tier headroom, so a built index is uploaded there
once and every later container downloads it instead of recomputing it.
"""

from pathlib import Path

import gridfs

from findocqa.config import settings
from findocqa.retrieval import bm25_index, vector_store
from findocqa.storage.mongo import get_client

_BUCKET = "index_cache"


def _fs() -> gridfs.GridFS:
    return gridfs.GridFS(get_client()[settings.mongodb_db_name], collection=_BUCKET)


def _files_for(variant: str) -> dict[str, Path]:
    index_path, metadata_path = vector_store.index_files(variant)
    return {
        f"{variant}_index.faiss": index_path,
        f"{variant}_chunks.jsonl": metadata_path,
        f"{variant}_bm25.pkl": bm25_index.bm25_file(variant),
    }


def upload_index(variant: str) -> None:
    """Call after building an index locally so future containers can
    download it instead of rebuilding. Replaces any previously-cached
    version for this variant."""
    fs = _fs()
    for name, path in _files_for(variant).items():
        for existing in fs.find({"filename": name}):
            fs.delete(existing._id)
        with path.open("rb") as f:
            fs.put(f, filename=name)


def download_index(variant: str) -> bool:
    """Downloads a previously-cached index into the local file layout
    vector_store/bm25_index read from. Returns True if a cached index
    was found and downloaded, False if nothing is cached for this
    variant yet (caller should build one and upload it)."""
    fs = _fs()
    files = _files_for(variant)
    if not all(fs.exists({"filename": name}) for name in files):
        return False
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        grid_out = fs.find_one({"filename": name})
        with path.open("wb") as f:
            f.write(grid_out.read())
    return True
