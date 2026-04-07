"""
Vector store — FAISS-backed index with persistence.

Vectors are L2-normalised so inner-product search equals cosine similarity.
The index is rebuilt from scratch when a document is deleted (FAISS doesn't
support in-place deletion for flat indices).
"""

import os
import numpy as np
import faiss
import threading
from app.config import get_settings

settings = get_settings()

_lock = threading.Lock()
_index: faiss.IndexFlatIP | None = None
_dim: int = 1536  # ada-002 dimension


def _index_path() -> str:
    return settings.faiss_index_path + ".index"


def _load_or_create() -> faiss.IndexFlatIP:
    global _index, _dim
    os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
    path = _index_path()
    if os.path.isfile(path):
        idx = faiss.read_index(path)
        _dim = idx.d
        return idx
    return faiss.IndexFlatIP(_dim)


def _get_index() -> faiss.IndexFlatIP:
    global _index
    if _index is None:
        _index = _load_or_create()
    return _index


def _save():
    faiss.write_index(_get_index(), _index_path())


def add_vectors(vectors: list[list[float]]) -> list[int]:
    """Add vectors and return their FAISS integer IDs (sequential)."""
    with _lock:
        idx = _get_index()
        start = idx.ntotal
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        idx.add(arr)
        _save()
        return list(range(start, idx.ntotal))


def search(query_vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
    """Return [(faiss_id, score)] sorted by descending cosine similarity."""
    with _lock:
        idx = _get_index()
        if idx.ntotal == 0:
            return []
        arr = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(arr)
        k = min(top_k, idx.ntotal)
        scores, ids = idx.search(arr, k)
        return [(int(ids[0][i]), float(scores[0][i])) for i in range(k) if ids[0][i] != -1]


def rebuild_index(vectors_by_id: dict[int, list[float]]):
    """
    Rebuild the index keeping only the provided faiss IDs.
    Used after document deletion.
    vectors_by_id: {old_faiss_id: embedding_vector}
    """
    global _index
    with _lock:
        new_index = faiss.IndexFlatIP(_dim)
        if vectors_by_id:
            arr = np.array(list(vectors_by_id.values()), dtype="float32")
            faiss.normalize_L2(arr)
            new_index.add(arr)
        _index = new_index
        _save()
    # Return mapping old_id -> new_id (positional order)
    return {old: new for new, old in enumerate(vectors_by_id.keys())}


def get_vector(faiss_id: int) -> list[float] | None:
    """Retrieve a stored vector by its FAISS sequential ID."""
    with _lock:
        idx = _get_index()
        if faiss_id >= idx.ntotal:
            return None
        vec = idx.reconstruct(faiss_id)
        return vec.tolist()


def total_vectors() -> int:
    with _lock:
        return _get_index().ntotal
