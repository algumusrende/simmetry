from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


def _require_hnswlib():
    try:
        import hnswlib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            'hnswlib is not installed. Install with: pip install "simfast[ann-hnsw]"'
        ) from e
    return hnswlib


@dataclass
class HNSWIndex:
    """Thin wrapper around hnswlib.Index with a stable, friendly API."""

    dim: int
    space: Literal["cosine", "l2", "ip"]
    index: object
    n_items: int

    def query(self, q, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        labels, distances = self.index.knn_query(q, k=k)
        return labels[0], distances[0]


def build_hnsw(
    X,
    space: Literal["cosine", "l2", "ip"] = "cosine",
    ef_construction: int = 200,
    M: int = 16,
    ef: int = 50,
) -> HNSWIndex:
    """Build an HNSW ANN index for vectors.

    Notes:
    - For space="cosine", hnswlib uses cosine distance (smaller is closer).
    - You can keep your core package dependency-free; this is an optional extra.
    """
    hnswlib = _require_hnswlib()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n, dim).")

    n, dim = X.shape
    idx = hnswlib.Index(space=space, dim=dim)
    idx.init_index(max_elements=n, ef_construction=int(ef_construction), M=int(M))
    idx.add_items(X, np.arange(n, dtype=np.int32))
    idx.set_ef(int(ef))
    return HNSWIndex(dim=dim, space=space, index=idx, n_items=n)
