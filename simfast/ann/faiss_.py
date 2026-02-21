from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _require_faiss():
    try:
        import faiss
    except Exception as e:
        raise ImportError(
            'faiss is not installed. Install with: pip install "simfast[ann-faiss]"'
        ) from e
    return faiss


@dataclass
class FaissIndex:
    dim: int
    metric: Literal["l2", "ip"]
    index: object
    n_items: int

    def query(self, q, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        distances, labels = self.index.search(q, k)
        return labels[0], distances[0]


def build_faiss(X, metric: Literal["l2", "ip"] = "ip") -> FaissIndex:
    faiss = _require_faiss()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n, dim).")
    n, dim = X.shape

    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
    elif metric == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError("metric must be 'l2' or 'ip'.")

    index.add(X)
    return FaissIndex(dim=dim, metric=metric, index=index, n_items=n)
