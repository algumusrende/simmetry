from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .api import pairwise
from .utils.numpy_utils import as_2d


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


@dataclass
class SimIndex:
    metric: str = "cosine"
    backend: Literal["exact", "hnsw", "faiss"] = "exact"
    X: np.ndarray | None = None
    _ann: Any = None

    def add(self, X) -> SimIndex:
        X = as_2d(X).astype(np.float32, copy=False)
        self.X = X
        if self.backend == "exact":
            return self

        if self.backend == "hnsw":
            from .ann.hnsw import build_hnsw
            space = "cosine" if self.metric == "cosine" else ("l2" if self.metric in {"euclidean_sim"} else "ip")
            self._ann = build_hnsw(X, space=space)
            return self

        if self.backend == "faiss":
            from .ann.faiss_ import build_faiss
            if self.metric == "cosine":
                Xn = _normalize_rows(X.astype(np.float32))
                self.X = Xn
                self._ann = build_faiss(Xn, metric="ip")
            elif self.metric == "dot":
                self._ann = build_faiss(X, metric="ip")
            elif self.metric == "euclidean_sim":
                self._ann = build_faiss(X, metric="l2")
            else:
                raise ValueError(f"faiss backend supports metric in {{cosine,dot,euclidean_sim}}, got {self.metric}")
            return self

        raise ValueError(f"Unknown backend: {self.backend}")

    def query(self, q, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None:
            raise ValueError("Index is empty. Call add(X) first.")
        qv = np.asarray(q, dtype=np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        elif qv.ndim != 2 or qv.shape[0] != 1:
            raise ValueError("q must be a 1D vector or a 2D array with shape (1, dim).")
        if qv.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Query dimension mismatch: q has {qv.shape[1]} features but index has {self.X.shape[1]}."
            )
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")
        k = min(k, self.X.shape[0])

        if self.backend == "exact":
            S = pairwise(qv, self.X, metric=self.metric).reshape(-1)
            idx = np.argpartition(-S, kth=k - 1)[:k]
            idx = idx[np.argsort(-S[idx])]
            return idx, S[idx]

        labels, distances = self._ann.query(qv, k=k)
        return labels.astype(np.int64), distances.astype(np.float32)
