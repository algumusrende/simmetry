from __future__ import annotations

import numpy as np

from ..utils.numpy_utils import as_2d

try:
    from numba import njit, prange
except Exception:
    njit = None
    prange = range


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


if njit is not None:
    @njit(cache=True, fastmath=True, parallel=True)
    def _euclidean_sim_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        m, d = X.shape
        n = Y.shape[0]
        out = np.empty((m, n), dtype=np.float64)
        for i in prange(m):
            for j in range(n):
                s = 0.0
                for k in range(d):
                    diff = X[i, k] - Y[j, k]
                    s += diff * diff
                out[i, j] = 1.0 / (1.0 + np.sqrt(s))
        return out


    @njit(cache=True, fastmath=True, parallel=True)
    def _manhattan_sim_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        m, d = X.shape
        n = Y.shape[0]
        out = np.empty((m, n), dtype=np.float64)
        for i in prange(m):
            for j in range(n):
                s = 0.0
                for k in range(d):
                    s += abs(X[i, k] - Y[j, k])
                out[i, j] = 1.0 / (1.0 + s)
        return out


def pairwise_numpy(X, Y=None, metric: str = "cosine") -> np.ndarray:
    X = as_2d(X).astype(np.float64, copy=False)
    Y = X if Y is None else as_2d(Y).astype(np.float64, copy=False)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: X has {X.shape[1]} features but Y has {Y.shape[1]} features."
        )

    metric = metric.lower().strip()

    if metric == "cosine":
        Xn = _normalize_rows(X)
        Yn = _normalize_rows(Y)
        return Xn @ Yn.T

    if metric == "dot":
        return X @ Y.T

    if metric == "euclidean_sim":
        if njit is not None:
            return _euclidean_sim_numba(X, Y)
        diff = X[:, None, :] - Y[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        return 1.0 / (1.0 + d)

    if metric == "manhattan_sim":
        if njit is not None:
            return _manhattan_sim_numba(X, Y)
        diff = np.abs(X[:, None, :] - Y[None, :, :])
        d = np.sum(diff, axis=2)
        return 1.0 / (1.0 + d)

    if metric == "pearson":
        Xc = X - X.mean(axis=1, keepdims=True)
        Yc = Y - Y.mean(axis=1, keepdims=True)
        Xn = _normalize_rows(Xc)
        Yn = _normalize_rows(Yc)
        return Xn @ Yn.T

    raise KeyError(f"pairwise not implemented for metric={metric}")
