from __future__ import annotations

import numpy as np

from ..utils.numpy_utils import as_2d


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def pairwise_numpy(X, Y=None, metric: str = "cosine") -> np.ndarray:
    X = as_2d(X).astype(np.float64, copy=False)
    Y = X if Y is None else as_2d(Y).astype(np.float64, copy=False)

    metric = metric.lower().strip()

    if metric == "cosine":
        Xn = _normalize_rows(X)
        Yn = _normalize_rows(Y)
        return Xn @ Yn.T

    if metric == "dot":
        return X @ Y.T

    if metric == "euclidean_sim":
        diff = X[:, None, :] - Y[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        return 1.0 / (1.0 + d)

    if metric == "manhattan_sim":
        diff = np.abs(X[:, None, :] - Y[None, :, :])
        d = np.sum(diff, axis=2)
        return 1.0 / (1.0 + d)

    if metric == "pearson":
        # center rows, then cosine on centered vectors
        Xc = X - X.mean(axis=1, keepdims=True)
        Yc = Y - Y.mean(axis=1, keepdims=True)
        Xn = _normalize_rows(Xc)
        Yn = _normalize_rows(Yc)
        return Xn @ Yn.T

    raise KeyError(f"pairwise not implemented for metric={metric}")
