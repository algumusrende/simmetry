from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .core import euclidean_2d, haversine_km

_POINT_METRICS = {
    "euclidean_2d": euclidean_2d,
    "haversine_km": haversine_km,
}


def _as_points(x) -> list[tuple[float, float]]:
    pts = list(x)
    out: list[tuple[float, float]] = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("Each point must be a 2-item tuple/list: (lat, lon) or (x, y).")
        out.append((float(p[0]), float(p[1])))
    return out


def pairwise_points(
    A: Sequence[tuple[float, float]],
    B: Sequence[tuple[float, float]] | None = None,
    metric: str = "haversine_km",
) -> np.ndarray:
    """Return a pairwise similarity matrix for point inputs."""
    metric = metric.lower().strip()
    if metric not in _POINT_METRICS:
        raise KeyError(f"Unknown point metric for pairwise_points: {metric}")

    fn = _POINT_METRICS[metric]
    PA = _as_points(A)
    PB = PA if B is None else _as_points(B)

    m = len(PA)
    n = len(PB)
    out = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        ai = PA[i]
        for j in range(n):
            out[i, j] = fn(ai, PB[j])
    return out


def topk_points(
    query: tuple[float, float],
    corpus: Sequence[tuple[float, float]],
    k: int = 10,
    metric: str = "haversine_km",
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k point matches by scoring against the full corpus."""
    S = pairwise_points([query], corpus, metric=metric).reshape(-1)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, S.shape[0])
    idx = np.argpartition(-S, kth=k - 1)[:k]
    idx = idx[np.argsort(-S[idx])]
    return idx, S[idx]
