from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .registry import get


def _is_string(x: Any) -> bool:
    return isinstance(x, str)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.number))


def _is_vector_like(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        return x.ndim == 1 and np.issubdtype(x.dtype, np.number)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return all(_is_number(v) for v in x)
    return False


def _is_matrix_like(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        return x.ndim == 2 and np.issubdtype(x.dtype, np.number)
    return False


def _is_point_like(x: Any) -> bool:
    return isinstance(x, (tuple, list)) and len(x) == 2 and all(_is_number(v) for v in x)


def _is_string_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and (len(x) == 0 or all(isinstance(v, str) for v in x))


def _is_set_like(x: Any) -> bool:
    return isinstance(x, (set, frozenset))


def _auto_metric(a: Any, b: Any) -> str:
    if _is_string_seq(a) and _is_string_seq(b):
        return "jaro_winkler"
    if _is_string(a) and _is_string(b):
        return "jaro_winkler"
    if _is_point_like(a) and _is_point_like(b):
        return "haversine_km"
    if _is_set_like(a) and _is_set_like(b):
        return "jaccard"
    if _is_vector_like(a) and _is_vector_like(b):
        return "cosine"
    return "cosine"


def similarity(
    a: Any,
    b: Any,
    metric: str | Mapping[str, str] | None = "auto",
    *,
    weights: Mapping[str, float] | None = None,
) -> Any:
    if metric is None or (isinstance(metric, str) and metric.lower().strip() == "auto"):
        metric = _auto_metric(a, b)

    if isinstance(metric, Mapping) and isinstance(a, Mapping) and isinstance(b, Mapping):
        total_w = 0.0
        total = 0.0
        for field, mname in metric.items():
            if field not in a or field not in b:
                continue
            w = float(weights.get(field, 1.0)) if weights is not None else 1.0
            total += w * float(get(mname).fn(a[field], b[field]))
            total_w += w
        return 0.0 if total_w == 0.0 else float(total / total_w)

    if not isinstance(metric, str):
        raise TypeError("metric must be a string name, a field->metric mapping, or 'auto'/None.")

    metric = metric.lower().strip()

    if _is_string_seq(a) and _is_string_seq(b):
        from .strings.pairwise import pairwise_strings
        return pairwise_strings(a, b, metric=metric)

    if _is_matrix_like(a) and _is_matrix_like(b):
        return pairwise(a, b, metric=metric)

    m = get(metric)
    return float(m.fn(a, b))


def pairwise(X, Y=None, metric: str = "cosine"):
    from .vectors.pairwise import pairwise_numpy

    return pairwise_numpy(X, Y, metric=metric)


def topk(query, X, k: int = 10, metric: str = "cosine") -> tuple[np.ndarray, np.ndarray]:
    S = pairwise(np.asarray(query), X, metric=metric).reshape(-1)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, S.shape[0])
    idx = np.argpartition(-S, kth=k - 1)[:k]
    idx = idx[np.argsort(-S[idx])]
    return idx, S[idx]
