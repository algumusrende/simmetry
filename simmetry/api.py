"""Core dispatch API: similarity, pairwise, topk, infer_metric."""

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


def _is_string_list(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and (len(x) == 0 or isinstance(x[0], str))


def _is_point_list(x: Any) -> bool:
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and isinstance(x[0], (tuple, list))
        and len(x[0]) == 2
        and all(_is_number(v) for v in x[0])
    )


def _is_point_like(x: Any) -> bool:
    """Return True for tuples of exactly 2 numbers with valid lat/lon ranges.

    Restricted to ``tuple`` (not ``list``) to avoid ambiguity with 2D numeric
    vectors. Values outside [-90, 90] × [-180, 180] fall through to ``cosine``.
    """
    if not (isinstance(x, tuple) and len(x) == 2 and all(_is_number(v) for v in x)):
        return False
    lat, lon = float(x[0]), float(x[1])
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


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
        return "haversine_sim"
    if _is_set_like(a) and _is_set_like(b):
        return "jaccard"
    if _is_vector_like(a) and _is_vector_like(b):
        return "cosine"
    return "cosine"


def infer_metric(a: Any, b: Any) -> str:
    """Return the metric name that ``similarity(..., metric="auto")`` would select.

    Selection order:
    1. ``list[str]`` / ``tuple[str]`` (including empty) → ``jaro_winkler``
    2. ``str`` + ``str`` → ``jaro_winkler``
    3. ``tuple`` of 2 numbers with valid lat/lon range → ``haversine_sim``
    4. ``set`` / ``frozenset`` → ``jaccard``
    5. numeric vectors → ``cosine``
    6. fallback → ``cosine``
    """
    return _auto_metric(a, b)


def similarity(
    a: Any,
    b: Any,
    metric: str | Mapping[str, str] | None = "auto",
    *,
    weights: Mapping[str, float] | None = None,
) -> Any:
    """Compute similarity between two inputs.

    Supports:
    - Scalar values with an explicit or auto-selected metric.
    - String sequences (``list[str]`` / ``tuple[str]``) → pairwise similarity matrix.
    - 2D NumPy arrays → vector pairwise similarity matrix.
    - Composite dict records when ``metric`` is a ``{field: metric_name}`` mapping.

    Args:
        a: First input.
        b: Second input.
        metric: Metric name, ``"auto"`` / ``None`` for automatic selection, or a
            ``{field: metric_name}`` mapping for composite records.
        weights: ``{field: weight}`` mapping used with composite record metrics.

    Returns:
        A float similarity score, or an ndarray for batch inputs.
    """
    if metric is None or (isinstance(metric, str) and metric.lower().strip() == "auto"):
        metric = infer_metric(a, b)

    if isinstance(metric, Mapping) and isinstance(a, Mapping) and isinstance(b, Mapping):
        missing = [f for f in metric if f not in a or f not in b]
        if missing:
            raise KeyError(
                f"Fields {missing} appear in the metric mapping but are absent from one or "
                f"both records. Record keys: a={sorted(a.keys())}, b={sorted(b.keys())}."
            )
        total_w = 0.0
        total = 0.0
        for field, mname in metric.items():
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


def pairwise(X, Y=None, metric: str = "cosine") -> np.ndarray:
    """Return a pairwise similarity matrix.

    Dispatches automatically based on input type:

    - ``list[str]`` / ``tuple[str]`` → :func:`~simmetry.strings.pairwise_strings`
    - list of 2-element numeric tuples/lists → :func:`~simmetry.points.pairwise_points`
    - NumPy arrays / numeric sequences → vector pairwise

    Args:
        X: Input data — strings, points, or vectors of shape (m, d).
        Y: Second set of inputs. Defaults to ``X`` (self-similarity).
        metric: Metric name appropriate for the input type.

    Returns:
        ndarray of shape (m, n).
    """
    if _is_string_list(X):
        from .strings.pairwise import pairwise_strings
        return pairwise_strings(X, Y, metric=metric)

    if _is_point_list(X):
        from .points.pairwise import pairwise_points
        return pairwise_points(X, Y, metric=metric)

    from .vectors.pairwise import pairwise_numpy
    return pairwise_numpy(X, Y, metric=metric)


def topk(
    query,
    X,
    k: int = 10,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k indices and scores for a query vector over ``X``.

    Results are sorted by score descending (highest similarity first).

    Args:
        query: 1D query vector of shape (d,).
        X: Corpus array of shape (n, d).
        k: Number of results to return.
        metric: Vector metric name.

    Returns:
        ``(indices, scores)`` both of length k, sorted descending.
    """
    S = pairwise(np.asarray(query), X, metric=metric).reshape(-1)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1.")
    k = min(k, S.shape[0])
    idx = np.argpartition(-S, kth=k - 1)[:k]
    idx = idx[np.argsort(-S[idx])]
    return idx, S[idx]
