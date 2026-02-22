from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .jaro import jaro_winkler
from .levenshtein import levenshtein
from .ngrams import ngram_jaccard, token_jaccard

_STRING_METRICS: dict[str, Callable[[str, str], float]] = {
    "levenshtein": levenshtein,
    "jaro_winkler": jaro_winkler,
    "ngram_jaccard": ngram_jaccard,
    "token_jaccard": token_jaccard,
}


def pairwise_strings(
    A: Sequence[str],
    B: Sequence[str] | None = None,
    metric: str = "levenshtein",
) -> np.ndarray:
    """Return a pairwise string similarity matrix for the selected metric."""
    metric = metric.lower().strip()
    if metric not in _STRING_METRICS:
        raise KeyError(f"Unknown string metric for pairwise_strings: {metric}")

    fn = _STRING_METRICS[metric]
    if B is None:
        B = A

    m = len(A)
    n = len(B)
    out = np.empty((m, n), dtype=np.float64)

    for i in range(m):
        ai = A[i]
        for j in range(n):
            out[i, j] = fn(ai, B[j])
    return out


def topk_strings(
    query: str,
    corpus: Sequence[str],
    k: int = 10,
    metric: str = "levenshtein",
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k string matches by scoring against the full corpus."""
    S = pairwise_strings([query], corpus, metric=metric).reshape(-1)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, S.shape[0])
    idx = np.argpartition(-S, kth=k - 1)[:k]
    idx = idx[np.argsort(-S[idx])]
    return idx, S[idx]
