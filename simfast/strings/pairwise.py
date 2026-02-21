from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

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
    B: Optional[Sequence[str]] = None,
    metric: str = "levenshtein",
) -> np.ndarray:
    """Pairwise similarity for strings.

    Parameters
    ----------
    A : sequence[str]
        First list of strings (size m)
    B : sequence[str] | None
        Second list of strings (size n). If None, uses B=A.
    metric : str
        One of: levenshtein, jaro_winkler, ngram_jaccard, token_jaccard

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (m, n)
    """
    metric = metric.lower().strip()
    if metric not in _STRING_METRICS:
        raise KeyError(f"Unknown string metric for pairwise_strings: {metric}")

    fn = _STRING_METRICS[metric]
    if B is None:
        B = A

    m = len(A)
    n = len(B)
    out = np.empty((m, n), dtype=np.float64)

    # Simple tight loops; keeps overhead low (fast enough for thousands of strings).
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k for strings by computing similarities to the entire corpus."""
    S = pairwise_strings([query], corpus, metric=metric).reshape(-1)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, S.shape[0])
    idx = np.argpartition(-S, kth=k - 1)[:k]
    idx = idx[np.argsort(-S[idx])]
    return idx, S[idx]
