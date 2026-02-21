from __future__ import annotations


def _ngrams(s: str, n: int) -> set[str]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    A = _ngrams(a, n)
    B = _ngrams(b, n)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union


def token_jaccard(a: str, b: str) -> float:
    A = set(a.split())
    B = set(b.split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union
