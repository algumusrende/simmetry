from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def _to_set(x: Iterable[T]) -> set[T]:
    return x if isinstance(x, set) else set(x)


def jaccard(a: Iterable[T], b: Iterable[T]) -> float:
    """Jaccard similarity: |A∩B| / |A∪B|; 1.0 for identical sets, 0.0 for disjoint."""
    A = _to_set(a)
    B = _to_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def dice(a: Iterable[T], b: Iterable[T]) -> float:
    """Sørensen-Dice similarity: 2|A∩B| / (|A| + |B|); 1.0 for identical sets."""
    A = _to_set(a)
    B = _to_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return (2.0 * len(A & B)) / (len(A) + len(B))


def overlap(a: Iterable[T], b: Iterable[T]) -> float:
    """Overlap coefficient: |A∩B| / min(|A|, |B|); 1.0 when one set contains the other."""
    A = _to_set(a)
    B = _to_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def tversky(a: Iterable[T], b: Iterable[T], alpha: float = 1.0, beta: float = 1.0) -> float:
    """Tversky index: |A∩B| / (|A∩B| + α|A\\B| + β|B\\A|).

    Special cases: alpha=beta=1.0 → Jaccard; alpha=beta=0.5 → Dice.
    Both empty sets return 1.0.
    """
    A = _to_set(a)
    B = _to_set(b)
    if not A and not B:
        return 1.0
    inter = len(A & B)
    denom = inter + alpha * len(A - B) + beta * len(B - A)
    if denom == 0.0:
        return 1.0
    return inter / denom
