from __future__ import annotations

from typing import Iterable, Set, TypeVar

T = TypeVar("T")


def _to_set(x: Iterable[T]) -> Set[T]:
    return x if isinstance(x, set) else set(x)


def jaccard(a: Iterable[T], b: Iterable[T]) -> float:
    A = _to_set(a)
    B = _to_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def dice(a: Iterable[T], b: Iterable[T]) -> float:
    A = _to_set(a)
    B = _to_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return (2.0 * inter) / (len(A) + len(B))


def overlap(a: Iterable[T], b: Iterable[T]) -> float:
    A = _to_set(a)
    B = _to_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return inter / min(len(A), len(B))
