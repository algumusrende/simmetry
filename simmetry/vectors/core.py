from __future__ import annotations

import numpy as np


def cosine(a, b) -> float:
    """Cosine similarity in [-1, 1]; returns 0.0 when either vector is zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_distance(a, b) -> float:
    """Cosine distance (1 - cosine similarity); 0.0 for identical directions."""
    return 1.0 - cosine(a, b)


def dot(a, b) -> float:
    """Raw inner product of two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b))


def euclidean_sim(a, b) -> float:
    """Similarity in (0, 1] derived from Euclidean distance; 1.0 = same point."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(1.0 / (1.0 + float(np.linalg.norm(a - b))))


def manhattan_sim(a, b) -> float:
    """Similarity in (0, 1] derived from Manhattan distance; 1.0 = same point."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(1.0 / (1.0 + float(np.sum(np.abs(a - b)))))


def pearson(a, b) -> float:
    """Pearson correlation coefficient in [-1, 1]; 0.0 for constant vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError(f"Vectors must be the same length for pearson, got {a.size} vs {b.size}.")
    if a.size == 0:
        return 0.0
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a0, b0) / denom)


def hamming(a, b) -> float:
    """Normalized Hamming similarity for equal-length sequences; 1.0 = identical.

    Works on any array-like input (numeric vectors, character lists, etc.).
    Raises ``ValueError`` when inputs have different shapes.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(
            f"Inputs must have the same shape for hamming, got {a.shape} vs {b.shape}."
        )
    if a.size == 0:
        return 1.0
    return float(1.0 - np.mean(a != b))
