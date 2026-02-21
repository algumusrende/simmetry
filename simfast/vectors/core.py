from __future__ import annotations

import numpy as np


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def dot(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b))


def euclidean_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = float(np.linalg.norm(a - b))
    return float(1.0 / (1.0 + d))


def manhattan_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = float(np.sum(np.abs(a - b)))
    return float(1.0 / (1.0 + d))


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError("Vectors must be same length for pearson.")
    if a.size == 0:
        return 0.0
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a0, b0) / denom)
