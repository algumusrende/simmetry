import numpy as np
from simfast import available, similarity


def test_available_has_some_metrics():
    metrics = available()
    assert "cosine" in metrics
    assert "levenshtein" in metrics
    assert "haversine_km" in metrics
    assert "jaccard" in metrics


def test_similarity_vectors_cosine():
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    c = np.array([0, 1, 0])
    assert similarity(a, b, "cosine") == 1.0
    assert similarity(a, c, "cosine") == 0.0
