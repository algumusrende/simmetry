import numpy as np
import pytest

from simmetry import available, similarity


def test_available_has_some_metrics():
    metrics = available()
    assert "cosine" in metrics
    assert "levenshtein" in metrics
    assert "haversine_sim" in metrics
    assert "jaccard" in metrics
    assert "hamming" in metrics
    assert "tversky" in metrics
    assert "cosine_distance" in metrics


def test_haversine_km_not_in_registry():
    assert "haversine_km" not in available()


def test_available_by_kind():
    assert "cosine" in available("vector")
    assert "levenshtein" in available("string")
    assert "haversine_sim" in available("point")
    assert "jaccard" in available("set")


def test_similarity_vectors_cosine():
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    c = np.array([0, 1, 0])
    assert similarity(a, b, "cosine") == pytest.approx(1.0)
    assert similarity(a, c, "cosine") == pytest.approx(0.0)


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        similarity("a", "b", "nonexistent_metric_xyz")
