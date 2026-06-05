import numpy as np
import pytest

from simmetry import pairwise, similarity, topk
from simmetry.vectors import cosine_distance, hamming


def test_pairwise_cosine_shape():
    X = np.eye(5)
    S = pairwise(X, metric="cosine")
    assert S.shape == (5, 5)
    assert np.allclose(np.diag(S), 1.0)


def test_pairwise_dot():
    X = np.eye(3)
    S = pairwise(X, metric="dot")
    assert np.allclose(S, np.eye(3))


def test_euclidean_sim_monotonic():
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([0.0, 2.0])
    assert similarity(a, b, "euclidean_sim") > similarity(a, c, "euclidean_sim")


def test_pairwise_dim_mismatch_message():
    X = np.random.randn(3, 4)
    Y = np.random.randn(5, 2)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        pairwise(X, Y, metric="cosine")


def test_topk_dim_mismatch_message():
    X = np.random.randn(6, 4)
    q = np.random.randn(3)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        topk(q, X, k=2, metric="cosine")


def test_topk_returns_sorted():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 16))
    q = rng.standard_normal(16)
    idx, scores = topk(q, X, k=10, metric="cosine")
    assert len(idx) == 10
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_cosine_distance_is_complement():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert similarity(a, b, "cosine") + cosine_distance(a, b) == pytest.approx(1.0)
    assert cosine_distance(a, a) == pytest.approx(0.0)


def test_cosine_distance_pairwise():
    X = np.eye(3)
    D = pairwise(X, metric="cosine_distance")
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D[0, 1], 1.0)


def test_hamming_identical():
    assert hamming([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_hamming_all_different():
    assert hamming([0, 0, 0], [1, 1, 1]) == pytest.approx(0.0)


def test_hamming_partial():
    assert hamming([1, 0, 1], [1, 1, 1]) == pytest.approx(2 / 3)


def test_hamming_empty():
    assert hamming([], []) == pytest.approx(1.0)


def test_hamming_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        hamming([1, 2], [1, 2, 3])


def test_hamming_via_similarity():
    a = np.array([1, 0, 1, 1])
    b = np.array([1, 1, 0, 1])
    s = similarity(a, b, "hamming")
    assert s == pytest.approx(2 / 4)


def test_pairwise_hamming_shape():
    X = np.eye(4)
    S = pairwise(X, metric="hamming")
    assert S.shape == (4, 4)
    assert np.allclose(np.diag(S), 1.0)
