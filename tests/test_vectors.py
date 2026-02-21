import numpy as np
import pytest

from simfast import pairwise, similarity, topk


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
