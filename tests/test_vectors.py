import numpy as np
from simfast import pairwise, similarity


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
