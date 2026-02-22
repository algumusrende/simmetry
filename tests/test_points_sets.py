import numpy as np

from simmetry import similarity
from simmetry.points import pairwise_points, topk_points


def test_haversine_similarity_range():
    a = (41.0082, 28.9784)
    b = (41.015, 28.95)
    s = similarity(a, b, "haversine_km")
    assert 0.0 < s < 1.0


def test_set_jaccard_dice_overlap():
    A = {1, 2, 3}
    B = {2, 3, 4}
    assert similarity(A, B, "jaccard") == 2 / 4
    assert similarity(A, B, "overlap") == 2 / 3
    assert similarity(A, B, "dice") == (2 * 2) / (3 + 3)


def test_pairwise_points_shape_and_diagonal():
    pts = [(41.0, 29.0), (41.01, 29.01), (40.9, 28.9)]
    S = pairwise_points(pts, metric="haversine_km")
    assert S.shape == (3, 3)
    assert np.allclose(np.diag(S), 1.0)


def test_topk_points_returns_sorted():
    pts = [(41.0, 29.0), (41.001, 29.001), (42.0, 30.0), (39.0, 27.0)]
    idx, scores = topk_points((41.0, 29.0), pts, k=3, metric="haversine_km")
    assert len(idx) == 3
    assert scores[0] >= scores[1] >= scores[2]
    assert int(idx[0]) == 0
