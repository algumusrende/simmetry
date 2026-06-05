import numpy as np
import pytest

from simmetry import similarity
from simmetry.points import haversine_km, haversine_sim, pairwise_points, topk_points
from simmetry.sets import tversky


def test_haversine_km_utility():
    a = (0.0, 0.0)
    b = (1.0, 0.0)
    d = haversine_km(a, b)
    assert np.isclose(d, 111.1950802335329, atol=1e-6)


def test_haversine_sim_range():
    assert similarity((41.0, 29.0), (41.0, 29.0), "haversine_sim") == pytest.approx(1.0)
    s = similarity((0.0, 0.0), (1.0, 0.0), "haversine_sim")
    assert 0.0 < s < 1.0


def test_haversine_sim_antipodal_near_zero():
    s = haversine_sim((90.0, 0.0), (-90.0, 0.0))
    assert s == pytest.approx(0.0, abs=1e-6)


def test_haversine_km_not_registered():
    with pytest.raises(KeyError):
        similarity((0.0, 0.0), (1.0, 0.0), "haversine_km")


def test_set_jaccard_dice_overlap():
    A = {1, 2, 3}
    B = {2, 3, 4}
    assert similarity(A, B, "jaccard") == pytest.approx(2 / 4)
    assert similarity(A, B, "overlap") == pytest.approx(2 / 3)
    assert similarity(A, B, "dice") == pytest.approx((2 * 2) / (3 + 3))


def test_tversky_equals_jaccard_at_defaults():
    A = {1, 2, 3}
    B = {2, 3, 4}
    assert tversky(A, B) == pytest.approx(similarity(A, B, "jaccard"))


def test_tversky_equals_dice_at_half():
    A = {1, 2, 3}
    B = {2, 3, 4}
    assert tversky(A, B, alpha=0.5, beta=0.5) == pytest.approx(similarity(A, B, "dice"))


def test_tversky_empty_sets():
    assert tversky(set(), set()) == pytest.approx(1.0)
    assert tversky(set(), {1}) == pytest.approx(0.0)


def test_tversky_via_similarity():
    A = {1, 2, 3}
    B = {2, 3, 4}
    assert similarity(A, B, "tversky") == pytest.approx(tversky(A, B))


def test_pairwise_points_haversine_sim_shape_and_diagonal():
    pts = [(41.0, 29.0), (41.01, 29.01), (40.9, 28.9)]
    S = pairwise_points(pts, metric="haversine_sim")
    assert S.shape == (3, 3)
    assert np.allclose(np.diag(S), 1.0)


def test_pairwise_points_haversine_km_diagonal_zero():
    pts = [(41.0, 29.0), (41.01, 29.01), (40.9, 28.9)]
    S = pairwise_points(pts, metric="haversine_km")
    assert S.shape == (3, 3)
    assert np.allclose(np.diag(S), 0.0)


def test_topk_points_haversine_sim_sorted_descending():
    pts = [(41.0, 29.0), (41.001, 29.001), (42.0, 30.0), (39.0, 27.0)]
    idx, scores = topk_points((41.0, 29.0), pts, k=3, metric="haversine_sim")
    assert len(idx) == 3
    assert scores[0] >= scores[1] >= scores[2]
    assert int(idx[0]) == 0


def test_topk_points_haversine_km_sorted_ascending():
    pts = [(41.0, 29.0), (41.001, 29.001), (42.0, 30.0), (39.0, 27.0)]
    idx, scores = topk_points((41.0, 29.0), pts, k=3, metric="haversine_km")
    assert len(idx) == 3
    assert scores[0] <= scores[1] <= scores[2]
    assert int(idx[0]) == 0
