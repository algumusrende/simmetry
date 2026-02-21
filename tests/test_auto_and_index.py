import numpy as np
from simfast import SimIndex, similarity


def test_auto_string_similarity():
    s = similarity("akbank", "ak bank", metric="auto")
    assert 0.0 <= s <= 1.0
    assert s > 0.7


def test_auto_set_similarity():
    s = similarity({1, 2, 3}, {2, 3, 4}, metric="auto")
    assert s == 2 / 4


def test_auto_point_similarity():
    s = similarity((41.0, 29.0), (41.01, 29.01), metric="auto")
    assert 0.0 < s < 1.0


def test_composite_dict_similarity():
    a = {"name": "Ali Can", "loc": (41.0, 29.0)}
    b = {"name": "Ali Can Gumus", "loc": (41.0, 29.1)}
    score = similarity(
        a,
        b,
        metric={"name": "jaro_winkler", "loc": "haversine_km"},
        weights={"name": 0.7, "loc": 0.3},
    )
    assert 0.0 <= score <= 1.0


def test_simindex_exact():
    X = np.random.randn(1000, 32).astype("float32")
    idx, scores = SimIndex(metric="cosine", backend="exact").add(X).query(X[0], k=5)
    assert len(idx) == 5
    assert scores[0] >= scores[-1]
