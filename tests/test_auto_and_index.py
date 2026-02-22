import numpy as np

from simmetry import SimIndex, infer_metric, similarity


def test_auto_string_similarity():
    s = similarity("samplecorp", "sample corp", metric="auto")
    assert 0.0 <= s <= 1.0
    assert s > 0.7


def test_infer_metric_examples():
    assert infer_metric("foo", "bar") == "jaro_winkler"
    assert infer_metric([], []) == "jaro_winkler"
    assert infer_metric((41.0, 29.0), (41.01, 29.01)) == "haversine_km"
    assert infer_metric({1, 2}, {2, 3}) == "jaccard"


def test_auto_empty_string_batch():
    out = similarity([], [], metric="auto")
    assert out.shape == (0, 0)


def test_auto_set_similarity():
    s = similarity({1, 2, 3}, {2, 3, 4}, metric="auto")
    assert s == 2 / 4


def test_auto_point_similarity():
    s = similarity((41.0, 29.0), (41.01, 29.01), metric="auto")
    assert 0.0 < s < 1.0


def test_composite_dict_similarity():
    a = {"name": "Entity One", "loc": (41.0, 29.0)}
    b = {"name": "Entity One Extended", "loc": (41.0, 29.1)}
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


def test_simindex_query_dim_mismatch_message():
    X = np.random.randn(20, 8).astype("float32")
    index = SimIndex(metric="cosine", backend="exact").add(X)
    try:
        index.query(np.random.randn(7).astype("float32"), k=3)
    except ValueError as e:
        assert "Query dimension mismatch" in str(e)
    else:
        raise AssertionError("Expected ValueError for dimension mismatch.")
