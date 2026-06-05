import numpy as np
import pytest

from simmetry import SimIndex, infer_metric, similarity


def test_auto_string_similarity():
    s = similarity("samplecorp", "sample corp", metric="auto")
    assert 0.0 <= s <= 1.0
    assert s > 0.7


def test_infer_metric_examples():
    assert infer_metric("foo", "bar") == "jaro_winkler"
    assert infer_metric([], []) == "jaro_winkler"
    assert infer_metric((41.0, 29.0), (41.01, 29.01)) == "haversine_sim"
    assert infer_metric({1, 2}, {2, 3}) == "jaccard"


def test_infer_metric_list_not_point():
    assert infer_metric([1.0, 2.0], [3.0, 4.0]) == "cosine"


def test_auto_empty_string_batch():
    out = similarity([], [], metric="auto")
    assert out.shape == (0, 0)


def test_auto_set_similarity():
    s = similarity({1, 2, 3}, {2, 3, 4}, metric="auto")
    assert s == pytest.approx(2 / 4)


def test_auto_point_similarity():
    s = similarity((41.0, 29.0), (41.01, 29.01), metric="auto")
    assert 0.0 < s <= 1.0


def test_auto_point_returns_similarity_not_km():
    s = similarity((41.0, 29.0), (41.01, 29.01), metric="auto")
    assert s <= 1.0


def test_composite_dict_similarity():
    a = {"name": "Entity One", "loc": (41.0, 29.0)}
    b = {"name": "Entity One Extended", "loc": (41.0, 29.1)}
    score = similarity(
        a,
        b,
        metric={"name": "jaro_winkler", "loc": "haversine_sim"},
        weights={"name": 0.7, "loc": 0.3},
    )
    assert 0.0 <= score <= 1.0


def test_composite_dict_missing_field_raises():
    a = {"name": "Alice"}
    b = {"name": "Bob"}
    with pytest.raises(KeyError, match="absent from one or both records"):
        similarity(a, b, metric={"name": "jaro_winkler", "city": "jaro_winkler"})


def test_simindex_exact():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 32)).astype("float32")
    idx, scores = SimIndex(metric="cosine", backend="exact").add(X).query(X[0], k=5)
    assert len(idx) == 5
    assert scores[0] >= scores[-1]


def test_simindex_exact_self_is_first():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 16)).astype("float32")
    idx, scores = SimIndex(metric="cosine", backend="exact").add(X).query(X[42], k=1)
    assert int(idx[0]) == 42
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_simindex_query_dim_mismatch_message():
    X = np.random.randn(20, 8).astype("float32")
    index = SimIndex(metric="cosine", backend="exact").add(X)
    with pytest.raises(ValueError, match="Query dimension mismatch"):
        index.query(np.random.randn(7).astype("float32"), k=3)


def test_pairwise_dispatches_strings():
    from simmetry import pairwise
    S = pairwise(["cat", "car", "bar"], metric="levenshtein")
    assert S.shape == (3, 3)
    assert np.allclose(np.diag(S), 1.0)


def test_pairwise_dispatches_points():
    from simmetry import pairwise
    pts = [(41.0, 29.0), (41.1, 29.1), (40.9, 28.9)]
    S = pairwise(pts, metric="haversine_sim")
    assert S.shape == (3, 3)
    assert np.allclose(np.diag(S), 1.0)


def test_pairwise_dispatches_cross_strings():
    from simmetry import pairwise
    S = pairwise(["cat", "car"], ["bar", "bat", "car"], metric="levenshtein")
    assert S.shape == (2, 3)


def test_register_and_use_custom_metric():
    from simmetry import available, register, similarity
    name = "_test_exact_match"
    if name not in available():
        register(name, lambda a, b: 1.0 if a == b else 0.0, kind="generic")
    assert similarity("foo", "foo", name) == pytest.approx(1.0)
    assert similarity("foo", "bar", name) == pytest.approx(0.0)
    assert name in available()


def test_simindex_hnsw_top_result():
    hnswlib = pytest.importorskip("hnswlib")  # noqa: F841
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 16)).astype("float32")
    idx, scores = SimIndex(metric="cosine", backend="hnsw").add(X).query(X[0], k=5)
    assert int(idx[0]) == 0
    assert scores[0] >= scores[-1]


def test_simindex_faiss_top_result():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 16)).astype("float32")
    idx, scores = SimIndex(metric="cosine", backend="faiss").add(X).query(X[0], k=5)
    assert int(idx[0]) == 0
    assert scores[0] >= scores[-1]
