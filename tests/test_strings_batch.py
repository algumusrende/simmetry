import numpy as np

from simfast.strings import pairwise_strings, topk_strings


def test_pairwise_strings_shape():
    A = ["abc", "abd", ""]
    S = pairwise_strings(A, metric="levenshtein")
    assert S.shape == (3, 3)
    assert np.allclose(np.diag(S), 1.0)


def test_topk_strings_returns_sorted():
    corpus = ["samplecorp", "examplefinance", "testgroup", "demoorg"]
    idx, scores = topk_strings("samplecorp", corpus, k=3, metric="levenshtein")
    assert len(idx) == 3
    assert scores[0] >= scores[1] >= scores[2]
