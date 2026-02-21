import importlib

import pytest


def test_ann_imports():
    mod = importlib.import_module("simfast.ann")
    assert mod is not None


def test_hnsw_raises_helpful_error():
    from simfast.ann.hnsw import build_hnsw

    with pytest.raises(ImportError) as e:
        build_hnsw([[0.0, 1.0]])
    assert "simfast[ann-hnsw]" in str(e.value)


def test_faiss_raises_helpful_error():
    from simfast.ann.faiss_ import build_faiss

    with pytest.raises(ImportError) as e:
        build_faiss([[0.0, 1.0]])
    assert "simfast[ann-faiss]" in str(e.value)
