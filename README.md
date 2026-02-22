# simmetry

Similarity scores for **strings**, **vectors**, **points**, and **sets** with a small, NumPy-first API.

[PyPI (simmetry)](https://pypi.org/project/simmetry/)

## Install

```bash
pip install simmetry
pip install "simmetry[fast]"
```

- `simmetry[fast]`: enables optional Numba acceleration for `pairwise(..., metric="euclidean_sim")` and `pairwise(..., metric="manhattan_sim")`
- ANN extras:
  - `pip install "simmetry[ann-hnsw]"`
  - `pip install "simmetry[ann-faiss]"`

## Project Status

- Current package: [`simmetry` on PyPI](https://pypi.org/project/simmetry/)
- Current version in this repo: `1.0.1`
- Maturity: **Alpha** (API may change; pin exact/minor versions in production)
- Versioning: semantic versioning target, but pre-hardening changes may still occur in minor releases until `1.x` stabilizes

## Quickstart

### One function

```python
from simmetry import similarity

similarity("kitten", "sitting", metric="levenshtein")
similarity([1, 2, 3], [1, 2, 4], metric="cosine")
similarity((41.1, 29.0), (41.2, 29.1), metric="haversine_km")
similarity({1, 2, 3}, {2, 3, 4}, metric="jaccard")
```

### Pairwise matrices (vectors)

```python
import numpy as np
from simmetry import pairwise

X = np.random.randn(1000, 128)
S = pairwise(X, metric="cosine")
```

### Top-k search (exact)

```python
import numpy as np
from simmetry import topk

X = np.random.randn(5000, 64)
q = np.random.randn(64)
idx, scores = topk(q, X, k=10, metric="cosine")
```

## Available Metrics

```python
from simmetry import available

available()
available("vector")
available("string")
available("point")
available("set")
```

### Vectors
- `cosine`, `dot`, `euclidean_sim`, `manhattan_sim`, `pearson`

### Strings
- `levenshtein` (normalized similarity)
- `jaro_winkler`
- `ngram_jaccard` (character n-gram set Jaccard)
- `token_jaccard` (whitespace token set Jaccard)

### Points / Geo
- `euclidean_2d`
- `haversine_km`

### Sets
- `jaccard`, `dice`, `overlap`

## Auto Metric Selection (Deterministic)

Auto mode is not random and not learned. It applies fixed type-based rules.

```python
from simmetry import infer_metric, similarity

infer_metric("samplecorp", "sample corp")     # "jaro_winkler"
infer_metric((41.0, 29.0), (41.1, 29.1))      # "haversine_km"
infer_metric({1, 2, 3}, {2, 3, 4})            # "jaccard"

similarity("samplecorp", "sample corp")       # uses inferred metric
```

Selection order:

1. `list[str]` / `tuple[str]` (including empty lists) -> batch strings (`jaro_winkler`)
2. `str` + `str` -> `jaro_winkler`
3. 2-number tuples/lists -> `haversine_km`
4. `set` / `frozenset` -> `jaccard`
5. numeric vectors -> `cosine`
6. fallback -> `cosine`

## Batch String APIs

```python
from simmetry.strings import pairwise_strings, topk_strings

S = pairwise_strings(
    ["item_one", "item_two"],
    ["item_one", "item_alt"],
    metric="jaro_winkler",
)
idx, scores = topk_strings(
    "samplecorp",
    ["samplecorp", "examplefinance", "testgroup"],
    k=2,
    metric="levenshtein",
)
```

## ANN Top-k (Optional)

For very large vector corpora (100k+), exact `topk()` can be slow.

### hnswlib

```python
import numpy as np
from simmetry.ann import build_hnsw

X = np.random.randn(200_000, 128).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)

index = build_hnsw(X, space="cosine")
labels, distances = index.query(X[0], k=10)
```

### faiss

```python
import numpy as np
from simmetry.ann import build_faiss

X = np.random.randn(200_000, 128).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)

index = build_faiss(X, metric="ip")
labels, scores = index.query(X[0], k=10)
```

## `SimIndex` (Exact or ANN)

```python
import numpy as np
from simmetry import SimIndex

X = np.random.randn(50_000, 128).astype("float32")
index = SimIndex(metric="cosine", backend="exact").add(X)
idx, scores = index.query(X[0], k=10)
```

## Composite Records

```python
from simmetry import similarity

a = {"name": "Entity One", "city": "CityAlpha", "loc": (41.0, 29.0)}
b = {"name": "Entity One Extended", "city": "CityAlpha", "loc": (41.01, 28.99)}

score = similarity(
    a,
    b,
    metric={"name": "jaro_winkler", "loc": "haversine_km"},
    weights={"name": 0.7, "loc": 0.3},
)
```

## Benchmarks

The project includes a benchmark harness in [`bench/run.py`](./bench/run.py). Comparative benchmarks against `rapidfuzz`, `scikit-learn`, and ANN libraries are not published yet.

Run locally:

```bash
python bench/run.py
```

## Scope and Roadmap

Current focus is a compact core with predictable APIs and optional ANN.

Planned additions (not implemented yet):

- String metrics: Hamming, BM25-style text ranking helpers, string-level Sorensen-Dice variants
- Point APIs: batch pairwise/top-k utilities for geo points
- Published comparative benchmarks (RapidFuzz / sklearn / faiss baselines)
- Hosted docs site

## License

MIT
