# simmetry

Similarity scores for **strings**, **vectors**, **points**, and **sets** with a small, NumPy-first API.

## Install

```bash
pip install simmetry
pip install "simmetry[fast]"      # Numba acceleration for euclidean_sim / manhattan_sim
pip install "simmetry[ann-hnsw]"  # hnswlib ANN backend
pip install "simmetry[ann-faiss]" # FAISS ANN backend
pip install "simmetry[ann]"       # both ANN backends
```

## Quickstart

### One function

```python
from simmetry import similarity

similarity("kitten", "sitting", metric="levenshtein")
similarity([1, 2, 3], [1, 2, 4], metric="cosine")
similarity((41.1, 29.0), (41.2, 29.1), metric="haversine_sim")  # returns [0, 1]
similarity({1, 2, 3}, {2, 3, 4}, metric="jaccard")
```

### Pairwise matrices

```python
import numpy as np
from simmetry import pairwise

X = np.random.randn(1000, 128)
S = pairwise(X, metric="cosine")           # (1000, 1000) similarity matrix
D = pairwise(X, metric="cosine_distance")  # 1 - cosine, sklearn-compatible
```

### Top-k search

```python
import numpy as np
from simmetry import topk

X = np.random.randn(5000, 64)
q = np.random.randn(64)
idx, scores = topk(q, X, k=10, metric="cosine")
# sorted descending — highest similarity first
```

## Available Metrics

```python
from simmetry import available

available()           # all metrics
available("vector")
available("string")
available("point")
available("set")
```

### Vectors

| Metric | Range | Notes |
|--------|-------|-------|
| `cosine` | [-1, 1] | |
| `cosine_distance` | [0, 2] | `1 - cosine` |
| `dot` | unbounded | raw inner product |
| `euclidean_sim` | (0, 1] | `1 / (1 + dist)` |
| `manhattan_sim` | (0, 1] | `1 / (1 + dist)` |
| `pearson` | [-1, 1] | correlation coefficient |
| `hamming` | [0, 1] | normalized, equal-length sequences |

### Strings

| Metric | Notes |
|--------|-------|
| `levenshtein` | normalized edit distance |
| `jaro_winkler` | prefix-weighted character matching |
| `ngram_jaccard` | character trigram Jaccard (default n=3) |
| `token_jaccard` | whitespace-token Jaccard |

### Points / Geo

| Metric | Range | Notes |
|--------|-------|-------|
| `euclidean_2d` | (0, 1] | 2D Cartesian similarity |
| `haversine_sim` | [0, 1] | geographic similarity; antipodal ≈ 0 |

`haversine_km` is available as a utility (not a registered metric) that returns raw kilometers:

```python
from simmetry.points import haversine_km

km = haversine_km((40.7, -74.0), (51.5, -0.1))  # ~5 570 km
```

### Sets

| Metric | Formula |
|--------|---------|
| `jaccard` | \|A∩B\| / \|A∪B\| |
| `dice` | 2\|A∩B\| / (\|A\| + \|B\|) |
| `overlap` | \|A∩B\| / min(\|A\|, \|B\|) |
| `tversky` | \|A∩B\| / (\|A∩B\| + α\|A\\B\| + β\|B\\A\|) |

`tversky` with `alpha=beta=1` equals Jaccard; `alpha=beta=0.5` equals Dice.

## Auto Metric Selection

```python
from simmetry import infer_metric, similarity

infer_metric("samplecorp", "sample corp")  # "jaro_winkler"
infer_metric((41.0, 29.0), (41.1, 29.1))  # "haversine_sim"
infer_metric({1, 2, 3}, {2, 3, 4})        # "jaccard"

similarity("samplecorp", "sample corp")    # uses inferred metric
```

Selection order:

1. `list[str]` / `tuple[str]` (including empty) → `jaro_winkler`
2. `str` + `str` → `jaro_winkler`
3. `tuple` of 2 numbers with valid lat/lon range → `haversine_sim`
4. `set` / `frozenset` → `jaccard`
5. numeric vectors → `cosine`
6. fallback → `cosine`

!!! note
    Only `tuple` inputs (not `list`) trigger the geo heuristic to avoid ambiguity
    with 2D numeric vectors. Pass `metric="haversine_sim"` explicitly when in doubt.

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

## Batch Point APIs

```python
from simmetry.points import pairwise_points, topk_points

pts = [(41.0, 29.0), (41.01, 29.01), (40.9, 28.9)]
S = pairwise_points(pts, metric="haversine_sim")
idx, scores = topk_points((41.0, 29.0), pts, k=2, metric="haversine_sim")
```

## ANN Backends

```python
import numpy as np
from simmetry import SimIndex

X = np.random.randn(200_000, 128).astype("float32")

# Exact
index = SimIndex(metric="cosine", backend="exact").add(X)

# HNSW  (pip install "simmetry[ann-hnsw]")
index = SimIndex(metric="cosine", backend="hnsw").add(X)

# FAISS (pip install "simmetry[ann-faiss]")
index = SimIndex(metric="cosine", backend="faiss").add(X)

idx, scores = index.query(X[0], k=10)
# All backends return (indices, similarities) — not raw distances
```

## Composite Records

```python
from simmetry import similarity

a = {"name": "Entity One", "loc": (41.0, 29.0)}
b = {"name": "Entity One Extended", "loc": (41.01, 28.99)}

score = similarity(
    a, b,
    metric={"name": "jaro_winkler", "loc": "haversine_sim"},
    weights={"name": 0.7, "loc": 0.3},
)
```

## Custom Metrics

```python
from simmetry import register, similarity

def my_metric(a, b):
    return 1.0 if a == b else 0.0

register("exact_match", my_metric, kind="generic")
similarity("foo", "foo", metric="exact_match")  # 1.0
```

## License

MIT
