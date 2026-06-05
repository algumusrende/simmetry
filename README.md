# simmetry

Similarity scores for **strings**, **vectors**, **points**, and **sets** with a small, NumPy-first API.

[PyPI](https://pypi.org/project/simmetry/) · [GitHub](https://github.com/algumusrende/simmetry) · [Docs](https://algumusrende.github.io/simmetry/) · [Changelog](./CHANGELOG.md)

## Install

```bash
pip install simmetry
pip install "simmetry[fast]"      # Numba acceleration for euclidean_sim / manhattan_sim
pip install "simmetry[ann-hnsw]"  # hnswlib ANN backend
pip install "simmetry[ann-faiss]" # FAISS ANN backend
pip install "simmetry[ann]"       # both ANN backends
```

## Project Status

[![PyPI](https://img.shields.io/pypi/v/simmetry)](https://pypi.org/project/simmetry/)

- Maturity: **Beta** (API stabilising; pin to minor versions in production)
- Versioning: semantic versioning; breaking changes bump the minor version until `2.0`

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

# Vectors
X = np.random.randn(1000, 128)
S = pairwise(X, metric="cosine")          # (1000, 1000)
D = pairwise(X, metric="cosine_distance") # 1 - cosine, sklearn-compatible

# Strings
S = pairwise(["cat", "car", "bar"], metric="levenshtein")  # (3, 3)

# Points
pts = [(41.0, 29.0), (41.1, 29.1), (40.9, 28.9)]
S = pairwise(pts, metric="haversine_sim")  # (3, 3)
```

### Top-k search (exact)

```python
import numpy as np
from simmetry import topk

X = np.random.randn(5000, 64)
q = np.random.randn(64)
idx, scores = topk(q, X, k=10, metric="cosine")
# idx and scores are sorted descending (highest similarity first)
```

## Available Metrics

```python
from simmetry import available

available()           # all registered metrics
available("vector")
available("string")
available("point")
available("set")
```

### Vectors

| Metric | Returns |
|--------|---------|
| `cosine` | [-1, 1] |
| `cosine_distance` | [0, 2] · `1 - cosine` |
| `dot` | unbounded inner product |
| `euclidean_sim` | (0, 1] · `1 / (1 + dist)` |
| `manhattan_sim` | (0, 1] · `1 / (1 + dist)` |
| `pearson` | [-1, 1] |
| `hamming` | [0, 1] · normalized for equal-length sequences |

### Strings

| Metric | Notes |
|--------|-------|
| `levenshtein` | normalized edit distance |
| `jaro_winkler` | prefix-weighted character matching |
| `ngram_jaccard` | character trigram Jaccard (default n=3) |
| `token_jaccard` | whitespace-token Jaccard |
| `hamming_str` | normalized Hamming for equal-length strings |
| `bm25` | BM25 relevance score normalized to [0, 1] |

### Points / Geo

| Metric | Returns |
|--------|---------|
| `euclidean_2d` | (0, 1] · 2D Cartesian similarity |
| `haversine_sim` | [0, 1] · geographic similarity (antipodal ≈ 0) |

`haversine_km` is available as a **utility function** (not a registered metric) that returns raw kilometers:

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
Call it directly with custom weights: `tversky(A, B, alpha=0.3, beta=0.7)`.

## Auto Metric Selection

Auto mode applies fixed type-based rules — no learning, no randomness.

```python
from simmetry import infer_metric, similarity

infer_metric("samplecorp", "sample corp")   # "jaro_winkler"
infer_metric((41.0, 29.0), (41.1, 29.1))   # "haversine_sim"
infer_metric({1, 2, 3}, {2, 3, 4})         # "jaccard"

similarity("samplecorp", "sample corp")     # uses inferred metric
```

Selection order:

1. `list[str]` / `tuple[str]` (including empty) → `jaro_winkler`
2. `str` + `str` → `jaro_winkler`
3. `tuple` of 2 numbers with valid lat/lon range ([-90, 90] × [-180, 180]) → `haversine_sim`
4. `set` / `frozenset` → `jaccard`
5. numeric vectors → `cosine`
6. fallback → `cosine`

> **Note:** Only `tuple` inputs (not `list`) trigger the geo heuristic to avoid ambiguity
> with 2D numeric vectors. `[1.0, 2.0]` routes to `cosine`; `(1.0, 2.0)` routes to `haversine_sim`.
> Pass `metric="haversine_sim"` explicitly when in doubt.

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

## Batch Point APIs (Geo / 2D)

```python
from simmetry.points import pairwise_points, topk_points

pts = [(41.0, 29.0), (41.01, 29.01), (40.9, 28.9)]
S = pairwise_points(pts, metric="haversine_sim")              # similarity matrix
idx, scores = topk_points((41.0, 29.0), pts, k=2, metric="haversine_sim")
```

`haversine_km` is also accepted by `pairwise_points` and `topk_points` for raw-distance
use cases. `topk_points(..., metric="haversine_km")` ranks by ascending distance (nearest first).

## ANN Top-k (Optional)

For large vector corpora (100k+), use approximate nearest neighbour backends.

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

`SimIndex` provides a unified interface across all backends.
`query()` always returns `(indices, similarities)` — ANN distances are converted
to similarities internally so results are directly comparable across backends.

```python
import numpy as np
from simmetry import SimIndex

X = np.random.randn(50_000, 128).astype("float32")
index = SimIndex(metric="cosine", backend="exact").add(X)
idx, scores = index.query(X[0], k=10)
# scores are cosine similarities, sorted descending
```

Switch backends without changing the query code:

```python
index_hnsw  = SimIndex(metric="cosine", backend="hnsw").add(X)
index_faiss = SimIndex(metric="cosine", backend="faiss").add(X)
```

## Composite Records

Weight multiple fields with different metrics:

```python
from simmetry import similarity

a = {"name": "Entity One", "city": "CityAlpha", "loc": (41.0, 29.0)}
b = {"name": "Entity One Extended", "city": "CityAlpha", "loc": (41.01, 28.99)}

score = similarity(
    a,
    b,
    metric={"name": "jaro_winkler", "loc": "haversine_sim"},
    weights={"name": 0.7, "loc": 0.3},
)
```

Fields missing from either record raise `KeyError` with a descriptive message.

## Custom Metrics

```python
from simmetry import register, similarity

def my_metric(a, b):
    return 1.0 if a == b else 0.0

register("exact_match", my_metric, kind="generic")
similarity("foo", "foo", metric="exact_match")  # 1.0
```

## Scope and Roadmap

Planned additions:

- `pairwise()` cross-type dispatch for composite inputs
- BM25 corpus-level ranking (multi-document IDF)

## License

MIT
