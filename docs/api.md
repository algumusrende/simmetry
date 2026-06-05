# API Reference

## Top-level functions

### `similarity(a, b, metric="auto", *, weights=None)`

Compute similarity between two inputs.

```python
from simmetry import similarity
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `a` | any | — | First input |
| `b` | any | — | Second input |
| `metric` | `str \| dict \| None` | `"auto"` | Metric name, `"auto"`, or `{field: metric}` mapping for composite records |
| `weights` | `dict[str, float] \| None` | `None` | Field weights for composite records |

Returns a `float` for scalar inputs or an `ndarray` for batch inputs.

---

### `pairwise(X, Y=None, metric="cosine")`

Return a pairwise similarity matrix for vector inputs.

```python
from simmetry import pairwise

S = pairwise(X, metric="cosine")          # (m, m) self-similarity
S = pairwise(X, Y, metric="euclidean_sim") # (m, n)
```

---

### `topk(query, X, k=10, metric="cosine")`

Return exact top-k indices and scores for a query vector.
Results are sorted descending (highest similarity first).

```python
from simmetry import topk

idx, scores = topk(q, X, k=10, metric="cosine")
```

---

### `infer_metric(a, b)`

Return the metric name that `similarity(..., metric="auto")` would select.

```python
from simmetry import infer_metric

infer_metric("hello", "world")           # "jaro_winkler"
infer_metric((41.0, 29.0), (42.0, 30.0)) # "haversine_sim"
```

---

### `available(kind=None)`

Return registered metric names, optionally filtered by kind.

```python
from simmetry import available

available()           # all metrics
available("vector")   # "cosine", "dot", ...
available("string")   # "levenshtein", ...
available("point")    # "haversine_sim", ...
available("set")      # "jaccard", ...
```

---

### `register(name, fn, kind="generic")`

Register a custom metric function.

```python
from simmetry import register

register("my_metric", lambda a, b: 1.0 if a == b else 0.0, kind="generic")
```

---

## `SimIndex`

Unified vector similarity index with exact and optional ANN backends.

```python
from simmetry import SimIndex

index = SimIndex(metric="cosine", backend="exact")
index.add(X)
idx, scores = index.query(q, k=10)
```

All backends return `(indices, similarities)` — ANN distances are converted to similarities internally.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `str` | `"cosine"` | Similarity metric |
| `backend` | `"exact" \| "hnsw" \| "faiss"` | `"exact"` | Index backend |

### Methods

#### `.add(X) → SimIndex`

Store vectors and build the index. Chainable.

#### `.query(q, k=10) → (ndarray, ndarray)`

Return `(indices, similarities)` for the k most similar vectors, sorted descending.

---

## Vectors

```python
from simmetry.vectors import cosine, dot, euclidean_sim, manhattan_sim, pearson, hamming, cosine_distance
from simmetry.vectors.pairwise import pairwise_numpy
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `cosine(a, b)` | array-like, array-like | float in [-1, 1] |
| `cosine_distance(a, b)` | array-like, array-like | float in [0, 2] |
| `dot(a, b)` | array-like, array-like | float (unbounded) |
| `euclidean_sim(a, b)` | array-like, array-like | float in (0, 1] |
| `manhattan_sim(a, b)` | array-like, array-like | float in (0, 1] |
| `pearson(a, b)` | array-like, array-like | float in [-1, 1] |
| `hamming(a, b)` | array-like, array-like | float in [0, 1] |

---

## Strings

```python
from simmetry.strings import (
    levenshtein, levenshtein_distance,
    jaro_winkler,
    ngram_jaccard, token_jaccard,
    pairwise_strings, topk_strings,
)
```

| Function | Notes |
|----------|-------|
| `levenshtein(a, b)` | Normalized similarity in [0, 1] |
| `levenshtein_distance(a, b)` | Raw integer edit distance |
| `jaro_winkler(a, b, prefix_scale=0.1, max_prefix=4)` | [0, 1] |
| `ngram_jaccard(a, b, n=3)` | Character n-gram Jaccard |
| `token_jaccard(a, b)` | Whitespace-token Jaccard |
| `pairwise_strings(A, B=None, metric="levenshtein")` | Returns (m, n) ndarray |
| `topk_strings(query, corpus, k=10, metric="levenshtein")` | Returns (indices, scores) |

---

## Points / Geo

```python
from simmetry.points import (
    euclidean_2d,
    haversine_km,   # utility — returns km, not a registered metric
    haversine_sim,
    pairwise_points, topk_points,
)
```

| Function | Returns | Notes |
|----------|---------|-------|
| `euclidean_2d(a, b)` | float in (0, 1] | 2D Cartesian |
| `haversine_km(a, b)` | float (km) | **not registered** — utility only |
| `haversine_sim(a, b, scale_km=...)` | float in [0, 1] | default scale = half Earth circumference |
| `pairwise_points(A, B=None, metric="haversine_sim")` | (m, n) ndarray | |
| `topk_points(query, corpus, k=10, metric="haversine_sim")` | (indices, scores) | |

---

## Sets

```python
from simmetry.sets import jaccard, dice, overlap, tversky
```

| Function | Signature | Notes |
|----------|-----------|-------|
| `jaccard(a, b)` | Iterable, Iterable | \|A∩B\| / \|A∪B\| |
| `dice(a, b)` | Iterable, Iterable | 2\|A∩B\| / (\|A\| + \|B\|) |
| `overlap(a, b)` | Iterable, Iterable | \|A∩B\| / min(\|A\|, \|B\|) |
| `tversky(a, b, alpha=1.0, beta=1.0)` | Iterable, Iterable | Generalises Jaccard/Dice |

---

## ANN (Optional)

```python
from simmetry.ann import build_hnsw, HNSWIndex, build_faiss, FaissIndex
```

### `build_hnsw(X, space="cosine", ef_construction=200, M=16, ef=50)`

Build an HNSW index. Requires `pip install "simmetry[ann-hnsw]"`.

### `build_faiss(X, metric="ip")`

Build a Faiss flat index. Requires `pip install "simmetry[ann-faiss]"`.

Both return an index object with a `.query(q, k=10)` method returning `(labels, raw_distances)`.
Use `SimIndex` to get unified `(indices, similarities)` output.
