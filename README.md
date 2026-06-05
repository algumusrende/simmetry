# simmetry

[![PyPI](https://img.shields.io/pypi/v/simmetry)](https://pypi.org/project/simmetry/)
[![Tests](https://github.com/algumusrende/simmetry/actions/workflows/test.yml/badge.svg)](https://github.com/algumusrende/simmetry/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-algumusrende.github.io-blue)](https://algumusrende.github.io/simmetry/)

One API for similarity across **strings**, **vectors**, **geo points**, and **sets** — with batch operations, ANN indexing, and composite record matching built in.

```python
from simmetry import similarity

similarity("samplecorp", "sample corp")                        # 0.97  (auto: jaro_winkler)
similarity([1, 0, 0], [0, 1, 0], metric="cosine")             # 0.0
similarity((41.0, 29.0), (41.1, 29.1), metric="haversine_sim") # 0.999
similarity({"ML", "Python"}, {"Python", "AI"}, metric="jaccard") # 0.33
```

## Install

```bash
pip install simmetry
pip install "simmetry[fast]"       # Numba acceleration for euclidean/manhattan pairwise
pip install "simmetry[ann-hnsw]"   # hnswlib ANN backend
pip install "simmetry[ann-faiss]"  # FAISS ANN backend
pip install "simmetry[ann]"        # both ANN backends
```

## Project Status

[![PyPI](https://img.shields.io/pypi/v/simmetry)](https://pypi.org/project/simmetry/)

- Maturity: **Beta** — API stabilising; pin to minor versions in production
- Versioning: semantic versioning; breaking changes bump the minor until `2.0`

---

## Use cases

### Fuzzy string matching

Find the closest matches for a query string against a list of candidates:

```python
from simmetry.strings import topk_strings

companies = [
    "Apple Inc", "Apple Corp", "Appel Inc",
    "Google LLC", "Alphabet Inc", "Microsoft",
]

idx, scores = topk_strings("Apple Inc.", companies, k=3, metric="jaro_winkler")
for i, s in zip(idx, scores):
    print(f"{companies[i]:<20} {s:.3f}")
# Apple Inc            0.993
# Apple Corp           0.966
# Appel Inc            0.963
```

### Text ranking with BM25

Rank documents against a query without a prebuilt index:

```python
from simmetry.strings import topk_strings

docs = [
    "python library for string similarity",
    "fast vector search with FAISS",
    "string matching and distance metrics in python",
    "machine learning model deployment",
]

idx, scores = topk_strings(
    "python string similarity", docs, k=2, metric="bm25"
)
for i, s in zip(idx, scores):
    print(f"{docs[i]:<45} {s:.3f}")
# string matching and distance metrics in python  0.833
# python library for string similarity            0.667
```

### Pairwise similarity matrices

`pairwise()` dispatches automatically — pass strings, points, or vectors:

```python
from simmetry import pairwise
import numpy as np

# Strings — returns (m, n) similarity matrix
names = ["New York", "New York City", "NYC", "Los Angeles"]
S = pairwise(names, metric="jaro_winkler")
# S[0, 1] → 0.962   (New York vs New York City)
# S[0, 3] → 0.409   (New York vs Los Angeles)

# Vectors
X = np.random.randn(500, 64)
S = pairwise(X, metric="cosine")  # (500, 500)

# Geo points
locations = [(41.0, 29.0), (41.1, 29.1), (40.0, 28.0)]
S = pairwise(locations, metric="haversine_sim")  # (3, 3)
```

### Large-scale vector search

`SimIndex` wraps exact and ANN backends behind a single interface.
All backends return `(indices, similarities)` — not raw distances:

```python
import numpy as np
from simmetry import SimIndex

X = np.random.randn(100_000, 128).astype("float32")

# Swap backends without changing query code
index = SimIndex(metric="cosine", backend="exact").add(X)   # exact
index = SimIndex(metric="cosine", backend="hnsw").add(X)    # ~10x faster
index = SimIndex(metric="cosine", backend="faiss").add(X)   # GPU-ready

idx, scores = index.query(X[0], k=10)
# idx    → array([    0, 23451, 87302, ...])
# scores → array([1.000, 0.812, 0.798, ...])  sorted descending
```

### Geo proximity search

```python
from simmetry.points import topk_points

landmarks = [
    (41.0082, 28.9784),  # Istanbul
    (48.8566,  2.3522),  # Paris
    (51.5074, -0.1278),  # London
    (40.7128, -74.0060), # New York
]

query = (41.0, 29.0)  # near Istanbul
idx, scores = topk_points(query, landmarks, k=2, metric="haversine_sim")
for i, s in zip(idx, scores):
    print(f"landmark {i}  similarity={s:.4f}")
# landmark 0  similarity=0.9999
# landmark 1  similarity=0.9657
```

### Entity / record matching

Match structured records with different metrics per field and custom weights:

```python
from simmetry import similarity

record_a = {
    "name": "Acme Corporation",
    "city": "New York",
    "location": (40.71, -74.01),
}
record_b = {
    "name": "ACME Corp",
    "city": "New York City",
    "location": (40.73, -74.00),
}

score = similarity(
    record_a,
    record_b,
    metric={
        "name":     "jaro_winkler",
        "city":     "token_jaccard",
        "location": "haversine_sim",
    },
    weights={"name": 0.5, "city": 0.2, "location": 0.3},
)
# score → ~0.91
```

### Set / tag similarity

```python
from simmetry import similarity

a = {"python", "machine-learning", "nlp"}
b = {"python", "deep-learning", "nlp"}

similarity(a, b, "jaccard")   # 0.5   — |intersection| / |union|
similarity(a, b, "dice")      # 0.667 — harmonic mean weighting
similarity(a, b, "overlap")   # 0.667 — overlap coefficient

# Tversky: penalise b's extras more than a's
from simmetry.sets import tversky
tversky(a, b, alpha=0.2, beta=0.8)  # asymmetric
```

---

## Available metrics

### Vectors

| Metric | Range | Notes |
|--------|-------|-------|
| `cosine` | [-1, 1] | direction similarity |
| `cosine_distance` | [0, 2] | `1 − cosine`; sklearn-compatible |
| `dot` | unbounded | raw inner product |
| `euclidean_sim` | (0, 1] | `1 / (1 + dist)` |
| `manhattan_sim` | (0, 1] | `1 / (1 + dist)` |
| `pearson` | [-1, 1] | correlation coefficient |
| `hamming` | [0, 1] | normalized; equal-length sequences |

### Strings

| Metric | Notes |
|--------|-------|
| `levenshtein` | normalized edit distance |
| `jaro_winkler` | prefix-weighted; good for names |
| `ngram_jaccard` | character trigram Jaccard (default n=3) |
| `token_jaccard` | whitespace-token Jaccard |
| `hamming_str` | normalized Hamming; equal-length strings only |
| `bm25` | BM25 relevance [0, 1]; asymmetric ranking helper |

### Points / Geo

| Metric | Range | Notes |
|--------|-------|-------|
| `euclidean_2d` | (0, 1] | 2D Cartesian similarity |
| `haversine_sim` | [0, 1] | geographic; antipodal ≈ 0, same point = 1 |

`haversine_km` is available as a utility (not a registered metric):

```python
from simmetry.points import haversine_km
km = haversine_km((40.7, -74.0), (51.5, -0.1))  # 5 570.0 km
```

### Sets

| Metric | Formula |
|--------|---------|
| `jaccard` | \|A∩B\| / \|A∪B\| |
| `dice` | 2\|A∩B\| / (\|A\| + \|B\|) |
| `overlap` | \|A∩B\| / min(\|A\|, \|B\|) |
| `tversky(alpha, beta)` | generalises Jaccard (α=β=1) and Dice (α=β=0.5) |

---

## Auto metric selection

Pass `metric="auto"` (the default) and simmetry picks the right metric:

```python
from simmetry import similarity, infer_metric

infer_metric("hello", "world")            # "jaro_winkler"
infer_metric((41.0, 29.0), (42.0, 30.0)) # "haversine_sim"
infer_metric({1, 2, 3}, {2, 3, 4})       # "jaccard"
infer_metric([1.0, 2.0], [3.0, 4.0])     # "cosine"
```

> **Note:** Only `tuple` inputs trigger the geo heuristic, not `list`. This avoids
> ambiguity with 2D numeric vectors. Use `metric="haversine_sim"` explicitly when in doubt.

---

## Custom metrics

```python
from simmetry import register, similarity

def prefix_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return sum(ca == cb for ca, cb in zip(a, b)) / max(len(a), len(b))

register("prefix", prefix_sim, kind="string")
similarity("hello", "help", metric="prefix")  # 0.75
```

---

## Scope and Roadmap

Planned additions:

- BM25 corpus-level ranking (multi-document IDF)
- `pairwise()` cross-type dispatch for composite inputs

## License

MIT
