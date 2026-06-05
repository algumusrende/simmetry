# simmetry

**simmetry** is a Python library for computing similarity between strings, vectors, geo points, and sets — using a single, consistent API.

Whether you're deduplicating company records, ranking documents against a query, searching a million embeddings, or checking how close two GPS coordinates are, simmetry gives you the same interface: `similarity(a, b, metric=...)`.

```bash
pip install simmetry
```

---

## Why simmetry?

Most similarity tasks in Python require juggling multiple libraries: `rapidfuzz` for strings, `scikit-learn` for vectors, `geopy` for coordinates, and custom code for everything else. Each has a different API, different output conventions, and no way to combine them.

simmetry unifies this under one interface — and adds batch operations, ANN indexing, and composite record matching on top.

---

## The basics

### One function for everything

`similarity(a, b)` automatically picks the right metric based on input type:

```python
from simmetry import similarity

# Strings → jaro_winkler
similarity("New York", "New York City")
# 0.962

# Vectors → cosine
similarity([1.0, 0.0, 0.5], [0.9, 0.1, 0.5])
# 0.993

# Geo coordinates → haversine_sim (0 = antipodal, 1 = same point)
similarity((41.0082, 28.9784), (41.0122, 28.9760))
# 0.9998

# Sets → jaccard
similarity({"python", "ml", "nlp"}, {"python", "nlp", "cv"})
# 0.5
```

You can always specify the metric explicitly:

```python
similarity("kitten", "sitting", metric="levenshtein")  # 0.571
similarity("kitten", "sitting", metric="jaro_winkler") # 0.746
similarity("kitten", "sitting", metric="ngram_jaccard") # 0.286
```

### Checking which metric will be used

```python
from simmetry import infer_metric

infer_metric("hello", "world")             # "jaro_winkler"
infer_metric((51.5, -0.1), (48.8, 2.3))   # "haversine_sim"
infer_metric({1, 2, 3}, {3, 4, 5})        # "jaccard"
infer_metric([0.1, 0.9], [0.8, 0.2])      # "cosine"
```

!!! tip "Tuple vs list for geo coordinates"
    Only `tuple` inputs trigger the geo heuristic, not `list`.
    `(41.0, 29.0)` → `haversine_sim`, but `[41.0, 29.0]` → `cosine`.
    This avoids ambiguity with 2D numeric vectors. Pass `metric="haversine_sim"` explicitly when in doubt.

---

## String similarity

### Choosing the right string metric

Each string metric is suited to different tasks:

| Metric | Best for | Example |
|--------|----------|---------|
| `jaro_winkler` | Names, company names, short strings with typos | "McDonald's" vs "McDonalds" |
| `levenshtein` | General edit distance, codes, identifiers | "colour" vs "color" |
| `ngram_jaccard` | Longer text, OCR output, transliterations | character-level overlap |
| `token_jaccard` | Multi-word phrases, documents | token set overlap |
| `hamming_str` | Fixed-length codes, DNA sequences | "ATCG" vs "ATGG" |
| `bm25` | Ranking documents against a query | search relevance |

### Finding the best match in a list

`topk_strings` returns the k closest matches from a corpus, sorted by score:

```python
from simmetry.strings import topk_strings

# Scenario: you have a canonical list of company names and a noisy input
canonical = [
    "Apple Inc.",
    "Google LLC",
    "Microsoft Corporation",
    "Amazon.com Inc.",
    "Meta Platforms Inc.",
]

query = "Microsft Corp"  # typo + truncation
idx, scores = topk_strings(query, canonical, k=3, metric="jaro_winkler")

for i, score in zip(idx, scores):
    print(f"  {canonical[i]:<25}  {score:.3f}")
# Microsoft Corporation      0.931
# Apple Inc.                 0.545
# Google LLC                 0.542
```

### All-vs-all pairwise matrix

`pairwise_strings` computes similarity between every pair across two lists:

```python
from simmetry.strings import pairwise_strings
import numpy as np

list_a = ["New York", "Los Angeles", "Chicago"]
list_b = ["New York City", "LA", "Chicago IL"]

S = pairwise_strings(list_a, list_b, metric="jaro_winkler")
# S.shape == (3, 3)
# S[i, j] = similarity between list_a[i] and list_b[j]

print(S.round(3))
# [[0.962 0.506 0.547]   ← "New York" vs each
#  [0.486 0.711 0.506]   ← "Los Angeles" vs each
#  [0.593 0.433 0.944]]  ← "Chicago" vs each
```

Self-similarity (pass only one list):

```python
tags = ["machine-learning", "deep-learning", "reinforcement-learning", "nlp"]
S = pairwise_strings(tags, metric="ngram_jaccard")
# diagonal is 1.0; off-diagonal shows character-level overlap
```

Or use the top-level `pairwise()` which detects the input type automatically:

```python
from simmetry import pairwise

S = pairwise(["cat", "car", "bar", "bat"], metric="levenshtein")
```

---

## Text ranking with BM25

`bm25` treats the first argument as a query and the second as a document, returning a relevance score in [0, 1]. It's designed for ranking — use it with `topk_strings` to search a corpus without building an index:

```python
from simmetry.strings import topk_strings

corpus = [
    "introduction to machine learning with python",
    "deep learning for natural language processing",
    "python string similarity and distance metrics",
    "vector databases and embedding search",
    "fuzzy matching for data deduplication",
    "similarity search at scale with FAISS",
]

query = "python similarity search"
idx, scores = topk_strings(query, corpus, k=3, metric="bm25")

for i, score in zip(idx, scores):
    print(f"  {corpus[i]:<48}  {score:.3f}")
# python string similarity and distance metrics      1.000
# similarity search at scale with FAISS             0.500
# fuzzy matching for data deduplication             0.333
```

BM25 is **asymmetric** — it measures how well a document satisfies a query, not general string closeness. For symmetric string comparison, use `jaro_winkler` or `levenshtein`.

---

## Vector similarity

### Pairwise matrix for embeddings

If you have a set of embeddings (from a model, TF-IDF, or anything else), `pairwise()` gives you the full similarity matrix:

```python
import numpy as np
from simmetry import pairwise

# 1000 sentence embeddings, 384 dimensions
embeddings = np.random.randn(1000, 384).astype("float32")

S = pairwise(embeddings, metric="cosine")
# S.shape == (1000, 1000)
# S[i, j] = cosine similarity between embedding i and embedding j
# diagonal is 1.0
```

Cross-similarity between two sets:

```python
queries = np.random.randn(10, 384)    # query embeddings
documents = np.random.randn(500, 384) # document embeddings

S = pairwise(queries, documents, metric="cosine")
# S.shape == (10, 500)
```

### Top-k nearest neighbours (exact)

```python
from simmetry import topk

q = embeddings[0]
idx, scores = topk(q, embeddings, k=5, metric="cosine")
# Results are sorted descending — highest similarity first
# idx[0] == 0, scores[0] == 1.0 (the query itself)
```

### All vector metrics

```python
from simmetry import similarity
import numpy as np

a = np.array([1.0, 0.3, 0.8])
b = np.array([0.9, 0.4, 0.7])

similarity(a, b, "cosine")         # 0.998  direction similarity
similarity(a, b, "euclidean_sim")  # 0.874  1 / (1 + euclid_dist)
similarity(a, b, "manhattan_sim")  # 0.769  1 / (1 + manhattan_dist)
similarity(a, b, "pearson")        # 0.995  correlation
similarity(a, b, "dot")            # 1.49   raw inner product
similarity(a, b, "cosine_distance") # 0.002 1 - cosine (for distance-based pipelines)
```

---

## Geo / location similarity

`haversine_sim` returns a similarity in [0, 1] — 1.0 for the same point, approaching 0 for antipodal points:

```python
from simmetry import similarity
from simmetry.points import topk_points, pairwise_points

istanbul  = (41.0082, 28.9784)
paris     = (48.8566,  2.3522)
london    = (51.5074, -0.1278)
new_york  = (40.7128, -74.006)
singapore = (1.3521,  103.8198)

# How similar are the locations?
similarity(istanbul, london, "haversine_sim")   # 0.966
similarity(istanbul, new_york, "haversine_sim") # 0.933
similarity(london, singapore, "haversine_sim")  # 0.889

# Find the closest cities to istanbul
cities = [istanbul, paris, london, new_york, singapore]
names  = ["Istanbul", "Paris", "London", "New York", "Singapore"]

idx, scores = topk_points(istanbul, cities, k=3, metric="haversine_sim")
for i, s in zip(idx, scores):
    print(f"  {names[i]:<12}  {s:.4f}")
# Istanbul      1.0000  (itself)
# London        0.9660
# Paris         0.9657

# Pairwise distance matrix (raw km, not similarity)
from simmetry.points import haversine_km
D = pairwise_points(cities[:3], metric="haversine_km")
# D[i, j] = km between city i and city j
```

---

## Set similarity

Sets, lists, or any iterables work as input:

```python
from simmetry import similarity
from simmetry.sets import tversky

a = {"python", "machine-learning", "nlp", "transformers"}
b = {"python", "deep-learning", "nlp", "computer-vision"}

similarity(a, b, "jaccard")   # 0.333  |intersection| / |union|
similarity(a, b, "dice")      # 0.5    harmonic-mean weighting
similarity(a, b, "overlap")   # 0.5    |intersection| / |smaller set|
```

**Tversky** is a generalisation with asymmetric weights — useful when false positives and false negatives have different costs:

```python
# Standard Jaccard (alpha=beta=1)
tversky(a, b, alpha=1.0, beta=1.0)  # 0.333

# Standard Dice (alpha=beta=0.5)
tversky(a, b, alpha=0.5, beta=0.5)  # 0.5

# Penalise items in b but not in a more heavily
tversky(a, b, alpha=0.2, beta=0.8)  # 0.556
```

---

## Composite record matching

When you have structured records with multiple fields, you can specify a metric per field and a weight for each:

```python
from simmetry import similarity

# Entity resolution: are these two records the same company?
record_a = {
    "name":     "Acme Corporation",
    "city":     "New York",
    "location": (40.7128, -74.006),
    "tags":     {"manufacturing", "b2b", "industrial"},
}
record_b = {
    "name":     "ACME Corp.",
    "city":     "New York City",
    "location": (40.7200, -74.010),
    "tags":     {"b2b", "industrial", "wholesale"},
}

score = similarity(
    record_a,
    record_b,
    metric={
        "name":     "jaro_winkler",
        "city":     "token_jaccard",
        "location": "haversine_sim",
        "tags":     "jaccard",
    },
    weights={
        "name":     0.4,
        "city":     0.1,
        "location": 0.2,
        "tags":     0.3,
    },
)
# score → 0.883  → likely the same entity
```

Each field can use any registered metric — strings, vectors, points, and sets can all appear in the same record.

---

## Large-scale vector search with SimIndex

For large corpora (100k+ vectors), exact pairwise search becomes too slow. `SimIndex` wraps exact and ANN backends behind a single interface, always returning `(indices, similarities)` — not raw distances:

```python
import numpy as np
from simmetry import SimIndex

# 500k product embeddings, 128 dimensions
X = np.random.randn(500_000, 128).astype("float32")

# Exact search — fine up to ~50k
index = SimIndex(metric="cosine", backend="exact").add(X)

# HNSW — approximate, ~10-50x faster  (pip install "simmetry[ann-hnsw]")
index = SimIndex(metric="cosine", backend="hnsw").add(X)

# FAISS — GPU-ready, best for very large corpora  (pip install "simmetry[ann-faiss]")
index = SimIndex(metric="cosine", backend="faiss").add(X)

# Query is identical regardless of backend
query = X[0]
idx, scores = index.query(query, k=10)

print(idx)     # array([    0, 42317, 91823, ...])
print(scores)  # array([1.000, 0.923, 0.917, ...])  ← similarities, not distances
```

Switch backends to benchmark or scale without changing any query code.

---

## Custom metrics

Register any function as a named metric and use it with the full API:

```python
from simmetry import register, similarity, pairwise

def jaro(a: str, b: str) -> float:
    """Example: plug in your own implementation."""
    ...

register("my_jaro", jaro, kind="string")

# Now works with the full API
similarity("hello", "helo", metric="my_jaro")

from simmetry.strings import pairwise_strings
S = pairwise_strings(["hello", "world"], metric="my_jaro")
```

---

## All available metrics

```python
from simmetry import available

available()           # all registered metrics
available("vector")   # cosine, cosine_distance, dot, euclidean_sim, hamming, manhattan_sim, pearson
available("string")   # bm25, hamming_str, jaro_winkler, levenshtein, ngram_jaccard, token_jaccard
available("point")    # euclidean_2d, haversine_sim
available("set")      # dice, jaccard, overlap, tversky
```

---

## License

MIT · [GitHub](https://github.com/algumusrende/simmetry) · [PyPI](https://pypi.org/project/simmetry/) · [Changelog](changelog.md)
