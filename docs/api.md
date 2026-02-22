# API Reference

This page documents the public API manually (the project currently does not expose generated API docs from docstrings).

## Top-level (`simmetry`)

### `similarity(a, b, metric="auto", *, weights=None)`

Computes similarity for:

- scalar strings
- numeric vectors
- geo points (`(lat, lon)`)
- sets
- batch strings (list/tuple of strings)
- batch numeric matrices (`numpy.ndarray`, 2D)
- composite mappings when `metric` is a field-to-metric mapping

Return type:

- `float` for scalar inputs
- `numpy.ndarray` for batch inputs

### `infer_metric(a, b) -> str`

Returns the metric name `similarity(..., metric="auto")` would choose.

Use this when you want auto-dispatch convenience but still want explicit observability.

### `pairwise(X, Y=None, metric="cosine") -> np.ndarray`

Vector pairwise similarity matrix.

Supported metrics:

- `cosine`
- `dot`
- `euclidean_sim`
- `manhattan_sim`
- `pearson`

### `topk(query, X, k=10, metric="cosine") -> (idx, scores)`

Exact top-k search over vectors using the pairwise implementation.

### `SimIndex`

Unified vector search wrapper.

Backends:

- `exact`
- `hnsw` (optional dependency)
- `faiss` (optional dependency)

Methods:

- `.add(X)`
- `.query(q, k=10)`

## String submodule (`simmetry.strings`)

- `levenshtein(a, b)`
- `levenshtein_distance(a, b)`
- `jaro_winkler(a, b)`
- `ngram_jaccard(a, b, n=3)`
- `token_jaccard(a, b)`
- `pairwise_strings(A, B=None, metric="levenshtein")`
- `topk_strings(query, corpus, k=10, metric="levenshtein")`

## Metrics registry

- `register(name, fn, kind="generic")`
- `get(name)`
- `available(kind=None)`
