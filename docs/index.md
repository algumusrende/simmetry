# simmetry

`simmetry` provides similarity scoring for strings, vectors, geo points, and sets with a small NumPy-first API.

## Install

```bash
pip install simmetry
pip install "simmetry[fast]"
```

Optional extras:

- `fast` -> Numba acceleration for selected pairwise vector metrics
- `ann-hnsw` -> `hnswlib`
- `ann-faiss` -> `faiss-cpu`

## Core API

- `similarity(a, b, metric=...)`
- `infer_metric(a, b)`
- `pairwise(X, Y=None, metric="cosine")`
- `topk(query, X, k=10, metric="cosine")`
- `SimIndex(...).add(X).query(q, k)`

See the API reference page for signatures and behavior notes.
