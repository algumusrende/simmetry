# Auto Metric Selection

`simmetry` supports `metric="auto"` for convenience. The selection is deterministic and based on input types.

## Inspecting auto selection

Use `infer_metric(a, b)` to see what will be chosen:

```python
from simmetry import infer_metric

infer_metric("samplecorp", "sample corp")      # "jaro_winkler"
infer_metric((41.0, 29.0), (41.1, 29.1))       # "haversine_km"
infer_metric({1, 2, 3}, {2, 3, 4})             # "jaccard"
infer_metric([1, 2, 3], [1, 2, 4])             # "cosine"
```

## Selection order

Order matters:

1. String sequences (`list[str]`, `tuple[str]`, including empty) -> batch strings (`jaro_winkler`)
2. Scalar strings -> `jaro_winkler`
3. 2-number lists/tuples -> `haversine_km`
4. Sets/frozensets -> `jaccard`
5. Numeric vectors -> `cosine`
6. Fallback -> `cosine`

## Why this exists

Auto mode is intended for quick exploratory usage. In production code, prefer explicit metrics for predictability and reviewability.
