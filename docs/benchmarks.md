# Benchmarks

The project includes a basic benchmark harness in `bench/run.py`.

## What exists today

- Internal timing for:
  - vector `pairwise(..., metric="cosine")`
  - exact `topk(..., metric="cosine")`

## What is missing (planned)

- Comparative benchmarks versus:
  - `rapidfuzz` (string metrics)
  - `scikit-learn` pairwise metrics
  - faiss/hnsw exact-vs-ann comparisons under controlled settings
- Published benchmark tables with fixed hardware/software metadata

## How to run

```bash
python bench/run.py
```

For reproducible results, record:

- CPU model
- Python version
- NumPy version
- Whether `simmetry[fast]` (Numba) is installed
- Dataset shape and random seed
