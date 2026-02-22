# Benchmarks

This directory contains lightweight benchmark scripts for `simmetry`.

## Current script

- `run.py`: internal baseline timings for vector `pairwise` and exact `topk`

## Run

```bash
python bench/run.py
```

## Notes

- Results are machine-dependent
- Comparative benchmarks against other libraries are not yet published
- If `simmetry[fast]` is installed, some pairwise vector metrics (`euclidean_sim`, `manhattan_sim`) can use Numba acceleration
