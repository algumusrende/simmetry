# Changelog

All notable changes to **simfast** will be documented in this file.

The format is based on **Keep a Changelog**, and this project adheres to **Semantic Versioning**.

## [1.0.1] - 2026-02-21
### Added
- Optional Numba acceleration for `pairwise(..., metric="euclidean_sim" | "manhattan_sim")` when installed via `simfast[fast]`.

### Changed
- Improved validation and error messages for vector dimension mismatches.
- Fixed `similarity([], [], metric="auto")` to route to string similarity batch behavior.
- Project cleanup for public/PyPI release packaging.

## [1.0.0] - 2026-02-21
### Added
- Auto similarity (`metric="auto"`) across strings/vectors/points/sets.
- Batch string APIs (`pairwise_strings`, `topk_strings`).
- Optional ANN module (`hnswlib` / `faiss-cpu`) via extras.
- Unified `SimIndex` with `exact` / `hnsw` / `faiss` backends.
- Composite similarity for dict records (field metrics + weights).
