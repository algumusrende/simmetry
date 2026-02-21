# Changelog

All notable changes to **simfast** will be documented in this file.

The format is based on **Keep a Changelog**, and this project adheres to **Semantic Versioning**.

## [1.0.1] - 2026-02-21
### Added
- GitHub Actions CI (tests on Linux/macOS/Windows, Python 3.10–3.13).
- Build job producing sdist + wheel and uploading artifacts.
- Release checklist for PyPI.

### Changed
- Updated `pyproject.toml` metadata

## [1.0.0] - 2026-02-21
### Added
- Auto similarity (`metric="auto"`) across strings/vectors/points/sets.
- Batch string APIs (`pairwise_strings`, `topk_strings`).
- Optional ANN module (`hnswlib` / `faiss-cpu`) via extras.
- Unified `SimIndex` with `exact` / `hnsw` / `faiss` backends.
- Composite similarity for dict records (field metrics + weights).
