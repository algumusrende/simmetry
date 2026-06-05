# Changelog

All notable changes to **simmetry** will be documented in this file.

The format is based on **Keep a Changelog**, and this project adheres to **Semantic Versioning**.

## [1.2.0] - 2026-06-05

### Added
- `hamming_str` — normalized Hamming similarity for equal-length strings. Registered as a `string` metric; available in `pairwise_strings` and `topk_strings`.
- `bm25` — BM25 text relevance score normalized to [0, 1]. Uses term-frequency weighting with uniform IDF; designed as a ranking helper via `topk_strings(..., metric="bm25")`. Registered as a `string` metric.
- `pairwise()` now dispatches automatically by input type: `list[str]` routes to `pairwise_strings`, lists of 2-element numeric tuples/lists route to `pairwise_points`, NumPy arrays and numeric sequences route to the existing vector implementation.
- Module docstrings added to `api.py`, `registry.py`, and `index.py`.
- `pytest-cov` added to `[dev]` extras.
- PyPI version badge in README (replaces hardcoded version string).

## [1.1.0] - 2026-06-05

### Added
- `haversine_sim(a, b, scale_km=...)` — geographic similarity in [0, 1] registered as the `haversine_sim` metric. Antipodal points score ~0; same point scores 1.0. Default `scale_km` is half the Earth's circumference (~20 015 km).
- `hamming` — normalized Hamming similarity for equal-length sequences (vectors, character lists, etc.). Registered as a `vector` metric.
- `cosine_distance` — `1 - cosine`, registered as a `vector` metric for sklearn-compatible workflows.
- `tversky` — Tversky index for sets: `|A∩B| / (|A∩B| + α|A\B| + β|B\A|)`. Defaults (α=β=1) equal Jaccard; α=β=0.5 equals Dice. Registered as a `set` metric.
- `pairwise()` now supports `hamming` and `cosine_distance`.
- `SimIndex.query()` now converts ANN backend distances to similarities consistently — all backends return `(indices, similarities)` in the same scale.
- Field presence validation in `similarity()` for composite dict records — raises `KeyError` with a clear message when a metric field is absent from either record.
- `[ann]` extra now installs both `hnswlib` and `faiss-cpu`.
- CI matrix: test workflow covering Python 3.10–3.13 and NumPy 1.x / 2.x.

### Changed
- **Breaking:** `haversine_km` is no longer registered as a similarity metric. Import it directly from `simmetry.points` or `simmetry` as a utility that returns raw kilometers. Use `haversine_sim` as the registered metric.
- **Breaking:** Auto-metric for 2-number tuples now returns `haversine_sim` instead of `haversine_km`, and the `_is_point_like` heuristic is tightened to tuples only (not lists) within valid lat/lon ranges ([-90, 90] × [-180, 180]).
- `pairwise_points` and `topk_points` default metric changed from `haversine_km` to `haversine_sim`.
- Package description updated to remove unsubstantiated "Blazing-fast" claim.
- `Homepage` project URL corrected to the GitHub repository.
- Development Status classifier promoted from Alpha to Beta.

## [1.0.3] - 2026-03-15

### Changed
- `haversine_km` now returns geographic distance in kilometers instead of a normalized similarity score.
- `topk_points(..., metric="haversine_km")` now ranks by nearest distance first.
- Updated point tests and README examples to match the distance-based behavior.

## [1.0.2] - 2026-02-22

### Added
- Point batch APIs: `simmetry.points.pairwise_points()` and `simmetry.points.topk_points()`.
- README examples for batch geo/point similarity.

### Changed
- Release alignment workflow: PyPI and GitHub releases will track the same version tag.

## [1.0.1] - 2026-02-21

### Added
- Optional Numba acceleration for `pairwise(..., metric="euclidean_sim" | "manhattan_sim")` when installed via `simmetry[fast]`.

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
