# Changelog

All notable changes to **simmetry** will be documented in this file.

The format is based on **Keep a Changelog**, and this project adheres to **Semantic Versioning**.

## [1.1.0] - 2026-06-05

### Added
- `haversine_sim(a, b, scale_km=...)` — geographic similarity in [0, 1] registered as the `haversine_sim` metric.
- `hamming` — normalized Hamming similarity for equal-length sequences. Registered as a `vector` metric.
- `cosine_distance` — `1 - cosine`, registered as a `vector` metric for sklearn-compatible workflows.
- `tversky` — Tversky index for sets: `|A∩B| / (|A∩B| + α|A\B| + β|B\A|)`. Defaults (α=β=1) equal Jaccard; α=β=0.5 equals Dice.
- `pairwise()` now supports `hamming` and `cosine_distance`.
- `SimIndex.query()` now converts ANN backend distances to similarities consistently.
- Field presence validation in `similarity()` for composite dict records.
- `[ann]` extra now installs both `hnswlib` and `faiss-cpu`.
- CI test matrix covering Python 3.10–3.13 and NumPy 1.x / 2.x.
- GitHub Pages documentation site.

### Changed
- **Breaking:** `haversine_km` is no longer a registered metric. Use `haversine_sim` as the metric; import `haversine_km` directly for raw km values.
- **Breaking:** Auto-metric for 2-number tuples now returns `haversine_sim`. `_is_point_like` restricted to tuples (not lists) within valid lat/lon ranges.
- `pairwise_points` and `topk_points` default metric changed from `haversine_km` to `haversine_sim`.
- Package description, Homepage URL, and classifier updated in `pyproject.toml`.
- Version bumped to `1.1.0`.

## [1.0.3] - 2026-03-15

### Changed
- `haversine_km` now returns geographic distance in kilometers instead of a normalized score.
- `topk_points(..., metric="haversine_km")` ranks by nearest distance first.

## [1.0.2] - 2026-02-22

### Added
- Point batch APIs: `pairwise_points()` and `topk_points()`.

## [1.0.1] - 2026-02-21

### Added
- Optional Numba acceleration for `euclidean_sim` / `manhattan_sim` via `simmetry[fast]`.

## [1.0.0] - 2026-02-21

### Added
- Initial release: auto similarity, string/vector/point/set metrics, ANN support, composite records.
