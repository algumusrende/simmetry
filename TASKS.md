# simmetry — Fix & Enhancement Task List

Generated from full codebase + PyPI audit.

---

## PyPI / Packaging

- [ ] Fix `Homepage` URL — change from `https://pypi.org/project/simmetry/` to `https://github.com/algumusrende/simmetry`
- [ ] Fix `[ann]` optional dependency — either add `faiss-cpu>=1.7.4` to it or remove the undocumented `ann` extra entirely
- [ ] Fix package description — remove "Blazing-fast", replace with `"NumPy-first similarity scores for strings, vectors, points, and sets."`
- [ ] Add missing project URLs — `Changelog` and `Bug Tracker` entries in `[project.urls]`
- [ ] Resolve version vs classifier contradiction — either move to `0.x` versioning or promote classifier from `Alpha` to `Beta`
- [ ] Add numpy 2.x compatibility — either add upper bound `numpy>=1.23,<3` or add a CI matrix job testing against numpy 2.x

---

## README

- [ ] Remove `pairwise_points` / `topk_points` from the metrics list — they are batch functions, not metric names; move them to the "Batch Point APIs" section only
- [ ] Clarify `[ann]` vs `[ann-hnsw]` vs `[ann-faiss]` — document what each extra installs
- [ ] Add a note that `topk()` (vectors) returns unsorted results — until it's fixed, users need to know
- [ ] Clarify `SimIndex.query()` return semantics — document that `hnsw`/`faiss` backends return distances, `exact` returns similarities
- [ ] Document the 2-element list footgun — warn that `similarity([x, y], [a, b])` infers `haversine_km`, not cosine

---

## Correctness Bugs

- [ ] Fix `topk()` (vectors) — sort results — currently returns unsorted top-k; `topk_strings` and `topk_points` both sort, vectors should too
- [ ] Fix `levenshtein()` zero-length edge case — `levenshtein("", "")` raises `ZeroDivisionError`; should return `1.0`
- [ ] Fix `SimIndex.query()` return semantics — normalize backend distances to similarities so all backends return `(indices, similarities)` consistently

---

## API Design

- [ ] Resolve `haversine_km` metric contract — keep `haversine_km` as a standalone utility (not registered as a metric), add `haversine_sim(a, b, scale_km=20000.0)` that returns a `[0,1]` similarity and register that instead
- [ ] Fix auto-metric `_is_point_like` footgun — 2D numeric lists/tuples incorrectly route to `haversine_km`; tighten the heuristic or require explicit `metric=` for points
- [ ] Add helpful error for missing composite dict fields — `similarity(a, b, weights={"x": ...})` raises bare `KeyError`; validate fields upfront

---

## Type Annotations & Docs

- [ ] Add type annotations throughout all source files (`mypy`-clean)
- [ ] Add docstrings to all public functions and classes
- [ ] Add module-level `__all__` where missing

---

## Tests

- [ ] Add test: `levenshtein("", "")` — currently crashes
- [ ] Add test: `levenshtein("", "abc")` — one-sided empty string
- [ ] Add test: `topk()` results are sorted — currently no assertion on order
- [ ] Add test: `similarity([1.0, 2.0], [3.0, 4.0])` routing — verify the haversine footgun is either fixed or explicitly tested
- [ ] Add test: `SimIndex` exact vs hnsw vs faiss return consistency — same query, same top result
- [ ] Add test: composite dict with missing field raises clear error
- [ ] Add test: NaN vector inputs

---

## CI

- [ ] Add numpy 2.x job to CI matrix
- [ ] Add Python 3.13 job if not already tested
- [ ] Add benchmark job (non-blocking) or remove the bench claim from README until published

---

## Enhancements (Roadmap)

- [ ] Add `hamming` similarity (vectors + strings of equal length)
- [ ] Add `tversky` set similarity (generalizes Jaccard/Dice with asymmetric weights)
- [ ] Add `cosine_distance` alias (`1 - cosine`) for sklearn-compatible users
- [ ] Add `pairwise()` dispatch for string and point inputs (not just vectors)
- [ ] Publish comparative benchmarks (RapidFuzz, sklearn, faiss baselines)
- [ ] Set up hosted docs site
