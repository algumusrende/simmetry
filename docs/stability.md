# Stability and Risk

`simmetry` is currently an **Alpha** project.

## Practical implications

- APIs may change as missing features are added
- Performance characteristics may change between releases
- Some categories (especially points/geo and string metrics) are intentionally narrow in scope today

## Production usage guidance

- Pin versions (`==` or conservative ranges)
- Prefer explicit metrics instead of `auto` in critical paths
- Add regression tests around expected score ranges/ordering
- Validate ANN backends independently if using optional extras

## Versioning policy (current)

The project aims to follow semantic versioning, but during early stabilization you should expect occasional breaking changes in minor releases. This will tighten as the API surface settles.
