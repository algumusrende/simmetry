# Release checklist (PyPI)

## Pre-flight
- [ ] Ensure `main/master` is green (CI passing).
- [ ] Update version in `pyproject.toml` (SemVer).
- [ ] Update `CHANGELOG.md` with the new version/date and highlights.
- [ ] Run locally:
  - [ ] `pip install -e ".[dev]"`
  - [ ] `pytest -q`
  - [ ] `python bench/run.py` (optional sanity performance check)

## Build
- [ ] Clean old artifacts:
  - [ ] `rm -rf dist/ build/ *.egg-info`
- [ ] Build:
  - [ ] `python -m pip install -U build`
  - [ ] `python -m build`
- [ ] Inspect:
  - [ ] `python -m pip install -U twine`
  - [ ] `twine check dist/*`

## Publish
- [ ] TestPyPI (recommended first):
  - [ ] `twine upload -r testpypi dist/*`
  - [ ] Verify install: `pip install -i https://test.pypi.org/simple/ simfast`
- [ ] PyPI:
  - [ ] `twine upload dist/*`

## Post-release
- [ ] Create a GitHub Release with notes from the changelog.
- [ ] Tag the release: `git tag vX.Y.Z && git push --tags`
- [ ] Announce (optional): LinkedIn / X / community.
