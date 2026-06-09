<!-- Copyright 2026 the OpenDataLoader-PDF authors. SPDX-License-Identifier: MIT -->
# Releasing to PyPI

The package builds to a pure-Python wheel + sdist with `hatchling`. It installs
the `odl-pdf` console script (`odl_pdf.__main__:main`).

## Steps

```bash
# 0. Pre-flight: clean tree, green tests, version bumped in pyproject.toml.
uv run pytest -q

# 1. Build wheel + sdist:
uv build
#   -> dist/odl_pdf-X.Y.Z-py3-none-any.whl
#   -> dist/odl_pdf-X.Y.Z.tar.gz

# 2. Validate the artifacts:
uvx twine check dist/*

# 3. (Optional) smoke-test in a throwaway venv:
uv venv /tmp/odl-smoke && source /tmp/odl-smoke/bin/activate
uv pip install dist/odl_pdf-*.whl
odl-pdf some.pdf -f markdown
deactivate

# 4. Upload (TestPyPI first if you want a dry run):
uvx twine upload --repository testpypi dist/*   # optional
uvx twine upload dist/*

# 5. Tag the release:
git tag v0.1.0 && git push origin v0.1.0
```

## Notes

- **Core has no cloud/AI deps.** The base install pulls only `pikepdf`, `pypdf`,
  and `cryptography`. The hybrid (vision) mode is an extra:
  `pip install 'odl-pdf[hybrid]'` (adds `boto3` + `pypdfium2`).
- **Versioning.** Bump `version` in `pyproject.toml`; the wheel/sdist names and
  the `Version:` metadata follow automatically.
