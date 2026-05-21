# Development Setup

Get a working development environment in five minutes.

## Clone and install

```bash
git clone https://github.com/alemnalo/csttool.git
cd csttool

python -m venv .venv
source .venv/bin/activate

pip install -e ".[test]"
```

The `test` extra pulls in pytest plus the documentation toolchain (`mkdocs`, `mkdocs-material`, `mkdocstrings`).

## System dependencies

`csttool` calls a few external binaries. Install whichever are missing:

```bash
# Debian / Ubuntu
sudo apt install dcm2niix libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libffi-dev

# macOS
brew install dcm2niix pango libffi
```

`dcm2niix` is needed for DICOM import. The `libpango` / `libffi` packages back WeasyPrint, which renders the metrics PDF.

## Sanity-check the install

```bash
csttool doctor
```

`doctor` verifies that Python is recent enough and that every required Python library imports cleanly. Any failure printed here is the right place to start debugging.

## Run the test suite

```bash
pytest                       # everything
pytest tests/unit            # fast unit tests only
pytest -k "extract"          # subset by keyword
pytest -x --pdb              # stop on first failure and drop into pdb
```

The integration suite under `tests/integration/` exercises the full CLI on small synthetic data and is the slow part. Plan for a few minutes.

## Build the docs locally

```bash
mkdocs serve
```

This serves the site at <http://127.0.0.1:8000> with hot reload. To produce a static build (and surface broken links):

```bash
mkdocs build --strict
```

`--strict` turns warnings — broken internal links, missing images, mkdocstrings symbol-not-found — into errors. Always run it before pushing a docs change.

## Updating dependencies

`pyproject.toml` is the source of truth. After editing it:

```bash
pip install -e ".[test]"
```

There is no lockfile; pin versions in your PR description if you have a reason to require an exact version.

## See also

- [Code Style](code-style.md)
- [Architecture](architecture.md)
