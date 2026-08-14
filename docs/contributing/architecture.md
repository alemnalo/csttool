# Architecture

A short tour of the `csttool` codebase. The aim is to give a contributor enough orientation to find the right module for a change. For *why* the pipeline is shaped the way it is, see [Design Decisions](../explanation/design-decisions.md).

## Top-level layout

```
csttool/
├── src/csttool/        # the package
├── tests/              # pytest suite (unit + integration)
├── docs/               # this mkdocs site
├── thesis/             # supporting LaTeX thesis
├── scripts/            # one-off scripts (figure generation, validation)
├── reports/            # generated artefacts (not user-facing)
├── pyproject.toml      # package metadata & dependencies
└── mkdocs.yml          # docs build config
```

## Package layout (`src/csttool/`)

```
csttool/
├── __init__.py
├── cli/                  # argparse setup and subcommand entry points
│   ├── __init__.py       # main() — entry point registered in pyproject.toml
│   └── commands/         # one file per subcommand (cmd_check, cmd_run, ...)
├── ingest/               # DICOM → NIfTI conversion and dataset validation
├── bids/                 # BIDS layout helpers (raw and derivatives)
├── preprocess/           # denoising, unringing, motion correction, masking
│   └── modules/gradient_validation.py  # AU21: bval/bvec sanity + DWI bvec reorientation
├── tracking/             # CSA-ODF, deterministic propagation, whole-brain tractography
├── extract/              # atlas registration + ROI-based CST extraction
├── metrics/              # scalar metrics, tract profiles, HTML/PDF reports
├── validation/           # bundle-vs-reference comparison (Dice, overlap, etc.)
├── batch/                # multi-subject orchestration (BIDS + manifest)
├── reproducibility/      # RNG seeding, provenance logging, version capture
└── data/                 # bundled atlases and template files
```

Each top-level module exposes a small public surface in its `__init__.py`. The corresponding CLI subcommand lives in `cli/commands/` and is a thin wrapper around that public surface — argparse parsing → call the public function → handle exit codes.

## Request flow: `csttool run`

The full pipeline path through the codebase:

1. `cli/__init__.py:main` parses the global argv and dispatches by subcommand name.
2. `cli/commands/run.py:cmd_run` parses run-specific flags and calls each stage in turn:
3. `ingest.import_subject` → DICOM/NIfTI ingestion, BIDS-style staging.
4. `preprocess.preprocess` → optional reslice, MPPCA denoise (Patch2Self/NLMeans optional), brain-mask via median Otsu, optional Gibbs unringing, optional motion correction (with b-vector rotation).
5. `tracking.modules.run_tracking` → CSA-ODF model fit, deterministic propagation with `ThresholdStoppingCriterion`, writes whole-brain `.trk`.
6. `extract.extract_cst` → atlas registration (DIPY `SymmetricDiffeomorphicRegistration`), ROI dilation, filter streamlines by chosen method.
7. `metrics.compute_metrics` → analyse each hemisphere, compute laterality index, render HTML report, optionally rasterise to PDF.

Each stage is independently CLI-callable (`csttool preprocess`, `csttool track`, etc.) — the public entry function is the same one `run` invokes.

## Reproducibility module

`reproducibility/` is the home for cross-cutting concerns that affect every stage:

- RNG seeding (`--rng-seed`, default `42`).
- Provenance logging: the tracking and bilateral-metrics stage reports embed
  a provenance block (git commit, Python version, dependency versions, platform,
  hardware info, thread environment) in their JSON reports. The HTML/PDF report
  presents a filtered subset for scientific readers.

If you add a new stochastic step anywhere in the pipeline, plumb it through this module so determinism guarantees stay intact.

## Tests

```
tests/
├── unit/             # pure-Python, no I/O on real datasets
├── integration/      # CLI invocations on small synthetic data
├── edge_cases/        # AU31: pathological inputs (zero streamlines, all-zero
│                     # DWI, single direction, misordered bvec/bval,
│                     # truncated NIfTI, DICOM missing tags)
└── fixtures/         # tiny synthetic DWI volumes, gradient files
```

Unit tests live next to the module they cover when practical (the layout mirrors `src/csttool/`). Integration tests live under `tests/integration/` and are slower. The `tests/edge_cases/` package (AU31) characterises pathological inputs across module boundaries — it pins current behaviour so a regression that turns a clear error into a silent failure (or vice versa) is caught.

## Where to make common changes

| Change | Module |
|---|---|
| New CLI flag | `cli/commands/<command>.py` + corresponding public function + `docs/reference/parameters.md` + per-command reference page |
| New extraction method | `extract/modules/` + choice in `cli/commands/extract.py` + note in `explanation/design-decisions.md` |
| New scalar metric | `metrics/modules/` + report template under `metrics/modules/reports/` |
| New denoising backend | `preprocess/modules/` + dispatch in `preprocess/preprocess.py` |
| New atlas | `data/` + helper in `extract/` |

## See also

- [Code Style](code-style.md)
- [Design Decisions](../explanation/design-decisions.md)
- [Tractography](../explanation/tractography.md)
