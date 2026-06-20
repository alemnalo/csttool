# csttool

Production CLI tool for CST assessment from DWI/DTI. Public-facing: docs build via mkdocs, CI runs tests on every push. Not the home for thesis prose or granular dev-log investigations — see Related Repos below.

## Environment
- Python >=3.10 (pyproject.toml) — note: global CLAUDE.md prefers 3.11+; this repo's floor is 3.10, don't bump without checking CI matrix.
- Setup: `conda env create -f environment.yml && conda activate csttool`
- Test: `pytest`. Docs: `mkdocs build --strict` / `mkdocs serve`.

## Where things live
See `docs/contributing/architecture.md` for module layout and `docs/explanation/design-decisions.md` for rationale — don't duplicate either here.

## Settled, don't re-derive
- RNG seed defaults to 42; every stochastic step must plumb through `reproducibility/`.
- Every CLI run writes `provenance.json` (versions, exact command line, resolved paths).
- Bidirectional CST extraction's artifact_index diagnostic (see CHANGELOG) is the agreed method for separating cortical-placement artifacts from genuine L/R asymmetry — don't propose alternatives without checking CHANGELOG first.

## Related repos
- `../csttool-devlog` — bug investigations and feature planning docs that used to live in this repo's docs/. Check there before re-investigating a known issue (e.g. ROI asymmetry has 4 investigation docs already).
- `../masterthesis` — thesis prose + validation experiments. Do NOT recreate a `thesis/` directory here; it was deliberately removed in commit f9ee9a8.
