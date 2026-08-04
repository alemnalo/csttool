# Troubleshooting

Symptom → fix for the failures users hit most often.

## Pipeline failures

### Extraction returns very few (or no) streamlines

**Symptom**: `csttool extract` completes but `cst_left.trk` / `cst_right.trk` contain a handful of streamlines or are empty.

**Likely causes and fixes**:

1. **Field-of-view does not cover both CST endpoints.** Run `csttool check-dataset --dwi <file>`. If brainstem coverage is missing, the acquisition cannot be salvaged. Otherwise see [Data formats](data-formats.md#fov-and-resolution).
2. **Whole-brain tractogram is too sparse.** Re-run `track` with `--seed-density 2` (or higher). With the default `--fa-thr 0.2` this typically yields a 2–4× denser whole-brain bundle.
3. **Registration failed.** Open `extract/visualizations/registration_qc.png` (requires `--save-visualizations`). Misaligned atlas → re-run with `--fast-registration` disabled, or check that the input FA map is not flipped (use `csttool check-dataset` to validate).

### Patch2Self denoising produces short streamlines

**Symptom**: After `--denoise-method patch2self`, tractography produces noticeably shorter streamlines and the FA map looks overly smooth.

**Fix**: Patch2Self assumes a sufficient number of diffusion volumes (~30+). For low-volume acquisitions, switch to NLMeans:

```bash
csttool preprocess --nifti raw.nii.gz --out ./preproc --denoise-method nlmeans
```

### `extract` fails with a coordinate-system error

**Symptom**: Error mentions affine, voxel-to-world, or coordinate validation.

**Fix**: This usually means the tractogram and FA map come from different processing runs with different reslicing. Re-run `track` and `extract` against the same `preprocess` output. As a last-resort debug, pass `--skip-coordinate-validation` — but treat any extraction it produces with suspicion.

### Degenerate / pathological inputs

These are caught and reported rather than failing silently:

- **Truncated NIfTI** (`load`/`track`): the header loads but the data array cannot be read, raising a clear `NIfTI file appears truncated or corrupt` error naming the path — re-convert from DICOM or restore the file.
- **All-zero DWI** (`track`): yields FA = 0 everywhere and an empty white-matter mask; `fit_tensors` warns `No white-matter voxels found … tractography will produce zero streamlines`. This is a valid but useless input — the run completes with empty output.
- **Single-direction DWI** (`track`): too few directions for the requested SH order; `validate_sh_order` warns and reduces the order automatically.
- **Misordered / malformed bvec/bval**: rejected at load by the gradient-table validator (see [Data requirements](../getting-started/data-requirements.md#gradient-table-validation-bvalsbvecs)).
- **Zero-streamline tractogram** (`extract`): returns a well-formed empty result (zero counts, `extraction_rate = 0`), not a crash.
- **DICOM with missing tags** (`import`): defaulted to empty/zero values and classified as unsuitable for tractography, rather than crashing.

## Installation

### WeasyPrint fails to install or render PDFs

**Symptom**: `csttool` prints `⚠ PDF skipped: weasyprint not installed` or `⚠ PDF skipped: weasyprint's native libraries (Cairo/Pango/GDK-PixBuf) could not be loaded`. The HTML report is still produced.

**Recommended fix (all platforms, including Windows)**: install via conda, which bundles the native libraries cleanly.

```bash
conda install -c conda-forge weasyprint pango cairo gdk-pixbuf
```

**Pip-only fix (Linux/macOS)**: install the optional extra and the system libraries.

```bash
pip install 'csttool[reports]'

# Debian / Ubuntu
sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libffi-dev libcairo2 libgdk-pixbuf2.0-0

# macOS
brew install pango cairo gdk-pixbuf libffi
```

**Windows + pip**: not officially supported. PDF generation will fail with a missing-DLL error even after `pip install 'csttool[reports]'` because the native libraries are not bundled. Use the conda install path.

If you do not need PDF reports, omit `--generate-pdf`; HTML, JSON, and CSV outputs are produced regardless.

### Missing FSL / MRtrix dependencies

**Symptom**: `csttool doctor` flags missing external binaries.

**Fix**: `csttool` itself does not require FSL or MRtrix, but the atlases shipped via `csttool fetch-data` are derived from FSL data. Install FSL only if you intend to compare against FSL-tractography pipelines or run advanced QC.

### `dcm2niix` not found

**Symptom**: `csttool import --dicom ...` fails with `dcm2niix: command not found`.

**Fix**:

```bash
# Debian / Ubuntu
sudo apt install dcm2niix

# macOS
brew install dcm2niix
```

Or install via conda: `conda install -c conda-forge dcm2niix`.

## Reproducibility

### Same input, different `.trk` files across runs

By default tracking is seeded (`--rng-seed 42`) and is bitwise reproducible. If you see drift between runs:

1. Confirm you did **not** pass `--random`, which disables seeding.
2. Pin library versions — Dipy and Numpy minor-version bumps can change floating-point output.
3. See the design rationale in [Limitations](../explanation/limitations.md).

## Related

- [Known Limitations](../explanation/limitations.md)
- [Data formats](data-formats.md)
- [`check-dataset` reference](../reference/cli/check_dataset.md)
