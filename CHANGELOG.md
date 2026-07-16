# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`--extraction-method bidirectional`** on `csttool run` — two-pass seeding with
  per-side count-bounded intersection and a forward/reverse artifact diagnostic.

  **Motivation:** Atlas-based motor cortex ROIs land at slightly different positions
  relative to the GM/WM boundary on each side, causing the forward-seeded (motor→brainstem)
  pass to produce asymmetric streamline counts. Brainstem-seeded reverse tracking is
  inherently symmetric (confirmed on in-vivo data: R/L = 0.987). Bidirectional seeding
  removes the cortical placement artifact while preserving genuine unilateral asymmetry
  (e.g. stroke, tumour) — a bilateral-symmetry cap is intentionally NOT applied.

  **Algorithm (three steps):**
  1. *Forward pass* — seed from left and right motor cortex ROIs separately; keep
     streamlines that reach the brainstem.
  2. *Reverse pass* — seed from brainstem ROI; keep streamlines that reach each
     motor cortex ROI, yielding `bs_to_left` and `bs_to_right` bundles.
  3. *Per-side count-bounded selection* — voxelise the reverse bundles into density
     maps; cap each side independently at `min(N_forward, N_reverse)`; from each
     forward bundle take the top streamlines ranked by spatial overlap score with the
     corresponding reverse density map.

  **Diagnostic (`artifact_index`):** Per-side forward/reverse inflation ratios are
  reported. When the two ratios diverge (`artifact_index > 0.20`), residual L/R count
  asymmetry is likely a cortical-interface artifact; when they agree, residual asymmetry
  is likely structural (genuine biology or pathology). This lets the method correct the
  artifact without masking pathology.

  **Result on personal in-vivo data:** streamline count LI = +0.002 (271 L / 270 R),
  vs −0.128 for passthrough. Matches the brainstem-seeded ground-truth (LI = +0.007).

  **New files:**
  - `src/csttool/extract/modules/bidirectional_filtering.py`
  - `docs/fixes/bidirectional_seeding.md`
  - `docs/explanation/design-decisions.md` — new section on bidirectional seeding

  **Modified files:**
  - `src/csttool/extract/__init__.py` — export `extract_cst_bidirectional`
  - `src/csttool/cli/__init__.py` — `run` choices extended
  - `src/csttool/cli/commands/extract.py` — guard + `run_bidirectional_extraction`
  - `src/csttool/cli/commands/run.py` — routing branch added

### Fixed

- **Streamlines are now reoriented before along-tract profiling.** `compute_tract_profile`
  resampled each streamline in its **stored point order** and averaged across the bundle by
  point index. Tractography stores each streamline as `[backward_from_seed][forward_from_seed]`,
  so a bundle mixes orientations: averaging index *i* aligned one streamline's pontine end
  with another's precentral end. AFQ/Yeh-style bundle profiles reorient to a reference
  direction first; csttool did not.

  Three consumers already assumed index 0 was the inferior (pontine) end and none verified
  it: `compute_localized_metrics`'s region bins, and the `'Pontine Level'` label the profile
  plots draw at position 0.

  **Fix.** `orient_streamlines_inferior_to_superior` (new, exported from `csttool.metrics`)
  flips each streamline so it runs from its inferior to its superior end, and
  `compute_tract_profile` applies it before sampling. Orientation is decided by comparing the
  mean Z of the first and last quartiles — more robust than the two endpoints alone, which are
  the noisiest part of a streamline. Ties fall through to Y, then X, then keep the stored
  order, so the choice is deterministic and depends only on the streamline itself. The rule is
  anchored to anatomy rather than to a bundle centroid, because index 0 = pontine is hardcoded
  downstream: a merely self-consistent bundle could be flipped as a unit and silently swap the
  pontine and precentral labels. Streamlines arrive in RASMM world coordinates, where +Z is
  superior by the NIfTI standard; only the two ends of the same streamline are compared, so
  the result is translation-invariant and unaffected by the subject not being recentered. A
  bundle that does not run predominantly superior-inferior now warns, since the labels are
  meaningless for one that doesn't.

  **Scope of the defect — active, and it moves reported numbers.** Measured on in-vivo CST
  data: **~20% of streamlines ran superior→inferior** (L 79.5% / R 82.4% inferior→superior),
  so bundles were genuinely mixed rather than coherent by luck. The contamination compressed
  the two ends of the profile toward each other, so correcting it *steepens* the true
  pontine→PLIC→precentral gradient:

  | Region | Left | Right |
  |---|---|---|
  | pontine | +4.1% | +4.7% |
  | PLIC | −0.4% | −0.6% |
  | precentral | −4.4% | −5.8% |

  The middle barely moves because the mixing is symmetric there. Regional FA laterality
  indices shift by 1.20× (pontine), 1.01× (PLIC) and 2.76× (precentral), all remaining inside
  the 0.05 "symmetric" band.

  **Headline `mean`/`std`/`median` FA/MD/RD/AD and every morphology metric are unchanged
  (measured delta exactly 0.0).** `sample_scalar_along_tract` pools every point of every
  streamline, so it is invariant to orientation and point order. Any report's regional values
  and profile figures change; its global table does not.

- **Region bins no longer silently return zeros for profiles other than 20 points long.**
  `compute_localized_metrics` hardcoded `profile[0:7]/[7:14]/[14:20]` and returned all-zeros
  for any other length, even though `n_points` is a caller-settable parameter of
  `compute_tract_profile`. The split is now proportional to the profile length. At the default
  `n_points=20` the bounds are identical to before, so no reported number changes from this.

- **The RNG seed now reaches the `roi-seeded` and `bidirectional` extraction methods.**
  `track_from_seeds` — the tracker behind both — constructed DIPY's `LocalTracking`
  without a `random_seed` argument and did not accept one, so `--rng-seed` could not
  reach the two methods that `run` exposes. This contradicted the project's
  seed-everything-stochastic policy for exactly the methods it foregrounds.

  `random_seed` is now a parameter of `track_from_seeds`, `extract_cst_roi_seeded` and
  `extract_cst_bidirectional`, plumbed from the CLI via `RunContext.rng_tracking_seed()`
  and recorded in each method's `parameters` block in the extraction report. The
  bidirectional method applies one seed to all three passes (forward left, forward right,
  reverse) so the RNG cannot become a source of L/R difference or forward/reverse
  inflation — the quantities `artifact_index` exists to measure.

  **Scope of the defect — latent, not active.** With the deterministic peaks direction
  getter these methods use, DIPY's `LocalTracking` never consumes the RNG that
  `random_seed` seeds (it is consumed only by probabilistic direction getters and by
  `randomize_forward_direction`, which is off). Verified on DIPY 1.12.1: seeds 42, 7 and
  `None` produce bitwise-identical bundles. **No previously reported result changes**, and
  results produced before this fix were reproducible. The guarantee now holds by
  construction rather than as a side effect of the direction getter's determinism.
  Regression tests: `tests/reproducibility/test_extraction_determinism.py`.

- **`csttool run` accepts `--rng-seed`** (default 42), and passes it to both the tracking
  and extraction stages. The flag previously existed only on `csttool track`, so the seed
  was unsettable on the only command that can run `roi-seeded` / `bidirectional`; it was
  fixed at 42 by `cmd_track`'s fallback.

- **`csttool run` no longer consumes its `--nifti` input.** The BIDS reorganizer used
  `shutil.move` unconditionally; when preprocessing was skipped (pass-through mode), the
  user-supplied DWI/bval/bvec/json files were relocated into the output tree and
  disappeared from their original location. Fix: detect whether each source path lives
  inside the run's output directory and copy when it does not.
- **CST extraction report preserved in BIDS reports.** The reorganizer renamed every
  JSON in `extraction/logs/` to `*_log-extraction.json`, causing the registration report
  to overwrite the CST extraction report (last-write-wins). Fix: use per-filename tag
  overrides so the registration report becomes `*_log-registration.json` and the CST
  extraction stats (including the new `forward_reverse_ratio_*` and `artifact_index`
  fields) survive as `*_log-extraction.json`.

---

## [0.5.0] - 2026-04-23

### Added

- **MPPCA** denoising added
- **BIDS-native output layout** — `csttool run` now writes a BIDS derivatives tree by
  default, with no extra flags required. All outputs are moved (not copied) into the
  subject directory; stage working directories are removed unconditionally afterwards.
  - `sub-<id>/[ses-<id>/]dwi/` — preprocessed NIfTI, bval/bvec, scalar maps
    (`space-orig_model-DTI_param-{FA,MD,RD,AD}_dwimap.nii.gz`), derivative JSON sidecars
  - `sub-<id>/[ses-<id>/]dwi/tractography/` — whole-brain, CST left, CST right, and
    bilateral combined tractograms
  - `sub-<id>/[ses-<id>/]figures/` — QC images renamed with stage and label entities
    (`stage-{preproc,tracking,extraction,metrics}_qc-{label}.png`)
  - `sub-<id>/[ses-<id>/]reports/` — HTML/PDF reports, metrics JSON/CSV, pipeline logs
  - `dataset_description.json`, `participants.tsv`, and `participants.json` at dataset root
  - `SourceDatasets: [{"URL": "bids::"}]` when derivatives are nested under a raw BIDS root
- **`--bids-out`** flag on `csttool run` — overrides the derivatives root (default: `--out`)
- **`--session-id`** flag on `csttool run` — sets the BIDS session label
- **`--bids-out`** flag on `csttool batch` — writes `dataset_description.json` and
  `participants.tsv` at the dataset root after the batch completes
- **Raw BIDS import** via `csttool import --dicom <dir> --raw-bids <out>` — produces a
  fully BIDS-compliant raw dataset (`DatasetType: raw`)
  - Subject label derived from SHA-256 hash of `PatientID` by default (anonymised)
  - Session label derived from `StudyDate`
  - `--keep-phi` flag to use `PatientID` directly (prints PHI warning)
  - `--subject-id` and `--session-id` to override auto-derivation
- **`dcm2niix` promoted to primary DICOM converter** — handles Siemens, GE, Philips, and
  Hitachi; generates BIDS JSON sidecars automatically. Falls back to `dicom2nifti` with a
  `fallback_used` flag and a vendor-specific warning for known-unreliable vendors.
- **`pydicom`** added as a required dependency
- **`bids/output.py`** module: `write_dataset_description`, `update_participants_tsv`,
  `write_participants_json`, `bids_filename`, `write_derivative_sidecar`,
  `sanitize_bids_label`, `parse_dicom_age`, `hash_patient_id`
- **`manufacturer`** field added to import report JSON and series info JSON
- 25 unit tests for BIDS output helpers and QC image naming (`tests/bids/`)

### Changed

- Output layout is BIDS derivatives by default — the flat stage-directory structure
  (`tracking/`, `extraction/`, `metrics/`, `preprocessing/`) is internal working state
  only, removed after finalization
- QC images explicitly routed to `figures/` with systematic names regardless of flags;
  output contract is stable across all flag combinations
- Reports (HTML, PDF) and tabular outputs (metrics JSON, CSV) routed to `reports/`,
  distinct from QC images in `figures/`
- Default denoising method changed to `mppca` from `nlmeans`. Reason: `mppca` requires no
  assumptions to be made on coil count or noise distribution of DWI data.
- `nlmeans` parameter `num_threads` changed from `-1` to `1`. Reason: single thread CPU reduces performance but ensures determinism when `nlmeans` denoising is implemented.
- `gibbs_removal` changed so that `num_threads = 1`, analogously to above

## [0.4.0] - 2026-01-28

### Added

- **Coordinate validation system** to prevent silent failures from tractogram/FA
  coordinate mismatches
  - Automatic validation checks bounding box overlap, detects voxel vs world space,
    and verifies orientation
  - New `--skip-coordinate-validation` flag to bypass validation (not recommended)
- **Hemisphere separation QC visualization** showing left/right CST bundles separately
  with midline reference
  - Displays cross-hemisphere contamination metrics
  - Color-coded QC status (green for good separation, red for warnings)
- **`--quiet` flag** for `extract`, `run`, and `batch` commands
- Documentation updates:
  - Expanded [limitations.md](docs/explanation/limitations.md)
  - Updated [data-requirements.md](docs/getting-started/data-requirements.md) with
    coordinate space requirements

### Changed

- Extract command now validates tractogram coordinates against FA space before processing
- All QC visualization outputs now include hemisphere separation view by default when
  using `--save-visualizations`

## [0.3.1] - 2026-01-25

### Changed

- Unified batch analysis metrics with the single-subject report format
- `batch_metrics.csv` now includes:
  - All diffusivity scalars (MD, RD, AD) in addition to FA
  - Localized metrics for pontine, PLIC, and precentral regions
  - Consistent column naming conventions

## [0.3.0] - 2026-01-22

### Added

- Robust batch processing system (`csttool batch`)
- Manifest-based and BIDS-directory based batch execution
- Comprehensive pre-flight validation for batch inputs
- Consolidated CSV and JSON reporting for batch runs
- Parallel processing with timeout handling

## [0.2.1] - 2026-01-20

### Changed

- Refactored CLI into a modular package structure (`src/csttool/cli/`)
- Moved CLI entry point commands to separate modules

### Fixed

- Updated `dipy.core.gradients.gradient_table` call to use `bvecs` keyword argument
- Enabled `copy_header=True` in `image.resample_to_img` to preserve header information

## [0.2.0] - 2026-01-20

### Changed

- Refined CST extraction logic:
  - Added mutual exclusivity filter for motor cortices
  - Added midline rejection filter (streamlines cannot cross X=0 sagittal plane)
- Major refactor of PDF report generation:
  - Replaced inline generation with Jinja2 templating
  - Switched to WeasyPrint for HTML-to-PDF conversion

### Fixed

- Resolved ROI placement asymmetry caused by orientation mismatch (LAS vs RAS)
- Fixed `csttool run` failure when using NIfTI input (`--nifti`) instead of DICOM
- Re-enabled SyN non-linear registration for improved ROI symmetry
- Adjusted affine handling to respect original subject orientation
