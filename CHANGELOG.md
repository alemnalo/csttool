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

### Changed

- **The default denoising method is now `mppca`, on every command.** MPPCA estimates its own
  noise level from the eigenvalue distribution of local PCA patches, so it requires neither a
  receiver-coil count nor an assumption about the noise distribution, and unlike `patch2self`
  it does not need bvals.

  This supersedes the previous entry claiming the same thing. **That claim was false**: the
  flip had been applied only to `run_preprocessing`'s signature default, while all three CLI
  parsers still passed an explicit `default="nlmeans"` that overrode it. No CLI run ever used
  MPPCA. The entry was also filed under the already-released 0.5.0; it has been moved here and
  corrected.

  The default is now defined once, in `csttool/defaults.py` as `DEFAULT_DENOISE_METHOD`, and
  read by the CLI parsers, the command wrappers, `run_preprocessing` and `denoise`. It was
  previously restated in eight places, which is why a one-line change looked complete and was
  not. `tests/test_cli_denoise_default.py` asserts the value each command actually resolves,
  rather than the library signature that looked right while the tool did the opposite.

  **What this retires.** `rician=False` (AU2) and the PIESNO coil count `N` (AU25) exist only
  in the `nlmeans` branch of `denoise()`, so both now leave the default path. They still apply
  if `--denoise-method nlmeans` is chosen explicitly. MPPCA also exposes no thread-count or
  seed parameter, so the multithreading non-determinism behind AU1/AU7 cannot arise on the
  default path; MPPCA output is bitwise identical across repeated runs.

  **Users who need the old behaviour** should pass `--denoise-method nlmeans` explicitly.
  Denoised output, and every metric downstream of it, will change for anyone relying on the
  default.

- **`nlmeans` now runs single-threaded** (`num_threads` `-1` → `1`). Multithreaded reduction
  order varies between runs, so the previous setting made denoising non-deterministic — the
  bug behind audit finding AU1. Single-threaded is slower but reproducible.

- **`gibbs_removal` now runs single-threaded** (`num_processes` `-1` → `1`), the same defect
  class as above (AU7), latent behind `--unring`.

  (These two entries were previously filed under the released 0.5.0; they describe 2026-07-16
  work and have been moved here.)

- **`dipy` is now pinned to `>=1.9,<2`** in `pyproject.toml` and `environment.yml`. It was
  entirely unpinned, so an environment rebuild could silently install a version that breaks
  tracking outright.

  The upper bound is load-bearing rather than cautious. DIPY deprecated passing
  `EuDXDirectionGetter`-based objects (`PeaksAndMetrics`) as `LocalTracking`'s
  `direction_getter` in 1.12.0 and raises `ExpiredDeprecationError` from 2.0.0. Both the
  whole-brain and the extraction tracking paths do exactly that, so DIPY 2.0 would break them.
  Lifting the bound means migrating to `dipy.tracking.tracker.eudx_tracking`.

  The lower bound reflects `estimate_directions` passing `CsaOdfModel(sh_order_max=...)`, the
  post-1.9 parameter name. Developed and tested against 1.12.1.

- **The headline FA/MD/RD/AD mean is now per-streamline (one vote per streamline), not
  point-weighted.** `sample_scalar_along_tract` pools every point of every streamline, so
  longer streamlines cast more votes; the report's global table and the laterality indices
  used that length-biased mean, while the displayed tract profile used the
  per-streamline-normalised summary. The two could disagree, and the LI could shift with
  bundle geometry (audit finding AU10). The thesis's "FA symmetric, count asymmetric"
  conclusion rests on this mean.

  **What changed.** Each scalar block (`fa`/`md`/`rd`/`ad`) now carries two honestly-named
  summaries:
  - *Headline* (`mean`/`std`/`median`/`min`/`max`/`n_streamlines`): per-streamline. Each
    streamline contributes one value (its point-mean), so the summary is length-unbiased.
    `std` is the between-streamline SD of streamline means — the spread of the quantity
    whose mean is reported — not the point-pool scatter, so every report's "±" now means
    between-streamline variability (much smaller than the old voxel scatter). `min`/`max`
    bound the same per-streamline distribution. The global LIs and the report's global
    table consume this.
  - *Point-pool* (`mean_point_weighted` … `n_samples`): every sampled point, length-biased.
    Preserved for QC and bias auditing per the AU5/AU24 instinct of labelling constructed
    numbers rather than removing them; not used as the headline.

  The headline uses the per-streamline mean rather than the profile-derived mean: the
  profile resamples to a fixed arc length and drops streamlines with fewer than five valid
  points — a different population. The four repeated per-scalar blocks in
  `analyze_cst_hemisphere` are now built by one `_compute_scalar_metrics` helper (single
  source of truth for the key set), and `sample_scalar_per_streamline` is the new public,
  length-unbiased counterpart to `sample_scalar_along_tract`. The single-subject and batch
  CSVs gained `*_mean_point_weighted` columns so the bias is visible in the output.

  **Impact — measured, not inferred, on both subjects** (post-AU9 code, DIPY 1.12.1).
  Length spans only ~146–322 points (mean ~227) and corr(length, streamline-mean-FA) is
  weak (−0.20 to +0.29), so the bias is real and signed as theory predicts but small:

  | Subject | Side | point-weighted (old headline) | per-streamline (new headline) |
  |---|---|---|---|
  | healthy control | L / R | 0.4975 / 0.4785 | 0.4984 / 0.4804 |
  | ALS patient | L / R | 0.4149 / 0.4133 | 0.4130 / 0.4134 |

  The FA-LI moves +0.01944 → +0.01833 (healthy) and +0.00194 → −0.00059 (ALS) — all three
  definitions stay well inside the 0.05 "symmetric" band, so the "FA symmetric" conclusion
  survives on both. AU10 is therefore a correctness/coherence fix, not a results-changing
  one (unlike AU9 on the regional values). The thesis subject moves *less* than the healthy
  control did. **Re-deriving the thesis global FA/MD/RD/AD table and the LI figures from the
  new headline is a separate follow-up**, deliberately out of scope of this code fix.

### Fixed

- **Hemisphere splits now use the warped MNI midline, not world X=0 (AU11); the
  Jacobian hemisphere stats now use the full affine and a per-mm gradient (AU19).**
  `register_mni_to_subject` reorients the subject to RAS but never recentres it, so
  the anatomical midline sits at world X = M, not necessarily 0. Four sites split on
  X=0: the streamline midline-crossing exclusion, `sample_peduncle_fa`, the Jacobian
  hemisphere split, and the atlas-warp centroid QC. Bundle L/R assignment is ROI-based
  (which motor cortex a streamline traverses), so the bundles were never wrong — but
  the *diagnostics* were, and `sample_peduncle_fa` fed the "lower left peduncle FA"
  finding behind [[07 Decisions]] D3.

  **Fix — one source of truth.** `compute_warped_midline` (new) warps the MNI X=0
  plane (a whole-brain left mask, `X_mni < 0`) into subject space and returns three
  views of the same surface: a per-voxel `hemisphere_mask` (the anatomical L/R label,
  used by `sample_peduncle_fa` and the Jacobian split), a signed `midline_distance`
  volume (mm; used by the streamline midline-crossing exclusion, since a single
  scalar plane is a poor fit — the warped midline curves with SyN and a whole-brain
  median is biased by asymmetric sub-cortical structures), and a scalar `midline_x`
  (for the centroid sign QC). `register_mni_to_subject` computes it once and threads
  it through `warp_atlas_to_subject` → `extract_cst_passthrough` / `sample_peduncle_fa`.
  The `MIDLINE_TOLERANCE_MM = 8.0` exclusion threshold itself is unchanged (still
  provisional — AU14).

  **AU19, two defects.** (1) `compute_jacobian_hemisphere_stats` took `np.gradient`
  per **voxel index**; the SyN forward field is in **mm** (world units) on
  dipy 1.9–1.12.1 — verified empirically from `DiffeomorphicMap._warp_forward`, not
  assumed — so the gradient must be per mm. The per-voxel-index code scaled the
  Jacobian spread by the voxel size (≈2× on 2 mm isotropic data). (2) The hemisphere
  split used the axis-aligned `affine[0,0]*i + affine[0,3]`, dropping the
  `affine[0,1]*j` and `affine[0,2]*k` terms; these are non-zero even after RAS
  reorientation (scanner shear survives), so the full affine is now used (and the
  per-voxel mask is preferred when available).

  **Impact — measured on both real subjects (post-fix, DIPY 1.12.1).**

  *The premise "M ≠ 0" is **not** confirmed by the gold-standard estimate.* The
  warped MNI midline lands at world X ≈ 0 for both subjects (cortical L/R boundary:
  −0.6 mm healthy, −0.3 mm ALS). The proxies that suggested otherwise — the
  FA>0.15 brain centroid and the brainstem centroid — are unreliable: the brainstem
  is anatomically right-shifted in the ALS subject (centroid +4.9 mm vs warped
  midline −0.3 mm), which is anatomy, not a recentering failure. **M ≈ 0 globally does
  not make a flat X=0 *peduncle* split correct**, however: the cerebral peduncles are
  only ~10–15 mm wide and sit against an anatomically asymmetric brainstem, so a flat
  cut at the global midline mislabels peduncle voxels that the per-voxel warped-MNI
  mask labels correctly — which is exactly why the peduncle FA moves (below) even
  though the midline's median X is near 0.

  | Peduncle FA (superior 30% brainstem, dilated) | Left | Right | L−R |
  |---|---|---|---|
  | healthy, X=0 split (the bug) | 0.240 | 0.262 | −0.022 |
  | healthy, warped-MNI mask (fix) | 0.250 | 0.248 | +0.002 |
  | ALS, X=0 split (the bug) | 0.201 | 0.207 | −0.006 |
  | ALS, warped-MNI mask (fix) | 0.206 | 0.205 | +0.001 |

  The X=0 split manufactures the "lower-left ~8%" asymmetry on the healthy control
  (the same magnitude and direction as D3's sub-1280 finding, 0.285 vs 0.310); the
  warped-MNI midline makes it symmetric. The ALS (thesis) subject is symmetric under
  every split. **D3's "lower-left peduncle FA is a data property" conclusion is
  revised** — see [[07 Decisions]] D3 and `docs/explanation/design-decisions.md`.
  (D3's actual subject, sub-1280, is no longer on disk, so its exact 0.285/0.310
  cannot be re-derived; the healthy control reproduces the identical artifact pattern.)

  | Jacobian (per hemisphere) | L mean±std | R mean±std |
  |---|---|---|
  | reported (buggy, per-voxel-index) | 1.000±0.388 | 0.999±0.330 |
  | healthy (fix, per-mm + mask) | 0.996±0.204 | 1.005±0.190 |
  | ALS (fix, per-mm + mask) | 1.001±0.192 | 0.998±0.195 |

  The std roughly halves (the voxel-size factor); the L/R symmetry the Jacobian was
  quoted as evidence for survives. **Bundle assignment and final CST counts are
  unaffected** (ROI-based); only the diagnostics move. Regression tests:
  `tests/extract/test_midline.py` (off-centre fixtures — a centred brain cannot detect
  a midline bug).

 `compute_tract_profile`
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

- **Profile figures now label the anatomical regions the report actually tabulates.** The
  profile plots drew `'Pontine Level'`, `'PLIC'` and `'Precentral Gyrus'` at 0 / 50 / 100% of
  tract position, annotating the axis ticks. Read as point landmarks that is defensible — the
  tract does start at the pons and end at the precentral gyrus — and each label did fall
  inside the region it named. But the regional table uses those same three names for bin
  averages over 0-35 / 35-70 / 70-100%, so the figure and the table used one vocabulary for
  two different things, with two of the three labels sitting at the extreme edge of their
  range.

  Region extents now come from a single `TRACT_REGIONS` definition that
  `compute_localized_metrics` bins with and the figures label from, so the numbers and the
  pictures cannot drift apart. Labels sit at the centre of each region, the bands are shaded,
  and the x-ticks mark the region boundaries rather than 0/50/100.

  This is a legibility change: no metric value moves, and no label previously pointed at the
  wrong region. Note the region boundaries themselves remain conventional rather than
  validated against an atlas.

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
