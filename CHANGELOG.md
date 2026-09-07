# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`csttool.viz.layout` — shared millimetre-based figure layout.** The one-page
  report has always been laid out at its final physical size, so 1 Matplotlib
  point printed as 1 point; every other figure was sized in arbitrary inches and
  scaled at include time, which is why the stage QC PNGs set their type at an
  effective 3–6 pt wherever they were placed. The report's machinery is now a
  shared module: `figure_mm` / `axes_mm` (physically sized Figures and Axes),
  `TypeScale` and `KeyMetrics` (frozen type/legend scales), `PanelRowBands` and
  `panel_row_geometry` (the panel-row arithmetic, generalized from the strip's
  fixed four panels and 194 mm to N panels on any width), `draw_key_row` /
  `fit_key_fontsize` / `fit_caption_fontsize` (the per-panel key band), and
  `content_bbox_2d` / `union_bbox` / `square_bbox` / `apply_bbox` /
  `crop_axes_to_content` (cropping to anatomy, preserving the radiological
  x-inversion). **No document dimension lives in the module** — callers pass the
  width their page needs. `metrics.modules.visualizations.qc_strip_geometry` is
  now a thin binding of `panel_row_geometry` to the report's own page, and
  `test_qc_strip_geometry_matches_shared_primitive` pins the two together.

- **`viz.render.render_mask_overlay`** — binary-mask overlay as a translucent
  RGBA fill plus an optional same-hue outline, replacing per-module ad-hoc
  overlays. Fill and outline are independently switchable, so one primitive
  covers filled, contour-only and hybrid representations.

- **`viz.style.save_figure_exact`** — saves a figure at exactly the canvas it was
  laid out at, for figures built with `layout.figure_mm`.

- **`plot_profile_matrix(..., size_mm=, filename=)`** — the report's profile
  matrix can now be typeset on a measure other than the report's content width.
  The figure is otherwise identical, so the same function serves a document with
  a different text width instead of that document scaling the figure down.

- **Registration template provenance in the extraction report.** Which template
  served as the moving image is a licensing fact, not a tuning detail: the
  pipeline prefers FSL-licensed FMRIB58_FA and silently falls back to the
  permissively licensed bundled MNI152 when it has not been fetched. Nothing
  recorded the outcome, so the licence attaching to a published registration
  figure could only be reconstructed from pixel intensities and file mtimes.
  `register_mni_to_subject` now resolves the choice once through
  `describe_registration_template` and returns it as `result['template']` —
  name, modality, tier, licence, source URL, manifest SHA-256, resolved path,
  and the reason the preferred template was declined. It is written to the
  registration report JSON and, via a new `registration=` argument to
  `save_extraction_report`, to `registration.template` in the extraction report
  JSON. Reports written without that argument keep their previous shape.

### Fixed

- **Binary masks rendered through a continuous colormap.** The white-matter mask
  panel and the tensor-maps brain-mask column drew their masks as
  `imshow(mask, cmap='Blues')`. A constant-valued array normalizes to the
  colormap's *low* end, so both overlays rendered near-white while their legend
  swatches showed saturated blue — the legend did not describe the mark. Both now
  draw an explicit RGBA fill at the stated colour and alpha.

- **Registration QC figure misattributed the template it displayed.**
  `plot_registration_comparison` hardcoded `MNI template` in the middle-column
  title, the suptitle and the overlay legend, and took no argument saying what
  had actually been registered. Whenever FMRIB58_FA was used — the default once
  `fetch-data` has run — every QC PNG named the wrong template, and named the
  permissively licensed one in place of the FSL-licensed one. The function now
  takes `template_label` and the pipeline passes the resolved template's display
  name; the default stays `MNI template` for callers that cannot know.

- **The white-matter mask legend omitted the dilation.** The mask on the figure is
  `binary_dilation((FA > tau) & brain_mask, iterations=1)` — the mask that
  actually seeds and stops tracking — but the key read `White Matter (FA > 0.2)`,
  and the voxel count in the title is likewise post-dilation. The key now names
  the dilation. No mask changed.

- **Registration QC column headers named the row, not the column.** They were
  written as `f'{view_name}\n<column>'` on row 0 only, so all three columns were
  headed "Axial" while the row labels down the left correctly read
  Axial/Coronal/Sagittal — the figure read as nine axial panels. Headers now name
  the column ("Subject FA" / "MNI template (warped)" / "Overlay"), and the overlay
  column gained a colour key, which it never had.

### Changed

- **One export path for every figure.** 22 call sites across the four stage
  visualization modules carried their own `dpi=150` and bbox/facecolor arguments,
  so `style.save_figure`'s documented dpi=200 policy described nothing that
  happened and the stage PNGs disagreed with the report figures. All 22 now go
  through `style.save_figure`. **Every stage QC PNG is therefore 4/3 larger in
  each dimension**; the six metrics figures that previously saved without an
  explicit facecolor also gain the policy's white background. Figure content is
  unchanged. Two tests enforce the single path.

- **Along-tract bins are named positionally, not anatomically.** The three bins
  are fixed ranges of a 20-node normalized profile assigned by index; nothing in
  the pipeline verifies them against an anatomical landmark, and their extent in
  millimetres varies between subjects and between hemispheres of one subject.
  "Pontine Level" / "PLIC" / "Precentral Gyrus" asserted a correspondence the
  computation does not establish — a reader could take a value in the "PLIC" row
  as a measurement of the posterior limb of the internal capsule. They are now
  **Inferior / Central / Superior** in the profile figures, the profile-matrix
  legend, the report's regional-metrics table and the node-homology axis label.
  **The JSON keys `pontine` / `plic` / `precentral` are unchanged** — they are the
  schema of every persisted metrics JSON and batch CSV column, and renaming them
  would invalidate existing derivatives for a presentation change.


## [0.6.0] - 2026-08-29

### Added

- **`MaxFraction` in the CST density sidecar.** The extraction stage computed
  the volume's true maximum density fraction and printed it, then discarded it,
  so the one number saying how dense the densest voxel actually got could not be
  recovered from the derivatives. It is now persisted alongside the existing
  `Denominator` / `StreamlineCountLeft` / `StreamlineCountRight` keys. Purely
  additive; the density calculation, its bilateral normalization and the NIfTI
  values are unchanged. It is deliberately **not** the report's colour-scale cap
  — the QC strip saturates at the 99th percentile of non-zero voxels and records
  that separately as `DensityDisplayVmax`.

- **Preprocessing provenance ledger.** The preprocessing report JSON gains
  `schema_version`, a `provenance` block (the same `get_provenance_dict()` output
  the tracking and metrics reports already carry: git commit, Python and dependency
  versions, platform, hardware, thread environment) and `stages` — an **ordered**
  list of what actually happened, in the order it happened. Each entry records the
  stage, who performed it, whether it was requested, its status, method, backend,
  parameters, input and output geometry (shape, zooms, axis codes), gradient-transform
  status, warnings and skip reason. The status vocabulary is closed: `executed`,
  `not_requested`, `failed_continued`, `declared_external` — so "we didn't ask for
  it" is finally distinguishable from "we asked and it failed". Declared external
  correction is *prepended*, because it happened before csttool received the data:
  a thesis-style run serialises as external TOPUP/EDDY → csttool denoise → mask.
  Versions live in the shared provenance block only, never duplicated per stage.
  Purely additive — every previous key is unchanged, and `save_preprocessed` called
  without the new keyword arguments writes exactly the old report.
- **`--input-corrected {unknown,none,topup-eddy,eddy-only,other}` on `run` and
  `preprocess`.** csttool skips its own preprocessing by default, so most runs
  operate on data some other tool has already corrected — and there was no way to
  say so. The flag records what was done to the input *before* csttool received it,
  as a user declaration: `{"declared": …, "verified_by_csttool": false, "source":
  "user-declaration"}`. csttool never inspects the data to confirm it, never infers
  it from BIDS metadata, and never lets it change a processing decision. Default
  `unknown`, which is deliberately distinct from `none` ("nobody said" is not
  "nothing was done"). The declaration lands in the preprocessing report JSON and,
  through the existing metadata channel, in the bilateral metrics JSON.
  Batch inherits it through the documented per-subject manifest
  `"options": {"input_corrected": …}`.
  - Declaring `topup-eddy`/`eddy-only` together with `--perform-motion-correction`
    prints a high-visibility **warning** about likely double motion correction and
    records it in the provenance — it does not block. An unverified declaration
    must not be able to veto an explicit flag.
  - A pass-through run with no declaration now prints a two-line advisory saying
    the results assume externally corrected input. Console only; no file changes.

- **Report enrichment: dispersion, uncertainty, node homology, and a 1×4 QC strip.**
  The one-page A4 report now says more per millimetre, and says what it does not
  know.
  - **Along-tract profiles carry their dispersion.** Each of the four profile
    lines is backed by the per-node **interquartile range across contributing
    streamlines**, with each band's quartile edges stroked in its own hue so the
    blue/orange overlap stays traceable. The centre line remains the **mean** —
    the twelve regional values and their laterality indices are derived from that
    exact array, so a median centre line would silently change every published
    regional metric. Costs 0 mm: the band is inside the existing figure. New
    keys `profile_p25` / `profile_p75` / `profile_n` per scalar per hemisphere.
  - **Bootstrap standard errors** for every headline mean, every regional mean,
    mean length and every laterality index, in `bilateral_metrics.json` and (for
    the headline values) in the CSV. A new `metrics.uncertainty` block records
    the method, resample count, seed and — mandatorily — the **scope**: these
    SEs are conditional on the retained bundle and do *not* include tracking,
    seeding, registration, preprocessing or acquisition variability. Streamline
    count and tract volume deliberately carry no SE. Not shown in the report
    tables (a deliberate scope decision; the data is serialised so the display
    can be added later without recomputation).
  - **Node-homology statement.** The regional table compares node *i* on each
    side; the measured L–R node offset and mean length difference now sit on its
    caption row, with the full per-node arrays in `metrics.node_homology`.
    Descriptive only: no threshold, no PASS/FAIL, no colour, no suppression of
    any regional value, and **no flag key in the schema** for a downstream
    consumer to depend on.
  - **The 1×3 tractography triptych is replaced by a 1×4 QC strip**
    (`plot_report_qc_strip`, 194 × 44 mm): DEC-FA, CST density, extraction ROIs
    and final CST over FA, on **one** shared coronal slice, with one legend, one
    caption and one colourbar. The triptych answered a single question
    (streamlines over FA) from three angles; the strip answers four different
    ones in the same page region. Every missing input degrades its own panel to a
    labelled grayscale-FA slot — the strip never reflows to three, so a reader
    can see which question went unanswered.
  - **New product `*_space-orig_desc-CSTroi_dseg.nii.gz`** — the three warped
    extraction ROIs as a `uint8` label map (1 brainstem, 2 motor-left, 3
    motor-right) on the FA grid, with `Labels` / `Space` / `VoxelCounts` in its
    sidecar. Reaches BIDS `dwi/`, and is exposed to `csttool metrics` via a new
    `--roi-dseg` flag with a sibling-glob fallback.

  Four inherited defects are fixed rather than carried forward:
  1. **The report QC panels now genuinely share one slice.** The standalone
     panels each select their own, and two of them were called with
     `density=None` and so fell through to the level-4 anatomical fallback while
     a third got level 1 — the shared-slice guarantee held only inside the
     prototype driver. The strip is now the one place it actually holds; the
     standalone panels keep their own per-call selection.
  2. **ROI masks reach a consumer.** They were written into `extraction/nifti/`,
     which `csttool run` deletes wholesale, so nothing downstream could read
     them. The `dseg` product is rescued to BIDS `dwi/` before that cleanup,
     alongside the V1 and density products.
  3. **The strip's streamline subsample is deterministic.** It is seeded from
     `DEFAULT_SEED`, not `viz.utils.viz_rng`, whose `VIZ_SEED` derives from
     Python's builtin `hash()` of a string and is randomised per process unless
     `PYTHONHASHSEED` is set. (The global `VIZ_SEED` defect and its remaining
     consumers are unchanged and still documented.)
  4. **`render_streamline_overlay` drew in the wrong frame.** It decided slab
     membership in world millimetres — correctly — but then plotted the *world*
     points onto an `imshow` whose data coordinates are voxel indices, putting
     the bundle far off the anatomy and autoscaling the axes out to contain it.
     Points are now converted to voxel coordinates before plotting. This also
     corrects the standalone CST-over-FA panel.

  Two further rendering corrections found by the print review:
  `render_density_overlay`'s `imshow` reset the axes limits and silently undid
  the background's radiological x-inversion, so the density panel was mirrored
  relative to its neighbours with its R/L markers on the wrong sides; and
  `add_direction_legend` placed its labels beyond the arrow tips, where at report
  size they overflowed the glyph box and printed over the anatomy.

  **Page budget.** Measured with WeasyPrint at every step. The report uses
  **271.4 mm** of the 281 mm printable height (the layout test asserts ≤ 275 mm),
  verified as exactly one A4 page across all six review cases — nominal, oblique,
  V1 withheld, density withheld, dseg withheld, and one empty hemisphere. Getting
  there needed the profile matrix at 94 mm rather than 97, and the node-homology
  statement on the regional table's existing caption row rather than in a block
  of its own. The fourth documented mitigation — cropping the strip's field of
  view to the brain-mask bounding box — was measured and **rejected**: the brain
  fills the full superior-inferior extent, so cropping makes the panels taller,
  not shorter.

  Two DEC code paths now exist: the world-frame one in the report strip, and the
  legacy voxel-frame one in the tracking stage's developer QC figure. The latter
  should be migrated to the world-frame product in a follow-up.
  `plot_tractogram_qc_triptych` and `compute_tract_profile` are unchanged,
  exported and still tested; only the report path moved off them.

- **Visualization refactor: new scientific data products and standalone QC panels.**
  Two new **unconditional** numerical NIfTI data products are now written by the
  producing pipeline stages and persist to BIDS derivatives, separate from any
  rendering:
  - `*_desc-V1_dwimap.nii.gz` — the principal diffusion eigenvector (V1) rotated
    into the anatomical **world (RAS+) frame** by the orthonormal polar factor of
    the affine (`csttool/spatial.py`, a new dependency-free leaf module). Stored as
    a 5-D `(X,Y,Z,1,3)` float32 volume with `NIFTI_INTENT_VECTOR` and a
    self-describing sidecar (`VectorFrame`, `AffineDeterminant`, `ObliquityRad`,
    `ShearMagnitude`). The world rotation is the scientifically correct DEC basis
    for oblique and LAS/LPS acquisitions where the voxel frame differs from the
    anatomical frame (plain `color_fa` is wrong there). Written whenever `track`
    runs.
  - `*_desc-CSTdensity_dwimap.nii.gz` — the fraction of distinct retained
    bilateral CST streamlines visiting each voxel at least once
    (`dipy.tracking.utils.density_map` + csttool's denominator, grid guarantee and
    step-size gap guard; `csttool/extract/modules/density.py`). Range `[0,1]`;
    sidecar records the exact definition, the denominator and per-hemisphere
    counts. Written whenever `extract` runs.
  Three **standalone prototype** QC panels replace the single-question 1×3
  tractography triptych with three complementary questions, sharing one
  data-driven coronal slice (documented fallback chain) and rendered via a new
  composable renderer layer (`csttool/viz/render.py`) that never creates a Figure:
  - `_dec_fa.png` (`stage-tracking_qc-decfa`) — DEC-FA: does the diffusion field
    support the anatomy?
  - `_cst_density.png` (`stage-extraction_qc-density`) — is the bundle coherent
    and symmetric?
  - `_cst_over_fa.png` (`stage-metrics_qc-cstoverfa`) — does the CST follow the
    expected course (physical-millimetre slab, contiguous-run polylines)?
  These prototypes are **not** embedded in the PDF report — the legacy triptych
  remains the report QC figure unchanged until independent scientific review passes
  (see `../csttool-devlog/report-improvement/visualization-refactoring-plan.md` §2.11/§14).

- **Six trust-chain QC panels and their diagnostics module.** The existing
  figures answer "what did the pipeline do?"; every scientific figure plotted a
  point estimate (a mean line, a bar, a laterality index) with no statement of
  its support. A new `csttool/metrics/modules/qc_stats.py` computes the missing
  diagnostics and `qc_figures.py` composes them into four standalone panels
  (`--save-visualizations` on `csttool metrics`), each answering exactly one
  question about whether the published numbers are believable:
  - `_qc_tissue_plausibility.png` (`stage-metrics_qc-tissue`) — the joint FA-MD
    distribution of every voxel the CST visits, each weighted by its CST
    density, with the free-water corner (FA < 0.2 **and** MD > 2.0e-3 mm²/s)
    outlined and its mass fraction reported per hemisphere. The only panel that
    questions the *input* to the metrics rather than the tractography: a bundle
    can be anatomically perfect and still report an elevated MD because part of
    its mass sits in voxels adjacent to the ventricles.
  - `_qc_v1_angle.png` (`stage-metrics_qc-v1angle`) — the acute angle between
    each streamline's tangent and the local world-frame V1, along the tract. A
    bulge is the CST/CC/SLF crossing region announcing itself, which identifies
    the profile nodes least supported by the tensor. Validated end-to-end on an
    oblique subject (0.496 rad), where a voxel-frame V1 would show a systematic
    offset and the world-frame product does not.
  - `_qc_profile_dispersion_<scalar>.png` (`qc-dispersion-<scalar>`) — per-node
    median, IQR and 5–95 band across streamlines. `compute_tract_profile` already
    built this matrix and discarded it at its `np.mean`; `qc_stats.profile_matrix`
    returns it, and `matrix.mean(axis=0)` reproduces that function exactly, so
    the band describes the published mean rather than a near neighbour of it.
  - `_qc_sampling_saturation.png` (`qc-saturation`) — the headline
    length-unbiased mean recomputed over subsamples of the bundle, plus the
    bootstrap standard error at full N (which the subsampling curve cannot show,
    having zero variance at 100 % by construction). Seeded from `DEFAULT_SEED`,
    not `viz.utils.VIZ_SEED` — the latter derives from Python's builtin `hash` of
    a string and is randomised per process unless `PYTHONHASHSEED` is set.
  - `_qc_node_homology.png` (`qc-nodehomology`) — the world Z and arc length of
    each profile node per hemisphere, plus the two length distributions. The
    20-node parameterisation is *relative*, so unequal bundle lengths mean node
    *i* sits at a different anatomical level on each side and every regional
    laterality index is confounded by geometry rather than microstructure.
    Nothing else in the figure set checks this.
  - `_qc_profile_attrition.png` (`qc-attrition`) — the four gates a streamline
    passes between extraction and the published profile, per hemisphere, plus
    the per-point retention. `compute_tract_profile` drops short streamlines and
    out-of-bounds points silently; `qc_stats.attrition_funnel` makes the discards
    auditable. The failure it exists to catch is *asymmetric* attrition, which
    would leave the two hemispheres' regional metrics computed on different
    populations. On a well-formed run the answer is "none" — the panel states
    that verdict in words so a flat funnel reads as a pass.
  All six sample exactly as the metrics they audit do, and are standalone — not
  embedded in the PDF report.

- **`csttool metrics --v1` and `--density`.** `--v1` points the V1-angle panel
  at the world-frame eigenvector map; when omitted it is looked for beside the
  FA map, where `save_tracking_outputs` writes it. `--density` points the
  tissue-plausibility panel at the CST density product; when omitted it is looked
  for in the extraction stage's `scalar_maps/` beside the tractograms. `csttool
  run` plumbs both through — the tracking result dict now carries `v1_path` and
  each extraction result dict carries `density_path`.

- **`../csttool-devlog/report-improvement/generate_qc_review.py` and `qc-figure-review.md`.** A
  driver that regenerates every QC panel for the validation subjects into a
  stable, versioned review location (`../csttool-devlog/report-improvement/qc-review/`, previous
  sets archived under `_previous/`), and the scientific review of the six
  trust-chain panels that decides which of them are worth keeping. Findings:
  QC-8 (node homology) and QC-5 (dispersion) earn standalone-QC status; QC-6
  (saturation) is an unbiased estimator plotted against sample size and can
  never look different across subjects — its only content is the bootstrap SE,
  which belongs in the metrics JSON rather than in a 140 mm figure; QC-7 is a
  check that happens to have a figure. Report integration is deferred.

- **Two new CLI flags.** `--save-visualizations` is added to `csttool track` and
  `csttool metrics` (it already existed on `preprocess`, `extract`, and `run`),
  making the DEC-FA and CST-over-FA panels reachable as documented CLI
  invocations and closing the parser inconsistency the figure inventory flagged.

- **Redesigned one-page A4 PDF report.** The clinical report is restructured to
  the approved layout: a compact header with a laterality legend and the LI
  formula, a three-column methods band (Acquisition / Processing / Space &
  Orientation) with terminology verified against the implementation (DTI scalar
  model, CSA ODF direction model, deterministic LocalTracking, FA-mask seeding),
  titled global and regional tables with consistent units/precision and
  blue/orange LI colour semantics (no red), a large 2×2 along-tract profile
  matrix (FA/MD/RD/AD) with a single shared legend and shared bottom x-axis, a
  compact 1×3 tractography-QC triptych (sagittal/coronal/axial) sharing one
  grayscale FA scale and one colorbar, and a readable reproducibility footer
  (hardware retained; git commit stays in JSON only). The orientation code is
  derived dynamically from the FA affine (`viz.geometry.orientation_code`),
  never hardcoded. Report formatting helpers are centralized and unit-tested
  (`reports.format_*`, `build_report_context`), and the two report-generation
  paths (library `generate_complete_report` and CLI `cmd_metrics`) now share one
  composite-figure pipeline. The obsolete "Metrics Extracted In" badge is
  removed.

- **`median_length` morphology statistic (additive).**
  `compute_morphology` now also returns `median_length` (`float(np.median(lengths))`;
  `0.0` for the empty case). Additive only: the length laterality index still
  uses `mean_length` (AU10), so no existing metric value changes. The report's
  global Length row now shows a genuine `median (min–max)` cell; legacy
  morphology without `median_length` renders an em dash rather than
  substituting the mean (a mean is never displayed as a median). New CSV columns
  `left_median_length_mm` / `right_median_length_mm`; the JSON `metrics.*.morphology`
  block gains the key. See `docs/explanation/design-decisions.md`.

- **Unmocked end-to-end test of the extraction scientific core (AU28).**
  `tests/integration/test_pipeline.py` patched
  `register_mni_to_subject`, `load_mni_template`, `fetch_harvard_oxford`,
  `extract_cst_passthrough` and `validate_tractogram_coordinates` with
  `MagicMock`, so registration, warping, ROI construction and filtering had no
  automated end-to-end coverage. New `tests/integration/test_unmocked_extract_core.py`
  runs the *real* implementations of all four stages on a small, fully-synthetic,
  self-contained scene (a 40³ MNI template + synthetic Harvard-Oxford-style atlas
  shipped as session fixtures in `tests/conftest.py`, so CI never depends on the
  FSL-licensed Tier-2 `fetch-data` atlases). It asserts the four planted CST
  streamlines are recovered and split by the correct hemisphere, the junk
  streamline is excluded, and `validate_tractogram_coordinates` passes on the
  synthetic tractogram — one of the five sites AU28 named as mocked. The only
  piece still substituted is `fetch_harvard_oxford`, which fetches a license-gated
  atlas that cannot be assumed present in CI.

- **Automated tests of registration quality / ROI placement / atlas-warp label
  preservation (AU33).** The QC in `warp_atlas_to_subject` (label-count change,
  motor-centroid side, motor Z-difference) only `print()`ed and
  `verify_atlas_labels` returned a dict no test asserted on, so there was no
  automated test of registration quality, ROI placement, or atlas-warp label
  preservation. New `tests/extract/test_registration_quality.py` asserts on the
  real registration+warp of the synthetic scene: the planted +6 mm translation is
  recovered (warped atlas motor centroids land at MNI+DX), the per-hemisphere
  Jacobian mean is ~1.0 with no folding, the warped-MNI midline is balanced and
  shifted off-centre, atlas labels survive warping (set unchanged, no spurious
  labels), and the motor L/R centroids straddle the midline and sit in the same
  axial plane. Direct unit tests of the extracted QC helper cover the
  previously-impossible label-loss and hemisphere-swap detector paths.


- **Formal edge-case suite (AU31).** A new `tests/edge_cases/` package
  characterises the six pathological-input classes the three audits named —
  zero streamlines, all-zero DWI, single direction, misordered bvec/bval,
  truncated NIfTI, DICOM missing tags — pinning current behaviour so a
  regression is caught. Targeted hardening accompanies the suite: a truncated
  NIfTI now raises a clear `NIfTI file appears truncated or corrupt` error at
  the load boundary (instead of a raw `EOFError` deep in a stage); an all-zero
  DWI now warns `No white-matter voxels found` (instead of a silent empty
  output). The remaining classes already behaved correctly (misordered
  gradients via AU21; zero-streamline extraction returns a well-formed empty
  result; missing DICOM tags default and classify as unsuitable) and are now
  pinned. 16 new tests.

- **`--fit-method`** on `csttool track` and `csttool run` — exposes the DTI tensor fit
  method (`OLS`, `WLS`, `NLLS`, `RT`) as an explicit parameter, pinned to `WLS` by
  default. Previously `fit_tensors.py` and the extraction modules relied on DIPY's
  version-dependent default. The parameter sweep can now include fit method, which
  is a first-order determinant of FA/MD bias at low SNR. (AU13)

- **`--npeaks`** on `csttool run` — pins the number of ODF peaks extracted per voxel
  in the roi-seeded and bidirectional tracking paths, defaulting to `1` (single
  principal direction, matching the whole-brain `estimate_directions` default).
  Previously these paths inherited DIPY's version-dependent default. (AU17)

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

- **Removed the unused `qc_has_colorbar` report-context key.** It was built and
  handed to the Jinja template on every render and the template never read it.
  A regression test now asserts every key the report context supplies is
  actually referenced by the template.

- **Report QC strip: the CST density panel now says what it measures and where
  its scale saturates.** The colourbar was labelled `streamline fraction`, which
  could equally have meant a fraction of one streamline, of one hemisphere's
  streamlines, or of every streamline generated. It is the fraction of the
  *bilateral* retained population (`n_left + n_right`), and that denominator is
  the one thing a reader cannot guess — it is also what sets the ceiling, since
  a voxel every left streamline visits reads `n_left / n_total`, about 43% on
  the validation subject, not 100%. The label now reads **Fraction of bilateral
  CST streamlines**.

  Endpoints are shown as percentages (`0%`, `≥7.4%`) rather than raw fractions,
  a display change only — the stored volume remains a float in `[0, 1]` and the
  metric is untouched. The upper endpoint carries `≥` whenever the scale
  saturates below the data's true maximum, which it usually does: the scale top
  is the 99th percentile of non-zero voxels, and on the validation subject the
  true maximum is 1.9x that, with 71 voxels sharing the top colour. Labelling
  that endpoint as a bare number named it as the maximum, which it is not. Where
  the cap happens to reach the maximum, no `≥` is claimed. The shared caption
  gains `density scale capped at non-zero P99`, stating how the scale top was
  chosen whenever a density panel is drawn.

  The figure sidecar's `DensityVmax` held the display cap under a name that
  reads like a data maximum. It is replaced by `DensityMaxFraction` (the
  volume's real maximum), `DensityDisplayVmax`, `DensityDisplayPercentile`,
  `DensityDisplayPercentileBasis` and `DensityDisplayClipped`, and the
  visualization code names the two apart throughout. **The density metric itself
  is unchanged** — same bilateral denominator, same DIPY `density_map`
  numerator, same densification guard — and a new test pins it against a silent
  switch to side-normalized density.

  Also: the colourbar's end ticks keep a margin from their column edge (solving
  the bar width against the ticks previously landed the wider one exactly on the
  boundary), and the shared caption is measured and shrunk rather than clipped
  if the slice, rule, slab and density note together outgrow the strip.

- **Report: the `FA background` subtitle is removed from the Tractography QC
  heading.** It was true of panels 2-4 only — panel 1 is the DEC image itself,
  not an overlay on FA — and each panel now names its own content in its own
  key, so it was an orphaned and partly false annotation.

- **Report QC strip: panel-level annotation is now owned by the panel it
  describes, and the layout is derived rather than allocated.** The strip's four
  panels each got a fixed rectangle, which `imshow`'s `aspect='equal'` then
  shrank and re-centred at draw time (Matplotlib's default `adjustable='box'`).
  Annotations anchored to the Axes moved with it; annotations anchored to the
  precomputed millimetre bands did not. The result depended on the subject's
  acquisition matrix: the panel titles landed at 44.13 mm on a 24×20×18 grid
  (clipped by the exact-bbox save) and at 40.58 mm on a 128×128×76 one, leaving
  ~5.1 mm of the 44 mm canvas as dead white in the wrong places. Two furniture
  bands were also smaller than the type they held — the 2.2 mm colourbar-label
  band held a label needing 2.77 mm, so it overprinted the shared caption by a
  measured 0.46 mm in every report.

  New `qc_strip_geometry()` computes the image box from the displayed slice's
  aspect, so `apply_aspect` is a no-op and the furniture bands hold what they
  were measured for. Panel width is constant and a tall grid letterboxes instead
  of narrowing the panel, because the key band underneath is width-critical.
  Figure height is now derived and bounded by `QC_STRIP_MIN/MAX_HEIGHT_MM`;
  `QC_STRIP_SIZE_MM` remains as the nominal size and the CSS sets width only.
  Measured page use: 273.8 mm of 281, against a 6 mm headroom requirement.

  Each panel now carries its own key in its own column: the DEC axis key (moved
  out of the image, where it was a 6.1 mm opaque inset over ~3.5% of panel 1),
  the density colourbar with its `vmax` as a flanking tick rather than a clause
  inside a 47.45 mm sentence, the ROI colour key (greyed, not dropped, when the
  display slab never reaches an ROI), and the hemisphere counts. The shared
  caption keeps only what is true of all four panels — plane, selection rule,
  slab and orientation convention. Panel titles rise to 9 pt, matching the
  profile matrix's subplot titles, and never report status: a degraded panel
  keeps its name and states the reason in its in-panel note. No scientific
  content changed — same slice selection, same FA-grid guard, same degradation
  behaviour, same sidecar keys plus `FigureSizeMm` and `PanelAspect`.

  Every panel is now the same three zones in the same order — title, image,
  fixed-height key — with the two separations that were previously missing made
  into real bands: `_STRIP_IMAGE_KEY_GAP_MM` between an image and its own key,
  and a deliberately larger `_STRIP_KEY_CAPTION_GAP_MM` above the shared
  caption, so the caption reads as a caption for the composite rather than a
  fifth legend under panel 4. Key labels are set on one shared typographic
  baseline (`va='baseline'`); centring each string's bounding box instead put a
  row of all-caps labels 0.127 mm off a row with ascenders. The density
  colourbar is thicker, its end ticks straddle the bar rather than sitting on a
  third line, and its width is *measured* from those ticks — a centred bar gives
  each tick half the leftover room, so the binding constraint is twice the wider
  tick, not their sum. Panel gutters widen 1.5 → 2.6 mm.

  The extra whitespace is paid for in image scale, not page height: the page had
  1.2 mm of headroom left, so `_STRIP_IMAGE_MAX_MM` drops 30.0 → 25.5 mm and the
  typical panel now letterboxes by ~2.9 mm a side. The letterbox is black
  against a slice whose own margins are black, so it costs scale and nothing
  else. Page use 273.8 → 274.3 mm of 281, against the 6 mm headroom the layout
  test demands.

- **Metadata-schema correction: the pass-through preprocessing status no longer
  asserts external preprocessing.** `pipeline_metadata['preprocessing']['status']`
  read `"Skipped (External Preprocessing Used)"` on every default run — stating as
  fact something nobody had declared and csttool cannot check. The status now says
  only what csttool did (`"Skipped"`, plus `performed_by_csttool: false`), and what
  happened to the input beforehand lives in the new, explicitly declared
  `external_correction` block. The two facts are separate because they have
  different epistemic status. The old string is not retained anywhere: keeping a
  misleading claim alongside the honest one would just be a contradiction with two
  spellings. Consumers should read `status` plus `external_correction.declared`.

- **`warp_atlas_to_subject` now returns an assertable QC dict (AU33).** The
  atlas-warp QC checks (label-set preservation, motor-centroid world
  coordinates and side-of-midline / Z-difference flags) were `print()`-only, so a
  registration that silently dropped a label or swapped a hemisphere would warn
  to stdout and pass. `warp_atlas_to_subject` now returns `(warped_atlas, qc)`
  and `warp_harvard_oxford_to_subject` exposes them as `cortical_qc` /
  `subcortical_qc`. The QC logic is extracted into a pure, testable helper
  `compute_atlas_warp_qc` so callers and tests can assert on it without
  re-deriving it. Label-preservation now compares the full label *set* (a
  resample can split/merge labels while keeping the count), and the original
  label set is recomputed on the resampled grid so a Harvard-Oxford→MNI-grid
  resample is not misreported as a label change. No algorithm or metric value
  changes; verbose printing is preserved.


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

- **Tensor `fit_method` and ODF `npeaks` are now explicit, pinned parameters** in every
  code path that creates a `TensorModel` or calls `peaks_from_model`. Previously these
  inherited DIPY's version-dependent defaults, which could change between DIPY releases
  and silently shift FA/MD values and tracking behaviour. The defaults (`fit_method='WLS'`,
  `npeaks=1`) match the current DIPY 1.9+ behaviour and preserve backward compatibility
  with all previously reported numbers. See `--fit-method` and `--npeaks` under Added.
  (AU13, AU17)

- **Motor cortex ROIs are now clamped at the anatomical midline** using the warped-MNI
  `hemisphere_mask` from AU11. Previously, dilated Harvard-Oxford motor labels could
  bleed across the midline (documented: up to 9.5 mm into the contralateral hemisphere)
  and the mutual-exclusivity filter only caught streamlines hitting *both* ROIs — a bled
  ROI silently accepted wrong-hemisphere streamlines and attributed contralateral FA
  values to the wrong side. The bidirectional method corrects the count artifact
  but not this ROI-placement artifact on FA sampling. Clamping is applied after
  dilation in `create_cst_roi_masks`; when `hemisphere_mask` is unavailable the scalar
  `midline_x` is used as a fallback. Both are already carried by the `warped` dict.
  (AU12)

- **Removed `assess_clinical_significance` and `compute_effect_size`** from the public
  API. Both were dead code (never called from any pipeline path) but were exported in
  `metrics.__all__`. `compute_effect_size` fabricated a pooled standard deviation by
  assuming a 10% coefficient of variation, then divided a single-subject L−R difference
  by it and called the result "Cohen's d" — with N=1 per side no effect size is defined.
  The interpretation strings from `assess_clinical_significance` did reach the JSON
  report via `save_json_report`, so downstream JSON consumers may notice their absence.
  (AU15)

### Fixed

- **Sub-seeds derived from Python's builtin `hash()` were randomised per process,
  making every subsampled QC figure irreproducible.** ⚠️ **Figure-changing; no
  scientific output affected.** `viz.utils.VIZ_SEED`, `RunContext.rng_viz` and
  `RunContext.rng_perturb` all derived their sub-seed as
  `hash(f"{seed}:label") & 0xFFFFFFFF`. Python salts `hash()` of a `str` per process
  (PEP 456) unless `PYTHONHASHSEED` is set, so each took a *new value on every
  invocation*. `deterministic_subsample` consequently drew a different subset of
  streamlines on each run, despite its name. All three now derive through
  `reproducibility.context.derive_seed()`, which uses BLAKE2b and is stable across
  processes, platforms and Python versions.
  - **Scope: rendering only.** A determinism run over four conditions (DICOM and
    NIfTI, single- and 4-threaded, with and without motion correction) confirmed
    every `.nii.gz`, `.trk`, `.bval` and `.bvec` was **bitwise identical** between
    repeat runs, and every metric in `*_metrics.json` was unchanged. The split was
    exact: all 17 figures that render their full input matched byte-for-byte, and
    all 11 that subsample differed. Because these figures save with a tight bounding
    box, a different subset changes the axis extent and hence the canvas size, which
    is why some diffs covered ~18 % of pixels rather than a few strokes.
  - Two call sites had already worked around this locally by seeding from
    `DEFAULT_SEED` (the sampling-saturation panel and the CST-over-FA panel); their
    behaviour is unchanged, and the docstrings explaining *why* have been corrected
    now that the root cause is gone.
- **The whole-brain tractography QC panel drew from the unseeded global NumPy RNG.**
  `run_tractography()` selected its 5 000 display streamlines with
  `np.random.choice`, which is neither seeded nor isolated — it varied between runs
  and perturbed the global stream for anything drawing after it. Now uses `viz_rng()`
  and sorts the indices.
- **`mean_closest_distance()` returned different MDF values for identical inputs.**
  ⚠️ **Output-changing for `csttool validate`.** When a bundle exceeded
  `num_samples` (default 1 000) it was subsampled with the unseeded global
  `np.random.choice`, so the reported `mdf_symmetric`, `mdf_cand_to_ref`,
  `mdf_ref_to_cand` and `mdf_std` moved between runs on the same tractograms. The
  draw is now seeded from a new `seed` parameter (default `DEFAULT_SEED`) via
  `derive_seed()`. Unlike the figure defects above, this one affected *numbers*.
- **Regression coverage.** `tests/reproducibility/test_seed_derivation.py` asserts
  cross-process stability by spawning fresh interpreters under differing
  `PYTHONHASHSEED` values, including a guard test that fails if `hash()` ever stops
  being salted. In-process assertions cannot catch this class of defect — the salt is
  fixed within a process, which is why the original bug survived the existing suite.


- **The dicom2nifti fallback silently mirrored the b-vectors on Siemens data.** ⚠️
  **Output-changing; affects every run imported without `dcm2niix` on PATH.**
  `dicom2nifti`'s Siemens path projects the gradient direction onto the image axes as
  `(bvec·read, −bvec·phase, bvec·slice)` — negating the phase axis — while the affine
  it writes uses `(read, phase, slice)` unflipped. Its b-vectors are therefore
  reflected relative to the voxel frame of its own NIfTI. csttool now negates the
  phase component back before reorienting to RAS.
  - **Nothing could see it.** A reflection of the gradient table leaves FA and MD
    exactly invariant (`D → Q D Qᵀ` preserves eigenvalues), so the scalar maps, the
    unit-norm/count/b0 validators, and per-axis `|V1|` summaries were all bit-identical
    with the bug present. Only the *signed* direction field was wrong. Measured on a
    71-direction Siemens acquisition: whole-brain mean streamline length 23.4 mm and
    CST extraction of 8 left / 1 right, versus 45.3 mm and 637 / 649 once corrected —
    the latter matching the `dcm2niix` reference exactly.
  - **Other vendors are deliberately not "fixed".** dicom2nifti's GE path reads
    patient-frame private tags with an *x* inversion and Philips has its own path
    again; neither matches the Siemens convention and neither is validated here.
    Their gradients are passed through unchanged with an explicit warning rather
    than guessed at. Install `dcm2niix` for those vendors.
  - Any tractography produced through the dicom2nifti fallback on Siemens data
    should be regenerated.
- **The motion-correction QC figure reported nonsense.** `dipy.align.motion_correction`
  returns a `(4, 4, n_volumes)` array, but `plot_motion_correction_summary` called
  `len()` on it (→ 4) and iterated the first axis, so each "affine" was a `(4, n)`
  slice and every plotted translation and rotation was garbage. A 71-volume run was
  titled "4 volumes" and reported a 284° maximum rotation where the true value was
  0.47°. It now normalises the stack, derives angles from the orthonormal rotation
  component, and reports the same geodesic `max_rotation_deg` the preprocessing report
  records — so the figure and the ledger can no longer disagree.

- **`--perform-motion-correction` shipped unrotated b-vectors.** ⚠️ **Output-changing,
  and previous motion-corrected outputs were wrong.** Motion correction resampled
  every DWI volume onto the reference pose but the original `.bvec` was copied
  through unchanged, so the data and its gradient table described different
  anatomies. Fitting a tensor to that pair biases FA/MD and tilts the principal
  eigenvector — silently, since nothing about the files looks malformed. The
  per-volume transforms estimated by the registration are now applied to the
  corresponding b-vectors (Leemans & Jones 2009) and the rotated `.bvec` is written
  next to the preprocessed NIfTI, where tractography picks it up automatically.
  New module `preprocess/modules/reorient_gradients.py`.
  - The rotation direction comes from DIPY's own `reorient_bvecs` primitive, not
    from re-derived mathematics; the world→voxel frame conversion uses the image
    affine's **orthonormal orientation** (voxel scaling deliberately excluded — a
    b-vector is a physical direction, not an index-space displacement). A test
    asserts the result is invariant to voxel size, and another pins the
    sign-flipped (LAS) case where a naive rotation would go the wrong way.
  - **Scope is unchanged and stated plainly**: affine, between-volume motion
    correction. No eddy-current model, no outlier replacement, no slice-to-volume
    estimation, no susceptibility-distortion correction.
  - If anything in the correction step fails, the motion-corrected data is
    discarded along with it, so what ships is always a mutually consistent
    data/gradient pair.
  - **Anyone who has used this flag should re-run**: the tensors, tractograms and
    metrics derived from those outputs were computed against mismatched gradients.
- **A failed motion correction was only visible in the output filename.** The
  preprocessing report now records `motion_correction_requested` alongside
  `motion_correction`, plus `bvecs_rotated`, `max_rotation_deg` and a `warnings`
  list, so requested-but-failed is distinguishable from never-requested without
  parsing the `_mc`/`_nomc` suffix.

- **`--b0-threshold` was parsed and thrown away.** The flag is advertised on
  `preprocess`, `track` and `run`, but only `check-dataset` ever read it — every
  gradient-table construction site used the hard-coded default instead, so a
  non-default value silently did nothing on the three commands that matter. The
  parsed value now reaches `load_dataset` → `validate_gradient_table`,
  `get_gtab_for_preproc`, Patch2Self, and (via `run`) both the preprocess and track
  sub-namespaces, so one execution has exactly one b0 threshold. The default is
  unchanged at 50 s/mm², so runs that do not pass the flag are unaffected.
- **Brain masking re-derived its own b0 set.** `background_segmentation` thresholded
  `gtab.bvals` against the module-level constant with a strict `<`, while the
  gradient table was built with `<=` and the execution's threshold. It now reads
  `gtab.b0s_mask`, so masking and the tensor fit cannot disagree about which volumes
  are b0. Affects only runs passing a non-default `--b0-threshold` — at the default,
  `gradient_table` rewrites sub-threshold b-values to 0 and the two rules happened to
  coincide.

- **`csttool batch` never actually preprocessed anything.** ⚠️ **Output-changing.**
  `BatchConfig.preprocessing` (documented default: enabled) was flattened onto the
  worker's argument namespace under the name `preprocessing`, but `cmd_run` reads
  `args.preprocess`. Nothing bridged the two, so every batch subject took the
  pass-through branch and was tracked on raw, un-denoised, un-masked data — silently,
  and regardless of `--preprocessing`/`--no-preprocessing`. The namespace construction
  is now a testable helper, `_build_run_namespace`, which translates the name at the
  worker boundary. The dataclass field keeps its name on purpose: `compute_config_hash`
  hashes `asdict(config)`, so renaming it would invalidate every existing `_done.json`
  resume marker. Config hashes are unchanged (pinned by a test).
  **Batch outputs will differ from previous runs**: subjects now really are denoised
  and skull-stripped before tractography, which is what the docs have always claimed.
  Pass `--no-preprocessing` to keep the old effective behaviour explicitly.
- **`csttool batch --denoise-method none` is no longer offered.** The choice existed
  in the batch parser but `denoise()` has never implemented it (it raises
  `ValueError`). While batch preprocessing was broken the branch was unreachable;
  with the fix above it would have failed every subject that selected it. Removed
  rather than implemented — an "input is already denoised" mode is a feature, not a
  repair. Scripts passing it now get an argparse error instead of silently doing
  nothing.
- **Stale `BatchConfig.denoise_method` default.** The dataclass still defaulted to
  `"nlmeans"` after the shared default moved to `mppca`. Unreachable through the CLI
  (`cmd_batch` always sets it) but a trap for library callers constructing
  `BatchConfig` directly; it now reads `DEFAULT_DENOISE_METHOD`.

- **One-page A4 report: the redesign actually fits now.** The redesigned report
  rendered on two pages (the QC triptych and the reproducibility footer spilled
  onto page 2) while the page div hid the overflow with a fixed height plus
  `overflow: hidden`. Root cause of the budget error: both composite figures
  were saved through the house-style `savefig.bbox="tight"` policy, so the PNGs
  were cropped-and-padded to an unpredictable aspect and printed ~5 mm taller
  than designed; on top of that the methods band's CSS-grid `<dl>` laid out at
  ~41 mm instead of the budgeted ~16 mm. The two report figures are now saved at
  their exact figure bbox (`PROFILE_MATRIX_SIZE_MM`, `QC_TRIPTYCH_SIZE_MM` are
  the single source of truth, asserted against the CSS widths by a test), the
  methods band uses flex rows and is content-sized, section gaps and table row
  padding are tightened, and the fixed height / `overflow: hidden` are gone so a
  regression shows up as a second page instead of silently clipped content. The
  page-count test is joined by a headroom test and a pathological-content case
  (very long subject ID, CPU, platform and dependency strings).

- **Duplicate and clipped along-tract profile legend.** The report drew a
  template-level legend in the profiles section *and* the legend inside the
  Matplotlib figure; the template copy was clipped. The template legend is
  removed — the figure owns the single legend (Left CST / Right CST / PLIC
  region), because only it can be positioned against the subplot geometry.

- **Overlapping along-tract axis and landmark labels.** "Pontine Level", "PLIC"
  and "Precentral Gyrus" were drawn per bottom-row panel in 9 pt italics, larger
  than the 9 pt subplot titles they sat under, and collided with the x tick
  labels. They now live in one shared anatomical-region strip spanning both
  columns below the grid, in regular sans at 6.5 pt (smaller than the titles),
  centred over their own `TRACT_REGIONS` intervals, alongside a clean numeric
  tract-position axis. Printed type sizes are named constants
  (`_RPT_TITLE_PT` … `_RPT_REGION_PT`) and a test asserts region labels stay
  smaller than subplot titles and are drawn exactly once each.

- **QC triptych geometry: unequal panels and an image/colorbar collision.** The
  sagittal, coronal and axial slices have different source shapes (e.g. 72×96 vs
  96×96), so the three panels rendered at different apparent sizes, and the
  colorbar — created with `ax=[...]`, stealing space from the shared axes area —
  landed on top of the axial panel. Each view is now letterboxed onto a common
  data canvas (the union of the three extents, centred), giving all three panels
  the same physical box and the same scale with no anatomical distortion, and
  the colorbar has its own GridSpec column snapped to the image row's real
  vertical extent. All three images and the colorbar share one
  `Normalize(0, 1)` instance by construction. Tests assert equal panel boxes,
  non-overlap, and the shared norm. The documented `slice_indices` override was
  also silently ignored (`_render_qc_slice` always recomputed the centre slice);
  it is now honoured.

- **Reproducibility footer typography and content.** The footer mixed
  proportional prose with a monospace, open-ended dependency list that wrapped
  and pushed the report onto page 2. It is now fixed-height and set in one
  proportional face: hardware and Python/platform on line 1 (long CPU and
  platform strings truncated horizontally, `platform.platform()`'s hyphen soup
  reduced to "Linux 6.17.0-40-generic (x86_64)"), a summarised thread-limit line
  ("Thread limits: unset" / "Threads: OMP=1, MKL=1") instead of one entry per raw
  environment variable, and a *fixed* dependency subset (NumPy, SciPy, DIPY,
  NiBabel, Matplotlib) so two reports are comparable line by line. Git commit and
  command line remain JSON-only.

- **Regional-metrics subtitle removed.** "Means at key landmarks" was imprecise:
  the values are regional profile-bin means over the `TRACT_REGIONS` intervals,
  not measurements at isolated landmarks.

- **Report orientation label documented correctly.** The label was already
  derived from the FA affine, but `docs/explanation/design-decisions.md` claimed
  csttool reorients DWI to RAS during preprocessing, which made a correct `LAS`
  label look like a bug. Only the `dicom2nifti` fallback reorients to RAS (for
  gradient consistency, AU21); the primary `dcm2niix` path preserves the
  scanner's native orientation, which is LAS for a typical Siemens axial DWI
  series. Docs and the `convert_series` docstring are corrected, and tests now
  round-trip RAS, LAS and an oblique PSR affine through the report context.

- **Affine-equality assertion in coordinate validation (AU22).**
  `validate_tractogram_coordinates` previously checked only that streamline
  coordinates looked like mm and fell within the reference volume ±15 mm —
  bounding-box overlap, not affine equality — so a tractogram and FA map in
  mismatched coordinate spaces could pass when their world bounds happened to
  overlap, then yield anatomically-plausible-but-wrong extraction (GLM §3.10,
  Qwen §3.9). The validator now explicitly asserts that the tractogram's stored
  affine matches the reference affine (translation ≤ 1.0 mm, rotation/scale ≤
  1e-3 elementwise), reusing the exact tolerances from
  `validation.bundle_comparison.check_spatial_compatibility` so the two
  validators agree. For `.trk` this replaces reliance on a silent side effect of
  DIPY's `load_tractogram` header check with a clear, actionable error naming
  both affines. For headerless formats (`.tck`, text) that carry no affine, a
  warning now surfaces that the affine could not be asserted — previously a
  `.tck` in a different voxel space passed silently with zero warnings.
  `--skip-coordinate-validation` still bypasses everything.

- **Gradient-table validation at load (AU21).** `gradient_table` was built at
  every load site (`load_dataset`, `get_gtab_for_preproc`) with no checks that
  bvecs are unit-normalised, bvals are non-negative, a b=0 volume exists, or that
  bvecs/bvals counts match. A malformed table would silently corrupt the tensor
  fit. All three audits (GLM §3.9, Qwen §3.10) flagged this. A new leaf module
  `preprocess/modules/gradient_validation.py` now hard-fails with actionable
  messages (naming offending indices) at load, and threads the single-source-of-
  truth `DEFAULT_B0_THRESHOLD` into the gtab's `b0s_mask` (DIPY's own default of
  50 only matched by coincidence). The existing opt-in `csttool check-dataset`
  advisory assessment is unchanged.

- **dicom2nifti b-vector reorientation (AU21).** dicom2nifti's
  `reorient_nifti=True` reorients the **image** to LAS but never reorients the
  **bvecs**, leaving them in the scanner's voxel space — a silent gradient flip
  that corrupts the tensor fit, which the validator cannot detect from the files
  alone. All three DICOM conversion sites (`load_dataset`, `cli/utils.resolve_nifti`,
  `ingest/modules/convert_series`) now convert with `reorient_nifti=False` and
  reorient **both image and bvecs to RAS+ together** via the new
  `reorient_dwi_to_ras` helper, matching the validated dcm2niix primary path's
  convention. `register_mni_to_subject` already reorients the subject to RAS
  itself, so this makes the fallback consistent with the primary path rather than
  introducing a new orientation. (AU21)


- **AU20 — b0 threshold is now a single source of truth** (`DEFAULT_B0_THRESHOLD = 50`
  in `defaults.py`). Previously hardcoded as ``< 50`` in seven places across
  `background_segmentation.py`, `estimate_directions.py`, `denoise.py`, and
  `visualizations.py`. A dataset with b=30 shells or a non-zero lowest b-value
  would have been silently mis-partitioned. Exposed via `--b0-threshold` on
  `preprocess` and `run` (was only on `check-dataset`).

- **Parameter transparency — all pipeline defaults now live in `defaults.py`** and
  are read by the CLI parsers and command wrappers. Previously many defaults were
  repeated as bare literals in `add_argument(default=…)` and `getattr(…, literal)`
  across the CLI layer. The defaults module now covers: fa_threshold, seed_density,
  step_size, sh_order, rng_seed, sphere_name, peak extraction thresholds, ROI
  dilation, streamline length bounds, midline tolerance, LI interpretation bands,
  and the artifact_index cutoff. (Parameter transparency §2)

- **Provenance now captures hardware details** (CPU model, core count, RAM, GPU),
  **threading environment variables** (`OMP_NUM_THREADS` etc.), **the full command line**
  (`sys.argv`), and **five additional dependency versions** (nilearn, dicom2nifti,
  pydicom, matplotlib, scikit-learn, h5py, cython) that the audits (AU8) identified
  as missing. `--provenance` on the CLI prints the full provenance dict and exits.
  The `CLAUDE.md` claim about `provenance.json` is now accurate.
  (Parameter transparency §2, §3)

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

  *The premise "M ≠ 0" is confirmed.* The warped MNI midline does **not** land at
  world X = 0: `compute_warped_midline` returns a median world X of −3.58 mm
  (healthy control) and +0.84 mm (ALS). Measuring the warped L/R boundary directly by
  Z band, the healthy control's midline sits ~4 mm left of X = 0 at every level
  (brainstem −4.75 mm, mid −3.91 mm, cortex −3.68 mm). The subject is reoriented to
  RAS but never recentered, exactly as the finding says.

  *The midline is also not a plane.* It curves with the non-linear warp: the ALS
  subject's midline runs from +2.94 mm at the brainstem to −2.91 mm at the cortex, a
  ~6 mm range. **No single X value can be correct at both ends**, which is why the
  per-voxel `hemisphere_mask` — not a scalar plane — is the primary output.

  Both effects land on a structure that cannot absorb them: the cerebral peduncles are
  only ~10–15 mm wide, so a cut several mm off the true midline mislabels a large
  fraction of them. That is why the peduncle FA moves (below).

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
