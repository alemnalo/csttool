# Output formats

csttool writes a BIDS derivatives dataset. This page describes every file produced
by `csttool run` and the BIDS compliance level of each output type.

---

## BIDS compliance levels

| Output | Level | Notes |
| --- | --- | --- |
| Preprocessed DWI, scalar maps, brain mask | **BIDS MRI Derivatives** | Follows BEP016 entity conventions; passes `bids-validator` |
| Tractograms (`.trk`) | **BIDS-adjacent container** | Stored under a BIDS derivatives tree with BIDS-like naming; not covered by any finalised tractography schema |
| `figures/` QC images, `reports/` HTML/PDF | **Ancillary** | BIDS derivatives explicitly permits non-standard ancillary files alongside compliant outputs |

The defensible thesis statement: *"The raw import is fully BIDS-compliant. Derivative
NIfTIs adhere to BIDS MRI Derivatives naming conventions. Tractography outputs are
placed in a BIDS-derivatives-compatible container using csttool-specific naming, due
to the absence of a finalised tractography specification."*

---

## Dataset-level files

Written once at the derivatives root. Skipped if already present (safe to re-run).

| File | Description |
| --- | --- |
| `dataset_description.json` | BIDS-required metadata: `Name`, `BIDSVersion`, `DatasetType: derivative`, `GeneratedBy`. Includes `SourceDatasets: [{"URL": "bids::"}]` when the derivatives directory is nested under the raw BIDS root. |
| `participants.tsv` | One row per subject: `participant_id`, `age`, `sex`. Updated after each subject; file-locked for safe concurrent batch writes. |
| `participants.json` | Column definitions and units for `participants.tsv`. |

---

## Per-subject files

All files follow BIDS entity ordering: `sub` → `ses` → `space` → `desc` → `model` → `param`.
The session level (`ses-<label>/`) is omitted when `--session-id` is not set.

### `dwi/` — scientific derivatives

#### Preprocessed DWI

| File | Description |
| --- | --- |
| `*_space-orig_desc-preproc_dwi.nii.gz` | Denoised, skull-stripped DWI in native space |
| `*_space-orig_desc-preproc_dwi.bval` | b-values |
| `*_space-orig_desc-preproc_dwi.bvec` | Gradient directions |
| `*_space-orig_desc-preproc_dwi.json` | BIDS sidecar (from dcm2niix if available, otherwise generated) |

#### Scalar maps

| File | Description |
| --- | --- |
| `*_space-orig_model-DTI_param-FA_dwimap.nii.gz` | Fractional anisotropy |
| `*_space-orig_model-DTI_param-MD_dwimap.nii.gz` | Mean diffusivity |
| `*_space-orig_model-DTI_param-RD_dwimap.nii.gz` | Radial diffusivity |
| `*_space-orig_model-DTI_param-AD_dwimap.nii.gz` | Axial diffusivity |
| `*_space-orig_desc-V1_dwimap.nii.gz` | Principal diffusion eigenvector (V1) in the anatomical world (RAS+) frame. Written as a 5-D `(X,Y,Z,1,3)` float32 volume with `NIFTI_INTENT_VECTOR` so the header alone declares it a vector field. The eigenvectors are rotated from the b-vec (voxel) frame into the world frame by the orthonormal polar factor of the affine; the sidecar records `VectorFrame: "world-RAS"`, the affine determinant, obliquity and shear diagnostics so a reader can reproduce the DEC from this file + the FA map. Unconditional whenever `track` runs. |
| `*_space-orig_desc-CSTdensity_dwimap.nii.gz` | CST streamline density: the fraction of distinct retained bilateral CST streamlines that visit each voxel at least once. `float32`, range `[0,1]`, on the native FA grid. The sidecar records the exact definition, the denominator (`StreamlineCountLeft + StreamlineCountRight`) and whether a step-size gap guard densified the streamlines. Unconditional whenever `extract` runs. |

Each scalar map has a `.json` derivative sidecar recording `Sources`, `Description`,
`CommandLine`, and `GeneratedAt`. The V1 and density sidecars additionally carry
`VectorFrame` / `AffineDeterminant` / `ObliquityRad` / `ShearMagnitude` and
`Denominator` / `StreamlineCount{Left,Right}` / `Densified` respectively.

#### Segmentations

| File | Description |
| --- | --- |
| `*_space-orig_desc-CSTroi_dseg.nii.gz` | Discrete segmentation of the three warped Harvard-Oxford ROIs that constrained CST extraction. `uint8` on the **native FA grid**, values `0` background, `1` brainstem, `2` motor-left, `3` motor-right. `dseg` is the BIDS discrete-segmentation suffix. Written whenever `extract` runs. |

The extraction stage also writes three individual `roi_*.nii.gz` masks under
`extraction/nifti/`, but those are working files: they do not reach the
derivatives tree. The `dseg` is the persisted product, and it is written on the
FA grid specifically so a reader (and the report's QC strip) can overlay it on
FA, the density volume and the streamlines with no resampling. Its sidecar
carries `Labels` (the value → name map), `Space: "orig (FA grid)"` and
`VoxelCounts` per label alongside the standard keys.

#### `tractography/` — tractograms

| File | Description |
| --- | --- |
| `*_space-orig_desc-wholebrain_tractogram.trk` | Full whole-brain tractogram |
| `*_space-orig_desc-CSTleft_tractogram.trk` | Left corticospinal tract |
| `*_space-orig_desc-CSTright_tractogram.trk` | Right corticospinal tract |
| `*_space-orig_desc-CSTbilateral_tractogram.trk` | Bilateral CST (combined) |

All tractograms are in native DWI space (`space-orig`).

---

### `figures/` — QC images

Diagnostic images produced when `--save-visualizations` is passed. Named
`sub-<id>_[ses-<label>_]stage-<stage>_qc-<label>.png`. In addition to the legacy
stage QC images, the visualization refactor adds three complementary report QC
panels (currently standalone prototypes; not embedded in the PDF report until
scientific review passes):

| Label | File suffix | Stage | Question answered |
| --- | --- | --- | --- |
| `decfa` | `_dec_fa.png` | tracking | Does the local diffusion direction field support the reconstructed anatomy? |
| `density` | `_cst_density.png` | extraction | Is the extracted bundle spatially coherent and left/right symmetric? |
| `cstoverfa` | `_cst_over_fa.png` | metrics | Does the extracted CST follow the expected anatomical course? |

All three share one deterministic coronal slice chosen by a documented,
data-driven fallback chain (`bilateral_occupancy` → `surviving_hemisphere` →
`roi_occupancy` → `anatomical_centroid`); the rule used is recorded in the
panel's JSON sidecar and the figure caption. Each panel is 90 mm × 90 mm at
200 dpi as a standalone prototype; final report dimensions are fixed only after
review.

Six further panels ask whether the *numbers* are trustworthy rather than
whether the picture looks right. They are plots, not slices, so they carry no
slice rule; each is 140 mm × 90 mm at 200 dpi and writes a JSON sidecar holding
the statistics behind it, so the panel's claim is recoverable without re-running
the pipeline. Like the three above, they are standalone and not embedded in the
PDF report.

| Label | File suffix | Stage | Question answered |
| --- | --- | --- | --- |
| `tissue` | `_qc_tissue_plausibility.png` | metrics | Are the reported scalars coming from voxels that behave like white matter, or from voxels contaminated by CSF partial volume? |
| `v1angle` | `_qc_v1_angle.png` | metrics | Are the streamlines following the local principal diffusion direction, or being pushed through crossing-fibre voxels? |
| `dispersion-<scalar>` | `_qc_profile_dispersion_<scalar>.png` | metrics | Is the mean profile a consensus of the bundle, or an average over streamlines that disagree? |
| `saturation` | `_qc_sampling_saturation.png` | metrics | Is the headline mean converged with respect to streamline count — would a re-run give the same number? |
| `attrition` | `_qc_profile_attrition.png` | metrics | How much of the bundle was silently discarded when the profile was built? |
| `nodehomology` | `_qc_node_homology.png` | metrics | Does profile node *i* refer to the same anatomical level in both hemispheres? |

The `v1angle` panel needs the world-frame V1 product; `csttool metrics` takes it
via `--v1` and otherwise looks for `{stem}_v1.nii.gz` beside the FA map. The
`tissue` panel needs the MD map (`--md`) and the CST density product, taken via
`--density` and otherwise looked for in the extraction stage's `scalar_maps/`
beside the tractograms. A panel whose inputs are missing is skipped with a
message; the others are still written.

These six sample exactly as the metrics they audit do — the same
inferior-to-superior reorientation, the same nearest-neighbour lookup, the same
drop of out-of-bounds points, and the same resample-to-20-nodes rule — so
`dispersion`'s band describes the published mean rather than a near neighbour of
it. The counts printed on the dispersion panel are the streamlines that survived
profile sampling, not the ones extraction retained; the difference is recorded as
`left_attrition` / `right_attrition` in the sidecar.

#### `stage-preproc`

| `qc-` label | Contents |
| --- | --- |
| `denoising` | Before/after denoising comparison (3 orthogonal views + residuals) |
| `gibbs` | Before/after Gibbs unringing (only if `--unring`) |
| `brainmask` | Brain mask overlaid on b0 volume |
| `motion` | Translation/rotation plots (only if `--perform-motion-correction`) |
| `summary` | Multi-panel preprocessing summary |

#### `stage-tracking`

| `qc-` label | Contents |
| --- | --- |
| `tensormaps` | FA, MD, RGB direction map, mask in 3 views |
| `wmmask` | White matter mask QC |
| `streamlines` | 2D streamline projections (FA anatomy + world coordinates) |
| `stats` | Length histogram, seed density, cumulative distribution |
| `summary` | Multi-panel tracking summary |

#### `stage-extraction`

| `qc-` label | Contents |
| --- | --- |
| `registration` | MNI→subject registration quality (subject FA, warped MNI, overlay) |
| `jacobian` | Jacobian determinant map (deformation magnitude) |
| `roimasks` | Motor cortex and brainstem ROIs overlaid on FA |
| `cst` | Extracted bilateral CST streamlines in world coordinates |
| `hemispheres` | Hemisphere separation QC with midline reference and contamination metrics |
| `summary` | ROI + CST + statistics combined |

#### `stage-metrics`

| `qc-` label | Contents |
| --- | --- |
| `tractprofile` | FA profile along normalised tract length (bilateral comparison) |
| `bilateral` | Bilateral comparison bar charts |
| `profiles` | Stacked FA/MD/RD/AD profiles |
| `tractogram-axial` | Tractogram QC axial view |
| `tractogram-sagittal` | Tractogram QC sagittal view |
| `tractogram-coronal` | Tractogram QC coronal view |
| `tissue` | Density-weighted FA–MD joint histogram with the free-water corner outlined |
| `v1angle` | Streamline tangent vs local V1, along the tract (median, IQR, 5–95) |
| `dispersion-fa` | FA profile dispersion across streamlines (median, IQR, 5–95) |
| `qc-strip` | The 1×4 report QC strip embedded in the PDF (DEC-FA, CST density, extraction ROIs, CST over FA on one shared coronal slice) |
| `saturation` | Mean FA vs subsample size, with the bootstrap SE at full N |
| `attrition` | Streamline survival funnel from extraction to profile, plus per-point retention |
| `nodehomology` | Per-node world Z and streamline-length distributions, left vs right |

---

### `reports/` — reports, tabular outputs, and pipeline logs

#### User-facing reports

| File | Description |
| --- | --- |
| `*_report.html` | Interactive HTML clinical report (self-contained; embeds all QC images) |
| `*_report.pdf` | Clinical report, exactly one A4 portrait page (requires WeasyPrint) |

The report embeds exactly two figures, each generated at its final print size and
placed by CSS at that same width, so the printed height is the designed height:

| Figure | Size | Contents |
| --- | --- | --- |
| Along-tract profile matrix | 194 × 94 mm | FA / MD / RD / AD profiles, left and right, each line backed by the per-node **interquartile range across contributing streamlines**. The centre line is the mean — the same array the twelve regional values are derived from — so the band is dispersion around the published number, not a different statistic. |
| Report QC strip | 194 × 44 mm | 1×4: DEC-FA, CST density, extraction ROIs, final CST over FA, all on **one** shared coronal slice with one legend, one caption and one colourbar. Replaces the earlier 1×3 triptych, which showed the same information from three angles. |

Every strip panel degrades independently: a missing world-frame V1, density
volume or ROI segmentation leaves a labelled grayscale-FA slot in place rather
than reflowing the strip to three panels, so a reader can see *which* question
went unanswered. The shared slice comes from the same documented fallback chain
the standalone panels use (`bilateral_occupancy` → `surviving_hemisphere` →
`roi_occupancy` → `anatomical_centroid`), recorded with the slab thickness,
streamline counts, density `vmax` and per-ROI slab voxel counts in the strip's
JSON sidecar.

#### Tabular outputs

| File | Description |
| --- | --- |
| `*_metrics.json` | Complete bilateral metrics with acquisition, processing, and provenance metadata |
| `*_metrics.csv` | Flat CSV table for group-level analysis (one row per subject) |

##### Uncertainty and dispersion fields

Each per-scalar block under `metrics.left` / `metrics.right` carries, alongside
the existing summary:

| Key | Meaning |
| --- | --- |
| `profile_p25`, `profile_p75` | Per-node 25th/75th percentile across the streamlines that contributed to `profile` — the band the report draws. |
| `profile_n` | How many streamlines contributed. May be lower than `n_streamlines`: a streamline needs one in-bounds sample to reach the headline mean but five to enter the profile. |
| `bootstrap_se` | Bootstrap standard error of the headline `mean`. |
| `pontine_se`, `plic_se`, `precentral_se` | Bootstrap SE of each regional mean. |

`metrics.left.morphology` gains `bootstrap_se_length`; each entry under
`metrics.asymmetry` gains `laterality_index_se`. Streamline count and tract
volume deliberately carry **no** SE: a count *is* its own sample size, and a
volume is a set-union voxel count rather than a mean over a resamplable
population, so a bootstrap of it would estimate something else while reading as
an SE of a mean.

`metrics.uncertainty` states the method, the resample count, the seed, and the
scope — the last of which is the important one: these standard errors are
**conditional on the retained bundle**. They quantify how much a reported mean
would move if a different subset of *those* streamlines had been sampled. They do
not include tracking, seeding, registration, preprocessing or acquisition
variability; a re-run of tractography produces a different bundle, not a resample
of this one.

`metrics.node_homology` records how far apart profile node *i* actually sits on
the two sides (`max_abs_z_difference_mm`, `length_difference_mm`, the two 20-element
per-node difference arrays, the per-side length means and counts). It is
descriptive only: **no threshold, flag or pass/fail key exists in the schema**,
and no regional value is ever suppressed or annotated on the basis of it. Both
difference fields are `null` when either hemisphere has no streamlines.

The CSV appends, at the end of the row so column positions are unchanged:
`left_mean_length_se`, `right_mean_length_se`, `left_<scalar>_bootstrap_se`,
`right_<scalar>_bootstrap_se`, `<scalar>_laterality_index_se`,
`node_homology_max_abs_z_diff_mm` and `node_homology_length_diff_mm`. A missing
standard error is written as an **empty cell, never `0.0`** — a zero SE claims a
mean is exactly determined, which a serialiser must not assert on behalf of an
input that never carried one. Regional SEs and the per-node arrays stay out of
the flat table.

#### Pipeline logs (provenance)

| File | Description |
| --- | --- |
| `*_log-import.json` | DICOM import report: converter used, `fallback_used`, warnings, scanner manufacturer |
| `*_log-series.json` | DICOM series analysis: acquisition parameters, suitability score |
| `*_log-preproc.json` | Preprocessing report: methods applied, motion statistics |
| `*_log-tracking.json` | Tracking report: parameters, streamline counts, timing |
| `*_log-extraction.json` | Extraction report: ROI approach, streamline counts per hemisphere |

---

## Dataset root

| File | Description |
| --- | --- |
| `sub-<id>_pipeline_report.json` | Step-by-step pipeline execution log with timing and error information |

---

## Raw BIDS import output

`csttool import --dicom <dir> --raw-bids <out>` produces a separate raw dataset
(not a derivatives dataset):

    <out>/
    ├── dataset_description.json   (DatasetType: "raw")
    ├── participants.tsv
    ├── participants.json
    └── sub-<id>/
        └── ses-<date>/
            └── dwi/
                ├── sub-<id>_ses-<date>_dwi.nii.gz
                ├── sub-<id>_ses-<date>_dwi.bval
                ├── sub-<id>_ses-<date>_dwi.bvec
                └── sub-<id>_ses-<date>_dwi.json

This output is fully BIDS-compliant and passes `bids-validator`. Pass it as input to
`csttool run --nifti` or `csttool batch --bids-dir` for downstream analysis.
