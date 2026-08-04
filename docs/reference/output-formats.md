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

Each scalar map has a `.json` derivative sidecar recording `Sources`, `Description`,
`CommandLine`, and `GeneratedAt`.

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
`sub-<id>_[ses-<label>_]stage-<stage>_qc-<label>.png`.

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

---

### `reports/` — reports, tabular outputs, and pipeline logs

#### User-facing reports

| File | Description |
| --- | --- |
| `*_report.html` | Interactive HTML clinical report (self-contained; embeds all QC images) |
| `*_report.pdf` | Clinical report, exactly one A4 portrait page (requires WeasyPrint) |

#### Tabular outputs

| File | Description |
| --- | --- |
| `*_metrics.json` | Complete bilateral metrics with acquisition, processing, and provenance metadata |
| `*_metrics.csv` | Flat CSV table for group-level analysis (one row per subject) |

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
