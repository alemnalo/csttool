# `csttool run`

Run the complete CST analysis pipeline for a single subject. Outputs a BIDS
derivatives dataset by default — no extra flags required.

---

## Usage

=== "DICOM input"

    ```bash
    csttool run \
        --dicom /path/to/dicoms \
        --out /data/bids/derivatives/csttool \
        --subject-id 001 \
        --generate-pdf
    ```

=== "NIfTI input"

    ```bash
    csttool run \
        --nifti /data/bids/sub-001/ses-01/dwi/sub-001_ses-01_dwi.nii.gz \
        --out /data/bids/derivatives/csttool \
        --subject-id 001 \
        --session-id 01 \
        --generate-pdf
    ```

---

## Pipeline steps

| Step | Name | Description |
| --- | --- | --- |
| 1 | **Check** | Validates environment and dependencies |
| 2 | **Import** | Converts DICOM to NIfTI or validates existing NIfTI |
| 3 | **Preprocess** | Denoises, skull-strips, optionally corrects motion |
| 4 | **Track** | Generates a whole-brain deterministic tractogram |
| 5 | **Extract** | Isolates left and right CST using atlas-based ROI filtering |
| 6 | **Metrics** | Computes FA/MD/RD/AD along tracts and generates reports |

---

## Output layout

All outputs are written as a BIDS derivatives tree. The `--out` directory is the
dataset root. Stage working directories are created internally and removed after
finalization — they are never part of the final output.

    <out>/
    ├── dataset_description.json
    ├── participants.tsv
    ├── participants.json
    ├── sub-<id>/
    │   └── ses-<label>/          (omitted if --session-id not set)
    │       ├── dwi/
    │       │   ├── sub-<id>_ses-<label>_space-orig_desc-preproc_dwi.nii.gz
    │       │   ├── sub-<id>_ses-<label>_space-orig_desc-preproc_dwi.bval
    │       │   ├── sub-<id>_ses-<label>_space-orig_desc-preproc_dwi.bvec
    │       │   ├── sub-<id>_ses-<label>_space-orig_desc-preproc_dwi.json
    │       │   ├── sub-<id>_ses-<label>_space-orig_model-DTI_param-FA_dwimap.nii.gz
    │       │   ├── sub-<id>_ses-<label>_space-orig_model-DTI_param-MD_dwimap.nii.gz
    │       │   ├── sub-<id>_ses-<label>_space-orig_model-DTI_param-RD_dwimap.nii.gz
    │       │   ├── sub-<id>_ses-<label>_space-orig_model-DTI_param-AD_dwimap.nii.gz
    │       │   └── tractography/
    │       │       ├── sub-<id>_ses-<label>_space-orig_desc-wholebrain_tractogram.trk
    │       │       ├── sub-<id>_ses-<label>_space-orig_desc-CSTleft_tractogram.trk
    │       │       ├── sub-<id>_ses-<label>_space-orig_desc-CSTright_tractogram.trk
    │       │       └── sub-<id>_ses-<label>_space-orig_desc-CSTbilateral_tractogram.trk
    │       ├── figures/
    │       │   ├── sub-<id>_ses-<label>_stage-preproc_qc-brainmask.png
    │       │   ├── sub-<id>_ses-<label>_stage-preproc_qc-denoising.png
    │       │   ├── sub-<id>_ses-<label>_stage-tracking_qc-tensormaps.png
    │       │   ├── sub-<id>_ses-<label>_stage-extraction_qc-registration.png
    │       │   └── ...
    │       └── reports/
    │           ├── sub-<id>_ses-<label>_report.html
    │           ├── sub-<id>_ses-<label>_report.pdf
    │           ├── sub-<id>_ses-<label>_metrics.json
    │           ├── sub-<id>_ses-<label>_metrics.csv
    │           └── sub-<id>_ses-<label>_log-tracking.json
    └── sub-<id>_pipeline_report.json

!!! note "BIDS compliance levels"
    - **`dwi/`** NIfTIs and bval/bvec — BIDS MRI Derivatives compliant; passes `bids-validator`
    - **Tractograms** — BIDS-adjacent container (no finalised BIDS tractography schema exists)
    - **`figures/` and `reports/`** — ancillary files; BIDS derivatives explicitly permits these

### Overriding the derivatives root

By default `--bids-out` equals `--out`. To place derivatives inside an existing raw BIDS
dataset (so that `SourceDatasets: bids::` resolves correctly):

    csttool run \
        --nifti /data/bids/sub-001/ses-01/dwi/sub-001_ses-01_dwi.nii.gz \
        --out /data/bids/derivatives/csttool \
        --bids-out /data/bids/derivatives/csttool \
        --subject-id 001 --session-id 01

---

## Options

### Input

| Flag | Description |
| --- | --- |
| `--dicom <dir>` | Path to DICOM directory |
| `--nifti <file>` | Path to existing NIfTI (skips import step) |
| `--out <dir>` | Output directory (BIDS dataset root) |
| `--subject-id <id>` | Subject label (without `sub-` prefix) |
| `--session-id <id>` | Session label (without `ses-` prefix) |
| `--bids-out <dir>` | Override BIDS derivatives root (default: same as `--out`) |
| `--series <n>` | DICOM series number to convert (1-indexed) |
| `--field-strength <T>` | Field strength override (Tesla) |
| `--echo-time <ms>` | Echo time override (ms) |

### Preprocessing

| Flag | Description |
| --- | --- |
| `--preprocess` | Enable preprocessing (denoising + brain masking). Default: skipped. |
| `--denoise-method` | `nlmeans` (default), `patch2self`, or `mppca` |
| `--unring` | Enable Gibbs unringing |
| `--perform-motion-correction` | Enable between-volume motion correction |
| `--target-voxel-size X Y Z` | Reslice to target voxel size (mm) |
| `--coil-count <n>` | Receiver coil count for PIESNO (nlmeans only) |

### Tractography

| Flag | Description |
| --- | --- |
| `--fa-thr <f>` | FA threshold for stopping and seeding (default: 0.2) |
| `--seed-density <n>` | Seeds per voxel (default: 1) |
| `--step-size <mm>` | Tracking step size (default: 0.5) |
| `--sh-order <n>` | Spherical harmonic order for CSA ODF (default: 6) |

### Extraction

| Flag | Description |
| --- | --- |
| `--extraction-method` | `passthrough` (default), `endpoint`, `roi-seeded`, or `bidirectional` |
| `--dilate-brainstem <n>` | Brainstem ROI dilation iterations (default: 2) |
| `--dilate-motor <n>` | Motor cortex ROI dilation iterations (default: 1) |
| `--min-length <mm>` | Minimum streamline length (default: 20) |
| `--max-length <mm>` | Maximum streamline length (default: 200) |

### Output and pipeline control

| Flag | Description |
| --- | --- |
| `--generate-pdf` | Generate PDF clinical report |
| `--save-visualizations` | Save QC images at each pipeline stage |
| `--skip-check` | Skip environment check step |
| `--continue-on-error` | Continue pipeline if a step fails |
| `--verbose` | Detailed output |
| `--quiet` | Suppress progress messages (errors still shown) |

---

## Related

- [Output formats](../output-formats.md) — detailed description of every output file
- [`csttool import`](import.md) — standalone DICOM import and raw BIDS organisation
- [`csttool batch`](batch.md) — run the pipeline over many subjects
