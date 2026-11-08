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
    │       │   ├── sub-<id>_ses-<label>_space-orig_desc-preferredSeedLeft_mask.nii.gz
    │       │   ├── sub-<id>_ses-<label>_space-orig_desc-preferredSeedRight_mask.nii.gz
    │       │   ├── transforms/
    │       │   │   ├── <subject>_mni_to_subject_affine.npy
    │       │   │   ├── <subject>_syn_forward_displacement.nii.gz
    │       │   │   ├── <subject>_syn_backward_displacement.nii.gz
    │       │   │   └── <subject>_registration_transform.json
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
    │       │   └── ... (JSON sidecars accompany auditable QC panels)
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
| `--denoise-method` | `mppca` (default), `patch2self`, or `nlmeans` |
| `--b0-threshold` | b-value at or below which a volume counts as b=0, in s/mm² (default: 50). Drives the gradient table, brain masking and Patch2Self. |
| `--input-corrected` | Declare correction applied to the input *before* csttool saw it: `unknown` (default), `none`, `topup-eddy`, `eddy-only`, `other`. Recorded as a user declaration, never verified. |
| `--unring` | Enable Gibbs unringing |
| `--perform-motion-correction` | Enable between-volume motion correction |
| `--target-voxel-size X Y Z` | Reslice to target voxel size (mm) |
| `--coil-count <n>` | Receiver coil count for PIESNO (nlmeans only) |

### Tractography

| Flag | Description |
| --- | --- |
| `--fa-thr <f>` | FA threshold for legacy stopping and FA-derived WM support (default: 0.2); preferred mode does not use FA as a hard stopping surface |
| `--seed-density <n>` | Seeds per voxel axis (default: 1) |
| `--step-size <mm>` | Tracking step size (default: 0.5) |
| `--sh-order <n>` | Requested spherical harmonic order for single-shell CSA ODF (default: 6; reduced when the selected shell has insufficient directions) |
| `--brain-mask <path>` | Use this DWI-space brain mask (NIfTI) instead of csttool's automatic background segmentation. Must be 3D and already on the DWI grid (same shape and affine) — it is never resampled; any nonzero voxel counts as brain. Omit for automatic masking. |
| `--wm-mask <path>` | Use this DWI-space white matter mask (NIfTI) verbatim as the seeding and peak-estimation territory. Must be 3D and already on the DWI grid (same shape and affine) — it is never resampled and never morphologically edited; any nonzero voxel counts as white matter. Takes precedence over `--wm-method`. |
| `--wm-method <m>` | How to obtain the WM mask when `--wm-mask` is absent: `auto` (default) uses DIPY's DAM tissue segmentation when the acquisition has ≥2 non-b0 shells and falls back to the legacy `dilate(FA > --fa-thr)` mask otherwise; `dam` forces DAM (falls back with a warning if inapplicable); `legacy` pins the historical mask for bit-exact reproducibility. |
| `--dam-wm-threshold <f>` | P-map slope cut below which DAM calls a voxel white matter (default 0.5, DAM only). |
| `--dam-low-signal <f>` | b=0 signal level below which DAM refuses to classify a voxel (default 50; lower for small-magnitude data). |

#### CST reconstruction architecture (`--tracking-mode`)

| Flag | Description |
| --- | --- |
| `--tracking-mode <m>` | `preferred` = role-separated, PLIC-seeded, dual-target pipeline (brain support, model mask, seed mask and stopping resolved independently; JHU posterior-limb-of-internal-capsule seeds; registration masks converted from the internal RAS grid to the original DWI grid before tracking; automatic supported single-shell CSA; CMC when PVE maps are supplied, else ROI-constrained DWI-only). `legacy-fa` (default) retains FA-seeded/FA-stopped behaviour for comparison. |
| `--model-mask <path>` | Optional DWI-grid mask over which CSA peaks are estimated in preferred mode (never resampled; must match the FA grid). Omit to use the permissive brain mask (recommended). An external WM mask here informs peak estimation only — it does not become the seed or the stopping boundary. |
| `--seed-mask <path>` | Optional DWI-grid binary seed region overriding the JHU PLIC seeds in preferred mode (never resampled; must match the FA grid). |
| `--wm-pve <path>` / `--gm-pve <path>` / `--csf-pve <path>` | Continuous WM/GM/CSF partial-volume maps (DWI grid) enabling CMC stopping in preferred mode (GM = valid termination, CSF = invalid). All three required together; hard label maps are refused. Omit for the ROI-constrained DWI-only domain. |
| `--midline-tolerance-mm <mm>` | Allowed medial excursion before a streamline is excluded as commissural/contralateral (default 8). |

> Preferred mode is deterministic and remains opt-in. The residual validation
> investigation found two independent correctness defects: registration-derived
> RAS arrays had been paired with the original LAS DWI affine during tracking,
> and CSA had pooled every shell despite being a single-shell model. Preferred
> mode now inverse-reorients every registration mask to the DWI grid, validates
> physical hemisphere placement, and records the automatically selected shell.
> `legacy-fa` remains the default pending validation beyond the two-subject
> regression cohort; this is a cohort-evidence decision, not a silent fallback.

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
