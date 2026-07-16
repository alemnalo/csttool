# Preprocess Module Walkthrough

The `preprocess` module prepares raw diffusion data for tractography by reducing noise and artifacts.

## Core Capability: `csttool preprocess`

This command runs a sequential pipeline of image correction steps.

### Usage

```bash
csttool preprocess \
    --nifti raw_dwi.nii.gz \
    --out preprocess_results \
    --denoise-method patch2self \
    --unring \
    --perform-motion-correction \
    --save-visualizations
```

### Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--nifti` | path | — | Input 4D NIfTI DWI file. Mutually exclusive with `--dicom`. |
| `--dicom` | path | — | Input DICOM directory; converted internally. |
| `--out` | path | — | Output directory (created on demand). **Required**. |
| `--denoise-method` | `mppca` \| `patch2self` \| `nlmeans` | `mppca` | Denoising algorithm. MPPCA estimates its own noise level and needs no extra information. Use `patch2self` for modern high-volume acquisitions, or `nlmeans` for low-volume acquisitions where Patch2Self over-smooths. |
| `--coil-count` | int | `4` | Number of receiver coils, used by NLMeans noise estimation. Ignored by `mppca` and `patch2self`. |
| `--unring` | flag | off | Apply Kellner Gibbs-unringing to attenuate ringing near sharp tissue boundaries. |
| `--perform-motion-correction` | flag | off | Affine-register all volumes to the first b=0 volume. |
| `--target-voxel-size` | 3×float | — | Reslice to the given isotropic voxel size in mm (e.g. `2 2 2`). |
| `--save-visualizations` | flag | off | Write QC plots (denoising residuals, motion parameters) to `visualizations/`. |
| `--verbose` | flag | off | Print per-step diagnostics. |

### Pipeline Steps

1.  **Denoising**:
    -   **MPPCA** (Default): Marchenko-Pastur PCA. Estimates the noise level itself from the eigenvalue distribution of local PCA patches, so it needs neither a receiver-coil count nor an assumption about the noise distribution. Deterministic across runs.
    -   **Patch2Self**: Self-supervised method that denoises each volume using information from the others. Requires bvals and enough directions; well suited to modern high-volume acquisitions.
    -   **NLMeans**: Non-local means denoising (classic approach). Requires `--coil-count` for PIESNO noise estimation and assumes Gaussian noise; both are guesses on most modern multi-channel data, which is why it is no longer the default.
2.  **Gibbs Unringing** (Optional):
    -   Corrects for Gibbs ringing artifacts near sharp contrast boundaries (e.g., skull/cortex).
3.  **Motion Correction** (Optional):
    -   Registers all volumes to the first b=0 image using an affine transformation to correct for subject movement.
4.  **Reslicing** (Optional):
    -   Resamples the data to a target isotropic voxel size (e.g., 2mm iso).
5.  **Brain Masking**:
    -   Computes a binary brain mask using median_otsu to strip the skull.

### Output

-   `dti_preproc.nii.gz`: The fully corrected 4D DWI series.
-   `dti_preproc_mask.nii.gz`: The computed brain mask.
-   `visualizations/`:
    -   `denoising_residuals.png`: Visual check of what was removed (should be noise, not anatomy).
    -   `motion_correction_plot.png`: Estimated translation/rotation parameters over time.

## Example Output

```text
Starting preprocessing for sub-001...
Input: 96x96x60 (65 volumes)

1. Gibbs Unringing
   - Axis: 2 (z-axis)
   - Splits: 3

2. Motion Correction
   - Reference: Volume 0 (b=0)
   - Corrected 64 volumes
   - Max translation: 1.2mm (Vol 42)
   - Max rotation: 0.8 deg (Vol 42)

3. Denoising (Patch2Self)
   - Model: OLS
   - Shift: True
   - Sigma estimation: Local PCA

✓ Preprocessing complete: /processed/sub-001/preprocessing/dti_preproc.nii.gz
```
