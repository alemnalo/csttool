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
| `--perform-motion-correction` | flag | off | Affine between-volume motion correction. b-vectors are rotated by the estimated per-volume transforms and the rotated `.bvec` is written alongside the output. Motion only — no eddy-current, outlier or susceptibility correction. |
| `--target-voxel-size` | 3×float | — | Reslice to the given isotropic voxel size in mm (e.g. `2 2 2`). |
| `--b0-threshold` | float | `50` | b-value at or below which a volume counts as b=0, in s/mm². Sets the gradient table's b0 partition, which also drives brain masking and Patch2Self. |
| `--input-corrected` | `unknown` \| `none` \| `topup-eddy` \| `eddy-only` \| `other` | `unknown` | Declare what correction was applied to the input *before* csttool received it. Recorded as a user declaration in the provenance; never verified, and it changes no processing decision. |
| `--save-visualizations` | flag | off | Write QC plots (denoising residuals, motion parameters) to `visualizations/`. |
| `--verbose` | flag | off | Print per-step diagnostics. |

### Pipeline Steps

The steps run in this order. It is the order the code executes, which is not the
order they are usually listed in — masking happens *before* Gibbs unringing and
motion correction, because the mask constrains the registration.

1.  **Load** — read the NIfTI (or convert DICOM), read the `.bval`/`.bvec`
    sidecars, and build a validated gradient table using `--b0-threshold`.
2.  **Reslicing** (optional, `--target-voxel-size`):
    -   Resamples the data to a target voxel size. The physical gradient frame is
        untouched, so b-vectors are unaffected.
3.  **Denoising**:
    -   **MPPCA** (default): Marchenko-Pastur PCA. Estimates the noise level itself
        from the eigenvalue distribution of local PCA patches, so it needs neither a
        receiver-coil count nor an assumption about the noise distribution.
        Deterministic across runs.
    -   **Patch2Self**: self-supervised; denoises each volume using information from
        the others. Requires bvals and enough directions; well suited to modern
        high-volume acquisitions.
    -   **NLMeans**: non-local means. Requires `--coil-count` for PIESNO noise
        estimation and assumes Gaussian noise; both are guesses on most modern
        multi-channel data, which is why it is no longer the default.
4.  **Brain Masking**:
    -   `median_otsu` over the volumes the gradient table marks as b=0.
5.  **Gibbs Unringing** (optional, `--unring`):
    -   Kellner subvoxel-shift correction for ringing near sharp contrast
        boundaries (e.g. skull/cortex).
6.  **Motion Correction** (optional, `--perform-motion-correction`):
    -   The b=0 volumes are registered to each other and averaged into a reference;
        every volume is then affine-registered to that reference.
    -   **The b-vectors are rotated by the estimated per-volume transforms** and the
        rotated `.bvec` is written next to the output, so the data and its gradient
        table describe the same anatomy (Leemans & Jones 2009). Tractography picks
        the rotated sidecar up automatically.
    -   Scope: affine, between-volume motion only. No eddy-current model, no outlier
        replacement, no slice-to-volume estimation, no susceptibility-distortion
        correction. See [limitations](../../explanation/limitations.md).
    -   If the correction fails, the run continues with the **uncorrected** data and
        the **original** gradients — always a mutually consistent pair — and the
        report records `motion_correction_requested: true` with
        `motion_correction: false`.
7.  **Save** — write the NIfTI, the gradient sidecars, the brain mask and the
    report JSON.

### Output

With input `sub-001.nii.gz`, the stem is `sub-001_dwi_preproc_mc` when motion
correction was applied and `sub-001_dwi_preproc_nomc` otherwise:

-   `{stem}.nii.gz` — the preprocessed 4D DWI series.
-   `{stem}.bval` — copied from the input, unchanged.
-   `{stem}.bvec` — copied from the input, or **rotated** when motion correction ran.
-   `{stem}_mask.nii.gz` — the computed brain mask.
-   `{stem}_report.json` — processing report, including the ordered stage ledger and
    the provenance block. See [output formats](../output-formats.md#preprocessing-report).
-   `visualizations/` (with `--save-visualizations`) — QC figures: denoising
    comparison, brain-mask overlay, motion summary, and a preprocessing summary.

## Example Output

```text
PREPROCESSING: Loaded data with shape (96, 96, 60, 65)
PREPROCESSING: Current voxel size: (2.0, 2.0, 2.0) mm
PREPROCESSING: Denoising complete (mppca)
PREPROCESSING: Brain masking complete
PREPROCESSING: Motion correction complete
PREPROCESSING: b-vectors rotated (max estimated head rotation 0.84°)
  ✓ Saved preprocessed data: .../sub-001_dwi_preproc_mc.nii.gz
  ✓ Copied bval: .../sub-001_dwi_preproc_mc.bval
  ✓ Wrote transformed bvec: .../sub-001_dwi_preproc_mc.bvec
  ✓ Saved brain mask: .../sub-001_dwi_preproc_mc_mask.nii.gz
  ✓ Saved processing report: .../sub-001_dwi_preproc_mc_report.json
PREPROCESSING: Saved outputs to .../preprocessing

PREPROCESSING COMPLETED
```
