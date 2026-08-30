# Tracking Module Walkthrough

The `tracking` module performs deterministic whole-brain tractography to reconstruct white matter pathways.

## Core Capability: `csttool track`

This command generates a whole-brain tractogram from preprocessed diffusion data using the Constant Solid Angle (CSA) ODF model.

### Gradient File Discovery

The command automatically discovers `.bval` and `.bvec` files by stripping common preprocessing suffixes (e.g., `_preproc`, `_dwi_preproc`) from the input filename and searching for matching gradient files in the same directory. Both singular (`.bval`/`.bvec`) and plural (`.bvals`/`.bvecs`) extensions are supported.

### Usage

```bash
csttool track \
    --nifti dti_preproc.nii.gz \
    --out tracking_results \
    --fa-thr 0.2 \
    --sh-order 6 \
    --seed-density 1 \
    --step-size 0.5
```

### Optional Parameters

- `--subject-id`: Subject identifier for output naming (default: extracted from filename)
- `--step-size`: Tracking step size in millimetres (default: 0.5)
- `--seed-density`: Seeds per voxel in the seed mask (default: 1)
- `--fa-thr`: FA threshold for stopping and seeding (default: 0.2)
- `--sh-order`: Maximum spherical harmonic order for CSA ODF model (default: 6)
- `--rng-seed`: Random seed for reproducible tractography (default: None, non-deterministic)
- `--brain-mask`: Path to a DWI-space brain mask (NIfTI) to use instead of automatic background segmentation. Must be 3D and already on the DWI grid (same shape and affine) — it is never resampled; any nonzero voxel counts as brain. Omit for automatic masking
- `--use-brain-mask-stop`: Stop tracking at brain mask boundary in addition to FA threshold
- `--show-plots`: Enable QC plots for segmentation and tractography
- `--verbose`: Print detailed processing information

### Algorithm Steps

1.  **Brain Masking**:
    -   Refines the brain mask using median_otsu to ensure tracking is confined to the brain.
2.  **Tensor Fitting (DTI)**:
    -   Fits the classical Tensor model to compute scalar maps:
        -   **FA**: Fractional Anisotropy (microstructural integrity).
        -   **MD**: Mean Diffusivity (overall diffusion magnitude).
        -   **RD/AD**: Radial/Axial Diffusivity.
3.  **Direction Estimation (CSA)**:
    -   Fits the Constant Solid Angle (CSA) model to the high-angular resolution data (HARDI).
    -   Computes Orientation Distribution Functions (ODFs) to resolve crossing fibers.
4.  **Seeding**:
    -   Generates seeds in all voxels where `FA > fa_thr` (approx. White Matter).
    -   Seeds are placed randomly within voxels based on `seed_density`.
5.  **Tracking**:
    -   Uses **LocalTracking** with peaks-based direction getter (PeaksAndMetrics inherits from EuDXDirectionGetter, see [here](https://docs.dipy.org/dev/_modules/dipy/direction/peaks.html)).
    -   **Stop Criteria**: Tracking stops if FA drops below threshold. Optionally, with `--use-brain-mask-stop`, also stops at brain boundary.
    -   **Turning Constraint**: Controlled by the direction getter's internal parameters.
6.  **Output**:
    -   `tractogram.trk`: The resulting whole-brain streamlines.
    -   `dti_FA.nii.gz`: FA map (reference for downstream registration).
    -   `dti_MD.nii.gz`: MD map.
    -   `dti_RD.nii.gz`: RD (Radial Diffusivity) map.
    -   `dti_AD.nii.gz`: AD (Axial Diffusivity) map.

## Example Output

```text
Subject: sub-001
Loading preprocessed data from dti_preproc.nii.gz...

Step 1: Brain masking with median Otsu
Step 2: Tensor fit and scalar measures
  - FA/MD/RD/AD maps created.

Step 3: Direction field estimation (CSA ODF model)
  - SH order: 6
  - Peaks found.

Step 4: Stopping criterion and seed generation
  - FA threshold: 0.2
  - Seed density: 1
  - Generated 620,250 seeds within white matter mask.

Step 5: Deterministic tracking
  - Algorithm: LocalTracking
  - Step size: 0.5mm
  - Turning constraint: direction getter controlled

Step 6: Saving outputs
  - Tractogram: tracking/tractogram.trk
  - FA map: tracking/dti_FA.nii.gz
  - MD map: tracking/dti_MD.nii.gz
  - RD map: tracking/dti_RD.nii.gz
  - AD map: tracking/dti_AD.nii.gz

TRACKING COMPLETE
Whole-brain streamlines: 452,180
```
