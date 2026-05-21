# Data Requirements

This page describes the input data formats and requirements for csttool.

---

## Supported Input Formats

csttool accepts diffusion MRI data in two formats:

| Format | Description |
|--------|-------------|
| **DICOM** | Raw scanner output; automatically converted to NIfTI |
| **NIfTI** | Pre-converted `.nii` or `.nii.gz` with gradient files |

---

## NIfTI Input

### Required Files

For NIfTI input, you need **three files** with matching base names:

```
mdwi.nii.gz     # 4D diffusion-weighted image
dwi.bval       # b-values (one per volume)
dwi.bvec       # b-vectors (3 × N matrix)
```

!!! note "Gradient file extensions"
    csttool supports both singular (`.bval`, `.bvec`) and plural (`.bvals`, `.bvecs`) extensions. These are automatically detected.

### File Specifications

| File | Format | Description |
|------|--------|-------------|
| `.nii.gz` | 4D NIfTI | Shape: (X, Y, Z, N) where N = number of volumes |
| `.bval` | Space-separated text | N b-values (e.g., `0 1000 1000 1000...`) |
| `.bvec` | Space-separated text | 3 rows × N columns (gradient directions) |

### Example b-value File

```
0 1000 1000 1000 1000 2000 2000 2000
```

### Example b-vector File

```
0.0  0.5774  -0.5774   0.5774  -0.5774   0.7071  -0.7071   0.0
0.0  0.5774   0.5774  -0.5774   0.5774   0.0      0.7071   0.7071
0.0  0.5774   0.5774   0.5774  -0.5774   0.7071   0.0     -0.7071
```

---

## DICOM Input

For DICOM input, provide the path to the study directory:

```
study_folder/
├── series_001/          # T1-weighted (ignored)
├── series_002/          # DTI sequence (auto-selected)
│   ├── IM-0001.dcm
│   ├── IM-0002.dcm
│   └── ...
└── series_003/          # Another sequence
```

csttool will:

1. Scan all series in the directory
2. Analyze each for tractography suitability
3. Auto-select the best diffusion series
4. Convert to NIfTI with gradient files

!!! tip "Manual series selection"
    Use `csttool import --series-index N` to manually select a specific series.

---

## Image Acquisition Requirements

### Essential Requirements

| Parameter | Requirement | Rationale |
|-----------|-------------|-----------|
| **Modality** | Diffusion-weighted MRI | Required for tractography |
| **b-values** | At least one b=0 + one b>0 | Minimum for DTI fitting |
| **Directions** | ≥6 unique gradient directions | Tensor estimation |

### Recommended Specifications

For reliable CST extraction:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| **b-value** | 1000–3000 s/mm² | Higher b improves angular resolution |
| **Directions** | ≥30 | Better fiber orientation estimation |
| **Resolution** | ≤2.5 mm isotropic | Finer detail in tract anatomy |
| **FOV coverage** | Full brain + brainstem | CST runs from cortex to brainstem |

!!! warning "Incomplete field of view"
    CST extraction will fail if the scan does not cover the **motor cortex** through the **brainstem**. Ensure full craniocaudal coverage.

---

## Coordinate Space Requirements

### Critical: World Space (mm) Required

Tractogram streamlines **must be in world coordinates (millimeters)**, not voxel indices. This is a common source of errors when using tractograms from external sources.

| Property | Correct (mm) | Incorrect (voxels) |
|----------|-------------|-------------------|
| Coordinate range | -100 to +100 | 0 to 256 |
| Typical values | -80.5, 45.2, 62.0 | 45, 120, 78 |
| Negative coordinates | Common (RAS space) | Rare or absent |

### Automatic Validation

csttool automatically validates coordinate systems when running extraction. If your tractogram is in voxel space, you will see an error like:

```
Error: Coordinate validation failed. This can lead to incorrect results.
Errors: Coordinate space mismatch detected: values suggest voxel indices, not mm
```

### Converting Voxel to World Coordinates

If your tractogram is in voxel space, convert it before using csttool:

```python
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space

# Load with reference image
ref_img = nib.load("fa.nii.gz")
sft = load_tractogram("tractogram.trk", ref_img, to_space=Space.VOX)

# Convert to world coordinates
sft.to_rasmm()

# Save converted tractogram
save_tractogram(sft, "tractogram_rasmm.trk")
```

!!! note "Skipping validation (not recommended)"
    You can bypass coordinate validation with `--skip-coordinate-validation`, but this is strongly discouraged as it can produce anatomically plausible but incorrect results.

---

## Validating Your Data

Run `csttool doctor` to verify your environment, then:

```bash
# Check DICOM study
csttool import --dicom /path/to/study --out /path/to/output

# Check NIfTI files
csttool import --nifti /path/to/dwi.nii.gz --out /path/to/output
```

The import step will report:

- Number of volumes
- b-values present
- Number of gradient directions
- Any warnings about data quality

---

## Recommended Datasets

For testing or learning, use these publicly available diffusion MRI datasets:

| Dataset | b-value | Directions | Access |
|---------|---------|------------|--------|
| [HCP Young Adult](https://www.humanconnectome.org/) | 1000/2000/3000 | 90×3 shells | Registration required |
| [Brainlife Open Datasets](https://brainlife.io/datasets) | Various | Various | Free |
| [OpenNeuro dMRI](https://openneuro.org/) | Various | Various | Free |

Note that registration may be required.

---

## Next Steps

- [Quick Start](quickstart.md) — Run your first analysis
- [Troubleshooting](../how-to/troubleshooting.md) — Common data issues and fixes
