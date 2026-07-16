# Quick Start

Run the entire CST assessment pipeline in one command.

---

## The `csttool run` Command

=== "NIfTI Input"

    ```bash
    csttool run --nifti /path/to/dwi.nii.gz --out /path/to/output \
        --subject-id sub-01 --generate-pdf --save-visualizations
    ```

=== "DICOM Input"

    ```bash
    csttool run --dicom /path/to/study/ --out /path/to/output \
        --subject-id sub-01 --generate-pdf --save-visualizations
    ```

!!! tip "Need sample data?"
    See [recommended datasets](data-requirements.md#recommended-datasets) for freely available diffusion MRI data.

!!! note "Runtime & disk space"
    Typical runtime is **2–10 minutes** depending on data size, hardware and chosen parameters. Notably, `patch2self` denoising and the `--perform-motion-correction` flag will increase runtime (in the latter case quite substantially). The pipeline generates up to **500 MB** of output files (tractograms, scalar maps, reports) per pipeline run.

---

## What Each Step Does

The pipeline runs 6 steps automatically:

| Step | Name | Description |
|------|------|-------------|
| 1 | **Check** | Validates your environment and dependencies |
| 2 | **Import** | Loads NIfTI/DICOM data and extracts gradient information |
| 3 | **Preprocess** | Denoises, corrects motion, and segments the brain |
| 4 | **Track** | Generates a whole-brain tractogram using CSD-based tractography |
| 5 | **Extract** | Isolates left and right CST using atlas-based ROI filtering |
| 6 | **Metrics** | Computes FA/MD/RD/AD along tracts and generates reports |

---

## Example Output

A successful run looks like this:

```
======================================================================
PIPELINE COMPLETE
======================================================================
Subject ID:     sub-01
Total time:     3.3 minutes (201 seconds)

Step timing:
  ✓ check       : 0.0s
  ✓ import      : 0.0s
  ✓ preprocess  : 7.7s
  ✓ track       : 72.2s
  ✓ extract     : 110.9s
  ✓ metrics     : 10.1s

✓ All steps completed successfully!
======================================================================
```

---

## Output Structure

csttool writes a **BIDS derivatives dataset** at `--out`. No extra flags required.

    output/
    ├── dataset_description.json
    ├── participants.tsv
    ├── participants.json
    ├── sub-01/
    │   ├── dwi/
    │   │   ├── sub-01_space-orig_desc-preproc_dwi.nii.gz
    │   │   ├── sub-01_space-orig_desc-preproc_dwi.bval
    │   │   ├── sub-01_space-orig_desc-preproc_dwi.bvec
    │   │   ├── sub-01_space-orig_model-DTI_param-FA_dwimap.nii.gz
    │   │   ├── sub-01_space-orig_model-DTI_param-MD_dwimap.nii.gz
    │   │   ├── sub-01_space-orig_model-DTI_param-RD_dwimap.nii.gz
    │   │   ├── sub-01_space-orig_model-DTI_param-AD_dwimap.nii.gz
    │   │   └── tractography/
    │   │       ├── sub-01_space-orig_desc-wholebrain_tractogram.trk
    │   │       ├── sub-01_space-orig_desc-CSTleft_tractogram.trk
    │   │       ├── sub-01_space-orig_desc-CSTright_tractogram.trk
    │   │       └── sub-01_space-orig_desc-CSTbilateral_tractogram.trk
    │   ├── figures/                        (if --save-visualizations)
    │   │   ├── sub-01_stage-preproc_qc-brainmask.png
    │   │   ├── sub-01_stage-tracking_qc-tensormaps.png
    │   │   ├── sub-01_stage-extraction_qc-registration.png
    │   │   └── ...
    │   └── reports/
    │       ├── sub-01_report.html
    │       ├── sub-01_report.pdf           (if --generate-pdf)
    │       ├── sub-01_metrics.json
    │       └── sub-01_metrics.csv
    └── sub-01_pipeline_report.json

See [Output formats](../reference/output-formats.md) for a full description of every file.

---

## Key Options

| Flag | Description |
|------|-------------|
| `--nifti` / `--dicom` | Input format (provide path to data) |
| `--out` | Output directory |
| `--denoise-method` | `nlmeans` (default), `patch2self`, or `mppca` |
| `--generate-pdf` | Generate PDF report |
| `--save-visualizations` | Save QC visualizations at each step |
| `--subject-id` | Subject ID |
| `--verbose` | Verbose output |

Run `csttool run --help` for all options.

---

## What's Next?

- **[Data Requirements](data-requirements.md)** — Input format specifications and recommended datasets
- **[Troubleshooting](../how-to/troubleshooting.md)** — Common issues and fixes
- **[CLI Reference](../reference/cli/run.md)** — Full `csttool run` documentation
