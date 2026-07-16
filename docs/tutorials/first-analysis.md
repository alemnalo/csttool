# Your First CST Analysis

A complete walkthrough: from a single raw DWI acquisition to a PDF metrics report. Allow 30–60 minutes of wall-clock time on a modern workstation.

## Prerequisites

- `csttool` installed and available on the `PATH`. See [Installation](../getting-started/installation.md).
- A DWI acquisition with whole-brain coverage including the brainstem, with at least 30 gradient directions and `.bval` / `.bvec` sidecars. See [Data requirements](../getting-started/data-requirements.md).
- ~5 GB of free disk space for intermediate outputs.

Throughout this tutorial, `~/data/sub-001/dwi/` stands for the directory containing your raw subject; substitute your own path.

## Step 0 — Verify the environment

```bash
csttool doctor
csttool fetch-data --accept-fsl-license
```

`doctor` confirms that Python, DIPY, ANTs and supporting libraries are present. `fetch-data` downloads the FMRIB58_FA and Harvard-Oxford atlases used for registration and ROI extraction. The download runs once per machine.

## Step 1 — Inspect the data

```bash
csttool check-dataset --dwi ~/data/sub-001/dwi/sub-001_dwi.nii.gz
```

You should see a quality summary covering b-values, gradient direction coverage, voxel size and brain-coverage estimate. Resolve any **error**-severity findings before continuing; **warnings** are informational. See the [`check-dataset` reference](../reference/cli/check_dataset.md) for the meaning of each line.

## Step 2 — Run the full pipeline

The simplest invocation chains every stage end-to-end:

```bash
csttool run \
    --nifti ~/data/sub-001/dwi/sub-001_dwi.nii.gz \
    --out  ~/derivatives \
    --subject-id sub-001 \
    --denoise-method patch2self \
    --perform-motion-correction \
    --bids-out \
    --generate-pdf
```

What each flag does, briefly:

- `--bids-out` — write outputs in BIDS-Derivatives layout under `~/derivatives/csttool/sub-001/`.
- `--denoise-method patch2self` — self-supervised denoising. The default is `mppca`, which estimates its own noise level; `patch2self` suits modern high-volume acquisitions, and `nlmeans` acquisitions with <30 volumes.
- `--perform-motion-correction` — affine align all volumes to the first b=0.
- `--generate-pdf` — render the HTML report to PDF (requires WeasyPrint; see [Troubleshooting](../how-to/troubleshooting.md#weasyprint-fails-to-install-or-render-pdfs) if this fails).

The run prints stage banners as it progresses: `CHECK → IMPORT → PREPROCESS → TRACK → EXTRACT → METRICS`.

## Step 3 — Read the report

Open the PDF:

```bash
xdg-open ~/derivatives/csttool/sub-001/metrics/sub-001_report.pdf
```

The report contains:

1. **Acquisition header** — subject ID, scanner field strength, voxel size, b-values, number of directions.
2. **Hemispheric metrics table** — left and right CST: mean FA, MD, RD, AD, streamline count, volume.
3. **Bilateral comparison** — laterality index per metric and a profile-correlation score.
4. **Tract profiles** — line plots of each scalar along the tract from brainstem to motor cortex.
5. **QC overlays** — axial, sagittal and coronal slices with the extracted bundles overlaid on the FA map.

A symmetric subject typically shows |LI| < 0.05 for FA. Larger asymmetries warrant inspection of the QC overlays — most often they reveal a registration glitch rather than a true biological finding.

!!! tip "Interpreting the profiles"
    The tract profile is resampled to 100 equidistant points along each streamline, so the x-axis spans the bundle from one endpoint to the other. A consistent left/right offset across the whole profile suggests global tract-wide difference; a localised dip points at a specific anatomical level.

## Step 4 — Inspect intermediate outputs (optional)

If something in the report looks wrong, follow the chain back:

- `~/derivatives/csttool/sub-001/preproc/dti_preproc.nii.gz` — denoised, motion-corrected DWI.
- `~/derivatives/csttool/sub-001/tracking/whole_brain.trk` — whole-brain deterministic tractogram.
- `~/derivatives/csttool/sub-001/extract/cst_{left,right}.trk` — extracted bilateral CST.
- `~/derivatives/csttool/sub-001/extract/visualizations/registration_qc.png` — atlas-to-subject registration overlay.

Load the `.trk` files in [MI-Brain](https://www.imeka.ca/mi-brain/) or [Trackvis](http://trackvis.org/) for visual inspection.

## What next

- [Batch processing](batch-processing.md) — run the same pipeline over multiple subjects.
- [Troubleshooting](../how-to/troubleshooting.md) — common pitfalls.
- [Tractography](../explanation/tractography.md) — what is happening inside the `track` and `extract` stages.
