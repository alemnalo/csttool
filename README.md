# csttool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://alemnalo.github.io/csttool/)

`csttool` is a command-line tool for automated assessment of the corticospinal tract (CST) from diffusion-weighted MRI (DWI/DTI) data. It runs a full analysis pipeline — from DICOM import through tractography, CST extraction, and quantitative metrics — and produces BIDS-aligned derivative outputs.

## What it does

The pipeline runs in six sequential steps:

| Step | Command | Description |
|------|---------|-------------|
| 1 | `doctor` | Verify Python environment and external dependencies |
| 2 | `import` | Convert DICOM to NIfTI, or validate an existing NIfTI dataset |
| 3 | `preprocess` | Denoise (patch2self or NLMEANS), skull-strip, optional motion correction and Gibbs unringing |
| 4 | `track` | Whole-brain deterministic tractography using a CSA ODF model |
| 5 | `extract` | Atlas-based bilateral CST extraction via MNI152 registration and Harvard-Oxford ROI masks |
| 6 | `metrics` | Per-side and bilateral metrics (FA, MD, morphology, tract profiles, asymmetry indices) with PDF/HTML reports |

Steps can be run individually or as a single `run` command. A `batch` command supports multi-subject processing via a BIDS dataset or JSON manifest.

## Inputs

- **DICOM directory** — study root or series directory (converted with `dcm2niix`)
- **NIfTI dataset** — `.nii` / `.nii.gz` + bval + bvec files (DICOM conversion skipped)
- Optional: BIDS JSON sidecar, subject/session labels, field strength, echo time overrides

## Outputs

All outputs are written to a BIDS-aligned derivative directory:

- Preprocessed DWI (denoised, brain-masked)
- Scalar maps: FA, MD
- Whole-brain tractogram (`.trk`)
- Left and right CST tractograms (`.trk`)
- Per-subject metrics in JSON and TSV
- HTML and PDF clinical reports
- QC visualizations (optional)

## Installation

### Recommended: conda (Linux, macOS, Windows)

Conda handles both the Python stack and the external `dcm2niix` binary, plus the native libraries (Cairo, Pango, GDK-PixBuf) that the PDF report backend depends on. This is the only path that gives full functionality on Windows out of the box.

```bash
git clone https://github.com/alemnalo/csttool.git
cd csttool
conda env create -f environment.yml
conda activate csttool
csttool fetch-data --accept-fsl-license
```

### Advanced: pip

```bash
pip install csttool             # core pipeline + HTML reports
pip install 'csttool[reports]'  # adds PDF rendering (Linux/macOS)
```

On Windows, `pip install 'csttool[reports]'` will install `weasyprint` but its native dependencies (Cairo, Pango) are not bundled — PDF generation will fail at runtime with a missing-DLL error. Use the conda install path on Windows if you need PDF reports. HTML reports work on all platforms regardless.

You will also need to install `dcm2niix` separately:

```bash
sudo apt install dcm2niix        # Debian / Ubuntu (check version >= 1.0.20220720)
brew install dcm2niix            # macOS
conda install -c conda-forge dcm2niix    # any platform
```

If `dcm2niix` is not on `PATH`, csttool falls back to `dicom2nifti`. The fallback does not produce BIDS JSON sidecars and is less reliable for non-Siemens data.

### Atlas data (required for extraction)

```bash
csttool fetch-data --accept-fsl-license
```

Downloads the FMRIB58_FA template and Harvard-Oxford atlas under the FSL non-commercial license.

### Development install

```bash
git clone https://github.com/alemnalo/csttool.git
cd csttool
pip install -e '.[reports,test]'
```

## Usage

**Full pipeline from DICOM:**

```bash
csttool run --dicom /path/to/dicom --out /path/to/output --subject-id sub-01 --save-visualizations --generate-pdf
```

**Full pipeline from existing NIfTI:**

```bash
csttool run --nifti sub-01_dwi.nii.gz --out /path/to/output --subject-id sub-01
```

**Run individual steps:**

```bash
csttool preprocess --dwi sub-01_dwi.nii.gz --out preproc/
csttool track --nifti preproc/sub-01_dwi_preproc.nii.gz --out tracking/
csttool extract --tractogram tracking/sub-01_tractogram.trk --nifti preproc/sub-01_dwi_preproc.nii.gz --out extract/
csttool metrics --cst-left extract/sub-01_cst-left.trk --cst-right extract/sub-01_cst-right.trk --out metrics/
```

**Batch processing (BIDS dataset):**

```bash
csttool batch --bids /path/to/bids --out /path/to/derivatives
```

**Batch processing (manifest):**

```bash
csttool batch --manifest subjects.json --out /path/to/derivatives
```

**Assess DWI data quality before running:**

```bash
csttool check-dataset --dwi sub-01_dwi.nii.gz --verbose
```

## Testing

```bash
pytest tests/
```
