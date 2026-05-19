# csttool Documentation

Welcome to the csttool documentation! 

**csttool** is a Python-based command-line tool for automated assessment of the corticospinal tract (CST) using diffusion-weighted MRI (DW-MRI) data.

## Quick Start
```bash
# Install (recommended: conda)
git clone https://github.com/alemnalo/csttool.git
cd csttool
conda env create -f environment.yml
conda activate csttool
csttool fetch-data --accept-fsl-license

# Run complete pipeline
csttool run --dicom /path/to/dicom --out results --subject-id sub-01
```

See the [Installation Guide](getting-started/installation.md) for the pip path and Windows notes, or the [Quick Start Guide](getting-started/quickstart.md) for analysis examples.

## Pipeline Overview
```mermaid
graph LR
    A[DICOM / NIfTI] --> B[Import]
    B --> C[Preprocess]
    C --> D[Track]
    D --> E[Extract]
    E --> F[Metrics / Reports]
```

csttool processes diffusion MRI data through six stages:

1. **Check** - Verify environment and dependencies
2. **Import** - Convert DICOM to NIfTI format
3. **Preprocess** - Denoise and skull strip
4. **Track** - Generate whole-brain tractogram
5. **Extract** - Isolate bilateral CST using anatomical ROIs
6. **Metrics** - Compute microstructural measures and generate reports

## Documentation Structure

This documentation follows the [Diátaxis framework](https://diataxis.fr/).

### [Tutorials](tutorials/first-analysis.md)
*COMING SOON*

### [How-To Guides](how-to/multiple-subjects.md)
*COMING SOON*

### [Reference](reference/cli/check.md)
*COMING SOON*

### [Explanation](explanation/diffusion-mri.md)
*COMING SOON*

## License

csttool is released under the MIT License. See [LICENSE](https://github.com/alemnalo/csttool/blob/main/LICENSE) for details.