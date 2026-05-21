# Installation

## Requirements

- **Python 3.10+** (handled automatically by the conda install path)
- **dcm2niix** ≥ 1.0.20220720 for DICOM import (optional; a `dicom2nifti` fallback is bundled)
- **WeasyPrint** + Cairo/Pango (optional; only needed for PDF reports — HTML reports work without them)

---

## Recommended: conda (Linux, macOS, Windows)

Conda handles the Python stack, the `dcm2niix` binary, and the native libraries (Cairo, Pango, GDK-PixBuf) that the PDF backend depends on. This is the only install path that gives full PDF functionality on Windows out of the box.

```bash
git clone https://github.com/alemnalo/csttool.git
cd csttool
conda env create -f environment.yml
conda activate csttool
csttool fetch-data --accept-fsl-license
```

That's the entire install. Skip to [Verify Installation](#verify-installation).

---

## Advanced: pip

Use this path if you already manage Python environments yourself and don't want conda.

```bash
pip install csttool             # core pipeline + HTML reports
pip install 'csttool[reports]'  # adds PDF rendering
```

You will also need to install `dcm2niix` separately — see [System Dependencies](#system-dependencies) below.

!!! warning "Windows + pip + PDF reports"
    On Windows, `pip install 'csttool[reports]'` installs `weasyprint` but **not** its native dependencies (Cairo, Pango, GDK-PixBuf). PDF generation will fail at runtime with a missing-DLL error. Use the conda install path if you need PDF reports on Windows.

    HTML reports work on all platforms regardless of install path.

---

## System Dependencies

### dcm2niix

Primary DICOM converter used by `csttool import` and `csttool run`. Handles Siemens, GE, Philips, and Hitachi scanners and generates BIDS-compliant JSON sidecars automatically. csttool falls back to `dicom2nifti` when `dcm2niix` is absent, but the fallback is less reliable for non-Siemens vendors and does not produce sidecars.

Minimum version: **1.0.20220720**.

=== "Ubuntu/Debian"

    ```bash
    sudo apt install dcm2niix
    # Check version — apt may ship older releases:
    dcm2niix -v
    ```

=== "macOS"

    ```bash
    brew install dcm2niix
    ```

=== "conda"

    ```bash
    conda install -c conda-forge 'dcm2niix>=1.0.20220720'
    ```

=== "Windows"

    Use the conda install path above, or download a release binary from the
    [dcm2niix GitHub releases](https://github.com/rordenlab/dcm2niix/releases) and add it to PATH.

### WeasyPrint native libraries (pip path only)

The conda install path handles these automatically. If you installed csttool via pip and want PDF reports, you need Cairo, Pango, and GDK-PixBuf on the system.

=== "Ubuntu/Debian"

    ```bash
    sudo apt install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libcairo2
    ```

=== "Fedora/RHEL"

    ```bash
    sudo dnf install pango gdk-pixbuf2 cairo libffi-devel
    ```

=== "macOS"

    ```bash
    brew install pango gdk-pixbuf cairo libffi
    ```

=== "Windows"

    Not officially supported via pip. Use the conda install path instead — it bundles the native libraries cleanly.

    If you must use pip on Windows, you can try the [GTK3 runtime installer](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer), but this path is fragile. PDF generation will fall back to a clear error message and the HTML report will still be produced.

!!! info "Graceful degradation"
    If WeasyPrint or its native libraries are missing, `csttool` skips PDF generation with an actionable message pointing at the HTML report. The rest of the pipeline runs unaffected.

---

## Development Install

For contributing or local development:

```bash
git clone https://github.com/alemnalo/csttool.git
cd csttool

# Option A: conda env (handles all native deps)
conda env create -f environment.yml
conda activate csttool

# Option B: pip + venv
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
pip install -e '.[reports,test]'
```

---

## Verify Installation

```bash
# Check CLI is accessible
csttool --version

# Verify environment and dependencies
csttool doctor
```

---

## Download Atlas Data

CST extraction requires FSL-licensed anatomical atlases. Download them using:

```bash
csttool fetch-data
```

This will:

- Display the FSL non-commercial license terms
- Prompt for acceptance
- Download FMRIB58_FA template and Harvard-Oxford atlases (~2 MB)
- Verify checksums and install to user data directory

!!! info "License Requirement"
    The FSL atlases are licensed for **non-commercial use only**. By downloading, you confirm your use is for academic research or educational purposes. For commercial use, you must obtain a commercial license from the University of Oxford.

For automated/scripted installations:

```bash
csttool fetch-data --accept-fsl-license
```

See the [fetch-data reference](../reference/cli/fetch-data.md) for detailed information.

---

## Python Dependencies

csttool installs the following packages automatically:

| Package | Purpose | Install group |
| --- | --- | --- |
| `dipy` | Diffusion MRI processing, tractography | core |
| `nibabel` | NIfTI file I/O | core |
| `nilearn` | Neuroimaging utilities, atlas handling | core |
| `pydicom` | Read DICOM metadata (PatientID, age, scanner info) | core |
| `dicom2nifti` | DICOM to NIfTI conversion (fallback) | core |
| `numpy`, `scipy` | Numerical computing | core |
| `matplotlib` | Visualizations | core |
| `jinja2` | HTML report templating | core |
| `weasyprint` | PDF report rendering (optional) | `[reports]` |

---

## Next Steps

- [Quick Start Guide](quickstart.md) — Run your first CST analysis
- [Data Requirements](data-requirements.md) — Prepare your input data
