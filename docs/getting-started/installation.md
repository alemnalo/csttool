# Installation

## Requirements

- **Python 3.10+**
- **System libraries** for PDF generation (WeasyPrint)

---

## Quick Install

```bash
pip install git+https://github.com/alemnalo/csttool.git
```

---

## System Dependencies

### dcm2niix (recommended)

`dcm2niix` is the primary DICOM converter used by `csttool import` and `csttool run`. It handles
Siemens, GE, Philips, and Hitachi scanners and generates BIDS-compliant JSON sidecars automatically.
csttool falls back to `dicom2nifti` when `dcm2niix` is absent, but the fallback is less reliable
for non-Siemens vendors.

=== "Ubuntu/Debian"

    ```bash
    sudo apt install dcm2niix
    ```

=== "macOS"

    ```bash
    brew install dcm2niix
    ```

=== "conda"

    ```bash
    conda install -c conda-forge dcm2niix
    ```

### WeasyPrint

csttool uses [WeasyPrint](https://weasyprint.org/) for PDF report generation, which requires system-level libraries.

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

    Install [GTK3 runtime](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer).

!!! warning "WeasyPrint installation issues"
    If you encounter errors during installation or PDF generation, see the 
    [WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation).

---

## Development Install

For contributing or local development:

```bash
# Clone repository
git clone https://github.com/alemnalo/csttool.git
cd csttool

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in editable mode with test dependencies
pip install -e ".[test]"
```

---

## Verify Installation

```bash
# Check CLI is accessible
csttool --version

# Verify environment and dependencies
csttool check
```

Expected output:

```text
csttool version 0.5.0
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

| Package | Purpose |
| --- | --- |
| `dipy` | Diffusion MRI processing, tractography |
| `nibabel` | NIfTI file I/O |
| `nilearn` | Neuroimaging utilities, atlas handling |
| `pydicom` | Read DICOM metadata (PatientID, age, scanner info) |
| `dicom2nifti` | DICOM to NIfTI conversion (fallback) |
| `numpy`, `scipy` | Numerical computing |
| `matplotlib` | Visualizations |
| `weasyprint`, `jinja2` | HTML/PDF report generation |

---

## Next Steps

- [Quick Start Guide](quickstart.md) — Run your first CST analysis
- [Data Requirements](data-requirements.md) — Prepare your input data
