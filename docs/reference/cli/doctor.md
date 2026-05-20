# doctor Command

The `doctor` command runs comprehensive environment diagnostics and reports issues with actionable fix instructions. It replaces the deprecated `csttool check`.

## Usage

```bash
csttool doctor
```

## Checks Performed

| Section | What is checked |
|---|---|
| **Python packages** | All core dependencies imported and version-stamped |
| **External tools** | `dcm2niix` on PATH, version ≥ 1.0.20220720 |
| **Atlas data** | FMRIB58_FA and Harvard-Oxford atlases present and valid |
| **System** | User data and cache directories writable; CPU count and OpenMP state |
| **Report backend** | HTML (jinja2) always available; PDF (weasyprint + Cairo/Pango) probed and reported as optional |

## Example Output

```text
============================================================
csttool doctor
============================================================
  csttool:  0.5.0
  Python:   3.11.15
  Platform: Linux x86_64

Python packages
---------------
  ✓ numpy: 2.4.6
  ✓ scipy: 1.17.1
  ✓ cython: 3.2.4
  ✓ dipy: 1.12.1
  ✓ nibabel: 5.4.0
  ✓ nilearn: 0.13.1
  ✓ pydicom: 3.0.2
  ✓ dicom2nifti: unknown
  ✓ platformdirs: 4.9.6
  ✓ jinja2: 3.1.6

External tools
--------------
  ✓ dcm2niix: v1.0.20260416  [/usr/bin/dcm2niix]

Atlas data
----------
  ✓ FMRIB58_FA + Harvard-Oxford atlases: /home/user/.local/share/csttool

System
------
  ✓ User data dir: /home/user/.local/share/csttool
  ✓ Cache dir: /home/user/.cache/csttool
  ✓ CPUs: 8: OMP_NUM_THREADS not set (dipy uses all cores)

Report backend
--------------
  ✓ HTML (jinja2): available
  ✓ PDF (weasyprint): 67.0

============================================================
  ✓  Environment ready
```

## Interpreting Results

| Symbol | Meaning |
|---|---|
| `✓` | Check passed |
| `○` | Optional component — missing, but pipeline still runs |
| `✗` | Required component missing — pipeline will fail |

The PDF backend is always optional. Missing weasyprint or its native libraries will not fail `doctor`; the fix command is printed and HTML reports will still be produced.

## Common Issues

**Atlas data not installed:**
```bash
csttool fetch-data --accept-fsl-license
```

**dcm2niix too old or missing:**
```bash
conda install -c conda-forge 'dcm2niix>=1.0.20220720'
```

**PDF rendering fails on Windows:**
```bash
conda install -c conda-forge weasyprint pango cairo gdk-pixbuf
```

## See Also

- [Installation](../../getting-started/installation.md)
- [Troubleshooting](../../how-to/troubleshooting.md)
- [fetch-data](fetch-data.md)
