# `csttool batch`

Process multiple subjects in batch mode. Runs the full pipeline for each subject in
parallel, with fault isolation, resume support, and optional BIDS derivatives output.

---

## Usage

=== "Manifest mode"

    ```bash
    csttool batch \
        --manifest study_manifest.json \
        --out /data/derivatives/csttool \
        --bids-out /data/derivatives/csttool \
        --preprocessing \
        --denoise-method patch2self
    ```

=== "BIDS auto-discovery"

    ```bash
    csttool batch \
        --bids-dir /data/bids \
        --out /data/derivatives/csttool \
        --bids-out /data/derivatives/csttool \
        --include "sub-0*" \
        --exclude "sub-099"
    ```

---

## Input modes

### Manifest

A JSON manifest explicitly defines subjects, their input files, and per-subject options.
Recommended for reproducibility.

    {
      "global_options": {
        "preprocessing": true,
        "denoise_method": "patch2self"
      },
      "subjects": [
        {
          "id": "sub-001",
          "nifti": "/data/bids/sub-001/ses-01/dwi/sub-001_ses-01_dwi.nii.gz"
        },
        {
          "id": "sub-002",
          "dicom": "/raw/sub-002/dicoms/",
          "session": "ses-01"
        }
      ]
    }

### BIDS auto-discovery

Scans a BIDS directory for `sub-*/[ses-*/]dwi/` directories and processes each one.
Use `--include` and `--exclude` glob patterns to filter subjects.

---

## BIDS derivatives output

Pass `--bids-out` to write a BIDS derivatives dataset at the specified path. After the
batch completes, csttool writes:

- `dataset_description.json` (once, skipped if already present)
- `participants.tsv` (one row per successful subject, file-locked for concurrent writes)
- `participants.json` (column definitions, once)

The per-subject output tree under `--bids-out` follows the same layout as `csttool run`.
See [Output formats](../output-formats.md) for details.

!!! tip "Recommended path"
    Place derivatives inside the raw BIDS dataset so that provenance resolves correctly:

        --bids-out /data/bids/derivatives/csttool

---

## Architecture

1. **Orchestrator** — iterates subjects, checks completion markers (`_done.json`),
   acquires per-subject file locks, and spawns worker processes.
2. **Worker** — runs the full pipeline in an isolated process; crashes do not affect
   other subjects.
3. **Finalization** — on success, moves `_work/` content to the final subject directory
   and writes `_done.json`.

---

## Options

### Input

| Flag | Description |
| --- | --- |
| `--manifest <file>` | Path to JSON manifest |
| `--bids-dir <dir>` | BIDS directory for auto-discovery |
| `--out <dir>` | Output root directory |
| `--bids-out <dir>` | BIDS derivatives root (writes `dataset_description.json` and `participants.tsv`) |
| `--include <patterns>` | Subject ID glob patterns to include (auto-discovery only) |
| `--exclude <patterns>` | Subject ID glob patterns to exclude |

### Processing

| Flag | Description |
| --- | --- |
| `--preprocessing` / `--no-preprocessing` | Enable or skip preprocessing (default: enabled) |
| `--denoise-method` | `mppca` (default), `patch2self`, `nlmeans`, or `none` |
| `--generate-pdf` | Generate PDF report for each subject |

### Pipeline control

| Flag | Description |
| --- | --- |
| `--force` | Re-process all subjects, ignoring completion markers |
| `--dry-run` | Show execution plan without processing |
| `--validate-only` | Run pre-flight validation and exit |
| `--keep-work` | Retain `_work/` directories after success |
| `--timeout-minutes <n>` | Per-subject timeout (default: 120) |
| `--verbose` | Detailed output |
| `--quiet` | Suppress progress messages |

---

## Related

- [Output formats](../output-formats.md) — per-subject output layout
- [`csttool run`](run.md) — single-subject pipeline
- [`csttool import`](import.md) — DICOM to raw BIDS conversion
