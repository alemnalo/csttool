# Parameters Reference

A single-page reference of every command-line flag accepted by `csttool`, grouped by command. For prose explanations of what each command does, see [CLI Overview](cli/overview.md) and the per-command pages.

Conventions:

- *type* `path` means an absolute or relative filesystem path (`pathlib.Path`).
- A flag listed without a default is **required** for that command.
- Boolean flags toggle on when present; default is *off* unless noted.

## `check`

Verify the Python environment and required dependencies.

| Flag | Type | Default | Description |
|---|---|---|---|
| *(no arguments)* | | | Runs version and import checks. |

## `check-dataset`

Assess DWI acquisition quality.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dwi` | path | — | DWI NIfTI file. **Required**. |
| `--bval` | path | auto | b-value sidecar. Auto-discovered if omitted. |
| `--bvec` | path | auto | b-vector sidecar. Auto-discovered if omitted. |
| `--json` | path | auto | BIDS JSON sidecar. Auto-discovered if omitted. |
| `--b0-threshold` | float | `50.0` | b-value below which a volume is treated as b=0. |
| `--verbose` | flag | off | Detailed acquisition-quality report. |

## `fetch-data`

Download FSL-licensed atlas data.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--accept-fsl-license` | flag | off | Acknowledge the FSL non-commercial licence and proceed. |

## `import`

Convert DICOM to NIfTI or validate an existing NIfTI dataset.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dicom` | path | — | DICOM input directory. Mutually exclusive with `--nifti`. |
| `--nifti` | path | — | Existing NIfTI input. Mutually exclusive with `--dicom`. |
| `--out` | path | — | Output directory. **Required**. |
| `--subject-id` | string | derived | Subject label written into BIDS names. |
| `--session-id` | string | — | Optional session label. |
| `--series` | int | — | DICOM series number to convert (omit to auto-select). |
| `--scan-only` | flag | off | Inspect input without writing anything. |
| `--field-strength` | float | from header | Override scanner field strength (Tesla). |
| `--echo-time` | float | from header | Override echo time. |
| `--raw-bids` | flag | off | Emit a raw-BIDS dataset (vs. derivatives layout). |
| `--keep-phi` | flag | off | Do not strip PHI tags from DICOM during conversion. |
| `--verbose` | flag | off | Detailed conversion log. |

## `preprocess`

Denoise, unring, motion-correct and brain-mask DWI.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--nifti` | path | — | DWI NIfTI input. Mutually exclusive with `--dicom`. |
| `--dicom` | path | — | DICOM input (converted internally). |
| `--out` | path | — | Output directory. **Required**. |
| `--denoise-method` | `nlmeans` \| `patch2self` \| `mppca` | `nlmeans` | Denoising algorithm. |
| `--coil-count` | int | `4` | NLMeans noise estimation parameter. |
| `--unring` | flag | off | Apply Kellner Gibbs-unringing. |
| `--perform-motion-correction` | flag | off | Affine-register volumes to first b=0. |
| `--target-voxel-size` | 3×float | — | Reslice to given isotropic voxel size in mm. |
| `--save-visualizations` | flag | off | Write QC plots. |
| `--verbose` | flag | off | |

## `track`

Whole-brain deterministic tractography (CSA-ODF).

| Flag | Type | Default | Description |
|---|---|---|---|
| `--nifti` | path | — | Preprocessed DWI. **Required**. |
| `--out` | path | — | Output directory. **Required**. |
| `--subject-id` | string | derived | Subject label. |
| `--fa-thr` | float | `0.2` | FA stopping threshold. |
| `--seed-density` | int | `1` | Seeds per voxel (3D grid). |
| `--step-size` | float | `0.5` | Streamline integration step in mm. |
| `--sh-order` | int | `6` | Spherical-harmonic order for CSA-ODF. |
| `--rng-seed` | int | `42` | Random seed for deterministic tracking. |
| `--random` | flag | off | Disable seeding (non-deterministic run). |
| `--use-brain-mask-stop` | flag | off | Use brain mask as an additional stopping criterion. |
| `--show-plots` | flag | off | Interactive plots during tracking. |
| `--verbose` | flag | off | |

## `extract`

Atlas-based ROI filtering for bilateral CST.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--tractogram` | path | — | Whole-brain `.trk` from `track`. **Required**. |
| `--fa` | path | — | FA map for registration target. **Required**. |
| `--out` | path | — | Output directory. **Required**. |
| `--subject-id` | string | `subject` | Subject label. |
| `--extraction-method` | `endpoint` \| `passthrough` | `endpoint` | ROI filtering strategy. |
| `--dilate-brainstem` | int | `2` | Brainstem ROI dilation (voxels). |
| `--dilate-motor` | int | `1` | Motor-cortex ROI dilation (voxels). |
| `--min-length` | float | `20.0` | Minimum streamline length (mm). |
| `--max-length` | float | `200.0` | Maximum streamline length (mm). |
| `--fast-registration` | flag | off | Use a faster, lower-precision ANTs registration profile. |
| `--save-visualizations` | flag | off | Write QC images. |
| `--skip-coordinate-validation` | flag | off | Bypass affine sanity checks (debugging only). |
| `--verbose` | flag | off | |
| `--quiet` | flag | off | Suppress all but errors. |

## `metrics`

Bilateral CST metrics and reports.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cst-left` | path | — | Left CST `.trk`. **Required**. |
| `--cst-right` | path | — | Right CST `.trk`. **Required**. |
| `--out` | path | — | Output directory. **Required**. |
| `--fa` | path | — | FA map (needed for profiles). |
| `--md` | path | — | MD map. |
| `--rd` | path | — | RD map. |
| `--ad` | path | — | AD map. |
| `--subject-id` | string | `subject` | Subject label. |
| `--space` | string | — | Coordinate space label recorded in metadata. |
| `--generate-pdf` | flag | off | Render the HTML report to PDF. |
| `--verbose` | flag | off | |

## `validate`

Compare extracted bundles against reference tractograms.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cand-left` | path | — | Candidate left CST. **Required**. |
| `--cand-right` | path | — | Candidate right CST. **Required**. |
| `--ref-left` | path | — | Reference left CST. **Required**. |
| `--ref-right` | path | — | Reference right CST. **Required**. |
| `--ref-space` | path | — | Reference-space NIfTI used to rasterize streamlines. **Required**. |
| `--output-dir` | path | `./validation_output` | Output directory. |
| `--visualize` | flag | off | Save overlay images. |
| `--disable-hemisphere-check` | flag | off | Skip the left/right hemisphere consistency check. |

## `run`

Full single-subject pipeline. Accepts the union of `import`, `preprocess`, `track`, `extract` and `metrics` flags, plus orchestration controls. Notable additions:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--bids-out` | flag | off | Emit a BIDS-Derivatives output layout. |
| `--generate-pdf` | flag | off | Forwarded to `metrics`. |
| `--extraction-method` | `endpoint` \| `passthrough` \| `roi-seeded` \| `bidirectional` | `endpoint` | `run` exposes two additional methods not available in standalone `extract` (`roi-seeded`, `bidirectional`). |

See [`csttool run`](cli/run.md) for the full list.

## `batch`

Multi-subject orchestration.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--manifest` | path | — | JSON manifest of subjects. Mutually exclusive with `--bids-dir`. |
| `--bids-dir` | path | — | BIDS dataset root; subjects auto-discovered. |
| `--out` | path | — | Output directory. **Required**. |
| `--bids-out` | flag | off | BIDS-Derivatives layout. |
| `--include` | list | — | Subject IDs to include. |
| `--exclude` | list | — | Subject IDs to skip. |
| `--force` | flag | off | Re-run subjects with existing outputs. |
| `--dry-run` | flag | off | Print the planned actions without running. |
| `--validate-only` | flag | off | Validate manifest / BIDS discovery only. |
| `--keep-work` | flag | off | Keep intermediate work directories. |
| `--timeout-minutes` | int | — | Per-subject timeout. |
| `--preprocessing` / `--no-preprocessing` | flag | on | Toggle the preprocess stage. |
| `--denoise-method` | `nlmeans` \| `patch2self` \| `mppca` | `nlmeans` | Forwarded to `preprocess`. |
| `--generate-pdf` | flag | off | Forwarded to `metrics`. |
| `--verbose` / `--quiet` | flag | off | |
