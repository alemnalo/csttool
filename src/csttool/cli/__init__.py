
from __future__ import annotations

import argparse
from pathlib import Path

from .. import __version__
from ..defaults import DEFAULT_DENOISE_METHOD
from .utils import add_io_arguments
from .commands.check import cmd_check
from .commands.doctor import cmd_doctor
from .commands.check_dataset import cmd_check_dataset
from .commands.import_cmd import cmd_import
from .commands.preprocess import cmd_preprocess
from .commands.track import cmd_track
from .commands.extract import cmd_extract
from .commands.metrics import cmd_metrics
from .commands.run import cmd_run
from .commands.batch import cmd_batch
from .commands.validate import add_validate_parser, cmd_validate
from .commands.fetch_data import cmd_fetch_data

def main() -> None:
    """Entrypoint for the csttool CLI."""
    parser = argparse.ArgumentParser(
        prog="csttool",
        description="CST assessment tool using DTI data."
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command")

    # -------------------------------------------------------------------------
    # doctor subtool
    # -------------------------------------------------------------------------
    p_doctor = subparsers.add_parser("doctor", help="Run comprehensive environment diagnostics")
    p_doctor.set_defaults(func=cmd_doctor)

    # -------------------------------------------------------------------------
    # check subtool (deprecated alias for doctor)
    # -------------------------------------------------------------------------
    p_check = subparsers.add_parser("check", help="Run environment checks (deprecated: use doctor)")
    p_check.set_defaults(func=cmd_check)

    # -------------------------------------------------------------------------
    # check-dataset subtool
    # -------------------------------------------------------------------------
    p_check_dataset = subparsers.add_parser(
        "check-dataset",
        help="Assess acquisition quality of a DWI dataset"
    )
    p_check_dataset.add_argument(
        "--dwi",
        type=Path,
        required=True,
        help="Path to DWI NIfTI file"
    )
    p_check_dataset.add_argument(
        "--bval",
        type=Path,
        help="Path to bval file (optional, guessed if not provided)"
    )
    p_check_dataset.add_argument(
        "--bvec",
        type=Path,
        help="Path to bvec file (optional, guessed if not provided)"
    )
    p_check_dataset.add_argument(
        "--json",
        type=Path,
        help="Path to BIDS JSON sidecar (optional)"
    )
    p_check_dataset.add_argument(
        "--b0-threshold",
        type=float,
        default=50.0,
        help="B-value threshold for b=0 volumes (default: 50 s/mm²)"
    )
    p_check_dataset.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed assessment info and per-shell analysis"
    )
    p_check_dataset.set_defaults(func=cmd_check_dataset)

    # -------------------------------------------------------------------------
    # import subtool (with ingest module support)
    # -------------------------------------------------------------------------
    p_import = subparsers.add_parser(
        "import",
        help="Import DICOM data or load NIfTI with series selection and validation"
    )
    p_import.add_argument(
        "--dicom",
        type=Path,
        help="Path to DICOM directory (can be study root with multiple series)"
    )
    p_import.add_argument(
        "--nifti",
        type=Path,
        help="Path to existing NIfTI file (skip DICOM conversion)"
    )
    p_import.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for converted files"
    )
    p_import.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject identifier for output naming"
    )
    p_import.add_argument(
        "--series",
        type=int,
        default=None,
        help="Series number to convert (as shown in scan output, 1-indexed)"
    )
    p_import.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan and analyze series, don't convert"
    )
    p_import.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
    p_import.add_argument(
        "--field-strength",
        type=float,
        default=None,
        help="MRI field strength in Tesla (e.g. 3.0). Overrides BIDS JSON if provided."
    )
    p_import.add_argument(
        "--echo-time",
        type=float,
        default=None,
        help="Echo time in milliseconds (e.g. 89.0). Overrides BIDS JSON if provided."
    )
    p_import.add_argument(
        "--raw-bids",
        type=Path,
        default=None,
        metavar="BIDS_DIR",
        help=(
            "Organise output as a raw BIDS dataset at BIDS_DIR. "
            "Subject and session labels are derived from DICOM PatientID and StudyDate "
            "(anonymised by default). Requires --dicom."
        )
    )
    p_import.add_argument(
        "--keep-phi",
        action="store_true",
        default=False,
        help=(
            "Disable anonymisation: use PatientID directly as the subject label. "
            "WARNING: the output dataset will contain protected health information (PHI)."
        )
    )
    p_import.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session label (without 'ses-' prefix) for --raw-bids. Overrides StudyDate."
    )
    p_import.set_defaults(func=cmd_import)

    # -------------------------------------------------------------------------
    # preprocess subtool
    # -------------------------------------------------------------------------
    p_preproc = subparsers.add_parser(
        "preprocess",
        help="Run preprocessing pipeline (denoise, skull strip, motion correction)"
    )
    add_io_arguments(p_preproc)
    p_preproc.add_argument(
        "--coil-count",
        type=int,
        default=4,
        help="Number of receiver coils for PIESNO noise estimation. Only used with --denoise-method nlmeans (ignored for patch2self and mppca)."
    )
    p_preproc.add_argument(
        "--denoise-method",
        type=str,
        default=DEFAULT_DENOISE_METHOD,
        choices=["nlmeans", "patch2self", "mppca"],
        help=f"Denoising method: nlmeans, patch2self, or mppca (default: {DEFAULT_DENOISE_METHOD})."
    )
    p_preproc.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save QC visualizations to output directory."
    )
    p_preproc.add_argument(
        "--perform-motion-correction",
        action="store_true",
        help="Enable between volume motion correction (disabled by default)."
    )
    p_preproc.add_argument(
        "--unring",
        action="store_true",
        help="Enable Gibbs unringing (disabled by default)."
    )
    p_preproc.add_argument(
        "--target-voxel-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Target voxel size in mm (x, y, z). If provided, data will be resliced to this voxel size."
    )
    p_preproc.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information."
    )
    p_preproc.set_defaults(func=cmd_preprocess)

    # -------------------------------------------------------------------------
    # track subtool
    # -------------------------------------------------------------------------
    p_track = subparsers.add_parser(
        "track",
        help="Run whole-brain deterministic tractography on preprocessed data"
    )
    p_track.add_argument(
        "--nifti",
        type=Path,
        required=True,
        help="Path to preprocessed NIfTI (.nii or .nii.gz).",
    )
    p_track.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject identifier for output naming (default: extracted from filename)."
    )
    p_track.add_argument(
        "--fa-thr",
        type=float,
        default=0.2,
        help="FA threshold for stopping and seeding (default 0.2).",
    )
    p_track.add_argument(
        "--seed-density",
        type=int,
        default=1,
        help="Seeds per voxel in the seed mask (default 1).",
    )
    p_track.add_argument(
        "--step-size",
        type=float,
        default=0.5,
        help="Tracking step size in millimetres (default 0.5).",
    )
    p_track.add_argument(
        "--sh-order",
        type=int,
        default=6,
        help="Maximum spherical harmonic order for CSA ODF model (default 6).",
    )
    p_track.add_argument(
        "--show-plots",
        action="store_true",
        help="Enable QC plots for segmentation and tractography.",
    )
    p_track.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information."
    )
    p_track.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for reproducible tractography (default: 42). Use --random to disable."
    )
    p_track.add_argument(
        "--random",
        action="store_const",
        const=None,
        dest="rng_seed",
        help="Disable deterministic seeding (use random initialization)."
    )
    p_track.add_argument(
        "--use-brain-mask-stop",
        action="store_true",
        help="Stop tracking at brain mask boundary in addition to FA threshold."
    )
    p_track.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for tractogram and scalar maps.",
    )
    p_track.set_defaults(func=cmd_track)

    # -------------------------------------------------------------------------
    # extract subtool
    # -------------------------------------------------------------------------
    p_extract = subparsers.add_parser(
        "extract",
        help="Extract bilateral CST from whole-brain tractogram using atlas-based ROI filtering"
    )
    p_extract.add_argument(
        "--tractogram",
        type=Path,
        required=True,
        help="Path to whole-brain tractogram (.trk)"
    )
    p_extract.add_argument(
        "--fa",
        type=Path,
        required=True,
        help="FA map for registration and analysis"
    )
    p_extract.add_argument(
        "--subject-id",
        type=str,
        default="subject",
        help="Subject identifier for output naming"
    )
    p_extract.add_argument(
        "--dilate-brainstem",
        type=int,
        default=2,
        help="Dilation iterations for brainstem ROI (default: 2)"
    )
    p_extract.add_argument(
        "--dilate-motor",
        type=int,
        default=1,
        help="Dilation iterations for motor cortex ROI (default: 1)"
    )
    p_extract.add_argument(
        "--min-length",
        type=float,
        default=20.0,
        help="Minimum streamline length in mm (default: 20)"
    )
    p_extract.add_argument(
        "--max-length",
        type=float,
        default=200.0,
        help="Maximum streamline length in mm (default: 200)"
    )
    # === NEW ARGUMENTS ===
    p_extract.add_argument(
        "--extraction-method",
        type=str,
        choices=["endpoint", "passthrough"],
        default="passthrough",
        help="CST extraction method: endpoint (strict) or passthrough (permissive). Default: passthrough"
    )
    # === END NEW ARGUMENTS ===
    p_extract.add_argument(
        "--fast-registration",
        action="store_true",
        help="Use faster but less accurate registration"
    )
    p_extract.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save QC visualizations"
    )
    p_extract.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
    p_extract.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages (errors still shown)"
    )
    p_extract.add_argument(
        "--skip-coordinate-validation",
        action="store_true",
        help="Skip coordinate validation (NOT RECOMMENDED - can cause silent failures)"
    )
    p_extract.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for extracted CST tractograms"
    )
    p_extract.set_defaults(func=cmd_extract)

    # -------------------------------------------------------------------------
    # metrics subtool
    # -------------------------------------------------------------------------
    p_metrics = subparsers.add_parser(
        "metrics",
        help="Compute bilateral CST metrics and generate reports"
    )
    p_metrics.add_argument(
        "--cst-left",
        type=Path,
        required=True,
        help="Path to left CST tractogram (.trk)"
    )
    p_metrics.add_argument(
        "--cst-right",
        type=Path,
        required=True,
        help="Path to right CST tractogram (.trk)"
    )
    p_metrics.add_argument(
        "--fa",
        type=Path,
        help="FA map for microstructural analysis"
    )
    p_metrics.add_argument(
        "--md",
        type=Path,
        help="MD map for microstructural analysis"
    )
    p_metrics.add_argument(
        "--rd",
        type=Path,
        help="RD (radial diffusivity) map for microstructural analysis"
    )
    p_metrics.add_argument(
        "--ad",
        type=Path,
        help="AD (axial diffusivity) map for microstructural analysis"
    )
    p_metrics.add_argument(
        "--subject-id",
        type=str,
        default="subject",
        help="Subject identifier for reports"
    )
    p_metrics.add_argument(
        "--space",
        type=str,
        default="Native Space",
        help="Space declaration for report header (default: Native Space)"
    )
    p_metrics.add_argument(
        "--generate-pdf",
        action="store_true",
        help="Generate PDF clinical report"
    )
    p_metrics.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information."
    )
    p_metrics.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for metrics and reports"
    )
    p_metrics.set_defaults(func=cmd_metrics)

    # -------------------------------------------------------------------------
    # run subtool - COMPLETE PIPELINE
    # -------------------------------------------------------------------------
    p_run = subparsers.add_parser(
        "run",
        help="Run complete pipeline: check → import → preprocess → track → extract → metrics"
    )
    
    # Input options
    p_run.add_argument(
        "--dicom",
        type=Path,
        help="Path to DICOM directory"
    )
    p_run.add_argument(
        "--nifti",
        type=Path,
        help="Path to existing NIfTI file (skip import step)"
    )
    p_run.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (subdirectories created for each step)"
    )
    p_run.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject identifier for all outputs"
    )
    p_run.add_argument(
        "--space",
        type=str,
        default="Native Space",
        help="Space declaration for report header (default: Native Space)"
    )
    
    # Import options
    p_run.add_argument(
        "--series",
        type=int,
        default=None,
        help="DICOM series number to convert (1-indexed)"
    )
    p_run.add_argument(
        "--field-strength",
        type=float,
        default=None,
        help="MRI field strength in Tesla (e.g. 3.0). Overrides BIDS JSON if provided."
    )
    p_run.add_argument(
        "--echo-time",
        type=float,
        default=None,
        help="Echo time in milliseconds (e.g. 89.0). Overrides BIDS JSON if provided."
    )
    
    # Preprocessing options
    p_run.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable preprocessing step (only denoising and brain masking by default). Default: Skipped."
    )
    p_run.add_argument(
        "--coil-count",
        type=int,
        default=4,
        help="Number of receiver coils for PIESNO. Only used with --denoise-method nlmeans (ignored for patch2self and mppca)."
    )
    p_run.add_argument(
        "--denoise-method",
        type=str,
        default=DEFAULT_DENOISE_METHOD,
        choices=["nlmeans", "patch2self", "mppca"],
        help=f"Denoising method: nlmeans, patch2self, or mppca (default: {DEFAULT_DENOISE_METHOD})"
    )
    p_run.add_argument(
        "--unring",
        action="store_true",
        help="Enable Gibbs unringing during preprocessing"
    )
    p_run.add_argument(
        "--perform-motion-correction",
        action="store_true",
        help="Enable motion correction during preprocessing"
    )
    p_run.add_argument(
        "--target-voxel-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Target voxel size in mm (x, y, z). If provided, data will be resliced to this voxel size."
    )
    
    # Tracking options
    p_run.add_argument(
        "--fa-thr",
        type=float,
        default=0.2,
        help="FA threshold for tracking (default: 0.2)"
    )
    p_run.add_argument(
        "--seed-density",
        type=int,
        default=1,
        help="Seeds per voxel (default: 1)"
    )
    p_run.add_argument(
        "--step-size",
        type=float,
        default=0.5,
        help="Tracking step size in mm (default: 0.5)"
    )
    p_run.add_argument(
        "--sh-order",
        type=int,
        default=6,
        help="Spherical harmonic order (default: 6)"
    )
    p_run.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for reproducible tracking and extraction (default: 42)."
    )

    # Extraction options
    p_run.add_argument(
        "--dilate-brainstem",
        type=int,
        default=2,
        help="Dilation iterations for brainstem ROI (default: 2)"
    )
    p_run.add_argument(
        "--dilate-motor",
        type=int,
        default=1,
        help="Dilation iterations for motor cortex ROI (default: 1)"
    )
    p_run.add_argument(
        "--min-length",
        type=float,
        default=20.0,
        help="Minimum streamline length in mm (default: 20)"
    )
    p_run.add_argument(
        "--max-length",
        type=float,
        default=200.0,
        help="Maximum streamline length in mm (default: 200)"
    )
    p_run.add_argument(
        "--extraction-method",
        type=str,
        choices=["endpoint", "passthrough", "roi-seeded", "bidirectional"],
        default="passthrough",
        help="CST extraction method: endpoint, passthrough, roi-seeded, or bidirectional "
             "(two-pass seeding with intersection — most symmetric). Default: passthrough"
    )
    p_run.add_argument(
        "--seed-fa-threshold",
        type=float,
        default=0.15,
        help="FA threshold for roi-seeded tracking (default: 0.15)"
    )

    # Metrics options
    p_run.add_argument(
        "--generate-pdf",
        action="store_true",
        help="Generate PDF clinical report"
    )
    
    # Pipeline control options
    p_run.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip environment check step"
    )
    p_run.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if a step fails"
    )
    p_run.add_argument(
        "--show-plots",
        action="store_true",
        help="Show QC plots during processing"
    )
    p_run.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save QC visualizations for each pipeline stage"
    )
    p_run.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
    p_run.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages (errors still shown)"
    )
    p_run.add_argument(
        "--bids-out",
        type=Path,
        default=None,
        metavar="DERIVATIVES_DIR",
        help=(
            "Write outputs as a BIDS derivatives dataset at DERIVATIVES_DIR. "
            "Creates dataset_description.json, participants.tsv, and renames output "
            "files to BIDS entity conventions. Backward-compatible: omit to use "
            "the existing flat layout."
        )
    )
    p_run.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session label (without 'ses-' prefix) for BIDS output organisation."
    )

    p_run.set_defaults(func=cmd_run)

    # -------------------------------------------------------------------------
    # batch subtool (BIDS / Manifest oriented)
    # -------------------------------------------------------------------------
    p_batch = subparsers.add_parser(
        "batch",
        help="Process multiple subjects/sessions in batch mode"
    )
    
    # Input options
    batch_input = p_batch.add_mutually_exclusive_group(required=True)
    batch_input.add_argument(
        "--manifest",
        type=Path,
        help="Path to JSON manifest file"
    )
    batch_input.add_argument(
        "--bids-dir",
        type=Path,
        help="Path to BIDS directory for auto-discovery"
    )
    
    p_batch.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output root directory where results will be organized"
    )
    
    # Batch control options
    p_batch.add_argument(
        "--force",
        action="store_true",
        help="Re-process all subjects, ignoring prior completions"
    )
    p_batch.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show execution plan without processing"
    )
    p_batch.add_argument(
        "--validate-only",
        action="store_true",
        help="Perform preflight validation only and exit"
    )
    p_batch.add_argument(
        "--keep-work",
        action="store_true",
        help="Retain _work/ directories even on success"
    )
    p_batch.add_argument(
        "--timeout-minutes",
        type=int,
        default=120,
        help="Per-subject timeout in minutes (default: 120)"
    )
    
    # Filters (only used with --bids-dir)
    p_batch.add_argument(
        "--include",
        nargs="+",
        help="Subject ID patterns to include (glob-style, e.g. sub-01*)"
    )
    p_batch.add_argument(
        "--exclude",
        nargs="+",
        help="Subject ID patterns to exclude"
    )

    # Processing options (similar to 'run' command)
    p_batch.add_argument(
        "--denoise-method",
        type=str,
        default=DEFAULT_DENOISE_METHOD,
        choices=["nlmeans", "patch2self", "mppca", "none"],
        help=f"Denoising method: nlmeans, patch2self, mppca, or none (default: {DEFAULT_DENOISE_METHOD})"
    )
    
    preproc_group = p_batch.add_mutually_exclusive_group()
    preproc_group.add_argument(
        "--preprocessing",
        dest="preprocessing",
        action="store_true",
        default=True,
        help="Run preprocessing steps (default)"
    )
    preproc_group.add_argument(
        "--no-preprocessing",
        dest="preprocessing",
        action="store_false",
        help="Skip preprocessing steps"
    )
    
    p_batch.add_argument(
        "--generate-pdf",
        action="store_true",
        help="Generate PDF clinical reports for each subject"
    )
    p_batch.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
    p_batch.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages (errors still shown)"
    )
    p_batch.add_argument(
        "--bids-out",
        type=Path,
        default=None,
        metavar="DERIVATIVES_DIR",
        help=(
            "Write BIDS dataset_description.json and participants.tsv at DERIVATIVES_DIR "
            "and rename output files to BIDS entity conventions. "
            "Recommended path: <raw_bids>/derivatives/csttool."
        )
    )

    p_batch.set_defaults(func=cmd_batch)

    # -------------------------------------------------------------------------
    # fetch-data subtool
    # -------------------------------------------------------------------------
    p_fetch_data = subparsers.add_parser(
        "fetch-data",
        help="Download FSL-licensed atlas data (FMRIB58_FA, Harvard-Oxford)"
    )
    p_fetch_data.add_argument(
        "--accept-fsl-license",
        action="store_true",
        help="Accept FSL non-commercial license without interactive prompt"
    )
    p_fetch_data.set_defaults(func=cmd_fetch_data)

    # -------------------------------------------------------------------------
    # validate subtool
    # -------------------------------------------------------------------------
    add_validate_parser(subparsers)

    # -------------------------------------------------------------------------
    # Parse and execute
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
