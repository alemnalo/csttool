
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from time import time

from dipy.io import read_bvals_bvecs

from csttool import __version__
from csttool.defaults import DEFAULT_B0_THRESHOLD
from csttool.preprocess.modules.gradient_validation import (
    validate_gradient_table,
)
from csttool.preprocess.modules.load_dataset import load_dataset as load_dataset_module

def add_io_arguments(p: argparse.ArgumentParser) -> None:
    """Add common IO arguments for import and preprocess."""
    p.add_argument("--dicom", type=Path, help="Path to DICOM directory.")
    p.add_argument(
        "--nifti",
        type=Path,
        help="Path to NIfTI file (.nii or .nii.gz). "
             "Sidecars (.bval/.bvec) must be in the same folder with the same stem."
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (also used for converted NIfTI)."
    )


def resolve_nifti(args: argparse.Namespace) -> Path:
    """Resolve a NIfTI path from DICOM or explicit NIfTI."""
    nii: Path | None = None

    # DICOM case
    # Check for DICOM directory manually to avoid funcs.py dependency
    is_dicom = False
    if args.dicom and args.dicom.is_dir():
        if any(f.suffix.lower() == ".dcm" for f in args.dicom.iterdir()):
            is_dicom = True

    if is_dicom:
        print(f"  → Converting DICOM to NIfTI...")
        stem = args.dicom.name

        if not args.out:
            pass

        import dicom2nifti
        import nibabel as nib
        import numpy as np
        from csttool.preprocess.modules.gradient_validation import (
            reorient_dwi_to_ras,
        )

        nifti_dir = args.out / "nifti"
        nifti_dir.mkdir(parents=True, exist_ok=True)
        output_nii = nifti_dir / (stem + ".nii.gz")

        print(f"  → Output: {output_nii}")
        # AU21: convert with reorient_nifti=False (image + bvecs in the same
        # scanner voxel space), then reorient both to RAS together — see
        # gradient_validation.reorient_dwi_to_ras. dicom2nifti's reorient_nifti
        # reorients only the image, leaving bvecs in scanner space and silently
        # flipping gradient directions relative to the stored array.
        result = dicom2nifti.dicom_series_to_nifti(
            str(args.dicom),
            str(output_nii),
            reorient_nifti=False,
        )
        bval_p = result.get("BVAL_FILE")
        bvec_p = result.get("BVEC_FILE")
        if bvec_p and bval_p:
            from dipy.io import read_bvals_bvecs
            raw_img = nib.load(str(output_nii))
            _bvals, raw_bvecs = read_bvals_bvecs(str(bval_p), str(bvec_p))
            ras_img, ras_bvecs = reorient_dwi_to_ras(raw_img, raw_bvecs)
            nib.save(ras_img, str(output_nii))
            # Only the bvec changes under reorientation; bvals are invariant.
            np.savetxt(bvec_p, ras_bvecs.T, fmt="%.8f")
        nii = output_nii

    else:
        # Provided DICOM path is invalid
        if args.dicom and args.dicom.is_dir():
             # Check again if it was just empty or no dcm files
             pass
        elif args.dicom:
            print(f"  ⚠️ {args.dicom} is not a valid DICOM directory, falling back to NIfTI")

        # Use explicit NIfTI if given
        if args.nifti:
            nii = args.nifti
        else:
            # Try to find a NIfTI in the output directory
            print(f"  → Searching for NIfTI in {args.out}...")
            # Use glob to find candidates
            if args.out and args.out.exists():
                candidates = list(args.out.glob("*.nii.gz")) + list(args.out.glob("*.nii"))
                if not candidates:
                    raise FileNotFoundError(
                        "No NIfTI file found. Provide --nifti or a valid --dicom directory."
                    )
                nii = candidates[0]
            else:
                 if args.nifti is None:
                     raise FileNotFoundError("Output directory does not exist and no NIfTI provided.")

    if not isinstance(nii, Path):
        nii = Path(nii)

    if not nii.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nii}")

    return nii


def load_with_preproc(nii: Path, *, b0_threshold: float = DEFAULT_B0_THRESHOLD):
    """Load data and gradients using modules.load_dataset."""
    nifti_dir = str(nii.parent)
    name = nii.name

    if name.endswith(".nii.gz"):
        fname = name[:-7]
    elif name.endswith(".nii"):
        fname = name[:-4]
    else:
        raise ValueError("NIfTI must end with .nii or .nii.gz")

    # Use the modular load_dataset
    # Note: modules.load_dataset signature is (dir_path: str, fname: str)
    # and returns (nii_img, gtab, nifti_dir_path)
    
    nii_img, gtab, _, metadata = load_dataset_module(
        dir_path=nifti_dir,
        fname=fname,
        b0_threshold=b0_threshold,
    )
    
    data = nii_img.get_fdata()
    affine = nii_img.affine
    hdr = nii_img.header
    
    return data, affine, hdr, gtab, metadata


def get_gtab_for_preproc(
    preproc_nii: Path, *, b0_threshold: float = DEFAULT_B0_THRESHOLD
):
    """
    Given a preprocessed NIfTI path like <stem>_preproc.nii.gz,
    find the .bval and .bvec next to it and build gtab.

    Supports both .bval/.bvec and .bvals/.bvecs extensions.

    ``b0_threshold`` is the execution's authoritative b0 threshold
    (``--b0-threshold``); it must match the one preprocessing used, so that
    ``gtab.b0s_mask`` partitions the series identically on both sides.
    """
    name = preproc_nii.name

    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        raise ValueError("Preprocessed NIfTI must end with .nii or .nii.gz")

    base_dir = preproc_nii.parent
    
    # Try to find matching bval/bvec files with exact stem match
    try:
        bval, bvec = find_gradient_files(base_dir, stem)
    except FileNotFoundError:
        # If not found, try removing _preproc suffix variations
        found = False
        for suffix in ["_preproc", "_dwi_preproc_nomc", "_dwi_preproc_mc", "_dwi_preproc"]:
            if stem.endswith(suffix):
                orig_stem = stem[:-len(suffix)]
                try:
                    bval, bvec = find_gradient_files(base_dir, orig_stem)
                    found = True
                    break
                except FileNotFoundError:
                    continue
        
        if not found:
            raise FileNotFoundError(
                f"Missing gradient files for {stem}. "
                f"Expected .bval/.bvec or .bvals/.bvecs next to the NIfTI file."
            )

    print(f"  → bval: {bval}")
    print(f"  → bvec: {bvec}")

    bvals, bvecs = read_bvals_bvecs(str(bval), str(bvec))
    # AU21: validate the gradient table at load rather than letting a malformed
    # table silently corrupt the tensor fit. Uses the single-source-of-truth
    # b0 threshold so the gtab's b0s_mask matches the rest of the pipeline.
    gtab = validate_gradient_table(
        bvals, bvecs, b0_threshold=b0_threshold
    )
    return gtab


def extract_stem_from_filename(filename: str) -> str:
    """Extract clean stem from filename, removing common suffixes."""
    path = Path(filename)
    stem = path.stem  # Removes .gz if present
    
    # Handle .nii.gz case
    if stem.endswith('.nii'):
        stem = stem[:-4]
    
    # Remove common suffixes
    suffixes_to_remove = [
        '_dwi_preproc_nomc',
        '_dwi_preproc_mc', 
        '_preproc_nomc',
        '_preproc_mc',
        '_preproc',
        '_dwi'
    ]
    
    for suffix in suffixes_to_remove:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    
    # If stem is empty or very short, use original name
    if len(stem) < 3:
        stem = path.stem.replace('.nii', '')
    
    return stem


def find_gradient_files(base_path: Path, stem: str) -> tuple[Path, Path]:
    """
    Find .bval/.bvec files with flexible extension support.
    
    Supports both singular (.bval, .bvec) and plural (.bvals, .bvecs) extensions.
    
    Args:
        base_path: Directory containing the gradient files
        stem: Base filename without extension
        
    Returns:
        tuple: (bval_path, bvec_path)
        
    Raises:
        FileNotFoundError: If gradient files cannot be found
    """
    # Try both singular and plural extensions
    for bval_ext, bvec_ext in [('.bval', '.bvec'), ('.bvals', '.bvecs')]:
        bval = base_path / f"{stem}{bval_ext}"
        bvec = base_path / f"{stem}{bvec_ext}"
        
        if bval.exists() and bvec.exists():
            return bval, bvec
    
    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"Could not find gradient files for '{stem}' in {base_path}. "
        f"Tried: {stem}.bval/.bvec and {stem}.bvals/.bvecs"
    )


def find_files_recursive(directory: Path, pattern: str) -> list[Path]:
    """Recursively find files matching a pattern."""
    return list(directory.rglob(pattern))


def save_pipeline_report(
    output_dir: Path,
    subject_id: str,
    step_results: dict,
    step_times: dict,
    failed_steps: list,
    start_time: float
) -> Path:
    """Save pipeline execution report as JSON."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    # log_dir = output_dir / "logs"
    # log_dir.mkdir(exist_ok=True)
    
    report = {
        'subject_id': subject_id,
        'pipeline_version': __version__,
        'execution': {
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_seconds': time() - start_time,
            'success': len(failed_steps) == 0
        },
        'step_times': step_times,
        'failed_steps': failed_steps,
        'step_results': {
            step: {
                'success': result.get('success', False),
                'error': result.get('error') if 'error' in result else None
            }
            for step, result in step_results.items()
        }
    }
    
    report_path = output_dir / f"{subject_id}_pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report_path
