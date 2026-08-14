"""
convert_series.py

Convert DICOM series to NIfTI format with gradient files.

Converter priority:
  1. dcm2niix  (primary — vendor-agnostic, auto-generates BIDS JSON sidecars)
  2. dicom2nifti (fallback — used when dcm2niix is not on PATH or fails)
"""

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


try:
    from dicom2nifti import convert_dicom as _d2n
except ImportError:
    _d2n = None


# Vendors where dicom2nifti fallback is known to be unreliable
_UNRELIABLE_FALLBACK_VENDORS = {"philips", "ge", "ge medical systems", "hitachi"}


def _read_manufacturer(dicom_dir: Path) -> Optional[str]:
    """DICOM Manufacturer (0008,0070) from the first readable file, or None.

    Needed because dicom2nifti's b-vector convention is vendor-specific; the
    caller-supplied ``vendor`` argument is optional and often unset.
    """
    try:
        import pydicom
    except ImportError:
        return None
    for path in sorted(Path(dicom_dir).rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        value = getattr(ds, "Manufacturer", None)
        if value:
            return str(value)
    return None


def convert_dicom_to_nifti(
    dicom_dir: Path,
    output_dir: Path,
    output_name: Optional[str] = None,
    reorient: bool = True,
    verbose: bool = True,
    vendor: Optional[str] = None,
) -> Dict:
    """
    Convert a DICOM series to NIfTI format.

    Tries dcm2niix first (handles all major scanner vendors and generates
    BIDS JSON sidecars automatically). Falls back to dicom2nifti if dcm2niix
    is not available or fails on this particular series.

    Parameters
    ----------
    dicom_dir : Path
    output_dir : Path
    output_name : str, optional
        Base filename without extension. Defaults to directory name.
    reorient : bool
        Reorient to standard orientation (dicom2nifti fallback only).
    verbose : bool
    vendor : str, optional
        Scanner manufacturer string for warning logic.

    Returns
    -------
    dict with keys:
        nifti_path, bval_path, bvec_path, json_path,
        success, warnings, converter, fallback_used
    """
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)

    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        output_name = _sanitize_filename(dicom_dir.name)

    result: Dict = {
        "nifti_path": None,
        "bval_path": None,
        "bvec_path": None,
        "json_path": None,
        "success": False,
        "warnings": [],
        "converter": None,
        "fallback_used": False,
    }

    # ------------------------------------------------------------------
    # Primary: dcm2niix
    # ------------------------------------------------------------------
    dcm2niix_bin = shutil.which("dcm2niix")
    if dcm2niix_bin:
        try:
            _run_dcm2niix(dicom_dir, output_dir, output_name, result, verbose)
            if result["success"]:
                result["converter"] = "dcm2niix"
                return result
        except Exception as e:
            warn = f"dcm2niix failed: {e}"
            result["warnings"].append(warn)
            if verbose:
                print(f"    ⚠️  {warn}")
                print("    → Falling back to dicom2nifti...")
    else:
        if verbose:
            print("    ⚠️  dcm2niix not found on PATH — using dicom2nifti fallback")
            print("    Install: brew install dcm2niix  /  apt install dcm2niix")
        result["warnings"].append(
            "dcm2niix not found on PATH; used dicom2nifti fallback. "
            "Install dcm2niix for better multi-vendor support."
        )

    # ------------------------------------------------------------------
    # Fallback: dicom2nifti
    # ------------------------------------------------------------------
    if vendor and vendor.lower() in _UNRELIABLE_FALLBACK_VENDORS:
        result["warnings"].append(
            f"Vendor '{vendor}' detected. dicom2nifti fallback may produce "
            "incorrect gradient files for this vendor. Install dcm2niix."
        )
        if verbose:
            print(f"    ⚠️  Known problematic vendor for dicom2nifti: {vendor}")

    if _d2n is None:
        raise ImportError(
            "Neither dcm2niix nor dicom2nifti is available. "
            "Install dcm2niix (recommended) or: pip install dicom2nifti"
        )

    try:
        _run_dicom2nifti(dicom_dir, output_dir, output_name, result, reorient, verbose)
        if result["success"]:
            result["converter"] = "dicom2nifti"
            result["fallback_used"] = True
            return result
    except Exception as e:
        result["warnings"].append(f"dicom2nifti also failed: {e}")
        raise RuntimeError(
            f"DICOM conversion failed with all available methods. "
            f"Errors: {'; '.join(result['warnings'])}"
        )

    return result


def _run_dcm2niix(
    dicom_dir: Path,
    output_dir: Path,
    output_name: str,
    result: Dict,
    verbose: bool,
) -> None:
    """Run dcm2niix and populate *result* in-place."""
    if verbose:
        print("    → Converting via dcm2niix...")
        print(f"    • Input:  {dicom_dir}")

    cmd: List[str] = [
        "dcm2niix",
        "-z", "y",
        "-b", "y",          # write BIDS JSON sidecar
        "-f", output_name,
        "-o", str(output_dir),
        str(dicom_dir),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "dcm2niix returned non-zero exit code")

    nii_files = sorted(output_dir.glob(f"{output_name}*.nii.gz"))
    if not nii_files:
        nii_files = sorted(output_dir.glob(f"{output_name}*.nii"))
    if not nii_files:
        raise RuntimeError("dcm2niix produced no NIfTI output")

    result["nifti_path"] = nii_files[0]
    result["success"] = True

    bval_files = sorted(output_dir.glob(f"{output_name}*.bval"))
    if bval_files:
        result["bval_path"] = bval_files[0]
    else:
        result["warnings"].append("No .bval file generated by dcm2niix")

    bvec_files = sorted(output_dir.glob(f"{output_name}*.bvec"))
    if bvec_files:
        result["bvec_path"] = bvec_files[0]
    else:
        result["warnings"].append("No .bvec file generated by dcm2niix")

    json_files = sorted(output_dir.glob(f"{output_name}*.json"))
    if json_files:
        result["json_path"] = json_files[0]

    if verbose:
        print(f"    ✓ NIfTI: {result['nifti_path']}")
        if result["bval_path"]:
            print(f"    ✓ bval:  {result['bval_path']}")
        if result["json_path"]:
            print(f"    ✓ BIDS sidecar: {result['json_path']}")


def _run_dicom2nifti(
    dicom_dir: Path,
    output_dir: Path,
    output_name: str,
    result: Dict,
    reorient: bool,
    verbose: bool,
) -> None:
    """Run dicom2nifti and populate *result* in-place.

    AU21: dicom2nifti's ``reorient_nifti`` reorients the *image* array to LAS
    but never reorients the *bvecs*, leaving them in the original scanner voxel
    space — a silent gradient flip that corrupts the tensor fit. We therefore
    ignore dicom2nifti's reorientation flag, convert in the native (scanner)
    voxel space where image and bvecs are mutually consistent, and reorient
    *both* to RAS together via ``reorient_dwi_to_ras``. RAS+ is chosen because it
    is a deterministic canonical target, not because it matches the dcm2niix
    primary path — dcm2niix preserves the scanner's native orientation (commonly
    LAS for Siemens axial DWI), so the two import paths can legitimately produce
    different voxel-orientation codes for the same series. The ``reorient``
    argument is kept for API compatibility but now only controls whether a RAS
    reorientation is applied (True, the default) — there is no longer a
    dicom2nifti LAS step.
    """
    import nibabel as nib
    import numpy as np
    from dipy.io import read_bvals_bvecs
    from csttool.preprocess.modules.gradient_validation import (
        correct_dicom2nifti_bvecs,
        reorient_dwi_to_ras,
    )

    if verbose:
        print("    → Converting via dicom2nifti...")

    output_nii = output_dir / f"{output_name}.nii.gz"

    conversion_result = _d2n.dicom_series_to_nifti(
        str(dicom_dir),
        output_file=str(output_nii),
        # Native (scanner) voxel space: image + bvecs consistent. We reorient
        # to RAS ourselves below so the bvecs come along.
        reorient_nifti=False,
    )

    nii_path = conversion_result.get("NII_FILE")
    if not nii_path or not Path(nii_path).exists():
        raise RuntimeError("dicom2nifti produced no output file")

    bval_path = conversion_result.get("BVAL_FILE")
    bvec_path = conversion_result.get("BVEC_FILE")

    # Reorient image + bvecs to RAS together, rewriting the on-disk bvec sidecar
    # so every later stage reads a consistent pair. b-values are invariant.
    if bvec_path and bval_path and reorient:
        try:
            raw_img = nib.load(str(nii_path))
            _bvals, raw_bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
            # dicom2nifti writes b-vectors in a vendor-specific frame that is
            # NOT the voxel frame of the NIfTI it wrote (Siemens: the phase axis
            # is inverted). Undo that first, so the RAS reorientation below
            # carries a gradient table that already matches its image.
            raw_bvecs, bvec_note = correct_dicom2nifti_bvecs(
                raw_bvecs, _read_manufacturer(dicom_dir)
            )
            result["warnings"].append(f"dicom2nifti gradients — {bvec_note}")
            if verbose:
                print(f"    → gradients: {bvec_note}")
            ras_img, ras_bvecs = reorient_dwi_to_ras(raw_img, raw_bvecs)
            nib.save(ras_img, str(nii_path))
            np.savetxt(str(bvec_path), ras_bvecs.T, fmt="%.8f")
        except Exception as e:
            result["warnings"].append(
                f"DWI bvec reorientation failed: {e}. The image and bvecs may be "
                "in inconsistent voxel spaces; verify before tractography."
            )
            if verbose:
                print(f"    ⚠️  {e}")

    result["nifti_path"] = Path(nii_path)
    result["success"] = True

    if bval_path and Path(bval_path).exists():
        result["bval_path"] = Path(bval_path)
    else:
        result["warnings"].append("No .bval file generated")

    if bvec_path and Path(bvec_path).exists():
        result["bvec_path"] = Path(bvec_path)
    else:
        result["warnings"].append("No .bvec file generated")

    _check_if_gradients_needed(result, verbose)

    if verbose:
        print(f"    ✓ NIfTI: {result['nifti_path']}")


def _sanitize_filename(name: str) -> str:
    replacements = {
        " ": "_", "/": "_", "\\": "_", ":": "_",
        "*": "_", "?": "_", '"': "_", "<": "_",
        ">": "_", "|": "_",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    result = result.strip("_.")
    return result if result else "converted"


def _check_if_gradients_needed(result: Dict, verbose: bool) -> None:
    import nibabel as nib

    if not result.get("nifti_path"):
        return
    try:
        img = nib.load(str(result["nifti_path"]))
        shape = img.shape
        if len(shape) == 3:
            result["warnings"] = [
                w for w in result["warnings"]
                if "bval" not in w.lower() and "bvec" not in w.lower()
            ]
            if verbose:
                print("    • 3D volume — gradient files not required")
        elif len(shape) == 4 and shape[3] > 1:
            if not result["bval_path"] or not result["bvec_path"]:
                result["warnings"].append(
                    "4D diffusion data detected but gradient files are missing. "
                    "Tractography requires bval/bvec files."
                )
                if verbose:
                    print(f"    ⚠️  4D data ({shape[3]} volumes) requires gradient files")
    except Exception as e:
        if verbose:
            print(f"    ⚠️  Could not verify image dimensions: {e}")


def validate_conversion(
    nifti_path: Path,
    bval_path: Optional[Path] = None,
    bvec_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """
    Validate converted NIfTI and gradient files.

    Returns a dict with keys: valid, nifti_valid, gradients_valid,
    data_shape, n_volumes, n_gradients, issues.
    """
    import nibabel as nib
    import numpy as np

    validation: Dict = {
        "valid": False,
        "nifti_valid": False,
        "gradients_valid": False,
        "data_shape": None,
        "n_volumes": 0,
        "n_gradients": 0,
        "issues": [],
    }

    try:
        img = nib.load(str(nifti_path))
        data_shape = img.shape
        validation["nifti_valid"] = True
        validation["data_shape"] = data_shape
        validation["n_volumes"] = data_shape[3] if len(data_shape) == 4 else 1

        if verbose:
            print("    • NIfTI Validation:")
            print(f"    ├─ Shape: {data_shape}")
            print(f"    ├─ Voxel size: {img.header.get_zooms()[:3]}")
            print(f"    └─ Data type: {img.get_data_dtype()}")
    except Exception as e:
        validation["issues"].append(f"Invalid NIfTI: {e}")
        return validation

    if validation["n_volumes"] > 1 and bval_path and bvec_path:
        try:
            bvals = np.loadtxt(str(bval_path)).flatten()
            bvecs = np.loadtxt(str(bvec_path))
            if bvecs.shape[0] == 3:
                bvecs = bvecs.T
            validation["n_gradients"] = len(bvals)

            if len(bvals) != validation["n_volumes"]:
                validation["issues"].append(
                    f"Volume/gradient mismatch: {validation['n_volumes']} volumes "
                    f"but {len(bvals)} b-values"
                )
            elif bvecs.shape[0] != len(bvals):
                validation["issues"].append(
                    f"bval/bvec mismatch: {len(bvals)} b-values "
                    f"but {bvecs.shape[0]} vectors"
                )
            else:
                validation["gradients_valid"] = True

            if verbose:
                print("    • Gradient Validation:")
                print(f"    ├─ B-values: {len(bvals)}")
                print(f"    ├─ Unique b-values: {sorted(set(bvals.astype(int)))}")
                print(f"    └─ Gradient vectors: {bvecs.shape}")
        except Exception as e:
            validation["issues"].append(f"Error reading gradient files: {e}")
    elif validation["n_volumes"] == 1:
        validation["gradients_valid"] = True

    validation["valid"] = (
        validation["nifti_valid"]
        and validation["gradients_valid"]
        and not validation["issues"]
    )

    if verbose:
        status = "✓" if validation["valid"] else "✗"
        print(f"  {status} Conversion validation {'passed' if validation['valid'] else 'failed'}")
        for issue in validation["issues"]:
            print(f"    • {issue}")

    return validation


def convert_with_fallback(
    dicom_dir: Path,
    output_dir: Path,
    output_name: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Thin wrapper kept for API compatibility. Delegates to convert_dicom_to_nifti().
    """
    return convert_dicom_to_nifti(dicom_dir, output_dir, output_name, verbose=verbose)
