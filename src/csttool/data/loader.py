"""
Data loader for csttool atlas and template files.

Provides two-tier data access:
- Tier 1: Bundled data (MNI152) via importlib.resources - always available
- Tier 2: User-fetched data (FMRIB58_FA, Harvard-Oxford) via platformdirs - requires fetch-data command
"""

import json
from importlib.resources import files, as_file
from pathlib import Path
from typing import Tuple, Optional

import nibabel as nib
import numpy as np
from platformdirs import user_data_dir

from .manifest import get_manifest_entry, verify_checksum


class DataNotInstalledError(FileNotFoundError):
    """Raised when required data files are not installed."""
    pass


# User data directory for Tier 2 files (cross-platform)
_USER_DATA_DIR = Path(user_data_dir("csttool", ensure_exists=False))
_VALIDATION_STAMP_FILE = _USER_DATA_DIR / ".validated"


def get_user_data_dir() -> Path:
    """
    Get the user data directory for Tier 2 files.

    Returns
    -------
    Path
        Platform-specific user data directory
        - Linux: ~/.local/share/csttool
        - macOS: ~/Library/Application Support/csttool
        - Windows: C:\\Users\\<username>\\AppData\\Local\\csttool
    """
    return _USER_DATA_DIR


def _load_validation_stamp() -> dict:
    """Load validation stamp file."""
    if not _VALIDATION_STAMP_FILE.exists():
        return {}
    try:
        with open(_VALIDATION_STAMP_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_validation_stamp(stamp_data: dict) -> None:
    """Save validation stamp file."""
    _USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_VALIDATION_STAMP_FILE, "w") as f:
        json.dump(stamp_data, f, indent=2)


def _validate_if_needed(file_path: Path, manifest_key: str) -> None:
    """
    Validate file checksum if needed (smart caching based on size/mtime).

    Parameters
    ----------
    file_path : Path
        Path to the file to validate
    manifest_key : str
        Key in the manifest for this file

    Raises
    ------
    DataNotInstalledError
        If file doesn't exist or checksum validation fails
    """
    if not file_path.exists():
        raise DataNotInstalledError(f"Data file not found: {file_path}")

    # Get expected checksum from manifest
    manifest_entry = get_manifest_entry(manifest_key)
    expected_sha256 = manifest_entry["sha256"]

    # Load validation stamp
    stamp_data = _load_validation_stamp()
    file_key = str(file_path)

    # Check if file was previously validated and unchanged
    if file_key in stamp_data:
        stamp_entry = stamp_data[file_key]
        current_size = file_path.stat().st_size
        current_mtime = file_path.stat().st_mtime

        # If size and mtime match, skip re-validation
        if (stamp_entry.get("size") == current_size and
            stamp_entry.get("mtime") == current_mtime and
            stamp_entry.get("sha256_ok") is True):
            return

    # File is new or modified, verify checksum
    if not verify_checksum(file_path, expected_sha256):
        raise DataNotInstalledError(
            f"Checksum verification failed for {file_path.name}.\n"
            f"Expected: {expected_sha256}\n"
            f"The file may be corrupted. Please run 'csttool fetch-data --accept-fsl-license' to re-download."
        )

    # Update validation stamp
    stamp_data[file_key] = {
        "size": file_path.stat().st_size,
        "mtime": file_path.stat().st_mtime,
        "sha256_ok": True,
    }
    _save_validation_stamp(stamp_data)


# ===== Tier 1: Bundled Data (MNI152) =====

def load_mni152_template() -> Tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    """
    Load bundled MNI152 T1 1mm template (always available).

    Uses importlib.resources to access bundled data within the package.
    Data is eagerly loaded and copied to ensure independence from file handles.

    Returns
    -------
    template_img : Nifti1Image
        NIfTI image object
    template_data : ndarray
        Image data array
    template_affine : ndarray
        4x4 affine transformation matrix

    Examples
    --------
    >>> img, data, affine = load_mni152_template()
    >>> print(data.shape)  # (197, 233, 189) for MNI152 1mm
    """
    # Access bundled resource via importlib.resources
    ref = files("csttool.data").joinpath("mni152", "MNI152_T1_1mm.nii.gz")

    # Use as_file() context manager to get filesystem path
    with as_file(ref) as path:
        img = nib.load(str(path))
        # Eagerly copy data before exiting context (file handle independence)
        data = img.get_fdata().copy()
        affine = img.affine.copy()

    # Return new NIfTI image with copied data
    return nib.Nifti1Image(data, affine), data, affine


# ===== Tier 2: User-Fetched Data (FMRIB58_FA, Harvard-Oxford) =====

def get_fmrib58_fa_path() -> Path:
    """
    Get path to FMRIB58_FA 1mm template (user-fetched).

    Returns
    -------
    Path
        Path to FMRIB58_FA_1mm.nii.gz

    Raises
    ------
    DataNotInstalledError
        If template not found or checksum validation fails.
        Error message includes instructions for fetching data.

    Examples
    --------
    >>> path = get_fmrib58_fa_path()
    >>> img = nib.load(path)
    """
    path = _USER_DATA_DIR / "fmrib58_fa" / "FMRIB58_FA_1mm.nii.gz"

    if not path.exists():
        raise DataNotInstalledError(
            "FMRIB58_FA template not found.\n"
            "Run 'csttool fetch-data --accept-fsl-license' to download "
            "FSL-licensed atlas data (non-commercial use only)."
        )

    _validate_if_needed(path, "fmrib58_fa/FMRIB58_FA_1mm.nii.gz")
    return path


def get_fmrib58_fa_skeleton_path() -> Path:
    """
    Get path to FMRIB58_FA skeleton 1mm template (user-fetched).

    Returns
    -------
    Path
        Path to FMRIB58_FA-skeleton_1mm.nii.gz

    Raises
    ------
    DataNotInstalledError
        If template not found or checksum validation fails
    """
    path = _USER_DATA_DIR / "fmrib58_fa" / "FMRIB58_FA-skeleton_1mm.nii.gz"

    if not path.exists():
        raise DataNotInstalledError(
            "FMRIB58_FA skeleton template not found.\n"
            "Run 'csttool fetch-data --accept-fsl-license' to download "
            "FSL-licensed atlas data (non-commercial use only)."
        )

    _validate_if_needed(path, "fmrib58_fa/FMRIB58_FA-skeleton_1mm.nii.gz")
    return path


def get_harvard_oxford_path(atlas_name: str, resolution: str = "1mm") -> Path:
    """
    Get path to Harvard-Oxford atlas (user-fetched).

    Parameters
    ----------
    atlas_name : str
        Atlas type: "cortical" or "subcortical"
    resolution : str, optional
        Resolution: "1mm" or "2mm" (default: "1mm")

    Returns
    -------
    Path
        Path to the Harvard-Oxford atlas file

    Raises
    ------
    DataNotInstalledError
        If atlas not found or checksum validation fails
    ValueError
        If invalid atlas_name or resolution

    Examples
    --------
    >>> cort_path = get_harvard_oxford_path("cortical", "1mm")
    >>> subcort_path = get_harvard_oxford_path("subcortical", "1mm")
    """
    # Validate inputs
    if atlas_name not in ("cortical", "subcortical"):
        raise ValueError(
            f"Invalid atlas_name '{atlas_name}'. "
            "Must be 'cortical' or 'subcortical'."
        )

    if resolution not in ("1mm", "2mm"):
        raise ValueError(
            f"Invalid resolution '{resolution}'. "
            "Must be '1mm' or '2mm'."
        )

    # Map to filename
    atlas_type = "cort" if atlas_name == "cortical" else "sub"
    filename = f"HarvardOxford-{atlas_type}-maxprob-thr25-{resolution}.nii.gz"
    path = _USER_DATA_DIR / "harvard_oxford" / filename
    manifest_key = f"harvard_oxford/{filename}"

    if not path.exists():
        raise DataNotInstalledError(
            f"Harvard-Oxford {atlas_name} atlas ({resolution}) not found.\n"
            "Run 'csttool fetch-data --accept-fsl-license' to download "
            "FSL-licensed atlas data (non-commercial use only)."
        )

    _validate_if_needed(path, manifest_key)
    return path


# JHU ICBM-DTI-81 label values for the internal capsule (FSL JHU-labels.xml).
# The atlas stores the posterior limb as 19 (right) and 20 (left); these feed
# the preferred CST pipeline's deep-white-matter seed. Anterior limb (17/18)
# and retrolenticular part (21/22) are recorded for validation context only.
JHU_PLIC_RIGHT_LABEL = 19
JHU_PLIC_LEFT_LABEL = 20
JHU_ATLAS_VERSION = "JHU-ICBM-DTI-81 (FSL data_atlases 2103.0)"


def get_jhu_labels_path(resolution: str = "1mm") -> Path:
    """Path to the JHU ICBM-DTI-81 white-matter label atlas (user-fetched).

    Same tier-2 contract as :func:`get_harvard_oxford_path`: the file must be
    installed in the csttool data directory (``csttool fetch-data`` stages it
    from the FSL data_atlases distribution). Returns the validated path.

    Parameters
    ----------
    resolution : str
        "1mm" or "2mm" (default "1mm").
    """
    if resolution not in ("1mm", "2mm"):
        raise ValueError(
            f"Invalid resolution '{resolution}'. Must be '1mm' or '2mm'."
        )
    filename = f"JHU-ICBM-labels-{resolution}.nii.gz"
    path = _USER_DATA_DIR / "jhu" / filename
    manifest_key = f"jhu/{filename}"
    if not path.exists():
        raise DataNotInstalledError(
            f"JHU ICBM-DTI-81 label atlas ({resolution}) not found at {path}.\n"
            "Run 'csttool fetch-data --accept-fsl-license' to install "
            "FSL-licensed atlas data (non-commercial use only). The PLIC seed "
            "ROIs for --tracking-mode preferred are read from this atlas."
        )
    _validate_if_needed(path, manifest_key)
    return path


def is_data_installed() -> bool:
    """
    Check if all Tier 2 data files are installed.

    Returns
    -------
    bool
        True if all Tier 2 files are present and valid, False otherwise

    Examples
    --------
    >>> if not is_data_installed():
    ...     print("Please run 'csttool fetch-data --accept-fsl-license'")
    """
    try:
        # Check FMRIB58_FA files
        get_fmrib58_fa_path()
        get_fmrib58_fa_skeleton_path()

        # Check Harvard-Oxford files (1mm and 2mm)
        for resolution in ("1mm", "2mm"):
            get_harvard_oxford_path("cortical", resolution)
            get_harvard_oxford_path("subcortical", resolution)

        return True
    except DataNotInstalledError:
        return False
