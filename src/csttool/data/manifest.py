"""
Data manifest with SHA256 checksums, source URLs, and metadata for all atlas/template files.
"""

import hashlib
from pathlib import Path
from typing import Dict, Any

# Data manifest with checksums, sources, and licensing information
DATA_MANIFEST: Dict[str, Dict[str, Any]] = {
    # Tier 1: Bundled in package (permissive license)
    "mni152/MNI152_T1_1mm.nii.gz": {
        "sha256": "78af50d4c8392da2fb74d0565bba1c56fc3520989d7e49fd6887b7166e79834a",
        "source_url": "https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/",
        "license": "BSD-like (ICBM)",
        "version": "ICBM 2009c Nonlinear Symmetric",
        "fsl_tag": None,
        "size_bytes": 2761282,
    },

    # Tier 2: User-fetched (FSL non-commercial license)
    # FMRIB58_FA templates
    "fmrib58_fa/FMRIB58_FA_1mm.nii.gz": {
        "sha256": "1268fd84b658451c9a85799a636688bc1f6947fdf8de2d9cf7f62320de1b3837",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_standard/-/raw/2208.0/FMRIB58_FA_1mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "FSL data_standard",
        "fsl_tag": "2208.0",
        "size_bytes": 2641696,
    },
    "fmrib58_fa/FMRIB58_FA-skeleton_1mm.nii.gz": {
        "sha256": "db7a69531396504ecf1555061831dd909c524b6d5575d5c1b280a340eea9c8de",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_standard/-/raw/2208.0/FMRIB58_FA-skeleton_1mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "FSL data_standard",
        "fsl_tag": "2208.0",
        "size_bytes": 533933,
    },

    # Harvard-Oxford atlases (1mm resolution)
    "harvard_oxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz": {
        "sha256": "f71ecc19595ffc85e782dea04563e83c9eb73a7a60c415fb6660c23285200dc1",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "FSL data_atlases",
        "fsl_tag": "2103.0",
        "size_bytes": 138203,
    },
    "harvard_oxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz": {
        "sha256": "5772ccce4a0444c9b29c071390db086b7bb0e2074662d2ccecc8c9820fff0fa7",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "FSL data_atlases",
        "fsl_tag": "2103.0",
        "size_bytes": 166519,
    },

    # Harvard-Oxford atlases (2mm resolution)
    "harvard_oxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz": {
        "sha256": "7397ffdbae7559e0f0aa6237f998dc0f6aca3f9db9009f6ffa637c0c62b686ca",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "FSL data_atlases",
        "fsl_tag": "2103.0",
        "size_bytes": 27867,
    },
    "harvard_oxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz": {
        "sha256": "72140df8117250d915b753ca2937e078c917525d206e6185e2c1b4ab703fbfcc",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "FSL data_atlases",
        "fsl_tag": "2103.0",
        "size_bytes": 36627,
    },

    # JHU ICBM-DTI-81 white-matter label atlas (MNI152). Source of the
    # posterior-limb-of-internal-capsule (PLIC) seed ROIs for the preferred
    # CST pipeline: label 19 = PLIC right, 20 = PLIC left.
    "jhu/JHU-ICBM-labels-1mm.nii.gz": {
        "sha256": "3c3f5d2f1250a3df60982acff35a75b99fd549a05d5f8124a63f78221aa0ec16",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/JHU/JHU-ICBM-labels-1mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "JHU-ICBM-DTI-81 (FSL data_atlases)",
        "fsl_tag": "2103.0",
        "size_bytes": 52088,
    },
    "jhu/JHU-ICBM-labels-2mm.nii.gz": {
        "sha256": "79a3badf78e91c861a10ec5b8eb277e17e6558d39ef93bfd1e365e7464cb9ebf",
        "source_url": "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/2103.0/JHU/JHU-ICBM-labels-2mm.nii.gz",
        "license": "FSL non-commercial",
        "version": "JHU-ICBM-DTI-81 (FSL data_atlases)",
        "fsl_tag": "2103.0",
        "size_bytes": 9634,
    },
}


def get_manifest_entry(relative_key: str) -> Dict[str, Any]:
    """
    Get manifest entry for a data file.

    Parameters
    ----------
    relative_key : str
        Relative path key (e.g., "mni152/MNI152_T1_1mm.nii.gz")

    Returns
    -------
    dict
        Manifest entry with sha256, source_url, license, version, etc.

    Raises
    ------
    KeyError
        If the key is not found in the manifest
    """
    if relative_key not in DATA_MANIFEST:
        raise KeyError(
            f"Data file '{relative_key}' not found in manifest. "
            f"Available keys: {list(DATA_MANIFEST.keys())}"
        )
    return DATA_MANIFEST[relative_key]


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """
    Verify SHA256 checksum of a file.

    Parameters
    ----------
    file_path : Path
        Path to the file to verify
    expected_sha256 : str
        Expected SHA256 checksum (hex string)

    Returns
    -------
    bool
        True if checksum matches, False otherwise
    """
    if not file_path.exists():
        return False

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    actual_checksum = sha256_hash.hexdigest()
    return actual_checksum == expected_sha256


def get_tier2_files() -> Dict[str, Dict[str, Any]]:
    """
    Get all Tier 2 (user-fetched) data files from the manifest.

    Returns
    -------
    dict
        Dictionary of Tier 2 files with their manifest entries
    """
    return {
        key: value
        for key, value in DATA_MANIFEST.items()
        if value.get("fsl_tag") is not None
    }
