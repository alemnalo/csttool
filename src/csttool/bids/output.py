"""
output.py

BIDS-compliant output helpers for csttool.

Three compliance tiers:
  - Raw import (dcm2niix output):        fully BIDS-compliant
  - Derivative NIfTIs (preproc, maps):   BIDS-aligned derivatives
  - Tractograms / HTML reports:          BIDS-adjacent container
                                         (no finalised BIDS tractography schema)
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock

from csttool import __version__

# ---------------------------------------------------------------------------
# BIDS entity order (BIDS spec §4.1)
# ---------------------------------------------------------------------------
_ENTITY_ORDER = [
    "sub", "ses", "task", "acq", "ce", "rec", "dir",
    "run", "mod", "echo", "flip", "inv", "mt",
    "space", "res", "den", "label", "desc",
    "model", "param",
]


def bids_filename(
    subject: str,
    suffix: str,
    extension: str,
    session: Optional[str] = None,
    **entities: str,
) -> str:
    """
    Build a BIDS-compliant filename.

    Parameters
    ----------
    subject : str
        Subject label without 'sub-' prefix (e.g. '001').
    suffix : str
        BIDS suffix (e.g. 'dwi', 'mask', 'dwimap', 'tractogram').
    extension : str
        File extension including leading dot (e.g. '.nii.gz', '.trk').
    session : str, optional
        Session label without 'ses-' prefix.
    **entities
        Additional BIDS entities as keyword arguments
        (e.g. space='orig', desc='preproc').

    Returns
    -------
    str
        BIDS filename, e.g.
        'sub-001_ses-01_space-orig_desc-preproc_dwi.nii.gz'.
    """
    parts: Dict[str, str] = {}
    parts["sub"] = subject
    if session:
        parts["ses"] = session
    parts.update(entities)

    ordered = []
    for key in _ENTITY_ORDER:
        if key in parts:
            ordered.append(f"{key}-{parts[key]}")
    # Any extra entities not in the canonical list are appended
    for key, val in parts.items():
        if key not in _ENTITY_ORDER and key not in ("sub", "ses"):
            ordered.append(f"{key}-{val}")

    name = "_".join(ordered) + f"_{suffix}{extension}"
    return name


def sanitize_bids_label(raw: str, max_len: int = 20) -> str:
    """
    Clean an arbitrary string for use as a BIDS entity value.

    Rules:
    - Replace non-alphanumeric characters with hyphens
    - Collapse consecutive hyphens
    - Strip leading/trailing hyphens
    - Ensure the result starts with a letter (prepend 'x' if not)
    - Truncate to max_len

    Parameters
    ----------
    raw : str
    max_len : int

    Returns
    -------
    str
        Safe BIDS label, never empty.
    """
    label = re.sub(r"[^A-Za-z0-9]", "-", raw)
    label = re.sub(r"-{2,}", "-", label)
    label = label.strip("-")
    if not label:
        label = "x"
    elif not label[0].isalpha():
        label = "x" + label
    return label[:max_len]


def parse_dicom_age(tag_value: Any) -> Optional[float]:
    """
    Parse a DICOM PatientAge tag value to decimal years.

    Handles:
      '034Y' → 34.0
      '018M' → 1.5
      '002W' → 0.038
      '005D' → 0.014

    Returns None if the value is absent or unparseable.
    """
    if tag_value is None:
        return None
    s = str(tag_value).strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d+)([YMWD])", s, re.IGNORECASE)
    if not m:
        return None
    amount, unit = int(m.group(1)), m.group(2).upper()
    if unit == "Y":
        return float(amount)
    if unit == "M":
        return round(amount / 12, 3)
    if unit == "W":
        return round(amount / 52.18, 3)
    if unit == "D":
        return round(amount / 365.25, 3)
    return None


def write_dataset_description(
    output_root: Path,
    version: str = __version__,
    raw_bids_root: Optional[Path] = None,
    dataset_type: str = "derivative",
) -> Path:
    """
    Write a BIDS dataset_description.json at output_root.

    Only creates the file if it does not already exist, so repeated
    pipeline runs do not overwrite existing provenance.

    Parameters
    ----------
    output_root : Path
    version : str
        csttool version string.
    raw_bids_root : Path, optional
        If provided and output_root is a descendant of raw_bids_root,
        adds SourceDatasets with the correct bids:: URI.
    dataset_type : str
        'derivative' or 'raw'.

    Returns
    -------
    Path to the written file.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dest = output_root / "dataset_description.json"

    if dest.exists():
        return dest

    desc: Dict[str, Any] = {
        "Name": "csttool derivatives" if dataset_type == "derivative" else "csttool raw BIDS",
        "BIDSVersion": "1.9.0",
        "DatasetType": dataset_type,
        "GeneratedBy": [
            {
                "Name": "csttool",
                "Version": version,
            }
        ],
    }

    if dataset_type == "derivative":
        # Include SourceDatasets only when the derivative is nested under the raw
        if raw_bids_root is not None:
            try:
                output_root.relative_to(Path(raw_bids_root).resolve())
                desc["SourceDatasets"] = [{"URL": "bids::"}]
            except ValueError:
                pass  # not a subdirectory — omit SourceDatasets

    dest.write_text(json.dumps(desc, indent=2) + "\n")
    return dest


def write_participants_json(output_root: Path) -> Path:
    """
    Write participants.json column definitions (only if absent).
    """
    output_root = Path(output_root)
    dest = output_root / "participants.json"
    if dest.exists():
        return dest

    definitions = {
        "participant_id": {
            "Description": "Unique participant identifier"
        },
        "age": {
            "Description": "Participant age at time of scan",
            "Units": "years",
        },
        "sex": {
            "Description": "Biological sex",
            "Levels": {"M": "Male", "F": "Female", "O": "Other"},
        },
    }
    dest.write_text(json.dumps(definitions, indent=2) + "\n")
    return dest


def update_participants_tsv(
    output_root: Path,
    subject_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Append one row to participants.tsv, ensuring no duplicate subjects.

    Uses fcntl.flock for safe concurrent batch writes.

    Parameters
    ----------
    output_root : Path
    subject_id : str
        Must already include the 'sub-' prefix.
    metadata : dict, optional
        Keys: 'age' (float or None), 'sex' (str or None).

    Returns
    -------
    Path to participants.tsv.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dest = output_root / "participants.tsv"

    if metadata is None:
        metadata = {}

    age_val = metadata.get("age")
    sex_raw = metadata.get("sex", "n/a")
    sex_val = _normalise_sex(sex_raw)
    age_str = f"{age_val:.1f}" if isinstance(age_val, (int, float)) else "n/a"

    with FileLock(str(dest) + ".lock"):
        with open(dest, "a+") as fh:
            fh.seek(0)
            content = fh.read()
            lines = content.splitlines()

            existing_ids = set()
            has_header = False
            if lines:
                header = lines[0]
                has_header = header.startswith("participant_id")
                for line in lines[1:]:
                    parts = line.split("\t")
                    if parts:
                        existing_ids.add(parts[0])

            if subject_id in existing_ids:
                return dest

            fh.seek(0, 2)  # append position
            if not has_header:
                fh.write("participant_id\tage\tsex\n")
            fh.write(f"{subject_id}\t{age_str}\t{sex_val}\n")

    write_participants_json(output_root)
    return dest


def write_derivative_sidecar(
    nifti_path: Path,
    sources: List[str],
    description: str,
    command_line: Optional[str] = None,
    software_versions: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Write a minimal BIDS-compliant JSON sidecar for a derived NIfTI.

    Parameters
    ----------
    nifti_path : Path
        Path to the NIfTI file; sidecar written alongside it.
    sources : list of str
        BIDS URIs for source files, e.g.
        ['bids::sub-001/ses-01/dwi/sub-001_ses-01_dwi.nii.gz'].
    description : str
        Human-readable description of the derivative.
    command_line : str, optional
        Full CLI command used.
    software_versions : dict, optional
        {'dipy': '1.9.0', ...}

    Returns
    -------
    Path to the sidecar JSON.
    """
    nifti_path = Path(nifti_path)
    # Strip .nii.gz or .nii to get the sidecar stem
    stem = nifti_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    sidecar = nifti_path.parent / f"{stem}.json"

    content: Dict[str, Any] = {
        "Sources": sources,
        "SpatialReference": "native",
        "Description": description,
        "GeneratedAt": datetime.now().isoformat(),
    }
    if command_line:
        content["CommandLine"] = command_line
    if software_versions:
        steps = [
            {"Name": name, "Software": name, "Version": ver}
            for name, ver in software_versions.items()
        ]
        content["ProcessingSteps"] = steps

    sidecar.write_text(json.dumps(content, indent=2) + "\n")
    return sidecar


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_sex(raw: Any) -> str:
    """Normalise DICOM PatientSex tag to BIDS-style M/F/O/n/a."""
    if raw is None:
        return "n/a"
    s = str(raw).strip().upper()
    if s in ("M", "MALE"):
        return "M"
    if s in ("F", "FEMALE"):
        return "F"
    if s in ("O", "OTHER"):
        return "O"
    return "n/a"


def hash_patient_id(patient_id: str) -> str:
    """
    Return an 8-char hex digest of a patient ID for anonymised subject labels.
    """
    return hashlib.sha256(patient_id.encode()).hexdigest()[:8]
