import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

@dataclass
class StudyInfo:
    """General metadata and global options for a batch run."""
    name: str = "Unnamed Study"
    description: str = ""
    global_options: Dict[str, Any] = field(default_factory=dict)

def load_manifest(path: Path) -> Tuple[StudyInfo, List[Dict[str, Any]]]:
    """
    Load and parse a JSON manifest file.
    
    Returns:
        Tuple of (StudyInfo, List of subject configurations)
    
    Raises:
        ValueError if manifest is invalid or missing required fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse manifest JSON: {e}")
        
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object")
        
    # extract study info
    study = StudyInfo(
        name=data.get("name", "Unnamed Study"),
        description=data.get("description", ""),
        global_options=data.get("options", {})
    )
    
    subjects_data = data.get("subjects", [])
    if not isinstance(subjects_data, list):
        raise ValueError("'subjects' field in manifest must be a list")
        
    subjects = []
    for i, s_data in enumerate(subjects_data):
        if not isinstance(s_data, dict):
            raise ValueError(f"Subject at index {i} must be an object")
            
        if "id" not in s_data:
            raise ValueError(f"Subject at index {i} is missing required 'id'")
            
        # Check for exclusivity of nifti/dicom
        has_nifti = "nifti" in s_data
        has_dicom = "dicom" in s_data
        
        if not has_nifti and not has_dicom:
            raise ValueError(f"Subject '{s_data['id']}' must specify either 'nifti' or 'dicom'")
        if has_nifti and has_dicom:
            raise ValueError(f"Subject '{s_data['id']}' cannot specify both 'nifti' and 'dicom'")
            
        # Standardize subject entry
        sub_entry = {
            "id": str(s_data["id"]),
            "session": s_data.get("session"),
            "nifti": Path(s_data["nifti"]) if has_nifti else None,
            "dicom": Path(s_data["dicom"]) if has_dicom else None,
            "series_uid": s_data.get("series_uid"),
            "options": s_data.get("options", {})
        }
        subjects.append(sub_entry)
        
    return study, subjects

def save_manifest_template(path: Path) -> None:
    """Generates a template manifest JSON file for users."""
    template = {
        "name": "Example Study",
        "description": "Template manifest for csttool batch processing",
        "options": {
            "denoise_method": "nlmeans",
            "generate_pdf": True
        },
        "subjects": [
            {
                "id": "sub-001",
                "session": "ses-01",
                "nifti": "/path/to/sub-001_ses-01_dwi.nii.gz"
            },
            {
                "id": "sub-002",
                "dicom": "/path/to/sub-002/dicom/",
                "series_uid": "1.2.840.113619.2.55.3..."
            }
        ]
    }
    
    with open(path, 'w') as f:
        json.dump(template, f, indent=2)
