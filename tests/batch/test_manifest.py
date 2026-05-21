import pytest
import json
from pathlib import Path
from csttool.batch.modules.manifest import load_manifest, save_manifest_template

def test_save_manifest_template(tmp_path):
    t_path = tmp_path / "template.json"
    save_manifest_template(t_path)
    assert t_path.exists()
    
    with open(t_path, 'r') as f:
        data = json.load(f)
    assert "subjects" in data
    assert len(data["subjects"]) > 0

def test_load_valid_manifest(tmp_path):
    m_path = tmp_path / "study.json"
    content = {
        "name": "Study Alpha",
        "options": {"preprocessing": False},
        "subjects": [
            {
                "id": "sub-01",
                "session": "ses-01",
                "nifti": "/data/sub-01.nii.gz",
                "options": {"sh_order": 4}
            },
            {
                "id": "sub-02",
                "dicom": "/data/sub-02/dcm",
                "series_uid": "1.2.840"
            }
        ]
    }
    with open(m_path, 'w') as f:
        json.dump(content, f)
        
    study, subjects = load_manifest(m_path)
    assert study.name == "Study Alpha"
    assert study.global_options["preprocessing"] is False
    assert len(subjects) == 2
    
    s1 = subjects[0]
    assert s1["id"] == "sub-01"
    assert s1["session"] == "ses-01"
    assert s1["nifti"].as_posix() == "/data/sub-01.nii.gz"
    assert s1["options"]["sh_order"] == 4
    
    s2 = subjects[1]
    assert s2["dicom"] is not None
    assert s2["series_uid"] == "1.2.840"

def test_load_invalid_manifest(tmp_path):
    m_path = tmp_path / "bad.json"
    
    # Missing 'id'
    with open(m_path, 'w') as f:
        json.dump({"subjects": [{"nifti": "foo.nii"}]}, f)
    with pytest.raises(ValueError, match="missing required 'id'"):
        load_manifest(m_path)
        
    # Both nifti and dicom
    with open(m_path, 'w') as f:
        json.dump({"subjects": [{"id": "s1", "nifti": "f.nii", "dicom": "d/"}]}, f)
    with pytest.raises(ValueError, match="cannot specify both 'nifti' and 'dicom'"):
        load_manifest(m_path)
        
    # Neither nifti nor dicom
    with open(m_path, 'w') as f:
        json.dump({"subjects": [{"id": "s1"}]}, f)
    with pytest.raises(ValueError, match="must specify either 'nifti' or 'dicom'"):
        load_manifest(m_path)
