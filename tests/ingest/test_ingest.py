import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from csttool.ingest import run_ingest_pipeline, scan_study


def test_scan_study_empty(tmp_path):
    """Test scanning empty directory."""
    series = scan_study(tmp_path)
    assert series == []


def test_scan_study_with_files(tmp_path):
    """Test scanning directory with mock DICOMs."""
    dcm_file = tmp_path / "test.dcm"
    with open(dcm_file, 'wb') as f:
        f.seek(128)
        f.write(b'DICM')

    series = scan_study(tmp_path)
    assert len(series) == 1
    assert series[0]['n_files'] == 1


@pytest.mark.skip(reason="integration test: requires real DICOM data and system dcm2niix")
def test_run_ingest_pipeline(tmp_path):
    """Full pipeline test — skipped in unit CI; run manually with real DICOM data."""
    pass
