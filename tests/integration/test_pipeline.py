import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from csttool.cli import main
import sys
import numpy as np
import nibabel as nib

# Helper to run CLI commands
def run_cli(args):
    """Run csttool CLI with arguments."""
    with patch.object(sys, 'argv', ['csttool'] + args):
        try:
            main()
            return True
        except SystemExit as e:
            return e.code == 0
        except Exception as e:
            print(f"CLI Error: {e}")
            return False

def test_cmd_check():
    """Test the check command."""
    assert run_cli(['check'])

def test_step_by_step_pipeline(synthetic_nifti_data, tmp_path):
    """
    Test the pipeline step-by-step:
    Preprocess -> Track -> Extract -> Metrics
    
    (Skipping import step here as we start with synthetic NIfTI)
    """
    out_dir = tmp_path / "output_manual"
    out_dir.mkdir()
    
    # 1. Preprocess
    # We use minimal args. PIESNO might fail on synthetic noise-free data, 
    # so we might see warnings, but it should produce output.
    s = synthetic_nifti_data
    assert run_cli(['preprocess', '--nifti', str(s), '--out', str(out_dir), '--verbose'])
    
    # Check output
    # Preprocess output dir changed to 'preprocessed' in recent updates?
    preproc_dir = out_dir / "preprocessed"
    if not preproc_dir.exists():
        preproc_dir = out_dir / "nifti"
    
    # If standard subdirs don't exist, check if files are in out_dir directly
    if not preproc_dir.exists():
        preproc_dir = out_dir

    # Find the output file
    # Note: Filename might be slightly different than expected, so use glob
    # Fix: Ensure we don't pick up the mask file (which also contains 'preproc' and ends in .nii.gz)
    found_files = list(preproc_dir.glob("*_preproc_*.nii.gz"))
    data_files = [f for f in found_files if "mask" not in f.name]
    
    assert len(data_files) > 0, f"No preprocessed data files found in {preproc_dir} or {out_dir}"
    preproc_file = data_files[0]
    
    assert preproc_file.exists()
    
    # 2. Track
    track_dir = out_dir / "tractography"
    assert run_cli([
        'track', 
        '--nifti', str(preproc_file), 
        '--out', str(track_dir),
        '--fa-thr', '0.1',  # Low threshold for synthetic data
        '--sh-order', '2',  # Low order for few gradients
        '--verbose'
    ])
    
    # Track command often appends suffixes, so rely on directory content or known structure
    # Usually track produces 'tractograms/' and 'scalar_maps/'
    trk_files = list((track_dir / "tractograms").glob("*.trk"))
    assert len(trk_files) > 0, "No tractogram produced"
    trk_file = trk_files[0]
    
    fa_files = list((track_dir / "scalar_maps").glob("*_fa.nii.gz"))
    assert len(fa_files) > 0, "No FA map produced"
    fa_file = fa_files[0]
    
    assert trk_file.exists()
    assert fa_file.exists()
    
    # 3. Extract
    extract_dir = out_dir / "cst"
    # We need to mock registration to avoid MNI fetching/alignment issues likely to fail on random data
    # We'll mock 'csttool.cli.cmd_extract' internals or just 'register_mni_to_subject'

    with patch('csttool.extract.modules.registration.register_mni_to_subject') as mock_reg, \
         patch('csttool.extract.modules.registration.load_mni_template') as mock_load, \
         patch('csttool.extract.modules.warp_atlas_to_subject.fetch_harvard_oxford') as mock_atlas, \
         patch('csttool.extract.modules.passthrough_filtering.extract_cst_passthrough') as mock_extract, \
         patch('csttool.extract.modules.coordinate_validation.validate_tractogram_coordinates') as mock_validate:

        # Mock coordinate validation to always pass
        mock_validate.return_value = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'tractogram_info': {'n_streamlines': 81},
            'reference_info': {'shape': (10, 10, 10), 'orientation': 'RAS', 'bounds_mm': {}}
        }

        # Mock extraction result
        dummy_streamlines = [np.zeros((10, 3)) for _ in range(5)]
        
        mock_extract.return_value = {
            'cst_left': dummy_streamlines,
            'cst_right': dummy_streamlines,
            'cst_combined': dummy_streamlines + dummy_streamlines,
            'cst_left_indices': [0, 1, 2, 3, 4],
            'cst_right_indices': [0, 1, 2, 3, 4],
            'stats': {
                'cst_left_count': 5,
                'cst_right_count': 5,
                'cst_total_count': 10,
                'extraction_rate': 100.0
            }
        }
         
        # Mock registration result
        mock_reg.return_value = {
            'mapping': MagicMock(),
            'affine_map': MagicMock(),
            'subject_affine': np.eye(4),
            'subject_shape': (10,10,10),
            'mni_affine': np.eye(4),
            'mni_shape': (10,10,10),
            'warped_template_path': None
        }
        mock_reg.return_value['mapping'].transform_inverse.return_value = np.zeros((10,10,10)) # transformed atlas
        mock_reg.return_value['mapping'].transform.return_value = np.zeros((10,10,10))
        mock_reg.return_value['affine_map'].transform_inverse.return_value = np.zeros((10,10,10)) 
        
        # Mock template loading
        mock_load.return_value = (MagicMock(), np.zeros((10,10,10)), np.eye(4))

        # Mock Harvard-Oxford atlas loading with real nibabel images
        mock_cort_img = nib.Nifti1Image(np.zeros((10,10,10)), np.eye(4))
        mock_subcort_img = nib.Nifti1Image(np.zeros((10,10,10)), np.eye(4))

        mock_atlas.return_value = {
            'cortical_path': '/mock/path/cortical.nii.gz',
            'subcortical_path': '/mock/path/subcortical.nii.gz',
            'cortical_img': mock_cort_img,
            'subcortical_img': mock_subcort_img,
        }

        assert run_cli([
            'extract',
            '--tractogram', str(trk_file),
            '--fa', str(fa_file),
            '--out', str(extract_dir),
            '--extraction-method', 'passthrough',
            '--verbose'
        ])
        
    cst_left_files = list((extract_dir / "trk").glob("*cst_left.trk"))
    if not cst_left_files:
        cst_left_files = list(extract_dir.glob("*cst_left.trk"))
    
    assert len(cst_left_files) > 0, "No Left CST file found"
    cst_left = cst_left_files[0]
    
    # We assume right also exists if left exists given our success
    cst_right_files = list((extract_dir / "trk").glob("*cst_right.trk"))
    if not cst_right_files:
        cst_right_files = list(extract_dir.glob("*cst_right.trk"))
    cst_right = cst_right_files[0]
    
    assert cst_left.exists()
    assert cst_right.exists()
    
    # 4. Metrics
    metrics_dir = out_dir / "metrics"
    assert run_cli([
        'metrics',
        '--cst-left', str(cst_left),
        '--cst-right', str(cst_right),
        '--fa', str(fa_file),
        '--out', str(metrics_dir),
        '--verbose'
    ])
    
    # Check for report or metrics file
    report_file = metrics_dir / "subject_bilateral_metrics.json"
    if not report_file.exists():
        report_file = metrics_dir / "subject_bilateral_report.json"
        
    assert report_file.exists(), f"Metrics report not found in {metrics_dir}"

def test_full_run_command(synthetic_dicom_dir, tmp_path):
    """Test the 'run' command (end-to-end)."""

    out_dir = tmp_path / "output_run"

    # Pre-create the synthetic NIfTI that the mocked ingest will "produce"
    nifti_path = out_dir / "dwi" / "subject_dwi.nii.gz"
    nifti_path.parent.mkdir(parents=True, exist_ok=True)

    shape = (10, 10, 10, 6)
    nib.save(nib.Nifti1Image(np.random.random(shape).astype(np.float32), np.eye(4)), nifti_path)
    bval_path = out_dir / "dwi" / "subject_dwi.bval"
    bvec_path = out_dir / "dwi" / "subject_dwi.bvec"

    np.savetxt(bval_path, [0, 1000, 1000, 1000, 1000, 1000], fmt='%d')
    bvecs = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]).T
    np.savetxt(bvec_path, bvecs, fmt='%.8f')

    mock_ingest_result = {
        'nifti_path': nifti_path,
        'bval_path': bval_path,
        'bvec_path': bvec_path,
        'subject_id': 'test_subj',
        'success': True,
        'warnings': [],
        'converter': 'mocked',
    }

    # Mock registration and extraction
    with patch('csttool.ingest.run_ingest_pipeline', return_value=mock_ingest_result), \
         patch('csttool.extract.modules.registration.register_mni_to_subject') as mock_reg, \
         patch('csttool.extract.modules.registration.load_mni_template') as mock_load, \
         patch('csttool.extract.modules.warp_atlas_to_subject.fetch_harvard_oxford') as mock_atlas, \
         patch('csttool.extract.modules.passthrough_filtering.extract_cst_passthrough') as mock_extract, \
         patch('csttool.extract.modules.coordinate_validation.validate_tractogram_coordinates') as mock_validate:

        # Mock coordinate validation to always pass
        mock_validate.return_value = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'tractogram_info': {'n_streamlines': 81},
            'reference_info': {'shape': (10, 10, 10), 'orientation': 'RAS', 'bounds_mm': {}}
        }

        # Mock extraction result
        # Streamlines should be a list of arrays (or Streamlines object), not a tuple
        dummy_streamlines = [np.zeros((10, 3)) for _ in range(5)]
        
        mock_extract.return_value = {
            'cst_left': dummy_streamlines,
            'cst_right': dummy_streamlines,
            'cst_combined': dummy_streamlines + dummy_streamlines,
            'cst_left_indices': [0, 1, 2, 3, 4],
            'cst_right_indices': [0, 1, 2, 3, 4],
            'stats': {
                'cst_left_count': 5,
                'cst_right_count': 5,
                'cst_total_count': 10,
                'extraction_rate': 100.0
            }
        }
         
        mock_reg.return_value = {
            'mapping': MagicMock(),
            'affine_map': MagicMock(),
            'subject_affine': np.eye(4),
            'subject_shape': shape[:3],
            'mni_affine': np.eye(4),
            'mni_shape': shape[:3],
            'warped_template_path': None
        }
        mock_reg.return_value['mapping'].transform_inverse.return_value = np.zeros((10,10,10)) # transformed atlas
        mock_reg.return_value['mapping'].transform.return_value = np.zeros((10,10,10))
        mock_reg.return_value['affine_map'].transform_inverse.return_value = np.zeros((10,10,10))

        # Mock Harvard-Oxford atlas loading with real nibabel images
        mock_cort_img = nib.Nifti1Image(np.zeros((10,10,10)), np.eye(4))
        mock_subcort_img = nib.Nifti1Image(np.zeros((10,10,10)), np.eye(4))

        mock_atlas.return_value = {
            'cortical_path': '/mock/path/cortical.nii.gz',
            'subcortical_path': '/mock/path/subcortical.nii.gz',
            'cortical_img': mock_cort_img,
            'subcortical_img': mock_subcort_img,
        }

        # run command logic: check -> import -> preproc -> track -> extract -> metrics
        
        # IMPORTANT: 'run' command inside `cli.py` might call `cmd_check`, `cmd_import` etc. directly.
        # We need to ensure that mocked resources are available during the call.
        
        # Also, run command takes --series. Our synthetic dir has one series.
        
        cmd = [
            'run',
            '--dicom', str(synthetic_dicom_dir.parent), # study dir
            '--out', str(out_dir),
            '--subject-id', 'test_subj',
            '--series', '1',
            '--skip-check', # skip env check to save time/noise
            '--sh-order', '2',
            '--verbose'
        ]
        
        # We need to make sure the pipeline actually finds the NIfTI we "converted"
        # The import step uses the result from run_ingest_pipeline, which calls mock_convert
        
        assert run_cli(cmd)
        
    # After _write_bids_derivatives, the metrics JSON is moved from metrics/ to
    # sub-<id>/reports/ and the stage directories are removed.
    reports_dir = out_dir / "sub-test_subj" / "reports"
    assert reports_dir.exists(), f"BIDS reports dir not found: {reports_dir}"
    metrics_files = list(reports_dir.glob("*metrics*.json"))
    assert len(metrics_files) > 0, f"No metrics JSON found in {reports_dir}"
