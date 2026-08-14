import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
from csttool.cli import main
import csttool.cli.commands.run # Ensure this is loaded

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

@patch('csttool.cli.commands.run.cmd_preprocess')
@patch('csttool.cli.commands.run.cmd_import')
@patch('csttool.cli.commands.run.cmd_track')
@patch('csttool.cli.commands.run.cmd_extract')
@patch('csttool.cli.commands.run.cmd_metrics')
def test_optional_preprocessing(mock_metrics, mock_extract, mock_track, mock_import, mock_preprocess, tmp_path):
    """
    Test that preprocessing is skipped by default and enabled with --preprocess.
    We mock the actual command implementations to just verify they are called.
    """
    out_dir = tmp_path / "output_test"
    nifti_file = tmp_path / "test.nii.gz"
    nifti_file.touch()
    
    # Mock import to return success
    mock_import.return_value = {'nifti_path': str(nifti_file)}
    
    # Mock other commands to return success/paths so pipeline continues
    mock_preprocess.return_value = {'preprocessed_path': str(out_dir / "preproc.nii.gz")}
    mock_track.return_value = {
        'tractogram_path': str(out_dir / "test.trk"),
        'fa_path': str(out_dir / "fa.nii.gz"),
        'md_path': str(out_dir / "md.nii.gz")
    }
    mock_extract.return_value = {
        'cst_left_path': str(out_dir / "cst_left.trk"),
        'cst_right_path': str(out_dir / "cst_right.trk"),
        'stats': {'cst_total_count': 100}
    }
    mock_metrics.return_value = {'success': True}

    # Case 1: Default (No --preprocess flag)
    # --------------------------------------
    cmd_default = [
        'run',
        '--nifti', str(nifti_file),
        '--out', str(out_dir / "default"),
        '--skip-check'
    ]
    
    assert run_cli(cmd_default)
    
    # Assert cmd_preprocess was NOT called
    mock_preprocess.assert_not_called()
    
    # Assert track was called with original nifti (passthrough)
    call_args = mock_track.call_args[0][0]
    assert str(call_args.nifti) == str(nifti_file)
    
    # Assert metrics was called with pipeline_metadata indicating skipped
    metrics_args = mock_metrics.call_args[0][0]
    # Check if pipeline_metadata exists in the args namespace
    assert hasattr(metrics_args, 'pipeline_metadata')
    # M4 metadata-schema correction: the status says only what csttool did.
    # It used to read 'Skipped (External Preprocessing Used)', which asserted
    # something about the input that nobody had declared and csttool cannot
    # verify. That claim is now a separate, explicitly declared field.
    preproc_meta = metrics_args.pipeline_metadata['preprocessing']
    assert preproc_meta['status'] == 'Skipped'
    assert preproc_meta['performed_by_csttool'] is False
    assert preproc_meta['external_correction'] == {
        'declared': 'unknown',
        'verified_by_csttool': False,
        'source': 'user-declaration',
    }
    
    # Reset mocks
    mock_preprocess.reset_mock()
    mock_track.reset_mock()
    
    # Case 2: With --preprocess flag
    # ------------------------------
    cmd_with_flag = [
        'run',
        '--nifti', str(nifti_file),
        '--out', str(out_dir / "flagged"),
        '--skip-check',
        '--preprocess'
    ]
    
    assert run_cli(cmd_with_flag)
    
    # Assert cmd_preprocess WAS called
    mock_preprocess.assert_called_once()
    
    # Assert track was called with preprocessed path
    call_args = mock_track.call_args[0][0]
    # We mocked preprocess to return "preproc.nii.gz", so track should receive that
    assert "preproc.nii.gz" in str(call_args.nifti)
