
import pytest
import numpy as np
import nibabel as nib
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from csttool.preprocess.modules.denoise import denoise
from csttool.preprocess.modules.load_dataset import load_dataset
from csttool.preprocess.modules.background_segmentation import background_segmentation
from csttool.preprocess.modules.gibbs_unringing import gibbs_unringing
from csttool.preprocess.modules.perform_motion_correction import perform_motion_correction
from csttool.preprocess.modules.save_preprocessed import save_preprocessed

# Helper for 4D synthetic data
@pytest.fixture
def synthetic_dwi_data():
    """Returns a synthetic 4D DWI data (10, 10, 10, 7)."""
    data = np.zeros((10, 10, 10, 7), dtype=np.float32)
    data[2:8, 2:8, 2:8, :] = 1.0  # Cube in the center
    # Add some random noise
    data += np.random.normal(0, 0.1, data.shape)
    return data

@pytest.fixture
def synthetic_gtab_fixture(synthetic_bvals, synthetic_bvecs):
    from dipy.core.gradients import gradient_table
    return gradient_table(synthetic_bvals, bvecs=synthetic_bvecs)

class TestDenoise:
    def test_denoise_nlmeans(self, synthetic_dwi_data):
        denoised = denoise(synthetic_dwi_data, denoise_method="nlmeans", N=4)
        assert denoised.shape == synthetic_dwi_data.shape
        assert not np.array_equal(denoised, synthetic_dwi_data)

    def test_denoise_patch2self(self, synthetic_dwi_data, synthetic_bvals):
        denoised = denoise(synthetic_dwi_data, bvals=synthetic_bvals, denoise_method="patch2self")
        assert denoised.shape == synthetic_dwi_data.shape
        assert not np.array_equal(denoised, synthetic_dwi_data)

    def test_denoise_with_mask(self, synthetic_dwi_data):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        denoised = denoise(synthetic_dwi_data, denoise_method="nlmeans", brain_mask=mask)
        assert denoised.shape == synthetic_dwi_data.shape

    def test_invalid_method(self, synthetic_dwi_data):
        with pytest.raises(ValueError, match="Invalid denoise method"):
            denoise(synthetic_dwi_data, denoise_method="invalid")

class TestLoadDataset:
    def test_load_nifti(self, tmp_path, synthetic_nifti, synthetic_bvals, synthetic_bvecs):
        # Setup dummy files
        fname = "test_data"
        nii_path = tmp_path / f"{fname}.nii.gz"
        nib.save(synthetic_nifti, nii_path)
        
        bval_path = tmp_path / f"{fname}.bval"
        np.savetxt(bval_path, synthetic_bvals)
        
        bvec_path = tmp_path / f"{fname}.bvec"
        np.savetxt(bvec_path, synthetic_bvecs.T)
        
        nii, gtab, nifti_dir, metadata = load_dataset(str(tmp_path), fname)
        
        assert isinstance(nii, nib.Nifti1Image)
        assert str(nifti_dir) == str(tmp_path)
        assert len(gtab.bvals) == len(synthetic_bvals)

    def test_gradient_file_detection_plural(self, tmp_path, synthetic_nifti, synthetic_bvals, synthetic_bvecs):
        # Test .bvals and .bvecs extensions
        fname = "test_data_plural"
        nib.save(synthetic_nifti, tmp_path / f"{fname}.nii.gz")
        np.savetxt(tmp_path / f"{fname}.bvals", synthetic_bvals)
        np.savetxt(tmp_path / f"{fname}.bvecs", synthetic_bvecs.T)
        
        nii, gtab, _, _ = load_dataset(str(tmp_path), fname)
        assert len(gtab.bvals) == len(synthetic_bvals)

    @patch('csttool.preprocess.modules.load_dataset.dicom2nifti.dicom_series_to_nifti')
    @patch('csttool.preprocess.modules.load_dataset.nib.load')
    def test_dicom_conversion(self, mock_load, mock_dicom2nifti, tmp_path):
        # Create a dummy dicom file
        dcm_dir = tmp_path / "dicom"
        dcm_dir.mkdir()
        (dcm_dir / "test.dcm").touch()
        
        fname = "converted"
        nifti_dir = tmp_path / "nifti"
        
        # Mock returns
        mock_dicom2nifti.return_value = {
            "NII_FILE": str(nifti_dir / f"{fname}.nii.gz"),
            "BVAL_FILE": str(nifti_dir / f"{fname}.bval"),
            "BVEC_FILE": str(nifti_dir / f"{fname}.bvec")
        }
        
        # We need to mock reading the bvals/bvecs if we want the rest to succeed, 
        # or we just mock read_bvals_bvecs
        with patch('csttool.preprocess.modules.load_dataset.read_bvals_bvecs') as mock_read:
            mock_read.return_value = (np.array([0, 1000]), np.array([[0,0,0], [1,0,0]]))
            
            nii, gtab, out_dir, metadata = load_dataset(str(dcm_dir), fname)
            
            mock_dicom2nifti.assert_called_once()
            assert str(out_dir) == str(nifti_dir)


class TestBackgroundSegmentation:
    def test_background_segmentation(self, synthetic_dwi_data, synthetic_gtab_fixture):
        masked_data, mask = background_segmentation(synthetic_dwi_data, gtab=synthetic_gtab_fixture)
        assert masked_data.shape == synthetic_dwi_data.shape
        assert mask.shape == synthetic_dwi_data.shape[:3]
        assert mask.dtype == np.uint8 or mask.dtype == bool # median_otsu returns mask as various types sometimes? 
        # Actually median_otsu returns mask as whatever underlying dipy returns, usually unit8 or bool. 
        # The function docstring says np.ndarray.

    def test_background_segmentation_no_gtab(self, synthetic_dwi_data):
        masked_data, mask = background_segmentation(synthetic_dwi_data, gtab=None)
        assert masked_data.shape == synthetic_dwi_data.shape
        assert mask.shape == synthetic_dwi_data.shape[:3]

class TestGibbsUnringing:
    def test_gibbs_unringing(self, synthetic_dwi_data):
        corrected = gibbs_unringing(synthetic_dwi_data, slice_axis=2)
        assert corrected.shape == synthetic_dwi_data.shape


    def test_invalid_axis(self, synthetic_dwi_data):
        with pytest.raises(ValueError):
            gibbs_unringing(synthetic_dwi_data, slice_axis=5)

class TestMotionCorrection:
    def test_perform_motion_correction(self, synthetic_dwi_data, synthetic_gtab_fixture, synthetic_affine):
        corrected, reg_affines = perform_motion_correction(
            synthetic_dwi_data, 
            synthetic_gtab_fixture, 
            synthetic_affine
        )
        assert corrected.shape == synthetic_dwi_data.shape
        # reg_affines is an array of shape (4, 4, num_volumes)
        assert reg_affines.shape[-1] == synthetic_dwi_data.shape[-1]

    def test_perform_motion_correction_with_mask(self, synthetic_dwi_data, synthetic_gtab_fixture, synthetic_affine):
        mask = np.zeros(synthetic_dwi_data.shape[:3], dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        
        corrected, reg_affines = perform_motion_correction(
            synthetic_dwi_data, 
            synthetic_gtab_fixture, 
            synthetic_affine,
            brain_mask=mask
        )
        assert corrected.shape == synthetic_dwi_data.shape

class TestSavePreprocessed:
    def test_save_preprocessed(self, tmp_path, synthetic_dwi_data, synthetic_affine):
        output_dir = tmp_path / "output"
        output_paths = save_preprocessed(
            synthetic_dwi_data,
            synthetic_affine,
            output_dir,
            "test_subj",
            create_report=True
        )
        
        assert (output_dir / "test_subj.nii.gz").exists()
        assert (output_dir / "test_subj_report.json").exists()
        assert output_paths['data'] == output_dir / "test_subj.nii.gz"

    def test_save_with_mask_and_grads(self, tmp_path, synthetic_dwi_data, synthetic_affine):
        output_dir = tmp_path / "output"
        
        # Create dummy source gradient files
        src_grads = tmp_path / "src"
        src_grads.mkdir()
        (src_grads / "orig.bval").touch()
        (src_grads / "orig.bvec").touch()
        
        mask = np.ones(synthetic_dwi_data.shape[:3])
        
        output_paths = save_preprocessed(
            synthetic_dwi_data,
            synthetic_affine,
            output_dir,
            "test_subj",
            gradient_files={'bval': src_grads / "orig.bval", 'bvec': src_grads / "orig.bvec"},
            brain_mask=mask
        )
        
        assert (output_dir / "test_subj.bval").exists()
        assert (output_dir / "test_subj.bvec").exists()
        assert (output_dir / "test_subj_mask.nii.gz").exists()


class TestPreprocessingVisualizationPaths:
    """Regression guard for NEW-1: preprocessing QC figures must land in a single
    ``<stage>/visualizations/`` directory (never a doubled
    ``visualizations/visualizations/``), which is exactly where the ``run`` BIDS
    reorg globs for them. A doubled path silently discarded every preproc figure.
    """

    def test_figures_land_in_single_visualizations_dir(
        self, tmp_path, synthetic_dwi_data, synthetic_gtab_fixture, synthetic_affine
    ):
        from csttool.preprocess.modules.visualizations import (
            save_all_preprocessing_visualizations,
        )

        stage_dir = tmp_path / "preprocessing"
        stage_dir.mkdir()
        mask = np.zeros(synthetic_dwi_data.shape[:3], dtype=bool)
        mask[2:8, 2:8, 2:8] = True

        save_all_preprocessing_visualizations(
            data_original=synthetic_dwi_data,
            data_denoised=synthetic_dwi_data * 0.9,
            data_masked=synthetic_dwi_data * 0.95,
            data_unringed=synthetic_dwi_data * 0.94,
            data_preprocessed=synthetic_dwi_data * 0.9,
            brain_mask=mask,
            gtab=synthetic_gtab_fixture,
            output_dir=stage_dir,
            stem="sub-x",
            denoise_method="nlmeans",
            affine=synthetic_affine,
            verbose=False,
        )

        viz_dir = stage_dir / "visualizations"
        pngs = sorted(p.name for p in viz_dir.glob("*.png"))
        # The BIDS reorg globs exactly this directory (non-recursive).
        assert pngs, "no preprocessing figures reached <stage>/visualizations/"
        for name in pngs:
            assert (viz_dir / name).stat().st_size > 1000  # non-empty
        # The old doubled path must not exist.
        assert not (viz_dir / "visualizations").exists()
