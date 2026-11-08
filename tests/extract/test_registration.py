import json
from unittest.mock import patch, MagicMock

import nibabel as nib
import numpy as np
import pytest
from csttool.extract.modules.registration import (
    compute_affine_registration,
    compute_syn_registration,
    load_mni_template,
    save_registration_transforms,
)

def test_load_mni_template():
    """Test MNI template loading (mocked)."""
    # Mock the bundled data loader
    with patch('csttool.extract.modules.registration.load_bundled_mni152') as mock_load:
        # Mock returns (img, data, affine) tuple
        mock_img = MagicMock()
        mock_data = np.zeros((10,10,10))
        mock_affine = np.eye(4)
        mock_load.return_value = (mock_img, mock_data, mock_affine)

        img, data, affine = load_mni_template(contrast="T1")

        assert data.shape == (10,10,10)
        assert affine.shape == (4,4)
        mock_load.assert_called_once()

def test_compute_affine_registration(synthetic_image_data, synthetic_affine):
    """Test affine registration wrapper."""
    # We mock the internal AffineRegistration to avoid compute
    with patch('csttool.extract.modules.registration.AffineRegistration') as mock_reg_class:
        mock_reg_inst = mock_reg_class.return_value
        # Mock optimize method
        mock_map = MagicMock()
        mock_map.affine = np.eye(4)
        mock_reg_inst.optimize.return_value = mock_map
        
        static = synthetic_image_data
        moving = synthetic_image_data # Same image, should register perfectly
        
        result = compute_affine_registration(
            static, synthetic_affine,
            moving, synthetic_affine,
            level_iters=[10, 10, 5]
        )
        
        assert result.affine.shape == (4,4)
        mock_reg_inst.optimize.assert_called()

def test_compute_syn_registration(synthetic_image_data, synthetic_affine):
    """Test SyN registration wrapper."""
    with patch('csttool.extract.modules.registration.SymmetricDiffeomorphicRegistration') as mock_syn_class:
        mock_syn_inst = mock_syn_class.return_value
        
        # Mock optimize output (mapping)
        mock_mapping = MagicMock()
        mock_syn_inst.optimize.return_value = mock_mapping
        
        result = compute_syn_registration(
            synthetic_image_data, synthetic_affine,
            synthetic_image_data, synthetic_affine,
            level_iters=[1, 1, 1],
            prealign=np.eye(4)
        )
        
        mock_syn_inst.optimize.assert_called()
        assert result == mock_mapping


def test_save_registration_transforms_preserves_fields_and_geometry(tmp_path):
    mapping = MagicMock()
    mapping.dim = 3
    mapping.is_inverse = False
    mapping.disp_shape = np.array([3, 4, 5])
    mapping.disp_grid2world = np.diag([2.0, 2.0, 2.0, 1.0])
    mapping.domain_shape = np.array([3, 4, 5])
    mapping.domain_grid2world = mapping.disp_grid2world
    mapping.codomain_shape = np.array([6, 7, 8])
    mapping.codomain_grid2world = np.diag([1.0, 1.0, 1.0, 1.0])
    mapping.prealign = np.diag([1.1, 1.0, 0.9, 1.0])
    forward = np.zeros((3, 4, 5, 3), dtype=np.float32)
    backward = np.ones_like(forward)
    mapping.get_forward_field.return_value = forward
    mapping.get_backward_field.return_value = backward
    affine_map = MagicMock()
    affine_map.affine = mapping.prealign

    paths = save_registration_transforms(
        mapping, affine_map, tmp_path, "sub-test"
    )

    np.testing.assert_array_equal(np.load(paths["affine"]), mapping.prealign)
    np.testing.assert_array_equal(
        np.asarray(nib.load(paths["syn_forward"]).dataobj), forward
    )
    np.testing.assert_array_equal(
        np.asarray(nib.load(paths["syn_backward"]).dataobj), backward
    )
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["displacement_units"] == "physical millimetres"
    assert metadata["dipy_mapping"]["displacement_shape"] == [3, 4, 5]
    assert metadata["dipy_mapping"]["codomain_shape"] == [6, 7, 8]
