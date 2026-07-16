import pytest
import numpy as np
import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import create_nifti_header

@pytest.fixture
def synthetic_affine():
    """Returns a simple identity affine for testing."""
    return np.eye(4)

@pytest.fixture
def synthetic_image_data():
    """Returns a small 10x10x10 synthetic 3D image."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    # Create some "features"
    data[2:8, 2:8, 2:8] = 1.0  # Cube in the center
    return data

@pytest.fixture
def synthetic_nifti(synthetic_image_data, synthetic_affine):
    """Returns a synthetic nibabel Nifti1Image."""
    return nib.Nifti1Image(synthetic_image_data, synthetic_affine)

@pytest.fixture
def synthetic_tractogram(synthetic_nifti, synthetic_affine):
    """Returns a simple synthetic StatefulTractogram."""
    # Create a few straight streamlines
    # Streamline 1: straight line along Z axis
    sl1 = np.array([[5, 5, 2], [5, 5, 3], [5, 5, 4], [5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 5, 8]], dtype=np.float32)
    # Streamline 2: slightly offset
    sl2 = np.array([[4, 4, 2], [4, 4, 3], [4, 4, 4], [4, 4, 5], [4, 4, 6], [4, 4, 7], [4, 4, 8]], dtype=np.float32)
    
    streamlines = [sl1, sl2]
    
    sft = StatefulTractogram(streamlines, synthetic_nifti, Space.RASMM)
    return sft

@pytest.fixture
def synthetic_z_gradient_data():
    """Returns a 10x10x10 volume ramping linearly from 0.0 to 1.0 along +Z.

    Unlike synthetic_image_data (a flat cube), this varies along the tract axis, so a
    profile computed on it changes if streamline orientation changes. A flat map cannot
    detect a flip, which is what let AU9 go unnoticed.
    """
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[:, :, :] = np.arange(10, dtype=np.float32) / 9.0
    return data

@pytest.fixture
def synthetic_z_gradient_nifti(synthetic_z_gradient_data, synthetic_affine):
    """Returns a Nifti1Image of the Z ramp. With an identity affine, world Z == voxel k."""
    return nib.Nifti1Image(synthetic_z_gradient_data, synthetic_affine)

@pytest.fixture
def synthetic_bvals():
    """Returns synthetic b-values (1 b0 and 6 DWIs)."""
    return np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])

@pytest.fixture
def synthetic_bvecs():
    """Returns synthetic b-vectors (1 b0 and 6 directions)."""
    # 6 directions along axes + diagonals
    bvecs = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])
    # Normalize non-zero vectors
    norm = np.linalg.norm(bvecs[1:], axis=1, keepdims=True)
    bvecs[1:] = bvecs[1:] / norm
    return bvecs

@pytest.fixture
def synthetic_gtab(synthetic_bvals, synthetic_bvecs):
    """Returns a synthetic gradient table."""
    from dipy.core.gradients import gradient_table
    return gradient_table(synthetic_bvals, bvecs=synthetic_bvecs)
