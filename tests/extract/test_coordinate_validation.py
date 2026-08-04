"""
Tests for coordinate validation module.

Tests the critical coordinate validation system that prevents silent failures.
"""

import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
from nibabel.streamlines import Field
from nibabel.streamlines.trk import TrkFile
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from csttool.extract.modules.coordinate_validation import validate_tractogram_coordinates


@pytest.fixture
def fa_image(tmp_path):
    """Create a mock FA image in RAS orientation."""
    # Create 64x64x64 FA volume with reasonable FA values
    fa_data = np.random.rand(64, 64, 64) * 0.7

    # Create affine with 2mm isotropic voxels centered around origin
    affine = np.array([
        [2.0, 0.0, 0.0, -64.0],
        [0.0, 2.0, 0.0, -64.0],
        [0.0, 0.0, 2.0, -64.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    img = nib.Nifti1Image(fa_data, affine)
    fa_path = tmp_path / "fa.nii.gz"
    nib.save(img, fa_path)

    return fa_path, img


@pytest.fixture
def valid_tractogram_world(tmp_path, fa_image):
    """Create a tractogram with valid world coordinates (mm)."""
    fa_path, fa_img = fa_image

    # Create streamlines in world coordinates (mm) that fall within FA volume
    streamlines = [
        np.array([
            [-50.0, -40.0, -30.0],
            [-30.0, -20.0, -10.0],
            [-10.0, 0.0, 10.0],
            [10.0, 20.0, 30.0],
            [30.0, 40.0, 50.0],
        ], dtype=np.float32),
        np.array([
            [-40.0, -30.0, -20.0],
            [-20.0, -10.0, 0.0],
            [0.0, 10.0, 20.0],
            [20.0, 30.0, 40.0],
        ], dtype=np.float32),
    ]

    sft = StatefulTractogram(streamlines, fa_img, Space.RASMM)
    trk_path = tmp_path / "valid_world.trk"
    save_tractogram(sft, str(trk_path), bbox_valid_check=False)

    return trk_path


@pytest.fixture
def invalid_tractogram_voxel(tmp_path, fa_image):
    """Create a tractogram with voxel coordinates (invalid)."""
    fa_path, fa_img = fa_image

    # Create streamlines that look like voxel indices (0-64 range)
    streamlines = [
        np.array([
            [10.0, 15.0, 20.0],
            [20.0, 25.0, 30.0],
            [30.0, 35.0, 40.0],
            [40.0, 45.0, 50.0],
        ], dtype=np.float32),
        np.array([
            [15.0, 20.0, 25.0],
            [25.0, 30.0, 35.0],
            [35.0, 40.0, 45.0],
        ], dtype=np.float32),
    ]

    # Save with identity affine to simulate voxel space
    identity_affine = np.eye(4)
    voxel_img = nib.Nifti1Image(np.zeros((64, 64, 64)), identity_affine)
    sft = StatefulTractogram(streamlines, voxel_img, Space.RASMM)
    trk_path = tmp_path / "invalid_voxel.trk"
    save_tractogram(sft, str(trk_path), bbox_valid_check=False)

    return trk_path


@pytest.fixture
def out_of_bounds_tractogram(tmp_path, fa_image):
    """Create a tractogram with coordinates far outside FA volume."""
    fa_path, fa_img = fa_image

    # Create streamlines with coordinates far outside the FA volume
    streamlines = [
        np.array([
            [-200.0, -200.0, -200.0],
            [-100.0, -100.0, -100.0],
            [200.0, 200.0, 200.0],
            [300.0, 300.0, 300.0],
        ], dtype=np.float32),
    ]

    sft = StatefulTractogram(streamlines, fa_img, Space.RASMM)
    trk_path = tmp_path / "out_of_bounds.trk"
    save_tractogram(sft, str(trk_path), bbox_valid_check=False)

    return trk_path


def test_valid_tractogram_passes(fa_image, valid_tractogram_world):
    """Test that a valid tractogram passes validation."""
    fa_path, _ = fa_image

    result = validate_tractogram_coordinates(
        str(valid_tractogram_world),
        str(fa_path),
        strict=False,
        verbose=False
    )

    assert result['valid'] is True
    assert len(result['errors']) == 0
    assert result['tractogram_info']['n_streamlines'] == 2


def test_voxel_space_tractogram_fails(fa_image, invalid_tractogram_voxel):
    """Test that a tractogram in voxel space fails validation."""
    fa_path, _ = fa_image

    result = validate_tractogram_coordinates(
        str(invalid_tractogram_voxel),
        str(fa_path),
        strict=False,
        verbose=False
    )

    assert result['valid'] is False
    # Either catches it as voxel space or as incompatible header
    assert (any('voxel' in err.lower() for err in result['errors']) or
            any('header' in err.lower() for err in result['errors']) or
            any('incompatible' in err.lower() for err in result['errors']))


def test_voxel_space_strict_mode_raises(fa_image, invalid_tractogram_voxel):
    """Test that strict mode raises ValueError for voxel space tractogram."""
    fa_path, _ = fa_image

    with pytest.raises(ValueError) as exc_info:
        validate_tractogram_coordinates(
            str(invalid_tractogram_voxel),
            str(fa_path),
            strict=True,
            verbose=False
        )

    # Should raise ValueError with validation/compatibility error message
    assert 'Cannot validate' in str(exc_info.value) or 'validation failed' in str(exc_info.value).lower()


def test_out_of_bounds_tractogram_fails(fa_image, out_of_bounds_tractogram):
    """Test that a tractogram with out-of-bounds coordinates fails validation."""
    fa_path, _ = fa_image

    result = validate_tractogram_coordinates(
        str(out_of_bounds_tractogram),
        str(fa_path),
        strict=False,
        verbose=False
    )

    assert result['valid'] is False
    assert any('bounding box' in err.lower() or 'exceeds' in err.lower()
               for err in result['errors'])


def test_validation_includes_metadata(fa_image, valid_tractogram_world):
    """Test that validation result includes reference and tractogram metadata."""
    fa_path, _ = fa_image

    result = validate_tractogram_coordinates(
        str(valid_tractogram_world),
        str(fa_path),
        strict=False,
        verbose=False
    )

    # Check reference info
    assert 'reference_info' in result
    assert 'shape' in result['reference_info']
    assert 'orientation' in result['reference_info']
    assert 'bounds_mm' in result['reference_info']

    # Check tractogram info
    assert 'tractogram_info' in result
    assert 'n_streamlines' in result['tractogram_info']
    assert 'bounds_mm' in result['tractogram_info']


def test_empty_tractogram_warns(fa_image, tmp_path):
    """Test that an empty tractogram produces a warning."""
    fa_path, fa_img = fa_image

    # Create empty tractogram
    sft = StatefulTractogram([], fa_img, Space.RASMM)
    trk_path = tmp_path / "empty.trk"
    save_tractogram(sft, str(trk_path), bbox_valid_check=False)

    result = validate_tractogram_coordinates(
        str(trk_path),
        str(fa_path),
        strict=False,
        verbose=False
    )

    assert any('no streamlines' in warn.lower() for warn in result['warnings'])
    assert result['tractogram_info']['n_streamlines'] == 0


def test_verbose_mode_prints_summary(fa_image, valid_tractogram_world, capsys):
    """Test that verbose mode prints validation summary."""
    fa_path, _ = fa_image

    validate_tractogram_coordinates(
        str(valid_tractogram_world),
        str(fa_path),
        strict=False,
        verbose=True
    )

    captured = capsys.readouterr()
    assert 'COORDINATE VALIDATION' in captured.out
    assert 'Reference:' in captured.out
    assert 'Tractogram:' in captured.out
    # Check for new format: ✓ or ✗ symbols
    assert ('✓ Coordinate validation passed' in captured.out or
            '✗ Coordinate validation failed:' in captured.out)


# ---------------------------------------------------------------------------
# AU22: explicit affine-equality assertion
# ---------------------------------------------------------------------------

def _fa_image(tmp_path, voxel_size, origin):
    """Build an FA map with a specific voxel size + origin (RAS)."""
    aff = np.eye(4)
    aff[:3, :3] = np.diag(voxel_size)
    aff[:3, 3] = origin
    img = nib.Nifti1Image(np.random.rand(64, 64, 64) * 0.7, aff)
    p = tmp_path / f"fa_{voxel_size[0]}_{origin[0]}.nii.gz"
    nib.save(img, p)
    return p, img


def test_affine_mismatch_trk_fails_explicitly(tmp_path):
    """AU22: a .trk whose affine differs from the FA map (but whose world
    bounds overlap) must fail the explicit affine check, not pass on
    bounding-box overlap alone. This is the audit's exact scenario.
    """
    fa_path, _ = _fa_image(tmp_path, (2.0, 2.0, 2.0), (-64, -64, -64))

    # A tractogram saved in a DIFFERENT voxel space (1.5mm, different origin)
    # whose streamlines' world bounds still overlap the FA bounds.
    trk_aff = np.diag([1.5, 1.5, 1.5, 1.0])
    trk_aff[:3, 3] = [-50, -50, -50]
    trk_img = nib.Nifti1Image(np.zeros((40, 40, 40)), trk_aff)
    sl = [np.array([[-40, -40, -40], [0, 0, 0], [40, 40, 40]], float)]
    sft = StatefulTractogram(sl, trk_img, Space.RASMM)
    trk = tmp_path / "mismatched.trk"
    save_tractogram(sft, str(trk), bbox_valid_check=False)

    result = validate_tractogram_coordinates(
        str(trk), str(fa_path), strict=False, verbose=False
    )
    assert result['valid'] is False
    # The explicit affine assertion is the error source (not just the header).
    assert any('affines do not match' in e for e in result['errors'])


def test_affine_mismatch_trk_strict_raises(tmp_path):
    """Strict mode raises ValueError naming the affine mismatch."""
    fa_path, _ = _fa_image(tmp_path, (2.0, 2.0, 2.0), (-64, -64, -64))
    trk_aff = np.diag([1.5, 1.5, 1.5, 1.0])
    trk_aff[:3, 3] = [-50, -50, -50]
    trk_img = nib.Nifti1Image(np.zeros((40, 40, 40)), trk_aff)
    sft = StatefulTractogram(
        [np.array([[-40, -40, -40], [40, 40, 40]], float)], trk_img, Space.RASMM
    )
    trk = tmp_path / "mismatched_strict.trk"
    save_tractogram(sft, str(trk), bbox_valid_check=False)

    with pytest.raises(ValueError, match="affines differ"):
        validate_tractogram_coordinates(
            str(trk), str(fa_path), strict=True, verbose=False
        )


def test_affine_match_trk_passes(tmp_path):
    """A .trk with the same affine as the FA map passes the explicit check."""
    fa_path, fa_img = _fa_image(tmp_path, (2.0, 2.0, 2.0), (-64, -64, -64))
    sl = [np.array([[-50, -40, -30], [30, 40, 50]], float)]
    sft = StatefulTractogram(sl, fa_img, Space.RASMM)
    trk = tmp_path / "matched.trk"
    save_tractogram(sft, str(trk), bbox_valid_check=False)

    result = validate_tractogram_coordinates(
        str(trk), str(fa_path), strict=False, verbose=False
    )
    assert result['valid'] is True
    assert not any('affines do not match' in e for e in result['errors'])


def test_tck_warns_no_affine_asserted(tmp_path):
    """AU22: .tck carries no spatial header, so affine equality cannot be
    asserted — the validator must surface this as a warning rather than a
    silent pass. Previously this case passed with no warnings at all.
    """
    fa_path, fa_img = _fa_image(tmp_path, (2.0, 2.0, 2.0), (-64, -64, -64))
    # Save the .tck against the *matching* FA space so it would otherwise pass.
    sft = StatefulTractogram(
        [np.array([[-50, -40, -30], [30, 40, 50]], float)], fa_img, Space.RASMM
    )
    tck = tmp_path / "t.tck"
    save_tractogram(sft, str(tck), bbox_valid_check=False)

    result = validate_tractogram_coordinates(
        str(tck), str(fa_path), strict=False, verbose=False
    )
    assert any('no spatial header' in w for w in result['warnings'])
    # A matching-space .tck is still valid (no error); the warning is the point.
    assert result['valid'] is True


def test_affine_mismatch_tck_still_warns_not_silent(tmp_path):
    """AU22 regression guard: a .tck in a *different* space must not pass
    silently (it previously returned valid=True with zero warnings). The
    headerless-format warning at least surfaces that the affine was unchecked.
    """
    fa_path, _ = _fa_image(tmp_path, (2.0, 2.0, 2.0), (-64, -64, -64))
    trk_img = nib.Nifti1Image(np.zeros((40, 40, 40)), np.diag([1.5, 1.5, 1.5, 1.0]))
    sft = StatefulTractogram(
        [np.array([[-40, -40, -40], [40, 40, 40]], float)], trk_img, Space.RASMM
    )
    tck = tmp_path / "mismatched.tck"
    save_tractogram(sft, str(tck), bbox_valid_check=False)

    result = validate_tractogram_coordinates(
        str(tck), str(fa_path), strict=False, verbose=False
    )
    # At minimum, the unchecked-affine warning fires (no silent pass).
    assert any('no spatial header' in w for w in result['warnings'])
