"""
Tests for AU21: gradient-table validation and DWI bvec reorientation at load.

Three independent audits (GLM §3.9, Qwen §3.10) found that ``gradient_table``
was built with no bvec/bval sanity checks, and that dicom2nifti's
``reorient_nifti=True`` reorients the image but not the bvecs — a silent
gradient flip. These tests pin both the hard-fail validation and the
image+bvec reorientation.

Per the roadmap meta-lesson, every check is verified against running code
and uses a *non-trivial* fixture (an identity/LAS-by-LAS affine would pass
vacuously against a broken reorientation — the AU9/AU29 lesson).
"""

import numpy as np
import nibabel as nib
import pytest

from csttool.preprocess.modules.gradient_validation import (
    GradientTableValidationError,
    validate_bvals_bvecs,
    validate_gradient_table,
    reorient_dwi_to_ras,
)


# ---------------------------------------------------------------------------
# validate_bvals_bvecs — the four AU21 checks
# ---------------------------------------------------------------------------

class TestValidateBvalsBvecs:
    """Each check must raise GradientTableValidationError with an actionable msg."""

    def test_valid_table_passes(self):
        bvals = np.array([0, 1000, 1000, 1000])
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        bv, bc = validate_bvals_bvecs(bvals, bvecs)
        assert bv.shape == (4,)
        assert bc.shape == (4, 3)

    def test_transposes_3_by_N_bvecs(self):
        """FSL/dicom2nifti store bvecs as (3, N); a (3,3) array is ambiguous
        and kept as-is (DIPY convention). Use valid unit DWI rows."""
        bvals = np.array([0, 1000, 1000])
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])  # (3, 3), b0 row zero
        bv, bc = validate_bvals_bvecs(bvals, bvecs)
        assert bc.shape == (3, 3)

    def test_transposes_tall_3_by_N(self):
        bvals = np.array([0, 1000, 1000, 1000])
        bvecs = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # (3, 4)
        bv, bc = validate_bvals_bvecs(bvals, bvecs)
        assert bc.shape == (4, 3)

    def test_negative_bval_rejected(self):
        bvals = np.array([0, 1000, -5])
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        with pytest.raises(GradientTableValidationError, match="negative"):
            validate_bvals_bvecs(bvals, bvecs)

    def test_non_finite_bval_rejected(self):
        bvals = np.array([0, 1000, np.nan])
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        with pytest.raises(GradientTableValidationError, match="non-finite"):
            validate_bvals_bvecs(bvals, bvecs)

    def test_no_b0_rejected(self):
        bvals = np.array([1000, 1000])
        bvecs = np.array([[1, 0, 0], [0, 1, 0]])
        with pytest.raises(GradientTableValidationError, match="No b=0"):
            validate_bvals_bvecs(bvals, bvecs)

    def test_no_b0_respects_custom_threshold(self):
        """A b=30 shell is b0 only if the threshold is raised — proving the
        threshold is actually plumbed, not hardcoded to 50."""
        bvals = np.array([30, 1000])
        bvecs = np.array([[1, 0, 0], [0, 1, 0]])
        with pytest.raises(GradientTableValidationError, match="No b=0"):
            validate_bvals_bvecs(bvals, bvecs, b0_threshold=10)
        # With a higher threshold the b=30 volume counts as b0 and it passes.
        bv, bc = validate_bvals_bvecs(bvals, bvecs, b0_threshold=50)
        assert bv.shape == (2,)

    def test_non_unit_dwi_bvec_rejected(self):
        bvals = np.array([0, 1000])
        bvecs = np.array([[0, 0, 0], [2, 0, 0]])  # norm 2, not unit
        with pytest.raises(GradientTableValidationError, match="unit-norm"):
            validate_bvals_bvecs(bvals, bvecs)

    def test_zero_dwi_bvec_rejected(self):
        """A DWI volume with a zero bvec is not a unit vector — a real corruption."""
        bvals = np.array([0, 1000])
        bvecs = np.array([[0, 0, 0], [0, 0, 0]])
        with pytest.raises(GradientTableValidationError, match="unit-norm"):
            validate_bvals_bvecs(bvals, bvecs)

    def test_b0_bvec_may_be_zero(self):
        """b0 bvecs are conventionally [0,0,0] (dcm2niix) and must NOT trip the
        unit-norm check, which applies only to DWI volumes."""
        bvals = np.array([0, 1000])
        bvecs = np.array([[0, 0, 0], [1, 0, 0]])
        bv, bc = validate_bvals_bvecs(bvals, bvecs)
        assert np.allclose(bc[0], 0)

    def test_count_mismatch_rejected(self):
        bvals = np.array([0, 1000, 1000])
        bvecs = np.array([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(GradientTableValidationError, match="count mismatch"):
            validate_bvals_bvecs(bvals, bvecs)

    def test_wrong_bvec_columns_rejected(self):
        bvals = np.array([0, 1000])
        bvecs = np.array([[0, 0], [1, 0]])  # 2 columns
        with pytest.raises(GradientTableValidationError, match="3 columns"):
            validate_bvals_bvecs(bvals, bvecs)

    def test_non_finite_dwi_bvec_rejected(self):
        bvals = np.array([0, 1000])
        bvecs = np.array([[0, 0, 0], [np.nan, 0, 0]])
        with pytest.raises(GradientTableValidationError, match="non-finite"):
            validate_bvals_bvecs(bvals, bvecs)


class TestValidateGradientTable:
    """The wrapper builds a real DIPY GradientTable using our b0 threshold."""

    def test_builds_gradient_table(self):
        from dipy.core.gradients import GradientTable
        bvals = np.array([0, 1000, 1000, 1000])
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        gtab = validate_gradient_table(bvals, bvecs)
        assert isinstance(gtab, GradientTable)
        assert gtab.bvals.shape == (4,)

    def test_uses_our_b0_threshold(self):
        """The gtab's b0s_mask must reflect our threshold, not DIPY's default 50.

        Uses a b=5 (always b0) plus a b=30 shell whose classification flips.
        """
        bvals = np.array([5, 30, 1000])
        bvecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # threshold=10 -> only b=5 is b0; b=30 is DWI
        gtab = validate_gradient_table(bvals, bvecs, b0_threshold=10)
        assert gtab.b0s_mask[0]
        assert not gtab.b0s_mask[1]
        # threshold=50 -> b=5 and b=30 are both b0
        gtab2 = validate_gradient_table(bvals, bvecs, b0_threshold=50)
        assert gtab2.b0s_mask[0]
        assert gtab2.b0s_mask[1]

    def test_propagates_validation_errors(self):
        bvals = np.array([0, -5])
        bvecs = np.array([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(GradientTableValidationError):
            validate_gradient_table(bvals, bvecs)


# ---------------------------------------------------------------------------
# reorient_dwi_to_ras — the dicom2nifti bvec-reorientation fix
# ---------------------------------------------------------------------------

def _lps_image(n_vols=3):
    """A DWI image in LPS-ish voxel space (x,y inverted, z normal).

    Deliberately non-RAS and non-identity: an identity affine would make a
    reorientation test pass vacuously (the AU9/AU29 lesson).
    """
    affine = np.array([
        [-1, 0, 0, 10],
        [0, -1, 0, 20],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    data = np.random.random((4, 4, 4, n_vols)).astype(np.float32)
    return nib.Nifti1Image(data, affine)


class TestReorientDwiToRas:
    def test_image_becomes_ras(self):
        img = _lps_image()
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        ras_img, ras_bvecs = reorient_dwi_to_ras(img, bvecs)
        assert nib.aff2axcodes(ras_img.affine) == ("R", "A", "S")

    def test_bvecs_reoriented_consistently(self):
        """The x and y axis directions must flip sign in the new voxel frame."""
        img = _lps_image()
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          dtype=float)
        ras_img, ras_bvecs = reorient_dwi_to_ras(img, bvecs)
        # x-dir (1,0,0) in LPS-voxel -> (-1,0,0) in RAS-voxel; y similarly.
        assert np.allclose(ras_bvecs[1], [-1, 0, 0])
        assert np.allclose(ras_bvecs[2], [0, -1, 0])
        assert np.allclose(ras_bvecs[3], [0, 0, 1])

    def test_b0_row_preserved_zero(self):
        img = _lps_image()
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        _, ras_bvecs = reorient_dwi_to_ras(img, bvecs)
        assert np.allclose(ras_bvecs[0], 0)

    def test_dwi_bvecs_renormalized_to_unit(self):
        img = _lps_image()
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        _, ras_bvecs = reorient_dwi_to_ras(img, bvecs)
        norms = np.linalg.norm(ras_bvecs[1:], axis=1)
        assert np.allclose(norms, 1.0)

    def test_round_trip_recovers_originals(self):
        """Applying the inverse voxel-frame change recovers the input bvecs."""
        img = _lps_image()
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        ras_img, ras_bvecs = reorient_dwi_to_ras(img, bvecs)
        R = np.linalg.inv(ras_img.affine[:3, :3]) @ img.affine[:3, :3]
        recovered = (np.linalg.inv(R) @ ras_bvecs.T).T
        nonzero = np.linalg.norm(bvecs, axis=1) > 0
        assert np.allclose(recovered[nonzero], bvecs[nonzero])

    def test_accepts_3_by_N_bvecs(self):
        img = _lps_image(n_vols=3)
        bvecs = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # (3, 3)
        ras_img, ras_bvecs = reorient_dwi_to_ras(img, bvecs)
        assert ras_bvecs.shape == (3, 3)

    def test_already_ras_is_noop(self):
        """An already-RAS image reorients to itself (no spurious flip)."""
        img = nib.Nifti1Image(
            np.random.random((4, 4, 4, 2)).astype(np.float32), np.eye(4)
        )
        bvecs = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        ras_img, ras_bvecs = reorient_dwi_to_ras(img, bvecs)
        assert nib.aff2axcodes(ras_img.affine) == ("R", "A", "S")
        assert np.allclose(ras_bvecs[1], [1, 0, 0])
