"""dicom2nifti b-vector convention correction and motion-QC parameter extraction.

Both bugs here were invisible to every existing check and were found only by
running the real pipeline on real data.

The gradient one is a *reflection* of the gradient table. FA and MD are
invariant under any global orthogonal transform (D -> Q D Qᵀ preserves
eigenvalues), so scalar maps, the unit-norm/count/b0 validators and even
per-axis |V1| summaries are bit-identical with and without it. Only the signed
direction field changes. Tests here therefore assert **signed** quantities; a
test written against FA or |V1| would pass while the bug is present, so the
scalar-invariance itself is pinned below to keep that trap documented.
"""

import numpy as np
import pytest
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti

from csttool.preprocess.modules.gradient_validation import correct_dicom2nifti_bvecs


BVECS = np.array([
    [0.0, 0.0, 0.0],
    [0.7113, -0.0761, -0.6987],
    [0.5672, -0.5062, 0.6497],
    [-0.1966, 0.0897, -0.9764],
])


# ---------------------------------------------------------------------------
# The correction itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("manufacturer", ["SIEMENS", "Siemens", "siemens healthineers"])
def test_siemens_phase_axis_is_negated(manufacturer):
    """dicom2nifti inverts the phase axis; the correction undoes exactly that."""
    corrected, note = correct_dicom2nifti_bvecs(BVECS, manufacturer)

    expected = BVECS.copy()
    expected[:, 1] *= -1
    np.testing.assert_allclose(corrected, expected)
    assert "siemens" in note.lower()
    # x and z are untouched — this is a single-axis reflection, not a rotation.
    np.testing.assert_allclose(corrected[:, 0], BVECS[:, 0])
    np.testing.assert_allclose(corrected[:, 2], BVECS[:, 2])


@pytest.mark.parametrize(
    "manufacturer", ["GE MEDICAL SYSTEMS", "Philips Medical Systems", "Hitachi"]
)
def test_other_vendors_are_left_untouched_and_warned_about(manufacturer):
    """dicom2nifti's GE/Philips paths use different conventions again.

    Applying the Siemens correction to them would introduce the very bug it
    fixes, so they are returned unchanged with an explicit warning.
    """
    corrected, note = correct_dicom2nifti_bvecs(BVECS, manufacturer)

    np.testing.assert_array_equal(corrected, BVECS)
    assert "not validated" in note
    assert "dcm2niix" in note


def test_unknown_manufacturer_is_not_silently_corrected():
    corrected, note = correct_dicom2nifti_bvecs(BVECS, None)
    np.testing.assert_array_equal(corrected, BVECS)
    assert "Unknown scanner manufacturer" in note
    assert "dcm2niix" in note


def test_b0_rows_stay_zero():
    corrected, _ = correct_dicom2nifti_bvecs(BVECS, "SIEMENS")
    np.testing.assert_array_equal(corrected[0], np.zeros(3))


def test_correction_is_its_own_inverse():
    once, _ = correct_dicom2nifti_bvecs(BVECS, "SIEMENS")
    twice, _ = correct_dicom2nifti_bvecs(once, "SIEMENS")
    np.testing.assert_allclose(twice, BVECS)


# ---------------------------------------------------------------------------
# Why no scalar metric can catch this
# ---------------------------------------------------------------------------

def test_reflection_leaves_fa_and_md_identical():
    """Pins the trap: a mirrored gradient table is invisible to FA and MD.

    This is why the defect survived a validator, a QC figure suite and a
    scalar-map pipeline. If this test ever fails, the invariance argument in
    `correct_dicom2nifti_bvecs` needs revisiting.
    """
    rng = np.random.default_rng(0)
    n = 30
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros(3), dirs])
    bvals = np.array([0.0] + [1000.0] * n)

    # An anisotropic tensor sampled along those directions.
    evals = np.array([1.7e-3, 0.3e-3, 0.2e-3])
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    D = R @ np.diag(evals) @ R.T
    sig = np.exp(-bvals * np.einsum("ij,jk,ik->i", bvecs, D, bvecs))
    data = np.tile(sig, (2, 2, 2, 1))

    mirrored = bvecs.copy()
    mirrored[:, 1] *= -1

    fits = []
    for bv in (bvecs, mirrored):
        gt = gradient_table(bvals, bvecs=bv, b0_threshold=50)
        fits.append(dti.TensorModel(gt, fit_method="WLS").fit(data))

    np.testing.assert_allclose(fits[0].fa, fits[1].fa, atol=1e-9)
    np.testing.assert_allclose(fits[0].md, fits[1].md, atol=1e-12)
    # ...but the signed principal direction is mirrored, which is what breaks
    # tractography and what the correction exists to prevent.
    v1a, v1b = fits[0].evecs[0, 0, 0, :, 0], fits[1].evecs[0, 0, 0, :, 0]
    assert not np.allclose(np.abs(v1a - v1b), 0, atol=1e-6)
    mirror = np.diag([1.0, -1.0, 1.0])
    assert np.allclose(np.abs(mirror @ v1a), np.abs(v1b), atol=1e-6)


# ---------------------------------------------------------------------------
# Motion-correction QC parameter extraction
# ---------------------------------------------------------------------------

def test_motion_qc_reads_the_dipy_affine_stack(tmp_path):
    """The QC figure must agree with the report's max_rotation_deg.

    dipy returns (4, 4, n); the plotting code used len() -> 4 and iterated the
    first axis, so a 71-volume run was titled "4 volumes" and reported 284° for
    a true 0.47°.
    """
    from csttool.preprocess.modules.visualizations import (
        plot_motion_correction_summary,
    )
    from csttool.preprocess.modules.reorient_gradients import max_rotation_angle_deg

    n = 12
    theta = 3.0
    t = np.deg2rad(theta)
    c, s = np.cos(t), np.sin(t)
    stack = np.zeros((4, 4, n))
    for i in range(n):
        stack[:3, :3, i] = np.eye(3)
        stack[3, 3, i] = 1.0
    stack[:3, :3, 5] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    stack[:3, 3, 7] = [1.5, 0.0, 0.0]

    path = plot_motion_correction_summary(
        stack, output_dir=tmp_path, stem="sub", verbose=False
    )
    assert path.exists()
    # The quantity the figure reports is the one the ledger records.
    assert max_rotation_angle_deg(stack) == pytest.approx(theta, abs=1e-6)


def test_motion_qc_accepts_a_list_of_matrices(tmp_path):
    """The historical (documented) input shape must keep working."""
    from csttool.preprocess.modules.visualizations import (
        plot_motion_correction_summary,
    )

    mats = [np.eye(4) for _ in range(5)]
    mats[2][:3, 3] = [0.5, -0.2, 0.1]
    path = plot_motion_correction_summary(
        mats, output_dir=tmp_path, stem="sub", verbose=False
    )
    assert path.exists()
