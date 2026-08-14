"""B-vector rotation for motion correction (M3).

`--perform-motion-correction` resampled every DWI volume and then copied the
*original* `.bvec` unchanged, so the shipped data/gradient pair was
geometrically inconsistent whenever the flag was used. These tests pin the
rotation's direction, its frame conversion, and the end-to-end chain.

The two named decision-gate tests are
`test_sign_flipped_orientation_conjugates_the_rotation` (frame) and
`test_planted_motion_is_recovered_end_to_end` (whole chain).
"""

import argparse
import shutil

import numpy as np
import nibabel as nib
import pytest
from dipy.core.gradients import gradient_table

from csttool.preprocess.modules.reorient_gradients import (
    max_rotation_angle_deg,
    rotate_bvecs_for_motion,
    rotation_part,
)


def rotz(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotx(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


BVALS = np.array([0.0, 1000.0, 1000.0, 1000.0])
BVECS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)
BVECS[3] /= np.linalg.norm(BVECS[3])
B0S_MASK = np.array([True, False, False, False])


def affine_stack(rotation: np.ndarray, n: int, translation=(0.0, 0.0, 0.0)):
    """A (4, 4, n) stack with the same world transform on every volume."""
    stack = np.zeros((4, 4, n))
    for i in range(n):
        stack[:3, :3, i] = rotation
        stack[:3, 3, i] = translation
        stack[3, 3, i] = 1.0
    return stack


# ---------------------------------------------------------------------------
# T3.1 — identity
# ---------------------------------------------------------------------------

def test_identity_motion_leaves_gradients_unchanged():
    out = rotate_bvecs_for_motion(
        BVALS, BVECS, affine_stack(np.eye(3), 4), B0S_MASK, np.eye(4)
    )
    np.testing.assert_allclose(out, BVECS, atol=1e-12)


def test_translation_alone_does_not_rotate_gradients():
    out = rotate_bvecs_for_motion(
        BVALS,
        BVECS,
        affine_stack(np.eye(3), 4, translation=(12.0, -7.0, 3.5)),
        B0S_MASK,
        np.eye(4),
    )
    np.testing.assert_allclose(out, BVECS, atol=1e-12)


# ---------------------------------------------------------------------------
# T3.2 — direction convention
# ---------------------------------------------------------------------------

def test_known_rotation_produces_the_inverse_bvec_rotation():
    """Motion by R means the gradient must be expressed as R^-1 g."""
    theta = 30.0
    out = rotate_bvecs_for_motion(
        BVALS, BVECS, affine_stack(rotz(theta), 4), B0S_MASK, np.eye(4)
    )
    expected = BVECS[1:] @ rotz(-theta).T
    np.testing.assert_allclose(out[1:], expected, atol=1e-8)
    # And emphatically not the forward rotation, which would double the error.
    assert not np.allclose(out[1:], BVECS[1:] @ rotz(theta).T, atol=1e-3)


# ---------------------------------------------------------------------------
# T3.3 — DECISION GATE: frame conjugation
# ---------------------------------------------------------------------------

def test_sign_flipped_orientation_conjugates_the_rotation():
    """LAS-stored data: the world rotation must be conjugated into voxel frame.

    With V = diag(-1, 1, 1), V^-1 Rz(-t) V == Rz(+t). Applying the world
    rotation naively would rotate the b-vectors the wrong way.
    """
    theta = 30.0
    V = np.diag([-1.0, 1.0, 1.0])
    image_affine = np.eye(4)
    image_affine[:3, :3] = V @ np.diag([2.0, 2.0, 2.0])

    out = rotate_bvecs_for_motion(
        BVALS, BVECS, affine_stack(rotz(theta), 4), B0S_MASK, image_affine
    )

    conjugated = np.linalg.inv(V) @ rotz(-theta) @ V
    expected = BVECS[1:] @ conjugated.T
    np.testing.assert_allclose(out[1:], expected, atol=1e-8)

    naive = BVECS[1:] @ rotz(-theta).T
    assert not np.allclose(out[1:], naive, atol=1e-3), (
        "sign-flipped storage must not behave like the RAS case"
    )
    # The conjugate of a z-rotation through an x-flip is the opposite z-rotation.
    np.testing.assert_allclose(conjugated, rotz(theta), atol=1e-12)


# ---------------------------------------------------------------------------
# T3.4 — DECISION GATE: anisotropic voxels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zooms", [(1.0, 1.0, 1.0), (2.0, 2.0, 2.5), (1.0, 1.0, 6.0)])
def test_result_is_invariant_to_voxel_size(zooms):
    """Voxel size is a sampling property; it cannot change a physical direction.

    This is what decides the M0 question of full-3x3 versus orthonormal frame
    conversion: A^-1 R^-1 A depends on the zooms, V^-1 R^-1 V does not.
    """
    theta = 25.0
    Vrot = rotz(17.0)  # an oblique orientation, so the conjugation is non-trivial
    image_affine = np.eye(4)
    image_affine[:3, :3] = Vrot @ np.diag(zooms)

    out = rotate_bvecs_for_motion(
        BVALS, BVECS, affine_stack(rotz(theta), 4), B0S_MASK, image_affine
    )

    expected = BVECS[1:] @ (np.linalg.inv(Vrot) @ rotz(-theta) @ Vrot).T
    np.testing.assert_allclose(out[1:], expected, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(out[1:], axis=1), 1.0, atol=1e-10)


def test_full_affine_conversion_would_disagree_for_anisotropic_voxels():
    """Guards the choice itself: the rejected formula gives a different answer.

    With an axis-aligned affine the correct answer is R^-1 g regardless of
    zooms. The full-3x3 conversion does not satisfy that, which is why it is
    not used.
    """
    theta = 25.0
    A = np.diag([1.0, 1.0, 6.0])
    image_affine = np.eye(4)
    image_affine[:3, :3] = A
    # Rotate about x, so the motion mixes the 1 mm and 6 mm axes.
    motion = rotx(theta)

    out = rotate_bvecs_for_motion(
        BVALS, BVECS, affine_stack(motion, 4), B0S_MASK, image_affine
    )
    np.testing.assert_allclose(out[1:], BVECS[1:] @ rotx(-theta).T, atol=1e-8)

    full_affine_result = BVECS[1:] @ (np.linalg.inv(A) @ rotx(-theta) @ A).T
    full_affine_result /= np.linalg.norm(full_affine_result, axis=1, keepdims=True)
    assert not np.allclose(out[1:], full_affine_result, atol=1e-3)


# ---------------------------------------------------------------------------
# T3.5 — b0 handling and shape contracts
# ---------------------------------------------------------------------------

def test_b0_rows_stay_exactly_zero_and_dwi_rows_stay_unit_norm():
    bvals = np.array([0.0, 1000.0, 0.0, 1000.0, 1000.0])
    bvecs = np.zeros((5, 3))
    bvecs[[1, 3, 4]] = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    b0s_mask = np.array([True, False, True, False, False])

    out = rotate_bvecs_for_motion(
        bvals, bvecs, affine_stack(rotz(40.0), 5), b0s_mask, np.eye(4)
    )

    assert np.array_equal(out[b0s_mask], np.zeros((2, 3)))
    np.testing.assert_allclose(np.linalg.norm(out[~b0s_mask], axis=1), 1.0, atol=1e-12)


def test_volume_count_mismatch_raises():
    with pytest.raises(ValueError, match="covers 3 volumes"):
        rotate_bvecs_for_motion(
            BVALS, BVECS, affine_stack(np.eye(3), 3), B0S_MASK, np.eye(4)
        )


def test_b0_threshold_disagreement_raises():
    """A threshold that reclassifies volumes must not silently misalign."""
    bvals = np.array([0.0, 30.0, 1000.0, 1000.0])
    bvecs = BVECS.copy()
    bvecs[1] = [0.0, 0.0, 1.0]
    b0s_mask = np.array([True, False, False, False])  # b=30 treated as DWI
    with pytest.raises(ValueError, match="does not reproduce"):
        rotate_bvecs_for_motion(
            bvals, bvecs, affine_stack(np.eye(3), 4), b0s_mask,
            np.eye(4), b0_threshold=50,
        )


def test_all_b0_series_is_returned_unchanged():
    bvals = np.zeros(3)
    bvecs = np.zeros((3, 3))
    mask = np.ones(3, bool)
    out = rotate_bvecs_for_motion(
        bvals, bvecs, affine_stack(rotz(10.0), 3), mask, np.eye(4)
    )
    np.testing.assert_array_equal(out, bvecs)


def test_rotation_part_and_max_angle():
    assert max_rotation_angle_deg(affine_stack(np.eye(3), 3)) == pytest.approx(0.0)
    stack = affine_stack(np.eye(3), 3)
    stack[:3, :3, 1] = rotz(12.0)
    assert max_rotation_angle_deg(stack) == pytest.approx(12.0, abs=1e-6)
    # scale and shear are discarded
    np.testing.assert_allclose(
        rotation_part(rotz(20.0) @ np.diag([2.5, 2.2, 1.0])), rotz(20.0), atol=1e-10
    )


# ---------------------------------------------------------------------------
# T3.6 — DECISION GATE: end-to-end planted-motion recovery
# ---------------------------------------------------------------------------

def _phantom(shape):
    """Asymmetric high-contrast volume: registration needs features to lock onto."""
    x, y, z = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    cx, cy, cz = (np.array(shape) - 1) / 2
    v = np.zeros(shape, float)
    v[((x - cx) / 11.0) ** 2 + ((y - cy) / 8.0) ** 2 + ((z - cz) / 8.0) ** 2 < 1] = 100.0
    v[((x - cx - 5) / 3.0) ** 2 + ((y - cy - 3) / 3.0) ** 2 + ((z - cz) / 3.0) ** 2 < 1] = 220.0
    v[((x - cx + 6) / 2.0) ** 2 + ((y - cy) / 2.0) ** 2 + ((z - cz - 4) / 2.0) ** 2 < 1] = 40.0
    return v


def test_planted_motion_is_recovered_end_to_end():
    """Plant a known rotation, correct it, and check the b-vector followed.

    Validates the whole chain at once: registration recovers approximately the
    planted motion, and the rotation is applied to the gradient in the right
    direction and the right frame. A sign error shows up as a ~2*theta miss.
    """
    from dipy.align.imaffine import AffineMap
    from csttool.preprocess.modules.perform_motion_correction import (
        perform_motion_correction,
    )

    shape = (32, 32, 24)
    image_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    image_affine[:3, 3] = -np.array(shape) * 1.0

    ref = _phantom(shape)
    theta = 10.0
    centre = (image_affine @ np.array([*((np.array(shape) - 1) / 2), 1.0]))[:3]
    T, Tinv = np.eye(4), np.eye(4)
    T[:3, 3], Tinv[:3, 3] = centre, -centre
    R = np.eye(4)
    R[:3, :3] = rotz(theta)
    motion = T @ R @ Tinv  # forward head motion, world coordinates

    moved = AffineMap(
        np.linalg.inv(motion),
        domain_grid_shape=shape, domain_grid2world=image_affine,
        codomain_grid_shape=shape, codomain_grid2world=image_affine,
    ).transform(ref)

    # Volume 2 (a DWI) is the one that moved.
    data = np.stack([ref, ref, moved, ref], axis=-1)
    gtab = gradient_table(BVALS, bvecs=BVECS, b0_threshold=50)

    _, reg_affines = perform_motion_correction(data, gtab, image_affine)

    rotated = rotate_bvecs_for_motion(
        gtab.bvals, gtab.bvecs, reg_affines, gtab.b0s_mask, image_affine
    )

    def angle_between(a, b):
        return np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))

    expected = rotz(-theta) @ BVECS[2]
    assert angle_between(rotated[2], expected) < 5.0, (
        f"moved volume: got {rotated[2]}, expected ~{expected}"
    )
    # The wrong-direction answer sits ~2*theta away and must be excluded.
    assert angle_between(rotated[2], rotz(theta) @ BVECS[2]) > 10.0
    # Stationary volumes barely move.
    for i in (1, 3):
        assert angle_between(rotated[i], BVECS[i]) < 5.0
    assert np.array_equal(rotated[0], np.zeros(3))


# ---------------------------------------------------------------------------
# T3.7 / T3.8 / T3.9 — orchestrator behaviour
# ---------------------------------------------------------------------------

def _write_dataset(directory, stem, n_dwi=6):
    directory.mkdir(parents=True, exist_ok=True)
    shape = (16, 16, 12)
    rng = np.random.default_rng(1)
    ref = _phantom(shape)
    data = np.stack([ref + rng.normal(0, 2, shape) for _ in range(n_dwi + 1)], axis=-1)

    bvals = np.array([0.0] + [1000.0] * n_dwi)
    dirs = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], float
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros(3), dirs[:n_dwi]])

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), directory / f"{stem}.nii.gz")
    np.savetxt(directory / f"{stem}.bval", bvals[None, :], fmt="%g")
    np.savetxt(directory / f"{stem}.bvec", bvecs.T, fmt="%.8f")
    return directory


def test_motion_corrected_run_writes_rotated_bvecs(tmp_path):
    from csttool.cli.commands.preprocess import cmd_preprocess
    from csttool.preprocess.modules.gradient_validation import validate_gradient_table
    from dipy.io import read_bvals_bvecs
    import json

    in_dir = _write_dataset(tmp_path / "in", "sub")
    out = tmp_path / "out"
    args = argparse.Namespace(
        nifti=in_dir / "sub.nii.gz", dicom=None, out=out, coil_count=4,
        denoise_method="mppca", target_voxel_size=None,
        perform_motion_correction=True, unring=False,
    )
    result = cmd_preprocess(args)
    assert result["motion_correction"] is True

    stem = "sub_dwi_preproc_mc"
    bvals_out, bvecs_out = read_bvals_bvecs(
        str(out / f"{stem}.bval"), str(out / f"{stem}.bvec")
    )
    bvals_in, bvecs_in = read_bvals_bvecs(
        str(in_dir / "sub.bval"), str(in_dir / "sub.bvec")
    )

    # b-values are untouched; b-vectors moved; the pair still validates.
    np.testing.assert_array_equal(bvals_out, bvals_in)
    assert not np.allclose(bvecs_out, bvecs_in, atol=1e-9)
    validate_gradient_table(bvals_out, bvecs_out)
    np.testing.assert_allclose(np.linalg.norm(bvecs_out[1:], axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(bvecs_out[0], np.zeros(3))

    report = json.loads((out / f"{stem}_report.json").read_text())
    params = report["processing_params"]
    assert params["motion_correction_requested"] is True
    assert params["motion_correction"] is True
    assert params["bvecs_rotated"] is True
    assert params["max_rotation_deg"] is not None

    # The rotated sidecar is what tractography will actually load.
    from csttool.cli.utils import get_gtab_for_preproc

    gtab = get_gtab_for_preproc(out / f"{stem}.nii.gz")
    np.testing.assert_allclose(gtab.bvecs, bvecs_out, atol=1e-7)
    assert not np.allclose(gtab.bvecs, bvecs_in, atol=1e-9)


def test_run_without_motion_correction_copies_gradients_byte_identically(tmp_path):
    from csttool.cli.commands.preprocess import cmd_preprocess

    in_dir = _write_dataset(tmp_path / "in", "sub", n_dwi=6)
    out = tmp_path / "out"
    args = argparse.Namespace(
        nifti=in_dir / "sub.nii.gz", dicom=None, out=out, coil_count=4,
        denoise_method="mppca", target_voxel_size=None,
        perform_motion_correction=False, unring=False,
    )
    cmd_preprocess(args)

    stem = "sub_dwi_preproc_nomc"
    assert (out / f"{stem}.bvec").read_bytes() == (in_dir / "sub.bvec").read_bytes()
    assert (out / f"{stem}.bval").read_bytes() == (in_dir / "sub.bval").read_bytes()


def test_failed_motion_correction_is_recorded_and_leaves_gradients_consistent(
    tmp_path, monkeypatch
):
    """A failed MC keeps the *uncorrected* data with its *original* gradients."""
    import json
    from csttool.cli.commands.preprocess import cmd_preprocess
    import csttool.preprocess.preprocess as preproc_mod

    def boom(*args, **kwargs):
        raise RuntimeError("registration blew up")

    monkeypatch.setattr(preproc_mod, "perform_motion_correction", boom)

    in_dir = _write_dataset(tmp_path / "in", "sub")
    out = tmp_path / "out"
    args = argparse.Namespace(
        nifti=in_dir / "sub.nii.gz", dicom=None, out=out, coil_count=4,
        denoise_method="mppca", target_voxel_size=None,
        perform_motion_correction=True, unring=False,
    )
    result = cmd_preprocess(args)
    assert result["motion_correction"] is False

    stem = "sub_dwi_preproc_nomc"
    assert (out / f"{stem}.bvec").read_bytes() == (in_dir / "sub.bvec").read_bytes()

    params = json.loads((out / f"{stem}_report.json").read_text())["processing_params"]
    assert params["motion_correction_requested"] is True
    assert params["motion_correction"] is False
    assert params["bvecs_rotated"] is False
    assert any("motion correction" in w.lower() for w in params["warnings"])
