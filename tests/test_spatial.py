"""Tests for ``csttool.spatial`` - affine / frame mathematics.

Covers the six affine cases of the visualization-refactoring-plan §13.2 (T-1
.. T-6) plus norm preservation (T-7) and zero-vector handling (T-8). These tests
are the scientific guard against the two most likely silent errors of the
refactor: applying the full affine (which rescales by voxel size, faking
anisotropy of direction) and "correcting" reflections (which would corrupt the
stored V1 field).
"""

import warnings

import numpy as np
import pytest

from csttool.spatial import (
    affine_rotation,
    affine_shear_magnitude,
    affine_obliquity_rad,
    rotate_vector_field_to_world,
)


# Affines used across the matrix. Each is a 4x4 voxel->RASMM map.
RAS_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
LAS_AFFINE = np.array([
    [-2.0, 0.0, 0.0, 90.0],
    [0.0, 2.0, 0.0, -75.0],
    [0.0, 0.0, 2.0, -70.0],
    [0.0, 0.0, 0.0, 1.0],
])
LPS_AFFINE = np.array([
    [-2.0, 0.0, 0.0, 90.0],
    [0.0, -2.0, 0.0, 75.0],
    [0.0, 0.0, 2.0, -70.0],
    [0.0, 0.0, 0.0, 1.0],
])
ANISO_AFFINE = np.diag([2.0, 2.0, 6.0, 1.0])


def _rot_z_affine(theta_deg, voxel=2.0):
    """An oblique affine: voxel axes rotated by ``theta`` about world Z."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    lin = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) * voxel
    aff = np.eye(4)
    aff[:3, :3] = lin
    return aff


def _sheared_affine():
    """An affine whose voxel axes are not mutually orthogonal in world space."""
    aff = np.eye(4)
    # Column 0 has a large component along the axis-1 direction -> shear.
    aff[:3, :3] = np.array([[2.0, 1.2, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 6.0]])
    return aff


def _is_orthonormal(R, tol=1e-10):
    return np.allclose(R @ R.T, np.eye(3), atol=tol) and np.allclose(R.T @ R, np.eye(3), atol=tol)


class TestAffineRotation:
    def test_t1_ras_identity(self):
        R, diag = affine_rotation(RAS_AFFINE)
        assert np.allclose(R, np.eye(3))
        assert abs(diag["det"] - 1.0) < 1e-12
        assert _is_orthonormal(R)
        # Canonical basis vectors map to themselves.
        for ax in range(3):
            v = np.zeros(3); v[ax] = 1.0
            assert np.allclose(R @ v, v)

    def test_t2_las_reflection(self):
        R, diag = affine_rotation(LAS_AFFINE)
        assert np.allclose(R, np.diag([-1.0, 1.0, 1.0]))
        assert diag["det"] < 0.0
        # |R @ e_x| == [1, 0, 0]: a reflection cannot change a DEC magnitude.
        assert np.allclose(np.abs(R @ np.array([1.0, 0, 0])), [1, 0, 0])
        assert _is_orthonormal(R)

    def test_t3_lps_det_plus_one(self):
        R, diag = affine_rotation(LPS_AFFINE)
        assert abs(diag["det"] - 1.0) < 1e-12
        # All three DEC magnitudes are unchanged under a pure rotation.
        for ax in range(3):
            v = np.zeros(3); v[ax] = 1.0
            assert np.allclose(np.abs(R @ v), [1, 1, 1] if False else np.abs(v))
        assert _is_orthonormal(R)

    def test_t4_anisotropic_is_identity(self):
        # The test that catches the "just apply the affine" error: applying M
        # to e_x would give [2,0,0] (rescaled), and to e_z would give [0,0,6].
        # R must be identity so unit directions stay unit.
        R, diag = affine_rotation(ANISO_AFFINE)
        assert np.allclose(R, np.eye(3))
        assert np.allclose(R @ np.array([1.0, 0, 0]), [1, 0, 0])
        assert np.allclose(R @ np.array([0.0, 0, 1]), [0, 0, 1])

    def test_t5_oblique_mixes_components(self):
        aff = _rot_z_affine(30.0)
        R, diag = affine_rotation(aff)
        assert _is_orthonormal(R)
        # A voxel-frame e_x maps to [cos30, sin30, 0] in world axes.
        ex = np.array([1.0, 0.0, 0.0])
        out = R @ ex
        assert np.isclose(out[0], np.cos(np.deg2rad(30.0)), atol=1e-10)
        assert np.isclose(out[1], np.sin(np.deg2rad(30.0)), atol=1e-10)
        assert np.isclose(out[2], 0.0, atol=1e-10)
        assert max(diag["obliquity_rad"]) > 0.0

    def test_t6_shear_warns_but_returns_orthonormal(self):
        aff = _sheared_affine()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            R, diag = affine_rotation(aff)
        assert _is_orthonormal(R, tol=1e-10)
        assert diag["shear_magnitude"] > 0.0
        # A UserWarning must be emitted (shear > 1e-6).
        assert any(issubclass(w.category, UserWarning) for w in caught)
        # The shear is large enough to set the flag (> 1e-2 default tol).
        assert diag["shear_warning"] is True

    def test_rejects_non_4x4(self):
        with pytest.raises(ValueError):
            affine_rotation(np.eye(3))

    def test_records_determinant_sign(self):
        _, diag = affine_rotation(RAS_AFFINE)
        assert diag["det"] == pytest.approx(1.0)
        _, diag = affine_rotation(LAS_AFFINE)
        assert diag["det"] == pytest.approx(-1.0)


class TestObliquityAndShear:
    def test_obliquity_zero_for_axis_aligned(self):
        # LAS and anisotropic affines are axis-aligned -> zero obliquity despite
        # sign and voxel size.
        for aff in (RAS_AFFINE, LAS_AFFINE, ANISO_AFFINE):
            assert all(o == 0.0 for o in affine_obliquity_rad(aff))

    def test_obliquity_nonzero_for_oblique(self):
        aff = _rot_z_affine(20.0)
        assert max(affine_obliquity_rad(aff)) > 0.0

    def test_shear_magnitude_zero_for_diagonal(self):
        assert affine_shear_magnitude(ANISO_AFFINE) == 0.0

    def test_shear_magnitude_positive_for_sheared(self):
        assert affine_shear_magnitude(_sheared_affine()) > 0.0


class TestRotateVectorField:
    def test_t7_norm_preservation(self):
        rng = np.random.default_rng(0)
        # Build a proper unit field of shape (N, 3).
        flat = rng.normal(size=(50, 3))
        flat /= np.linalg.norm(flat, axis=1, keepdims=True)
        for aff in (RAS_AFFINE, LAS_AFFINE, LPS_AFFINE, ANISO_AFFINE, _rot_z_affine(30.0)):
            world, _ = rotate_vector_field_to_world(flat, aff, normalize=False)
            norms = np.linalg.norm(world, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6)

    def test_t8_zero_vectors_stay_zero(self):
        field = np.zeros((4, 3))
        # A handful of zero rows mixed with unit rows.
        field[1] = [1.0, 0.0, 0.0]
        world, _ = rotate_vector_field_to_world(field, RAS_AFFINE)
        assert np.allclose(world[0], 0.0)
        assert np.allclose(world[2], 0.0)
        assert np.allclose(world[3], 0.0)
        assert not np.any(np.isnan(world))

    def test_canonical_basis_under_las(self):
        # e_x (voxel axis 0) is anatomical Left in LAS: world X = -1.
        field = np.eye(3)
        world, _ = rotate_vector_field_to_world(field, LAS_AFFINE, normalize=False)
        assert np.allclose(world[0], [-1.0, 0.0, 0.0])
        assert np.allclose(world[1], [0.0, 1.0, 0.0])
        assert np.allclose(world[2], [0.0, 0.0, 1.0])

    def test_output_dtype_float32(self):
        field = np.eye(3, dtype=np.float64)
        world, _ = rotate_vector_field_to_world(field, RAS_AFFINE)
        assert world.dtype == np.float32

    def test_normalize_default_returns_unit(self):
        field = np.tile([1.0, 0.0, 0.0], (3, 1))
        world, _ = rotate_vector_field_to_world(field, _rot_z_affine(45.0))
        norms = np.linalg.norm(world, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            rotate_vector_field_to_world(np.zeros((4, 2)), RAS_AFFINE)
