"""Tests for ``csttool.extract.modules.density`` - CST density data product.

Implements the visualization-refactoring-plan §13.2 density test matrix D-1..D-9.
The numerical primitive is DIPY's public ``density_map``; these tests assert
csttool's denominator, grid guarantee, step-size guard and error message, and
lock the unique-visit semantics that DIPY provides (§5.2).
"""

import numpy as np
import pytest

from csttool.extract.modules.density import compute_cst_density


# 4x4x4 grid, 1 mm isotropic, RAS.
AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])
SHAPE = (4, 4, 4)


def _sl(*pts):
    """Build an (N,3) streamline from integer voxel coords (world==voxel here)."""
    return np.asarray(pts, dtype=float)


class TestDensitySemantics:
    def test_d1_two_streamlines_known_fractions(self):
        # One streamline through voxel (0,0,0) only; one through (1,1,1) only.
        left = [_sl([0.0, 0.0, 0.0], [0.1, 0.1, 0.1])]
        right = [_sl([1.0, 1.0, 1.0], [1.1, 1.1, 1.1])]
        density, meta = compute_cst_density(left, right, AFFINE, SHAPE)
        assert meta["n_total"] == 2
        # Each visited voxel gets 1/2.
        assert np.isclose(density[0, 0, 0], 0.5)
        assert np.isclose(density[1, 1, 1], 0.5)
        assert density.sum() == pytest.approx(1.0)

    def test_d2_repeated_points_in_one_voxel_count_once(self):
        # 10 sampled points all inside voxel (0,0,0).
        pts = np.linspace([0.1, 0.1, 0.1], [0.4, 0.4, 0.4], 10)
        left = [pts]
        right = []
        density, meta = compute_cst_density(left, right, AFFINE, SHAPE)
        assert density[0, 0, 0] == 1.0  # the single streamline visits once
        assert meta["n_total"] == 1

    def test_d3_reentry_counted_once(self):
        # Streamline leaves voxel 0 and comes back to it.
        left = [_sl([0.1, 0.1, 0.1], [2.1, 2.1, 2.1], [0.4, 0.4, 0.4])]
        right = []
        density, meta = compute_cst_density(left, right, AFFINE, SHAPE)
        assert density[0, 0, 0] == 1.0
        assert density[2, 2, 2] == 1.0

    def test_d4_range_zero_to_one(self):
        left = [np.linspace([0.1, 0.1, 0.1], [3.4, 3.4, 3.4], 12)]
        right = [np.linspace([0.2, 0.2, 0.2], [3.3, 3.3, 3.3], 9)]
        density, meta = compute_cst_density(left, right, AFFINE, SHAPE)
        assert density.min() >= 0.0
        assert density.max() <= 1.0 + 1e-6

    def test_d5_denominator_and_integer_counts(self):
        left = [np.linspace([0.1, 0.1, 0.1], [3.4, 3.4, 3.4], 12)]
        right = [np.linspace([0.2, 0.2, 0.2], [3.3, 3.3, 3.3], 9),
                 np.linspace([1.1, 1.1, 1.1], [2.9, 2.9, 2.9], 7)]
        density, meta = compute_cst_density(left, right, AFFINE, SHAPE)
        assert meta["n_total"] == 3
        # density.max() * n_total must be an integer (raw count).
        assert float(np.round(density.max() * meta["n_total"], 6)).is_integer()

    def test_d6_doubling_right_changes_right_fraction(self):
        base_left = [_sl([0.1, 0.1, 0.1], [0.4, 0.4, 0.4])]
        one_right = [_sl([2.1, 2.1, 2.1], [2.4, 2.4, 2.4])]
        two_right = [_sl([2.1, 2.1, 2.1], [2.4, 2.4, 2.4]),
                     _sl([2.1, 2.1, 2.1], [2.4, 2.4, 2.4])]
        d1, _ = compute_cst_density(base_left, one_right, AFFINE, SHAPE)
        d2, _ = compute_cst_density(base_left, two_right, AFFINE, SHAPE)
        # Left raw count (voxel 0,0,0) is unchanged: 1/2 -> 1/3.
        assert d1[0, 0, 0] == pytest.approx(0.5)
        assert d2[0, 0, 0] == pytest.approx(1.0 / 3.0)
        # Right raw count doubles: 1/2 -> 2/3.
        assert d1[2, 2, 2] == pytest.approx(0.5)
        assert d2[2, 2, 2] == pytest.approx(2.0 / 3.0)

    def test_d7_empty_bundle_all_zero(self):
        density, meta = compute_cst_density([], [], AFFINE, SHAPE)
        assert meta["n_total"] == 0
        assert density.shape == SHAPE
        assert not np.any(density)
        # No ZeroDivisionError (the whole point).

    def test_d8_gap_guard_densifies(self):
        # A 3 mm step on a 1 mm grid: step_size > min_voxel/2 = 0.5 -> densify.
        left = [_sl([0.0, 0.0, 0.0], [3.0, 0.0, 0.0])]  # skips voxels 1,2
        right = []
        # Without the guard, voxels (1,0,0) and (2,0,0) would be zero.
        density_no, meta_no = compute_cst_density(left, right, AFFINE, SHAPE)
        assert meta_no["densified"] is False
        # With the guard on a 1mm grid (step 3 > 0.5), densification fires and
        # the intermediate voxels are filled.
        density_yes, meta_yes = compute_cst_density(
            left, right, AFFINE, SHAPE, step_size_mm=3.0)
        assert meta_yes["densified"] is True
        assert density_no[1, 0, 0] == 0.0 and density_no[2, 0, 0] == 0.0
        assert density_yes[1, 0, 0] > 0.0 and density_yes[2, 0, 0] > 0.0
        # Total fraction (1 streamline) is preserved: every voxel the streamline
        # passes counts 1/1.
        assert density_yes[0, 0, 0] == 1.0

    def test_d9_grid_mismatch_raises_csttool_valueerror(self):
        # Streamline point at voxel/world coordinate 9 on a 4-voxel grid.
        left = [_sl([9.0, 0.0, 0.0], [9.4, 0.4, 0.4])]
        right = []
        with pytest.raises(ValueError) as excinfo:
            compute_cst_density(left, right, AFFINE, SHAPE)
        msg = str(excinfo.value)
        # The csttool message must name the grids, not the bare DIPY index error.
        assert "target" in msg.lower() or "grid" in msg.lower()


class TestDensityMeta:
    def test_definition_is_exact_section_sentence(self):
        _, meta = compute_cst_density([], [], AFFINE, SHAPE)
        assert meta["definition"] == (
            "The fraction of distinct retained bilateral CST streamlines that "
            "visit each voxel at least once."
        )

    def test_n_left_right_split(self):
        left = [_sl([0.1, 0.1, 0.1], [0.4, 0.4, 0.4]),
                _sl([0.2, 0.2, 0.2], [0.4, 0.4, 0.4])]
        right = [_sl([2.1, 2.1, 2.1], [2.4, 2.4, 2.4])]
        _, meta = compute_cst_density(left, right, AFFINE, SHAPE)
        assert meta["n_left"] == 2
        assert meta["n_right"] == 1
        assert meta["n_total"] == 3
