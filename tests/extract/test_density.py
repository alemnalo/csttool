"""Tests for ``csttool.extract.modules.density`` - CST density data product.

Implements the visualization-refactoring-plan §13.2 density test matrix D-1..D-9.
The numerical primitive is DIPY's public ``density_map``; these tests assert
csttool's denominator, grid guarantee, step-size guard and error message, and
lock the unique-visit semantics that DIPY provides (§5.2).
"""

import numpy as np
import pytest

from csttool.extract.modules.density import (
    compute_cst_density,
    save_cst_density,
)


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

    def test_the_denominator_is_bilateral_not_side_normalized(self):
        """The regression guard for the metric's meaning.

        A voxel every retained *left* streamline visits, and no right one,
        reads ``n_left / (n_left + n_right)`` — it cannot reach 1.0 while the
        right bundle is non-empty. Side-normalized density (``n_L(v) / N_L``)
        would put that voxel at 1.0, which is a different quantity with a
        different interpretation; switching to it is a scientific decision, not
        a visualization tweak, so it must never happen silently.

        This is also the ceiling the colourbar label exists to explain: on the
        validation subject n_left=749 of n_total=1756, so a fully occupied left
        voxel tops out near 43%.
        """
        # Three identical left streamlines through one voxel, two right ones
        # elsewhere: the left voxel is visited by 3 of 5 retained streamlines.
        left = [_sl([0.1, 0.1, 0.1], [0.4, 0.4, 0.4]) for _ in range(3)]
        right = [_sl([2.1, 2.1, 2.1], [2.4, 2.4, 2.4]) for _ in range(2)]
        density, meta = compute_cst_density(left, right, AFFINE, SHAPE)

        assert meta["n_total"] == meta["n_left"] + meta["n_right"] == 5
        assert density[0, 0, 0] == pytest.approx(3.0 / 5.0)
        assert density[0, 0, 0] < 1.0, "a unilateral voxel must not reach 1.0"
        assert density[2, 2, 2] == pytest.approx(2.0 / 5.0)
        # Emptying the other side is what would make it 1.0 — proving the
        # denominator really is shared rather than per-hemisphere.
        solo, solo_meta = compute_cst_density(left, [], AFFINE, SHAPE)
        assert solo_meta["n_total"] == 3
        assert solo[0, 0, 0] == pytest.approx(1.0)

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


class TestDensitySidecar:
    """What ``save_cst_density`` records beside the volume.

    The sidecar is how a reader recovers what the fractions were divided by and
    how dense the densest voxel actually got, without re-deriving anything.
    """

    def _write(self, tmp_path, left, right):
        import json

        density, meta = compute_cst_density(left, right, AFFINE, SHAPE)
        nii = save_cst_density(density, meta, AFFINE, tmp_path, "sub-x")
        sidecar = json.loads(nii.with_name(
            nii.name.replace(".nii.gz", ".json")).read_text())
        return density, meta, sidecar

    def test_records_the_true_maximum_fraction(self, tmp_path):
        """It was computed and printed but never persisted, so the one number
        saying how dense the densest voxel got could not be recovered from the
        derivatives at all."""
        left = [_sl([0.1, 0.1, 0.1], [0.4, 0.4, 0.4]) for _ in range(3)]
        right = [_sl([2.1, 2.1, 2.1], [2.4, 2.4, 2.4])]
        density, meta, sidecar = self._write(tmp_path, left, right)

        assert sidecar["MaxFraction"] == pytest.approx(float(density.max()))
        assert sidecar["MaxFraction"] == pytest.approx(meta["max_fraction"])
        assert 0.0 <= sidecar["MaxFraction"] <= 1.0
        # Three of four retained streamlines share one voxel.
        assert sidecar["MaxFraction"] == pytest.approx(3.0 / 4.0)

    def test_the_true_maximum_is_not_the_report_display_cap(self, tmp_path):
        """The distinction this field exists to preserve. The QC strip saturates
        its colour scale at the 99th percentile of non-zero voxels and records
        that separately as ``DensityDisplayVmax``; on a real subject the true
        maximum is roughly twice it. The extraction sidecar carries the data's
        maximum and must never carry a display decision."""
        left = [_sl([0.1, 0.1, 0.1], [0.4, 0.4, 0.4]) for _ in range(3)]
        right = [_sl([2.1, 2.1, 2.1], [2.4, 2.4, 2.4])]
        _, _, sidecar = self._write(tmp_path, left, right)

        for display_key in ("DensityDisplayVmax", "DensityDisplayPercentile",
                            "Vmax", "P99"):
            assert display_key not in sidecar

    def test_the_existing_schema_is_unchanged(self, tmp_path):
        """Purely additive: every key a previous reader relied on is still
        there, with the same meaning."""
        left = [_sl([0.1, 0.1, 0.1], [0.4, 0.4, 0.4]),
                _sl([0.2, 0.2, 0.2], [0.4, 0.4, 0.4])]
        right = [_sl([2.1, 2.1, 2.1], [2.4, 2.4, 2.4])]
        _, meta, sidecar = self._write(tmp_path, left, right)

        assert sidecar["Denominator"] == meta["n_total"] == 3
        assert sidecar["StreamlineCountLeft"] == 2
        assert sidecar["StreamlineCountRight"] == 1
        assert sidecar["Units"] == "dimensionless fraction"
        assert sidecar["Densified"] is False
        assert "density_map" in sidecar["Numerator"]
        assert sidecar["Description"] == meta["definition"]

    def test_an_empty_bundle_records_a_zero_maximum(self, tmp_path):
        _, _, sidecar = self._write(tmp_path, [], [])
        assert sidecar["MaxFraction"] == 0.0
        assert sidecar["Denominator"] == 0
