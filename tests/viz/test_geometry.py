"""Tests for csttool.viz.geometry - orientation and spatial helpers.

These validate the *semantics* of the display convention (radiological: anatomical
Left drawn on the viewer's right) rather than exact pixels.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from csttool.viz import geometry as geo

# Two subject affines with OPPOSITE X handedness, both 2 mm iso, centred.
LAS_AFFINE = np.array([[-2, 0, 0, 90.0],   # voxel axis 0 -> anatomical Left  (X-sign -)
                       [0, 2, 0, -75.0],
                       [0, 0, 2, -70.0],
                       [0, 0, 0, 1.0]])
RAS_AFFINE = np.array([[2, 0, 0, -90.0],    # voxel axis 0 -> anatomical Right (X-sign +)
                       [0, 2, 0, -75.0],
                       [0, 0, 2, -70.0],
                       [0, 0, 0, 1.0]])


class TestVoxelWorldSign:
    def test_x_sign_matches_affine(self):
        assert geo.voxel_axis_world_x_sign(LAS_AFFINE, 0) == -1
        assert geo.voxel_axis_world_x_sign(RAS_AFFINE, 0) == +1

    def test_non_lr_axis_returns_zero(self):
        # voxel axis 1 (Y / anterior) and 2 (Z / superior) carry no L/R.
        assert geo.voxel_axis_world_x_sign(LAS_AFFINE, 1) == 0
        assert geo.voxel_axis_world_x_sign(LAS_AFFINE, 2) == 0

    def test_small_rotation_term_is_not_lr(self):
        # A realistic affine has small off-diagonal rotation terms; a tiny X
        # component on the anterior/superior axes must NOT be treated as L/R
        # (otherwise sagittal views wrongly get R/L markers).
        rot = np.array([[-1.997, 0.0, 0.106, 90.0],
                        [0.0, 2.0, -0.006, -75.0],
                        [0.106, 0.006, 1.997, -70.0],
                        [0.0, 0.0, 0.0, 1.0]])
        assert geo.voxel_axis_world_x_sign(rot, 0) == -1  # X-dominant axis
        assert geo.voxel_axis_world_x_sign(rot, 1) == 0   # Y axis, tiny X term
        assert geo.voxel_axis_world_x_sign(rot, 2) == 0   # Z axis, small X term


class TestRadiologicalImage:
    @pytest.mark.parametrize("view", ["axial", "coronal"])
    def test_negative_x_affine_needs_no_flip(self, view):
        # LAS: increasing horizontal voxel index already moves toward anat-Left,
        # so anat-Left is already on the viewer's right -> no inversion.
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((10, 10)))
        has_lr = geo.enforce_radiological_image(ax, LAS_AFFINE, view)
        assert has_lr is True
        assert not ax.xaxis_inverted()
        plt.close(fig)

    @pytest.mark.parametrize("view", ["axial", "coronal"])
    def test_positive_x_affine_is_flipped(self, view):
        # RAS: default display would put anat-Right on the right (neurological);
        # radiological enforcement must invert the x-axis.
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((10, 10)))
        has_lr = geo.enforce_radiological_image(ax, RAS_AFFINE, view)
        assert has_lr is True
        assert ax.xaxis_inverted()
        plt.close(fig)

    def test_sagittal_has_no_lr(self):
        fig, ax = plt.subplots()
        assert geo.enforce_radiological_image(ax, LAS_AFFINE, "sagittal") is False
        assert geo.enforce_radiological_image(ax, RAS_AFFINE, "sagittal") is False
        plt.close(fig)


class TestWorldXRadiological:
    def test_world_x_inverted_to_radiological(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-50, 50)  # ascending: Right(+X) on the right = neurological
        geo.set_world_x_radiological(ax)
        lo, hi = ax.get_xlim()
        assert lo > hi  # descending: anat-Left (-X) now on the viewer's right
        plt.close(fig)

    def test_already_radiological_is_left_untouched(self):
        fig, ax = plt.subplots()
        ax.set_xlim(50, -50)  # already descending
        geo.set_world_x_radiological(ax)
        lo, hi = ax.get_xlim()
        assert (lo, hi) == (50, -50)
        plt.close(fig)


class TestLRMarkers:
    def test_markers_are_R_left_and_L_right(self):
        # After radiological orientation, "R" sits at the left edge, "L" at the right.
        fig, ax = plt.subplots()
        geo.add_lr_markers(ax)
        texts = {t.get_text(): t.get_position()[0] for t in ax.texts}
        assert set(texts) == {"R", "L"}
        assert texts["R"] < texts["L"]  # R nearer the left edge
        plt.close(fig)


class TestSpatialHelpers:
    def test_pad_slice_to_square(self):
        sl = np.ones((6, 10))
        padded, extent = geo.pad_slice_to_square(sl)
        assert padded.shape == (10, 10)
        assert padded.sum() == sl.sum()  # padding adds zeros only

    def test_volume_world_bounds(self):
        bounds = geo.volume_world_bounds((10, 10, 10), LAS_AFFINE)
        assert len(bounds) == 3
        for lo, hi in bounds:
            assert hi > lo

    def test_voxel_world_roundtrip(self):
        pts = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
        world = geo.voxel_to_world(pts, LAS_AFFINE)
        back = geo.world_to_voxel(world, LAS_AFFINE)
        np.testing.assert_allclose(back, pts, atol=1e-9)


class TestOrientationCode:
    """orientation_code returns the 3-letter voxel orientation from the affine.

    This is the compact, scientifically exact orientation label shown once in the
    report's Methods band. It must be derived from the affine, never hardcoded,
    so a subject reoriented to RAS during preprocessing is reported as RAS.
    """

    def test_ras_affine(self):
        assert geo.orientation_code(RAS_AFFINE) == "RAS"

    def test_las_affine(self):
        assert geo.orientation_code(LAS_AFFINE) == "LAS"

    def test_lps_affine(self):
        # Negative X (Left), negative Y (Posterior), positive Z (Superior).
        lps = np.array([[-2, 0, 0, 90.0],
                        [0, -2, 0, -75.0],
                        [0, 0, 2, -70.0],
                        [0, 0, 0, 1.0]])
        assert geo.orientation_code(lps) == "LPS"

    def test_three_letters(self):
        code = geo.orientation_code(RAS_AFFINE)
        assert len(code) == 3


# ---------------------------------------------------------------------------
# select_qc_slice + slab_membership (visualization-refactor M4: §6.4, §6.5)
# ---------------------------------------------------------------------------
from csttool.viz.geometry import select_qc_slice, slab_membership, DEFAULT_SLAB_MM

ISO2 = np.diag([2.0, 2.0, 2.0, 1.0])
ANISO = np.diag([2.0, 2.0, 6.0, 1.0])


class TestSelectQcSlice:
    def test_s1_known_peak_returns_correct_index(self):
        # 10x10x10 volume with density only in coronal plane j=4.
        d = np.zeros((10, 10, 10))
        d[:, 4, :] = 0.5  # every voxel in the y=4 coronal plane is "visited"
        idx, prov = select_qc_slice(density=d, brain_mask=np.ones((10, 10, 10)),
                                    affine=ISO2, view="coronal")
        assert idx == 4
        assert prov["rule"] == "bilateral_occupancy"

    def test_s2_deterministic(self):
        d = np.zeros((10, 10, 10))
        d[:, 3, :] = 0.5; d[:, 7, :] = 0.5
        bm = np.ones((10, 10, 10))
        r1 = select_qc_slice(density=d, brain_mask=bm, affine=ISO2, view="coronal")
        r2 = select_qc_slice(density=d, brain_mask=bm, affine=ISO2, view="coronal")
        assert r1 == r2

    def test_s3_left_only_surviving_hemisphere(self):
        dL = np.zeros((10, 10, 10)); dL[:, 5, :] = 0.4
        dR = np.zeros((10, 10, 10))
        idx, prov = select_qc_slice(density_left=dL, density_right=dR,
                                    brain_mask=np.ones((10, 10, 10)),
                                    affine=ISO2, view="coronal")
        assert prov["rule"] == "surviving_hemisphere"
        assert idx == 5

    def test_s4_both_empty_rois_present(self):
        roi = np.zeros((10, 10, 10), dtype=bool); roi[:, 6, :] = True
        idx, prov = select_qc_slice(roi_masks=[roi], brain_mask=np.ones((10, 10, 10)),
                                    affine=ISO2, view="coronal")
        assert prov["rule"] == "roi_occupancy"
        assert idx == 6

    def test_s5_everything_empty_anatomical_centroid_never_raises(self):
        bm = np.zeros((10, 10, 10)); bm[:, :, 3:7] = 1
        idx, prov = select_qc_slice(brain_mask=bm, affine=ISO2, view="coronal")
        assert prov["rule"] == "anatomical_centroid"
        assert 0 <= idx < 10

    def test_s6_tie_break_nearest_centroid(self):
        # Two planes with equal occupancy; centroid between them picks the nearer.
        d = np.zeros((20, 20, 20))
        # Mass weighted to the right so centroid > 10, and tie at planes 8 and 12.
        d[:, 8, :] = 0.1
        d[:, 12, :] = 0.9  # heavier -> pulls centroid toward 12, so 12 wins
        idx, prov = select_qc_slice(density=d, brain_mask=np.ones((20, 20, 20)),
                                    affine=ISO2, view="coronal")
        assert idx == 12
        assert "tie" not in prov["tie_break"].lower() or "centroid" in prov["tie_break"].lower()

    def test_rejects_bad_view(self):
        with pytest.raises(ValueError):
            select_qc_slice(brain_mask=np.ones((4, 4, 4)), affine=ISO2, view="bogus")


class TestSlabMembership:
    def test_b1_thickness_boundary_2mm_iso(self):
        # Centre of coronal slice 0 in this affine is at world y=0; plane normal y.
        # thickness 10 -> membership within +/-5 mm.
        pts = np.array([[0.0, 4.9, 0.0], [0.0, 5.1, 0.0],
                        [0.0, -4.9, 0.0], [0.0, -5.1, 0.0]])
        m = slab_membership(pts, ISO2, "coronal", 0, 10.0)
        assert m[0] and not m[1]
        assert m[2] and not m[3]

    def test_b2_anisotropic_voxels_same_physical_membership(self):
        # Same physical world points must give the same membership under 2x2x6 as
        # under 2x2x2 (the regression test for the voxel->mm change).
        pts = np.array([[0.0, 4.9, 0.0], [0.0, 5.1, 0.0],
                        [0.0, -4.9, 0.0], [0.0, -5.1, 0.0]])
        m_iso = slab_membership(pts, ISO2, "coronal", 0, 10.0)
        m_aniso = slab_membership(pts, ANISO, "coronal", 0, 10.0)
        assert np.array_equal(m_iso, m_aniso)

    def test_b3_oblique_normal_follows_world_plane(self):
        # Rotate the affine 30 degrees about z; the coronal depth axis (voxel y)
        # now has a world direction of [-sin30, cos30, 0] = [-0.5, 0.866, 0].
        # Membership must follow that true normal, not the array axis.
        t = np.deg2rad(30.0)
        c, s = np.cos(t), np.sin(t)
        aff = np.eye(4)
        aff[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) * 2.0
        # A point straight along the world-Y axis is mostly aligned with the
        # rotated depth normal, so inside the 10mm slab.
        pts = np.array([[0.0, 4.0, 0.0]])
        m = slab_membership(pts, aff, "coronal", 0, 10.0)
        assert m[0]
        # A point along world-X projects onto the rotated normal with magnitude
        # |4 * (-0.5)| = 2.0 mm, still inside the ±5mm slab -> in. A point far
        # enough along world-X to be outside: |x * (-0.5)| > 5 -> |x| > 10.
        pts2 = np.array([[12.0, 0.0, 0.0]])
        m2 = slab_membership(pts2, aff, "coronal", 0, 10.0)
        assert not m2[0]

    def test_b4_contiguity(self):
        # Two runs in the slab, separated by an out-of-slab point, must not be
        # joined by render_streamline_overlay (asserted in test_render.py). Here
        # only the membership split is checked: 3 points in, 1 out, 2 in.
        pts = np.array([[0, 1, 0], [0, 2, 0], [0, 50, 0], [0, 3, 0], [0, 4, 0]],
                       dtype=float)
        m = slab_membership(pts, ISO2, "coronal", 0, 10.0)
        assert m.tolist() == [True, True, False, True, True]
