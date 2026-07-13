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
