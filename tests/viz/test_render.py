"""Tests for ``csttool.viz.render`` - composable panel renderers (M4, §13.2 R-1..R-5).

Every renderer draws into the Axes it is given and creates no new Figure (R-5).
These tests assert artist properties and decoded RGBA, never just "a file exists".
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from csttool.viz import geometry as geo
from csttool.viz import render
from csttool.viz import style


RAS = np.diag([2.0, 2.0, 2.0, 1.0])
LAS = np.array([[-2.0, 0, 0, 90.0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]])


class TestNoFigureCreated:
    def test_r5_no_new_figure_from_any_renderer(self):
        before = plt.get_fignums()
        fig, ax = plt.subplots()
        vol = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
        render.render_scalar_slice(ax, vol, RAS, "coronal", 4)
        rgb = np.zeros((8, 8, 8, 3), dtype=np.float32); rgb[..., 2] = 0.5
        render.render_rgb_slice(ax, rgb, RAS, "coronal", 4)
        render.render_density_overlay(ax, vol, RAS, "coronal", 4)
        render.render_mask_contour(ax, (vol > 0.5).astype(np.uint8), RAS, "coronal", 4)
        render.add_direction_legend(ax)
        plt.close(fig)
        assert plt.get_fignums() == before


class TestRenderRgbSlice:
    def test_r1_rejects_out_of_range(self):
        fig, ax = plt.subplots()
        rgb = np.zeros((4, 4, 4, 3), dtype=np.float32)
        rgb[..., 0] = 1.5  # out of range
        with pytest.raises(ValueError):
            render.render_rgb_slice(ax, rgb, RAS, "coronal", 0)
        plt.close(fig)

    def test_r1_rejects_wrong_dtype(self):
        fig, ax = plt.subplots()
        rgb = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        with pytest.raises(TypeError):
            render.render_rgb_slice(ax, rgb, RAS, "coronal", 0)
        plt.close(fig)

    def test_r2_radiological_convention(self):
        # The established csttool convention (tests/viz/test_geometry.py):
        # RAS coronal gets its x-axis inverted (to radiological); LAS coronal is
        # *already* radiological so it is left un-inverted. Both add L/R markers.
        rgb = np.zeros((4, 4, 4, 3), dtype=np.float32)

        fig, ax = plt.subplots()
        render.render_rgb_slice(ax, rgb, RAS, "coronal", 0)
        assert ax.xaxis_inverted()
        assert {t.get_text() for t in ax.texts} >= {"R", "L"}
        plt.close(fig)

        fig, ax = plt.subplots()
        render.render_rgb_slice(ax, rgb, LAS, "coronal", 0)
        # LAS is already radiological: no inversion, but markers still added.
        assert not ax.xaxis_inverted()
        assert {t.get_text() for t in ax.texts} >= {"R", "L"}
        plt.close(fig)


class TestRenderDensityOverlay:
    def test_r3_zero_density_fully_transparent(self):
        fig, ax = plt.subplots()
        # A density slice that is all-zero must produce alpha 0 everywhere.
        density = np.zeros((8, 8, 8), dtype=np.float32)
        im = render.render_density_overlay(ax, density, RAS, "coronal", 4,
                                           vmax=1.0, alpha_floor=0.0)
        rgba = im.cmap(im.norm(im.get_array()), bytes=False)
        # masked array: alpha is 0 where masked
        assert np.all(rgba[..., 3] == 0)
        plt.close(fig)

    def test_vmax_passed_to_imshow(self):
        fig, ax = plt.subplots()
        density = np.zeros((8, 8, 8), dtype=np.float32); density[4, 4, 4] = 0.7
        im = render.render_density_overlay(ax, density, RAS, "coronal", 4, vmax=0.7)
        # The mappable's norm vmax must be the value passed in.
        assert im.norm.vmax == pytest.approx(0.7)
        plt.close(fig)


class TestRenderStreamlineOverlay:
    def test_r4_deterministic_subsample_reproducible(self):
        # Two runs with the same seeded rng must produce identical Line2D vertex
        # arrays (the only stochastic element in the render layer).
        affine = RAS
        rng = np.random.default_rng(0)
        sls = [np.column_stack([rng.uniform(-5, 5, 20), np.full(20, 1.0),
                                rng.uniform(-5, 5, 20)]) for _ in range(40)]

        def _draw():
            fig, ax = plt.subplots()
            from csttool.viz.utils import viz_rng
            render.render_streamline_overlay(
                ax, sls, affine, "coronal", 4, color="red",
                thickness_mm=10.0, max_streamlines=20, rng=viz_rng(123),
            )
            verts = []
            for line in ax.get_lines():
                verts.append(np.asarray(line.get_xydata()))
            plt.close(fig)
            return verts

        a = _draw(); b = _draw()
        assert len(a) == len(b)
        for va, vb in zip(a, b):
            assert np.array_equal(va, vb)

    def test_b4_split_runs_no_chord(self):
        # A streamline that leaves and re-enters the slab must yield two separate
        # Line2D objects, not one polyline with a chord across the gap.
        affine = RAS  # 2mm iso -> coronal slice 0 centred at world y=0, ±5mm slab.
        # 3 in-slab points near y=0, then a far point at world y=50 (out), then 2
        # more in-slab points.
        sl = np.array([[0, 1, 0], [2, 1, 0], [4, 1, 0],
                       [0, 50, 0], [3, -1, 0], [5, 1, 0]], dtype=float)
        fig, ax = plt.subplots()
        render.render_streamline_overlay(ax, [sl], affine, "coronal", 0,
                                         color="blue", thickness_mm=10.0,
                                         max_streamlines=10, rng=np.random.default_rng(0))
        lines = ax.get_lines()
        assert len(lines) >= 2  # at least two separate runs
        # No line should span the gap: none should contain a y near 50.
        for line in lines:
            y = np.asarray(line.get_ydata())
            assert not np.any(np.abs(y - 50) < 5)
        plt.close(fig)
