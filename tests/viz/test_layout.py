"""Tests for ``csttool.viz.layout`` - physical (millimetre) figure layout.

These assert the two properties the module exists to guarantee:

* a figure asked for N millimetres **is** N millimetres, so a point of type in
  the source is a point of type on paper; and
* the generalized panel-row arithmetic is the *same* arithmetic the report QC
  strip has been reviewed at, so promoting it out of the report path did not
  quietly re-lay-out the report.

The cropping helpers are asserted against the invariant that matters
scientifically: cropping is a presentation operation and must never mirror a
panel that :func:`csttool.viz.geometry.enforce_radiological_image` has flipped.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from csttool.viz import geometry as geo
from csttool.viz import layout
from csttool.viz import render


RAS = np.diag([2.0, 2.0, 2.0, 1.0])
LAS = np.array([[-2.0, 0, 0, 90.0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]])


class TestPhysicalSizing:
    def test_figure_mm_is_exactly_that_many_mm(self):
        fig = layout.figure_mm(194.0, 94.0)
        w_mm, h_mm = np.array(fig.get_size_inches()) * layout.MM_PER_IN
        assert w_mm == pytest.approx(194.0)
        assert h_mm == pytest.approx(94.0)
        plt.close(fig)

    def test_figure_mm_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            layout.figure_mm(0.0, 10.0)
        with pytest.raises(ValueError):
            layout.figure_mm(10.0, -1.0)

    def test_axes_mm_lands_where_asked(self):
        fig = layout.figure_mm(100.0, 50.0)
        ax = layout.axes_mm(fig, 10.0, 5.0, 40.0, 20.0)
        x0, y0, w, h = ax.get_position().bounds
        assert x0 == pytest.approx(0.10)
        assert y0 == pytest.approx(0.10)
        assert w == pytest.approx(0.40)
        assert h == pytest.approx(0.40)
        plt.close(fig)

    def test_mm_inch_roundtrip(self):
        assert layout.in_to_mm(layout.mm_to_in(37.5)) == pytest.approx(37.5)


class TestTypeScale:
    def test_report_scale_orders_title_above_context(self):
        t = layout.REPORT_TYPE
        assert t.title > t.label > t.tick
        assert t.title > t.key, "a key must never out-shout a panel title"

    def test_scaled_multiplies_every_field(self):
        half = layout.REPORT_TYPE.scaled(0.5)
        assert half.title == pytest.approx(layout.REPORT_TYPE.title * 0.5)
        assert half.caption == pytest.approx(layout.REPORT_TYPE.caption * 0.5)

    def test_type_scale_is_frozen(self):
        with pytest.raises(Exception):
            layout.REPORT_TYPE.title = 12.0


class TestPanelRowGeometry:
    """The generalized arithmetic. The report's own contract is asserted in
    ``tests/metrics/test_report_qc_strip.py``; here we check it generalizes."""

    def test_panels_tile_the_width_without_overlap(self):
        for n in (1, 2, 3, 4, 6):
            geom = layout.panel_row_geometry(
                96, 60, width_mm=160.0, n_panels=n, image_max_mm=40.0)
            xs = geom["panel_x0_mm"]
            assert len(xs) == n
            w = geom["panel_w_mm"]
            for a, b in zip(xs, xs[1:]):
                assert b - a == pytest.approx(w + layout.REPORT_STRIP_BANDS.gap_mm)
            # The row is centred and fits.
            assert xs[0] >= -1e-9
            assert xs[-1] + w <= 160.0 + 1e-9

    def test_width_is_the_callers_not_the_modules(self):
        """No document dimension is baked in: the width out is the width in."""
        for width in (120.0, 160.0, 194.0):
            geom = layout.panel_row_geometry(
                96, 60, width_mm=width, n_panels=4, image_max_mm=25.5)
            assert geom["width_mm"] == pytest.approx(width)

    def test_wide_grid_is_width_bound(self):
        geom = layout.panel_row_geometry(
            160, 80, width_mm=160.0, n_panels=4, image_max_mm=100.0)
        assert geom["image_h_mm"] == pytest.approx(geom["panel_w_mm"] / 2.0)
        assert geom["box_aspect"] == pytest.approx(geom["aspect"])

    def test_tall_grid_letterboxes_rather_than_narrowing(self):
        wide = layout.panel_row_geometry(
            160, 80, width_mm=160.0, n_panels=4, image_max_mm=20.0)
        tall = layout.panel_row_geometry(
            80, 80, width_mm=160.0, n_panels=4, image_max_mm=20.0)
        assert tall["panel_w_mm"] == pytest.approx(wide["panel_w_mm"])
        assert tall["image_h_mm"] == pytest.approx(20.0)
        assert tall["box_aspect"] > tall["aspect"]

    def test_height_is_furniture_plus_image(self):
        bands = layout.PanelRowBands()
        geom = layout.panel_row_geometry(
            96, 60, width_mm=160.0, n_panels=4, bands=bands, image_max_mm=100.0)
        assert geom["height_mm"] == pytest.approx(
            bands.furniture_mm + geom["image_h_mm"])

    def test_bands_stack_without_overlap(self):
        bands = layout.PanelRowBands()
        geom = layout.panel_row_geometry(
            96, 60, width_mm=160.0, n_panels=4, bands=bands, image_max_mm=25.5)
        assert geom["caption_y0_mm"] < geom["key_y0_mm"] < geom["image_y0_mm"]
        assert (geom["key_y0_mm"] - geom["caption_y0_mm"]
                >= bands.caption_mm)
        assert (geom["image_y0_mm"] - geom["key_y0_mm"]) >= bands.key_mm
        assert geom["image_y0_mm"] + geom["image_h_mm"] <= geom["height_mm"] + 1e-9

    def test_height_floor_splits_slack_between_margins(self):
        geom = layout.panel_row_geometry(
            400, 60, width_mm=160.0, n_panels=4, image_max_mm=25.5,
            min_height_mm=60.0)
        assert geom["height_mm"] == pytest.approx(60.0)
        # Content is centred, so the bottom margin grew.
        assert geom["caption_y0_mm"] > layout.PanelRowBands().margin_bottom_mm

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            layout.panel_row_geometry(96, 60, width_mm=160.0, n_panels=0,
                                      image_max_mm=25.5)
        with pytest.raises(ValueError):
            layout.panel_row_geometry(0, 60, width_mm=160.0, n_panels=4,
                                      image_max_mm=25.5)


class TestContentBbox:
    def test_finds_the_content_and_pads_it(self):
        sl = np.zeros((20, 20))
        sl[5:15, 4:16] = 1.0
        box = layout.content_bbox_2d(sl, margin_frac=0.0)
        col_lo, col_hi, row_lo, row_hi = box
        assert (col_lo, col_hi) == pytest.approx((3.5, 15.5))
        assert (row_lo, row_hi) == pytest.approx((4.5, 14.5))

    def test_margin_scales_with_the_structure(self):
        sl = np.zeros((20, 20))
        sl[5:15, 5:15] = 1.0
        tight = layout.content_bbox_2d(sl, margin_frac=0.0)
        loose = layout.content_bbox_2d(sl, margin_frac=0.10)
        assert loose[0] < tight[0] and loose[1] > tight[1]

    def test_empty_slice_returns_none(self):
        assert layout.content_bbox_2d(np.zeros((8, 8))) is None

    def test_handles_rgb(self):
        sl = np.zeros((10, 10, 3))
        sl[2:5, 3:7, 1] = 0.8
        box = layout.content_bbox_2d(sl, margin_frac=0.0)
        assert box is not None
        assert box[0] == pytest.approx(2.5)

    def test_union_bbox_covers_all(self):
        a = (0.0, 4.0, 0.0, 4.0)
        b = (2.0, 9.0, -1.0, 3.0)
        assert layout.union_bbox([a, b, None]) == (0.0, 9.0, -1.0, 4.0)
        assert layout.union_bbox([None]) is None

    def test_square_bbox_preserves_centre(self):
        box = layout.square_bbox((0.0, 10.0, 0.0, 4.0))
        assert box[1] - box[0] == pytest.approx(box[3] - box[2])
        assert (box[0] + box[1]) / 2 == pytest.approx(5.0)
        assert (box[2] + box[3]) / 2 == pytest.approx(2.0)


class TestApplyBboxPreservesOrientation:
    """Cropping is presentation-only and must not undo the radiological flip."""

    def test_inverted_axis_stays_inverted(self):
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[3:7, 3:7, 3:7] = 1.0
        fig, ax = plt.subplots()
        # LAS gives a coronal view that must be x-inverted to read radiologically.
        render.render_scalar_slice(ax, vol, LAS, "coronal", 5)
        was_inverted = ax.xaxis_inverted()
        layout.apply_bbox(ax, (2.0, 8.0, 2.0, 8.0))
        assert ax.xaxis_inverted() == was_inverted
        lo, hi = sorted(ax.get_xlim())
        assert (lo, hi) == pytest.approx((2.0, 8.0))
        plt.close(fig)

    def test_none_is_a_noop(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 3); ax.set_ylim(0, 7)
        layout.apply_bbox(ax, None)
        assert ax.get_xlim() == (0, 3)
        assert ax.get_ylim() == (0, 7)
        plt.close(fig)

    def test_crop_axes_to_content_shrinks_the_view(self):
        vol = np.zeros((40, 40, 40), dtype=np.float32)
        vol[15:25, 15:25, 15:25] = 1.0
        fig, ax = plt.subplots()
        render.render_scalar_slice(ax, vol, RAS, "axial", 20)
        full = abs(np.diff(ax.get_xlim())[0])
        box = layout.crop_axes_to_content(ax, vol, "axial", 20, margin_frac=0.0)
        assert box is not None
        assert abs(np.diff(ax.get_xlim())[0]) < full
        plt.close(fig)

    def test_crop_on_empty_slice_leaves_axes_alone(self):
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        fig, ax = plt.subplots()
        render.render_scalar_slice(ax, vol, RAS, "axial", 5)
        before = (ax.get_xlim(), ax.get_ylim())
        assert layout.crop_axes_to_content(ax, vol, "axial", 5) is None
        assert (ax.get_xlim(), ax.get_ylim()) == before
        plt.close(fig)


class TestKeyRow:
    def test_draw_key_row_emits_one_swatch_and_label_per_entry(self):
        fig = layout.figure_mm(60.0, 10.0)
        ax = layout.axes_mm(fig, 2.0, 2.0, 56.0, 6.0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        entries = [("#1f77b4", "left"), ("#ff7f0e", "right"), (None, "no swatch")]
        layout.draw_key_row(ax, renderer, entries, 6.5, y=0.3)
        assert len(ax.lines) == 2, "an entry with colour=None draws no swatch"
        assert [t.get_text() for t in ax.texts] == ["left", "right", "no swatch"]
        plt.close(fig)

    def test_empty_entries_draw_nothing(self):
        fig = layout.figure_mm(60.0, 10.0)
        ax = layout.axes_mm(fig, 2.0, 2.0, 56.0, 6.0)
        fig.canvas.draw()
        layout.draw_key_row(ax, fig.canvas.get_renderer(), [], 6.5, y=0.3)
        assert not ax.lines and not ax.texts
        plt.close(fig)

    def test_fit_key_fontsize_shrinks_only_when_needed(self):
        fig = layout.figure_mm(60.0, 10.0)
        ax = layout.axes_mm(fig, 2.0, 2.0, 56.0, 6.0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        short = [[("#1f77b4", "L")]]
        assert layout.fit_key_fontsize(ax, renderer, short, 6.5) == pytest.approx(6.5)
        long = [[("#1f77b4", "an extremely long legend label" * 4)]]
        fitted = layout.fit_key_fontsize(ax, renderer, long, 6.5)
        assert layout.REPORT_KEY.min_pt <= fitted < 6.5
        plt.close(fig)

    def test_fit_key_fontsize_never_returns_below_floor(self):
        fig = layout.figure_mm(20.0, 10.0)
        ax = layout.axes_mm(fig, 1.0, 1.0, 18.0, 6.0)
        fig.canvas.draw()
        rows = [[("#1f77b4", "x" * 400)]]
        fitted = layout.fit_key_fontsize(ax, fig.canvas.get_renderer(), rows, 6.5)
        assert fitted == pytest.approx(layout.REPORT_KEY.min_pt)
        plt.close(fig)


class TestNoFigureLeak:
    def test_layout_helpers_create_no_stray_figures(self):
        before = plt.get_fignums()
        layout.panel_row_geometry(96, 60, width_mm=160.0, n_panels=3,
                                  image_max_mm=25.5)
        layout.content_bbox_2d(np.ones((4, 4)))
        layout.union_bbox([(0.0, 1.0, 0.0, 1.0)])
        assert plt.get_fignums() == before


class TestContentBboxClamping:
    """The box must not extend past the slice it was measured on.

    An image axis has its frame off and paints no patch, so any view beyond the
    data shows bare canvas - and anything positioned in axes fractions lands on
    it. The R/L markers are drawn white with a dark outline so they read against
    anatomy; on white paper they read as floating outside the panel.
    """

    def _full_width_slice(self):
        sl = np.zeros((20, 30))
        sl[:, :] = 1.0            # content reaches every edge
        return sl

    def test_clamped_box_stays_inside_the_slice(self):
        sl = self._full_width_slice()
        x0, x1, y0, y1 = layout.content_bbox_2d(sl, margin_frac=0.25)
        assert x0 >= -0.5 and x1 <= 30 - 0.5
        assert y0 >= -0.5 and y1 <= 20 - 0.5

    def test_unclamped_box_may_exceed_it(self):
        sl = self._full_width_slice()
        box = layout.content_bbox_2d(sl, margin_frac=0.25, clamp=False)
        assert box[0] < -0.5 and box[1] > 30 - 0.5

    def test_clamping_does_not_shrink_an_interior_box(self):
        sl = np.zeros((40, 40))
        sl[10:30, 10:30] = 1.0     # well inside, margin has room
        clamped = layout.content_bbox_2d(sl, margin_frac=0.10)
        loose = layout.content_bbox_2d(sl, margin_frac=0.10, clamp=False)
        assert clamped == pytest.approx(loose)

    def test_clamped_box_still_covers_all_content(self):
        sl = np.zeros((20, 30))
        sl[0, 0] = sl[19, 29] = 1.0
        x0, x1, y0, y1 = layout.content_bbox_2d(sl, margin_frac=0.5)
        assert x0 <= -0.5 + 1e-9 and x1 >= 29 - 0.5
        assert y0 <= -0.5 + 1e-9 and y1 >= 19 - 0.5
