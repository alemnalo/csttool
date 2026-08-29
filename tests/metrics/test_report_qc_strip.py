"""Tests for the 1x4 report QC strip.

The strip replaces a 1x3 triptych that showed the same information from three
angles. Its whole claim is that four panels answer four *different* questions on
**one** slice, so these tests assert the properties that claim rests on: four
slots always present, one shared slice index, one legend, one colourbar, equal
physical viewports, the documented colours, and byte-reproducibility.

Three defects the strip must fix rather than inherit are covered explicitly:
the shared slice (which never actually held outside a prototype driver), the
world-vs-voxel frame of the streamline overlay, and the ``viz_rng`` seeding that
made the legacy panel irreproducible across processes.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import nibabel as nib
import numpy as np
import pytest

from csttool.metrics.modules.visualizations import (
    format_density_percent,
    QC_STRIP_MAX_HEIGHT_MM,
    QC_STRIP_MIN_HEIGHT_MM,
    QC_STRIP_WIDTH_MM,
    plot_report_qc_strip,
    qc_strip_geometry,
)
from csttool.viz import style as _style

SHAPE = (24, 20, 18)
RAS_AFFINE = np.array([[2.0, 0, 0, -24.0],
                       [0, 2.0, 0, -20.0],
                       [0, 0, 2.0, -18.0],
                       [0, 0, 0, 1.0]])
LAS_AFFINE = np.array([[-2.0, 0, 0, 24.0],
                       [0, 2.0, 0, -20.0],
                       [0, 0, 2.0, -18.0],
                       [0, 0, 0, 1.0]])


# ---------------------------------------------------------------------------
# Synthetic subject: a brain-ish FA blob, a superior-inferior V1 field, a
# density map, an ROI segmentation and two bundles.
# ---------------------------------------------------------------------------

def _fa():
    xx, yy, zz = np.indices(SHAPE)
    fa = np.exp(-(((xx - 12) / 8.0) ** 2 + ((yy - 10) / 7.0) ** 2
                  + ((zz - 9) / 7.0) ** 2))
    return fa.clip(0, 1).astype(np.float32)


def _v1_superior():
    """A field pointing along world +Z everywhere: DEC must come out blue."""
    v1 = np.zeros(SHAPE + (3,), dtype=np.float32)
    v1[..., 2] = 1.0
    return v1


def _density():
    density = np.zeros(SHAPE, dtype=np.float32)
    density[8:11, 8:13, 4:14] = 0.4
    density[14:17, 8:13, 4:14] = 0.3
    return density


def _long_tailed_density():
    """A density whose true maximum sits well above its non-zero P99.

    The flat two-valued ``_density`` fixture cannot saturate — its P99 *is* its
    maximum — so it cannot exercise the clipped branch. Real subjects have a
    long tail: on the validation subject the maximum is 1.9x the P99 and 1% of
    visited voxels sit at or above it.
    """
    density = np.zeros(SHAPE, dtype=np.float32)
    rng = np.random.default_rng(0)
    body = (slice(6, 18), slice(6, 14), slice(3, 15))
    density[body] = rng.uniform(0.01, 0.10, size=density[body].shape)
    density[11:13, 9:11, 7:9] = 0.9   # the tail: a few very dense core voxels
    return density


def _dseg():
    dseg = np.zeros(SHAPE, dtype=np.uint8)
    dseg[10:15, 8:13, 2:5] = 1     # brainstem, low
    dseg[7:10, 8:13, 13:16] = 2    # motor left, high
    dseg[15:18, 8:13, 13:16] = 3   # motor right, high
    return dseg


def _bundle(x_world, n=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = np.linspace(-14.0, 12.0, 20)
        out.append(np.column_stack([
            np.full(20, x_world + rng.normal(0, 0.4)),
            np.full(20, 0.0 + rng.normal(0, 0.4)),
            z,
        ]))
    return out


def _write_nifti(path, data, affine):
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _write_trk(path, streamlines, affine):
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram

    reference = nib.Nifti1Image(np.zeros(SHAPE, np.float32), affine)
    sft = StatefulTractogram(streamlines, reference, Space.RASMM)
    save_tractogram(sft, str(path), bbox_valid_check=False)
    return path


@pytest.fixture
def subject(tmp_path):
    """Every strip input, written to disk as the real products are."""
    affine = RAS_AFFINE
    paths = {
        "fa_path": _write_nifti(tmp_path / "fa.nii.gz", _fa(), affine),
        "v1_path": _write_nifti(tmp_path / "v1.nii.gz", _v1_superior(), affine),
        "density_path": _write_nifti(tmp_path / "density.nii.gz", _density(), affine),
        "roi_dseg_path": _write_nifti(tmp_path / "dseg.nii.gz", _dseg(), affine),
        "cst_left_path": _write_trk(tmp_path / "left.trk", _bundle(-8.0, seed=1), affine),
        "cst_right_path": _write_trk(tmp_path / "right.trk", _bundle(8.0, seed=2), affine),
    }
    return paths


def _render(subject, tmp_path, subject_id="sub-strip", **overrides):
    kwargs = dict(subject)
    kwargs.update(overrides)
    return plot_report_qc_strip(
        output_dir=tmp_path, subject_id=subject_id, **kwargs
    )


def _sidecar(path):
    return json.loads(path.with_suffix(".json").read_text())


@pytest.fixture
def strip_probe(monkeypatch):
    """Capture the figure's geometry and artists at save time."""
    import csttool.metrics.modules.visualizations as viz

    captured = {}
    real_save = viz._style.save_figure

    def probe(fig, path, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        w_mm, h_mm = np.array(fig.get_size_inches()) * 25.4

        def mm(value):
            return value / fig.dpi * 25.4

        def bbox_mm(artist):
            try:
                box = artist.get_window_extent(renderer)
            except Exception:
                return None
            if box.width <= 0 or box.height <= 0:
                return None
            return (mm(box.x0), mm(box.y0), mm(box.x1), mm(box.y1))

        captured["size_in"] = tuple(fig.get_size_inches())
        captured["size_mm"] = (w_mm, h_mm)
        captured["axes"] = [
            {
                "position": ax.get_position(original=False),
                "box_mm": (mm(ax.get_window_extent(renderer).x0),
                           mm(ax.get_window_extent(renderer).y0),
                           mm(ax.get_window_extent(renderer).x1),
                           mm(ax.get_window_extent(renderer).y1)),
                "title": ax.get_title(),
                "title_pt": ax.title.get_fontsize(),
                "title_bbox_mm": bbox_mm(ax.title) if ax.get_title() else None,
                "images": list(ax.images),
                "lines": list(ax.lines),
                "line_colors": [ln.get_color() for ln in ax.lines],
                "collections": list(ax.collections),
                "texts": [t.get_text() for t in ax.texts],
                "text_colors": [t.get_color() for t in ax.texts],
                "text_pts": [t.get_fontsize() for t in ax.texts],
                "text_bboxes_mm": [bbox_mm(t) for t in ax.texts],
                # The anchor point, which for va='baseline' *is* the baseline —
                # unlike a bbox edge, it does not move with glyph content.
                "text_anchor_y_mm": [
                    mm(t.get_transform().transform(t.get_position())[1])
                    for t in ax.texts
                ],
                "text_vas": [t.get_va() for t in ax.texts],
                "line_bboxes_mm": [bbox_mm(ln) for ln in ax.lines],
                "xlabel": ax.get_xlabel(),
                "xlim": ax.get_xlim(),
                "ylim": ax.get_ylim(),
            }
            for ax in fig.axes
        ]
        captured["n_legends"] = len(fig.legends) + sum(
            1 for ax in fig.axes if ax.get_legend() is not None
        )
        return real_save(fig, path, **kwargs)

    monkeypatch.setattr(viz._style, "save_figure", probe)
    return captured


def _panels(probe):
    """The four image panels, left to right."""
    panels = [ax for ax in probe["axes"] if ax["images"] and ax["title"]]
    return sorted(panels, key=lambda ax: ax["position"].x0)


def _key_band(probe):
    """The four per-panel key columns, left to right.

    A key axes carries no image and no title, and spans one panel column rather
    than the whole strip (which is what distinguishes it from the caption).
    """
    width_mm = probe["size_mm"][0]
    keys = [ax for ax in probe["axes"]
            if not ax["images"] and not ax["title"] and not ax["collections"]
            and (ax["box_mm"][2] - ax["box_mm"][0]) < width_mm * 0.9]
    return sorted(keys, key=lambda ax: ax["box_mm"][0])


def _colourbar_axes(probe):
    """Panel 2's colourbar box, in mm. It is the one axes carrying a QuadMesh."""
    bars = [ax for ax in probe["axes"] if ax["collections"] and not ax["images"]]
    assert len(bars) == 1, f"expected one colourbar, found {len(bars)}"
    return bars[0]["box_mm"]


def _caption_axes(probe):
    """The one full-width caption row."""
    width_mm = probe["size_mm"][0]
    full = [ax for ax in probe["axes"]
            if not ax["images"] and not ax["title"]
            and (ax["box_mm"][2] - ax["box_mm"][0]) >= width_mm * 0.9]
    assert len(full) == 1, f"expected one caption row, found {len(full)}"
    return full[0]


def _caption_text(probe):
    return " ".join(_caption_axes(probe)["texts"])


def _furniture(probe):
    """Every annotation outside the images: titles, keys, caption.

    The R/L orientation glyphs are excluded — they are deliberately inside the
    image, one pair per panel, and are covered by their own test.
    """
    items = []
    for ax in probe["axes"]:
        if ax["title"] and ax["title_bbox_mm"]:
            items.append((ax["title"], ax["title_bbox_mm"]))
        for text, box in zip(ax["texts"], ax["text_bboxes_mm"]):
            if box and text.strip() not in ("R", "L"):
                items.append((text, box))
    return items


# ---------------------------------------------------------------------------

class TestFourSlots:
    def test_all_four_panels_present_and_titled(self, subject, tmp_path, strip_probe):
        _render(subject, tmp_path)
        panels = _panels(strip_probe)
        assert len(panels) == 4
        assert panels[0]["title"] == "DEC-FA"
        assert panels[1]["title"] == "CST density"
        assert panels[2]["title"] == "Extraction ROIs"
        assert panels[3]["title"] == "CST over FA"

    @pytest.mark.parametrize(
        "missing, title, note",
        [
            ("v1_path", "DEC-FA", "world-frame V1 not produced"),
            ("density_path", "CST density", "density map not produced"),
            ("roi_dseg_path", "Extraction ROIs", "ROI segmentation not produced"),
        ],
    )
    def test_a_missing_product_degrades_its_own_slot_only(
        self, subject, tmp_path, strip_probe, missing, title, note
    ):
        """The strip never reflows to three: a reader must be able to see which
        question went unanswered.

        The *title* names the panel and never reports status — a heading row
        that changes length from subject to subject cannot hold a consistent
        visual hierarchy. The reason goes in the in-panel note instead.
        """
        _render(subject, tmp_path, **{missing: None})
        panels = _panels(strip_probe)
        assert len(panels) == 4
        titles = [p["title"] for p in panels]
        assert title in titles
        assert not any("unavailable" in t for t in titles)
        assert any(note in t for p in panels for t in p["texts"])

    @pytest.mark.parametrize("empty", ["cst_left_path", "cst_right_path", "both"])
    def test_empty_hemispheres_still_render_four_panels(
        self, subject, tmp_path, strip_probe, empty
    ):
        overrides = ({"cst_left_path": None, "cst_right_path": None}
                     if empty == "both" else {empty: None})
        path = _render(subject, tmp_path, **overrides)
        assert len(_panels(strip_probe)) == 4
        assert path.exists()

    def test_every_input_missing_still_renders_four_panels(
        self, subject, tmp_path, strip_probe
    ):
        """Only FA is required; everything else degrades in place."""
        _render(subject, tmp_path, v1_path=None, density_path=None,
                roi_dseg_path=None, cst_left_path=None, cst_right_path=None)
        assert len(_panels(strip_probe)) == 4

    def test_degraded_panels_are_named_in_the_sidecar(self, subject, tmp_path):
        path = _render(subject, tmp_path, v1_path=None, density_path=None)
        degraded = _sidecar(path)["DegradedPanels"]
        assert "dec_fa" in degraded and "density" in degraded
        assert "roi_dseg" not in degraded


class TestSharedSlice:
    def test_one_slice_index_is_recorded(self, subject, tmp_path):
        sidecar = _sidecar(_render(subject, tmp_path))
        assert isinstance(sidecar["SliceIndex"], int)
        assert sidecar["SliceSelectionRule"] == "bilateral_occupancy"

    def test_every_panel_shows_the_same_plane(self, subject, tmp_path, strip_probe):
        """The defect this fixes: the standalone panels select independently, and
        two of them are called with ``density=None``, so they land on the level-4
        fallback while a third gets level 1. Here one selection feeds all four,
        which is observable as four identical background slices."""
        path = _render(subject, tmp_path)
        index = _sidecar(path)["SliceIndex"]
        expected = _fa()[:, index, :].T
        panels = _panels(strip_probe)
        # Panels 2-4 share the grayscale FA background; panel 1 is the DEC RGB
        # of the same plane, so its luminance follows the same slice.
        for panel in panels[1:]:
            np.testing.assert_allclose(panels[1]["images"][0].get_array(), expected,
                                       rtol=0, atol=0)
            assert panel["images"][0].get_array().shape == expected.shape

    def test_density_drives_the_slice_when_available(self, subject, tmp_path):
        with_density = _sidecar(_render(subject, tmp_path, subject_id="a"))
        without = _sidecar(_render(subject, tmp_path, subject_id="b",
                                   density_path=None))
        assert with_density["SliceSelectionRule"] == "bilateral_occupancy"
        assert without["SliceSelectionRule"] in ("roi_occupancy",
                                                 "anatomical_centroid")

    def test_roi_dseg_serves_the_level_three_fallback(self, subject, tmp_path):
        """Both bundles empty and no density: the ROI segmentation is what keeps
        the strip on a plane the extraction's own targets occupy."""
        sidecar = _sidecar(_render(subject, tmp_path, density_path=None,
                                   cst_left_path=None, cst_right_path=None))
        assert sidecar["SliceSelectionRule"] == "roi_occupancy"

    def test_slice_selection_never_fails(self, subject, tmp_path):
        sidecar = _sidecar(_render(subject, tmp_path, density_path=None,
                                   roi_dseg_path=None, cst_left_path=None,
                                   cst_right_path=None))
        assert sidecar["SliceSelectionRule"] == "anatomical_centroid"


class TestGeometry:
    def test_generated_at_its_designed_print_size(self, subject, tmp_path, strip_probe):
        """Width is fixed by the CSS; height is derived from the slice aspect,
        so it is asserted against the geometry function rather than a constant."""
        path = _render(subject, tmp_path)
        canvas_h, canvas_w = _fa()[:, _sidecar(path)["SliceIndex"], :].T.shape
        geom = qc_strip_geometry(canvas_w, canvas_h)
        w_mm, h_mm = QC_STRIP_WIDTH_MM, geom["height_mm"]
        np.testing.assert_allclose(
            strip_probe["size_in"], (w_mm / 25.4, h_mm / 25.4), rtol=1e-9
        )
        assert QC_STRIP_MIN_HEIGHT_MM <= h_mm <= QC_STRIP_MAX_HEIGHT_MM
        arr = np.asarray(mpimg.imread(str(path)))
        np.testing.assert_allclose(arr.shape[0] / arr.shape[1], h_mm / w_mm, rtol=2e-3)
        assert _sidecar(path)["FigureSizeMm"] == [round(w_mm, 3), round(h_mm, 3)]

    def test_four_panels_share_one_physical_viewport(self, subject, tmp_path,
                                                     strip_probe):
        """1 mm of brain is 1 mm of paper in every panel, so the four are
        directly comparable by eye."""
        _render(subject, tmp_path)
        panels = _panels(strip_probe)
        widths = {round(p["position"].width, 6) for p in panels}
        heights = {round(p["position"].height, 6) for p in panels}
        assert len(widths) == 1 and len(heights) == 1
        for panel in panels:
            assert panel["xlim"] == panels[0]["xlim"]
            assert panel["ylim"] == panels[0]["ylim"]

    def test_panels_are_not_stretched(self, subject, tmp_path, strip_probe):
        _render(subject, tmp_path)
        for panel in _panels(strip_probe):
            for image in panel["images"]:
                assert image.axes.get_aspect() == 1.0

    def test_panels_do_not_overlap(self, subject, tmp_path, strip_probe):
        _render(subject, tmp_path)
        panels = _panels(strip_probe)
        for left, right in zip(panels, panels[1:]):
            assert left["position"].x1 <= right["position"].x0 + 1e-9

    def test_lr_markers_appear_once_per_panel(self, subject, tmp_path, strip_probe):
        """Markers, not legends: a reader must not have to look at a neighbouring
        panel to orient the one they are reading. Exactly one pair each — the
        renderers used to stamp their own on top of the strip's."""
        _render(subject, tmp_path)
        for panel in _panels(strip_probe):
            assert panel["texts"].count("R") == 1
            assert panel["texts"].count("L") == 1


class TestPanelContent:
    def test_dec_panel_is_blue_dominant_for_a_superior_field(self, subject, tmp_path,
                                                             strip_probe):
        """The frame guard: a purely superior-inferior V1 must colour blue,
        because the stored field is in the world frame and B = S-I."""
        _render(subject, tmp_path)
        rgb = _panels(strip_probe)[0]["images"][0].get_array()
        rgb = np.asarray(rgb, dtype=float)
        bright = rgb.sum(axis=2) > 0.1
        assert bright.any()
        assert rgb[..., 2][bright].mean() > rgb[..., 0][bright].mean()
        assert rgb[..., 2][bright].mean() > rgb[..., 1][bright].mean()

    def test_dec_panel_is_blue_dominant_under_an_las_affine(self, tmp_path,
                                                            strip_probe):
        """And under a flipped in-plane affine, which is csttool's normal case:
        the colour encodes the *world* axis, not the voxel axis."""
        paths = {
            "fa_path": _write_nifti(tmp_path / "fa.nii.gz", _fa(), LAS_AFFINE),
            "v1_path": _write_nifti(tmp_path / "v1.nii.gz", _v1_superior(), LAS_AFFINE),
            "density_path": None, "roi_dseg_path": None,
            "cst_left_path": None, "cst_right_path": None,
        }
        plot_report_qc_strip(output_dir=tmp_path, subject_id="sub-las", **paths)
        rgb = np.asarray(_panels(strip_probe)[0]["images"][0].get_array(), dtype=float)
        bright = rgb.sum(axis=2) > 0.1
        assert rgb[..., 2][bright].mean() > rgb[..., 0][bright].mean()

    def test_the_sidecar_separates_the_true_maximum_from_the_display_cap(
        self, subject, tmp_path
    ):
        """Two different numbers. The display cap is a presentation choice; the
        maximum is a property of the data. They were previously one key named
        ``DensityVmax``, which reads like the second and held the first."""
        sidecar = _sidecar(_render(subject, tmp_path))
        assert 0 < sidecar["DensityDisplayVmax"] <= 1.0
        assert 0 < sidecar["DensityMaxFraction"] <= 1.0
        assert sidecar["DensityMaxFraction"] >= sidecar["DensityDisplayVmax"]
        assert sidecar["DensityDisplayPercentile"] == 99.0
        assert sidecar["DensityDisplayPercentileBasis"] == "non-zero voxels"
        assert isinstance(sidecar["DensityDisplayClipped"], bool)
        assert "DensityVmax" not in sidecar, "the misleading key must be gone"

    def test_the_colourbar_names_the_bilateral_denominator(self, subject,
                                                           tmp_path, strip_probe):
        """"streamline fraction" could equally have meant a fraction of one
        streamline, of one hemisphere's streamlines, or of every streamline
        generated. The denominator is the one thing a reader cannot guess, and
        it is what sets the ceiling: a voxel every left streamline visits reads
        n_left / (n_left + n_right), not 100%."""
        _render(subject, tmp_path)
        texts = [t for ax in strip_probe["axes"] for t in ax["texts"]]
        assert texts.count("Fraction of bilateral CST streamlines") == 1
        assert "streamline fraction" not in texts
        assert not any("vmax=" in t for t in texts)

    def test_the_colourbar_endpoints_are_percentages(self, subject, tmp_path,
                                                     strip_probe):
        """Display only — the stored volume stays a fraction in [0, 1]."""
        path = _render(subject, tmp_path)
        sidecar = _sidecar(path)
        texts = [t for ax in strip_probe["axes"] for t in ax["texts"]]
        assert "0%" in texts
        expected = format_density_percent(
            sidecar["DensityDisplayVmax"],
            saturated=sidecar["DensityDisplayClipped"],
        )
        assert expected in texts
        assert expected.endswith("%")
        # The raw fraction must not appear anywhere on the figure.
        assert f"{sidecar['DensityDisplayVmax']:.3g}" not in texts

    def test_a_clipped_endpoint_is_marked_as_a_floor_not_a_maximum(
        self, subject, tmp_path, strip_probe
    ):
        """A long-tailed density — like a real subject's, whose true maximum is
        ~1.9x its 99th percentile — puts voxels above the cap, so they share the
        top colour and a bare number would misname the maximum."""
        path = _render(subject, tmp_path,
                       density_path=_write_nifti(tmp_path / "tail.nii.gz",
                                                 _long_tailed_density(),
                                                 RAS_AFFINE))
        sidecar = _sidecar(path)
        assert sidecar["DensityDisplayClipped"] is True
        assert sidecar["DensityMaxFraction"] > sidecar["DensityDisplayVmax"]
        texts = [t for ax in strip_probe["axes"] for t in ax["texts"]]
        upper = [t for t in texts if t.startswith("\u2265")]
        assert len(upper) == 1, "the capped endpoint must carry a >= marker"
        assert format_density_percent(sidecar["DensityMaxFraction"]) not in texts, (
            "the true maximum must not be printed as the endpoint"
        )

    def test_an_unclipped_endpoint_is_stated_exactly(self, subject, tmp_path,
                                                     strip_probe):
        """The counterpart: where the cap happens to reach the true maximum,
        nothing saturates and no ">=" may be claimed."""
        path = _render(subject, tmp_path)
        sidecar = _sidecar(path)
        assert sidecar["DensityDisplayClipped"] is False, "fixture must not saturate"
        texts = [t for ax in strip_probe["axes"] for t in ax["texts"]]
        assert not any(t.startswith("\u2265") for t in texts)
        assert format_density_percent(sidecar["DensityDisplayVmax"]) in texts

    @pytest.mark.parametrize("long_tail", [False, True])
    def test_the_caption_states_the_clipping_rule_that_the_code_applies(
        self, subject, tmp_path, strip_probe, long_tail
    ):
        """The rule stated on the page must be the rule the code ran — a caption
        naming a percentile the visualization does not use is worse than none.

        It is stated whenever a density panel exists, saturating or not: the
        rule describes how the scale top was chosen, and a caption that explained
        the scale for some subjects and not others would be harder to read.
        """
        from csttool.metrics.modules.visualizations import (
            _DENSITY_DISPLAY_PERCENTILE,
        )

        density = _long_tailed_density() if long_tail else _density()
        path = _render(subject, tmp_path,
                       density_path=_write_nifti(tmp_path / "d.nii.gz", density,
                                                 RAS_AFFINE))
        assert "density scale capped at non-zero P99" in _caption_text(strip_probe)
        assert _DENSITY_DISPLAY_PERCENTILE == 99.0
        # And it is genuinely that percentile of the non-zero voxels.
        nonzero = density[density > 0]
        assert _sidecar(path)["DensityDisplayVmax"] == pytest.approx(
            float(np.percentile(nonzero, _DENSITY_DISPLAY_PERCENTILE))
        )

    def test_no_label_calls_the_display_cap_a_maximum(self, subject, tmp_path,
                                                      strip_probe):
        _render(subject, tmp_path)
        for ax in strip_probe["axes"]:
            for text in ax["texts"]:
                assert "max" not in text.lower(), f"{text!r} calls the cap a maximum"

    def test_no_colourbar_when_density_is_unavailable(self, subject, tmp_path,
                                                      strip_probe):
        _render(subject, tmp_path, density_path=None)
        texts = [t for ax in strip_probe["axes"] for t in ax["texts"]]
        assert not any("Fraction of bilateral" in t for t in texts)
        # And the caption must not advertise a clipping rule for a panel that
        # was never drawn.
        assert "capped at non-zero" not in _caption_text(strip_probe)

    def test_every_panel_is_radiological(self, subject, tmp_path, strip_probe):
        """All four must agree on which side of the page is anatomical left.
        The density overlay's imshow used to reset the axes limits and silently
        undo the background's radiological inversion, mirroring that one panel
        (and its R/L markers) relative to its neighbours."""
        _render(subject, tmp_path)
        panels = _panels(strip_probe)
        directions = {p["xlim"][0] > p["xlim"][1] for p in panels}
        assert len(directions) == 1, "panels disagree on left/right"
        assert directions == {True}, "RAS affine must be x-inverted to read radiologically"

    def test_roi_contours_use_the_documented_colours_and_no_red(
        self, subject, tmp_path, strip_probe
    ):
        import matplotlib.colors as mcolors

        _render(subject, tmp_path)
        roi_panel = _panels(strip_probe)[2]
        colors = set()
        for collection in roi_panel["collections"]:
            for rgba in np.atleast_2d(collection.get_edgecolor()):
                colors.add(mcolors.to_hex(rgba[:3]))
        expected = {
            mcolors.to_hex(_style.BRAINSTEM),
            mcolors.to_hex(_style.MOTOR_LEFT),
            mcolors.to_hex(_style.MOTOR_RIGHT),
        }
        assert colors == expected
        # The report's laterality policy: never red.
        assert mcolors.to_hex(_style.CANDIDATE) not in colors

    def test_roi_slab_voxel_counts_are_reported(self, subject, tmp_path):
        """The honesty check: a plane chosen for the bundle can miss a cortical
        ROI, so the counts that decide whether it did are recorded."""
        counts = _sidecar(_render(subject, tmp_path))["RoiSlabVoxelCounts"]
        assert set(counts) == {"brainstem", "motor_left", "motor_right"}
        assert all(isinstance(v, int) for v in counts.values())

    def test_roi_contours_are_slab_projected_not_single_plane(self, subject,
                                                              tmp_path):
        """Projecting through the 10 mm slab is what makes a cortical ROI show up
        at all on a plane chosen for the bundle."""
        from csttool.metrics.modules.visualizations import _slab_projected_volume

        index = _sidecar(_render(subject, tmp_path))["SliceIndex"]
        mask = _dseg() == 2
        projected, n_slab = _slab_projected_volume(mask, RAS_AFFINE, "coronal",
                                                   index, 10.0)
        single_plane = int(mask[:, index, :].sum())
        assert n_slab >= single_plane
        assert projected[:, index, :].sum() >= single_plane

    def test_streamlines_are_drawn_on_the_anatomy_not_in_world_mm(
        self, subject, tmp_path, strip_probe
    ):
        """The frame defect: the overlay decides slab membership in world mm but
        must *draw* in voxel coordinates, which is the frame the background
        image lives in. Plotting world points put the bundle off the slice
        entirely and autoscaled the axes out to contain it."""
        _render(subject, tmp_path)
        panel = _panels(strip_probe)[3]
        assert panel["lines"], "no streamlines drawn"
        x_lo, x_hi = sorted(panel["xlim"])
        y_lo, y_hi = sorted(panel["ylim"])
        for line in panel["lines"]:
            xs, ys = line.get_xdata(), line.get_ydata()
            assert xs.min() >= x_lo and xs.max() <= x_hi
            assert ys.min() >= y_lo and ys.max() <= y_hi

    def test_streamline_counts_are_recorded(self, subject, tmp_path):
        sidecar = _sidecar(_render(subject, tmp_path))
        assert sidecar["StreamlineCounts"] == {"left": 12, "right": 12}
        assert sidecar["StreamlinesDrawn"] == {"left": 12, "right": 12}


class TestSharedFurniture:
    def test_no_matplotlib_legend_anywhere(self, subject, tmp_path, strip_probe):
        """The keys are measured-and-placed swatch rows, not Legend artists: a
        Legend cannot be aligned to a panel column or held to a fixed band."""
        _render(subject, tmp_path)
        assert strip_probe["n_legends"] == 0

    def test_hemisphere_counts_live_with_the_trajectories_they_count(
        self, subject, tmp_path, strip_probe
    ):
        """The counts used to sit in the shared caption, away from panel 4."""
        _render(subject, tmp_path)
        texts = [t for ax in strip_probe["axes"] for t in ax["texts"]]
        assert texts.count("Left n=12") == 1
        assert texts.count("Right n=12") == 1

    def test_caption_states_only_the_facts_shared_by_all_four_panels(
        self, subject, tmp_path, strip_probe
    ):
        """Slice, rule, slab and orientation are true of the whole composite.
        Counts and the ROI key are not, and have moved to their own panels."""
        path = _render(subject, tmp_path)
        index = _sidecar(path)["SliceIndex"]
        caption = _caption_text(strip_probe)
        assert f"Coronal slice {index}" in caption
        assert "bilateral occupancy" in caption
        assert "10 mm display slab" in caption
        assert "radiological convention" in caption
        for absent in ("n=", "brainstem", "motor", "vmax"):
            assert absent not in caption

    def test_roi_key_swatches_use_the_documented_colours(self, subject, tmp_path,
                                                         strip_probe):
        """The key sits in panel 3's own column and its colour lives on the
        swatch, so it is the same artist property the contours are drawn with."""
        import matplotlib.colors as mcolors

        _render(subject, tmp_path)
        key = _key_band(strip_probe)[2]
        assert [t for t in key["texts"]] == ["brainstem", "motor L", "motor R"]
        assert [mcolors.to_hex(c) for c in key["line_colors"]] == [
            mcolors.to_hex(_style.BRAINSTEM),
            mcolors.to_hex(_style.MOTOR_LEFT),
            mcolors.to_hex(_style.MOTOR_RIGHT),
        ]

    def test_an_roi_the_slab_never_reaches_is_greyed_not_dropped(
        self, subject, tmp_path, strip_probe
    ):
        """A zero slab count must stay visible: silently dropping the entry
        would read as "this ROI was not requested"."""
        import matplotlib.colors as mcolors

        dseg = _dseg()
        dseg[dseg == 1] = 0           # brainstem present in the file, absent in the slab
        path = _write_nifti(tmp_path / "no_stem.nii.gz", dseg, RAS_AFFINE)
        _render(subject, tmp_path, roi_dseg_path=path)
        key = _key_band(strip_probe)[2]
        assert "brainstem" in key["texts"]
        assert mcolors.to_hex(key["line_colors"][0]) != mcolors.to_hex(_style.BRAINSTEM)

    def test_roi_key_is_omitted_when_there_is_no_segmentation(self, subject,
                                                              tmp_path, strip_probe):
        _render(subject, tmp_path, roi_dseg_path=None)
        texts = [t for ax in strip_probe["axes"] for t in ax["texts"]]
        assert "brainstem" not in texts


class TestGridGuard:
    def _wrong_grid(self, tmp_path, name, data):
        wrong = np.diag([2.0, 2.0, 2.0, 1.0])
        return _write_nifti(tmp_path / name, data, wrong)

    def test_mismatched_dseg_affine_raises(self, subject, tmp_path):
        bad = self._wrong_grid(tmp_path, "bad_dseg.nii.gz", _dseg())
        with pytest.raises(ValueError, match="not on the FA grid"):
            _render(subject, tmp_path, roi_dseg_path=bad)

    def test_mismatched_dseg_shape_raises(self, subject, tmp_path):
        bad = _write_nifti(tmp_path / "bad_shape.nii.gz",
                           np.zeros((8, 8, 8), np.uint8), RAS_AFFINE)
        with pytest.raises(ValueError, match="not on the FA grid"):
            _render(subject, tmp_path, roi_dseg_path=bad)

    def test_mismatched_density_raises(self, subject, tmp_path):
        bad = self._wrong_grid(tmp_path, "bad_density.nii.gz", _density())
        with pytest.raises(ValueError, match="not on the FA grid"):
            _render(subject, tmp_path, density_path=bad)

    def test_a_merely_absent_product_does_not_raise(self, subject, tmp_path):
        """Absent is a normal state and degrades; present-but-wrong-grid is a
        defect and raises. The two must not be conflated."""
        assert _render(subject, tmp_path,
                       roi_dseg_path=tmp_path / "does_not_exist.nii.gz").exists()


class TestDeterminism:
    def test_byte_reproducible_across_two_runs(self, subject, tmp_path):
        """This is the ``viz_rng`` fix under test: the legacy panel subsampled
        through a seed derived from Python's builtin ``hash`` of a string, which
        is randomised per process unless PYTHONHASHSEED is set."""
        a = _render(subject, tmp_path, subject_id="run-a")
        b = _render(subject, tmp_path, subject_id="run-b")
        np.testing.assert_array_equal(mpimg.imread(str(a)), mpimg.imread(str(b)))

    def test_seed_is_recorded_and_defaults_to_the_run_seed(self, subject, tmp_path):
        from csttool.reproducibility.context import DEFAULT_SEED

        assert _sidecar(_render(subject, tmp_path))["Seed"] == DEFAULT_SEED

    def test_a_different_seed_draws_a_different_subsample(self, subject, tmp_path):
        """Guards against the subsample being seeded but inert."""
        many = {
            **subject,
            "cst_left_path": _write_trk(tmp_path / "big_l.trk",
                                        _bundle(-8.0, n=40, seed=3), RAS_AFFINE),
            "cst_right_path": _write_trk(tmp_path / "big_r.trk",
                                         _bundle(8.0, n=40, seed=4), RAS_AFFINE),
        }
        a = plot_report_qc_strip(output_dir=tmp_path, subject_id="s1",
                                 max_streamlines=5, seed=1, **many)
        b = plot_report_qc_strip(output_dir=tmp_path, subject_id="s2",
                                 max_streamlines=5, seed=2, **many)
        assert not np.array_equal(mpimg.imread(str(a)), mpimg.imread(str(b)))


# ---------------------------------------------------------------------------
# Layout invariants.
#
# The defect this class exists for shipped because the committed fixture grid is
# (24, 20, 18) — coronal aspect 1.33, *narrower* than the axes box the strip
# allocated — while real acquisitions sit near 1.6, *wider*. The sign of that
# mismatch decides whether Matplotlib's apply_aspect shrinks the image box
# horizontally or vertically, so the suite exercised a geometry regime that
# never reached a report: titles drifted up to 2.5 mm off their panels, ~5 mm of
# the canvas went to dead white, and the colourbar label overprinted the shared
# caption by 0.46 mm. Every test below is therefore parametrised over shapes
# spanning both regimes, and asserts bbox relations rather than pixels.
# ---------------------------------------------------------------------------

LAYOUT_SHAPES = [
    (24, 20, 18),     # the legacy fixture: aspect 1.33, image-height cap binds
    (30, 24, 18),     # aspect 1.67: width-bound, the normal regime
    (34, 24, 18),     # aspect 1.89: width-bound, a short strip
    (18, 20, 24),     # aspect 0.75: portrait, letterboxes hard
]


def _layout_subject(tmp_path, shape, n_streamlines=12, density_scale=1.0):
    """Every strip input on an arbitrary grid, written to disk as products are."""
    sx, sy, sz = shape
    affine = np.array([[2.0, 0, 0, -float(sx)],
                       [0, 2.0, 0, -float(sy)],
                       [0, 0, 2.0, -float(sz)],
                       [0, 0, 0, 1.0]])
    xx, yy, zz = np.indices(shape)
    fa = np.exp(-(((xx - sx / 2) / (sx / 3.0)) ** 2
                  + ((yy - sy / 2) / (sy / 3.0)) ** 2
                  + ((zz - sz / 2) / (sz / 3.0)) ** 2)).clip(0, 1).astype(np.float32)
    v1 = np.zeros(shape + (3,), dtype=np.float32)
    v1[..., 2] = 1.0
    density = np.zeros(shape, dtype=np.float32)
    density[sx // 3:sx // 3 + 2, sy // 2 - 1:sy // 2 + 1, 2:sz - 2] = 0.4 * density_scale
    density[2 * sx // 3:2 * sx // 3 + 2, sy // 2 - 1:sy // 2 + 1, 2:sz - 2] = 0.3 * density_scale
    dseg = np.zeros(shape, dtype=np.uint8)
    dseg[sx // 2 - 1:sx // 2 + 1, sy // 2 - 1:sy // 2 + 1, 1:3] = 1
    dseg[sx // 3:sx // 3 + 2, sy // 2 - 1:sy // 2 + 1, sz - 3:sz - 1] = 2
    dseg[2 * sx // 3:2 * sx // 3 + 2, sy // 2 - 1:sy // 2 + 1, sz - 3:sz - 1] = 3

    def bundle(x_world, seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n_streamlines):
            z = np.linspace(-sz + 3.0, sz - 3.0, 20)
            out.append(np.column_stack([
                np.full(20, x_world + rng.normal(0, 0.4)),
                np.full(20, rng.normal(0, 0.4)),
                z,
            ]))
        return out

    tag = "x".join(str(v) for v in shape)
    return {
        "fa_path": _write_nifti(tmp_path / f"fa_{tag}.nii.gz", fa, affine),
        "v1_path": _write_nifti(tmp_path / f"v1_{tag}.nii.gz", v1, affine),
        "density_path": _write_nifti(tmp_path / f"den_{tag}.nii.gz", density, affine),
        "roi_dseg_path": _write_nifti(tmp_path / f"ds_{tag}.nii.gz", dseg, affine),
        "cst_left_path": _write_trk_shaped(
            tmp_path / f"l_{tag}.trk", bundle(-sx / 3.0, 1), affine, shape),
        "cst_right_path": _write_trk_shaped(
            tmp_path / f"r_{tag}.trk", bundle(sx / 3.0, 2), affine, shape),
    }


def _write_trk_shaped(path, streamlines, affine, shape):
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram

    reference = nib.Nifti1Image(np.zeros(shape, np.float32), affine)
    sft = StatefulTractogram(streamlines, reference, Space.RASMM)
    save_tractogram(sft, str(path), bbox_valid_check=False)
    return path


def _overlap_mm(a, b):
    """Overlapping width and height of two (x0, y0, x1, y1) boxes, in mm."""
    return (min(a[2], b[2]) - max(a[0], b[0]),
            min(a[3], b[3]) - max(a[1], b[1]))


@pytest.mark.parametrize("shape", LAYOUT_SHAPES,
                         ids=lambda s: "x".join(str(v) for v in s))
class TestFurnitureLayout:
    """Bbox relations that must hold whatever grid the subject was acquired on."""

    def test_figure_height_is_derived_and_stays_inside_its_clamp(
        self, tmp_path, strip_probe, shape
    ):
        """Height follows the slice aspect, but never past the page budget: the
        report measures 271.4 mm of 281 and the headroom test demands 6 mm."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        width_mm, height_mm = strip_probe["size_mm"]
        assert width_mm == pytest.approx(QC_STRIP_WIDTH_MM)
        assert QC_STRIP_MIN_HEIGHT_MM <= height_mm <= QC_STRIP_MAX_HEIGHT_MM

    def test_every_annotation_is_inside_the_canvas(self, tmp_path, strip_probe,
                                                   shape):
        """The strip is saved with its exact figure bbox, so anything outside is
        silently cut off — which is how the 7 pt titles used to lose their
        ascenders on a narrow-aspect grid."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        width_mm, height_mm = strip_probe["size_mm"]
        for text, (x0, y0, x1, y1) in _furniture(strip_probe):
            assert x0 >= -0.05 and x1 <= width_mm + 0.05, f"{text!r} off-canvas in x"
            assert y0 >= -0.05 and y1 <= height_mm + 0.05, f"{text!r} off-canvas in y"

    def test_no_two_annotations_overlap(self, tmp_path, strip_probe, shape):
        """The colourbar label used to overprint the shared caption by a
        measured 0.46 mm, because its band was 2.2 mm and its type needed 2.77."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        items = _furniture(strip_probe)
        for i, (text_a, box_a) in enumerate(items):
            for text_b, box_b in items[i + 1:]:
                dx, dy = _overlap_mm(box_a, box_b)
                assert dx <= 0.1 or dy <= 0.1, (
                    f"{text_a!r} overlaps {text_b!r} by {dx:.2f} x {dy:.2f} mm"
                )

    def test_each_title_sits_immediately_above_its_own_panel(
        self, tmp_path, strip_probe, shape
    ):
        """Unambiguous ownership, stated testably: the title is horizontally
        centred within its panel and touches the top of its image box. It used
        to float 2.5 mm below the reserved band on a wide grid and be clipped by
        the canvas edge on a narrow one — a function of the acquisition matrix."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        for panel in _panels(strip_probe):
            px0, _, px1, py1 = panel["box_mm"]
            tx0, ty0, tx1, _ = panel["title_bbox_mm"]
            centre = (tx0 + tx1) / 2.0
            assert px0 - 0.05 <= centre <= px1 + 0.05, "title not over its panel"
            assert -0.1 <= ty0 - py1 <= 1.5, (
                f"title of {panel['title']!r} sits {ty0 - py1:.2f} mm from its image"
            )

    def test_each_key_stays_inside_its_own_panel_column(self, tmp_path,
                                                        strip_probe, shape):
        """A key that spills into the neighbouring column is exactly the
        ambiguous ownership the shared caption used to create."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        for key in _key_band(strip_probe):
            cx0, _, cx1, _ = key["box_mm"]
            boxes = [b for b in key["text_bboxes_mm"] + key["line_bboxes_mm"] if b]
            assert boxes, "empty key column"
            assert min(b[0] for b in boxes) >= cx0 - 0.05
            assert max(b[2] for b in boxes) <= cx1 + 0.05

    def test_nothing_is_drawn_over_the_anatomy_but_the_orientation_glyphs(
        self, tmp_path, strip_probe, shape
    ):
        """The DEC direction key used to be a 6.1 mm opaque inset covering ~3.5%
        of panel 1, in the corner the CST descends through.

        Checked two ways, because the glyph was an ``inset_axes``: its text
        belonged to the inset, not to the panel, so a per-panel text check alone
        would not have seen it. No *axes* may overlap a panel's image box
        either.
        """
        _render(_layout_subject(tmp_path, shape), tmp_path)
        panels = _panels(strip_probe)
        for panel in panels:
            inside = [t for t, b in zip(panel["texts"], panel["text_bboxes_mm"]) if b]
            assert set(inside) <= {"R", "L"}, f"{inside} drawn over the anatomy"
        panel_boxes = [p["box_mm"] for p in panels]
        for ax in strip_probe["axes"]:
            if ax["box_mm"] in panel_boxes:
                continue
            for box in panel_boxes:
                dx, dy = _overlap_mm(ax["box_mm"], box)
                assert dx <= 0.1 or dy <= 0.1, (
                    f"an axes overlaps a panel image by {dx:.2f} x {dy:.2f} mm"
                )

    def test_the_four_key_columns_align_with_the_four_panels(
        self, tmp_path, strip_probe, shape
    ):
        _render(_layout_subject(tmp_path, shape), tmp_path)
        panels, keys = _panels(strip_probe), _key_band(strip_probe)
        assert len(keys) == 4, "every panel keeps a key column, degraded or not"
        for panel, key in zip(panels, keys):
            assert key["box_mm"][0] == pytest.approx(panel["box_mm"][0], abs=0.05)
            assert key["box_mm"][2] == pytest.approx(panel["box_mm"][2], abs=0.05)


    def test_the_four_image_rectangles_are_identical(self, tmp_path, strip_probe,
                                                     shape):
        """Same size and the same top and bottom edge, exactly — the reader
        compares the four by eye, so any difference reads as a difference in the
        data."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        boxes = [p["box_mm"] for p in _panels(strip_probe)]
        widths = {round(b[2] - b[0], 6) for b in boxes}
        heights = {round(b[3] - b[1], 6) for b in boxes}
        assert len(widths) == 1 and len(heights) == 1
        assert len({round(b[1], 6) for b in boxes}) == 1, "bottom edges differ"
        assert len({round(b[3], 6) for b in boxes}) == 1, "top edges differ"

    def test_the_four_titles_share_one_baseline_and_one_gap(self, tmp_path,
                                                            strip_probe, shape):
        """Consistent title position is half of the visual hierarchy; the images
        are top-aligned, so the titles must be too."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        panels = _panels(strip_probe)
        gaps = {round(p["title_bbox_mm"][1] - p["box_mm"][3], 4) for p in panels}
        assert len(gaps) == 1, f"title-to-image spacing differs: {gaps}"
        assert len({round(p["title_bbox_mm"][1], 4) for p in panels}) == 1

    def test_the_key_band_is_a_fixed_height_region_under_every_panel(
        self, tmp_path, strip_probe, shape
    ):
        """Same box for all four whatever the key contains, so DEC axes, a
        colourbar, ROI colours and hemisphere counts still line up as a row."""
        from csttool.metrics.modules.visualizations import _STRIP_KEY_MM

        _render(_layout_subject(tmp_path, shape), tmp_path)
        keys = _key_band(strip_probe)
        assert len(keys) == 4
        heights = {round(k["box_mm"][3] - k["box_mm"][1], 6) for k in keys}
        assert heights == {round(_STRIP_KEY_MM, 6)}
        assert len({round(k["box_mm"][1], 6) for k in keys}) == 1

    def test_every_key_label_sits_on_one_shared_baseline(self, tmp_path,
                                                         strip_probe, shape):
        """Set with ``va='baseline'`` for exactly this reason: centring each
        string's bounding box instead puts a row of all-caps labels 0.127 mm off
        a row with ascenders, and the four legends stop lining up."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        baselines = set()
        for key in _key_band(strip_probe):
            for text, va, y in zip(key["texts"], key["text_vas"],
                                   key["text_anchor_y_mm"]):
                assert va == "baseline", f"{text!r} is not set on a baseline"
                baselines.add(round(y, 4))
        # Two rows only: the shared label row, and panel 2's tick row on its bar.
        assert len(baselines) == 2, f"key labels sit on {len(baselines)} baselines"

    def test_the_caption_is_set_much_further_from_the_keys_than_they_are_from_the_images(
        self, tmp_path, strip_probe, shape
    ):
        """Measured between ink, not between bands: proximity is what tells a
        reader the caption belongs to the whole composite rather than to panel 4."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        keys = _key_band(strip_probe)
        image_bottom = min(p["box_mm"][1] for p in _panels(strip_probe))
        key_ink_top = max(b[3] for k in keys for b in k["text_bboxes_mm"] if b)
        key_ink_bottom = min(b[1] for k in keys for b in k["text_bboxes_mm"] if b)
        caption_ink_top = max(b[3] for b in _caption_axes(strip_probe)["text_bboxes_mm"] if b)

        image_to_key = image_bottom - key_ink_top
        key_to_caption = key_ink_bottom - caption_ink_top
        assert image_to_key > 0.8, "image and its key are touching"
        assert key_to_caption > 2.0 * image_to_key, (
            f"caption sits {key_to_caption:.2f} mm below the keys against "
            f"{image_to_key:.2f} mm between image and key — too close to read "
            "as a caption for the composite"
        )

    def test_the_colourbar_reads_as_one_unit(self, tmp_path, strip_probe, shape):
        """Label above, bar, and the two end ticks straddling it — the ticks
        must overlap the bar vertically, not sit on a third line below it."""
        _render(_layout_subject(tmp_path, shape), tmp_path)
        key = _key_band(strip_probe)[1]
        bar = _colourbar_axes(strip_probe)
        ticks = [b for t, b in zip(key["texts"], key["text_bboxes_mm"])
                 if b and t.rstrip("%").lstrip("\u2265").replace(".", "").isdigit()]
        label = [b for t, b in zip(key["texts"], key["text_bboxes_mm"])
                 if b and t == "Fraction of bilateral CST streamlines"][0]
        assert len(ticks) == 2
        for box in ticks:
            overlap = min(box[3], bar[3]) - max(box[1], bar[1])
            assert overlap > 0, "an end tick does not straddle the bar"
        assert label[1] > bar[3], "the quantity name must sit above the bar"
        # The bar is thick enough to read as a scale rather than a rule.
        assert bar[3] - bar[1] >= 2.0

    def test_type_hierarchy_holds(self, tmp_path, strip_probe, shape):
        """Titles outrank keys, and the strip's titles match the profile
        matrix's subplot titles exactly so the two figures cannot drift apart
        again — the strip used to be two points smaller at every level, with its
        primary legend set at the matrix's footnote size."""
        from csttool.metrics.modules.visualizations import _RPT_TITLE_PT

        _render(_layout_subject(tmp_path, shape), tmp_path)
        titles = {p["title_pt"] for p in _panels(strip_probe)}
        assert titles == {_RPT_TITLE_PT}
        key_pts = {pt for key in _key_band(strip_probe) for pt in key["text_pts"]}
        assert key_pts, "no key type found"
        assert max(key_pts) < min(titles)
        assert len(key_pts) == 1, "one type size for the whole key band"


@pytest.mark.parametrize("n_streamlines", [0, 7, 120_000])
def test_streamline_counts_do_not_reflow_the_layout(tmp_path, strip_probe,
                                                    n_streamlines):
    """Six digits must lay out like one. The counts sit in panel 4's key, so a
    long one would push into panel 3's column rather than simply looking wide."""
    _render(_layout_subject(tmp_path, (30, 24, 18), n_streamlines=n_streamlines),
            tmp_path)
    width_mm, height_mm = strip_probe["size_mm"]
    for text, (x0, y0, x1, y1) in _furniture(strip_probe):
        assert x0 >= -0.05 and x1 <= width_mm + 0.05, f"{text!r} off-canvas"
    for key in _key_band(strip_probe):
        cx0, _, cx1, _ = key["box_mm"]
        for box in [b for b in key["text_bboxes_mm"] + key["line_bboxes_mm"] if b]:
            assert box[0] >= cx0 - 0.05 and box[2] <= cx1 + 0.05


@pytest.mark.parametrize("scale, expected", [(1e-4, "0.004%"), (1.0, "40%"),
                                             (105.0, "4200%")])
def test_density_endpoint_magnitude_does_not_reflow_the_colourbar(
    tmp_path, strip_probe, scale, expected
):
    """The endpoint is subject-adaptive and printed beside the bar. Under the
    old ``{vmax:.4f}`` inside a sentence, reaching two integer digits pushed the
    label into the neighbouring panels; the bar is now measured around whatever
    the endpoint needs, so the column absorbs the change instead."""
    _render(_layout_subject(tmp_path, (30, 24, 18), density_scale=scale), tmp_path)
    texts = [t for ax in strip_probe["axes"] for t in ax["texts"]]
    assert any(t.lstrip("\u2265") == expected for t in texts), texts
    key = _key_band(strip_probe)[1]
    cx0, _, cx1, _ = key["box_mm"]
    for box in [b for b in key["text_bboxes_mm"] if b]:
        assert box[0] >= cx0 - 0.05 and box[2] <= cx1 + 0.05


class TestStripGeometryFunction:
    """``qc_strip_geometry`` in isolation — it is the whole layout contract."""

    @pytest.mark.parametrize("canvas_w, canvas_h", [(160, 80), (128, 66), (96, 50)])
    def test_a_width_bound_grid_fills_its_panel_exactly(self, canvas_w, canvas_h):
        """No letterbox when the aspect fits: the panel box *is* the data's
        shape, which is what makes apply_aspect a no-op. The image-height cap
        binds below aspect ~1.83, so these are the grids wide enough to escape
        it."""
        from csttool.metrics.modules.visualizations import _STRIP_IMAGE_MAX_MM

        geom = qc_strip_geometry(canvas_w, canvas_h)
        assert geom["image_h_mm"] == pytest.approx(
            geom["panel_w_mm"] / geom["aspect"]
        )
        assert geom["box_aspect"] == pytest.approx(geom["aspect"])
        assert geom["image_h_mm"] <= _STRIP_IMAGE_MAX_MM

    @pytest.mark.parametrize("canvas_w, canvas_h", [(24, 18), (96, 60), (60, 80)])
    def test_a_tall_grid_letterboxes_rather_than_narrowing_the_panel(
        self, canvas_w, canvas_h
    ):
        """Panel width is constant so the key band keeps its room; the image
        gives way instead. Narrowing the panel to 40 mm at aspect 1.33 would
        leave the ROI key, which needs ~45 mm, nowhere to go.

        The letterbox is black against a slice whose own margins are black, so
        it costs image scale rather than legibility — which is what pays for the
        whitespace in the vertical stack."""
        from csttool.metrics.modules.visualizations import _STRIP_IMAGE_MAX_MM

        geom = qc_strip_geometry(canvas_w, canvas_h)
        full_width = qc_strip_geometry(160, 80)["panel_w_mm"]
        assert geom["panel_w_mm"] == pytest.approx(full_width)
        assert geom["image_h_mm"] == pytest.approx(_STRIP_IMAGE_MAX_MM)
        assert geom["box_aspect"] > geom["aspect"]

    @pytest.mark.parametrize("canvas_w, canvas_h",
                             [(24, 18), (96, 60), (160, 60), (20, 40), (300, 61)])
    def test_height_always_lands_inside_the_page_budget(self, canvas_w, canvas_h):
        geom = qc_strip_geometry(canvas_w, canvas_h)
        assert QC_STRIP_MIN_HEIGHT_MM <= geom["height_mm"] <= QC_STRIP_MAX_HEIGHT_MM
        assert geom["width_mm"] == pytest.approx(QC_STRIP_WIDTH_MM)

    def test_panels_are_evenly_spaced_and_span_the_content_width(self):
        geom = qc_strip_geometry(96, 60)
        x0 = geom["panel_x0_mm"]
        gaps = [x0[i + 1] - (x0[i] + geom["panel_w_mm"]) for i in range(3)]
        assert all(g == pytest.approx(gaps[0]) for g in gaps)
        assert x0[0] == pytest.approx(0.0)
        assert x0[3] + geom["panel_w_mm"] == pytest.approx(QC_STRIP_WIDTH_MM)

    def test_every_millimetre_of_the_stack_is_accounted_for(self):
        """Margin, title, image, gap, key, gap, caption, margin — in that order
        and summing exactly to the figure height. The dead white the old fixed
        allocation produced was ~5.1 mm of a 44 mm canvas."""
        from csttool.metrics.modules.visualizations import (
            _STRIP_CAPTION_MM, _STRIP_IMAGE_KEY_GAP_MM,
            _STRIP_KEY_CAPTION_GAP_MM, _STRIP_KEY_MM, _STRIP_MARGIN_TOP_MM,
            _STRIP_TITLE_MM,
        )

        geom = qc_strip_geometry(160, 80)
        assert geom["key_y0_mm"] == pytest.approx(
            geom["caption_y0_mm"] + _STRIP_CAPTION_MM + _STRIP_KEY_CAPTION_GAP_MM)
        assert geom["image_y0_mm"] == pytest.approx(
            geom["key_y0_mm"] + _STRIP_KEY_MM + _STRIP_IMAGE_KEY_GAP_MM)
        top = geom["image_y0_mm"] + geom["image_h_mm"] + _STRIP_TITLE_MM
        assert top + _STRIP_MARGIN_TOP_MM == pytest.approx(geom["height_mm"])

    def test_the_caption_is_set_further_from_the_keys_than_they_are_from_their_images(self):
        """Proximity is the only thing that distinguishes the shared caption
        from a fifth legend — both are centred type at the same size — so the
        gap above it must be the larger one."""
        from csttool.metrics.modules.visualizations import (
            _STRIP_IMAGE_KEY_GAP_MM, _STRIP_KEY_CAPTION_GAP_MM,
        )

        assert _STRIP_KEY_CAPTION_GAP_MM > _STRIP_IMAGE_KEY_GAP_MM

    def test_a_wider_grid_gives_a_shorter_strip(self):
        heights = [qc_strip_geometry(w, 60)["height_mm"] for w in (110, 130, 160)]
        assert heights == sorted(heights, reverse=True)


class TestDensityPercentFormatting:
    """``format_density_percent`` — display only, never the stored value."""

    @pytest.mark.parametrize("fraction, expected", [
        (0.074, "7.4%"),        # the validation subject's display cap
        (0.138952, "13.9%"),    # its true maximum
        (0.5, "50%"),           # no meaningless trailing zero
        (1.0, "100%"),
        (0.0, "0%"),
        (0.42665, "42.7%"),     # a fully occupied unilateral voxel
        (0.004, "0.4%"),
        (0.00042, "0.042%"),    # two significant figures below 1%
    ])
    def test_reads_as_a_percentage(self, fraction, expected):
        assert format_density_percent(fraction) == expected

    def test_saturation_marks_the_endpoint_as_a_floor(self):
        assert format_density_percent(0.074, saturated=True) == "\u22657.4%"
        assert format_density_percent(0.074, saturated=False) == "7.4%"

    def test_zero_never_carries_a_saturation_marker(self):
        """The lower endpoint is exact by construction."""
        assert format_density_percent(0.0, saturated=True) == "0%"

    def test_every_output_is_a_percentage(self):
        for fraction in (0.0, 1e-6, 0.001, 0.074, 0.5, 1.0):
            assert format_density_percent(fraction).endswith("%")

    def test_formatting_does_not_touch_the_underlying_value(self):
        """A formatter, not a transform: the caller's float is unchanged and the
        label is derived from it rather than replacing it."""
        value = 0.0740321
        assert format_density_percent(value) == "7.4%"
        assert value == 0.0740321


class TestAnnotationFitsWhateverItSays:
    """The two auto-fits that keep dynamic text inside its box."""

    def test_the_caption_shrinks_rather_than_clipping(self):
        """The figure is saved with its exact bbox, so an overlong caption is
        cut off at the canvas edge rather than wrapped. It grows with the slice
        index, the rule name, the slab and the density note, so it is measured."""
        import matplotlib.pyplot as plt
        from csttool.metrics.modules.visualizations import (
            _STRIP_CAPTION_MIN_PT, _STRIP_CAPTION_PT, _fit_caption_fontsize,
        )

        fig = plt.figure(figsize=(194 / 25.4, 47 / 25.4))
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        short = "Coronal slice 42 · 10 mm display slab"
        assert _fit_caption_fontsize(ax, renderer, short,
                                     _STRIP_CAPTION_PT) == _STRIP_CAPTION_PT

        long_text = short + " · " + ("a very long clause indeed " * 12)
        fitted = _fit_caption_fontsize(ax, renderer, long_text, _STRIP_CAPTION_PT)
        assert _STRIP_CAPTION_MIN_PT <= fitted < _STRIP_CAPTION_PT
        plt.close(fig)

    @pytest.mark.parametrize("shape", LAYOUT_SHAPES,
                             ids=lambda s: "x".join(str(v) for v in s))
    def test_the_colourbar_ticks_keep_clear_of_the_column_edge(self, tmp_path,
                                                               strip_probe, shape):
        """Solving the bar width against the ticks lands the wider one exactly
        on the boundary; the edge margin is what keeps it off.

        The threshold is an absolute millimetre figure, deliberately not
        ``_STRIP_CBAR_EDGE_MM`` — asserting against the constant under test
        would pass for any value of it, including zero.
        """
        _render(_layout_subject(tmp_path, shape, density_scale=1.0), tmp_path)
        key = _key_band(strip_probe)[1]
        cx0, _, cx1, _ = key["box_mm"]
        for text, box in zip(key["texts"], key["text_bboxes_mm"]):
            if box is None or not text.rstrip("%").lstrip("\u2265").replace(".", "").isdigit():
                continue
            assert box[0] - cx0 >= 0.5, f"{text!r} crowds the left column edge"
            assert cx1 - box[2] >= 0.5, f"{text!r} crowds the right column edge"
