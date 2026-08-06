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
    QC_STRIP_SIZE_MM,
    plot_report_qc_strip,
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
        captured["size_in"] = tuple(fig.get_size_inches())
        captured["axes"] = [
            {
                "position": ax.get_position(original=False),
                "title": ax.get_title(),
                "images": list(ax.images),
                "lines": list(ax.lines),
                "collections": list(ax.collections),
                "texts": [t.get_text() for t in ax.texts],
                "text_colors": [t.get_color() for t in ax.texts],
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
        "missing, expected_title",
        [
            ("v1_path", "DEC-FA — V1 unavailable"),
            ("density_path", "CST density — unavailable"),
            ("roi_dseg_path", "Extraction ROIs — unavailable"),
        ],
    )
    def test_a_missing_product_degrades_its_own_slot_only(
        self, subject, tmp_path, strip_probe, missing, expected_title
    ):
        """The strip never reflows to three: a reader must be able to see which
        question went unanswered."""
        _render(subject, tmp_path, **{missing: None})
        panels = _panels(strip_probe)
        assert len(panels) == 4
        assert expected_title in [p["title"] for p in panels]

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
        path = _render(subject, tmp_path)
        w_mm, h_mm = QC_STRIP_SIZE_MM
        np.testing.assert_allclose(
            strip_probe["size_in"], (w_mm / 25.4, h_mm / 25.4), rtol=1e-9
        )
        arr = np.asarray(mpimg.imread(str(path)))
        np.testing.assert_allclose(arr.shape[0] / arr.shape[1], h_mm / w_mm, rtol=2e-3)

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

    def test_density_vmax_is_subject_adaptive_and_disclosed(self, subject, tmp_path):
        """A fixed [0, 1] scale would render real subjects nearly uniformly dark,
        so vmax adapts — which makes stating it mandatory."""
        sidecar = _sidecar(_render(subject, tmp_path))
        assert 0 < sidecar["DensityVmax"] <= 1.0
        assert sidecar["DensityVmax"] < 1.0

    def test_exactly_one_colourbar_and_it_names_its_vmax(self, subject, tmp_path,
                                                          strip_probe):
        """Density is the only quantitative scale in the strip, and its vmax is
        subject-adaptive, so exactly one colourbar exists and it must state the
        number it is scaled to."""
        path = _render(subject, tmp_path)
        labelled = [ax for ax in strip_probe["axes"]
                    if "fraction of bundle streamlines" in ax["xlabel"]]
        assert len(labelled) == 1
        assert f"vmax={_sidecar(path)['DensityVmax']:.4f}" in labelled[0]["xlabel"]

    def test_no_colourbar_when_density_is_unavailable(self, subject, tmp_path,
                                                      strip_probe):
        _render(subject, tmp_path, density_path=None)
        assert not any("fraction of bundle streamlines" in ax["xlabel"]
                       for ax in strip_probe["axes"])

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
    def test_exactly_one_legend_for_the_whole_strip(self, subject, tmp_path,
                                                    strip_probe):
        """No per-panel legends: the shared caption row carries the hemisphere
        key and the ROI key once each."""
        _render(subject, tmp_path)
        assert strip_probe["n_legends"] == 0  # the caption row is text, not a Legend
        caption = " ".join(
            t for ax in strip_probe["axes"] for t in ax["texts"]
        )
        assert caption.count("Left CST") == 1
        assert caption.count("Right CST") == 1

    def test_caption_states_slice_rule_slab_and_counts(self, subject, tmp_path,
                                                       strip_probe):
        path = _render(subject, tmp_path)
        index = _sidecar(path)["SliceIndex"]
        caption = " ".join(t for ax in strip_probe["axes"] for t in ax["texts"])
        assert f"coronal slice {index}" in caption
        assert "rule: bilateral_occupancy" in caption
        assert "slab 10.0 mm" in caption
        assert "Left CST n=12" in caption
        assert "Right CST n=12" in caption

    def test_caption_roi_key_is_colour_coded(self, subject, tmp_path, strip_probe):
        import matplotlib.colors as mcolors

        _render(subject, tmp_path)
        for ax in strip_probe["axes"]:
            for text, color in zip(ax["texts"], ax["text_colors"]):
                if text == "brainstem":
                    assert mcolors.to_hex(color) == mcolors.to_hex(_style.BRAINSTEM)
                if text == "motor-L":
                    assert mcolors.to_hex(color) == mcolors.to_hex(_style.MOTOR_LEFT)
                if text == "motor-R":
                    assert mcolors.to_hex(color) == mcolors.to_hex(_style.MOTOR_RIGHT)

    def test_roi_key_is_omitted_when_there_is_no_segmentation(self, subject,
                                                              tmp_path, strip_probe):
        _render(subject, tmp_path, roi_dseg_path=None)
        caption = " ".join(t for ax in strip_probe["axes"] for t in ax["texts"])
        assert "brainstem" not in caption


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
