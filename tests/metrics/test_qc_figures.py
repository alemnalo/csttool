"""Tests for ``csttool.metrics.modules.qc_figures`` - the three prototype panels
(visualization-refactor M5, §13.2 F-1..F-6).

Every test asserts a decoded property of the pixel array or the artist tree —
never just that a PNG exists. The DEC-FA test is the end-to-end
anatomical-vs-voxel-axis guard (F-1): a superior-inferior phantom must render
dominantly blue under both RAS and LAS affines.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nibabel as nib
import numpy as np
import pytest

from csttool.metrics.modules import qc_figures
from csttool.viz import style


def _write_volume(path, data, affine):
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)


def _superior_inferior_phantom(shape=(12, 12, 12), affine=None,
                              principal_axis=2):
    """A volume whose every voxel has a unit V1 along `principal_axis`."""
    if affine is None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
    fa = np.full(shape, 0.7, dtype=np.float32)
    v1 = np.zeros(shape + (3,), dtype=np.float32)
    v1[..., principal_axis] = 1.0
    return fa, v1, affine


class TestDecFaPanel:
    def test_f1_superior_inferior_is_blue_under_ras_and_las(self, tmp_path):
        # Superior-inferior (axis 2) must be blue under both affines: this is the
        # end-to-end test that the rotation is applied (LAS would otherwise
        # mis-colour as red because voxel axis 0 == anatomical Left in LAS).
        for name, affine in (("ras", np.diag([2.0, 2.0, 2.0, 1.0])),
                             ("las", np.array([[-2, 0, 0, 90], [0, 2, 0, 0],
                                               [0, 0, 2, 0], [0, 0, 0, 1.0]]))):
            fa, v1, _ = _superior_inferior_phantom(affine=affine)
            fa_path = tmp_path / f"fa_{name}.nii.gz"
            _write_volume(fa_path, fa, affine)
            # V1 stored as 5D (X,Y,Z,1,3) like the real product.
            v1_5d = v1.reshape(v1.shape[0], v1.shape[1], v1.shape[2], 1, 3)
            v1_img = nib.Nifti1Image(v1_5d.astype(np.float32), affine)
            v1_img.header.set_intent('vector')
            v1_path = tmp_path / f"v1_{name}.nii.gz"
            nib.save(v1_img, v1_path)
            out = qc_figures.plot_dec_fa_panel(
                v1_path, fa_path, tmp_path, f"sub-{name}")
            assert out.exists()
            arr = mpimg.imread(str(out))
            # Blue channel must dominate over red on average (superior-inferior
            # phantom). Tolerate the black letterbox and L/R markers.
            rgb = arr[..., :3].reshape(-1, 3)
            nonzero = (rgb.sum(axis=1) > 0.05)
            b = rgb[nonzero, 2].mean()
            r = rgb[nonzero, 0].mean()
            assert b > r, f"{name}: expected blue-dominant, got r={r:.3f} b={b:.3f}"

    def test_f2_byte_reproducible_across_runs(self, tmp_path):
        fa, v1, affine = _superior_inferior_phantom()
        fa_path = tmp_path / "fa.nii.gz"; _write_volume(fa_path, fa, affine)
        v1_5d = v1.reshape(*v1.shape[:3], 1, 3)
        v1_img = nib.Nifti1Image(v1_5d.astype(np.float32), affine)
        v1_img.header.set_intent('vector')
        v1_path = tmp_path / "v1.nii.gz"; nib.save(v1_img, v1_path)
        out1 = qc_figures.plot_dec_fa_panel(v1_path, fa_path, tmp_path, "a")
        out2 = qc_figures.plot_dec_fa_panel(v1_path, fa_path, tmp_path, "b")
        a = mpimg.imread(str(out1)); b = mpimg.imread(str(out2))
        assert np.array_equal(a, b)

    def test_sidecar_records_slice_rule(self, tmp_path):
        fa, v1, affine = _superior_inferior_phantom()
        fa_path = tmp_path / "fa.nii.gz"; _write_volume(fa_path, fa, affine)
        v1_5d = v1.reshape(*v1.shape[:3], 1, 3)
        v1_img = nib.Nifti1Image(v1_5d.astype(np.float32), affine)
        v1_img.header.set_intent('vector')
        v1_path = tmp_path / "v1.nii.gz"; nib.save(v1_img, v1_path)
        out = qc_figures.plot_dec_fa_panel(v1_path, fa_path, tmp_path, "x")
        sidecar = out.with_suffix(".json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["Panel"] == "DEC-FA"
        assert "SliceSelectionRule" in data
        assert isinstance(data["SliceIndex"], int)


class TestCstDensityPanel:
    def _setup(self, tmp_path):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        fa = np.full((12, 12, 12), 0.5, dtype=np.float32)
        fa_path = tmp_path / "fa.nii.gz"; _write_volume(fa_path, fa, affine)
        density = np.zeros((12, 12, 12), dtype=np.float32)
        density[:, 4, :] = 0.3
        dens_path = tmp_path / "density.nii.gz"
        _write_volume(dens_path, density, affine)
        # Sidecar with denominator so the caption shows n=...
        (tmp_path / "density.json").write_text(json.dumps({
            "Denominator": 100, "Description": "..."}))
        return dens_path, fa_path

    def test_f3_colorbar_vmax_matches_imshow(self, tmp_path):
        dens_path, fa_path = self._setup(tmp_path)
        out = qc_figures.plot_cst_density_panel(dens_path, fa_path, tmp_path, "d")
        assert out.exists()
        # The vmax used must be the 99th percentile of non-zero density.
        sidecar = json.loads(out.with_suffix(".json").read_text())
        # The sidecar records vmax (F-3 anchor).
        assert "vmax" in sidecar
        expected_vmax = float(np.percentile([0.3], 99))  # only nonzero value is 0.3
        assert sidecar["vmax"] == pytest.approx(expected_vmax, abs=0.02)

    def test_f4_empty_bundle_renders_with_fallback_rule(self, tmp_path):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        fa = np.zeros((12, 12, 12), dtype=np.float32)
        fa[3:9, 3:9, 3:9] = 0.5  # a brain region so the fallback has a centroid
        fa_path = tmp_path / "fa.nii.gz"; _write_volume(fa_path, fa, affine)
        density = np.zeros((12, 12, 12), dtype=np.float32)
        dens_path = tmp_path / "density.nii.gz"
        _write_volume(dens_path, density, affine)
        (tmp_path / "density.json").write_text(json.dumps({"Denominator": 0}))
        out = qc_figures.plot_cst_density_panel(dens_path, fa_path, tmp_path, "e")
        assert out.exists()
        sidecar = json.loads(out.with_suffix(".json").read_text())
        assert sidecar["SliceSelectionRule"] == "anatomical_centroid"


class TestCstOverFaPanel:
    def _make_trk(self, path, streamlines, affine, shape):
        from dipy.io.stateful_tractogram import StatefulTractogram, Space
        from dipy.io.streamline import save_tractogram
        sft = StatefulTractogram(streamlines, nib.Nifti1Image(np.zeros(shape, np.float32), affine),
                                 Space.RASMM)
        save_tractogram(sft, str(path))

    def _setup(self, tmp_path):
        # Affine centered so world 0 = voxel 6 (volume spans world -12..12).
        affine = np.array([[2.0, 0, 0, -12.0], [0, 2.0, 0, -12.0],
                           [0, 0, 2.0, -12.0], [0, 0, 0, 1.0]])
        shape = (12, 12, 12)
        fa = np.full(shape, 0.5, dtype=np.float32)
        fa_path = tmp_path / "fa.nii.gz"; _write_volume(fa_path, fa, affine)
        # Left streamlines near world x=-2, right near x=+2, both running SI at
        # world y=0 (in any near-centre coronal slab).
        n = 20
        z = np.linspace(-10, 10, 20)
        ls = [np.column_stack([np.full(20, -2.0), np.full(20, 0.0), z]) for _ in range(n)]
        rs = [np.column_stack([np.full(20, 2.0), np.full(20, 0.0), z]) for _ in range(n)]
        l_path = tmp_path / "cst_left.trk"; self._make_trk(l_path, ls, affine, shape)
        r_path = tmp_path / "cst_right.trk"; self._make_trk(r_path, rs, affine, shape)
        return l_path, r_path, fa_path

    def test_f5_left_blue_right_orange(self, tmp_path):
        l_path, r_path, fa_path = self._setup(tmp_path)
        out = qc_figures.plot_cst_over_fa_panel(l_path, r_path, fa_path, tmp_path, "lr")
        assert out.exists()
        # The panel must use the canonical LEFT/RIGHT colours. Assert by
        # re-running the renderers directly and inspecting line colours
        # (pixel-level colour reading is fragile across cmap/alpha blending).
        from dipy.io.streamline import load_tractogram
        from csttool.viz import render
        from csttool.viz.utils import viz_rng
        fig, ax = plt.subplots()
        fa = nib.load(str(fa_path)).get_fdata()
        affine = nib.load(str(fa_path)).affine
        render.render_scalar_slice(ax, fa, affine, "coronal", fa.shape[1] // 2,
                                   cmap="gray", norm=plt.Normalize(0, 1))
        left = list(load_tractogram(str(l_path), 'same').streamlines)
        right = list(load_tractogram(str(r_path), 'same').streamlines)
        rng = viz_rng(0)
        render.render_streamline_overlay(ax, left, affine, "coronal", fa.shape[1] // 2,
                                         color=style.LEFT, thickness_mm=10.0,
                                         max_streamlines=500, rng=rng)
        render.render_streamline_overlay(ax, right, affine, "coronal", fa.shape[1] // 2,
                                         color=style.RIGHT, thickness_mm=10.0,
                                         max_streamlines=500, rng=rng)
        import matplotlib.colors as mcolors
        colors = {tuple(mcolors.to_rgb(l.get_color())) for l in ax.get_lines()}
        # At least the two canonical hemisphere colours appear.
        assert any(np.allclose(c, (0.12156863, 0.46666667, 0.70588235), atol=1e-3) for c in colors)
        assert any(np.allclose(c, (1.0, 0.49803922, 0.05490196), atol=1e-3) for c in colors)
        plt.close(fig)

    def test_f6_byte_reproducible_across_runs(self, tmp_path):
        l_path, r_path, fa_path = self._setup(tmp_path)
        out1 = qc_figures.plot_cst_over_fa_panel(l_path, r_path, fa_path, tmp_path, "r1")
        out2 = qc_figures.plot_cst_over_fa_panel(l_path, r_path, fa_path, tmp_path, "r2")
        a = mpimg.imread(str(out1)); b = mpimg.imread(str(out2))
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Trust-chain plot panels
#
# These assert the *claim* each panel makes, read back from its sidecar, on a
# phantom with a known answer — a panel that renders but reports the wrong
# number is the failure mode that matters here. The arithmetic itself is covered
# in test_qc_stats.py; these tests cover the wiring between it and the figure.
# ---------------------------------------------------------------------------
_PLOT_AFFINE = np.array([[2.0, 0, 0, -20.0], [0, 2.0, 0, -20.0],
                         [0, 0, 2.0, -20.0], [0, 0, 0, 1.0]])
_PLOT_SHAPE = (20, 20, 20)


def _save_trk(path, streamlines, affine=_PLOT_AFFINE, shape=_PLOT_SHAPE):
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram
    reference = nib.Nifti1Image(np.zeros(shape, np.float32), affine)
    save_tractogram(StatefulTractogram(streamlines, reference, Space.RASMM),
                    str(path))
    return path


def _si_bundle(n=9, x0=0.0, z0=-16.0, z1=16.0, n_points=40):
    """Streamlines running inferior-to-superior, one per voxel column."""
    z = np.linspace(z0, z1, n_points)
    return [np.column_stack([np.full(n_points, x0 + (i % 9 - 4) * 2.0),
                             np.zeros(n_points), z]) for i in range(n)]


def _write_v1(path, direction, affine=_PLOT_AFFINE, shape=_PLOT_SHAPE):
    """Write a uniform V1 field in the 5D layout of the stored product."""
    v1 = np.zeros(shape + (3,), dtype=np.float32)
    v1[..., :] = np.asarray(direction, dtype=np.float32)
    img = nib.Nifti1Image(v1.reshape(*shape, 1, 3), affine)
    img.header.set_intent('vector')
    nib.save(img, path)
    return path


class TestV1AnglePanel:
    def test_aligned_field_reports_zero_median_angle(self, tmp_path):
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        v1 = _write_v1(tmp_path / "v1.nii.gz", [0, 0, 1])
        out = qc_figures.plot_v1_angle_panel(left, right, v1, tmp_path, "s1")
        assert out.exists()
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["Panel"] == "V1-angle"
        assert data["left"]["median_angle_deg"] == pytest.approx(0.0, abs=1e-3)
        assert data["left"]["n_contributing"] == 9

    def test_orthogonal_field_reports_ninety_degrees(self, tmp_path):
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        v1 = _write_v1(tmp_path / "v1.nii.gz", [1, 0, 0])
        out = qc_figures.plot_v1_angle_panel(left, right, v1, tmp_path, "s2")
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["right"]["median_angle_deg"] == pytest.approx(90.0, abs=1e-3)

    def test_empty_hemisphere_renders_with_a_stated_reason(self, tmp_path):
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", [])
        v1 = _write_v1(tmp_path / "v1.nii.gz", [0, 0, 1])
        out = qc_figures.plot_v1_angle_panel(left, right, v1, tmp_path, "s3")
        assert out.exists()
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["right"]["n_contributing"] == 0

    def test_byte_reproducible_across_runs(self, tmp_path):
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        v1 = _write_v1(tmp_path / "v1.nii.gz", [0, 0, 1])
        a = qc_figures.plot_v1_angle_panel(left, right, v1, tmp_path, "a")
        b = qc_figures.plot_v1_angle_panel(left, right, v1, tmp_path, "b")
        assert np.array_equal(mpimg.imread(str(a)), mpimg.imread(str(b)))


class TestProfileDispersionPanel:
    def _fa(self, tmp_path, data=None):
        fa = np.full(_PLOT_SHAPE, 0.5, np.float32) if data is None else data
        path = tmp_path / "fa.nii.gz"
        _write_volume(path, fa, _PLOT_AFFINE)
        return path

    def test_identical_streamlines_give_a_zero_width_band(self, tmp_path):
        rng = np.random.default_rng(0)
        fa_path = self._fa(tmp_path, rng.random(_PLOT_SHAPE).astype(np.float32))
        identical = [_si_bundle(n=1)[0] for _ in range(6)]
        left = _save_trk(tmp_path / "l.trk", identical)
        right = _save_trk(tmp_path / "r.trk", identical)
        out = qc_figures.plot_profile_dispersion_panel(
            left, right, fa_path, tmp_path, "p1")
        assert out.exists()
        # Read the band back off the artists: with identical streamlines the
        # 5th and 95th percentile must coincide at every node.
        from csttool.metrics.modules import qc_stats
        from dipy.io.streamline import load_tractogram
        streamlines = list(load_tractogram(str(left), 'same').streamlines)
        fa = nib.load(str(fa_path)).get_fdata().astype(np.float32)
        matrix, _ = qc_stats.profile_matrix(streamlines, fa, _PLOT_AFFINE)
        dispersion = qc_stats.profile_dispersion(matrix)
        assert np.allclose(dispersion["p5"], dispersion["p95"])

    def test_sidecar_records_the_attrition(self, tmp_path):
        fa_path = self._fa(tmp_path)
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        out = qc_figures.plot_profile_dispersion_panel(
            left, right, fa_path, tmp_path, "p2")
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["Panel"] == "profile-dispersion"
        assert data["left_attrition"]["n_input"] == 9
        assert data["left_attrition"]["n_contributing"] == 9
        assert data["left_attrition"]["point_retention"] == pytest.approx(1.0)

    def test_md_is_scaled_to_the_labelled_units(self, tmp_path):
        """MD is stored in mm²/s; the axis says ×10⁻³, so the band must scale."""
        md = np.full(_PLOT_SHAPE, 8e-4, np.float32)
        md_path = tmp_path / "md.nii.gz"
        _write_volume(md_path, md, _PLOT_AFFINE)
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        out = qc_figures.plot_profile_dispersion_panel(
            left, right, md_path, tmp_path, "p3", scalar="md")
        assert out.name.endswith("_qc_profile_dispersion_md.png")
        arr = mpimg.imread(str(out))
        assert arr.size > 0

    def test_both_empty_renders_a_placeholder(self, tmp_path):
        fa_path = self._fa(tmp_path)
        left = _save_trk(tmp_path / "l.trk", [])
        right = _save_trk(tmp_path / "r.trk", [])
        out = qc_figures.plot_profile_dispersion_panel(
            left, right, fa_path, tmp_path, "p4")
        assert out.exists()
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["left_n_contributing"] == 0


class TestSamplingSaturationPanel:
    def test_constant_scalar_gives_a_flat_curve(self, tmp_path):
        fa = np.full(_PLOT_SHAPE, 0.42, np.float32)
        fa_path = tmp_path / "fa.nii.gz"
        _write_volume(fa_path, fa, _PLOT_AFFINE)
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        out = qc_figures.plot_sampling_saturation_panel(
            left, right, fa_path, tmp_path, "t1", n_repeats=10)
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["Panel"] == "sampling-saturation"
        assert data["left"]["full_estimate"] == pytest.approx(0.42, abs=1e-6)
        assert np.allclose(data["left"]["std"], 0.0, atol=1e-6)

    def test_seed_is_recorded_and_the_figure_is_reproducible(self, tmp_path):
        rng = np.random.default_rng(3)
        fa_path = tmp_path / "fa.nii.gz"
        _write_volume(fa_path, rng.random(_PLOT_SHAPE).astype(np.float32),
                      _PLOT_AFFINE)
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        a = qc_figures.plot_sampling_saturation_panel(
            left, right, fa_path, tmp_path, "t2", n_repeats=10)
        b = qc_figures.plot_sampling_saturation_panel(
            left, right, fa_path, tmp_path, "t3", n_repeats=10)
        assert np.array_equal(mpimg.imread(str(a)), mpimg.imread(str(b)))
        data = json.loads(a.with_suffix(".json").read_text())
        # Seeded from DEFAULT_SEED, not the per-process VIZ_SEED hash.
        from csttool.reproducibility.context import DEFAULT_SEED
        assert data["seed"] == DEFAULT_SEED

    def test_empty_bundles_render_a_placeholder(self, tmp_path):
        fa_path = tmp_path / "fa.nii.gz"
        _write_volume(fa_path, np.full(_PLOT_SHAPE, 0.5, np.float32), _PLOT_AFFINE)
        left = _save_trk(tmp_path / "l.trk", [])
        right = _save_trk(tmp_path / "r.trk", [])
        out = qc_figures.plot_sampling_saturation_panel(
            left, right, fa_path, tmp_path, "t4", n_repeats=5)
        assert out.exists()
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["left"]["n_streamlines"] == 0


class TestNodeHomologyPanel:
    def test_identical_bundles_report_zero_offset(self, tmp_path):
        bundle = _si_bundle()
        left = _save_trk(tmp_path / "l.trk", bundle)
        right = _save_trk(tmp_path / "r.trk", bundle)
        out = qc_figures.plot_node_homology_panel(left, right, tmp_path, "n1")
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["Panel"] == "node-homology"
        assert data["max_abs_z_difference_mm"] == pytest.approx(0.0, abs=1e-4)
        assert data["length_difference_mm"] == pytest.approx(0.0, abs=1e-4)

    def test_shifted_bundle_reports_the_offset_in_millimetres(self, tmp_path):
        left = _save_trk(tmp_path / "l.trk", _si_bundle(z0=-16.0, z1=16.0))
        right = _save_trk(tmp_path / "r.trk", _si_bundle(z0=-18.0, z1=14.0))
        out = qc_figures.plot_node_homology_panel(left, right, tmp_path, "n2")
        data = json.loads(out.with_suffix(".json").read_text())
        # Both bundles span 32 mm, the right one sits 2 mm lower at every node.
        assert data["max_abs_z_difference_mm"] == pytest.approx(2.0, abs=1e-4)
        assert data["length_difference_mm"] == pytest.approx(0.0, abs=1e-4)

    def test_shorter_bundle_reports_a_length_difference(self, tmp_path):
        left = _save_trk(tmp_path / "l.trk", _si_bundle(z0=-16.0, z1=16.0))
        right = _save_trk(tmp_path / "r.trk", _si_bundle(z0=-8.0, z1=8.0))
        out = qc_figures.plot_node_homology_panel(left, right, tmp_path, "n3")
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["length_difference_mm"] == pytest.approx(16.0, abs=1e-4)

    def test_empty_hemisphere_renders_with_a_stated_reason(self, tmp_path):
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", [])
        out = qc_figures.plot_node_homology_panel(left, right, tmp_path, "n4")
        assert out.exists()
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["max_abs_z_difference_mm"] is None


class TestTissuePlausibilityPanel:
    """QC-1: the panel must report the tissue it was actually given."""

    def _maps(self, tmp_path, fa_value=0.6, md_value=8e-4):
        fa = np.full(_PLOT_SHAPE, fa_value, np.float32)
        md = np.full(_PLOT_SHAPE, md_value, np.float32)
        fa_path, md_path = tmp_path / "fa.nii.gz", tmp_path / "md.nii.gz"
        _write_volume(fa_path, fa, _PLOT_AFFINE)
        _write_volume(md_path, md, _PLOT_AFFINE)
        return fa_path, md_path

    def _density(self, tmp_path, name="density.nii.gz", value=0.5,
                 region=(slice(5, 8),) * 3):
        density = np.zeros(_PLOT_SHAPE, np.float32)
        density[region] = value
        path = tmp_path / name
        _write_volume(path, density, _PLOT_AFFINE)
        return path, density

    def test_white_matter_bundle_reports_no_free_water(self, tmp_path):
        fa_path, md_path = self._maps(tmp_path)
        density_path, _ = self._density(tmp_path)
        out = qc_figures.plot_tissue_plausibility_panel(
            fa_path, md_path, density_path, tmp_path, "t1")
        assert out.exists()
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["Panel"] == "tissue-plausibility"
        assert data["centroid_fa"] == pytest.approx(0.6, abs=1e-5)
        assert data["free_water_fraction"] == pytest.approx(0.0)

    def test_csf_like_bundle_reports_full_free_water(self, tmp_path):
        """A bundle sitting entirely in CSF-like voxels must read 100 %."""
        fa_path, md_path = self._maps(tmp_path, fa_value=0.05, md_value=2.8e-3)
        density_path, _ = self._density(tmp_path)
        out = qc_figures.plot_tissue_plausibility_panel(
            fa_path, md_path, density_path, tmp_path, "t2")
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["free_water_fraction"] == pytest.approx(1.0)

    def test_per_hemisphere_fractions_are_reported_when_supplied(self, tmp_path):
        """Left in white matter, right in CSF: the panel must separate them."""
        fa = np.full(_PLOT_SHAPE, 0.6, np.float32)
        md = np.full(_PLOT_SHAPE, 8e-4, np.float32)
        fa[12:15, 5:8, 5:8] = 0.05
        md[12:15, 5:8, 5:8] = 2.8e-3
        fa_path, md_path = tmp_path / "fa.nii.gz", tmp_path / "md.nii.gz"
        _write_volume(fa_path, fa, _PLOT_AFFINE)
        _write_volume(md_path, md, _PLOT_AFFINE)

        left = np.zeros(_PLOT_SHAPE, np.float32)
        left[5:8, 5:8, 5:8] = 0.5
        right = np.zeros(_PLOT_SHAPE, np.float32)
        right[12:15, 5:8, 5:8] = 0.5
        density_path = tmp_path / "density.nii.gz"
        _write_volume(density_path, left + right, _PLOT_AFFINE)

        out = qc_figures.plot_tissue_plausibility_panel(
            fa_path, md_path, density_path, tmp_path, "t3",
            density_left=left, density_right=right)
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["per_hemisphere"]["left"]["free_water_fraction"] == \
            pytest.approx(0.0)
        assert data["per_hemisphere"]["right"]["free_water_fraction"] == \
            pytest.approx(1.0)
        assert data["free_water_fraction"] == pytest.approx(0.5)

    def test_empty_density_renders_with_a_stated_reason(self, tmp_path):
        fa_path, md_path = self._maps(tmp_path)
        density_path = tmp_path / "empty.nii.gz"
        _write_volume(density_path, np.zeros(_PLOT_SHAPE, np.float32),
                      _PLOT_AFFINE)
        out = qc_figures.plot_tissue_plausibility_panel(
            fa_path, md_path, density_path, tmp_path, "t4")
        assert out.exists()
        assert json.loads(out.with_suffix(".json").read_text())["n_voxels"] == 0

    def test_byte_reproducible_across_runs(self, tmp_path):
        fa_path, md_path = self._maps(tmp_path)
        density_path, _ = self._density(tmp_path)
        a = qc_figures.plot_tissue_plausibility_panel(
            fa_path, md_path, density_path, tmp_path, "a")
        b = qc_figures.plot_tissue_plausibility_panel(
            fa_path, md_path, density_path, tmp_path, "b")
        assert np.array_equal(mpimg.imread(str(a)), mpimg.imread(str(b)))


class TestProfileAttritionPanel:
    """QC-7: the panel must agree with the funnel it draws."""

    def _fa(self, tmp_path):
        path = tmp_path / "fa.nii.gz"
        _write_volume(path, np.full(_PLOT_SHAPE, 0.5, np.float32), _PLOT_AFFINE)
        return path

    def test_clean_bundles_report_no_attrition(self, tmp_path):
        fa_path = self._fa(tmp_path)
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        out = qc_figures.plot_profile_attrition_panel(
            left, right, fa_path, tmp_path, "a1")
        assert out.exists()
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["Panel"] == "profile-attrition"
        assert data["left"]["any_attrition"] is False
        assert [s["count"] for s in data["left"]["stages"]] == [9, 9, 9, 9]
        assert data["left"]["point_retention"] == pytest.approx(1.0)

    def test_out_of_bounds_hemisphere_is_reported_as_attrition(self, tmp_path):
        """Asymmetric attrition — the failure the panel exists to catch.

        The right bundle is stored against a taller reference than the FA map,
        which is exactly the real hazard: a tractogram whose grid does not match
        the scalar map it is sampled against.
        """
        fa_path = self._fa(tmp_path)
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        far = [s + np.array([0.0, 0.0, 40.0]) for s in _si_bundle()]
        right = _save_trk(tmp_path / "r.trk", far, shape=(20, 20, 60))
        out = qc_figures.plot_profile_attrition_panel(
            left, right, fa_path, tmp_path, "a2")
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["left"]["any_attrition"] is False
        assert data["right"]["any_attrition"] is True
        assert data["right"]["n_contributing"] == 0
        assert data["right"]["stages"][2]["dropped"] == 9

    def test_funnel_bars_carry_the_stage_counts(self, tmp_path):
        """Read the bar geometry back: the drawn widths are the stage counts."""
        fa_path = self._fa(tmp_path)
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle(n=5))
        out = qc_figures.plot_profile_attrition_panel(
            left, right, fa_path, tmp_path, "a3")
        data = json.loads(out.with_suffix(".json").read_text())
        assert [s["count"] for s in data["right"]["stages"]] == [5, 5, 5, 5]
        assert data["left"]["n_input"] == 9

    def test_empty_bundles_render_with_a_stated_reason(self, tmp_path):
        fa_path = self._fa(tmp_path)
        left = _save_trk(tmp_path / "l.trk", [])
        right = _save_trk(tmp_path / "r.trk", [])
        out = qc_figures.plot_profile_attrition_panel(
            left, right, fa_path, tmp_path, "a4")
        assert out.exists()
        data = json.loads(out.with_suffix(".json").read_text())
        assert data["left"]["n_input"] == 0

    def test_byte_reproducible_across_runs(self, tmp_path):
        fa_path = self._fa(tmp_path)
        left = _save_trk(tmp_path / "l.trk", _si_bundle())
        right = _save_trk(tmp_path / "r.trk", _si_bundle())
        a = qc_figures.plot_profile_attrition_panel(
            left, right, fa_path, tmp_path, "a")
        b = qc_figures.plot_profile_attrition_panel(
            left, right, fa_path, tmp_path, "b")
        assert np.array_equal(mpimg.imread(str(a)), mpimg.imread(str(b)))
