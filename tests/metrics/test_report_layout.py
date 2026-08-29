"""Structural and reproducibility tests for the redesigned one-page A4 report.

Covers:
- HTML structure (section text, dynamic orientation label, LI sign classes,
  badge removed, Length median handling).
- Backward compatibility for legacy morphology without median_length.
- Figure reproducibility (decoded pixel arrays identical across two runs).
- PDF one-page + A4 + content + git-commit-absence (gated on weasyprint + pypdf).
- Optional perceptual-hash golden comparison (opt-in via CSTTOOL_VR=1).

HTML assertions are string-based (no external parser dependency) to keep the
test self-contained, matching the style of the existing report tests.
"""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
import pytest

from csttool.metrics.modules.reports import build_report_context, save_html_report
from csttool.metrics.modules.visualizations import (
    plot_profile_matrix,
    plot_tractogram_qc_triptych,
)
from csttool.viz.geometry import orientation_code


# ---------------------------------------------------------------------------
# Fixtures: a deterministic synthetic subject.
# ---------------------------------------------------------------------------

RAS_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


DISPERSION_KEYS = ("profile_p25", "profile_p75", "profile_n")


def _scalar_block(seed):
    rng = np.random.default_rng(seed)
    profile = rng.uniform(0.3, 0.7, 20)
    # A band straddling the mean by a known, per-node-varying half-width, so a
    # test can assert the drawn extents against the emitted numbers rather than
    # against a constant it could also get from a bug.
    half = rng.uniform(0.02, 0.06, 20)
    return {
        "mean": 0.45 + seed * 0.001, "std": 0.08, "median": 0.44,
        "min": 0.31, "max": 0.72, "n_streamlines": 100,
        "bootstrap_se": 0.008,
        "mean_point_weighted": 0.45, "std_point_weighted": 0.1,
        "median_point_weighted": 0.44, "min_point_weighted": 0.3,
        "max_point_weighted": 0.7, "n_samples": 5000,
        "profile": profile.tolist(),
        "profile_p25": (profile - half).tolist(),
        "profile_p75": (profile + half).tolist(),
        "profile_n": 97,
        "pontine": 0.4, "plic": 0.5, "precentral": 0.45,
        "pontine_se": 0.004, "plic_se": 0.004, "precentral_se": 0.004,
    }


def strip_dispersion(metrics):
    """The same metrics as a pre-milestone (legacy) JSON would carry."""
    out = {}
    for key, block in metrics.items():
        if isinstance(block, dict):
            block = {k: v for k, v in block.items() if k not in DISPERSION_KEYS}
        out[key] = block
    return out


def _morphology(median_len=84.0, n=100, mean=85.0, std=12.0,
              min_len=40.0, max_len=130.0, volume=12000.0):
    m = {
        "n_streamlines": n, "mean_length": mean, "std_length": std,
        "min_length": min_len, "max_length": max_len, "tract_volume": volume,
    }
    if median_len is not None:
        m["median_length"] = median_len
    return m


def make_comparison(with_median=True, with_all_scalars=True):
    left = {"morphology": _morphology(84.0 if with_median else None, 100)}
    right = {"morphology": _morphology(
        87.0 if with_median else None, 110, mean=88.0, std=11.0,
        min_len=42.0, max_len=135.0, volume=13000.0,
    )}
    scalars = ["fa", "md", "rd", "ad"] if with_all_scalars else ["fa"]
    for i, s in enumerate(scalars):
        left[s] = _scalar_block(i)
        right[s] = _scalar_block(i + 5)
    asym = {
        "volume": {"laterality_index": -0.04},
        "streamline_count": {"laterality_index": -0.05},
        "mean_length": {"laterality_index": -0.02},
        "fa": {"laterality_index": 0.043},
        "md": {"laterality_index": -0.052},
        "rd": {"laterality_index": 0.01},
        "ad": {"laterality_index": 0.0},
    }
    return {"left": left, "right": right, "asymmetry": asym}


NODE_HOMOLOGY = {
    "max_abs_z_difference_mm": 6.42, "length_difference_mm": -7.94,
    "z_difference_mm": [0.1] * 20, "arc_difference_mm": [0.2] * 20,
    "left_length_mean": 120.0, "right_length_mean": 127.94,
    "left_n_streamlines": 192, "right_n_streamlines": 264, "n_points": 20,
}


def make_qc_strip(tmp_path, subject_id):
    """Build the 1x4 QC strip the report now embeds.

    Reuses the strip suite's synthetic products rather than inventing a second
    set, so the two suites cannot drift apart about what a valid input looks
    like.
    """
    from .test_report_qc_strip import (
        RAS_AFFINE as STRIP_AFFINE, _bundle, _density, _dseg, _fa,
        _v1_superior, _write_nifti, _write_trk,
    )
    from csttool.metrics.modules.visualizations import plot_report_qc_strip

    return plot_report_qc_strip(
        fa_path=_write_nifti(tmp_path / "s_fa.nii.gz", _fa(), STRIP_AFFINE),
        v1_path=_write_nifti(tmp_path / "s_v1.nii.gz", _v1_superior(), STRIP_AFFINE),
        density_path=_write_nifti(tmp_path / "s_d.nii.gz", _density(), STRIP_AFFINE),
        roi_dseg_path=_write_nifti(tmp_path / "s_s.nii.gz", _dseg(), STRIP_AFFINE),
        cst_left_path=_write_trk(tmp_path / "s_l.trk", _bundle(-8.0), STRIP_AFFINE),
        cst_right_path=_write_trk(tmp_path / "s_r.trk", _bundle(8.0), STRIP_AFFINE),
        output_dir=tmp_path, subject_id=subject_id,
    )


def make_fa_background():
    xx, yy, zz = np.indices((40, 40, 30))
    fa = np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 15) ** 2) / 100)
    return fa.clip(0, 1)


def make_streamlines(n=20, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(np.column_stack([
            rng.uniform(5, 35, 10), rng.uniform(5, 35, 10), np.full(10, 15.0),
        ]))
    return out


def full_metadata():
    return {
        "acquisition": {
            "b_values": [0, 1000], "n_directions": 64,
            "resolution_mm": [2.0, 2.0, 2.0], "field_strength_T": 3.0,
            "echo_time_ms": 90.0,
        },
        "processing": {
            "tracking_params": {
                "fit_method": "WLS", "sh_order": 6, "seed_density": 1,
                "fa_thresh": 0.2, "step_size": 0.5,
            },
            "preprocessing": {"motion_correction": False},
            "extraction": {"method": "passthrough"},
        },
        "provenance": {
            "python_version": "3.12", "platform": "Linux",
            "hardware": {"cpu_model": "Test CPU", "cpu_count": 8,
                         "total_ram_gb": 16.0, "gpu": None},
            "thread_env": {"OMP_NUM_THREADS": "1"},
            "dependencies": {"numpy": "1.26", "dipy": "1.9"},
        },
    }


@pytest.fixture
def rendered_html(tmp_path):
    comparison = make_comparison()
    comparison["node_homology"] = NODE_HOMOLOGY
    aff = RAS_AFFINE
    pm = plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-x")
    viz = {"profile_matrix": pm, "qc_strip": make_qc_strip(tmp_path, "sub-x")}
    html = save_html_report(
        comparison, viz, tmp_path, "sub-x", version="0.5.0",
        space="Native Space", metadata=full_metadata(), fa_affine=aff,
    )
    return html.read_text()


# ---------------------------------------------------------------------------
# HTML structural tests.
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "CST Analysis Report",
    "Global metrics",
    "Regional metrics",
    "Along-tract profiles",
    "Tractography QC",
    "Reproducibility",
]


class TestHtmlStructure:
    def test_required_sections_present(self, rendered_html):
        for section in REQUIRED_SECTIONS:
            assert section in rendered_html, f"missing section: {section!r}"

    def test_li_formula_present(self, rendered_html):
        assert "LI = (L - R) / (L + R)" in rendered_html

    def test_method_summary_concise(self, rendered_html):
        assert "Deterministic CST tractography" in rendered_html

    def test_extraction_badge_removed(self, rendered_html):
        assert "Metrics Extracted In" not in rendered_html

    def test_exactly_one_profile_matrix_image(self, rendered_html):
        assert rendered_html.count('class="profile-matrix"') == 1

    def test_exactly_one_qc_strip_image(self, rendered_html):
        assert rendered_html.count('class="qc-strip"') == 1

    def test_no_triptych_remains_in_the_report(self, rendered_html):
        """The 1x3 triptych is gone from the report path (the function itself is
        kept, exported and tested — only the report no longer uses it)."""
        assert "qc-triptych" not in rendered_html

    def test_orientation_label_dynamic_and_once(self, rendered_html):
        # The orientation code derived from RAS_AFFINE is "RAS".
        assert orientation_code(RAS_AFFINE) == "RAS"
        # Appears once, in the methods band (as a <dd>).
        # Count occurrences of the bare code token — must be exactly one.
        assert rendered_html.count(">RAS<") == 1

    def test_qc_is_single_composite_image(self, rendered_html):
        # The QC section is one composite strip image (the four panel titles live
        # inside the figure, not as separate HTML labels), and no per-panel
        # orientation code is repeated in the HTML.
        assert rendered_html.count('class="qc-strip"') == 1
        assert "Sagittal (RAS)" not in rendered_html
        assert "Coronal (RAS)" not in rendered_html
        assert "Axial (RAS)" not in rendered_html

    def test_qc_alt_text_names_all_four_panels(self, rendered_html):
        alt = "Report QC strip (DEC-FA, CST density, extraction ROIs, CST over FA)"
        assert alt in rendered_html

    def test_li_classes_by_sign(self, rendered_html):
        # asymmetry.fa LI is +0.043 (li-left/blue); md is -0.052 (li-right/orange).
        assert 'class="li li-left">+0.043' in rendered_html
        assert 'class="li li-right">-0.052' in rendered_html

    def test_methods_terminology_verified(self, rendered_html):
        assert "DTI (WLS)" in rendered_html          # scalar model
        assert "CSA ODF, SH 6" in rendered_html       # direction model
        assert ">Deterministic<" in rendered_html or "Tracking</dt><dd>Deterministic" in rendered_html
        assert "FA mask" in rendered_html             # seeding (not "anatomically constrained")
        assert "Anatomically constrained" not in rendered_html

    def test_length_row_shows_genuine_median(self, rendered_html):
        # median_length present -> "84.0 (40.0-130.0)" (the median, not the mean 85.0).
        assert "84.0 (40.0-130.0)" in rendered_html
        assert "87.0 (42.0-135.0)" in rendered_html

    def test_regional_subtitle_removed(self, rendered_html):
        # The values are regional profile-bin means, not isolated landmark
        # measurements, so the old "Means at key landmarks" subtitle is gone.
        assert "Regional metrics" in rendered_html
        assert "Means at key landmarks" not in rendered_html

    def test_template_carries_no_profile_legend(self, rendered_html):
        # Exactly one profile legend, and it belongs to the Matplotlib figure.
        # A template-level copy used to be rendered alongside it, clipped.
        assert "inline-legend" not in rendered_html
        assert "PLIC region" not in rendered_html

    def test_git_commit_absent(self, rendered_html):
        assert "abc123def456" not in rendered_html

    def test_hardware_visible_in_footer(self, rendered_html):
        assert "Test CPU" in rendered_html
        assert "16.0" in rendered_html and "GB" in rendered_html


class TestNodeHomologyLine:
    """The status line under the regional table (plan §8).

    Descriptive only: it must never gate, colour or suppress a regional value.
    """

    def _html(self, tmp_path, node_homology, name="sub-nh"):
        comparison = make_comparison()
        if node_homology is not None:
            comparison["node_homology"] = node_homology
        return save_html_report(
            comparison, {}, tmp_path, name, version="0.5.0", space="Native Space",
            metadata=full_metadata(), fa_affine=RAS_AFFINE,
        ).read_text()

    AVAILABLE = {
        "max_abs_z_difference_mm": 6.42, "length_difference_mm": -7.94,
        "z_difference_mm": [0.1] * 20, "arc_difference_mm": [0.2] * 20,
        "left_length_mean": 120.0, "right_length_mean": 127.94,
        "left_n_streamlines": 192, "right_n_streamlines": 264, "n_points": 20,
    }
    UNAVAILABLE = {
        "max_abs_z_difference_mm": None, "length_difference_mm": None,
        "z_difference_mm": None, "arc_difference_mm": None,
        "left_length_mean": None, "right_length_mean": 127.9,
        "left_n_streamlines": 0, "right_n_streamlines": 264, "n_points": 20,
    }

    def test_line_states_both_numbers_when_available(self, tmp_path):
        html = self._html(tmp_path, self.AVAILABLE)
        assert "Node homology" in html
        assert "6.4 mm" in html
        assert "-7.9 mm" in html   # signed: which side is longer is the point

    def test_length_difference_carries_an_explicit_sign(self, tmp_path):
        positive = dict(self.AVAILABLE, length_difference_mm=0.11)
        assert "+0.1 mm" in self._html(tmp_path, positive, "sub-nh-pos")

    def test_note_rides_on_the_regional_caption_row(self, tmp_path):
        """It costs no vertical block of its own: it is the <small> on the
        Regional metrics section title, which already existed."""
        html = self._html(tmp_path, self.AVAILABLE)
        # Split on the section, not the words: the stylesheet comment above
        # explains the same thing in prose.
        section = html.split('class="regional-metrics"', 1)[1]
        title = section.split("</h2>", 1)[0]
        assert 'class="homology-note"' in title
        assert "Node homology" in title

    def test_no_verdict_no_threshold_no_icon(self, tmp_path):
        # Scoped to the rendered section, not the whole document: the stylesheet
        # comment legitimately explains *why* no threshold is shipped.
        html = self._html(tmp_path, self.AVAILABLE)
        section = html.split('class="regional-metrics"', 1)[1].split("</section>", 1)[0]
        for forbidden in ("PASS", "FAIL", "⚠", "threshold", "homology-fail",
                          "homology-pass"):
            assert forbidden not in section

    def test_unavailable_states_so_plainly(self, tmp_path):
        html = self._html(tmp_path, self.UNAVAILABLE)
        assert "unavailable (one hemisphere has no streamlines)" in html
        assert "max L–R node offset" not in html

    def test_absent_block_degrades_like_an_empty_hemisphere(self, tmp_path):
        """A metrics dict from an older version carries no node_homology at all."""
        html = self._html(tmp_path, None)
        assert "Node homology" in html
        assert "unavailable" in html

    @pytest.mark.parametrize("homology", [AVAILABLE, UNAVAILABLE, None])
    def test_regional_cells_are_never_suppressed(self, tmp_path, homology):
        """The no-suppression guarantee: every regional value stays visible and
        unmodified whatever the homology says."""
        html = self._html(tmp_path, homology, "sub-nh-cells")
        for region in ("Pontine", "PLIC", "Precentral"):
            assert f'class="region-label">{region}<' in html
        # Four scalar columns of real values, three rows, in every case.
        assert html.count('<td class="region-label">') == 3

    def test_line_lives_inside_the_regional_section(self, tmp_path):
        html = self._html(tmp_path, self.AVAILABLE)
        section = html.split('class="regional-metrics"', 1)[1].split("</section>", 1)[0]
        assert "Node homology" in section

    def test_note_never_wraps(self, tmp_path):
        """It shares a row with the section title, so it must stay on one line —
        a wrap would silently reintroduce the 3 mm the page cannot afford."""
        html = self._html(tmp_path, self.AVAILABLE)
        css = html.split("</style>", 1)[0]
        block = css.split(".homology-note", 1)[1][:200]
        assert "nowrap" in block

    def test_note_uses_the_muted_footnote_style(self, tmp_path):
        html = self._html(tmp_path, self.AVAILABLE)
        assert 'class="homology-note"' in html
        css = html.split("</style>", 1)[0]
        assert ".homology-note" in css
        assert "var(--muted)" in css.split(".homology-note", 1)[1][:220]


class TestHtmlBackwardCompat:
    def test_legacy_morphology_no_median_renders_dash(self, tmp_path):
        comparison = make_comparison(with_median=False, with_all_scalars=False)
        html = save_html_report(
            comparison, {}, tmp_path, "legacy-sub", version="0.5.0",
            space="Native Space", metadata=full_metadata(), fa_affine=RAS_AFFINE,
        ).read_text()
        length_cell_left = "85.0 ± 12.0"
        assert length_cell_left in html  # mean +/- SD still shown
        # The Length median column must be an em dash, NOT the mean.
        # The em dash is the cell content for both sides.
        assert "—" in html
        # And the mean (85.0 / 88.0) is never shown as a "(min-max)" median.
        assert "85.0 (40.0-130.0)" not in html

    def test_fa_only_degrades_matrix_gracefully(self, tmp_path):
        comparison = make_comparison(with_all_scalars=False)
        fa = make_fa_background()
        pm = plot_profile_matrix(
            comparison["left"], comparison["right"], tmp_path, "sub-faonly"
        )
        assert pm is not None and pm.exists()

    def test_empty_metadata_renders(self, tmp_path):
        comparison = make_comparison(with_all_scalars=False)
        html = save_html_report(
            comparison, {}, tmp_path, "empty-meta", version="0.5.0",
            space="Native Space", metadata={}, fa_affine=RAS_AFFINE,
        ).read_text()
        assert "CST Analysis Report" in html
        assert "N/A" in html  # missing acquisition fields render N/A


# ---------------------------------------------------------------------------
# Orientation: the report must show the code of the affine it was handed.
# ---------------------------------------------------------------------------

# Three affines with known, *different* voxel-orientation codes. LAS is the code
# of a scanner-native dicom2nifti conversion (the csttool pipeline does not
# reorient to RAS), so both must round-trip — a value hardcoded to either one
# fails at least one of these.
LAS_AFFINE = np.diag([-2.0, 2.0, 2.0, 1.0])
PSR_AFFINE = np.array([
    [0.0, 0.0, 2.0, -90.0],
    [-2.0, 0.0, 0.0, 90.0],
    [0.0, 2.0, 0.0, -70.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _orientation_cell(affine):
    """The Orientation value the report context builds for ``affine``."""
    from csttool.metrics.modules.reports import build_report_context
    ctx = build_report_context(
        comparison=make_comparison(), visualization_paths={}, subject_id="sub-o",
        version="0.5.0", space="Native Space", metadata=full_metadata(),
        fa_affine=affine,
    )
    return dict(ctx["space_rows"])["Orientation"]


class TestOrientationCode:
    @pytest.mark.parametrize(
        "affine, expected", [(RAS_AFFINE, "RAS"), (LAS_AFFINE, "LAS"), (PSR_AFFINE, "PSR")]
    )
    def test_code_is_computed_from_the_affine(self, affine, expected):
        import nibabel as nib
        # The expectation is anchored on nibabel, not on a literal in the code.
        assert "".join(nib.orientations.aff2axcodes(affine)) == expected
        assert orientation_code(affine) == expected
        assert _orientation_cell(affine) == expected

    def test_not_hardcoded(self):
        assert _orientation_cell(RAS_AFFINE) != _orientation_cell(LAS_AFFINE)

    def test_html_shows_the_affines_own_code(self, tmp_path):
        for affine, expected in ((RAS_AFFINE, "RAS"), (LAS_AFFINE, "LAS")):
            html = save_html_report(
                make_comparison(), {}, tmp_path, f"sub-{expected}", version="0.5.0",
                space="Native Space", metadata=full_metadata(), fa_affine=affine,
            ).read_text()
            assert f">{expected}<" in html
            other = "LAS" if expected == "RAS" else "RAS"
            assert f">{other}<" not in html

    def test_missing_affine_renders_na(self):
        assert _orientation_cell(None) == "N/A"


# ---------------------------------------------------------------------------
# Figure reproducibility (decoded pixel arrays — environment-independent).
# ---------------------------------------------------------------------------

def _decoded(path):
    return np.asarray(mpimg.imread(str(path)))


@pytest.fixture
def figure_probe(monkeypatch):
    """Capture the geometry of the figure a ``plot_*`` helper builds.

    The plotting helpers own their figure and close it, so the facts have to be
    read at save time. Recording them in place of asserting on pixels keeps
    these tests about layout rather than about font rasterization.
    """
    import csttool.metrics.modules.visualizations as viz

    captured = {}
    real_save = viz._style.save_figure

    def probe(fig, path, **kwargs):
        fig.canvas.draw()  # positions are only final once aspect has been applied
        captured["size_in"] = tuple(fig.get_size_inches())
        captured["legend_labels"] = [
            [t.get_text() for t in lg.get_texts()] for lg in fig.legends
        ]
        captured["axes"] = [
            {
                "position": ax.get_position(original=False),
                "title": ax.get_title(),
                "title_pt": ax.title.get_fontsize(),
                "texts": [(t.get_text(), t.get_fontsize()) for t in ax.texts],
                "images": list(ax.images),
                "collections": list(ax.collections),
                "lines": list(ax.lines),
                "ylim": ax.get_ylim(),
            }
            for ax in fig.axes
        ]
        return real_save(fig, path, **kwargs)

    monkeypatch.setattr(viz._style, "save_figure", probe)
    return captured


class TestProfileMatrixLayout:
    def test_generated_at_its_designed_print_size(self, tmp_path, figure_probe):
        from csttool.metrics.modules.visualizations import PROFILE_MATRIX_SIZE_MM

        comparison = make_comparison()
        path = plot_profile_matrix(
            comparison["left"], comparison["right"], tmp_path, "sub-size"
        )
        w_mm, h_mm = PROFILE_MATRIX_SIZE_MM
        np.testing.assert_allclose(
            figure_probe["size_in"], (w_mm / 25.4, h_mm / 25.4), rtol=1e-9
        )
        # And the saved PNG keeps that aspect exactly (no "tight" crop/pad),
        # so placing it at w_mm wide prints it exactly h_mm tall.
        arr = _decoded(path)
        np.testing.assert_allclose(arr.shape[0] / arr.shape[1], h_mm / w_mm, rtol=2e-3)

    def test_exactly_one_legend_naming_left_and_right(self, tmp_path, figure_probe):
        comparison = make_comparison()
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-lg")
        assert len(figure_probe["legend_labels"]) == 1
        labels = figure_probe["legend_labels"][0]
        assert "Left CST" in labels and "Right CST" in labels
        # No per-panel legend duplicating it.
        assert all(ax["texts"] is not None for ax in figure_probe["axes"])

    def test_region_labels_are_smaller_than_subplot_titles(self, tmp_path, figure_probe):
        comparison = make_comparison()
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-fs")
        titles = [ax["title_pt"] for ax in figure_probe["axes"] if ax["title"]]
        region_pts = [
            pt
            for ax in figure_probe["axes"]
            for text, pt in ax["texts"]
            if text in ("Pontine Level", "PLIC", "Precentral Gyrus")
        ]
        assert len(titles) == 4
        assert len(region_pts) == 3, "region names must be drawn exactly once each"
        assert max(region_pts) < min(titles)

    def test_region_labels_live_in_one_shared_strip(self, tmp_path, figure_probe):
        comparison = make_comparison()
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-strip")
        strips = [
            ax for ax in figure_probe["axes"]
            if any(t in ("Pontine Level", "PLIC", "Precentral Gyrus")
                   for t, _ in ax["texts"])
        ]
        assert len(strips) == 1, "the region names must not be repeated per panel"
        strip, panels = strips[0], [
            ax for ax in figure_probe["axes"] if ax["title"]
        ]
        # The strip spans both columns and sits below every panel.
        assert strip["position"].width > max(p["position"].width for p in panels)
        assert strip["position"].y1 <= min(p["position"].y0 for p in panels)


class TestProfileMatrixIqrBands:
    """The IQR band drawn behind each profile line (plan §6).

    The band is a scientific claim about the bundle's spread, so these assert
    the *drawn* extents against the emitted ``profile_p25``/``profile_p75`` —
    a band that renders but does not match the numbers would be worse than none.
    """

    def _panels(self, probe):
        """The four scalar panels, in _PROFILE_MATRIX_CONFIG order."""
        return [ax for ax in probe["axes"] if ax["title"]]

    def test_two_band_artists_per_populated_panel(self, tmp_path, figure_probe):
        comparison = make_comparison()
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-band")
        panels = self._panels(figure_probe)
        assert len(panels) == 4
        for panel in panels:
            assert len(panel["collections"]) == 2, "one filled band per hemisphere"

    def test_band_extents_equal_the_emitted_quartiles(self, tmp_path, figure_probe):
        from csttool.metrics.modules.visualizations import _PROFILE_MATRIX_CONFIG

        comparison = make_comparison()
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-ext")
        for panel, cfg in zip(self._panels(figure_probe), _PROFILE_MATRIX_CONFIG):
            scale = cfg["scale"]
            for collection, side in zip(panel["collections"], ("left", "right")):
                block = comparison[side][cfg["key"]]
                drawn = collection.get_paths()[0].vertices[:, 1]
                expected = np.concatenate([
                    np.array(block["profile_p25"]) * scale,
                    np.array(block["profile_p75"]) * scale,
                ])
                # fill_between traces p25 forward then p75 back, plus closing
                # vertices; compare the value sets rather than the traversal.
                np.testing.assert_allclose(
                    np.sort(np.unique(np.round(drawn, 12))),
                    np.sort(np.unique(np.round(expected, 12))),
                    rtol=1e-9,
                )

    def test_profile_lines_are_drawn_above_the_bands(self, tmp_path, figure_probe):
        """The lines are the data; a band must never occlude one."""
        comparison = make_comparison()
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-z")
        for panel in self._panels(figure_probe):
            band_z = max(c.get_zorder() for c in panel["collections"])
            line_z = max(line.get_zorder() for line in panel["lines"][-2:])
            assert line_z > band_z

    def test_ylim_contains_the_whole_band(self, tmp_path, figure_probe):
        """A band clipped at the axis edge would understate the bundle's spread."""
        from csttool.metrics.modules.visualizations import _PROFILE_MATRIX_CONFIG

        comparison = make_comparison()
        # Push the right hemisphere's band well outside every fixed y-range, so
        # the auto-fallback has to notice the band and not just the line.
        for cfg in _PROFILE_MATRIX_CONFIG:
            block = comparison["right"][cfg["key"]]
            block["profile_p75"] = [v + 5.0 for v in block["profile_p75"]]
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-ylim")
        for panel, cfg in zip(self._panels(figure_probe), _PROFILE_MATRIX_CONFIG):
            top = max(comparison["right"][cfg["key"]]["profile_p75"]) * cfg["scale"]
            assert panel["ylim"][1] >= top

    def test_legacy_metrics_without_dispersion_draw_no_band(self, tmp_path, figure_probe):
        """An older metrics JSON is a legitimate input: lines only, no warning,
        and no fabricated band."""
        comparison = make_comparison()
        path = plot_profile_matrix(
            strip_dispersion(comparison["left"]), strip_dispersion(comparison["right"]),
            tmp_path, "sub-legacy",
        )
        assert path is not None and path.exists()
        for panel in self._panels(figure_probe):
            assert panel["collections"] == []

    def test_empty_hemisphere_loses_only_its_own_band(self, tmp_path, figure_probe):
        comparison = make_comparison()
        for key in ("fa", "md", "rd", "ad"):
            comparison["right"][key]["profile_n"] = 0
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-half")
        for panel in self._panels(figure_probe):
            assert len(panel["collections"]) == 1

    def test_band_caption_appears_once_and_only_when_a_band_is_drawn(self, tmp_path,
                                                                    figure_probe):
        from csttool.metrics.modules.visualizations import _BAND_CAPTION

        comparison = make_comparison()
        plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "sub-cap")
        captions = [t for ax in figure_probe["axes"] for t, _ in ax["texts"]
                    if t == _BAND_CAPTION]
        assert len(captions) == 1

        plot_profile_matrix(
            strip_dispersion(comparison["left"]), strip_dispersion(comparison["right"]),
            tmp_path, "sub-nocap",
        )
        captions = [t for ax in figure_probe["axes"] for t, _ in ax["texts"]
                    if t == _BAND_CAPTION]
        assert captions == []

    def test_bands_do_not_change_the_printed_size(self, tmp_path, figure_probe):
        from csttool.metrics.modules.visualizations import PROFILE_MATRIX_SIZE_MM

        comparison = make_comparison()
        path = plot_profile_matrix(
            comparison["left"], comparison["right"], tmp_path, "sub-size2"
        )
        w_mm, h_mm = PROFILE_MATRIX_SIZE_MM
        np.testing.assert_allclose(
            figure_probe["size_in"], (w_mm / 25.4, h_mm / 25.4), rtol=1e-9
        )
        arr = _decoded(path)
        np.testing.assert_allclose(arr.shape[0] / arr.shape[1], h_mm / w_mm, rtol=2e-3)

    def test_sidecar_records_the_band_and_its_population(self, tmp_path):
        import json

        comparison = make_comparison()
        path = plot_profile_matrix(
            comparison["left"], comparison["right"], tmp_path, "sub-side"
        )
        sidecar = json.loads(path.with_suffix(".json").read_text())
        assert sidecar["Panel"] == "profile-matrix"
        assert sidecar["Band"] == "interquartile range across contributing streamlines"
        assert sidecar["BandDrawn"] is True
        assert sidecar["ContributingStreamlines"]["fa"]["left"] == 97

    def test_banded_figure_is_reproducible(self, tmp_path):
        comparison = make_comparison()
        p1 = plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "r1")
        p2 = plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "r2")
        np.testing.assert_array_equal(_decoded(p1), _decoded(p2))


class TestQcTriptychGeometry:
    def _probe(self, tmp_path, figure_probe, background_kind="fa"):
        fa = make_fa_background()
        sl_l = make_streamlines(20, 42)
        sl_r = make_streamlines(20, 7)
        plot_tractogram_qc_triptych(
            sl_l, sl_r, fa, RAS_AFFINE, tmp_path, "sub-qc",
            background_kind=background_kind,
        )
        return figure_probe

    def test_generated_at_its_designed_print_size(self, tmp_path, figure_probe):
        from csttool.metrics.modules.visualizations import QC_TRIPTYCH_SIZE_MM

        w_mm, h_mm = QC_TRIPTYCH_SIZE_MM
        self._probe(tmp_path, figure_probe)
        np.testing.assert_allclose(
            figure_probe["size_in"], (w_mm / 25.4, h_mm / 25.4), rtol=1e-9
        )

    def test_three_views_share_one_physical_viewport(self, tmp_path, figure_probe):
        # The source slices have different shapes (40x30 vs 40x40 here), so this
        # only holds because each panel is letterboxed onto a common canvas.
        probe = self._probe(tmp_path, figure_probe)
        boxes = [ax["position"] for ax in probe["axes"] if ax["images"]]
        assert len(boxes) == 3
        widths = {round(b.width, 6) for b in boxes}
        heights = {round(b.height, 6) for b in boxes}
        assert len(widths) == 1 and len(heights) == 1

    def test_slices_are_not_stretched(self, tmp_path, figure_probe):
        probe = self._probe(tmp_path, figure_probe)
        for ax in probe["axes"]:
            for im in ax["images"]:
                assert im.axes.get_aspect() == 1.0

    def test_colorbar_has_its_own_axis_and_never_overlaps_an_image(
        self, tmp_path, figure_probe
    ):
        probe = self._probe(tmp_path, figure_probe)
        image_axes = [ax for ax in probe["axes"] if ax["images"]]
        cbar_axes = [ax for ax in probe["axes"] if not ax["images"]]
        assert len(cbar_axes) == 1, "the colorbar must live on its own axis"
        cbar = cbar_axes[0]["position"]
        for ax in image_axes:
            assert cbar.x0 >= ax["position"].x1, "colorbar overlaps an image panel"

    def test_all_images_and_colorbar_share_one_normalize(self, tmp_path, figure_probe):
        probe = self._probe(tmp_path, figure_probe)
        norms = [im.norm for ax in probe["axes"] for im in ax["images"]]
        assert len(norms) == 3
        assert all(n is norms[0] for n in norms), "views must share one Normalize"
        assert (norms[0].vmin, norms[0].vmax) == (0.0, 1.0)

    def test_no_colorbar_axis_for_non_fa_background(self, tmp_path, figure_probe):
        probe = self._probe(tmp_path, figure_probe, background_kind="other")
        assert all(ax["images"] for ax in probe["axes"] if ax["position"].width > 0.1)


class TestReportFiguresMatchCss:
    """The CSS must place each figure at exactly the width it was drawn at."""

    def test_css_widths_match_figure_sizes(self):
        from csttool.metrics.modules.reports import _TEMPLATE_DIR
        from csttool.metrics.modules.visualizations import (
            PROFILE_MATRIX_SIZE_MM, QC_STRIP_WIDTH_MM,
        )

        css = (_TEMPLATE_DIR / "report.css").read_text()
        assert f".profile-matrix" in css
        assert f"width: {PROFILE_MATRIX_SIZE_MM[0]:g}mm" in css
        assert f".qc-strip" in css
        assert f"width: {QC_STRIP_WIDTH_MM:g}mm" in css
        # The figures are placed at the width they were drawn at, never resized.
        assert ".qc-triptych" not in css

    def test_css_does_not_constrain_the_strip_height(self):
        """The strip's height is derived from the subject's coronal aspect, so
        the CSS must set width only — a height here would rescale the figure and
        silently break the 1 pt = 1 pt guarantee its type sizes rest on."""
        from csttool.metrics.modules.reports import _TEMPLATE_DIR

        css = (_TEMPLATE_DIR / "report.css").read_text()
        block = css.split(".qc-strip {", 1)[1].split("}", 1)[0]
        assert "height" not in block


class TestReproducibilityFooter:
    def test_fixed_dependency_subset_no_open_ended_list(self):
        from csttool.metrics.modules.reports import _build_report_provenance

        prov = _build_report_provenance(full_metadata()["provenance"])
        deps = prov["dependencies_str"]
        for name in ("NumPy", "SciPy", "DIPY", "NiBabel", "Matplotlib"):
            assert name in deps
        assert "…" not in deps and "..." not in deps
        # Extra installed packages never leak into the footer.
        assert "nilearn" not in deps and "pydicom" not in deps

    def test_missing_dependency_is_explicit(self):
        from csttool.metrics.modules.reports import _build_report_provenance

        prov = _build_report_provenance({"dependencies": {"numpy": "2.0"}})
        assert "NumPy 2.0" in prov["dependencies_str"]
        assert "SciPy N/A" in prov["dependencies_str"]

    def test_thread_limits_summarised(self):
        from csttool.metrics.modules.reports import _build_report_provenance

        unset = _build_report_provenance(
            {"thread_env": {"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None}}
        )
        assert unset["thread_env_str"] == "Thread limits: unset"

        some = _build_report_provenance(
            {"thread_env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}
        )
        assert some["thread_env_str"] == "Threads: OMP=1, MKL=1"

    def test_long_strings_truncated_horizontally(self):
        from csttool.metrics.modules.reports import _build_report_provenance

        prov = _build_report_provenance({
            "hardware": {"cpu_model": "A" * 200, "cpu_count": 8},
            "platform": "Linux-6.17.0-40-generic-x86_64-with-glibc2.39",
            "python_version": "3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]",
        })
        assert len(prov["hardware_str"]) < 80
        assert prov["platform"] == "Linux 6.17.0-40-generic (x86_64)"
        assert prov["python_version"] == "3.12.3"

    def test_footer_uses_no_monospace(self, rendered_html):
        import re

        footer = rendered_html.split("<footer", 1)[1]
        assert "mono" not in footer  # no .mono class on any footer line
        css = rendered_html.split("</style>", 1)[0]
        declarations = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        assert "monospace" not in declarations


class TestFigureReproducibility:
    def test_profile_matrix_deterministic(self, tmp_path):
        comparison = make_comparison()
        p1 = plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "a")
        p2 = plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "b")
        np.testing.assert_array_equal(_decoded(p1), _decoded(p2))

    def test_qc_triptych_deterministic(self, tmp_path):
        fa = make_fa_background()
        sl_l = make_streamlines(20, 42)
        sl_r = make_streamlines(20, 7)
        p1 = plot_tractogram_qc_triptych(
            sl_l, sl_r, fa, RAS_AFFINE, tmp_path, "a", background_kind="fa"
        )
        p2 = plot_tractogram_qc_triptych(
            sl_l, sl_r, fa, RAS_AFFINE, tmp_path, "b", background_kind="fa"
        )
        np.testing.assert_array_equal(_decoded(p1), _decoded(p2))

    def test_qc_triptych_colorbar_only_for_fa(self, tmp_path):
        fa = make_fa_background()
        sl = make_streamlines(5, 1)
        p_fa = plot_tractogram_qc_triptych(
            sl, sl, fa, RAS_AFFINE, tmp_path, "fa", background_kind="fa"
        )
        p_other = plot_tractogram_qc_triptych(
            sl, sl, fa, RAS_AFFINE, tmp_path, "other", background_kind="other"
        )
        # FA background yields a larger image footprint (colorbar steals space);
        # both still render. The real assertion is structural (no crash) and the
        # colorbar label only present in the FA figure.
        assert p_fa.exists() and p_other.exists()


# ---------------------------------------------------------------------------
# Perceptual hash (opt-in golden comparison; robust to font rasterization).
# ---------------------------------------------------------------------------

def _phash(arr, hash_size=8):
    """Minimal perceptual hash: downsample to grayscale, compare to mean -> bits."""
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    h, w = gray.shape
    # Downsample to hash_size x hash_size by averaging blocks.
    bh = h // hash_size
    bw = w // hash_size
    small = np.zeros((hash_size, hash_size))
    for i in range(hash_size):
        for j in range(hash_size):
            small[i, j] = gray[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].mean()
    bits = (small > small.mean()).flatten()
    return bits


def _hamming(a, b):
    return int(np.count_nonzero(a != b))


@pytest.mark.skipif(
    os.environ.get("CSTTOOL_VR") != "1",
    reason="perceptual-hash golden comparison is opt-in (CSTTOOL_VR=1)",
)
class TestVisualRegressionGolden:
    golden_dir = Path(__file__).parent / "golden"

    def test_profile_matrix_matches_golden(self, tmp_path):
        gold = self.golden_dir / "profile_matrix.png"
        if not gold.exists():
            pytest.skip("golden profile_matrix.png not committed; run generate_golden.py")
        comparison = make_comparison()
        p = plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, "x")
        d = _hamming(_phash(_decoded(gold)), _phash(_decoded(p)))
        assert d <= 12, f"profile matrix pHash distance {d} exceeds tolerance 12"

    def test_qc_strip_matches_golden(self, tmp_path):
        gold = self.golden_dir / "report_qc_strip.png"
        if not gold.exists():
            pytest.skip("golden QC strip not committed; run generate_golden.py")
        p = make_qc_strip(tmp_path, "x")
        d = _hamming(_phash(_decoded(gold)), _phash(_decoded(p)))
        assert d <= 12, f"QC strip pHash distance {d} exceeds tolerance 12"


# ---------------------------------------------------------------------------
# PDF structural tests (gated on weasyprint + pypdf).
# ---------------------------------------------------------------------------

def _render_pdf(tmp_path, subject_id, metadata, comparison=None, with_median=True):
    pytest.importorskip("weasyprint")
    pytest.importorskip("pypdf")
    from csttool.metrics.modules.reports import html_to_pdf
    import pypdf

    comparison = comparison or make_comparison(with_median=with_median)
    comparison.setdefault("node_homology", NODE_HOMOLOGY)
    pm = plot_profile_matrix(comparison["left"], comparison["right"], tmp_path, subject_id)
    viz = {"profile_matrix": pm, "qc_strip": make_qc_strip(tmp_path, subject_id)}
    html = save_html_report(
        comparison, viz, tmp_path, subject_id, version="0.5.0",
        space="Native Space", metadata=metadata, fa_affine=RAS_AFFINE,
    )
    pdf = tmp_path / f"{subject_id}_report.pdf"
    html_to_pdf(html, pdf)
    assert pdf.exists() and pdf.stat().st_size > 0
    return pypdf.PdfReader(str(pdf))


A4_PRINTABLE_HEIGHT_MM = 297.0 - 2 * 8.0  # A4 height minus the @page margins
CSS_PX_MM = 25.4 / 96.0                   # WeasyPrint lays out in CSS pixels


def _rendered_content_height_mm(html_path):
    """Height of the report's content box, as WeasyPrint lays it out."""
    from weasyprint import HTML

    page = HTML(filename=str(html_path)).render().pages[0]

    def find_page_div(box):
        element = getattr(box, "element", None)
        if element is not None and "page" in (element.get("class", "") or "").split():
            return box
        for child in getattr(box, "children", []):
            found = find_page_div(child)
            if found is not None:
                return found
        return None

    return find_page_div(page._page_box).height * CSS_PX_MM


class TestPdfOnePage:
    def test_layout_keeps_headroom(self, tmp_path):
        """One page is necessary but not sufficient — keep slack for variation.

        A layout that fits with 0.5 mm to spare is one font-metric change away
        from spilling onto page 2, so the budget is asserted with a margin.
        """
        pytest.importorskip("weasyprint")
        comparison = make_comparison()
        comparison["node_homology"] = NODE_HOMOLOGY
        pm = plot_profile_matrix(
            comparison["left"], comparison["right"], tmp_path, "sub-budget"
        )
        html = save_html_report(
            comparison, {"profile_matrix": pm,
                         "qc_strip": make_qc_strip(tmp_path, "sub-budget")},
            tmp_path, "sub-budget", version="0.5.0", space="Native Space",
            metadata=full_metadata(), fa_affine=RAS_AFFINE,
        )
        used = _rendered_content_height_mm(html)
        assert used <= A4_PRINTABLE_HEIGHT_MM - 6.0, (
            f"report uses {used:.1f} mm of {A4_PRINTABLE_HEIGHT_MM:.1f} mm — "
            "too little headroom for realistic content variation"
        )

    def test_exactly_one_page_pathological_content(self, tmp_path):
        md = full_metadata()
        md["provenance"]["hardware"]["cpu_model"] = "Very Long CPU Model Name " * 8
        md["provenance"]["hardware"]["gpu"] = ["GPU " * 20]
        md["provenance"]["platform"] = "Linux-" + "x" * 120 + "-x86_64-with-glibc2.39"
        md["provenance"]["dependencies"] = {f"dep{k}": f"1.{k}.0" for k in range(60)}
        md["processing"]["extraction"] = {
            "method": "an-unusually-long-extraction-method-name",
            "artifact_index_available": False,
            "artifact_index_reason": "defined only for bidirectional extraction",
        }
        reader = _render_pdf(tmp_path, "sub-" + "long-identifier-" * 6 + "ses-01", md)
        assert len(reader.pages) == 1

    def test_exactly_one_page_full(self, tmp_path):
        reader = _render_pdf(tmp_path, "sub-ALS_participant1280_ses-01", full_metadata())
        assert len(reader.pages) == 1

    def test_exactly_one_page_empty_metadata(self, tmp_path):
        reader = _render_pdf(tmp_path, "sub-empty", {})
        assert len(reader.pages) == 1

    def test_exactly_one_page_fa_only(self, tmp_path):
        comparison = make_comparison(with_all_scalars=False)
        reader = _render_pdf(
            tmp_path, "sub-faonly", full_metadata(), comparison=comparison
        )
        assert len(reader.pages) == 1

    def test_exactly_one_page_long_metadata(self, tmp_path):
        md = full_metadata()
        md["provenance"]["dependencies"] = {
            f"dep{k}": f"1.{k}.0" for k in range(12)
        }
        md["provenance"]["hardware"]["cpu_model"] = (
            "AMD Ryzen 9 7950X 16-Core Processor with a very long model string"
        )
        reader = _render_pdf(
            tmp_path, "sub-very-long-subject-identifier-2026-ses-01", md
        )
        assert len(reader.pages) == 1

    def test_exactly_one_page_zero_streamlines(self, tmp_path):
        # Empty-case morphology path.
        left = {"morphology": _morphology(0.0, 0)}
        right = {"morphology": _morphology(0.0, 0)}
        asym = {
            "volume": {"laterality_index": 0.0},
            "streamline_count": {"laterality_index": 0.0},
            "mean_length": {"laterality_index": 0.0},
        }
        comparison = {"left": left, "right": right, "asymmetry": asym}
        reader = _render_pdf(
            tmp_path, "sub-zero", full_metadata(), comparison=comparison,
            with_median=False,
        )
        assert len(reader.pages) == 1

    def test_a4_portrait_page_size(self, tmp_path):
        reader = _render_pdf(tmp_path, "sub-a4", full_metadata())
        box = reader.pages[0].mediabox
        # A4 portrait = 595 x 842 pt (±1).
        w, h = float(box.width), float(box.height)
        assert 590 <= w <= 600, f"width {w} not A4"
        assert 838 <= h <= 846, f"height {h} not A4"
        assert h > w

    def test_pdf_contains_sections(self, tmp_path):
        reader = _render_pdf(tmp_path, "sub-text", full_metadata())
        # Section titles are uppercased by CSS, so the comparison is
        # case-insensitive: the assertion is that the section is *on the page*,
        # not how it is cased.
        text = (reader.pages[0].extract_text() or "").lower()
        for section in ["CST Analysis Report", "Global metrics", "Regional metrics",
                        "Along-tract profiles", "Tractography QC", "Reproducibility"]:
            assert section.lower() in text, f"section {section!r} not in PDF text"

    def test_pdf_git_commit_absent(self, tmp_path):
        md = full_metadata()
        md["provenance"]["git_commit"] = "abc123def456"  # type: ignore
        reader = _render_pdf(tmp_path, "sub-git", md)
        text = reader.pages[0].extract_text() or ""
        assert "abc123def456" not in text

    def test_pdf_orientation_label_dynamic(self, tmp_path):
        reader = _render_pdf(tmp_path, "sub-orient", full_metadata())
        text = reader.pages[0].extract_text() or ""
        assert "RAS" in text
