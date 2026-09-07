"""Tests for csttool.viz.utils (deterministic sampling) and viz.style (policy)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from csttool.viz import utils, style


class TestDeterministicSubsample:
    def test_returns_same_sample_across_calls(self):
        items = list(range(1000))
        a = utils.deterministic_subsample(items, 50)
        b = utils.deterministic_subsample(items, 50)
        assert a == b
        assert len(a) == 50

    def test_small_input_returned_unchanged(self):
        items = [1, 2, 3]
        assert utils.deterministic_subsample(items, 10) == [1, 2, 3]

    def test_result_is_sorted_subset(self):
        out = utils.deterministic_subsample(list(range(100)), 10)
        assert out == sorted(out)
        assert all(x in range(100) for x in out)

    def test_does_not_touch_global_numpy_state(self):
        np.random.seed(1234)
        before = np.random.get_state()[1].copy()
        utils.deterministic_subsample(list(range(1000)), 20)
        after = np.random.get_state()[1]
        assert np.array_equal(before, after)


class TestStylePolicy:
    def test_canonical_hemisphere_colours(self):
        assert style.LEFT == "#1f77b4"
        assert style.RIGHT == "#ff7f0e"
        # ROI motor colours reuse the hemisphere colours for consistency.
        assert style.MOTOR_LEFT == style.LEFT
        assert style.MOTOR_RIGHT == style.RIGHT

    def test_savefig_dpi_policy(self):
        assert style.RCPARAMS["savefig.dpi"] == 200
        assert style.RCPARAMS["figure.dpi"] == 150
        assert style.SAVEFIG_DPI == 200

    def test_apply_house_style_sets_rcparams(self):
        import matplotlib as mpl
        style.apply_house_style()
        assert mpl.rcParams["savefig.dpi"] == 200
        assert mpl.rcParams["axes.spines.top"] is False

    def test_add_scalar_colorbar_creates_colorbar_axes(self):
        fig, ax = plt.subplots()
        im = ax.imshow(np.random.rand(10, 10), cmap=style.MD_CMAP)
        n_axes_before = len(fig.axes)
        style.add_scalar_colorbar(fig, im, ax, "MD (x10^-3 mm^2/s)")
        assert len(fig.axes) == n_axes_before + 1  # colorbar axis added
        plt.close(fig)

    def test_save_figure_writes_nonempty_file(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "fig.png"
        style.save_figure(fig, out)
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)


class TestSingleExportPath:
    """Every figure csttool writes goes through ``style.save_figure``.

    The policy docstring used to describe dpi=200 while 22 call sites passed
    their own ``dpi=150``, so the documented policy described nothing that
    actually happened and the stage QC PNGs disagreed with the report figures
    about resolution and background.
    """

    VIZ_MODULES = [
        "src/csttool/metrics/modules/visualizations.py",
        "src/csttool/extract/modules/visualizations.py",
        "src/csttool/preprocess/modules/visualizations.py",
        "src/csttool/tracking/modules/visualizations.py",
        "src/csttool/metrics/modules/qc_figures.py",
    ]

    def _repo_root(self):
        import pathlib
        return pathlib.Path(__file__).resolve().parents[2]

    def test_no_module_hardcodes_a_dpi(self):
        root = self._repo_root()
        offenders = []
        for rel in self.VIZ_MODULES:
            for i, line in enumerate((root / rel).read_text().splitlines(), 1):
                if "dpi=" in line and "SAVEFIG_DPI" not in line \
                        and "PROTOTYPE_DPI" not in line and not line.lstrip().startswith("#"):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        assert not offenders, "hardcoded DPI outside the export policy:\n" + "\n".join(offenders)

    def test_no_module_calls_savefig_directly(self):
        root = self._repo_root()
        offenders = []
        for rel in self.VIZ_MODULES:
            for i, line in enumerate((root / rel).read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if ".savefig(" in stripped or "plt.savefig(" in stripped:
                    offenders.append(f"{rel}:{i}: {stripped}")
        assert not offenders, (
            "figures must be saved through style.save_figure / save_figure_exact:\n"
            + "\n".join(offenders))

    def test_save_figure_exact_pins_the_canvas(self, tmp_path):
        import matplotlib.pyplot as plt
        from PIL import Image
        from csttool.viz import layout, style

        fig = layout.figure_mm(100.0, 50.0)
        fig.add_axes([0, 0, 1, 1]).plot([0, 1], [0, 1])
        out = tmp_path / "exact.png"
        style.save_figure_exact(fig, out)
        plt.close(fig)

        w, h = Image.open(out).size
        expected_w = 100.0 / 25.4 * style.SAVEFIG_DPI
        expected_h = 50.0 / 25.4 * style.SAVEFIG_DPI
        assert abs(w - expected_w) <= 1
        assert abs(h - expected_h) <= 1
