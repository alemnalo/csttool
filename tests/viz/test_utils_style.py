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
