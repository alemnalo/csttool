"""Tests for the enriched per-scalar metric block.

The block gained three dispersion keys and four bootstrap standard errors. The
governing requirement is that nothing it *already* published moved: the profile
is now derived from ``qc_stats.profile_matrix`` in the same pass that produces
the dispersion, instead of being recomputed by `compute_tract_profile`, and the
two must agree exactly rather than approximately — a profile that drifts in the
eighth decimal drifts every regional value and every regional laterality index
with it.
"""

import numpy as np
import pytest

from csttool.metrics.modules.unilateral_analysis import (
    analyze_cst_hemisphere,
    compute_localized_metrics,
    compute_morphology,
    compute_tract_profile,
)

AFFINE = np.array([[2.0, 0, 0, -20.0],
                   [0, 2.0, 0, -20.0],
                   [0, 0, 2.0, -20.0],
                   [0, 0, 0, 1.0]])
SHAPE = (20, 20, 20)


def _si_streamline(x=0.0, y=0.0, z0=-18.0, z1=18.0, n=40):
    z = np.linspace(z0, z1, n)
    return np.column_stack([np.full(n, x), np.full(n, y), z])


def _bundle(n_streamlines=8):
    """Streamlines one voxel apart, so they sample genuinely different columns."""
    return [
        _si_streamline(x=(i % 9 - 4) * 2.0, y=(i // 9 % 9 - 4) * 2.0)
        for i in range(n_streamlines)
    ]


def _varied_length_bundle(n_streamlines=8):
    """A bundle whose streamlines differ in length, so a length SE is non-zero.

    ``_bundle`` deliberately holds length constant (it isolates the scalar
    dispersion); a bootstrap of a constant is correctly 0.0, which would make a
    length-SE test pass for the wrong reason.
    """
    return [
        _si_streamline(x=(i % 9 - 4) * 2.0, z1=6.0 + 2.0 * i)
        for i in range(n_streamlines)
    ]


def _scalar_map(seed=0, dtype=np.float32):
    """A scalar map in the dtype the pipeline actually stores (float32)."""
    return np.random.default_rng(seed).random(SHAPE).astype(dtype)


def _block(streamlines, scalar_map):
    return analyze_cst_hemisphere(
        streamlines, fa_map=scalar_map, affine=AFFINE, hemisphere="left"
    )["fa"]


# ---------------------------------------------------------------------------
# The gate: no published value moved.
# ---------------------------------------------------------------------------

class TestPublishedValuesUnchanged:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_emitted_profile_is_identical_to_compute_tract_profile(self, dtype):
        """Exact, not ``allclose``. float32 maps are the pipeline's normal case,
        and a float64 promotion inside the shared resampler would shift the
        profile by ~5e-8 — invisible in a plot, but a changed published value."""
        scalar = _scalar_map(dtype=dtype)
        bundle = _bundle(12)
        assert _block(bundle, scalar)["profile"] == compute_tract_profile(
            bundle, scalar, AFFINE, n_points=20
        )

    def test_emitted_profile_is_identical_for_upsampled_streamlines(self):
        """Short streamlines take `np.interp`, a different branch of the rule."""
        scalar = _scalar_map(3)
        short = [_si_streamline(x=i * 2.0, n=6) for i in range(4)]
        assert _block(short, scalar)["profile"] == compute_tract_profile(
            short, scalar, AFFINE, n_points=20
        )

    def test_regional_means_are_identical(self):
        scalar = _scalar_map(1)
        bundle = _bundle(12)
        block = _block(bundle, scalar)
        expected = compute_localized_metrics(
            compute_tract_profile(bundle, scalar, AFFINE, n_points=20)
        )
        for region in ("pontine", "plic", "precentral"):
            assert block[region] == expected[region]


# ---------------------------------------------------------------------------
# The new dispersion keys.
# ---------------------------------------------------------------------------

class TestProfileDispersionKeys:
    def test_quartiles_are_ordered_per_node(self):
        """`p25 <= p75` is the assertable property. `p25 <= mean <= p75` is
        deliberately *not* asserted: a mean can sit outside its own IQR when the
        per-node distribution is skewed, and that divergence is a finding."""
        block = _block(_bundle(12), _scalar_map(2))
        p25 = np.array(block["profile_p25"])
        p75 = np.array(block["profile_p75"])
        assert np.all(p25 <= p75)

    def test_identical_streamlines_have_a_degenerate_band(self):
        identical = [_si_streamline() for _ in range(6)]
        block = _block(identical, _scalar_map(4))
        np.testing.assert_array_equal(block["profile_p25"], block["profile_p75"])

    def test_profile_n_counts_contributing_streamlines(self):
        block = _block(_bundle(12), _scalar_map(5))
        assert block["profile_n"] == 12

    def test_profile_n_may_differ_from_n_streamlines(self):
        """A streamline can reach the headline mean (one in-bounds point) and
        still be dropped from the profile (fewer than five)."""
        scalar = _scalar_map(6)
        bundle = _bundle(6) + [
            # Two points, only one of them inside the grid.
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 400.0]])
        ]
        block = _block(bundle, scalar)
        assert block["n_streamlines"] == 7
        assert block["profile_n"] == 6

    def test_bands_have_the_same_length_as_the_profile(self):
        block = _block(_bundle(9), _scalar_map(7))
        assert len(block["profile_p25"]) == len(block["profile"]) == 20
        assert len(block["profile_p75"]) == 20

    def test_bands_are_json_serialisable_lists(self):
        import json

        block = _block(_bundle(5), _scalar_map(8))
        json.dumps({k: block[k] for k in
                    ("profile", "profile_p25", "profile_p75", "profile_n")})


# ---------------------------------------------------------------------------
# The new standard errors.
# ---------------------------------------------------------------------------

class TestBootstrapStandardErrors:
    def test_headline_and_regional_ses_are_present_and_positive(self):
        block = _block(_bundle(12), _scalar_map(9))
        assert block["bootstrap_se"] > 0
        for key in ("pontine_se", "plic_se", "precentral_se"):
            assert block[key] > 0

    def test_constant_map_gives_zero_ses(self):
        """Known answer: nothing to resample, so nothing can move."""
        block = _block(_bundle(12), np.full(SHAPE, 0.42, dtype=np.float32))
        assert block["bootstrap_se"] == pytest.approx(0.0, abs=1e-12)
        assert block["plic_se"] == pytest.approx(0.0, abs=1e-12)

    def test_morphology_carries_a_length_se_but_no_volume_se(self):
        morphology = compute_morphology(_varied_length_bundle(12), AFFINE)
        assert morphology["bootstrap_se_length"] > 0
        assert "bootstrap_se_volume" not in morphology

    def test_deterministic_across_calls(self):
        scalar, bundle = _scalar_map(10), _bundle(12)
        assert _block(bundle, scalar)["bootstrap_se"] == _block(bundle, scalar)["bootstrap_se"]


# ---------------------------------------------------------------------------
# Degradation.
# ---------------------------------------------------------------------------

class TestEmptyBundle:
    def test_bands_are_zeros_and_ses_are_none(self):
        block = _block([], _scalar_map(11))
        assert block["profile"] == [0.0] * 20
        assert block["profile_p25"] == [0.0] * 20
        assert block["profile_p75"] == [0.0] * 20
        assert block["profile_n"] == 0
        assert block["bootstrap_se"] is None
        for key in ("pontine_se", "plic_se", "precentral_se"):
            assert block[key] is None

    def test_empty_morphology_length_se_is_none_not_zero(self):
        assert compute_morphology([], AFFINE)["bootstrap_se_length"] is None

    def test_empty_block_is_json_serialisable(self):
        """The empty profile used to be an ndarray, which json.dump cannot write."""
        import json

        json.dumps(_block([], _scalar_map(12)))


# ---------------------------------------------------------------------------
# Laterality-index standard errors.
# ---------------------------------------------------------------------------

class TestLateralityIndexSe:
    def _sides(self):
        left = analyze_cst_hemisphere(
            _varied_length_bundle(12), fa_map=_scalar_map(13), affine=AFFINE,
            hemisphere="left",
        )
        right = analyze_cst_hemisphere(
            _varied_length_bundle(10), fa_map=_scalar_map(14), affine=AFFINE,
            hemisphere="right",
        )
        return left, right

    def test_scalar_and_regional_indices_carry_an_se(self):
        from csttool.metrics.modules.bilateral_analysis import compute_laterality_indices

        asym = compute_laterality_indices(*self._sides())
        assert asym["fa"]["laterality_index_se"] > 0
        assert asym["fa_plic"]["laterality_index_se"] > 0
        assert asym["mean_length"]["laterality_index_se"] > 0

    def test_count_and_volume_indices_carry_no_se(self):
        """A count *is* its own sample size; a volume is a voxel-set union, not a
        mean over a resamplable population (plan §7.1)."""
        from csttool.metrics.modules.bilateral_analysis import compute_laterality_indices

        asym = compute_laterality_indices(*self._sides())
        assert asym["streamline_count"].get("laterality_index_se") is None
        assert asym["volume"].get("laterality_index_se") is None

    def test_legacy_metric_dicts_without_ses_render_none(self):
        """A metrics dict produced by an older version must not crash, and must
        not be given a fabricated zero."""
        from csttool.metrics.modules.bilateral_analysis import compute_laterality_indices

        left, right = self._sides()
        for side in (left, right):
            side["fa"].pop("bootstrap_se")
            side["morphology"].pop("bootstrap_se_length")
        asym = compute_laterality_indices(left, right)
        assert asym["fa"]["laterality_index_se"] is None
        assert asym["mean_length"]["laterality_index_se"] is None
        assert asym["fa"]["laterality_index"] != 0.0  # the index itself is intact
