"""
Length-bias tests for the headline scalar mean (audit finding AU10).

`sample_scalar_along_tract` pools every point of every streamline, so longer streamlines
cast more votes: the headline ``mean`` was point-weighted (length-biased), while the
displayed tract profile was per-streamline-normalised (one vote each). The laterality
indices and the report's global table used the biased ``mean``; the thesis's "FA symmetric,
count asymmetric" conclusion rests on it.

These tests pin the fix: the headline ``mean``/``std``/``median``/``min``/``max`` describe
the **per-streamline** population (one vote per streamline, length-unbiased), the global
laterality index uses that headline, and the length-biased point-pool summary is preserved
under honestly-named ``*_point_weighted`` keys.

The fixture is load-bearing. It must have **unequal streamline lengths correlated with the
scalar** or it cannot detect length bias at all: on a flat map, or with equal-length
streamlines, the per-streamline and point-weighted means are identical and every assertion
below would pass vacuously against the broken code. Here FA falls with Z, short streamlines
stay in the high-FA bottom and the long streamline extends into low-FA territory, so the
long streamline both has a lower mean FA *and* casts more point-votes — the exact condition
that separates the two summaries.
"""

import numpy as np
import pytest

from csttool.metrics import (
    analyze_cst_hemisphere,
    compute_laterality_indices,
    sample_scalar_along_tract,
)
from csttool.metrics.modules.unilateral_analysis import world_to_voxel


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _gradient_fa_map():
    """10x10x10 FA map that falls linearly from 0.9 (k=0) to 0.1 (k=9) along +Z."""
    fa = np.zeros((10, 10, 10), dtype=np.float32)
    fa[:, :, :] = 0.1 + (np.arange(9, -1, -1, dtype=np.float32) / 9.0) * 0.8
    return fa


def _vertical_streamline(z_values, x=5.0, y=5.0):
    """A straight vertical streamline through voxel column (x, y) at the given Z values."""
    n = len(z_values)
    return np.stack(
        [np.full(n, x), np.full(n, y), np.asarray(z_values, dtype=np.float32)],
        axis=1,
    ).astype(np.float32)


def _length_biased_bundle(n_short=3, n_long=1):
    """A bundle whose streamlines have unequal lengths correlated with the scalar.

    Short streamlines sample z=1..3 (high FA, ~0.72 mean); the long streamline samples
    z=1..8 (extends into low FA, ~0.50 mean). So the long streamline has a lower mean FA
    *and* more points, which is what makes point-weighting diverge from per-streamline
    averaging. Returns ``(streamlines, fa_map)``.
    """
    fa = _gradient_fa_map()
    short = _vertical_streamline([1.0, 2.0, 3.0])
    long = _vertical_streamline(np.arange(1.0, 9.0))  # z = 1..8
    streamlines = [short] * n_short + [long] * n_long
    return streamlines, fa


def _per_streamline_means(streamlines, scalar_map, affine):
    """Independent reimplementation of the per-streamline mean (one value per streamline).

    Recomputed in the test rather than calling the production function, so the assertion
    is an independent check, not a tautology.
    """
    means = []
    for streamline in streamlines:
        vals = []
        for point in streamline:
            vc = world_to_voxel(point, affine)
            if (0 <= vc[0] < scalar_map.shape[0] and
                    0 <= vc[1] < scalar_map.shape[1] and
                    0 <= vc[2] < scalar_map.shape[2]):
                vals.append(scalar_map[vc[0], vc[1], vc[2]])
        if len(vals) > 0:
            means.append(float(np.mean(vals)))
    return np.array(means)


# ---------------------------------------------------------------------------
# Headline is per-streamline, not point-weighted
# ---------------------------------------------------------------------------

@pytest.fixture
def length_biased():
    return _length_biased_bundle()


def test_headline_mean_is_per_streamline_not_point_weighted(length_biased, synthetic_affine):
    """The headline `mean` is the per-streamline mean (one vote each), not the point pool."""
    streamlines, fa = length_biased
    m = analyze_cst_hemisphere(
        streamlines, fa_map=fa, affine=synthetic_affine, hemisphere='left'
    )['fa']

    sl_means = _per_streamline_means(streamlines, fa, synthetic_affine)
    expected_per_streamline = float(np.mean(sl_means))
    expected_point_weighted = float(np.mean(
        sample_scalar_along_tract(streamlines, fa, synthetic_affine)
    ))

    # Non-vacuous guard: the two definitions must actually differ on this fixture.
    assert not np.isclose(expected_per_streamline, expected_point_weighted), (
        "fixture is vacuous: per-streamline and point-weighted means coincide, so it "
        "cannot detect length bias (need unequal lengths correlated with the scalar)"
    )

    assert np.isclose(m['mean'], expected_per_streamline), (
        f"headline mean is point-weighted: got {m['mean']}, expected per-streamline "
        f"{expected_per_streamline} (point-weighted was {expected_point_weighted})"
    )
    assert np.isclose(m['mean_point_weighted'], expected_point_weighted), (
        "preserved point-pool mean does not match sample_scalar_along_tract"
    )
    assert not np.isclose(m['mean'], m['mean_point_weighted']), (
        "headline and point-weighted means coincide on a non-vacuous fixture"
    )


def test_headline_block_is_coherent_per_streamline(length_biased, synthetic_affine):
    """mean/std/median/min/max/n_streamlines all describe the per-streamline population."""
    streamlines, fa = length_biased
    m = analyze_cst_hemisphere(
        streamlines, fa_map=fa, affine=synthetic_affine, hemisphere='left'
    )['fa']

    sl_means = _per_streamline_means(streamlines, fa, synthetic_affine)

    assert np.isclose(m['std'], float(np.std(sl_means)))
    assert np.isclose(m['median'], float(np.median(sl_means)))
    assert np.isclose(m['min'], float(np.min(sl_means)))
    assert np.isclose(m['max'], float(np.max(sl_means)))
    assert m['n_streamlines'] == len(sl_means)
    # Between-streamline SD is the spread of streamline means; the point-pool SD is the
    # spread of every voxel. They are different quantities and the point-pool one is larger.
    assert m['std'] < m['std_point_weighted']


def test_point_pool_summary_preserved(length_biased, synthetic_affine):
    """The length-biased point-pool summary is preserved under *_point_weighted keys."""
    streamlines, fa = length_biased
    m = analyze_cst_hemisphere(
        streamlines, fa_map=fa, affine=synthetic_affine, hemisphere='left'
    )['fa']

    pool = sample_scalar_along_tract(streamlines, fa, synthetic_affine)
    assert np.isclose(m['std_point_weighted'], float(np.std(pool)))
    assert np.isclose(m['median_point_weighted'], float(np.median(pool)))
    assert np.isclose(m['min_point_weighted'], float(np.min(pool)))
    assert np.isclose(m['max_point_weighted'], float(np.max(pool)))
    assert m['n_samples'] == len(pool)


def test_empty_bundle_emits_zeroed_block_with_all_keys(synthetic_affine):
    """Empty input yields a zeroed block that still carries every key (no KeyError)."""
    fa = _gradient_fa_map()
    m = analyze_cst_hemisphere(
        [], fa_map=fa, affine=synthetic_affine, hemisphere='left'
    )['fa']

    for key in ('mean', 'std', 'median', 'min', 'max', 'n_streamlines',
                'mean_point_weighted', 'std_point_weighted', 'median_point_weighted',
                'min_point_weighted', 'max_point_weighted', 'n_samples',
                'profile', 'pontine', 'plic', 'precentral'):
        assert key in m, f"missing key in empty block: {key}"
    assert m['n_streamlines'] == 0
    assert m['n_samples'] == 0


# ---------------------------------------------------------------------------
# The laterality index uses the per-streamline mean
# ---------------------------------------------------------------------------

def test_global_fa_li_uses_per_streamline_mean(synthetic_affine):
    """The global FA LI is computed from the per-streamline mean, not the point-weighted one.

    Left and right bundles have the same scalar map but different length distributions, so
    the per-streamline LI and the point-weighted LI differ. The headline LI must match the
    per-streamline one.
    """
    left_streamlines, fa = _length_biased_bundle(n_short=3, n_long=1)
    right_streamlines, _ = _length_biased_bundle(n_short=1, n_long=3)

    left = analyze_cst_hemisphere(
        left_streamlines, fa_map=fa, affine=synthetic_affine, hemisphere='left'
    )
    right = analyze_cst_hemisphere(
        right_streamlines, fa_map=fa, affine=synthetic_affine, hemisphere='right'
    )
    asym = compute_laterality_indices(left, right)

    left_sl = float(np.mean(_per_streamline_means(left_streamlines, fa, synthetic_affine)))
    right_sl = float(np.mean(_per_streamline_means(right_streamlines, fa, synthetic_affine)))
    left_pw = float(np.mean(sample_scalar_along_tract(left_streamlines, fa, synthetic_affine)))
    right_pw = float(np.mean(sample_scalar_along_tract(right_streamlines, fa, synthetic_affine)))

    li_per_streamline = (left_sl - right_sl) / (left_sl + right_sl)
    li_point_weighted = (left_pw - right_pw) / (left_pw + right_pw)

    # Non-vacuous guard: the two LIs must differ on this fixture.
    assert not np.isclose(li_per_streamline, li_point_weighted, atol=1e-4), (
        "fixture is vacuous: per-streamline and point-weighted LIs coincide"
    )

    assert np.isclose(asym['fa']['laterality_index'], li_per_streamline), (
        f"global FA LI used the point-weighted mean: got {asym['fa']['laterality_index']}, "
        f"expected per-streamline LI {li_per_streamline} (point-weighted was {li_point_weighted})"
    )


# ---------------------------------------------------------------------------
# The new public function
# ---------------------------------------------------------------------------

def test_sample_scalar_per_streamline_one_vote_each(length_biased, synthetic_affine):
    """`sample_scalar_per_streamline` returns one mean per streamline, length-unbiased."""
    from csttool.metrics import sample_scalar_per_streamline  # new in AU10

    streamlines, fa = length_biased
    means = sample_scalar_per_streamline(streamlines, fa, synthetic_affine)

    # One value per streamline (all have >=1 in-bounds point here).
    assert len(means) == len(streamlines)

    # Each value is the mean of that streamline's in-bounds points.
    for streamline, mean_val in zip(streamlines, means):
        vals = []
        for point in streamline:
            vc = world_to_voxel(point, synthetic_affine)
            if (0 <= vc[0] < fa.shape[0] and
                    0 <= vc[1] < fa.shape[1] and
                    0 <= vc[2] < fa.shape[2]):
                vals.append(fa[vc[0], vc[1], vc[2]])
        assert np.isclose(mean_val, np.mean(vals))

    # And it equals neither the point pool nor its mean by construction: averaging the
    # per-streamline means gives the headline, which differs from the point-weighted mean.
    pool = sample_scalar_along_tract(streamlines, fa, synthetic_affine)
    assert not np.isclose(np.mean(means), np.mean(pool))


def test_sample_scalar_per_streamline_empty(synthetic_affine):
    """Empty input returns an empty array, not an error."""
    from csttool.metrics import sample_scalar_per_streamline

    fa = _gradient_fa_map()
    out = sample_scalar_per_streamline([], fa, synthetic_affine)
    assert len(out) == 0
