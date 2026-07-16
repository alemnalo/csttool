"""
Orientation-robustness tests for the along-tract profile (audit findings AU9 / AU29).

`compute_tract_profile` resamples each streamline in its stored point order. DIPY's
`LocalTracking` stores each streamline as [backward_from_seed][forward_from_seed], so a
bundle mixes orientations — measured at ~20% reversed on real CST data. Averaging point
index i across a mixed bundle aligns one streamline's pontine end with another's
precentral end, which biases every regional FA/MD/RD/AD value and the profile figures.

These tests pin two distinct properties, and both are needed:

  - *invariance*: reversing streamlines must not change the profile.
  - *anatomy*: index 0 must be the inferior (pontine) end.

Invariance alone is not enough. An implementation that merely orients every streamline to
agree with `streamlines[0]` is self-consistent and passes the invariance tests, but leaves
a 50/50 chance the whole bundle is flipped — which silently swaps the `pontine` and
`precentral` labels that `compute_localized_metrics` and the plots assign by index.
`test_profile_index_zero_is_inferior_end` is the test that catches that.

All tests run on a Z-gradient scalar map. The flat `synthetic_nifti` cube cannot be used:
a constant map yields a constant profile that is invariant under flips whether or not the
bug is fixed, so every assertion would pass vacuously.
"""

import warnings

import numpy as np
import pytest

from csttool.metrics import (
    analyze_cst_hemisphere,
    compute_tract_profile,
    orient_streamlines_inferior_to_superior,
)


def _vertical_bundle(n_streamlines=6, n_points=12):
    """A CST-like bundle of straight lines ascending in +Z, all stored inferior->superior."""
    z = np.linspace(1.0, 8.0, n_points)
    return [
        np.stack([np.full(n_points, 4.0 + i * 0.1),
                  np.full(n_points, 4.0 + i * 0.1),
                  z], axis=1).astype(np.float32)
        for i in range(n_streamlines)
    ]


def _reverse_alternating(streamlines):
    """Reverse every other streamline, mimicking DIPY's mixed [backward][forward] storage."""
    return [s[::-1] if i % 2 else s for i, s in enumerate(streamlines)]


def _profile(streamlines, gradient_map, affine):
    return np.asarray(compute_tract_profile(streamlines, gradient_map, affine, n_points=20))


@pytest.mark.parametrize("store", ["forward", "reversed", "mixed"])
def test_profile_index_zero_is_inferior_end(
    store, synthetic_z_gradient_data, synthetic_affine
):
    """Index 0 is the inferior end however the bundle happens to be stored.

    This is the load-bearing test: it pins *anatomical* orientation, not merely mutual
    consistency between streamlines. The "reversed" case is the one that matters — a
    bundle stored entirely superior-to-inferior is already self-consistent, so a
    centroid-relative fix would leave it profiling backwards and only this assertion
    would notice.
    """
    forward = _vertical_bundle()
    bundle = {
        "forward": forward,
        "reversed": [s[::-1] for s in forward],
        "mixed": _reverse_alternating(forward),
    }[store]

    profile = _profile(bundle, synthetic_z_gradient_data, synthetic_affine)

    assert profile[0] < profile[-1], (
        "profile does not ascend on an ascending-Z map: index 0 is not the inferior end"
    )


def test_profile_invariant_to_streamline_reversal(synthetic_z_gradient_data, synthetic_affine):
    """The AU29 test: reversing half the bundle must not change the profile."""
    forward = _vertical_bundle()
    expected = _profile(forward, synthetic_z_gradient_data, synthetic_affine)

    # Guard against a vacuous pass: on a flat map every profile is invariant regardless.
    assert not np.allclose(expected[0], expected[-1]), (
        "profile is constant - a flat scalar map cannot detect an orientation flip"
    )

    mixed = _profile(_reverse_alternating(forward), synthetic_z_gradient_data, synthetic_affine)

    assert np.allclose(mixed, expected), "profile changed when half the streamlines were reversed"


def test_fully_reversed_bundle_matches_forward_bundle(
    synthetic_z_gradient_data, synthetic_affine
):
    """A wholly-reversed bundle profiles identically, not as a mirror image."""
    forward = _vertical_bundle()
    expected = _profile(forward, synthetic_z_gradient_data, synthetic_affine)
    reversed_all = _profile(
        [s[::-1] for s in forward], synthetic_z_gradient_data, synthetic_affine
    )

    assert np.allclose(reversed_all, expected)


def test_localized_metrics_pontine_is_inferior_end(
    synthetic_z_gradient_data, synthetic_affine
):
    """End-to-end: the 'pontine' bin is the inferior end and survives reversal."""
    forward = _vertical_bundle()
    mixed = _reverse_alternating(forward)

    fa_fwd = analyze_cst_hemisphere(
        streamlines=forward, fa_map=synthetic_z_gradient_data,
        affine=synthetic_affine, hemisphere='left',
    )['fa']
    fa_mixed = analyze_cst_hemisphere(
        streamlines=mixed, fa_map=synthetic_z_gradient_data,
        affine=synthetic_affine, hemisphere='left',
    )['fa']

    assert fa_fwd['pontine'] < fa_fwd['precentral'], "pontine bin is not the inferior end"
    for region in ('pontine', 'plic', 'precentral'):
        assert np.isclose(fa_mixed[region], fa_fwd[region]), (
            f"{region} changed when half the streamlines were reversed"
        )


def test_headline_mean_unchanged_by_reversal(synthetic_z_gradient_data, synthetic_affine):
    """Headline mean/std/median are order-invariant, so AU9's reorientation cannot move them.

    AU10 changed the headline from the point pool to the per-streamline mean (one vote per
    streamline). That stays order-invariant in exact arithmetic - reversing a streamline
    does not change the set of points it samples, so its mean is unchanged - but the
    per-streamline computation sums each streamline's points *before* averaging across
    streamlines, so floating-point summation order can shift the result by ~1e-7. The
    assertion is tolerance-based rather than bit-exact; any *real* order-dependence (the
    AU9 defect moved the profile by ~4%) is orders of magnitude larger. The preserved
    point-pool summary stays bit-exact under reversal, which is asserted alongside.
    """
    forward = _vertical_bundle()
    fa_fwd = analyze_cst_hemisphere(
        streamlines=forward, fa_map=synthetic_z_gradient_data,
        affine=synthetic_affine, hemisphere='left',
    )['fa']
    fa_mixed = analyze_cst_hemisphere(
        streamlines=_reverse_alternating(forward), fa_map=synthetic_z_gradient_data,
        affine=synthetic_affine, hemisphere='left',
    )['fa']

    for key in ('mean', 'std', 'median'):
        assert np.isclose(fa_mixed[key], fa_fwd[key], atol=1e-6), (
            f"headline {key} is order-dependent: {fa_fwd[key]} vs {fa_mixed[key]}"
        )
    # The point-pool summary is a single flat sum, so it stays bit-exact under reversal.
    for key in ('mean_point_weighted', 'std_point_weighted'):
        assert fa_mixed[key] == fa_fwd[key], f"point-pool {key} is order-dependent"


def test_input_streamlines_not_mutated(synthetic_z_gradient_data, synthetic_affine):
    """Callers reuse their bundles after profiling (cli/commands/metrics.py QC previews)."""
    streamlines = _reverse_alternating(_vertical_bundle())
    before = [np.array(s, copy=True) for s in streamlines]

    compute_tract_profile(streamlines, synthetic_z_gradient_data, synthetic_affine)

    for original, current in zip(before, streamlines):
        assert np.array_equal(original, current), "input streamlines were mutated in place"


def test_reorientation_is_idempotent():
    """Reorienting twice equals reorienting once - it is applied per scalar map."""
    once = orient_streamlines_inferior_to_superior(_reverse_alternating(_vertical_bundle()))
    twice = orient_streamlines_inferior_to_superior(once)

    for a, b in zip(once, twice):
        assert np.array_equal(a, b)


def test_non_vertical_bundle_warns():
    """A bundle that does not run inferior-superior gets a warning: index 0 is meaningless."""
    n = 12
    x = np.linspace(1.0, 8.0, n)
    horizontal = [
        np.stack([x, np.full(n, 4.0 + i * 0.1), np.full(n, 4.0)], axis=1).astype(np.float32)
        for i in range(4)
    ]

    with pytest.warns(UserWarning, match="not.*superior-inferior|dominant"):
        orient_streamlines_inferior_to_superior(horizontal)


def test_vertical_bundle_does_not_warn():
    """A CST-like bundle must not trip the verticality warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        orient_streamlines_inferior_to_superior(_vertical_bundle())


def test_degenerate_and_horizontal_streamlines():
    """Single-point and flat-Z streamlines are handled without error, deterministically."""
    degenerate = [
        np.array([[5.0, 5.0, 5.0]], dtype=np.float32),                    # single point
        np.array([[5.0, 5.0, 3.0], [5.0, 5.0, 3.0]], dtype=np.float32),   # dz == 0, dy == 0, dx == 0
        np.array([[1.0, 5.0, 3.0], [4.0, 5.0, 3.0]], dtype=np.float32),   # dz == 0, resolved on X
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = orient_streamlines_inferior_to_superior(degenerate)
        second = orient_streamlines_inferior_to_superior(degenerate)

    for a, b in zip(first, second):
        assert np.array_equal(a, b), "reorientation is not deterministic for degenerate input"
