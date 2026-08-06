"""Tests for ``csttool.metrics.modules.qc_stats`` — the trust-chain diagnostics.

Every test asserts a *known* answer on a synthetic bundle rather than a
self-consistency property: a bundle running along +Z sampled against a V1 field
along +Z must give 0 degrees, identical streamlines must give zero dispersion, a
bundle shifted 10 mm superiorly must give a 10 mm node offset. A diagnostic that
only agrees with itself would not catch a sign error or a frame error.

The one exception is `test_profile_matrix_mean_reproduces_compute_tract_profile`,
which is deliberately a consistency test: it is the contract that makes the
dispersion band describe the published mean rather than a near neighbour of it.
"""

import numpy as np
import pytest

from csttool.metrics.modules import qc_stats
from csttool.metrics.modules.unilateral_analysis import (
    compute_tract_profile,
    sample_scalar_per_streamline,
)

AFFINE = np.array([[2.0, 0, 0, -20.0],
                   [0, 2.0, 0, -20.0],
                   [0, 0, 2.0, -20.0],
                   [0, 0, 0, 1.0]])
SHAPE = (20, 20, 20)


def _si_streamline(x=0.0, y=0.0, z0=-18.0, z1=18.0, n=40):
    """A straight streamline running inferior -> superior at (x, y)."""
    z = np.linspace(z0, z1, n)
    return np.column_stack([np.full(n, x), np.full(n, y), z])


def _bundle(n_streamlines=8, x=0.0, **kwargs):
    """A bundle whose streamlines each occupy a *distinct* voxel column.

    The 2 mm spacing matters: streamlines closer together than one voxel sample
    identical values under nearest-neighbour lookup, which would give the bundle
    zero between-streamline variance and make the dispersion and saturation
    tests pass vacuously.
    """
    return [
        _si_streamline(x=x + (i % 9 - 4) * 2.0, y=(i // 9 % 9 - 4) * 2.0, **kwargs)
        for i in range(n_streamlines)
    ]


def _uniform_v1(direction):
    v1 = np.zeros(SHAPE + (3,), dtype=np.float32)
    v1[..., :] = np.asarray(direction, dtype=np.float32)
    return v1


class TestTangentV1Angles:
    def test_aligned_field_gives_zero_degrees(self):
        angles, meta = qc_stats.tangent_v1_angles(
            _bundle(), _uniform_v1([0, 0, 1]), AFFINE)
        assert angles.shape == (8, 20)
        assert np.allclose(angles, 0.0, atol=1e-4)
        assert meta["n_contributing"] == 8
        assert meta["median_angle_deg"] == pytest.approx(0.0, abs=1e-4)

    def test_orthogonal_field_gives_ninety_degrees(self):
        angles, _ = qc_stats.tangent_v1_angles(
            _bundle(), _uniform_v1([1, 0, 0]), AFFINE)
        assert np.allclose(angles, 90.0, atol=1e-4)

    def test_forty_five_degree_field(self):
        direction = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
        angles, _ = qc_stats.tangent_v1_angles(
            _bundle(), _uniform_v1(direction), AFFINE)
        assert np.allclose(angles, 45.0, atol=1e-3)

    def test_eigenvector_sign_is_irrelevant(self):
        """|cos| is taken: a field pointing inferiorly is still aligned."""
        up, _ = qc_stats.tangent_v1_angles(
            _bundle(), _uniform_v1([0, 0, 1]), AFFINE)
        down, _ = qc_stats.tangent_v1_angles(
            _bundle(), _uniform_v1([0, 0, -1]), AFFINE)
        assert np.allclose(up, down, atol=1e-6)

    def test_input_streamline_order_is_irrelevant(self):
        """Streamlines stored superior-to-inferior give the same angles."""
        forward = _bundle()
        reversed_bundle = [s[::-1] for s in forward]
        a, _ = qc_stats.tangent_v1_angles(forward, _uniform_v1([0, 0, 1]), AFFINE)
        b, _ = qc_stats.tangent_v1_angles(reversed_bundle,
                                          _uniform_v1([0, 0, 1]), AFFINE)
        assert np.allclose(a, b, atol=1e-6)

    def test_accepts_five_dimensional_stored_product(self):
        """The on-disk V1 product is (X, Y, Z, 1, 3)."""
        v1 = _uniform_v1([0, 0, 1])
        v1_5d = v1.reshape(*SHAPE, 1, 3)
        angles, _ = qc_stats.tangent_v1_angles(_bundle(), v1_5d, AFFINE)
        assert np.allclose(angles, 0.0, atol=1e-4)

    def test_zero_vectors_are_excluded_not_counted_as_aligned(self):
        v1 = np.zeros(SHAPE + (3,), dtype=np.float32)
        angles, meta = qc_stats.tangent_v1_angles(_bundle(), v1, AFFINE)
        assert angles.shape == (0, 20)
        assert meta["n_contributing"] == 0
        assert meta["n_points_zero_vector"] > 0

    def test_out_of_bounds_points_are_counted(self):
        far = [_si_streamline(x=500.0)]
        angles, meta = qc_stats.tangent_v1_angles(
            far, _uniform_v1([0, 0, 1]), AFFINE)
        assert angles.shape == (0, 20)
        assert meta["n_points_out_of_bounds"] == 40

    def test_empty_bundle(self):
        angles, meta = qc_stats.tangent_v1_angles([], _uniform_v1([0, 0, 1]), AFFINE)
        assert angles.shape == (0, 20)
        assert meta["n_input"] == 0

    def test_rejects_non_vector_field(self):
        with pytest.raises(ValueError, match="vector field"):
            qc_stats.tangent_v1_angles(_bundle(), np.zeros(SHAPE), AFFINE)


class TestProfileMatrix:
    def test_mean_reproduces_compute_tract_profile(self):
        """The contract: the band must describe the published mean exactly."""
        rng = np.random.default_rng(0)
        scalar = rng.random(SHAPE).astype(np.float32)
        bundle = _bundle(n_streamlines=12)
        matrix, _ = qc_stats.profile_matrix(bundle, scalar, AFFINE)
        expected = compute_tract_profile(bundle, scalar, AFFINE)
        assert np.allclose(matrix.mean(axis=0), expected)

    def test_identical_streamlines_have_zero_dispersion(self):
        scalar = np.random.default_rng(1).random(SHAPE).astype(np.float32)
        identical = [_si_streamline() for _ in range(6)]
        matrix, _ = qc_stats.profile_matrix(identical, scalar, AFFINE)
        dispersion = qc_stats.profile_dispersion(matrix)
        assert np.allclose(dispersion["p5"], dispersion["p95"])
        assert dispersion["n"] == 6

    def test_profile_follows_a_known_gradient(self):
        """A scalar increasing with world Z must give a monotone profile."""
        scalar = np.zeros(SHAPE, dtype=np.float32)
        scalar[:] = np.arange(SHAPE[2], dtype=np.float32)[None, None, :]
        matrix, _ = qc_stats.profile_matrix(_bundle(), scalar, AFFINE)
        profile = matrix.mean(axis=0)
        # Non-decreasing rather than strictly increasing: nearest-neighbour
        # lookup makes consecutive nodes land in the same voxel when the
        # streamline is sampled more finely than the grid.
        assert np.all(np.diff(profile) >= 0), "inferior-to-superior ordering lost"
        assert profile[-1] > profile[0]

    def test_attrition_counts_out_of_bounds_bundle(self):
        scalar = np.ones(SHAPE, dtype=np.float32)
        matrix, attrition = qc_stats.profile_matrix(
            [_si_streamline(x=500.0)], scalar, AFFINE)
        assert matrix.shape == (0, 20)
        assert attrition["n_no_inbounds"] == 1
        assert attrition["n_contributing"] == 0
        assert attrition["point_retention"] == 0.0

    def test_attrition_counts_partially_sampled_streamline(self):
        """A streamline with 1-4 in-bounds points is dropped, and said so."""
        scalar = np.ones(SHAPE, dtype=np.float32)
        # Three points inside the grid, the rest far outside it.
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0],
                           [500.0, 0.0, 0.0], [600.0, 0.0, 0.0]])
        # This fixture runs predominantly along X by construction, so the
        # inferior-to-superior reorientation correctly objects.
        with pytest.warns(UserWarning, match="predominantly along X"):
            matrix, attrition = qc_stats.profile_matrix([points], scalar, AFFINE)
        assert matrix.shape == (0, 20)
        assert attrition["n_insufficient_inbounds"] == 1
        assert attrition["points_total"] == 5
        assert attrition["points_in_bounds"] == 3
        assert attrition["point_retention"] == pytest.approx(0.6)

    def test_attrition_counts_too_short_streamline(self):
        scalar = np.ones(SHAPE, dtype=np.float32)
        _, attrition = qc_stats.profile_matrix(
            [np.array([[0.0, 0.0, 0.0]])], scalar, AFFINE)
        assert attrition["n_too_short"] == 1

    def test_empty_bundle(self):
        matrix, attrition = qc_stats.profile_matrix(
            [], np.ones(SHAPE, np.float32), AFFINE)
        assert matrix.shape == (0, 20)
        assert attrition["n_input"] == 0
        assert attrition["point_retention"] is None


class TestProfileDispersion:
    def test_percentiles_are_per_node(self):
        matrix = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
        dispersion = qc_stats.profile_dispersion(matrix)
        assert np.allclose(dispersion["p50"], [1.0, 11.0])
        assert np.allclose(dispersion["mean"], [1.0, 11.0])
        assert dispersion["n"] == 3

    def test_empty_matrix(self):
        dispersion = qc_stats.profile_dispersion(np.zeros((0, 20)))
        assert dispersion["n"] == 0
        assert dispersion["p50"] is None


class TestSubsampleStability:
    def _scalar(self):
        return np.random.default_rng(2).random(SHAPE).astype(np.float32)

    def test_full_estimate_matches_the_published_statistic(self):
        """The curve must describe the length-unbiased mean the report prints."""
        scalar = self._scalar()
        bundle = _bundle(n_streamlines=30)
        stability = qc_stats.subsample_stability(bundle, scalar, AFFINE)
        expected = sample_scalar_per_streamline(bundle, scalar, AFFINE).mean()
        assert stability["full_estimate"] == pytest.approx(expected)

    def test_full_fraction_has_no_subsampling_variance(self):
        stability = qc_stats.subsample_stability(
            _bundle(n_streamlines=30), self._scalar(), AFFINE)
        assert stability["fractions"][-1] == 1.0
        assert stability["std"][-1] == pytest.approx(0.0)

    def test_bootstrap_reports_variance_the_curve_cannot(self):
        stability = qc_stats.subsample_stability(
            _bundle(n_streamlines=30), self._scalar(), AFFINE)
        assert stability["bootstrap_full"]["std"] > 0

    def test_constant_scalar_gives_a_flat_curve(self):
        """Known answer: with no between-streamline variance, N cannot matter."""
        scalar = np.full(SHAPE, 0.42, dtype=np.float32)
        stability = qc_stats.subsample_stability(
            _bundle(n_streamlines=30), scalar, AFFINE)
        assert np.allclose(stability["mean"], 0.42, atol=1e-6)
        assert np.allclose(stability["std"], 0.0, atol=1e-6)
        assert stability["bootstrap_full"]["std"] == pytest.approx(0.0, abs=1e-6)

    def test_deterministic_for_a_given_seed(self):
        scalar = self._scalar()
        bundle = _bundle(n_streamlines=30)
        a = qc_stats.subsample_stability(bundle, scalar, AFFINE, seed=7)
        b = qc_stats.subsample_stability(bundle, scalar, AFFINE, seed=7)
        assert np.array_equal(a["estimates"], b["estimates"])

    def test_different_seeds_draw_different_subsamples(self):
        scalar = self._scalar()
        bundle = _bundle(n_streamlines=30)
        a = qc_stats.subsample_stability(bundle, scalar, AFFINE, seed=7)
        b = qc_stats.subsample_stability(bundle, scalar, AFFINE, seed=8)
        assert not np.array_equal(a["estimates"], b["estimates"])

    def test_empty_bundle(self):
        stability = qc_stats.subsample_stability(
            [], np.ones(SHAPE, np.float32), AFFINE)
        assert stability["n_streamlines"] == 0
        assert stability["full_estimate"] is None


class TestNodeGeometry:
    def test_arc_length_and_length_are_millimetres(self):
        straight = [_si_streamline(z0=0.0, z1=10.0, n=11)]
        geometry = qc_stats.node_geometry(straight)
        assert geometry["arc_length_mm"][0] == pytest.approx(0.0)
        assert geometry["arc_length_mm"][-1] == pytest.approx(10.0)
        assert geometry["lengths_mm"][0] == pytest.approx(10.0)
        assert geometry["length_mean"] == pytest.approx(10.0)

    def test_node_z_is_ordered_inferior_to_superior(self):
        geometry = qc_stats.node_geometry(_bundle())
        assert np.all(np.diff(geometry["node_z"]) > 0)

    def test_stored_orientation_does_not_change_node_positions(self):
        forward = _bundle()
        backward = [s[::-1] for s in forward]
        assert np.allclose(qc_stats.node_geometry(forward)["node_z"],
                           qc_stats.node_geometry(backward)["node_z"])

    def test_empty_bundle(self):
        geometry = qc_stats.node_geometry([])
        assert geometry["n_streamlines"] == 0
        assert geometry["node_z"] is None


class TestCompareNodeGeometry:
    def test_identical_bundles_are_perfectly_homologous(self):
        bundle = _bundle()
        comparison = qc_stats.compare_node_geometry(bundle, list(bundle))
        assert np.allclose(comparison["z_difference_mm"], 0.0)
        assert comparison["max_abs_z_difference_mm"] == pytest.approx(0.0)
        assert comparison["length_difference_mm"] == pytest.approx(0.0)

    def test_superior_shift_appears_as_a_node_offset(self):
        """Known answer: a bundle 10 mm higher gives a 10 mm offset at every node."""
        left = _bundle()
        right = [s - np.array([0.0, 0.0, 10.0]) for s in left]
        comparison = qc_stats.compare_node_geometry(left, right)
        assert np.allclose(comparison["z_difference_mm"], 10.0, atol=1e-6)
        assert comparison["max_abs_z_difference_mm"] == pytest.approx(10.0)

    def test_shorter_bundle_appears_as_a_length_difference(self):
        left = _bundle(z0=-18.0, z1=18.0)      # 36 mm
        right = _bundle(z0=-8.0, z1=8.0)       # 16 mm
        comparison = qc_stats.compare_node_geometry(left, right)
        assert comparison["length_difference_mm"] == pytest.approx(20.0, abs=1e-6)

    def test_empty_hemisphere_yields_no_comparison(self):
        comparison = qc_stats.compare_node_geometry(_bundle(), [])
        assert comparison["z_difference_mm"] is None
        assert comparison["max_abs_z_difference_mm"] is None


class TestTissuePlausibility:
    """QC-1: the density-weighted FA-MD joint distribution.

    The known answers here are the weighting and the free-water accounting: a
    voxel that carries twice the density must count twice, and the free-water
    fraction must be a mass fraction rather than a voxel fraction.
    """

    def _volumes(self, fa_value=0.6, md_value=8e-4):
        fa = np.full(SHAPE, fa_value, dtype=np.float32)
        md = np.full(SHAPE, md_value, dtype=np.float32)
        density = np.zeros(SHAPE, dtype=np.float32)
        return fa, md, density

    def test_single_tissue_bundle_has_that_tissue_as_its_centroid(self):
        fa, md, density = self._volumes()
        density[5:8, 5:8, 5:8] = 0.5
        stats = qc_stats.tissue_plausibility(fa, md, density)
        assert stats["n_voxels"] == 27
        assert stats["centroid_fa"] == pytest.approx(0.6, abs=1e-5)
        assert stats["centroid_md"] == pytest.approx(8e-4, abs=1e-9)
        assert stats["free_water_fraction"] == pytest.approx(0.0)

    def test_centroid_is_density_weighted_not_voxel_counted(self):
        """Known answer: one voxel at weight 3 outvotes three voxels at weight 1."""
        fa, md, density = self._volumes()
        fa[5, 5, 5] = 0.8
        fa[6, 6, 6:9] = 0.2
        density[5, 5, 5] = 0.9
        density[6, 6, 6:9] = 0.1
        # (0.9*0.8 + 0.3*0.2) / 1.2
        assert qc_stats.tissue_plausibility(fa, md, density)["centroid_fa"] == \
            pytest.approx((0.9 * 0.8 + 0.3 * 0.2) / 1.2, abs=1e-6)

    def test_free_water_fraction_is_a_mass_fraction(self):
        """Half the voxels but a tenth of the mass in the free-water corner."""
        fa, md, density = self._volumes()
        density[5, 5, 5] = 0.9
        density[6, 6, 6] = 0.1
        fa[6, 6, 6], md[6, 6, 6] = 0.05, 2.8e-3     # CSF-like
        stats = qc_stats.tissue_plausibility(fa, md, density)
        assert stats["n_voxels"] == 2
        assert stats["free_water_fraction"] == pytest.approx(0.1, abs=1e-6)

    def test_free_water_needs_both_low_fa_and_high_md(self):
        """High MD alone (oedematous WM) is not free water; nor is low FA alone."""
        fa, md, density = self._volumes()
        density[5, 5, 5:8] = 1.0
        fa[5, 5, 5], md[5, 5, 5] = 0.6, 2.8e-3      # high MD, high FA
        fa[5, 5, 6], md[5, 5, 6] = 0.05, 8e-4       # low FA, low MD
        fa[5, 5, 7], md[5, 5, 7] = 0.05, 2.8e-3     # both -> free water
        stats = qc_stats.tissue_plausibility(fa, md, density)
        assert stats["free_water_fraction"] == pytest.approx(1 / 3, abs=1e-6)

    def test_zero_density_voxels_are_excluded(self):
        """The bundle's own voxels, not the whole brain: FA elsewhere is ignored."""
        fa, md, density = self._volumes()
        fa[...] = 0.05                      # a brain of noise
        fa[5, 5, 5] = 0.7
        density[5, 5, 5] = 1.0
        stats = qc_stats.tissue_plausibility(fa, md, density)
        assert stats["n_voxels"] == 1
        assert stats["centroid_fa"] == pytest.approx(0.7, abs=1e-6)

    def test_histogram_mass_equals_the_density_mass(self):
        """Nothing may leave the figure: clipping keeps out-of-range mass in."""
        fa, md, density = self._volumes()
        density[4:9, 4:9, 4:9] = 0.4
        md[4:6, 4:6, 4:6] = 9e-3            # far beyond the MD axis
        stats = qc_stats.tissue_plausibility(fa, md, density)
        assert stats["histogram"].sum() == pytest.approx(stats["density_mass"])
        assert stats["n_md_above_range"] == 8

    def test_weighted_median_matches_the_unweighted_one_at_equal_weights(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        weighted = qc_stats._weighted_percentile(values, np.ones(5), 50)
        assert weighted == pytest.approx(float(np.median(values)), abs=1e-9)

    def test_mismatched_grids_raise(self):
        fa, md, density = self._volumes()
        with pytest.raises(ValueError, match="share a grid"):
            qc_stats.tissue_plausibility(fa, md[:-1], density)

    def test_empty_bundle_yields_no_centroid(self):
        fa, md, density = self._volumes()
        stats = qc_stats.tissue_plausibility(fa, md, density)
        assert stats["n_voxels"] == 0
        assert stats["centroid_fa"] is None
        assert stats["free_water_fraction"] is None


class TestAttritionFunnel:
    """QC-7: the survival funnel from extraction to the published profile."""

    def _scalar(self):
        rng = np.random.default_rng(1)
        return rng.random(SHAPE).astype(np.float32)

    def test_clean_bundle_reports_no_attrition(self):
        _, attrition = qc_stats.profile_matrix(_bundle(), self._scalar(), AFFINE)
        funnel = qc_stats.attrition_funnel(attrition)
        assert funnel["any_attrition"] is False
        assert [stage["count"] for stage in funnel["stages"]] == [8, 8, 8, 8]
        assert funnel["streamline_retention"] == pytest.approx(1.0)
        assert funnel["point_retention"] == pytest.approx(1.0)

    def test_out_of_bounds_bundle_is_counted_at_the_right_gate(self):
        """A bundle entirely outside the grid dies at the in-bounds gate."""
        outside = [s + np.array([500.0, 0.0, 0.0]) for s in _bundle()]
        _, attrition = qc_stats.profile_matrix(outside, self._scalar(), AFFINE)
        funnel = qc_stats.attrition_funnel(attrition)
        counts = [stage["count"] for stage in funnel["stages"]]
        assert counts == [8, 8, 0, 0]
        assert funnel["stages"][2]["dropped"] == 8
        assert funnel["any_attrition"] is True
        assert funnel["streamline_retention"] == pytest.approx(0.0)

    def test_partial_point_loss_shows_as_point_retention_below_one(self):
        """Half of each streamline outside the grid: retention ~= 0.5."""
        half_out = [s + np.array([0.0, 0.0, 18.0]) for s in _bundle(z0=-18, z1=18)]
        _, attrition = qc_stats.profile_matrix(half_out, self._scalar(), AFFINE)
        funnel = qc_stats.attrition_funnel(attrition)
        assert 0.4 < funnel["point_retention"] < 0.6
        assert funnel["any_attrition"] is True
        # No streamline was lost — only points. The funnel must say so.
        assert funnel["n_contributing"] == funnel["n_input"]

    def test_stage_counts_are_monotonically_non_increasing(self):
        _, attrition = qc_stats.profile_matrix(_bundle(), self._scalar(), AFFINE)
        counts = [stage["count"]
                  for stage in qc_stats.attrition_funnel(attrition)["stages"]]
        assert counts == sorted(counts, reverse=True)

    def test_gate_label_follows_the_min_points_threshold(self):
        _, attrition = qc_stats.profile_matrix(_bundle(), self._scalar(), AFFINE)
        funnel = qc_stats.attrition_funnel(attrition, min_points=7)
        assert "7" in funnel["stages"][3]["label"]

    def test_inconsistent_counters_are_rejected(self):
        """The funnel is arithmetic over the counters; a bad dict must not pass."""
        bogus = {"n_input": 10, "n_too_short": 0, "n_no_inbounds": 0,
                 "n_insufficient_inbounds": 0, "n_contributing": 4,
                 "points_total": 10, "points_in_bounds": 10,
                 "point_retention": 1.0}
        with pytest.raises(AssertionError):
            qc_stats.attrition_funnel(bogus)

    def test_empty_bundle_yields_no_retention(self):
        funnel = qc_stats.attrition_funnel(
            qc_stats.profile_matrix([], self._scalar(), AFFINE)[1])
        assert funnel["n_input"] == 0
        assert funnel["streamline_retention"] is None
        assert funnel["any_attrition"] is False


class TestBootstrapMeanSe:
    """The SE behind every ``bootstrap_se`` the report serialises."""

    def test_constant_population_has_zero_se(self):
        """Known answer: with no variance to resample, the mean cannot move."""
        assert qc_stats.bootstrap_mean_se(np.full(200, 0.42)) == 0.0

    def test_recovers_the_analytic_standard_error(self):
        """A bootstrap of a mean must land on sigma/sqrt(n) at B=1000."""
        sample = np.random.default_rng(11).normal(0.5, 0.1, 400)
        analytic = sample.std(ddof=0) / np.sqrt(sample.size)
        se = qc_stats.bootstrap_mean_se(sample)
        assert abs(se - analytic) / analytic < 0.10

    def test_deterministic_for_a_given_seed(self):
        sample = np.random.default_rng(3).normal(0, 1, 200)
        assert (qc_stats.bootstrap_mean_se(sample)
                == qc_stats.bootstrap_mean_se(sample))

    def test_default_seed_is_not_process_dependent(self):
        """DEFAULT_SEED, never viz_rng: the latter is randomised per process."""
        from csttool.reproducibility.context import DEFAULT_SEED

        sample = np.random.default_rng(4).normal(0, 1, 150)
        assert (qc_stats.bootstrap_mean_se(sample)
                == qc_stats.bootstrap_mean_se(sample, seed=DEFAULT_SEED))

    def test_different_seeds_draw_differently(self):
        sample = np.random.default_rng(5).normal(0, 1, 150)
        assert (qc_stats.bootstrap_mean_se(sample, seed=1)
                != qc_stats.bootstrap_mean_se(sample, seed=2))

    def test_empty_population_is_none_not_zero(self):
        """A zero SE is a scientific claim and must never be fabricated."""
        assert qc_stats.bootstrap_mean_se([]) is None

    def test_caller_supplied_rng_is_used(self):
        sample = np.random.default_rng(6).normal(0, 1, 100)
        given = qc_stats.bootstrap_mean_se(sample, rng=np.random.default_rng(9))
        assert given == qc_stats.bootstrap_mean_se(sample, seed=9)


class TestSubsampleStabilityUnchangedByRefactor:
    """`subsample_stability` must be byte-identical after `bootstrap_mean_se`
    was extracted from it.

    The golden numbers below were captured from the pre-refactor implementation.
    They exist because the refactor could only break this one way: by re-seeding
    instead of passing the caller's own generator, which would silently shift
    every published saturation number while every other test stayed green.
    """

    GOLDEN_BOOTSTRAP = {
        "std": 0.01308894423773185,
        "p5": 0.4818637861808141,
        "p95": 0.5261642270783583,
    }
    GOLDEN_ESTIMATES_SUM = 150.95950794922135
    GOLDEN_FULL_ESTIMATE = 0.5025627881288528

    def _stability(self):
        scalar = np.random.default_rng(2).random(SHAPE).astype(np.float32)
        return qc_stats.subsample_stability(_bundle(n_streamlines=30), scalar, AFFINE)

    def test_bootstrap_block_matches_golden(self):
        assert self._stability()["bootstrap_full"] == self.GOLDEN_BOOTSTRAP

    def test_subsampling_draws_match_golden(self):
        stability = self._stability()
        assert float(stability["estimates"].sum()) == self.GOLDEN_ESTIMATES_SUM
        assert stability["full_estimate"] == self.GOLDEN_FULL_ESTIMATE


class TestPropagateLiSe:
    def test_zero_hemisphere_ses_give_zero_li_se(self):
        # Exactly-determined means cannot move the index. The residual is the
        # float noise of subtracting a constant array's own mean.
        assert qc_stats.propagate_li_se(0.5, 0.4, 0.0, 0.0) == pytest.approx(0.0, abs=1e-15)

    def test_larger_hemisphere_ses_give_a_larger_li_se(self):
        small = qc_stats.propagate_li_se(0.5, 0.4, 0.001, 0.001)
        large = qc_stats.propagate_li_se(0.5, 0.4, 0.010, 0.010)
        assert large > small

    def test_matches_the_delta_method_to_within_monte_carlo_error(self):
        """The Monte Carlo propagation must agree with the analytic first-order
        result — otherwise the draws are not propagating what they claim to."""
        left, right, sl, sr = 0.49, 0.46, 0.0012, 0.0022
        total = left + right
        analytic = np.hypot(2 * right / total**2 * sl, 2 * left / total**2 * sr)
        assert abs(qc_stats.propagate_li_se(left, right, sl, sr) - analytic) / analytic < 0.10

    def test_missing_se_is_none(self):
        assert qc_stats.propagate_li_se(0.5, 0.4, None, 0.01) is None
        assert qc_stats.propagate_li_se(0.5, 0.4, 0.01, None) is None

    def test_undefined_index_is_none(self):
        assert qc_stats.propagate_li_se(0.0, 0.0, 0.01, 0.01) is None

    def test_deterministic(self):
        assert (qc_stats.propagate_li_se(0.5, 0.4, 0.01, 0.02)
                == qc_stats.propagate_li_se(0.5, 0.4, 0.01, 0.02))
