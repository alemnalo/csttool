"""
qc_stats.py - numerical QC diagnostics behind the trust-chain QC figures.

This is the **science** layer for the QC panels in ``qc_figures``: it computes
numbers and returns arrays and dicts, never a ``Figure``. The split mirrors the
one ``csttool.viz.render`` already follows (renderers draw, they never create or
save figures), so each diagnostic is unit-testable against a synthetic bundle
with a known answer and the figure layer stays composition-only.

Each function answers exactly one question a reader asks before trusting a
subject's CST metrics:

* ``tissue_plausibility`` — are the reported scalars coming from voxels that
  behave like white matter, or from voxels contaminated by CSF partial volume?
* ``tangent_v1_angles``   — are the streamlines following the local principal
  diffusion direction, or being pushed through voxels where the tensor does not
  support them?
* ``profile_matrix``      — is the mean profile a consensus of the bundle, or an
  average over streamlines that disagree? (It also reports the attrition that
  ``compute_tract_profile`` currently discards silently.)
* ``attrition_funnel``    — how much of the bundle was silently discarded on the
  way from extraction to the published profile?
* ``subsample_stability`` — is the headline mean converged with respect to the
  number of streamlines, i.e. would a re-run give the same number?
* ``node_geometry`` / ``compare_node_geometry`` — does node *i* mean the same
  anatomical level in both hemispheres?

Sampling conventions are deliberately identical to the metrics they audit: the
same inferior-to-superior reorientation
(`orient_streamlines_inferior_to_superior`), the same nearest-neighbour
world-to-voxel lookup, the same drop of out-of-bounds points, and the same
resample-to-``n_points`` rule as `compute_tract_profile`. A diagnostic that
sampled differently from the metric would describe a different quantity.
"""

import numpy as np

from ...reproducibility.context import DEFAULT_SEED
from .unilateral_analysis import (
    orient_streamlines_inferior_to_superior,
    sample_scalar_per_streamline,
    _sample_streamline_values,
)

# A streamline must yield at least this many in-bounds samples to enter a
# profile. Mirrors the threshold hard-coded in `compute_tract_profile`, so the
# attrition this module reports is the attrition that actually happened.
MIN_PROFILE_POINTS = 5

# Default subsampling grid for the saturation curve.
DEFAULT_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
DEFAULT_REPEATS = 50

# Resamples behind every bootstrap standard error quoted in the report's JSON and
# CSV. QC-6's 50 is adequate for a plotted band but leaves the SE itself with
# ~10 % relative error, which is visible in the third decimal of a value the
# report serialises at full float precision. 1000 draws of a mean over ~800
# float64 values costs ~10 ms per scalar per hemisphere.
REPORT_BOOTSTRAP_REPEATS = 1000

# Free-water regime for the tissue-plausibility joint histogram. Free water at
# body temperature diffuses at ~3.0e-3 mm^2/s isotropically; a voxel whose MD is
# above 2.0e-3 mm^2/s *and* whose FA is below 0.2 cannot be dominated by intact
# white matter, whatever else it contains. Both thresholds are conventional
# partial-volume heuristics, not a tissue classifier — the figure reports the
# mass that falls there, it does not claim those voxels are pure CSF.
FREE_WATER_MD_MIN = 2.0e-3   # mm^2/s
FREE_WATER_FA_MAX = 0.2
# Histogram support. MD beyond 3.0e-3 is free water by any account, so the top
# bin is open-ended rather than clipping mass out of the figure.
TISSUE_FA_RANGE = (0.0, 1.0)
TISSUE_MD_RANGE = (0.0, 3.0e-3)
TISSUE_BINS = 60


# ---------------------------------------------------------------------------
# Shared resampling — must match `compute_tract_profile` exactly
# ---------------------------------------------------------------------------
def _resample_series(values, n_points):
    """Resample a 1D series to ``n_points`` using the profile's own rule.

    Index-``linspace`` downsample when the series is long enough, linear
    interpolation upsample otherwise — byte-for-byte the rule in
    `compute_tract_profile`, so ``profile_matrix(...).mean(axis=0)`` reproduces
    `compute_tract_profile` rather than merely approximating it.

    The input dtype is deliberately **not** promoted. Scalar maps are stored
    float32, so `compute_tract_profile` downsamples and averages in float32;
    forcing float64 here would leave the two disagreeing in the eighth decimal
    and make ``profile`` change the moment the report derives it from this
    matrix. The upsample branch returns float64 either way, because that is what
    ``np.interp`` returns in `compute_tract_profile` too.
    """
    values = np.asarray(values)
    if len(values) >= n_points:
        indices = np.linspace(0, len(values) - 1, n_points).astype(int)
        return values[indices]
    x_original = np.linspace(0, 1, len(values))
    x_target = np.linspace(0, 1, n_points)
    return np.interp(x_target, x_original, values)


def _resample_points(points, n_points):
    """Resample an (N, 3) point sequence to ``n_points`` with the same rule."""
    points = np.asarray(points, dtype=float)
    if len(points) >= n_points:
        indices = np.linspace(0, len(points) - 1, n_points).astype(int)
        return points[indices]
    return np.column_stack([
        _resample_series(points[:, axis], n_points) for axis in range(3)
    ])


def _world_to_voxel_batch(points_world, affine):
    """Vectorised nearest-neighbour world -> voxel, matching `world_to_voxel`.

    Same rounding rule as the scalar helper in ``unilateral_analysis``; batched
    so a whole streamline costs one matrix product instead of a Python loop.
    """
    points_world = np.asarray(points_world, dtype=float).reshape(-1, 3)
    homogeneous = np.column_stack([points_world, np.ones(len(points_world))])
    voxel = (np.linalg.inv(affine) @ homogeneous.T).T[:, :3]
    return np.round(voxel).astype(int)


def _in_bounds(voxel, shape):
    """Boolean mask of voxel coordinates inside ``shape``."""
    return np.all((voxel >= 0) & (voxel < np.asarray(shape[:3])), axis=1)


def _weighted_percentile(values, weights, q):
    """Percentile of ``values`` under non-negative ``weights``.

    ``np.percentile`` has no weighted form. Sorting once and interpolating the
    cumulative weight (midpoint convention, so a uniform weight vector agrees
    with ``np.percentile``'s linear interpolation to within one sample) keeps
    this exact enough to quote in a sidecar.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0 or weights.sum() <= 0:
        return None
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(np.asarray(q, dtype=float) / 100.0, cumulative, values))


# ---------------------------------------------------------------------------
# QC-1: tissue plausibility of the sampled voxels
# ---------------------------------------------------------------------------
def tissue_plausibility(fa, md, density, *, bins=TISSUE_BINS,
                        fa_range=TISSUE_FA_RANGE, md_range=TISSUE_MD_RANGE,
                        free_water_fa_max=FREE_WATER_FA_MAX,
                        free_water_md_min=FREE_WATER_MD_MIN):
    """Joint FA-MD distribution of the voxels the CST metrics are drawn from.

    Every voxel the bundle visits is weighted by its CST density, so the
    histogram describes the population the *reported* scalars average over
    rather than the anatomy an unweighted mask would describe. A compact cloud
    at FA 0.4-0.7 / MD ~0.8e-3 means the mean FA is a microstructural
    statement; mass dragged toward the free-water corner means an elevated MD or
    a depressed FA is partial volume, not pathology.

    The three volumes must share a grid — they come off the same FA affine in
    the pipeline, and comparing voxel-for-voxel across different grids would be
    meaningless, so a mismatch raises rather than broadcasting.

    Parameters
    ----------
    fa, md : ndarray, shape (X, Y, Z)
        Tensor scalars on the native grid. ``md`` in mm^2/s (the units csttool
        writes), i.e. white matter sits near 0.8e-3, not 0.8.
    density : ndarray, shape (X, Y, Z)
        CST streamline density in [0, 1]. Voxels at exactly 0 are excluded:
        they contribute no weight and would only inflate ``n_voxels``.
    bins : int
    fa_range, md_range : (float, float)
        Histogram support. Values outside are clipped into the edge bins so no
        weight silently leaves the figure; ``n_md_above_range`` reports how much
        was clipped at the top.

    Returns
    -------
    dict with ``histogram`` (bins x bins, FA along axis 0, MD along axis 1,
    weighted by density), ``fa_edges``, ``md_edges``, ``n_voxels``,
    ``density_mass``, ``centroid_fa`` / ``centroid_md`` (density-weighted
    means), ``fa_percentiles`` / ``md_percentiles`` (density-weighted 5/25/50/
    75/95), ``free_water_fraction`` (share of the density mass inside the
    free-water box), ``free_water_fa_max``, ``free_water_md_min`` and
    ``n_md_above_range``. Everything is None/0 for an empty bundle.
    """
    fa = np.asarray(fa, dtype=float)
    md = np.asarray(md, dtype=float)
    density = np.asarray(density, dtype=float)
    if not (fa.shape == md.shape == density.shape):
        raise ValueError(
            "fa, md and density must share a grid, got "
            f"{fa.shape}, {md.shape}, {density.shape}"
        )

    empty = {
        "histogram": np.zeros((bins, bins)),
        "fa_edges": np.linspace(*fa_range, bins + 1),
        "md_edges": np.linspace(*md_range, bins + 1),
        "n_voxels": 0, "density_mass": 0.0,
        "centroid_fa": None, "centroid_md": None,
        "fa_percentiles": None, "md_percentiles": None,
        "free_water_fraction": None,
        "free_water_fa_max": float(free_water_fa_max),
        "free_water_md_min": float(free_water_md_min),
        "n_md_above_range": 0,
    }

    visited = (density > 0) & np.isfinite(fa) & np.isfinite(md)
    if not np.any(visited):
        return empty

    fa_values = fa[visited]
    md_values = md[visited]
    weights = density[visited]

    fa_clipped = np.clip(fa_values, *fa_range)
    md_clipped = np.clip(md_values, *md_range)
    histogram, fa_edges, md_edges = np.histogram2d(
        fa_clipped, md_clipped, bins=bins, range=[list(fa_range), list(md_range)],
        weights=weights,
    )

    mass = float(weights.sum())
    in_free_water = (fa_values < free_water_fa_max) & (md_values > free_water_md_min)
    percentiles = (5, 25, 50, 75, 95)

    return {
        "histogram": histogram,
        "fa_edges": fa_edges,
        "md_edges": md_edges,
        "n_voxels": int(np.count_nonzero(visited)),
        "density_mass": mass,
        "centroid_fa": float(np.average(fa_values, weights=weights)),
        "centroid_md": float(np.average(md_values, weights=weights)),
        "fa_percentiles": {
            f"p{p}": _weighted_percentile(fa_values, weights, p) for p in percentiles
        },
        "md_percentiles": {
            f"p{p}": _weighted_percentile(md_values, weights, p) for p in percentiles
        },
        "free_water_fraction": float(weights[in_free_water].sum() / mass),
        "free_water_fa_max": float(free_water_fa_max),
        "free_water_md_min": float(free_water_md_min),
        "n_md_above_range": int(np.count_nonzero(md_values > md_range[1])),
    }


# ---------------------------------------------------------------------------
# QC-2: streamline tangent vs local V1
# ---------------------------------------------------------------------------
def tangent_v1_angles(streamlines, v1, affine, n_points=20,
                      min_points=MIN_PROFILE_POINTS):
    """Angle between each streamline's tangent and the local principal direction.

    The angle is *acute by construction*: the sign of a diffusion eigenvector is
    arbitrary, so ``|cos(theta)|`` is taken before the arccos and the result lies
    in [0, 90] degrees. A value near 0 means the step the tracker took agreed
    with the tensor at that voxel; a large value means it did not, which is what
    a crossing-fibre region looks like from the streamline's point of view.

    Parameters
    ----------
    streamlines : sequence of (N, 3) ndarray
        Streamlines in RASMM world coordinates (mm).
    v1 : ndarray, shape (X, Y, Z, 3)
        Principal eigenvector field in the **world** frame — i.e. the stored
        ``*_desc-V1_dwimap.nii.gz`` product, already rotated by
        ``csttool.spatial.rotate_vector_field_to_world``. Passing a voxel-frame
        field silently produces wrong angles on any oblique acquisition.
    affine : ndarray, shape (4, 4)
        Affine of the V1 grid.
    n_points : int
        Nodes per streamline in the returned matrix (default 20, the profile's).
    min_points : int
        Minimum usable samples for a streamline to contribute.

    Returns
    -------
    angles : ndarray, shape (n_contributing, n_points)
        Degrees, inferior (pontine) end first.
    meta : dict
        ``n_input``, ``n_contributing``, ``n_points_sampled``,
        ``n_points_out_of_bounds``, ``n_points_zero_vector``, and
        ``median_angle_deg`` over every retained sample.
    """
    v1 = np.asarray(v1)
    if v1.ndim == 5:  # stored product is (X, Y, Z, 1, 3)
        v1 = v1.squeeze(axis=3)
    if v1.ndim != 4 or v1.shape[-1] != 3:
        raise ValueError(
            f"v1 must be a (X, Y, Z, 3) vector field, got shape {v1.shape}"
        )

    meta = {
        "n_input": len(streamlines),
        "n_contributing": 0,
        "n_points_sampled": 0,
        "n_points_out_of_bounds": 0,
        "n_points_zero_vector": 0,
        "median_angle_deg": None,
    }
    if len(streamlines) == 0:
        return np.zeros((0, n_points)), meta

    oriented = orient_streamlines_inferior_to_superior(streamlines)
    shape = v1.shape[:3]

    rows = []
    pooled = []
    for streamline in oriented:
        points = np.asarray(streamline, dtype=float)
        if len(points) < 2:
            continue

        # Central-difference tangent; endpoints fall back to one-sided
        # differences, which is what np.gradient does.
        tangents = np.gradient(points, axis=0)
        tangent_norm = np.linalg.norm(tangents, axis=1)

        voxel = _world_to_voxel_batch(points, affine)
        inside = _in_bounds(voxel, shape)
        meta["n_points_out_of_bounds"] += int(np.count_nonzero(~inside))
        if not np.any(inside):
            continue

        vectors = np.zeros_like(points)
        vectors[inside] = v1[voxel[inside, 0], voxel[inside, 1], voxel[inside, 2]]
        vector_norm = np.linalg.norm(vectors, axis=1)

        usable = inside & (vector_norm > 0) & (tangent_norm > 0)
        meta["n_points_zero_vector"] += int(
            np.count_nonzero(inside & (vector_norm <= 0))
        )
        if np.count_nonzero(usable) < min_points:
            continue

        cosines = np.abs(np.einsum(
            "ij,ij->i", tangents[usable] / tangent_norm[usable, None],
            vectors[usable] / vector_norm[usable, None],
        ))
        degrees = np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0)))

        meta["n_points_sampled"] += int(degrees.size)
        pooled.append(degrees)
        rows.append(_resample_series(degrees, n_points))

    meta["n_contributing"] = len(rows)
    if pooled:
        meta["median_angle_deg"] = float(np.median(np.concatenate(pooled)))
    if not rows:
        return np.zeros((0, n_points)), meta
    return np.vstack(rows), meta


# ---------------------------------------------------------------------------
# QC-5 + QC-7: the per-streamline profile matrix and its attrition
# ---------------------------------------------------------------------------
def profile_matrix(streamlines, scalar_map, affine, n_points=20,
                   min_points=MIN_PROFILE_POINTS):
    """Per-streamline x per-node scalar matrix, plus the attrition that produced it.

    `compute_tract_profile` builds exactly this matrix and then collapses it with
    ``np.mean``, discarding both the dispersion and the record of what it threw
    away. ``matrix.mean(axis=0)`` here equals that function's output; the second
    return value makes the discards auditable.

    Returns
    -------
    matrix : ndarray, shape (n_contributing, n_points)
        Inferior (pontine) end first. Empty with shape (0, n_points) if nothing
        contributed.
    attrition : dict
        ``n_input``; ``n_too_short`` (fewer than 2 points, rejected before
        sampling); ``n_no_inbounds`` (every point outside the scalar grid);
        ``n_insufficient_inbounds`` (between 1 and ``min_points`` - 1 usable
        samples); ``n_contributing``; ``points_total`` and ``points_in_bounds``
        over the streamlines that reached sampling; and ``point_retention``, the
        ratio of the two (None when nothing was sampled).
    """
    attrition = {
        "n_input": len(streamlines),
        "n_too_short": 0,
        "n_no_inbounds": 0,
        "n_insufficient_inbounds": 0,
        "n_contributing": 0,
        "points_total": 0,
        "points_in_bounds": 0,
        "point_retention": None,
    }
    if len(streamlines) == 0:
        return np.zeros((0, n_points)), attrition

    oriented = orient_streamlines_inferior_to_superior(streamlines)

    rows = []
    for streamline in oriented:
        points = np.asarray(streamline)
        if len(points) < 2:
            attrition["n_too_short"] += 1
            continue

        values = _sample_streamline_values(points, scalar_map, affine)
        attrition["points_total"] += len(points)
        attrition["points_in_bounds"] += len(values)

        if len(values) == 0:
            attrition["n_no_inbounds"] += 1
            continue
        if len(values) < min_points:
            attrition["n_insufficient_inbounds"] += 1
            continue

        rows.append(_resample_series(values, n_points))

    attrition["n_contributing"] = len(rows)
    if attrition["points_total"] > 0:
        attrition["point_retention"] = (
            attrition["points_in_bounds"] / attrition["points_total"]
        )
    if not rows:
        return np.zeros((0, n_points)), attrition
    return np.vstack(rows), attrition


def profile_dispersion(matrix, percentiles=(5, 25, 50, 75, 95)):
    """Per-node percentiles of a profile matrix.

    Returns a dict keyed ``p5``, ``p25``, ``p50``, ``p75``, ``p95`` (following
    ``percentiles``) plus ``mean`` and ``n``, each an ``n_points`` vector except
    ``n`` which is the contributing streamline count. Empty input yields None
    vectors so callers can render a placeholder without a special case.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return {f"p{int(p)}": None for p in percentiles} | {"mean": None, "n": 0}
    out = {
        f"p{int(p)}": np.percentile(matrix, p, axis=0) for p in percentiles
    }
    out["mean"] = matrix.mean(axis=0)
    out["n"] = int(matrix.shape[0])
    return out


def attrition_funnel(attrition, min_points=MIN_PROFILE_POINTS):
    """Turn a ``profile_matrix`` attrition dict into an ordered survival funnel.

    The four stages are the four gates a streamline passes on its way from
    extraction to the published profile, in the order `profile_matrix` applies
    them. Each stage carries the count that *survived* it and the reason the
    difference was dropped, so the figure never has to re-derive the arithmetic
    and the invariant below is checkable.

    Returns
    -------
    dict with ``stages`` (list of ``{'label', 'count', 'dropped', 'reason'}``,
    the first having ``dropped`` 0), ``n_input``, ``n_contributing``,
    ``streamline_retention`` (contributing / input, None for an empty bundle),
    ``points_total``, ``points_in_bounds``, ``point_retention``, and
    ``any_attrition`` — False only when every streamline and every point
    survived, which is the expected state for a run whose scalar map shares the
    tracking grid.
    """
    n_input = int(attrition["n_input"])
    after_length = n_input - int(attrition["n_too_short"])
    after_inbounds = after_length - int(attrition["n_no_inbounds"])
    contributing = after_inbounds - int(attrition["n_insufficient_inbounds"])
    # Guards against a caller hand-assembling an inconsistent dict.
    assert contributing == int(attrition["n_contributing"]), (
        "attrition counters do not sum to n_contributing"
    )

    stages = [
        {"label": "extracted", "count": n_input, "dropped": 0, "reason": None},
        {"label": "≥ 2 points", "count": after_length,
         "dropped": int(attrition["n_too_short"]),
         "reason": "too short to sample"},
        {"label": "≥ 1 in-bounds sample", "count": after_inbounds,
         "dropped": int(attrition["n_no_inbounds"]),
         "reason": "no point inside the scalar grid"},
        {"label": f"≥ {min_points} in-bounds samples", "count": contributing,
         "dropped": int(attrition["n_insufficient_inbounds"]),
         "reason": f"fewer than {min_points} usable samples"},
    ]

    retention = attrition["point_retention"]
    return {
        "stages": stages,
        "n_input": n_input,
        "n_contributing": contributing,
        "streamline_retention": (contributing / n_input) if n_input else None,
        "points_total": int(attrition["points_total"]),
        "points_in_bounds": int(attrition["points_in_bounds"]),
        "point_retention": retention,
        "any_attrition": bool(contributing != n_input
                              or (retention is not None and retention < 1.0)),
    }


# ---------------------------------------------------------------------------
# Nonparametric bootstrap of a mean — shared by the QC-6 saturation figure and
# by every ``bootstrap_se`` the report serialises.
# ---------------------------------------------------------------------------
def _bootstrap_means(values, n_repeats, rng):
    """``n_repeats`` means of ``values`` resampled with replacement at full N.

    The single place the draws are made, so the report's standard errors and the
    QC-6 figure's bootstrap band cannot describe subtly different resamplings.
    The generator is passed in rather than seeded here: `subsample_stability`
    hands over the generator it has already drawn its subsampling grid from, so
    its draw sequence — and therefore every published saturation number — is
    unchanged by the extraction of this helper.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    return np.array([
        values[rng.integers(0, n, size=n)].mean() for _ in range(n_repeats)
    ])


def bootstrap_mean_se(values, n_repeats=REPORT_BOOTSTRAP_REPEATS,
                      seed=DEFAULT_SEED, rng=None):
    """Nonparametric bootstrap SE of the mean of ``values``.

    The standard error of the *reported* mean: how much it would move if a
    different subset of **these** values had been sampled. It is conditional on
    the retained bundle and says nothing about tracking, seeding, registration
    or acquisition variability — see the ``uncertainty`` block the report
    serialises alongside it.

    Parameters
    ----------
    values : array-like
        The population the mean is taken over — one value per streamline for the
        headline scalars, one per streamline per region for the regional means.
    n_repeats : int
        Resamples with replacement at full N.
    seed : int
        Used only when ``rng`` is None. ``DEFAULT_SEED``, never
        ``viz.utils.VIZ_SEED``: the latter derives from Python's builtin
        ``hash`` of a string and is randomised per process.
    rng : numpy.random.Generator, optional
        An existing generator to draw from, so a caller that has its own draw
        sequence can keep it intact.

    Returns
    -------
    float or None
        The SD across the resampled means, or None for an empty population — a
        zero SE is a scientific claim and is never fabricated.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    if rng is None:
        rng = np.random.default_rng(seed)
    return float(_bootstrap_means(values, n_repeats, rng).std())


def propagate_li_se(left_value, right_value, left_se, right_se,
                    n_repeats=REPORT_BOOTSTRAP_REPEATS, seed=DEFAULT_SEED):
    """SE of ``LI = (L - R) / (L + R)`` from the two hemispheres' bootstrap SEs.

    The two hemispheres are separate bundles, so their bootstrap distributions
    are independent and are resampled independently — jointly, from one
    generator, so the propagation is a pure function of its inputs. Each draw
    perturbs the two means by their own bootstrap SE and recomputes the index;
    the SD across draws is the LI's standard error.

    ``compute_laterality_indices`` sees only the metric dicts (never the
    streamlines), so the hemisphere bootstraps enter here summarised by their
    SEs rather than as raw draws. That is the whole content of "independent
    hemispheres": nothing correlates the two perturbations.

    Returns None when either SE is missing or the two values sum to zero (the LI
    itself is undefined there and ``compute_li`` already reports 0.0/no data).
    """
    if left_se is None or right_se is None:
        return None
    total = left_value + right_value
    if total == 0:
        return None
    rng = np.random.default_rng(seed)
    left_draws = left_value + left_se * rng.standard_normal(n_repeats)
    right_draws = right_value + right_se * rng.standard_normal(n_repeats)
    totals = left_draws + right_draws
    if np.any(totals == 0):
        return None
    return float(((left_draws - right_draws) / totals).std())


# ---------------------------------------------------------------------------
# QC-6: sampling saturation
# ---------------------------------------------------------------------------
def subsample_stability(streamlines, scalar_map, affine,
                        fractions=DEFAULT_FRACTIONS, n_repeats=DEFAULT_REPEATS,
                        seed=DEFAULT_SEED):
    """How much the headline mean moves when the bundle is subsampled.

    The statistic resampled is the **length-unbiased per-streamline mean** —
    the same population `sample_scalar_per_streamline` feeds to the report's
    headline ``mean`` and the global laterality indices (audit finding AU10), so
    the curve describes the number that is actually published.

    Two distinct quantities are returned, because they answer different halves of
    "would a re-run give the same number?":

    * the **subsampling curve** (draws without replacement at each fraction)
      shows how the estimate behaves as a function of bundle size, i.e. whether
      the attained N is on the flat part of the curve;
    * the **bootstrap at full N** (draws with replacement, size N) is the
      standard error of the published mean itself, which subsampling without
      replacement cannot express because at 100 % it has zero variance.

    Seeded from ``DEFAULT_SEED`` rather than ``viz.utils.VIZ_SEED``, to keep
    this estimate independent of figure subsampling. (``VIZ_SEED`` was itself
    unstable across processes until it moved to
    :func:`~csttool.reproducibility.context.derive_seed`; both are reproducible
    now, and this call site is left on ``DEFAULT_SEED`` so the published
    numbers do not shift.)

    Returns
    -------
    dict with ``n_streamlines`` (streamlines contributing at least one in-bounds
    sample), ``full_estimate``, ``fractions``, ``counts``, ``estimates``
    (n_fractions x n_repeats), per-fraction ``mean``/``std``/``p5``/``p95``, and
    ``bootstrap_full`` = {``std``, ``p5``, ``p95``}. All None/empty when the
    bundle yields no samples.
    """
    per_streamline = sample_scalar_per_streamline(streamlines, scalar_map, affine)
    n = int(per_streamline.size)

    empty = {
        "n_streamlines": n,
        "full_estimate": None,
        "fractions": list(fractions),
        "counts": [],
        "estimates": np.zeros((0, 0)),
        "mean": [], "std": [], "p5": [], "p95": [],
        "bootstrap_full": {"std": None, "p5": None, "p95": None},
    }
    if n == 0:
        return empty

    rng = np.random.default_rng(seed)
    counts, estimates = [], []
    for fraction in fractions:
        count = int(max(1, min(n, round(fraction * n))))
        counts.append(count)
        if count == n:
            # Without replacement, every draw is the whole bundle.
            estimates.append(np.full(n_repeats, float(per_streamline.mean())))
            continue
        draws = np.array([
            per_streamline[rng.choice(n, size=count, replace=False)].mean()
            for _ in range(n_repeats)
        ])
        estimates.append(draws)

    estimates = np.vstack(estimates)
    # Same generator, same draw order as before `_bootstrap_means` was extracted,
    # so every published saturation number is byte-identical.
    boot = _bootstrap_means(per_streamline, n_repeats, rng)

    return {
        "n_streamlines": n,
        "full_estimate": float(per_streamline.mean()),
        "fractions": list(fractions),
        "counts": counts,
        "estimates": estimates,
        "mean": estimates.mean(axis=1).tolist(),
        "std": estimates.std(axis=1).tolist(),
        "p5": np.percentile(estimates, 5, axis=1).tolist(),
        "p95": np.percentile(estimates, 95, axis=1).tolist(),
        "bootstrap_full": {
            "std": float(boot.std()),
            "p5": float(np.percentile(boot, 5)),
            "p95": float(np.percentile(boot, 95)),
        },
    }


# ---------------------------------------------------------------------------
# QC-8: node homology between hemispheres
# ---------------------------------------------------------------------------
def node_geometry(streamlines, n_points=20):
    """Where each profile node actually sits, in millimetres.

    The 20-node parameterisation is *relative*: node ``i`` is ``i/(n-1)`` of the
    way along whatever was reconstructed, not a fixed anatomical level. This
    reports the physical position of each node so two hemispheres can be checked
    for homology before their per-node values are differenced.

    Returns
    -------
    dict with ``n_streamlines``; ``node_world`` (n_points, 3) mean world
    coordinate per node; ``node_z`` (n_points,) mean superior-inferior
    coordinate; ``node_z_std``; ``arc_length_mm`` (n_points,) mean cumulative
    distance from the inferior end; ``lengths_mm`` per streamline; and
    ``length_mean``/``length_median``/``length_std``. Empty bundle yields None
    arrays and ``n_streamlines`` 0.
    """
    empty = {
        "n_streamlines": 0, "node_world": None, "node_z": None,
        "node_z_std": None, "arc_length_mm": None, "lengths_mm": np.array([]),
        "length_mean": None, "length_median": None, "length_std": None,
    }
    if len(streamlines) == 0:
        return empty

    oriented = orient_streamlines_inferior_to_superior(streamlines)

    node_points, node_arcs, lengths = [], [], []
    for streamline in oriented:
        points = np.asarray(streamline, dtype=float)
        if len(points) < 2:
            continue
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        lengths.append(float(cumulative[-1]))
        node_points.append(_resample_points(points, n_points))
        node_arcs.append(_resample_series(cumulative, n_points))

    if not node_points:
        return empty

    node_points = np.stack(node_points)          # (n_streamlines, n_points, 3)
    node_arcs = np.vstack(node_arcs)             # (n_streamlines, n_points)
    lengths = np.asarray(lengths, dtype=float)

    return {
        "n_streamlines": int(node_points.shape[0]),
        "node_world": node_points.mean(axis=0),
        "node_z": node_points[:, :, 2].mean(axis=0),
        "node_z_std": node_points[:, :, 2].std(axis=0),
        "arc_length_mm": node_arcs.mean(axis=0),
        "lengths_mm": lengths,
        "length_mean": float(lengths.mean()),
        "length_median": float(np.median(lengths)),
        "length_std": float(lengths.std()),
    }


def compare_node_geometry(left_streamlines, right_streamlines, n_points=20):
    """Left/right node homology: is node ``i`` the same anatomical level?

    Returns ``{'left', 'right', 'z_difference_mm', 'arc_difference_mm',
    'max_abs_z_difference_mm', 'length_difference_mm'}``. The differences are
    left minus right, per node, and are None when either bundle is empty.

    A non-zero ``max_abs_z_difference_mm`` is the quantity that invalidates the
    regional laterality indices: if the left bundle sits systematically higher at
    node 7, the ``plic`` LI is comparing two different anatomical levels.
    """
    left = node_geometry(left_streamlines, n_points=n_points)
    right = node_geometry(right_streamlines, n_points=n_points)

    result = {
        "left": left, "right": right,
        "z_difference_mm": None, "arc_difference_mm": None,
        "max_abs_z_difference_mm": None, "length_difference_mm": None,
    }
    if left["node_z"] is None or right["node_z"] is None:
        return result

    z_difference = left["node_z"] - right["node_z"]
    result["z_difference_mm"] = z_difference
    result["arc_difference_mm"] = left["arc_length_mm"] - right["arc_length_mm"]
    result["max_abs_z_difference_mm"] = float(np.max(np.abs(z_difference)))
    result["length_difference_mm"] = left["length_mean"] - right["length_mean"]
    return result
