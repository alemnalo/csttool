"""
unilateral_analysis.py

Functions for analyzing a single CST hemisphere (left OR right).

This module computes:
- Morphological metrics (streamline count, length, volume)
- Microstructural metrics (FA, MD sampling)
- Tract profiles (along-the-tract distribution)
- Descriptive statistics
"""

import warnings

import numpy as np
from dipy.tracking.streamline import length
from dipy.tracking.utils import density_map


# Nominal extent of each anatomical region along the tract, as a fraction of tract length
# measured from the inferior (pontine) end. Single source of truth: `compute_localized_metrics`
# bins the profile with these, and the profile figures place their region labels with them, so
# the numbers and the pictures cannot drift apart.
#
# These boundaries are conventional, not validated against an atlas — the JHU ICBM-DTI-81
# landmark validation is outstanding. Treat the region names as nominal.
TRACT_REGIONS = (
    ('pontine', 0.0, 0.35),
    ('plic', 0.35, 0.70),
    ('precentral', 0.70, 1.0),
)


def analyze_cst_hemisphere(
    streamlines,
    fa_map=None,
    md_map=None,
    rd_map=None,
    ad_map=None,
    affine=None,
    hemisphere='unknown',
    tensor_valid_mask=None,
):
    """
    Complete analysis of a single CST hemisphere.
    
    Parameters
    ----------
    streamlines : Streamlines
        Streamlines for one hemisphere (left OR right)
    fa_map : ndarray, optional
        3D fractional anisotropy map
    md_map : ndarray, optional
        3D mean diffusivity map
    rd_map : ndarray, optional
        3D radial diffusivity map
    ad_map : ndarray, optional
        3D axial diffusivity map
    affine : ndarray, optional
        4x4 affine transformation matrix
    hemisphere : str
        'left' or 'right' for identification
        
    Returns
    -------
    metrics : dict
        Comprehensive metrics dictionary containing:
        - morphology: streamline count, length stats, volume
        - fa/md/rd/ad: per-scalar block with a length-unbiased headline
          (mean/std/median/min/max/n_streamlines, one vote per streamline), a preserved
          point-pool summary (mean_point_weighted … n_samples), and the along-tract
          profile with regional means (pontine/plic/precentral). See
          `_compute_scalar_metrics`.
        - hemisphere: identification string
    """
    
    print(f"\nAnalyzing {hemisphere.upper()} CST...")
    
    metrics = {
        'hemisphere': hemisphere,
        'morphology': compute_morphology(streamlines, affine)
    }
    
    # Microstructural analysis requires affine. The per-scalar block is identical across
    # scalars, so it is built once in `_compute_scalar_metrics` — the single source of
    # truth for the headline / point-pool / profile key set (previously this block was
    # copied verbatim per scalar, so any key change was 4x).
    _scalar_inputs = [
        ('fa', fa_map, False),
        ('md', md_map, True),
        ('rd', rd_map, True),
        ('ad', ad_map, True),
    ]
    for _name, _map, _is_diffusivity in _scalar_inputs:
        if _map is not None and affine is not None:
            metrics[_name] = _compute_scalar_metrics(
                streamlines,
                _map,
                affine,
                scalar_name=_name,
                tensor_valid_mask=tensor_valid_mask,
            )
            if metrics[_name]['n_streamlines'] > 0:
                _fmt = '.3e' if _is_diffusivity else '.3f'
                print(f"  {_name.upper()}: {metrics[_name]['mean']:{_fmt}} "
                      f"± {metrics[_name]['std']:{_fmt}}")
    
    return metrics


def _scalar_summary(values):
    """Descriptive stats of a 1-D array, or zeros if empty."""
    if len(values) == 0:
        return {k: 0.0 for k in ('mean', 'std', 'median', 'min', 'max')}
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }


def _compute_scalar_metrics(streamlines, scalar_map, affine, *,
                            scalar_name=None, tensor_valid_mask=None):
    """Build one scalar block with explicit, non-interchangeable populations.

    The headline gives each retained streamline one vote. The diagnostic point
    pool gives each valid point one vote. The profile gives each streamline one
    value per normalized node, reports mean plus IQR, and leaves nodes below the
    minimum support fraction missing rather than drawing through them.
    """
    # Local import: qc_stats imports orient_streamlines_inferior_to_superior,
    # sample_scalar_per_streamline and _sample_streamline_values from this module,
    # so a module-level import here would be circular.
    from . import qc_stats

    n_points = 20
    point_values, point_validity = sample_scalar_along_tract(
        streamlines,
        scalar_map,
        affine,
        tensor_valid_mask=tensor_valid_mask,
        scalar_name=scalar_name,
        return_validity=True,
    )
    streamline_means = sample_scalar_per_streamline(
        streamlines,
        scalar_map,
        affine,
        tensor_valid_mask=tensor_valid_mask,
        scalar_name=scalar_name,
    )

    matrix, attrition = qc_stats.profile_matrix(
        streamlines,
        scalar_map,
        affine,
        n_points=n_points,
        tensor_valid_mask=tensor_valid_mask,
        scalar_name=scalar_name,
    )
    dispersion = qc_stats.profile_dispersion(matrix, percentiles=(25, 50, 75))
    def _json_vector(values):
        if values is None:
            return [None] * n_points
        return [
            float(value) if np.isfinite(value) else None
            for value in np.asarray(values, dtype=float)
        ]
    if matrix.size:
        with np.errstate(invalid="ignore"):
            mean_profile = np.nanmean(matrix, axis=0)
        adequate = np.asarray(dispersion["support_fraction"]) >= 0.5
        profile = [
            float(value) if keep and np.isfinite(value) else None
            for value, keep in zip(mean_profile, adequate)
        ]
    else:
        profile = [None] * n_points
    localized = compute_localized_metrics(profile)

    headline = _scalar_summary(streamline_means)
    point_pool = _scalar_summary(point_values)

    empty_band = [None] * n_points
    return {
        **headline,
        'headline_weighting': (
            'one mean per retained streamline; each streamline has equal weight'
        ),
        'n_streamlines': int(len(streamline_means)),
        'bootstrap_se': qc_stats.bootstrap_mean_se(streamline_means),
        'mean_point_weighted': point_pool['mean'],
        'std_point_weighted': point_pool['std'],
        'median_point_weighted': point_pool['median'],
        'min_point_weighted': point_pool['min'],
        'max_point_weighted': point_pool['max'],
        'n_samples': int(len(point_values)),
        'point_pool_weighting': (
            'each valid sampled streamline point has equal weight'
        ),
        'sample_validity': point_validity,
        'profile': profile,
        'profile_center': 'mean',
        'profile_band': 'interquartile range',
        'profile_minimum_support_fraction': 0.5,
        'profile_p25': _json_vector(dispersion['p25']),
        'profile_median': _json_vector(dispersion['p50']),
        'profile_p75': _json_vector(dispersion['p75']),
        'profile_n': int(dispersion['n']),
        'node_support': dispersion['node_support'],
        'profile_attrition': attrition,
        'pontine': localized['pontine'],
        'plic': localized['plic'],
        'precentral': localized['precentral'],
        **_regional_bootstrap_se(matrix),
    }


def _regional_bootstrap_se(matrix):
    """Bootstrap SE of each regional mean, from the profile matrix rows.

    A region's value is the mean over its nodes of the per-node mean across
    streamlines, which is identically the mean across streamlines of each
    streamline's own regional mean. Resampling *that* per-streamline population
    is therefore the bootstrap of the number `compute_localized_metrics` reports,
    and it needs no extra sampling pass — the rows are already in hand.

    Returns ``{'pontine_se': ..., 'plic_se': ..., 'precentral_se': ...}``, each
    None for an empty bundle (a zero SE is never fabricated).
    """
    from . import qc_stats

    if matrix.size == 0:
        return {f'{name}_se': None for name, _, _ in TRACT_REGIONS}
    n = matrix.shape[1]
    out = {}
    for name, start, end in TRACT_REGIONS:
        region = matrix[:, int(start * n):int(end * n)]
        with np.errstate(invalid="ignore"):
            per_streamline = np.nanmean(region, axis=1)
        per_streamline = per_streamline[np.isfinite(per_streamline)]
        out[f'{name}_se'] = qc_stats.bootstrap_mean_se(per_streamline)
    return out


def compute_morphology(streamlines, affine):
    """
    Compute morphological properties of a streamline bundle.
    
    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines
    affine : ndarray
        4x4 affine transformation matrix
        
    Returns
    -------
    morphology : dict
        Dictionary containing:
        - n_streamlines: number of streamlines
        - mean_length: average streamline length in mm
        - median_length: median streamline length in mm (additive descriptive stat;
          the length LI uses mean_length, not this)
        - std_length: standard deviation of lengths
        - min_length: minimum streamline length
        - max_length: maximum streamline length
        - tract_volume: volume in mm³
        - bootstrap_se_length: bootstrap SE of mean_length, or None for an empty
          bundle. Length is a mean over a resamplable per-streamline population,
          so it carries an SE; ``tract_volume`` deliberately does not — it is a
          set-union voxel count, and bootstrapping it would estimate the
          variability of a coverage statistic while reading as an SE of a mean.
    """

    if len(streamlines) == 0:
        return {
            'n_streamlines': 0,
            'mean_length': 0.0,
            'median_length': 0.0,
            'std_length': 0.0,
            'min_length': 0.0,
            'max_length': 0.0,
            'tract_volume': 0.0,
            'bootstrap_se_length': None,
            'length_weighting': 'one geometric length per retained streamline',
            'volume_definition': (
                'union of nearest-voxel indices traversed by the bounded bundle'
            ),
        }
    
    # Compute streamline lengths
    lengths = np.array([length(s) for s in streamlines])
    
    # Compute tract volume by counting unique voxels
    # Get all points from all streamlines
    all_points = np.vstack(streamlines)
    
    # Transform to voxel coordinates using inverse affine
    inv_affine = np.linalg.inv(affine)
    all_points_hom = np.c_[all_points, np.ones(len(all_points))]
    voxel_coords = (inv_affine @ all_points_hom.T).T[:, :3]
    
    # Round to integer voxel indices
    voxel_indices = np.round(voxel_coords).astype(int)
    
    # Count unique voxels
    unique_voxels = np.unique(voxel_indices, axis=0)
    n_voxels = len(unique_voxels)
    
    # Compute voxel volume
    voxel_size = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    voxel_volume = np.prod(voxel_size)
    
    # Total tract volume
    tract_volume = n_voxels * voxel_volume
    
    # median_length is an additive descriptive statistic paired with mean_length;
    # the length laterality index still uses mean_length (see bilateral_analysis),
    # so adding the median changes no existing metric value. See
    # docs/explanation/design-decisions.md.
    from . import qc_stats

    morphology = {
        'n_streamlines': len(streamlines),
        'mean_length': float(np.mean(lengths)),
        'median_length': float(np.median(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': float(np.min(lengths)),
        'max_length': float(np.max(lengths)),
        'tract_volume': float(tract_volume),
        'bootstrap_se_length': qc_stats.bootstrap_mean_se(lengths),
        'length_weighting': 'one geometric length per retained streamline',
        'volume_definition': (
            'union of nearest-voxel indices traversed by the bounded bundle'
        ),
    }
    
    print(f"  Morphology: {morphology['n_streamlines']} streamlines, "
          f"{morphology['mean_length']:.1f} mm mean length, "
          f"{morphology['tract_volume']:.0f} mm³ volume")
    
    return morphology


def sample_scalar_nodes(points, scalar_map, affine, *,
                        tensor_valid_mask=None, scalar_name=None):
    """Trilinearly sample points with explicit support/validity reasons.

    When no tensor mask is supplied this retains the historical nearest-voxel
    behavior. Preferred tractometry supplies the persisted fit-validity mask;
    then all interpolation corners must lie inside that domain.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    scalar_map = np.asarray(scalar_map)
    reasons = {
        "out_of_bounds": 0,
        "outside_tensor_fit_domain": 0,
        "nonfinite": 0,
        "mathematically_invalid": 0,
    }
    if len(points) == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=bool), reasons

    homogeneous = np.c_[points, np.ones(len(points))]
    coords = (np.linalg.inv(affine) @ homogeneous.T).T[:, :3]
    values = np.full(len(points), np.nan, dtype=float)
    valid = np.zeros(len(points), dtype=bool)

    if tensor_valid_mask is None:
        rounded = np.round(coords).astype(int)
        inside = ((rounded >= 0) & (rounded < np.asarray(scalar_map.shape))).all(axis=1)
        reasons["out_of_bounds"] = int(np.count_nonzero(~inside))
        if inside.any():
            rv = rounded[inside]
            sampled = scalar_map[rv[:, 0], rv[:, 1], rv[:, 2]].astype(float)
            finite = np.isfinite(sampled)
            selected = np.flatnonzero(inside)
            values[selected[finite]] = sampled[finite]
            valid[selected[finite]] = True
            reasons["nonfinite"] = int(np.count_nonzero(~finite))
        return values, valid, reasons

    tensor_valid_mask = np.asarray(tensor_valid_mask, dtype=bool)
    if tensor_valid_mask.shape != scalar_map.shape:
        raise ValueError(
            "tensor validity mask and scalar map must share a grid, got "
            f"{tensor_valid_mask.shape} and {scalar_map.shape}"
        )
    lower = np.floor(coords).astype(int)
    upper = np.ceil(coords).astype(int)
    inside = ((lower >= 0) & (upper < np.asarray(scalar_map.shape))).all(axis=1)
    reasons["out_of_bounds"] = int(np.count_nonzero(~inside))

    support = np.zeros(len(points), dtype=bool)
    for index in np.flatnonzero(inside):
        x0, y0, z0 = lower[index]
        x1, y1, z1 = upper[index]
        support[index] = bool(
            tensor_valid_mask[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1].all()
        )
    reasons["outside_tensor_fit_domain"] = int(
        np.count_nonzero(inside & ~support)
    )

    usable = inside & support
    if usable.any():
        from scipy.ndimage import map_coordinates
        sampled = map_coordinates(
            scalar_map,
            coords[usable].T,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        ).astype(float)
        finite = np.isfinite(sampled)
        mathematically_valid = finite.copy()
        name = (scalar_name or "").lower()
        if name == "fa":
            mathematically_valid &= (sampled >= 0.0) & (sampled <= 1.0)
        elif name in {"md", "rd", "ad"}:
            mathematically_valid &= sampled >= 0.0
        selected = np.flatnonzero(usable)
        values[selected[mathematically_valid]] = sampled[mathematically_valid]
        valid[selected[mathematically_valid]] = True
        reasons["nonfinite"] = int(np.count_nonzero(~finite))
        reasons["mathematically_invalid"] = int(
            np.count_nonzero(finite & ~mathematically_valid)
        )
    return values, valid, reasons


def _sample_streamline_values(streamline, scalar_map, affine, *,
                              tensor_valid_mask=None, scalar_name=None):
    values, valid, _ = sample_scalar_nodes(
        streamline,
        scalar_map,
        affine,
        tensor_valid_mask=tensor_valid_mask,
        scalar_name=scalar_name,
    )
    return values[valid].tolist()


def sample_scalar_along_tract(streamlines, scalar_map, affine, *,
                              tensor_valid_mask=None, scalar_name=None,
                              return_validity=False):
    """Return the valid point pool, optionally with rejected-sample reasons."""
    all_values = []
    summary = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "rejected_by_reason": {
            "out_of_bounds": 0,
            "outside_tensor_fit_domain": 0,
            "nonfinite": 0,
            "mathematically_invalid": 0,
        },
        "interpolation": (
            "trilinear; all interpolation corners inside tensor-fit domain"
            if tensor_valid_mask is not None else "nearest voxel"
        ),
        "numeric_threshold_clipping": False,
        "observed_valid_extrema": {
            "minimum": None,
            "maximum": None,
            "interpretation": (
                "Observed mathematically valid extrema are retained and "
                "reported for review; they are not exclusion thresholds."
            ),
        },
    }
    for streamline_index, streamline in enumerate(streamlines):
        points = np.asarray(streamline, dtype=float)
        values, valid, reasons = sample_scalar_nodes(
            points,
            scalar_map,
            affine,
            tensor_valid_mask=tensor_valid_mask,
            scalar_name=scalar_name,
        )
        valid_indices = np.flatnonzero(valid)
        all_values.extend(values[valid])
        if len(valid_indices):
            extrema = summary["observed_valid_extrema"]
            for label, local_index in (
                ("minimum", int(np.argmin(values[valid]))),
                ("maximum", int(np.argmax(values[valid]))),
            ):
                point_index = int(valid_indices[local_index])
                candidate = {
                    "value": float(values[point_index]),
                    "streamline_index": int(streamline_index),
                    "point_index": point_index,
                    "world_coordinate_mm": [
                        float(coordinate) for coordinate in points[point_index]
                    ],
                }
                current = extrema[label]
                if (
                    current is None
                    or (label == "minimum"
                        and candidate["value"] < current["value"])
                    or (label == "maximum"
                        and candidate["value"] > current["value"])
                ):
                    extrema[label] = candidate
        summary["total_samples"] += int(len(values))
        summary["valid_samples"] += int(valid.sum())
        for reason, count in reasons.items():
            summary["rejected_by_reason"][reason] += int(count)
    summary["invalid_samples"] = (
        summary["total_samples"] - summary["valid_samples"]
    )
    summary["valid_fraction"] = (
        summary["valid_samples"] / summary["total_samples"]
        if summary["total_samples"] else None
    )
    result = np.asarray(all_values, dtype=float)
    return (result, summary) if return_validity else result


def sample_scalar_per_streamline(streamlines, scalar_map, affine, *,
                                 tensor_valid_mask=None, scalar_name=None):
    """Mean scalar per streamline; every retained streamline has one vote."""
    means = []
    for streamline in streamlines:
        vals = _sample_streamline_values(
            streamline,
            scalar_map,
            affine,
            tensor_valid_mask=tensor_valid_mask,
            scalar_name=scalar_name,
        )
        if vals:
            means.append(float(np.mean(vals)))
    return np.asarray(means, dtype=float)


def _end_to_end_delta(points, axis):
    """Mean coordinate of the last quartile minus that of the first, along `axis`."""
    k = max(1, len(points) // 4)
    return float(np.mean(points[-k:, axis]) - np.mean(points[:k, axis]))


def orient_streamlines_inferior_to_superior(streamlines):
    """
    Flip streamlines so each runs from its inferior end to its superior end.

    Tractography stores each streamline as ``[backward_from_seed][forward_from_seed]``, so a
    bundle mixes orientations. Averaging by point index across a mixed bundle aligns one
    streamline's pontine end with another's precentral end, biasing the tract profile and
    every regional metric derived from it (audit finding AU9).

    Consumers assume index 0 is the inferior (pontine) end: `compute_localized_metrics`
    bins the profile by index, and the profile plots label position 0 "Pontine Level". So
    the ordering is anchored to anatomy rather than made merely self-consistent between
    streamlines, which would leave the whole bundle liable to being flipped as a unit.

    Orientation is decided by comparing the mean Z of the first and last quartiles rather
    than the two endpoints alone. On healthy CST data the two rules agree on every
    streamline, but terminal points are the noisiest part of a streamline (that is where
    the stopping criterion fired) and the cortical end hooks laterally into the precentral
    gyrus, so the quartile mean is the more robust choice on the tortuous bundles this
    tool targets. It costs one extra mean per streamline and degrades to the endpoint rule
    for short streamlines.

    Parameters
    ----------
    streamlines : sequence of ndarray
        Streamlines in RASMM world coordinates (as returned by
        ``load_tractogram(path, 'same')``), where +Z is superior by the NIfTI standard.
        Because only the two ends of the same streamline are compared, the result is
        translation-invariant and therefore unaffected by the subject not being
        recentered (audit finding AU11).

    Returns
    -------
    oriented : list of ndarray
        A new list; the input streamlines are never modified in place.

    Warns
    -----
    UserWarning
        If the bundle does not run predominantly superior-inferior, in which case
        "index 0 is the inferior end" is not anatomically meaningful.
    """

    if len(streamlines) == 0:
        return []

    spans = np.array([
        np.abs(np.asarray(s)[-1] - np.asarray(s)[0])
        for s in streamlines if len(s) >= 2
    ])
    if len(spans) > 0:
        dominant = int(np.argmax(spans.mean(axis=0)))
        if dominant != 2:
            warnings.warn(
                f"Bundle runs predominantly along {'XYZ'[dominant]}, not Z: it is not a "
                "superior-inferior tract, so orienting it inferior-to-superior (and the "
                "pontine/precentral labels that assume it) is not meaningful.",
                UserWarning,
                stacklevel=2,
            )

    oriented = []
    for streamline in streamlines:
        points = np.asarray(streamline)
        if len(points) < 2:
            oriented.append(points)
            continue

        # Z decides. Ties fall through to Y then X, then keep the stored order, so the
        # result is deterministic and depends only on the streamline itself.
        for axis in (2, 1, 0):
            delta = _end_to_end_delta(points, axis)
            if delta != 0.0:
                oriented.append(points[::-1] if delta < 0 else points)
                break
        else:
            oriented.append(points)

    return oriented


def compute_tract_profile(streamlines, scalar_map, affine, n_points=20):
    """
    Compute normalized tract profile (average scalar along tract).
    
    This function samples scalar values along each streamline, normalizes
    each streamline to the same number of points, and averages across all
    streamlines to create a representative profile.

    Streamlines are reoriented inferior-to-superior first, so that point index i means the
    same anatomical position in every streamline (see
    `orient_streamlines_inferior_to_superior`). Without that step a mixed-orientation
    bundle averages one streamline's pontine end against another's precentral end. The
    input is not modified.

    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines in RASMM world coordinates (mm)
    scalar_map : ndarray
        3D scalar map (FA or MD)
    affine : ndarray
        4x4 affine transformation matrix
    n_points : int
        Number of points in output profile (default: 20)

    Returns
    -------
    profile : ndarray
        Average scalar values at n_points positions along the tract, ordered from the
        inferior (pontine) end to the superior (precentral) end.
    """

    if len(streamlines) == 0:
        return np.zeros(n_points)

    streamlines = orient_streamlines_inferior_to_superior(streamlines)

    all_profiles = []
    for streamline in streamlines:
        if len(streamline) < 2:
            continue
        streamline_scalars = _sample_streamline_values(
            streamline, scalar_map, affine
        )
        if len(streamline_scalars) < 5:
            continue
        if len(streamline_scalars) >= n_points:
            indices = np.linspace(
                0, len(streamline_scalars) - 1, n_points
            ).astype(int)
            normalized_profile = np.array(streamline_scalars)[indices]
        else:
            x_original = np.linspace(0, 1, len(streamline_scalars))
            x_target = np.linspace(0, 1, n_points)
            normalized_profile = np.interp(
                x_target, x_original, streamline_scalars
            )
        all_profiles.append(normalized_profile)

    if len(all_profiles) == 0:
        return np.zeros(n_points)
    return np.mean(all_profiles, axis=0).tolist()


def world_to_voxel(world_point, affine):
    """
    Convert world coordinates (mm) to voxel coordinates.

    Parameters
    ----------
    world_point : array-like
        3D point in world coordinates [x, y, z]
    affine : ndarray
        4x4 affine transformation matrix

    Returns
    -------
    voxel_coord : ndarray
        3D voxel coordinates [i, j, k] as integers
    """

    # Add homogeneous coordinate
    world_point_homogeneous = np.append(world_point, 1.0)

    # Apply inverse affine transformation
    voxel_coord_homogeneous = np.linalg.inv(affine) @ world_point_homogeneous

    # Convert to integer voxel coordinates
    voxel_coord = np.round(voxel_coord_homogeneous[:3]).astype(int)

    return voxel_coord


def compute_localized_metrics(profile):
    """
    Compute region-specific statistics from tract profile.

    Divides the tract profile into the regions defined by `TRACT_REGIONS`: pontine 0-35% of
    the tract, PLIC 35-70%, precentral 70-100%. Those boundaries are conventional rather than
    validated against an atlas, so the region names are nominal.

    This assumes the profile is ordered from the inferior (pontine) end to the superior
    (precentral) end, which `compute_tract_profile` guarantees by reorienting the bundle.
    On a profile built from a mixed-orientation bundle these bins are meaningless.

    Parameters
    ----------
    profile : list or ndarray
        Tract profile (scalar values along tract), as returned by
        `compute_tract_profile`. Any length of at least 3 is supported; the region
        boundaries scale with it, and at the default n_points=20 they fall at points
        0-6 / 7-13 / 14-19.

    Returns
    -------
    localized : dict
        Dictionary with 'pontine', 'plic', 'precentral' keys,
        each containing the mean value for that region
    """
    if profile is None or len(profile) < 3:  # need at least one point per region
        return {
            'pontine': 0.0,
            'plic': 0.0,
            'precentral': 0.0
        }

    profile_arr = np.array([
        np.nan if value is None else float(value) for value in profile
    ])
    n = len(profile_arr)
    localized = {}
    for name, start, end in TRACT_REGIONS:
        values = profile_arr[int(start * n):int(end * n)]
        localized[name] = (
            float(np.nanmean(values)) if np.isfinite(values).any() else None
        )
    return localized


def print_hemisphere_summary(metrics):
    """
    Print human-readable summary of hemisphere metrics.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from analyze_cst_hemisphere()
    """
    
    hemisphere = metrics['hemisphere'].upper()
    print(f"\n{'='*60}")
    print(f"{hemisphere} CST ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    morph = metrics['morphology']
    print(f"Streamline Count: {morph['n_streamlines']}")
    print(f"Mean Length: {morph['mean_length']:.1f} ± {morph['std_length']:.1f} mm")
    print(f"Length Range: [{morph['min_length']:.1f}, {morph['max_length']:.1f}] mm")
    print(f"Tract Volume: {morph['tract_volume']:.0f} mm³")
    
    if 'fa' in metrics:
        fa = metrics['fa']
        print(f"\nFractional Anisotropy:")
        print(f"  Mean: {fa['mean']:.3f} ± {fa['std']:.3f} "
              f"(per-streamline, n={fa['n_streamlines']})")
        print(f"  Range: [{fa['min']:.3f}, {fa['max']:.3f}]")
        print(f"  Point-weighted: {fa['mean_point_weighted']:.3f} ± "
              f"{fa['std_point_weighted']:.3f} (n={fa['n_samples']} samples)")
    
    if 'md' in metrics:
        md = metrics['md']
        print(f"\nMean Diffusivity:")
        print(f"  Mean: {md['mean']:.3e} ± {md['std']:.3e} "
              f"(per-streamline, n={md['n_streamlines']})")
        print(f"  Range: [{md['min']:.3e}, {md['max']:.3e}]")
        print(f"  Point-weighted: {md['mean_point_weighted']:.3e} ± "
              f"{md['std_point_weighted']:.3e} (n={md['n_samples']} samples)")
    
    print(f"{'='*60}\n")