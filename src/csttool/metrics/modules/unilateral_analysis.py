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
    hemisphere='unknown'
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
            metrics[_name] = _compute_scalar_metrics(streamlines, _map, affine)
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


def _compute_scalar_metrics(streamlines, scalar_map, affine):
    """Build the per-scalar metric block emitted by `analyze_cst_hemisphere`.

    Two summaries coexist, both honestly named so the headline and its preserved
    length-biased counterpart cannot be confused (audit finding AU10):

    - **Headline** (``mean``/``std``/``median``/``min``/``max``/``n_streamlines``):
      per-streamline. Each streamline contributes one value — the mean of its sampled
      points — so the summary is length-unbiased. The report's global table and the global
      laterality indices consume this. ``std`` is the between-streamline SD of streamline
      means (the spread of the quantity whose mean is reported), not the point-pool
      scatter; ``min``/``max`` bound the same per-streamline distribution.
    - **Point-pool** (``mean_point_weighted`` … ``n_samples``): every sampled point,
      length-biased (longer streamlines vote more). Preserved for QC and bias auditing
      per the AU5/AU24 instinct of labelling constructed numbers rather than removing
      them; not used as the headline.
    - **Profile** (``profile``/``pontine``/``plic``/``precentral``): per-streamline-
      normalised along-tract curve, arc-length resampled; unchanged.

    The headline uses the per-streamline mean rather than the profile-derived mean because
    the profile resamples to a fixed arc length and drops streamlines with fewer than five
    valid points — a different population. See `docs/explanation/design-decisions.md`.
    """
    point_values = sample_scalar_along_tract(streamlines, scalar_map, affine)
    streamline_means = sample_scalar_per_streamline(streamlines, scalar_map, affine)
    profile = compute_tract_profile(streamlines, scalar_map, affine, n_points=20)
    localized = compute_localized_metrics(profile)

    headline = _scalar_summary(streamline_means)
    point_pool = _scalar_summary(point_values)

    return {
        **headline,
        'n_streamlines': int(len(streamline_means)),
        'mean_point_weighted': point_pool['mean'],
        'std_point_weighted': point_pool['std'],
        'median_point_weighted': point_pool['median'],
        'min_point_weighted': point_pool['min'],
        'max_point_weighted': point_pool['max'],
        'n_samples': int(len(point_values)),
        'profile': profile,
        'pontine': localized['pontine'],
        'plic': localized['plic'],
        'precentral': localized['precentral'],
    }


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
        - std_length: standard deviation of lengths
        - min_length: minimum streamline length
        - max_length: maximum streamline length
        - tract_volume: volume in mm³
    """
    
    if len(streamlines) == 0:
        return {
            'n_streamlines': 0,
            'mean_length': 0.0,
            'std_length': 0.0,
            'min_length': 0.0,
            'max_length': 0.0,
            'tract_volume': 0.0
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
    
    morphology = {
        'n_streamlines': len(streamlines),
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': float(np.min(lengths)),
        'max_length': float(np.max(lengths)),
        'tract_volume': float(tract_volume)
    }
    
    print(f"  Morphology: {morphology['n_streamlines']} streamlines, "
          f"{morphology['mean_length']:.1f} mm mean length, "
          f"{morphology['tract_volume']:.0f} mm³ volume")
    
    return morphology


def _sample_streamline_values(streamline, scalar_map, affine):
    """In-bounds scalar values at each point of a single streamline.

    Shared by `sample_scalar_along_tract`, `sample_scalar_per_streamline` and
    `compute_tract_profile` so the world→voxel lookup, bounds check and
    nearest-neighbour sampling are defined once. Values from out-of-bounds points are
    dropped.
    """
    values = []
    for point in streamline:
        voxel_coord = world_to_voxel(point, affine)
        if (0 <= voxel_coord[0] < scalar_map.shape[0] and
                0 <= voxel_coord[1] < scalar_map.shape[1] and
                0 <= voxel_coord[2] < scalar_map.shape[2]):
            values.append(scalar_map[voxel_coord[0], voxel_coord[1], voxel_coord[2]])
    return values


def sample_scalar_along_tract(streamlines, scalar_map, affine):
    """Sample scalar values at every point along all streamlines (point pool).

    Every in-bounds point of every streamline is pooled into one flat array, so this is
    invariant to streamline orientation and to point order; it deliberately does not
    reorient (unlike `compute_tract_profile`). Summary statistics derived from it are
    therefore unaffected by AU9.

    The pool is **point-weighted**: a streamline with twice as many points casts twice
    as many votes. When streamline length is correlated with the scalar (it is, weakly,
    on in-vivo CST data) this biases the mean. `sample_scalar_per_streamline` is the
    length-unbiased counterpart (one vote per streamline); the report's headline mean and
    the global laterality indices use that one, not this. This function is retained for
    QC, for the reproducibility stability tests, and as the basis of the preserved
    ``mean_point_weighted`` summary.

    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines in world coordinates (mm)
    scalar_map : ndarray
        3D scalar map (e.g., FA or MD)
    affine : ndarray
        4x4 affine transformation matrix

    Returns
    -------
    scalar_values : ndarray
        Array of all sampled scalar values (flattened across all streamlines)
    """
    if len(streamlines) == 0:
        return np.array([])

    all_values = []
    for streamline in streamlines:
        all_values.extend(_sample_streamline_values(streamline, scalar_map, affine))
    return np.array(all_values)


def sample_scalar_per_streamline(streamlines, scalar_map, affine):
    """Mean scalar per streamline (one value per streamline, length-unbiased).

    Each streamline that has at least one in-bounds point contributes exactly one value —
    the mean of its sampled points — regardless of how many points it has. Summary
    statistics derived from the returned array are therefore **not** weighted by
    streamline length, unlike `sample_scalar_along_tract`. This is the population the
    report's headline ``mean``/``std``/``median``/``min``/``max`` and the global
    laterality indices describe (audit finding AU10).

    A streamline's point-mean is order-invariant, so this is unaffected by AU9 as well.

    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines in world coordinates (mm)
    scalar_map : ndarray
        3D scalar map (e.g., FA or MD)
    affine : ndarray
        4x4 affine transformation matrix

    Returns
    -------
    streamline_means : ndarray
        Mean scalar value for each streamline with at least one in-bounds point.
    """
    if len(streamlines) == 0:
        return np.array([])

    means = []
    for streamline in streamlines:
        vals = _sample_streamline_values(streamline, scalar_map, affine)
        if len(vals) > 0:
            means.append(float(np.mean(vals)))
    return np.array(means)


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

        # Sample scalar values at each point (shared per-point lookup)
        streamline_scalars = _sample_streamline_values(streamline, scalar_map, affine)

        if len(streamline_scalars) < 5:  # Need minimum points
            continue
        
        # Normalize to n_points
        if len(streamline_scalars) >= n_points:
            # Downsample
            indices = np.linspace(0, len(streamline_scalars)-1, n_points).astype(int)
            normalized_profile = np.array(streamline_scalars)[indices]
        else:
            # Upsample with interpolation
            x_original = np.linspace(0, 1, len(streamline_scalars))
            x_target = np.linspace(0, 1, n_points)
            normalized_profile = np.interp(x_target, x_original, streamline_scalars)
        
        all_profiles.append(normalized_profile)
    
    if len(all_profiles) == 0:
        return np.zeros(n_points)
    
    # Average across all streamlines
    final_profile = np.mean(all_profiles, axis=0)
    
    return final_profile.tolist()  # Convert to list for JSON serialization


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

    profile_arr = np.array(profile)
    n = len(profile_arr)

    return {
        name: float(np.mean(profile_arr[int(start * n):int(end * n)]))
        for name, start, end in TRACT_REGIONS
    }


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