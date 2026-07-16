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
        - fa: mean, std, median, profile, all sampled values
        - md: mean, std, median, profile, all sampled values
        - rd: mean, std, median, profile, all sampled values
        - ad: mean, std, median, profile, all sampled values
        - hemisphere: identification string
    """
    
    print(f"\nAnalyzing {hemisphere.upper()} CST...")
    
    metrics = {
        'hemisphere': hemisphere,
        'morphology': compute_morphology(streamlines, affine)
    }
    
    # Microstructural analysis requires affine
    if fa_map is not None and affine is not None:
        fa_values = sample_scalar_along_tract(streamlines, fa_map, affine)
        if len(fa_values) > 0:
            fa_profile = compute_tract_profile(streamlines, fa_map, affine, n_points=20)
            fa_localized = compute_localized_metrics(fa_profile)
            metrics['fa'] = {
                'mean': float(np.mean(fa_values)),
                'std': float(np.std(fa_values)),
                'median': float(np.median(fa_values)),
                'min': float(np.min(fa_values)),
                'max': float(np.max(fa_values)),
                'profile': fa_profile,
                'n_samples': len(fa_values),
                'pontine': fa_localized['pontine'],
                'plic': fa_localized['plic'],
                'precentral': fa_localized['precentral']
            }
            print(f"  FA: {metrics['fa']['mean']:.3f} ± {metrics['fa']['std']:.3f}")
        else:
            # Handle empty case
            metrics['fa'] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
                'profile': [], 'n_samples': 0,
                'pontine': 0.0, 'plic': 0.0, 'precentral': 0.0
            }
    
    if md_map is not None and affine is not None:
        md_values = sample_scalar_along_tract(streamlines, md_map, affine)
        if len(md_values) > 0:
            md_profile = compute_tract_profile(streamlines, md_map, affine, n_points=20)
            md_localized = compute_localized_metrics(md_profile)
            metrics['md'] = {
                'mean': float(np.mean(md_values)),
                'std': float(np.std(md_values)),
                'median': float(np.median(md_values)),
                'min': float(np.min(md_values)),
                'max': float(np.max(md_values)),
                'profile': md_profile,
                'n_samples': len(md_values),
                'pontine': md_localized['pontine'],
                'plic': md_localized['plic'],
                'precentral': md_localized['precentral']
            }
            print(f"  MD: {metrics['md']['mean']:.3e} ± {metrics['md']['std']:.3e}")
        else:
            # Handle empty case
            metrics['md'] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
                'profile': [], 'n_samples': 0,
                'pontine': 0.0, 'plic': 0.0, 'precentral': 0.0
            }
    
    if rd_map is not None and affine is not None:
        rd_values = sample_scalar_along_tract(streamlines, rd_map, affine)
        if len(rd_values) > 0:
            rd_profile = compute_tract_profile(streamlines, rd_map, affine, n_points=20)
            rd_localized = compute_localized_metrics(rd_profile)
            metrics['rd'] = {
                'mean': float(np.mean(rd_values)),
                'std': float(np.std(rd_values)),
                'median': float(np.median(rd_values)),
                'min': float(np.min(rd_values)),
                'max': float(np.max(rd_values)),
                'profile': rd_profile,
                'n_samples': len(rd_values),
                'pontine': rd_localized['pontine'],
                'plic': rd_localized['plic'],
                'precentral': rd_localized['precentral']
            }
            print(f"  RD: {metrics['rd']['mean']:.3e} ± {metrics['rd']['std']:.3e}")
        else:
            metrics['rd'] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
                'profile': [], 'n_samples': 0,
                'pontine': 0.0, 'plic': 0.0, 'precentral': 0.0
            }
    
    if ad_map is not None and affine is not None:
        ad_values = sample_scalar_along_tract(streamlines, ad_map, affine)
        if len(ad_values) > 0:
            ad_profile = compute_tract_profile(streamlines, ad_map, affine, n_points=20)
            ad_localized = compute_localized_metrics(ad_profile)
            metrics['ad'] = {
                'mean': float(np.mean(ad_values)),
                'std': float(np.std(ad_values)),
                'median': float(np.median(ad_values)),
                'min': float(np.min(ad_values)),
                'max': float(np.max(ad_values)),
                'profile': ad_profile,
                'n_samples': len(ad_values),
                'pontine': ad_localized['pontine'],
                'plic': ad_localized['plic'],
                'precentral': ad_localized['precentral']
            }
            print(f"  AD: {metrics['ad']['mean']:.3e} ± {metrics['ad']['std']:.3e}")
        else:
            metrics['ad'] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
                'profile': [], 'n_samples': 0,
                'pontine': 0.0, 'plic': 0.0, 'precentral': 0.0
            }
    
    return metrics


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


def sample_scalar_along_tract(streamlines, scalar_map, affine):
    """
    Sample scalar values at every point along all streamlines.

    The returned values are pooled across the whole bundle, so this is invariant to
    streamline orientation and to point order; it deliberately does not reorient (unlike
    `compute_tract_profile`). Summary statistics derived from it are therefore unaffected
    by AU9. Note they remain point-weighted, so longer streamlines contribute more
    samples - a separate concern (AU10).

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
    
    scalar_values = []
    
    for streamline in streamlines:
        for point in streamline:
            # Convert world coordinates to voxel coordinates
            voxel_coord = world_to_voxel(point, affine)
            
            # Check bounds
            if (0 <= voxel_coord[0] < scalar_map.shape[0] and
                0 <= voxel_coord[1] < scalar_map.shape[1] and
                0 <= voxel_coord[2] < scalar_map.shape[2]):
                
                scalar_value = scalar_map[voxel_coord[0], voxel_coord[1], voxel_coord[2]]
                scalar_values.append(scalar_value)
    
    return np.array(scalar_values)


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
        
        # Sample scalar values at each point
        streamline_scalars = []
        for point in streamline:
            voxel_coord = world_to_voxel(point, affine)
            
            if (0 <= voxel_coord[0] < scalar_map.shape[0] and
                0 <= voxel_coord[1] < scalar_map.shape[1] and
                0 <= voxel_coord[2] < scalar_map.shape[2]):
                
                scalar_value = scalar_map[voxel_coord[0], voxel_coord[1], voxel_coord[2]]
                streamline_scalars.append(scalar_value)
        
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

    Divides the tract profile into 3 anatomical regions by relative position:
    - Pontine: 0-35% of the tract
    - PLIC: 35-70%
    - Precentral: 70-100%

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
    lo, hi = int(0.35 * n), int(0.70 * n)

    return {
        'pontine': float(np.mean(profile_arr[:lo])),
        'plic': float(np.mean(profile_arr[lo:hi])),
        'precentral': float(np.mean(profile_arr[hi:]))
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
        print(f"  Mean: {fa['mean']:.3f} ± {fa['std']:.3f}")
        print(f"  Range: [{fa['min']:.3f}, {fa['max']:.3f}]")
        print(f"  Samples: {fa['n_samples']}")
    
    if 'md' in metrics:
        md = metrics['md']
        print(f"\nMean Diffusivity:")
        print(f"  Mean: {md['mean']:.3e} ± {md['std']:.3e}")
        print(f"  Range: [{md['min']:.3e}, {md['max']:.3e}]")
        print(f"  Samples: {md['n_samples']}")
    
    print(f"{'='*60}\n")