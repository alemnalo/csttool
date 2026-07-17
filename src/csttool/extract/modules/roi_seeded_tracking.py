"""
roi_seeded_tracking.py

ROI-seeded tractography for targeted tract extraction.
Seeds streamlines from specific ROIs and filters by target ROI traversal.

This approach is more effective than whole-brain + endpoint filtering
for long-range tracts like the CST that pass through crossing fiber regions.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines, length as streamline_length
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.direction import peaks_from_model
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere


def streamline_passes_through(streamline, mask, affine):
    """
    Check if any point along a streamline passes through a binary mask.
    
    Parameters
    ----------
    streamline : ndarray, shape (N, 3)
        Streamline points in world coordinates (mm).
    mask : ndarray
        3D binary mask.
    affine : ndarray, shape (4, 4)
        Affine matrix for the mask.
        
    Returns
    -------
    passes : bool
        True if streamline passes through mask.
    """
    inv_affine = np.linalg.inv(affine)
    ones = np.ones((len(streamline), 1))
    pts_h = np.hstack([streamline, ones])
    voxels = np.round((pts_h @ inv_affine.T)[:, :3]).astype(int)
    
    for vox in voxels:
        if (0 <= vox[0] < mask.shape[0] and
            0 <= vox[1] < mask.shape[1] and
            0 <= vox[2] < mask.shape[2]):
            if mask[vox[0], vox[1], vox[2]] > 0:
                return True
    return False


def generate_seeds_from_mask(mask, affine, density=2):
    """
    Generate seed points from a binary mask.
    
    Parameters
    ----------
    mask : ndarray
        3D binary mask defining seed region.
    affine : ndarray, shape (4, 4)
        Affine matrix for the mask.
    density : int, optional
        Number of seeds per voxel dimension. Default is 2.
        density=2 means 8 seeds per voxel (2³).
        
    Returns
    -------
    seeds : ndarray, shape (N, 3)
        Seed points in world coordinates (mm).
    """
    seeds = utils.seeds_from_mask(mask, affine, density=density)
    return seeds


def track_from_seeds(
    seeds,
    direction_getter,
    stopping_criterion,
    affine,
    step_size=0.5,
    return_all=False,
    random_seed=None
):
    """
    Perform deterministic tractography from specified seed points.

    Parameters
    ----------
    seeds : ndarray, shape (N, 3)
        Seed points in world coordinates.
    direction_getter : DirectionGetter
        DIPY direction getter (from peaks or ODF model).
    stopping_criterion : StoppingCriterion
        DIPY stopping criterion.
    affine : ndarray, shape (4, 4)
        Affine matrix.
    step_size : float, optional
        Step size in mm. Default is 0.5.
    return_all : bool, optional
        If True, return all streamlines including invalid ones.
    random_seed : int, optional
        Random seed for reproducible tracking. If None, results may vary
        between runs. Callers in the csttool pipeline should pass
        ``ctx.rng_tracking_seed()`` rather than a literal.

    Returns
    -------
    streamlines : Streamlines
        Generated streamlines.
    """
    # Note: max_angle is NOT a parameter of LocalTracking
    # Angular constraints are handled by the direction_getter (peaks)
    # via min_separation_angle during peak extraction
    streamline_generator = LocalTracking(
        direction_getter,
        stopping_criterion,
        seeds,
        affine,
        step_size=step_size,
        return_all=return_all,
        random_seed=random_seed
    )

    streamlines = Streamlines(streamline_generator)
    return streamlines


def filter_by_target_roi(streamlines, target_mask, affine, verbose=True):
    """
    Filter streamlines that pass through a target ROI.
    
    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines.
    target_mask : ndarray
        3D binary mask of target ROI.
    affine : ndarray, shape (4, 4)
        Affine matrix for the mask.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    filtered : Streamlines
        Streamlines passing through target ROI.
    indices : ndarray
        Indices of kept streamlines.
    """
    kept_indices = []
    
    for i, sl in enumerate(streamlines):
        if streamline_passes_through(sl, target_mask, affine):
            kept_indices.append(i)
        
        if verbose and (i + 1) % 10000 == 0:
            print(f"    Processed {i + 1:,} / {len(streamlines):,}...")
    
    kept_indices = np.array(kept_indices, dtype=np.intp)
    
    if len(kept_indices) > 0:
        filtered = Streamlines([streamlines[i] for i in kept_indices])
    else:
        filtered = Streamlines()
    
    if verbose:
        print(f"    Filtered: {len(streamlines):,} → {len(filtered):,} streamlines")
    
    return filtered, kept_indices


def filter_by_length(streamlines, min_length=20.0, max_length=200.0, verbose=True):
    """
    Filter streamlines by length.
    
    Parameters
    ----------
    streamlines : Streamlines
        Input streamlines.
    min_length : float
        Minimum length in mm.
    max_length : float
        Maximum length in mm.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    filtered : Streamlines
        Length-filtered streamlines.
    """
    lengths = np.array([streamline_length(sl) for sl in streamlines])
    mask = (lengths >= min_length) & (lengths <= max_length)
    
    filtered = Streamlines([sl for sl, keep in zip(streamlines, mask) if keep])
    
    if verbose:
        print(f"    Length filter ({min_length}-{max_length}mm): {len(streamlines):,} → {len(filtered):,}")
    
    return filtered


def extract_cst_roi_seeded(
    data,
    gtab,
    affine,
    brain_mask,
    motor_left_mask,
    motor_right_mask,
    brainstem_mask,
    fa_map=None,
    fa_threshold=0.15,
    seed_density=2,
    step_size=0.5,
    min_length=30.0,
    max_length=200.0,
    sh_order=6,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    random_seed=None,
    verbose=True,
    npeaks=1,
    fit_method='WLS'
):
    """
    Extract bilateral CST using ROI-seeded tractography.
    
    This method:
    1. Seeds streamlines from motor cortex (left and right separately)
    2. Tracks bidirectionally using CSA-ODF direction field
    3. Filters streamlines that reach the brainstem
    
    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N)
        DWI data.
    gtab : GradientTable
        Gradient table.
    affine : ndarray, shape (4, 4)
        Affine matrix.
    brain_mask : ndarray
        3D binary brain mask.
    motor_left_mask : ndarray
        3D binary mask for left motor cortex (seed ROI).
    motor_right_mask : ndarray
        3D binary mask for right motor cortex (seed ROI).
    brainstem_mask : ndarray
        3D binary mask for brainstem (target ROI).
    fa_map : ndarray, optional
        Pre-computed FA map. If None, will be computed.
    fa_threshold : float, optional
        FA threshold for stopping criterion. Default is 0.15.
    seed_density : int, optional
        Seeds per voxel dimension. Default is 2.
    step_size : float, optional
        Tracking step size in mm. Default is 0.5.
    min_length : float, optional
        Minimum streamline length in mm. Default is 30.
    max_length : float, optional
        Maximum streamline length in mm. Default is 200.
    sh_order : int, optional
        Spherical harmonic order for CSA-ODF. Default is 6.
    relative_peak_threshold : float, optional
        Threshold for peak extraction (relative to max). Default is 0.5.
        Lower values allow more peaks, enabling better crossing fiber handling.
    min_separation_angle : int, optional
        Minimum angle between peaks in degrees. Default is 25.
        Lower values allow sharper turns during tracking.
    random_seed : int, optional
        Random seed for reproducible tracking. Both hemispheres are tracked
        with the same seed, so the RNG cannot itself become a source of L/R
        difference. If None, results vary between runs.
    verbose : bool, optional
        Print progress information.
    npeaks : int, optional
        Number of peaks to extract per voxel. Default is 1.
    fit_method : str, optional
        DTI fit method passed to TensorModel when fa_map is not provided.
        Default 'WLS' (consistent with the pinned default in fit_tensors).

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'cst_left': Left CST streamlines
        - 'cst_right': Right CST streamlines
        - 'cst_combined': All CST streamlines
        - 'stats': Extraction statistics
        - 'parameters': Tracking parameters used
    """
    from dipy.reconst.dti import TensorModel
    
    if verbose:
        print("=" * 60)
        print("ROI-SEEDED CST EXTRACTION")
        print("=" * 60)
        if random_seed is not None:
            print(f"  → Random seed: {random_seed}")
        else:
            print("  → Random seeding disabled - results will vary between runs")

    # -------------------------------------------------------------------------
    # Step 1: Compute FA if not provided
    # -------------------------------------------------------------------------
    if fa_map is None:
        if verbose:
            print("\n[Step 1/5] Computing FA map...")
        tensor_model = TensorModel(gtab, fit_method=fit_method)
        tensor_fit = tensor_model.fit(data, mask=brain_mask)
        fa_map = tensor_fit.fa
        fa_map = np.clip(fa_map, 0, 1)
    else:
        if verbose:
            print("\n[Step 1/5] Using provided FA map...")
    
    # -------------------------------------------------------------------------
    # Step 2: Estimate fiber directions using CSA-ODF
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 2/5] Estimating fiber directions (CSA-ODF)...")
    
    # Validate SH order based on available gradient directions
    from csttool.tracking.modules.estimate_directions import validate_sh_order
    validated_sh_order = validate_sh_order(gtab, sh_order, verbose=verbose)
    
    csa_model = CsaOdfModel(gtab, sh_order=validated_sh_order)
    
    # Create white matter mask for direction estimation
    wm_mask = fa_map > fa_threshold
    
    # Peak extraction - angular constraints are set HERE, not in LocalTracking
    peaks = peaks_from_model(
        csa_model,
        data,
        default_sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        mask=wm_mask,
        npeaks=npeaks
    )
    
    if verbose:
        print(f"    Peaks computed for {np.sum(wm_mask):,} voxels")
    
    # -------------------------------------------------------------------------
    # Step 3: Setup stopping criterion
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[Step 3/5] Setting up stopping criterion (FA > {fa_threshold})...")
    
    stopping_criterion = ThresholdStoppingCriterion(fa_map, fa_threshold)
    
    # -------------------------------------------------------------------------
    # Step 4: Extract Left CST
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 4/5] Extracting Left CST...")
        print(f"    Generating seeds from left motor cortex (density={seed_density})...")
    
    left_seeds = generate_seeds_from_mask(motor_left_mask, affine, density=seed_density)
    
    if verbose:
        print(f"    Seeds: {len(left_seeds):,}")
        print(f"    Tracking...")
    
    left_streamlines_raw = track_from_seeds(
        left_seeds,
        peaks,
        stopping_criterion,
        affine,
        step_size=step_size,
        random_seed=random_seed
    )
    
    if verbose:
        print(f"    Generated: {len(left_streamlines_raw):,} streamlines")
        print(f"    Filtering by length...")
    
    left_streamlines_length = filter_by_length(
        left_streamlines_raw, 
        min_length=min_length, 
        max_length=max_length,
        verbose=verbose
    )
    
    if verbose:
        print(f"    Filtering by brainstem traversal...")
    
    cst_left, _ = filter_by_target_roi(
        left_streamlines_length,
        brainstem_mask,
        affine,
        verbose=verbose
    )
    
    if verbose:
        print(f"    ✓ Left CST: {len(cst_left):,} streamlines")
    
    # -------------------------------------------------------------------------
    # Step 5: Extract Right CST
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 5/5] Extracting Right CST...")
        print(f"    Generating seeds from right motor cortex (density={seed_density})...")
    
    right_seeds = generate_seeds_from_mask(motor_right_mask, affine, density=seed_density)
    
    if verbose:
        print(f"    Seeds: {len(right_seeds):,}")
        print(f"    Tracking...")
    
    right_streamlines_raw = track_from_seeds(
        right_seeds,
        peaks,
        stopping_criterion,
        affine,
        step_size=step_size,
        random_seed=random_seed
    )
    
    if verbose:
        print(f"    Generated: {len(right_streamlines_raw):,} streamlines")
        print(f"    Filtering by length...")
    
    right_streamlines_length = filter_by_length(
        right_streamlines_raw,
        min_length=min_length,
        max_length=max_length,
        verbose=verbose
    )
    
    if verbose:
        print(f"    Filtering by brainstem traversal...")
    
    cst_right, _ = filter_by_target_roi(
        right_streamlines_length,
        brainstem_mask,
        affine,
        verbose=verbose
    )
    
    if verbose:
        print(f"    ✓ Right CST: {len(cst_right):,} streamlines")
    
    # -------------------------------------------------------------------------
    # Combine results
    # -------------------------------------------------------------------------
    cst_combined = Streamlines(list(cst_left) + list(cst_right))
    
    stats = {
        'left_seeds': len(left_seeds),
        'right_seeds': len(right_seeds),
        'left_raw': len(left_streamlines_raw),
        'right_raw': len(right_streamlines_raw),
        'left_after_length': len(left_streamlines_length),
        'right_after_length': len(right_streamlines_length),
        'cst_left_count': len(cst_left),
        'cst_right_count': len(cst_right),
        'cst_total_count': len(cst_combined),
        'left_yield': len(cst_left) / len(left_seeds) * 100 if len(left_seeds) > 0 else 0,
        'right_yield': len(cst_right) / len(right_seeds) * 100 if len(right_seeds) > 0 else 0,
        'extraction_rate': len(cst_combined) / (len(left_seeds) + len(right_seeds)) * 100 if (len(left_seeds) + len(right_seeds)) > 0 else 0,
        'method': 'roi-seeded'
    }
    
    # Add length statistics if we have streamlines
    if len(cst_combined) > 0:
        cst_lengths = np.array([streamline_length(sl) for sl in cst_combined])
        stats['length_mean'] = float(np.mean(cst_lengths))
        stats['length_std'] = float(np.std(cst_lengths))
        stats['length_min'] = float(np.min(cst_lengths))
        stats['length_max'] = float(np.max(cst_lengths))
    
    parameters = {
        'fa_threshold': fa_threshold,
        'seed_density': seed_density,
        'step_size': step_size,
        'min_length': min_length,
        'max_length': max_length,
        'sh_order': validated_sh_order,
        'relative_peak_threshold': relative_peak_threshold,
        'min_separation_angle': min_separation_angle,
        'random_seed': random_seed,
        'npeaks': npeaks,
        'fit_method': fit_method,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("ROI-SEEDED EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nResults:")
        print(f"    Left CST:  {stats['cst_left_count']:,} streamlines ({stats['left_yield']:.2f}% yield)")
        print(f"    Right CST: {stats['cst_right_count']:,} streamlines ({stats['right_yield']:.2f}% yield)")
        print(f"    Total:     {stats['cst_total_count']:,} streamlines")
        
        if 'length_mean' in stats:
            print(f"    Length: mean={stats['length_mean']:.1f}, "
                  f"range=[{stats['length_min']:.1f}, {stats['length_max']:.1f}] mm")
    
    return {
        'cst_left': cst_left,
        'cst_right': cst_right,
        'cst_combined': cst_combined,
        'stats': stats,
        'parameters': parameters
    }