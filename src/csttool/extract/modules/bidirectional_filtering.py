"""
bidirectional_filtering.py

Bidirectional CST extraction via intersection of forward and reverse seeded runs.

Motivation: motor-cortex-seeded forward tracking produces direction-dependent
asymmetry because atlas ROI centres land at slightly different positions relative
to the GM/WM interface on each side. Brainstem-seeded reverse tracking is symmetric
because the brainstem has no such positional ambiguity. Retaining only streamlines
confirmed by both directions eliminates the cortical placement artifact.

Algorithm:
  Pass A (forward):  seed motor cortex (L/R) → filter by brainstem traversal
  Pass B (reverse):  seed brainstem        → filter by motor cortex (L/R) traversal
  Confirmed zone:    voxelise Pass B bundles → binary masks of traversed voxels
  Intersection:      from Pass A, keep streamlines that traverse the confirmed zone
"""

import numpy as np
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines, length as streamline_length
from dipy.tracking.utils import density_map

from .roi_seeded_tracking import (
    generate_seeds_from_mask,
    track_from_seeds,
    filter_by_target_roi,
    filter_by_length,
    streamline_passes_through,
)


def _voxelise(streamlines, affine, shape):
    """Return a density map (count) of voxels traversed by streamlines."""
    if len(streamlines) == 0:
        return np.zeros(shape, dtype=np.float32)
    return density_map(streamlines, affine, shape).astype(np.float32)


def _overlap_score(streamline, density, affine):
    """Sum of density values at each voxel the streamline passes through."""
    inv_affine = np.linalg.inv(affine)
    ones = np.ones((len(streamline), 1))
    pts_h = np.hstack([streamline, ones])
    voxels = np.round((pts_h @ inv_affine.T)[:, :3]).astype(int)
    score = 0.0
    for vox in voxels:
        if (0 <= vox[0] < density.shape[0] and
                0 <= vox[1] < density.shape[1] and
                0 <= vox[2] < density.shape[2]):
            score += density[vox[0], vox[1], vox[2]]
    return score


def extract_cst_bidirectional(
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
    fit_method='WLS',
):
    """
    Extract bilateral CST using bidirectional seeding with intersection.

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
        3D binary mask for left motor cortex.
    motor_right_mask : ndarray
        3D binary mask for right motor cortex.
    brainstem_mask : ndarray
        3D binary mask for brainstem.
    fa_map : ndarray, optional
        Pre-computed FA map. Computed from DTI if not provided.
    fa_threshold : float, optional
        FA tracking threshold. Default 0.15.
    seed_density : int, optional
        Seeds per voxel dimension. Default 2.
    step_size : float, optional
        Tracking step size in mm. Default 0.5.
    min_length : float, optional
        Minimum streamline length in mm. Default 30.
    max_length : float, optional
        Maximum streamline length in mm. Default 200.
    sh_order : int, optional
        Spherical harmonic order for CSA-ODF. Default 6.
    relative_peak_threshold : float, optional
        Relative peak threshold. Default 0.5.
    min_separation_angle : int, optional
        Minimum angle between ODF peaks in degrees. Default 25.
    random_seed : int, optional
        Random seed for reproducible tracking, applied to all three passes
        (forward left, forward right, reverse). The same seed is used for
        every pass so the RNG cannot itself become a source of L/R
        difference or of forward/reverse inflation — both of which this
        method exists to measure. If None, results vary between runs and
        ``artifact_index`` is not reproducible.
    verbose : bool, optional
        Print progress. Default True.
    npeaks : int, optional
        Number of ODF peaks extracted per voxel. Default 1 (single
        principal direction, consistent with whole-brain tracking).
    fit_method : str, optional
        DTI fit method passed to TensorModel. Default 'WLS' (consistent
        with the pinned default in fit_tensors).

    Returns
    -------
    result : dict
        Keys: cst_left, cst_right, cst_combined, stats, parameters.
    """
    from dipy.reconst.dti import TensorModel

    if verbose:
        print("=" * 60)
        print("BIDIRECTIONAL CST EXTRACTION")
        print("=" * 60)
        if random_seed is not None:
            print(f"  → Random seed: {random_seed}")
        else:
            print("  → Random seeding disabled - results will vary between runs")

    # ------------------------------------------------------------------
    # Step 1: FA + ODF peaks + stopping criterion
    # ------------------------------------------------------------------
    if fa_map is None:
        if verbose:
            print("\n[Step 1/6] Computing FA map...")
        tensor_model = TensorModel(gtab, fit_method=fit_method)
        tensor_fit = tensor_model.fit(data, mask=brain_mask)
        fa_map = np.clip(tensor_fit.fa, 0, 1)
    else:
        if verbose:
            print("\n[Step 1/6] Using provided FA map...")

    if verbose:
        print("\n[Step 2/6] Estimating fiber directions (CSA-ODF)...")

    from csttool.tracking.modules.estimate_directions import validate_sh_order
    validated_sh_order = validate_sh_order(gtab, sh_order, verbose=verbose)

    csa_model = CsaOdfModel(gtab, sh_order=validated_sh_order)
    wm_mask = fa_map > fa_threshold
    peaks = peaks_from_model(
        csa_model, data, default_sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        mask=wm_mask,
        npeaks=npeaks,
    )

    if verbose:
        print(f"    Peaks computed for {wm_mask.sum():,} voxels")
        print(f"\n[Step 3/6] Setting up stopping criterion (FA > {fa_threshold})...")

    stopping_criterion = ThresholdStoppingCriterion(fa_map, fa_threshold)
    shape = brain_mask.shape

    # ------------------------------------------------------------------
    # Step 2: Forward left (motor left → brainstem)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Step 4/6] Forward pass: motor cortex → brainstem...")

    left_seeds = generate_seeds_from_mask(motor_left_mask, affine, density=seed_density)
    right_seeds = generate_seeds_from_mask(motor_right_mask, affine, density=seed_density)

    if verbose:
        print(f"    Left motor seeds:  {len(left_seeds):,}")
        print(f"    Right motor seeds: {len(right_seeds):,}")

    left_fwd_all = filter_by_length(
        track_from_seeds(left_seeds, peaks, stopping_criterion, affine, step_size,
                         random_seed=random_seed),
        min_length, max_length, verbose=verbose,
    )
    left_fwd, _ = filter_by_target_roi(left_fwd_all, brainstem_mask, affine, verbose=verbose)

    right_fwd_all = filter_by_length(
        track_from_seeds(right_seeds, peaks, stopping_criterion, affine, step_size,
                         random_seed=random_seed),
        min_length, max_length, verbose=verbose,
    )
    right_fwd, _ = filter_by_target_roi(right_fwd_all, brainstem_mask, affine, verbose=verbose)

    if verbose:
        print(f"    Forward left → brainstem:  {len(left_fwd):,}")
        print(f"    Forward right → brainstem: {len(right_fwd):,}")

    # ------------------------------------------------------------------
    # Step 3: Reverse pass (brainstem → motor cortex, L and R)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Step 5/6] Reverse pass: brainstem → motor cortex...")

    bs_seeds = generate_seeds_from_mask(brainstem_mask, affine, density=seed_density)

    if verbose:
        print(f"    Brainstem seeds: {len(bs_seeds):,}")

    bs_all = filter_by_length(
        track_from_seeds(bs_seeds, peaks, stopping_criterion, affine, step_size,
                         random_seed=random_seed),
        min_length, max_length, verbose=verbose,
    )

    bs_to_left, _ = filter_by_target_roi(bs_all, motor_left_mask, affine, verbose=verbose)
    bs_to_right, _ = filter_by_target_roi(bs_all, motor_right_mask, affine, verbose=verbose)

    if verbose:
        print(f"    Brainstem → left motor:  {len(bs_to_left):,}")
        print(f"    Brainstem → right motor: {len(bs_to_right):,}")

    # ------------------------------------------------------------------
    # Step 4: Count-bounded intersection
    #
    # The naive "density > 0" intersection is a no-op: both forward and
    # reverse bundles traverse the same voxels (the CST territory), so
    # every forward streamline passes through the reverse confirmed zone.
    #
    # The source of the asymmetry is in HOW MANY seeds initiate valid CST
    # trajectories from each motor cortex ROI, not in where those
    # trajectories go.  The reverse (brainstem-seeded) count is the
    # symmetric, unbiased reference.
    #
    # Fix: cap the forward count at min(N_fwd, N_reverse) per side,
    # choosing the forward streamlines ranked highest by spatial overlap
    # with the reverse-pass density map.  This selects the "most
    # representative" forward streamlines and discards excess ones that
    # arose from the ROI placement advantage.
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Step 6/6] Count-bounded intersection (forward ranked by reverse density)...")

    dens_L = _voxelise(bs_to_left, affine, shape)
    dens_R = _voxelise(bs_to_right, affine, shape)

    if verbose:
        print(f"    Reverse density nonzero: left={int((dens_L > 0).sum()):,}  right={int((dens_R > 0).sum()):,}")

    def _select_top_n(forward_bundle, density, n_keep):
        if len(forward_bundle) == 0 or n_keep == 0:
            return Streamlines()
        scores = np.array([_overlap_score(s, density, affine) for s in forward_bundle])
        n_keep = min(n_keep, int((scores > 0).sum()))   # can't keep more than have overlap
        if n_keep == 0:
            return Streamlines()
        top_idx = np.argsort(scores)[-n_keep:]
        return Streamlines([forward_bundle[i] for i in sorted(top_idx)])

    # Per-side cap only — bilateral symmetry is NOT enforced. Each hemisphere is
    # bounded by its own reverse-pass count, which removes the cortical-interface
    # inflation per side without assuming the two sides should match. This
    # preserves genuine asymmetry (e.g. unilateral motor pathology) while still
    # correcting the placement artifact.
    n_keep_L = min(len(left_fwd), len(bs_to_left))
    n_keep_R = min(len(right_fwd), len(bs_to_right))

    if verbose:
        print(f"    Per-side cap — left:  min({len(left_fwd)}, {len(bs_to_left)}) = {n_keep_L}")
        print(f"    Per-side cap — right: min({len(right_fwd)}, {len(bs_to_right)}) = {n_keep_R}")

    cst_left  = _select_top_n(left_fwd,  dens_L, n_keep_L)
    cst_right = _select_top_n(right_fwd, dens_R, n_keep_R)

    if verbose:
        print(f"    After intersection — left:  {len(left_fwd):,} → {len(cst_left):,}")
        print(f"    After intersection — right: {len(right_fwd):,} → {len(cst_right):,}")

    cst_combined = Streamlines(list(cst_left) + list(cst_right))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    # Forward/reverse inflation per side. Comparing the two ratios indicates
    # whether residual L/R count asymmetry is methodological (interface artifact)
    # or structural (pathology / genuine biology).
    #   inflation_L ≈ inflation_R → both sides over-produced equally; residual
    #     asymmetry is structural.
    #   inflation_L ≠ inflation_R → one cortical ROI seeded more efficiently
    #     than the other; residual asymmetry is suspect.
    inflation_L = len(left_fwd) / max(len(bs_to_left), 1)
    inflation_R = len(right_fwd) / max(len(bs_to_right), 1)
    artifact_index = (
        abs(inflation_L - inflation_R) / max(inflation_L, inflation_R, 1.0)
    )

    stats = {
        'left_seeds': len(left_seeds),
        'right_seeds': len(right_seeds),
        'bs_seeds': len(bs_seeds),
        'left_forward_count': len(left_fwd),
        'right_forward_count': len(right_fwd),
        'bs_to_left_count': len(bs_to_left),
        'bs_to_right_count': len(bs_to_right),
        'cst_left_count': len(cst_left),
        'cst_right_count': len(cst_right),
        'cst_total_count': len(cst_combined),
        'total_input': len(left_fwd) + len(right_fwd),
        'left_intersection_rate': len(cst_left) / max(len(left_fwd), 1) * 100,
        'right_intersection_rate': len(cst_right) / max(len(right_fwd), 1) * 100,
        'left_yield': len(cst_left) / max(len(left_seeds), 1) * 100,
        'right_yield': len(cst_right) / max(len(right_seeds), 1) * 100,
        'extraction_rate': len(cst_combined) / max(len(left_seeds) + len(right_seeds), 1) * 100,
        'forward_reverse_ratio_left': inflation_L,
        'forward_reverse_ratio_right': inflation_R,
        'artifact_index': artifact_index,
        'method': 'bidirectional',
    }

    if len(cst_combined) > 0:
        lengths = np.array([streamline_length(s) for s in cst_combined])
        stats['length_mean'] = float(np.mean(lengths))
        stats['length_std'] = float(np.std(lengths))
        stats['length_min'] = float(np.min(lengths))
        stats['length_max'] = float(np.max(lengths))

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
        print("BIDIRECTIONAL EXTRACTION COMPLETE")
        print("=" * 60)
        lr = len(cst_right) / max(len(cst_left), 1)
        print(f"\n  Left CST:  {stats['cst_left_count']:,} streamlines ({stats['left_yield']:.2f}% yield)")
        print(f"  Right CST: {stats['cst_right_count']:,} streamlines ({stats['right_yield']:.2f}% yield)")
        print(f"  Total:     {stats['cst_total_count']:,} streamlines  R/L = {lr:.3f}")
        print(f"  Forward/reverse inflation — left: {inflation_L:.2f}  right: {inflation_R:.2f}")
        from csttool.defaults import DEFAULT_ARTIFACT_INDEX_THRESHOLD
        if artifact_index > DEFAULT_ARTIFACT_INDEX_THRESHOLD:
            print(f"  Artifact index: {artifact_index:.2f} (>{DEFAULT_ARTIFACT_INDEX_THRESHOLD:.2f}) — cortical interface "
                  f"asymmetry suspected; residual L/R count difference may be methodological.")
        else:
            print(f"  Artifact index: {artifact_index:.2f} (≤{DEFAULT_ARTIFACT_INDEX_THRESHOLD:.2f}) — inflation symmetric; "
                  f"residual L/R count difference is likely structural.")
        if 'length_mean' in stats:
            print(f"  Length: mean={stats['length_mean']:.1f}, "
                  f"range=[{stats['length_min']:.1f}, {stats['length_max']:.1f}] mm")

    return {
        'cst_left': cst_left,
        'cst_right': cst_right,
        'cst_combined': cst_combined,
        'stats': stats,
        'parameters': parameters,
    }
