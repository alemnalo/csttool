"""
passthrough_filtering.py

Filter streamlines that pass through (not just terminate in) ROI masks.
More permissive than endpoint filtering for long-range tracts.
"""

import numpy as np
from dipy.tracking.streamline import Streamlines, length
from scipy.ndimage import center_of_mass


def get_roi_geometry(mask, affine, name):
    """Compute and return ROI geometry metrics in world coordinates."""
    if mask.sum() == 0:
        return None

    # Get voxel coordinates of all mask voxels
    coords = np.argwhere(mask > 0)

    # Compute center of mass in voxel space
    com_vox = center_of_mass(mask.astype(float))

    # Convert COM to world coordinates
    com_world = affine @ np.array([com_vox[0], com_vox[1], com_vox[2], 1])
    com_world = com_world[:3]

    # Convert all voxel coordinates to world coordinates
    ones = np.ones((len(coords), 1))
    coords_h = np.hstack([coords, ones])
    world_coords = (affine @ coords_h.T).T[:, :3]

    # Get X extent
    x_min, x_max = world_coords[:, 0].min(), world_coords[:, 0].max()

    return {
        'name': name,
        'com': com_world,
        'x_min': x_min,
        'x_max': x_max,
        'midline_dist': abs(com_world[0]),
        'voxel_count': len(coords)
    }


def print_roi_geometry(geom):
    """Print ROI geometry in a formatted way."""
    if geom is None:
        return
    print(f"    {geom['name']}:")
    print(f"        COM (mm): X={geom['com'][0]:.1f}, Y={geom['com'][1]:.1f}, Z={geom['com'][2]:.1f}")
    print(f"        X extent: [{geom['x_min']:.1f}, {geom['x_max']:.1f}] mm")
    print(f"        Distance from midline: {geom['midline_dist']:.1f} mm")
    print(f"        Voxel count: {geom['voxel_count']:,}")


def get_roi_hemisphere_split(mask, affine, name, midline_x=0.0):
    """Analyze ROI voxel distribution by hemisphere (left vs right of the midline).

    The midline is the warped-MNI midline (``world X = midline_x``), not
    necessarily 0 (AU11).
    """
    if mask.sum() == 0:
        return None

    coords = np.argwhere(mask > 0)
    ones = np.ones((len(coords), 1))
    coords_h = np.hstack([coords, ones])
    world_coords = (affine @ coords_h.T).T[:, :3]

    left_count = np.sum(world_coords[:, 0] < midline_x)
    right_count = np.sum(world_coords[:, 0] >= midline_x)
    total_count = len(coords)
    lr_ratio = left_count / right_count if right_count > 0 else 0

    return {
        'name': name,
        'left_voxels': int(left_count),
        'right_voxels': int(right_count),
        'total_voxels': total_count,
        'lr_ratio': lr_ratio,
        'left_pct': (left_count / total_count * 100) if total_count > 0 else 0,
        'right_pct': (right_count / total_count * 100) if total_count > 0 else 0,
        'midline_x': midline_x,
    }


def print_roi_hemisphere_split(split):
    """Print ROI hemisphere split in a formatted way."""
    if split is None:
        return
    m = split.get('midline_x', 0.0)
    print(f"    {split['name']} hemisphere split (midline X={m:.1f}):")
    print(f"        Left (X<{m:.1f}):  {split['left_voxels']:,} voxels ({split['left_pct']:.1f}%)")
    print(f"        Right (X>={m:.1f}): {split['right_voxels']:,} voxels ({split['right_pct']:.1f}%)")
    print(f"        L/R ratio:   {split['lr_ratio']:.3f}")
    if abs(split['lr_ratio'] - 1.0) > 0.05:
        bias = "left" if split['lr_ratio'] > 1.0 else "right"
        pct_diff = abs(split['left_voxels'] - split['right_voxels']) / split['total_voxels'] * 100
        print(f"        -> Asymmetric: {pct_diff:.1f}% more voxels on {bias}")


def sample_peduncle_fa(fa_map, fa_affine, brainstem_mask, mask_affine,
                       hemisphere_mask=None, midline_x=None, verbose=True):
    """Sample FA in left vs right cerebral peduncle (superior brainstem).

    The cerebral peduncle is approximated as the superior 30% of the brainstem
    by Z coordinate. This is where CST fibers enter the brainstem.

    The left/right split uses, in order of preference (AU11):

    1. ``hemisphere_mask`` — the per-voxel warped-MNI L/R label (the anatomical
       midline warped into subject space). This is the gold standard: each
       brainstem voxel is assigned to the hemisphere it maps to in MNI, so the
       split is correct regardless of where the subject midline sits in world X.
    2. ``midline_x`` — the scalar warped-MNI midline world X. The split is
       ``world X < midline_x`` vs ``>= midline_x``.
    3. ``world X < 0`` — the legacy assumption (the bug AU11 fixes), used only when
       neither of the above is supplied so old callers/tests still run.

    Parameters
    ----------
    fa_map : ndarray
        3D FA map.
    fa_affine : ndarray, shape (4, 4)
        Affine matrix for FA map.
    brainstem_mask : ndarray
        Binary brainstem mask (subject / mask_affine voxel space).
    mask_affine : ndarray, shape (4, 4)
        Affine matrix for brainstem mask (subject space).
    hemisphere_mask : ndarray of bool, optional
        Per-voxel left-hemisphere label from ``compute_warped_midline``, on the
        same voxel grid as ``brainstem_mask``.
    midline_x : float, optional
        Scalar midline world X (from ``compute_warped_midline``).
    verbose : bool
        Print results.

    Returns
    -------
    result : dict
        Dictionary with left_mean_fa, right_mean_fa, and counts.
    """
    # Get brainstem voxel coords
    bs_coords = np.argwhere(brainstem_mask > 0)

    if len(bs_coords) == 0:
        return None

    # Convert to world coords
    ones = np.ones((len(bs_coords), 1))
    bs_world = (mask_affine @ np.hstack([bs_coords, ones]).T).T[:, :3]

    # Take superior portion (top 30% by Z) - cerebral peduncle region
    z_coords = bs_world[:, 2]
    z_threshold = np.percentile(z_coords, 70)
    superior_mask = z_coords >= z_threshold

    # Split by hemisphere (AU11): per-voxel warped-MNI label is preferred, then the
    # scalar warped-MNI midline, then the legacy X=0 assumption.
    if hemisphere_mask is not None:
        # Index the 3D warped-MNI L/R label at each brainstem voxel.
        left_hemi = hemisphere_mask[bs_coords[:, 0], bs_coords[:, 1], bs_coords[:, 2]]
        right_hemi = ~left_hemi
        split_source = 'warped-MNI hemisphere mask'
    else:
        x_coords = bs_world[:, 0]
        m = midline_x if midline_x is not None else 0.0
        left_hemi = x_coords < m
        right_hemi = ~left_hemi
        split_source = f'world X < {m:.2f} (midline)' if midline_x is not None else 'world X < 0 (legacy)'

    left_mask = left_hemi & superior_mask
    right_mask = right_hemi & superior_mask

    # Convert world coords to FA voxel space and sample
    inv_fa_affine = np.linalg.inv(fa_affine)

    left_fas = []
    right_fas = []
    for i in range(len(bs_coords)):
        if not (left_mask[i] or right_mask[i]):
            continue

        world = bs_world[i]
        fa_vox = np.round(inv_fa_affine @ np.append(world, 1))[:3].astype(int)
        if (0 <= fa_vox[0] < fa_map.shape[0] and
            0 <= fa_vox[1] < fa_map.shape[1] and
            0 <= fa_vox[2] < fa_map.shape[2]):
            fa_val = fa_map[fa_vox[0], fa_vox[1], fa_vox[2]]
            if left_mask[i]:
                left_fas.append(fa_val)
            elif right_mask[i]:
                right_fas.append(fa_val)

    result = {
        'left_mean_fa': np.mean(left_fas) if left_fas else 0,
        'right_mean_fa': np.mean(right_fas) if right_fas else 0,
        'left_n': len(left_fas),
        'right_n': len(right_fas),
    }

    if verbose and left_fas and right_fas:
        print(f"\n    Cerebral Peduncle FA (superior 30% of brainstem; split: {split_source}):")
        print(f"      Left:  mean FA = {result['left_mean_fa']:.3f} (n={result['left_n']})")
        print(f"      Right: mean FA = {result['right_mean_fa']:.3f} (n={result['right_n']})")
        fa_diff = result['left_mean_fa'] - result['right_mean_fa']
        if fa_diff < -0.01:
            print(f"      -> Left peduncle has lower FA ({-fa_diff:.3f} difference)")
        elif fa_diff > 0.01:
            print(f"      -> Right peduncle has lower FA ({fa_diff:.3f} difference)")
        else:
            print(f"      -> Similar FA in both peduncles")

    return result


def streamline_passes_through(streamline, mask, affine):
    """Check if any point along streamline passes through mask."""
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


def get_first_brainstem_entry(streamline, mask, affine):
    """Return X coordinate of first point entering brainstem mask.

    Used to diagnose whether hemisphere classification aligns with
    actual brainstem entry side.
    """
    inv_affine = np.linalg.inv(affine)
    ones = np.ones((len(streamline), 1))
    pts_h = np.hstack([streamline, ones])
    voxels = np.round((pts_h @ inv_affine.T)[:, :3]).astype(int)

    for i, vox in enumerate(voxels):
        if (0 <= vox[0] < mask.shape[0] and
            0 <= vox[1] < mask.shape[1] and
            0 <= vox[2] < mask.shape[2]):
            if mask[vox[0], vox[1], vox[2]] > 0:
                return streamline[i, 0]  # Return X in world coords
    return None


def extract_cst_passthrough(
    streamlines,
    masks,
    affine,
    min_length=20.0,
    max_length=200.0,
    midline_x=0.0,
    midline_distance=None,
    verbose=True
):
    """
    Extract bilateral CST using pass-through filtering.
    
    A streamline is included if it passes through BOTH the brainstem
    AND the corresponding motor cortex at any point along its length.
    
    Parameters
    ----------
    streamlines : Streamlines
        Whole-brain tractogram.
    masks : dict
        ROI masks with keys: 'brainstem', 'motor_left', 'motor_right'
    affine : ndarray, shape (4, 4)
        Affine matrix for masks.
    min_length : float
        Minimum streamline length in mm.
    max_length : float
        Maximum streamline length in mm.
    midline_x : float, optional
        Subject-space anatomical midline world X (from
        ``compute_warped_midline``). The subject is reoriented to RAS but never
        recentered, so the midline is not in general at world X=0; splitting the
        hemisphere diagnostics and the midline-crossing exclusion on X=0 (the
        legacy assumption) biases them when the subject is off-centre (AU11).
        Defaults to 0.0 (legacy) for backward compatibility; the pipeline passes
        the warped-MNI midline. Note this only affects *diagnostics* and the
        commissural-exclusion filter — final L/R bundle assignment is ROI-based,
        not plane-based, and is unaffected.
    midline_distance : ndarray of float, optional
        Signed distance (mm) to the warped MNI midline surface (from
        ``compute_warped_midline``), on the same voxel grid as ``masks``. When
        given, the midline-crossing exclusion uses the true *curved* midline rather
        than the scalar ``midline_x`` plane — a single scalar is a poor fit because
        the warped midline curves with SyN and a whole-brain median is biased by
        asymmetric sub-cortical structures. Defaults to None (legacy scalar
        plane). Same grid convention as ``affine`` (subject RAS voxel space).
    verbose : bool
        Print progress information.
        
    Returns
    -------
    result : dict
        Dictionary with 'cst_left', 'cst_right', 'cst_combined', 'stats'
    """
    if verbose:
        print("=" * 60)
        print("PASS-THROUGH CST EXTRACTION")
        print("=" * 60)
        print(f"\nInput: {len(streamlines):,} streamlines")
    
    brainstem = masks['brainstem']
    motor_left = masks['motor_left']
    motor_right = masks['motor_right']

    # AU11: when the signed distance-to-midline volume is available, precompute an
    # inverse-affine lookup so the midline-crossing exclusion can evaluate the true
    # curved midline at each streamline point. Otherwise the scalar midline_x plane
    # is used (legacy). Voxel-size is also used to keep the 8 mm tolerance in mm.
    use_signed_midline = midline_distance is not None
    if use_signed_midline:
        _inv_affine = np.linalg.inv(affine)
        _dist_shape = np.asarray(midline_distance.shape)
    
    # Length filtering
    if verbose:
        print(f"\n[Step 1/2] Length filtering ({min_length}-{max_length} mm)...")
    
    lengths = np.array([length(sl) for sl in streamlines])
    length_mask = (lengths >= min_length) & (lengths <= max_length)
    valid_indices = np.where(length_mask)[0]
    streamlines_filtered = Streamlines([streamlines[i] for i in valid_indices])
    
    if verbose:
        print(f"    {len(streamlines):,} → {len(streamlines_filtered):,} streamlines")

    # Analyze input tractogram hemisphere distribution (for asymmetry diagnosis)
    if verbose:
        print(f"\n    Analyzing input hemisphere distribution...")
    left_input = sum(1 for sl in streamlines_filtered if np.mean(sl[:, 0]) < midline_x)
    right_input = len(streamlines_filtered) - left_input
    lr_input = left_input / right_input if right_input > 0 else 0
    if verbose:
        print(f"    Input hemisphere distribution:")
        print(f"        Left (centroid X<{midline_x:.1f}):  {left_input:,}")
        print(f"        Right (centroid X>={midline_x:.1f}): {right_input:,}")
        print(f"        L/R Ratio: {lr_input:.3f}")

    # Pass-through filtering
    if verbose:
        print(f"\n[Step 2/2] Pass-through filtering...")

    cst_left_list = []
    cst_right_list = []
    bilateral_excluded_count = 0
    midline_excluded_count = 0

    # Diagnostic counters for per-stage asymmetry analysis
    passes_brainstem_count = 0
    left_bs_count = 0  # Brainstem passes from left hemisphere streamlines
    right_bs_count = 0  # Brainstem passes from right hemisphere streamlines
    passes_left_motor_count = 0
    passes_right_motor_count = 0

    # Brainstem entry point diagnostics (to check classification alignment)
    left_entry_xs = []  # X coords of brainstem entry for left-classified streamlines
    right_entry_xs = []  # X coords of brainstem entry for right-classified streamlines

    # Inferior extent diagnostics (for streamlines failing brainstem check)
    left_fail_min_zs = []  # Min Z for left-classified streamlines that fail brainstem
    right_fail_min_zs = []  # Min Z for right-classified streamlines that fail brainstem

    for i, sl in enumerate(streamlines_filtered):
        passes_bs = streamline_passes_through(sl, brainstem, affine)

        if passes_bs:
            passes_brainstem_count += 1  # Count brainstem hits

            # Track hemisphere of brainstem-passing streamlines (using centroid X)
            sl_centroid_x = np.mean(sl[:, 0])
            if sl_centroid_x < midline_x:
                left_bs_count += 1
            else:
                right_bs_count += 1

            # Record brainstem entry X for diagnostic (classification alignment check)
            entry_x = get_first_brainstem_entry(sl, brainstem, affine)
            if entry_x is not None:
                if sl_centroid_x < midline_x:
                    left_entry_xs.append(entry_x)
                else:
                    right_entry_xs.append(entry_x)

            # Check mutual exclusivity (bilateral motor)
            passes_left = streamline_passes_through(sl, motor_left, affine)
            passes_right = streamline_passes_through(sl, motor_right, affine)

            # Count motor cortex hits (before exclusion logic)
            if passes_left:
                passes_left_motor_count += 1
            if passes_right:
                passes_right_motor_count += 1

            if passes_left and passes_right:
                bilateral_excluded_count += 1
                continue
            
            # Check midline crossing with tolerance for registration imperfection
            # This catches streamlines that grossly cross hemispheres (commissural).
            # The midline is the warped-MNI midline (AU11): prefer the signed
            # distance volume (the true curved surface) when available, else the
            # scalar midline_x plane. The tolerance itself remains provisional (AU14).
            MIDLINE_TOLERANCE_MM = 8.0  # Allow minor medial excursion
            if use_signed_midline:
                # Map streamline points to voxels and look up the signed distance.
                ones = np.ones((len(sl), 1))
                voxels = np.floor((np.hstack([sl, ones]) @ _inv_affine.T)[:, :3]).astype(int)
                in_bounds = (
                    (voxels[:, 0] >= 0) & (voxels[:, 0] < _dist_shape[0]) &
                    (voxels[:, 1] >= 0) & (voxels[:, 1] < _dist_shape[1]) &
                    (voxels[:, 2] >= 0) & (voxels[:, 2] < _dist_shape[2])
                )
                if in_bounds.any():
                    vv = voxels[in_bounds]
                    dists = midline_distance[vv[:, 0], vv[:, 1], vv[:, 2]]
                    d_min, d_max = float(np.min(dists)), float(np.max(dists))
                else:
                    d_min, d_max = 0.0, 0.0
                crosses_deep = (d_min < -MIDLINE_TOLERANCE_MM and
                                d_max > MIDLINE_TOLERANCE_MM)
            else:
                x_coords = sl[:, 0]
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                crosses_deep = (x_min < midline_x - MIDLINE_TOLERANCE_MM and
                                x_max > midline_x + MIDLINE_TOLERANCE_MM)
            if crosses_deep:
                midline_excluded_count += 1
                continue
            
            if passes_left:
                cst_left_list.append(sl)
            elif passes_right:
                cst_right_list.append(sl)
        else:
            # Streamline failed brainstem check - record inferior extent for diagnostic
            min_z = np.min(sl[:, 2])
            sl_centroid_x = np.mean(sl[:, 0])
            if sl_centroid_x < midline_x:
                left_fail_min_zs.append(min_z)
            else:
                right_fail_min_zs.append(min_z)

        if verbose and (i + 1) % 50000 == 0:
            print(f"    Processed {i + 1:,} / {len(streamlines_filtered):,}...")
    
    cst_left = Streamlines(cst_left_list)
    cst_right = Streamlines(cst_right_list)
    cst_combined = Streamlines(cst_left_list + cst_right_list)
    
    # Compute L/R ratios for stats
    lr_bs = left_bs_count / right_bs_count if right_bs_count > 0 else 0

    stats = {
        'total_input': len(streamlines),
        'after_length_filter': len(streamlines_filtered),
        # Input hemisphere distribution
        'input_left': left_input,
        'input_right': right_input,
        'input_lr_ratio': lr_input,
        # Brainstem hemisphere split
        'passes_brainstem': passes_brainstem_count,
        'left_bs': left_bs_count,
        'right_bs': right_bs_count,
        'lr_bs': lr_bs,
        # Motor cortex counts (before exclusion)
        'passes_left_motor': passes_left_motor_count,
        'passes_right_motor': passes_right_motor_count,
        # Conditional motor yields
        'left_motor_yield': (passes_left_motor_count / left_bs_count * 100) if left_bs_count > 0 else 0,
        'right_motor_yield': (passes_right_motor_count / right_bs_count * 100) if right_bs_count > 0 else 0,
        # Final counts
        'cst_left_count': len(cst_left),
        'cst_right_count': len(cst_right),
        'cst_total_count': len(cst_combined),
        'bilateral_excluded': bilateral_excluded_count,
        'midline_excluded': midline_excluded_count,
        'extraction_rate': len(cst_combined) / len(streamlines) * 100 if len(streamlines) > 0 else 0,
    }
    
    if verbose:
        print(f"\n" + "=" * 60)
        print("PASS-THROUGH EXTRACTION COMPLETE")
        print("=" * 60)

        # Per-stage asymmetry analysis for diagnosis
        lr_motor = passes_left_motor_count / passes_right_motor_count if passes_right_motor_count > 0 else 0
        lr_final = len(cst_left) / len(cst_right) if len(cst_right) > 0 else 0

        print(f"\nPer-Stage Asymmetry Analysis:")
        print(f"    Stage                           Left      Right     L/R Ratio")
        print(f"    " + "-" * 60)
        print(f"    Input (post-length filter)  {left_input:>8,}  {right_input:>8,}     {lr_input:.3f}")
        print(f"    Pass through brainstem      {left_bs_count:>8,}  {right_bs_count:>8,}     {lr_bs:.3f}")
        print(f"    Pass through motor cortex   {passes_left_motor_count:>8,}  {passes_right_motor_count:>8,}     {lr_motor:.3f}")
        print(f"    Final (after exclusions)    {len(cst_left):>8,}  {len(cst_right):>8,}     {lr_final:.3f}")

        # Conditional motor yields (key diagnostic)
        left_yield = stats['left_motor_yield']
        right_yield = stats['right_motor_yield']
        print(f"\n    Conditional motor yields (P(motor | brainstem+hemisphere)):")
        print(f"      Left:  {left_yield:5.1f}% ({passes_left_motor_count:,} / {left_bs_count:,})")
        print(f"      Right: {right_yield:5.1f}% ({passes_right_motor_count:,} / {right_bs_count:,})")
        if right_yield > 0:
            yield_ratio = left_yield / right_yield
            print(f"      Yield ratio (L/R): {yield_ratio:.3f}")

        # Brainstem ROI geometry
        print(f"\n    Brainstem ROI Geometry:")
        bs_geom = get_roi_geometry(brainstem, affine, "brainstem")
        print_roi_geometry(bs_geom)

        # Brainstem hemisphere split analysis
        bs_split = get_roi_hemisphere_split(brainstem, affine, "brainstem", midline_x=midline_x)
        print_roi_hemisphere_split(bs_split)

        # Brainstem entry point distribution (classification alignment diagnostic)
        if left_entry_xs and right_entry_xs:
            left_entry_xs_arr = np.array(left_entry_xs)
            right_entry_xs_arr = np.array(right_entry_xs)
            print(f"\n    Brainstem entry point distribution:")
            print(f"      Left-classified streamlines (n={len(left_entry_xs_arr):,}):")
            print(f"        Entry X mean: {np.mean(left_entry_xs_arr):.1f} mm")
            left_correct = np.sum(left_entry_xs_arr < midline_x)
            print(f"        Enter left side (X<{midline_x:.1f}): {left_correct:,} ({left_correct / len(left_entry_xs_arr) * 100:.1f}%)")
            print(f"      Right-classified streamlines (n={len(right_entry_xs_arr):,}):")
            print(f"        Entry X mean: {np.mean(right_entry_xs_arr):.1f} mm")
            right_correct = np.sum(right_entry_xs_arr >= midline_x)
            print(f"        Enter right side (X>={midline_x:.1f}): {right_correct:,} ({right_correct / len(right_entry_xs_arr) * 100:.1f}%)")
            # Summary interpretation
            left_pct = left_correct / len(left_entry_xs_arr) * 100
            right_pct = right_correct / len(right_entry_xs_arr) * 100
            if left_pct > 80 and right_pct > 80:
                print(f"      -> Classification aligns with brainstem entry (likely FA/tracking asymmetry)")
            elif left_pct < 60 or right_pct < 60:
                print(f"      -> Classification mismatch detected (centroid X misleading)")
            else:
                print(f"      -> Moderate alignment (mixed classification)")

        # Inferior extent analysis (streamlines failing brainstem check)
        if left_fail_min_zs and right_fail_min_zs:
            left_fail_arr = np.array(left_fail_min_zs)
            right_fail_arr = np.array(right_fail_min_zs)
            print(f"\n    Inferior extent (streamlines failing brainstem check):")
            print(f"      Left-classified (n={len(left_fail_arr):,}):")
            print(f"        Mean min Z: {np.mean(left_fail_arr):.1f} mm")
            print(f"      Right-classified (n={len(right_fail_arr):,}):")
            print(f"        Mean min Z: {np.mean(right_fail_arr):.1f} mm")
            z_diff = np.mean(left_fail_arr) - np.mean(right_fail_arr)
            if z_diff > 2.0:
                print(f"      -> Left streamlines stop {z_diff:.1f} mm higher (earlier termination)")
            elif z_diff < -2.0:
                print(f"      -> Right streamlines stop {-z_diff:.1f} mm higher (earlier termination)")
            else:
                print(f"      -> Similar inferior extent (Δ = {z_diff:.1f} mm)")

        # ROI geometry (for diagnosing spatial positioning)
        print(f"\n    Motor ROI Geometry:")
        left_geom = get_roi_geometry(motor_left, affine, "motor_left")
        right_geom = get_roi_geometry(motor_right, affine, "motor_right")
        print_roi_geometry(left_geom)
        print_roi_geometry(right_geom)
        if left_geom and right_geom:
            midline_diff = right_geom['midline_dist'] - left_geom['midline_dist']
            if abs(midline_diff) > 2.0:
                closer = "right" if midline_diff < 0 else "left"
                print(f"    → {closer} motor ROI is {abs(midline_diff):.1f} mm closer to midline")

        # Highlight where asymmetry is introduced
        print(f"\n    Asymmetry changes:")
        if abs(lr_input - 1.0) > 0.05:
            print(f"      Input tractogram already asymmetric (L/R = {lr_input:.3f})")
        if abs(lr_bs - lr_input) > 0.05:
            print(f"      Brainstem check: {lr_input:.3f} → {lr_bs:.3f} (Δ = {lr_bs - lr_input:+.3f})")
        if abs(lr_motor - lr_bs) > 0.05:
            print(f"      Motor cortex check: {lr_bs:.3f} → {lr_motor:.3f} (Δ = {lr_motor - lr_bs:+.3f})")
        if abs(lr_final - lr_motor) > 0.05:
            print(f"      Exclusion filters: {lr_motor:.3f} → {lr_final:.3f} (Δ = {lr_final - lr_motor:+.3f})")
        if (abs(lr_input - 1.0) <= 0.05 and abs(lr_bs - lr_input) <= 0.05 and
            abs(lr_motor - lr_bs) <= 0.05 and abs(lr_final - lr_motor) <= 0.05):
            print(f"      No significant asymmetry changes detected")

        print(f"\nResults:")
        print(f"    Left CST:  {stats['cst_left_count']:,} streamlines")
        print(f"    Right CST: {stats['cst_right_count']:,} streamlines")
        print(f"    Total:     {stats['cst_total_count']:,} streamlines")
        print(f"    Rejected (Bilateral): {bilateral_excluded_count:,} streamlines")
        print(f"    Rejected (Midline):   {midline_excluded_count:,} streamlines")
    
    return {
        'cst_left': cst_left,
        'cst_right': cst_right,
        'cst_combined': cst_combined,
        'stats': stats
    }