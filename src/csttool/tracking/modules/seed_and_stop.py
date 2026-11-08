def seed_and_stop(fa, affine, white_matter=None, brain_mask=None, fa_thresh=0.2, density=1,
                  use_binary=False, use_brain_mask_stop=False, seed_mask=None,
                  verbose=False):
    """Generate seeds and stopping criterion for tractography.

    Args:
        fa (ndarray): Fractional anisotropy map (X, Y, Z).
        affine (ndarray): 4x4 affine transformation matrix.
        white_matter (ndarray): Binary white matter mask. Required if use_binary=True.
        brain_mask (ndarray): Binary brain mask. Used for boundary stopping if use_brain_mask_stop=True.
        fa_thresh (float): FA threshold for seeding and stopping (default 0.2).
        density (int): Seeds per voxel in seed mask (default 1).
        use_binary (bool): If True, use binary stopping on WM mask. If False (default), use FA threshold stopping.
        use_brain_mask_stop (bool): If True, also stop tracking at brain mask boundary.
            This provides a safety constraint to prevent streamlines from leaving the brain.
        seed_mask (ndarray, optional): Explicit seeding territory. When given,
            seeds come from this mask instead of ``fa >= fa_thresh``; the FA
            threshold *stopping* rule is unchanged. This is how an external or
            DAM white-matter mask replaces the FA-threshold seed territory in
            the default (non-binary) mode while keeping stopping orthogonal.
            None (the default) preserves the historical behaviour exactly.
        verbose (bool): Print processing details.

    Returns:
        tuple: (seeds, stopping_criterion)
            - seeds: Seed points in world coordinates (N, 3)
            - stopping_criterion: Stopping criterion for LocalTracking
    """
    import numpy as np
    from dipy.tracking.utils import seeds_from_mask
    from dipy.tracking.stopping_criterion import (
        ThresholdStoppingCriterion,
        BinaryStoppingCriterion
    )

    if use_binary:
        if white_matter is None:
            raise ValueError("white_matter mask required when use_binary=True")

        seeds = seeds_from_mask(white_matter, affine, density=density)
        stopping_criterion = BinaryStoppingCriterion(white_matter)

        if verbose:
            print("  → Seeding and stopping (binary mode)...")
            print(f"    • White matter voxels: {white_matter.sum():,}")
            print(f"    • Total seeds (density={density}): {len(seeds):,}")

    else:
        # Explicit seeding territory (external/DAM WM mask) overrides the
        # historical FA-threshold seed territory; stopping stays FA-based.
        if seed_mask is None:
            seed_mask = fa >= fa_thresh

        # If brain mask stopping is enabled, combine FA with brain mask
        if use_brain_mask_stop and brain_mask is not None:
            # Create a modified FA map that is 0 outside brain mask
            # This ensures tracking stops at brain boundary
            fa_masked = fa.copy()
            fa_masked[~brain_mask.astype(bool)] = 0.0
            stopping_criterion = ThresholdStoppingCriterion(fa_masked, fa_thresh)

            if verbose:
                print(f"  → Seeding and stopping (FA threshold={fa_thresh} + brain mask boundary)...")
                print(f"    • Seed mask voxels: {seed_mask.sum():,}")
                print(f"    • Brain mask voxels: {brain_mask.sum():,}")
                print(f"    • Total seeds (density={density}): {len(seeds_from_mask(seed_mask, affine, density=density)):,}")
        else:
            stopping_criterion = ThresholdStoppingCriterion(fa, fa_thresh)

            if verbose:
                print(f"  → Seeding and stopping (FA threshold={fa_thresh})...")
                print(f"    • Seed mask voxels: {seed_mask.sum():,}")

        seeds = seeds_from_mask(seed_mask, affine, density=density)

        if verbose and not (use_brain_mask_stop and brain_mask is not None):
            print(f"    • Total seeds (density={density}): {len(seeds):,}")

    return seeds, stopping_criterion