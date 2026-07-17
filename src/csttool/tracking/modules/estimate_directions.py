import warnings
from csttool.defaults import DEFAULT_RELATIVE_PEAK_THRESHOLD, DEFAULT_MIN_SEPARATION_ANGLE


def get_max_sh_order(n_directions: int) -> int:
    """
    Return maximum safe SH order for given number of gradient directions.
    
    The number of SH coefficients required for order l is (l+1)(l+2)/2.
    We require n_directions >= required coefficients for stable fitting.
    
    Args:
        n_directions: Number of unique non-b0 gradient directions.
        
    Returns:
        Maximum SH order that can be safely used.
    """
    # Minimum directions required: (l+1)(l+2)/2
    # l=2: 6, l=4: 15, l=6: 28, l=8: 45
    if n_directions >= 45:
        return 8
    elif n_directions >= 28:
        return 6
    elif n_directions >= 15:
        return 4
    else:
        return 2  # DTI only


def _count_gradient_directions(gtab) -> int:
    """Count unique non-b0 gradient directions from a GradientTable."""
    import numpy as np
    from csttool.defaults import DEFAULT_B0_THRESHOLD
    # Exclude b0 volumes at the configured threshold
    b0_threshold = DEFAULT_B0_THRESHOLD
    dwi_mask = gtab.bvals > b0_threshold
    
    if not dwi_mask.any():
        return 0
    
    # Get unique directions (normalize and count unique)
    bvecs = gtab.bvecs[dwi_mask]
    # Round to avoid floating point precision issues
    unique_dirs = np.unique(np.round(bvecs, decimals=4), axis=0)
    return len(unique_dirs)


def validate_sh_order(gtab, sh_order: int, verbose: bool = False) -> int:
    """
    Validate and potentially reduce SH order based on available gradient directions.
    
    Args:
        gtab: DIPY GradientTable.
        sh_order: Requested spherical harmonic order.
        verbose: Print validation details.
        
    Returns:
        Validated (possibly reduced) SH order.
    """
    n_directions = _count_gradient_directions(gtab)
    max_safe_order = get_max_sh_order(n_directions)
    
    if verbose:
        print(f"    • Gradient directions: {n_directions}")
        print(f"    • Maximum safe SH order: {max_safe_order}")
    
    if sh_order > max_safe_order:
        warnings.warn(
            f"Requested SH order {sh_order} requires ≥{(sh_order+1)*(sh_order+2)//2} "
            f"gradient directions, but only {n_directions} found. "
            f"Reducing to SH order {max_safe_order} for stable fitting.",
            UserWarning
        )
        return max_safe_order
    
    return sh_order


def estimate_directions(data, gtab, white_matter, sh_order=6, sphere_name="symmetric362", verbose=False):
    """Estimate principal diffusion directions using CSA ODF model. 
       Code adapted from https://docs.dipy.org/dev/examples_built/streamline_analysis/streamline_tools.html
    
    Args:
        data (ndarray): 4D DWI data array (X, Y, Z, N).
        gtab (GradientTable): DIPY gradient table.
        white_matter (ndarray): Binary white matter mask for direction estimation.
        sh_order (int): Maximum spherical harmonic order (default 6).
            Will be automatically reduced if insufficient gradient directions.
        sphere_name (str): Name of sphere to build model around. symmetric 362 for speed, default_sphere for accuracy.
        verbose (bool): Print processing details.
        
    Returns:
        PeaksAndMetrics: Direction field object containing:
            - peak_dirs: Principal directions per voxel
            - peak_values: ODF values at peaks
            - peak_indices: Indices into the sphere
            - gfa: Generalized fractional anisotropy
    """
    from dipy.reconst.shm import CsaOdfModel
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere

    sphere = get_sphere(name=sphere_name)
    
    # Validate and potentially reduce SH order based on available directions
    validated_sh_order = validate_sh_order(gtab, sh_order, verbose=verbose)
    
    if verbose:
        print(f"  → Estimating directions (CSA ODF, SH order={validated_sh_order})...")
        print(f"    • White matter voxels: {white_matter.sum():,}")
    
    csamodel = CsaOdfModel(gtab, sh_order_max=validated_sh_order)
    
    csapeaks = peaks_from_model(
        model=csamodel,
        data=data,
        sphere=sphere,
        relative_peak_threshold=DEFAULT_RELATIVE_PEAK_THRESHOLD,
        min_separation_angle=DEFAULT_MIN_SEPARATION_ANGLE,
        mask=white_matter,
        npeaks=1,  # Single direction for deterministic tracking
    )
    
    if verbose:
        # Count voxels with valid peaks
        valid_peaks = (csapeaks.peak_values[..., 0] > 0).sum()
        coverage = valid_peaks / white_matter.sum() * 100
        print(f"    • Voxels with valid peaks: {valid_peaks:,}")
        print(f"    • Coverage: {coverage:.1f}%")
    
    return csapeaks