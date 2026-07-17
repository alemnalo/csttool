"""
bilateral_analysis.py

Functions for bilateral CST comparison and asymmetry analysis.

This module takes outputs from unilateral_analysis and computes:
- Laterality indices (LI) for volume, FA, MD
- Asymmetry metrics
- Statistical comparisons
"""

import numpy as np


def compare_bilateral_cst(left_metrics, right_metrics):
    """
    Compare left and right CST metrics and compute asymmetry measures.
    
    Parameters
    ----------
    left_metrics : dict
        Metrics from analyze_cst_hemisphere() for left CST
    right_metrics : dict
        Metrics from analyze_cst_hemisphere() for right CST
        
    Returns
    -------
    comparison : dict
        Comprehensive bilateral comparison containing:
        - left: complete left hemisphere metrics
        - right: complete right hemisphere metrics
        - asymmetry: laterality indices and differences
    """
    
    print("\nComputing bilateral comparison...")
    
    comparison = {
        'left': left_metrics,
        'right': right_metrics,
        'asymmetry': compute_laterality_indices(left_metrics, right_metrics)
    }
    
    print_bilateral_summary(comparison)
    
    return comparison


def compute_laterality_indices(left_metrics, right_metrics):
    """
    Compute laterality indices for all available metrics.
    
    Laterality Index (LI) = (L - R) / (L + R)
    
    LI interpretation:
    - LI > 0: Left hemisphere larger/higher
    - LI < 0: Right hemisphere larger/higher  
    - LI ≈ 0: Symmetric
    - |LI| > 0.1: Potentially significant asymmetry
    
    Parameters
    ----------
    left_metrics : dict
        Left hemisphere metrics
    right_metrics : dict
        Right hemisphere metrics
        
    Returns
    -------
    asymmetry : dict
        Laterality indices and absolute differences for:
        - volume
        - streamline count
        - mean length
        - mean FA (if available)
        - mean MD (if available)
        - mean RD (if available)
        - mean AD (if available)
    """
    
    asymmetry = {}
    
    # Morphological asymmetry
    left_morph = left_metrics['morphology']
    right_morph = right_metrics['morphology']
    
    asymmetry['volume'] = compute_li(
        left_morph['tract_volume'],
        right_morph['tract_volume']
    )
    
    asymmetry['streamline_count'] = compute_li(
        left_morph['n_streamlines'],
        right_morph['n_streamlines']
    )
    
    asymmetry['mean_length'] = compute_li(
        left_morph['mean_length'],
        right_morph['mean_length']
    )
    
    # Microstructural asymmetry (global).
    # The headline mean is per-streamline (length-unbiased, AU10), so the global LI is
    # computed on the same population the report tabulates - not the point-weighted
    # mean that longer streamlines dominate. The regional LIs below use the
    # profile-derived values, which are per-streamline-normalised by construction.
    if 'fa' in left_metrics and 'fa' in right_metrics:
        asymmetry['fa'] = compute_li(
            left_metrics['fa']['mean'],
            right_metrics['fa']['mean']
        )

    if 'md' in left_metrics and 'md' in right_metrics:
        asymmetry['md'] = compute_li(
            left_metrics['md']['mean'],
            right_metrics['md']['mean']
        )

    if 'rd' in left_metrics and 'rd' in right_metrics:
        asymmetry['rd'] = compute_li(
            left_metrics['rd']['mean'],
            right_metrics['rd']['mean']
        )

    if 'ad' in left_metrics and 'ad' in right_metrics:
        asymmetry['ad'] = compute_li(
            left_metrics['ad']['mean'],
            right_metrics['ad']['mean']
        )

    # Localized microstructural asymmetry (per region)
    regions = ['pontine', 'plic', 'precentral']
    scalars = ['fa', 'md', 'rd', 'ad']

    for scalar in scalars:
        if scalar in left_metrics and scalar in right_metrics:
            for region in regions:
                if region in left_metrics[scalar] and region in right_metrics[scalar]:
                    key = f'{scalar}_{region}'
                    asymmetry[key] = compute_li(
                        left_metrics[scalar][region],
                        right_metrics[scalar][region]
                    )

    return asymmetry


def compute_li(left_value, right_value):
    """
    Compute laterality index and absolute difference.
    
    Parameters
    ----------
    left_value : float
        Metric value from left hemisphere
    right_value : float
        Metric value from right hemisphere
        
    Returns
    -------
    li_info : dict
        Dictionary containing:
        - laterality_index: (L-R)/(L+R)
        - absolute_difference: |L-R|
        - percent_difference: 100 * |L-R| / mean(L,R)
        - interpretation: text description
    """
    
    total = left_value + right_value
    
    if total == 0:
        return {
            'laterality_index': 0.0,
            'absolute_difference': 0.0,
            'percent_difference': 0.0,
            'interpretation': 'no data'
        }
    
    li = (left_value - right_value) / total
    abs_diff = abs(left_value - right_value)
    mean_value = (left_value + right_value) / 2
    pct_diff = 100 * abs_diff / mean_value if mean_value > 0 else 0.0
    
    # Interpret laterality
    from csttool.defaults import DEFAULT_LI_SYMMETRIC, DEFAULT_LI_MILD, DEFAULT_LI_MODERATE
    if abs(li) < DEFAULT_LI_SYMMETRIC:
        interpretation = 'symmetric'
    elif abs(li) < DEFAULT_LI_MILD:
        interpretation = 'mild asymmetry'
    elif abs(li) < DEFAULT_LI_MODERATE:
        interpretation = 'moderate asymmetry'
    else:
        interpretation = 'strong asymmetry'
    
    if li > 0:
        interpretation += ' (left > right)'
    elif li < 0:
        interpretation += ' (right > left)'
    
    return {
        'laterality_index': float(li),
        'absolute_difference': float(abs_diff),
        'percent_difference': float(pct_diff),
        'interpretation': interpretation,
        'left_value': float(left_value),
        'right_value': float(right_value)
    }


def print_bilateral_summary(comparison):
    """
    Print human-readable bilateral comparison summary.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    """
    
    print("\n" + "=" * 60)
    print("BILATERAL CST COMPARISON")
    print("=" * 60)
    
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    # Morphology comparison
    print("\nMORPHOLOGY:")
    print(f"  Streamline Count:")
    print(f"    Left:  {left['morphology']['n_streamlines']}")
    print(f"    Right: {right['morphology']['n_streamlines']}")
    print(f"    LI:    {asym['streamline_count']['laterality_index']:+.3f} "
          f"({asym['streamline_count']['interpretation']})")
    
    print(f"\n  Tract Volume:")
    print(f"    Left:  {left['morphology']['tract_volume']:.0f} mm³")
    print(f"    Right: {right['morphology']['tract_volume']:.0f} mm³")
    print(f"    LI:    {asym['volume']['laterality_index']:+.3f} "
          f"({asym['volume']['interpretation']})")
    print(f"    Diff:  {asym['volume']['absolute_difference']:.0f} mm³ "
          f"({asym['volume']['percent_difference']:.1f}%)")
    
    print(f"\n  Mean Length:")
    print(f"    Left:  {left['morphology']['mean_length']:.1f} mm")
    print(f"    Right: {right['morphology']['mean_length']:.1f} mm")
    print(f"    LI:    {asym['mean_length']['laterality_index']:+.3f} "
          f"({asym['mean_length']['interpretation']})")
    
    # Microstructural comparison
    if 'fa' in asym:
        print(f"\nFRACTIONAL ANISOTROPY:")
        print(f"    Left:  {left['fa']['mean']:.3f} ± {left['fa']['std']:.3f}")
        print(f"    Right: {right['fa']['mean']:.3f} ± {right['fa']['std']:.3f}")
        print(f"    LI:    {asym['fa']['laterality_index']:+.3f} "
              f"({asym['fa']['interpretation']})")
        print(f"    Diff:  {asym['fa']['absolute_difference']:.3f} "
              f"({asym['fa']['percent_difference']:.1f}%)")
    
    if 'md' in asym:
        print(f"\nMEAN DIFFUSIVITY:")
        print(f"    Left:  {left['md']['mean']:.3e} ± {left['md']['std']:.3e}")
        print(f"    Right: {right['md']['mean']:.3e} ± {right['md']['std']:.3e}")
        print(f"    LI:    {asym['md']['laterality_index']:+.3f} "
              f"({asym['md']['interpretation']})")
        print(f"    Diff:  {asym['md']['absolute_difference']:.3e} "
              f"({asym['md']['percent_difference']:.1f}%)")
    
    print("=" * 60)