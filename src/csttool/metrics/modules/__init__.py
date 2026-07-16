"""
Metrics computation modules.

This package contains modular components for CST analysis:
- unilateral_analysis: Analyze individual CST hemispheres
- bilateral_analysis: Compare left vs right CST
- visualizations: Generate all plots and figures
"""

from .unilateral_analysis import analyze_cst_hemisphere
from .bilateral_analysis import compare_bilateral_cst, compute_laterality_indices
from .visualizations import (
    plot_tract_profiles,
    plot_bilateral_comparison,
    create_summary_figure
)

__all__ = [
    # Unilateral
    'analyze_cst_hemisphere',
    
    # Bilateral
    'compare_bilateral_cst',
    'compute_laterality_indices',
    
    # Visualizations
    'plot_tract_profiles',
    'plot_bilateral_comparison',
    'create_summary_figure',
]