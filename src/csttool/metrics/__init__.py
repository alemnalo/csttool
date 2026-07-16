"""
CST Metrics Module

Modular metrics computation for bilateral CST analysis.

Structure:
- modules/unilateral_analysis: Analyze individual hemispheres
- modules/bilateral_analysis: Compare left vs right
- modules/visualizations: Generate all plots
"""

# Import key functions from modules
from .modules.unilateral_analysis import (
    analyze_cst_hemisphere,
    compute_morphology,
    sample_scalar_along_tract,
    compute_tract_profile,
    orient_streamlines_inferior_to_superior,
    print_hemisphere_summary
)

from .modules.bilateral_analysis import (
    compare_bilateral_cst,
    compute_laterality_indices,
    compute_li,
    print_bilateral_summary,
    assess_clinical_significance
)

from .modules.visualizations import (
    plot_tract_profiles,
    plot_bilateral_comparison,
    create_summary_figure,
    plot_asymmetry_radar,
    plot_stacked_profiles,
    plot_tractogram_qc_preview,
)

from .modules.reports import (
    save_json_report,
    save_csv_summary,
    save_html_report,
    save_pdf_report,
    generate_complete_report,
)

__all__ = [
    # Unilateral analysis
    'analyze_cst_hemisphere',
    'compute_morphology',
    'sample_scalar_along_tract',
    'compute_tract_profile',
    'orient_streamlines_inferior_to_superior',
    'print_hemisphere_summary',
    
    # Bilateral analysis
    'compare_bilateral_cst',
    'compute_laterality_indices',
    'compute_li',
    'print_bilateral_summary',
    'assess_clinical_significance',
    
    # Visualizations
    'plot_tract_profiles',
    'plot_bilateral_comparison',
    'create_summary_figure',
    'plot_asymmetry_radar',
    'plot_stacked_profiles',
    'plot_tractogram_qc_preview',
    
    # Reports
    'save_json_report',
    'save_csv_summary',
    'save_html_report',
    'save_pdf_report',
    'generate_complete_report',
]