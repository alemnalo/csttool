"""
Extract pipeline modules.

Exports the main functions for extraction of the left and right CST.
"""

from .registration import (
    load_mni_template,
    compute_affine_registration,
    compute_syn_registration,
    compute_jacobian_hemisphere_stats,
    compute_warped_midline,
    register_mni_to_subject,
    save_registration_report,
)

from .visualizations import (
    plot_registration_comparison,
    plot_cst_extraction,
    plot_hemisphere_separation,
    plot_jacobian_map,
    create_extraction_summary,
    save_all_extraction_visualizations,
)

__all__ = [
    'load_mni_template',
    'register_mni_to_subject',
    'save_registration_report',
    'compute_affine_registration',
    'compute_syn_registration',
    'compute_jacobian_hemisphere_stats',
    'compute_warped_midline',
    'plot_registration_comparison',
    'plot_cst_extraction',
    'plot_hemisphere_separation',
    'plot_jacobian_map',
    'create_extraction_summary',
    'save_all_extraction_visualizations',
]
