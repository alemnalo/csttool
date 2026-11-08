"""
Tracking pipeline modules.

Exports the main pipeline functions for tractography and visualization.
"""

from .brain_mask import (
    apply_brain_mask,
    load_external_mask,
    resolve_brain_mask,
)
from .wm_mask import (
    dam_shell_count,
    load_external_wm_mask,
    resolve_wm_mask,
    segment_wm_dam,
    WM_METHODS,
)
from .load_and_mask import load_and_mask
from .fit_tensors import fit_tensors, tensor_validity_mask
from .estimate_directions import (
    estimate_directions,
    get_max_sh_order,
    validate_sh_order,
    _count_gradient_directions,
)
from .seed_and_stop import seed_and_stop
from .run_tractography import run_tractography
from .save_tracking_outputs import save_tracking_outputs

from .visualizations import (
    plot_tensor_maps,
    plot_white_matter_mask,
    plot_streamlines_2d,
    plot_streamline_statistics,
    create_tracking_summary,
    save_all_tracking_visualizations,
)

__all__ = [
    # Pipeline functions
    'load_and_mask',
    'resolve_brain_mask',
    'load_external_mask',
    'apply_brain_mask',
    'fit_tensors',
    'tensor_validity_mask',
    'estimate_directions',
    'get_max_sh_order',
    'validate_sh_order',
    'resolve_wm_mask',
    'segment_wm_dam',
    'dam_shell_count',
    'load_external_wm_mask',
    'WM_METHODS',
    'seed_and_stop',
    'run_tractography',
    'save_tracking_outputs',
    # Visualization functions
    'plot_tensor_maps',
    'plot_white_matter_mask',
    'plot_streamlines_2d',
    'plot_streamline_statistics',
    'create_tracking_summary',
    'save_all_tracking_visualizations',
]