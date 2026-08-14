"""
Preprocessing package for csttool.

Exports the main preprocessing orchestrator and module functions.
"""

from .preprocess import run_preprocessing

from .modules.load_dataset import load_dataset
from .modules.denoise import denoise
from .modules.gibbs_unringing import gibbs_unringing
from .modules.background_segmentation import background_segmentation
from .modules.perform_motion_correction import perform_motion_correction
from .modules.reorient_gradients import (
    max_rotation_angle_deg,
    rotate_bvecs_for_motion,
)
from .modules.save_preprocessed import save_preprocessed

from .modules.visualizations import (
    plot_denoising_comparison,
    plot_gibbs_unringing_comparison,
    plot_brain_mask_overlay,
    plot_motion_correction_summary,
    create_preprocessing_summary,
    save_all_preprocessing_visualizations,
)

__all__ = [
    # Main orchestrator
    'run_preprocessing',
    # Individual modules
    'load_dataset',
    'denoise',
    'gibbs_unringing',
    'background_segmentation',
    'perform_motion_correction',
    'rotate_bvecs_for_motion',
    'max_rotation_angle_deg',
    'save_preprocessed',
    # Visualizations
    'plot_denoising_comparison',
    'plot_gibbs_unringing_comparison',
    'plot_brain_mask_overlay',
    'plot_motion_correction_summary',
    'create_preprocessing_summary',
    'save_all_preprocessing_visualizations',
]
