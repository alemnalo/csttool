"""
visualizations.py - Extract Module

Visualization functions for CST extraction QC.

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from pathlib import Path

from csttool.viz import geometry as _geo
from csttool.viz.utils import deterministic_subsample


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _pad_slice_to_square(image_slice, extent=None, pad_value=0.0):
    """
    Pad a 2D slice to a square shape.

    Returns the padded slice and an updated display extent that preserves
    the original pixel spacing.
    """
    height, width = image_slice.shape
    target_size = max(height, width)
    pad_y = target_size - height
    pad_x = target_size - width
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before
    pad_x_before = pad_x // 2
    pad_x_after = pad_x - pad_x_before

    padded = np.pad(
        image_slice,
        ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
        mode='constant',
        constant_values=pad_value
    )

    if extent is None:
        extent = (0, width, 0, height)

    x_min, x_max, y_min, y_max = extent
    dx = (x_max - x_min) / width if width else 1.0
    dy = (y_max - y_min) / height if height else 1.0
    padded_extent = (
        x_min - pad_x_before * dx,
        x_max + pad_x_after * dx,
        y_min - pad_y_before * dy,
        y_max + pad_y_after * dy
    )

    return padded, padded_extent


def _volume_world_bounds(volume_shape, affine):
    corners = np.array([
        [0, 0, 0],
        [volume_shape[0], 0, 0],
        [0, volume_shape[1], 0],
        [0, 0, volume_shape[2]],
        [volume_shape[0], volume_shape[1], 0],
        [volume_shape[0], 0, volume_shape[2]],
        [0, volume_shape[1], volume_shape[2]],
        [volume_shape[0], volume_shape[1], volume_shape[2]],
    ])
    corners_h = np.hstack([corners, np.ones((corners.shape[0], 1))])
    world = corners_h @ affine.T
    bounds = []
    for dim in range(3):
        bounds.append((world[:, dim].min(), world[:, dim].max()))
    return bounds


def _streamline_plane_limits(streamlines, d1, d2, fallback_bounds):
    min_d1 = None
    max_d1 = None
    min_d2 = None
    max_d2 = None
    for sl in streamlines:
        if sl.size == 0:
            continue
        sl_d1 = sl[:, d1]
        sl_d2 = sl[:, d2]
        sl_min_d1 = sl_d1.min()
        sl_max_d1 = sl_d1.max()
        sl_min_d2 = sl_d2.min()
        sl_max_d2 = sl_d2.max()
        min_d1 = sl_min_d1 if min_d1 is None else min(min_d1, sl_min_d1)
        max_d1 = sl_max_d1 if max_d1 is None else max(max_d1, sl_max_d1)
        min_d2 = sl_min_d2 if min_d2 is None else min(min_d2, sl_min_d2)
        max_d2 = sl_max_d2 if max_d2 is None else max(max_d2, sl_max_d2)

    if min_d1 is None or min_d2 is None:
        (min_d1, max_d1), (min_d2, max_d2) = fallback_bounds

    range_d1 = max_d1 - min_d1
    range_d2 = max_d2 - min_d2
    pad_d1 = range_d1 * 0.05 if range_d1 else 1.0
    pad_d2 = range_d2 * 0.05 if range_d2 else 1.0

    return (min_d1 - pad_d1, max_d1 + pad_d1), (min_d2 - pad_d2, max_d2 + pad_d2)


# =============================================================================
# REGISTRATION QC VISUALIZATION
# =============================================================================

def plot_registration_comparison(
    subject_fa,
    mni_warped,
    output_dir,
    subject_id=None,
    affine=None,
    verbose=True
):
    """
    Create registration QC visualization.
    
    Shows subject FA and warped MNI template side-by-side
    in three orthogonal views for registration QC.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    # Get slice indices
    mid_ax = subject_fa.shape[2] // 2
    mid_cor = subject_fa.shape[1] // 2
    mid_sag = subject_fa.shape[0] // 2
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    fig.suptitle(f"Registration QC - {subject_id or 'Subject'}\nMNI → Subject Space",
                 fontsize=14, fontweight='bold')
    
    views = [
        ('Axial', subject_fa[:, :, mid_ax], mni_warped[:, :, mid_ax]),
        ('Coronal', subject_fa[:, mid_cor, :], mni_warped[:, mid_cor, :]),
        ('Sagittal', subject_fa[mid_sag, :, :], mni_warped[mid_sag, :, :]),
    ]
    
    for row, (view_name, fa_slice, mni_slice) in enumerate(views):
        padded_fa, padded_extent = _pad_slice_to_square(
            fa_slice.T,
            extent=(0, fa_slice.T.shape[1], 0, fa_slice.T.shape[0]),
            pad_value=0.0
        )
        padded_mni, _ = _pad_slice_to_square(
            mni_slice.T,
            extent=(0, mni_slice.T.shape[1], 0, mni_slice.T.shape[0]),
            pad_value=0.0
        )
        # Subject FA
        axes[row, 0].imshow(
            padded_fa,
            cmap='gray',
            origin='lower',
            vmin=0,
            vmax=1,
            extent=padded_extent
        )
        axes[row, 0].set_title(f'{view_name}\nSubject FA' if row == 0 else '')
        axes[row, 0].axis('off')
        axes[row, 0].set_box_aspect(1)
        
        # MNI warped
        axes[row, 1].imshow(
            padded_mni,
            cmap='gray',
            origin='lower',
            extent=padded_extent
        )
        axes[row, 1].set_title(f'{view_name}\nMNI Warped' if row == 0 else '')
        axes[row, 1].axis('off')
        axes[row, 1].set_box_aspect(1)
        
        # Overlay
        axes[row, 2].imshow(
            padded_fa,
            cmap='gray',
            origin='lower',
            vmin=0,
            vmax=1,
            extent=padded_extent
        )
        axes[row, 2].imshow(
            padded_mni,
            cmap='hot',
            alpha=0.3,
            origin='lower',
            extent=padded_extent
        )
        axes[row, 2].set_title(f'{view_name}\nOverlay' if row == 0 else '')
        axes[row, 2].axis('off')
        axes[row, 2].set_box_aspect(1)
        
        # Add view label
        axes[row, 0].text(-0.15, 0.5, view_name, transform=axes[row, 0].transAxes,
                         fontsize=12, fontweight='bold', va='center', ha='right',
                         rotation=90)

        # Radiological orientation + L/R markers for all three panels of this view.
        if affine is not None:
            view = view_name.lower()
            for c in range(3):
                _geo.finalize_image_view(axes[row, c], affine, view)

    fig_path = viz_dir / f"{prefix}registration_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Registration QC: {fig_path}")
    
    return fig_path


# =============================================================================
# ROI MASK VISUALIZATION
# =============================================================================

def plot_roi_masks(
    fa,
    masks,
    output_dir,
    subject_id=None,
    affine=None,
    verbose=True
):
    """
    Create ROI masks visualization.
    
    Shows motor cortex and brainstem ROI masks overlaid on FA
    in three orthogonal views.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"ROI Masks - {subject_id or 'Subject'}\nMotor Cortex (L/R) + Brainstem",
                 fontsize=14, fontweight='bold')
    
    # Get masks (keys from create_cst_roi_masks: 'motor_left', 'motor_right', 'brainstem')
    motor_left = masks.get('motor_left', np.zeros_like(fa))
    motor_right = masks.get('motor_right', np.zeros_like(fa))
    brainstem = masks.get('brainstem', np.zeros_like(fa))
    
    views = [
        ('Axial', fa[:, :, mid_ax], motor_left[:, :, mid_ax], 
         motor_right[:, :, mid_ax], brainstem[:, :, mid_ax]),
        ('Coronal', fa[:, mid_cor, :], motor_left[:, mid_cor, :],
         motor_right[:, mid_cor, :], brainstem[:, mid_cor, :]),
        ('Sagittal', fa[mid_sag, :, :], motor_left[mid_sag, :, :],
         motor_right[mid_sag, :, :], brainstem[mid_sag, :, :]),
    ]
    
    for col, (view_name, fa_slice, ml_slice, mr_slice, bs_slice) in enumerate(views):
        padded_fa, padded_extent = _pad_slice_to_square(
            fa_slice.T,
            extent=(0, fa_slice.T.shape[1], 0, fa_slice.T.shape[0]),
            pad_value=0.0
        )
        padded_ml, _ = _pad_slice_to_square(
            ml_slice.T,
            extent=(0, ml_slice.T.shape[1], 0, ml_slice.T.shape[0]),
            pad_value=0.0
        )
        padded_mr, _ = _pad_slice_to_square(
            mr_slice.T,
            extent=(0, mr_slice.T.shape[1], 0, mr_slice.T.shape[0]),
            pad_value=0.0
        )
        padded_bs, _ = _pad_slice_to_square(
            bs_slice.T,
            extent=(0, bs_slice.T.shape[1], 0, bs_slice.T.shape[0]),
            pad_value=0.0
        )
        # Row 0: FA only
        axes[0, col].imshow(
            padded_fa,
            cmap='gray',
            origin='lower',
            vmin=0,
            vmax=0.8,
            extent=padded_extent
        )
        axes[0, col].set_title(f'{view_name}\nFA background')
        axes[0, col].axis('off')
        axes[0, col].set_box_aspect(1)
        
        # Row 1: FA with ROI overlays
        axes[1, col].imshow(
            padded_fa,
            cmap='gray',
            origin='lower',
            vmin=0,
            vmax=0.8,
            extent=padded_extent
        )
        
        # Create colored overlays
        # Motor cortex left (blue)
        ml_rgb = np.zeros((*padded_ml.shape, 4))  # RGBA array
        ml_rgb[padded_ml > 0] = [0, 0, 1, 0.6]  # Blue with alpha
        
        # Motor cortex right (red)
        mr_rgb = np.zeros((*padded_mr.shape, 4))
        mr_rgb[padded_mr > 0] = [1, 0, 0, 0.6]  # Red with alpha
        
        # Brainstem (green)
        bs_rgb = np.zeros((*padded_bs.shape, 4))
        bs_rgb[padded_bs > 0] = [0, 1, 0, 0.6]  # Green with alpha
        
        # Apply overlays
        axes[1, col].imshow(ml_rgb, origin='lower', extent=padded_extent)
        axes[1, col].imshow(mr_rgb, origin='lower', extent=padded_extent)
        axes[1, col].imshow(bs_rgb, origin='lower', extent=padded_extent)
        
        axes[1, col].set_title(f'{view_name}\nROI overlay')
        axes[1, col].axis('off')
        axes[1, col].set_box_aspect(1)

        # Enforce radiological orientation + L/R markers (axial/coronal only).
        if affine is not None:
            view = view_name.lower()
            _geo.finalize_image_view(axes[0, col], affine, view)
            _geo.finalize_image_view(axes[1, col], affine, view)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Motor Cortex Left'),
        Patch(facecolor='red', alpha=0.6, label='Motor Cortex Right'),
        Patch(facecolor='green', alpha=0.6, label='Brainstem'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11)
    
    fig_path = viz_dir / f"{prefix}roi_masks.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ ROI masks: {fig_path}")
    
    return fig_path


# =============================================================================
# JACOBIAN MAP VISUALIZATION (REGISTRATION QUALITY DIAGNOSTICS)
# =============================================================================

def plot_jacobian_map(
    jacobian_det,
    fa,
    output_dir,
    subject_id=None,
    affine=None,
    verbose=True
):
    """
    Visualize Jacobian determinant map overlaid on FA.

    Shows local expansion (red) and compression (blue) from registration.
    Asymmetric patterns indicate hemisphere-specific registration issues.

    Parameters
    ----------
    jacobian_det : ndarray
        3D array of Jacobian determinant values from registration.
    fa : ndarray
        3D FA map for background.
    output_dir : str or Path
        Output directory.
    subject_id : str, optional
        Subject identifier for filename.
    verbose : bool
        Print progress information.

    Returns
    -------
    fig_path : Path
        Path to saved visualization.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{subject_id}_" if subject_id else ""

    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Registration Jacobian - {subject_id or 'Subject'}\n"
                 "Red=expansion, Blue=compression", fontsize=14, fontweight='bold')

    views = [
        ('Axial', fa[:, :, mid_ax].T, jacobian_det[:, :, mid_ax].T),
        ('Coronal', fa[:, mid_cor, :].T, jacobian_det[:, mid_cor, :].T),
        ('Sagittal', fa[mid_sag, :, :].T, jacobian_det[mid_sag, :, :].T),
    ]

    for col, (name, fa_slice, jac_slice) in enumerate(views):
        padded_fa, padded_extent = _pad_slice_to_square(
            fa_slice,
            extent=(0, fa_slice.shape[1], 0, fa_slice.shape[0]),
            pad_value=0.0
        )
        padded_jac, _ = _pad_slice_to_square(
            jac_slice,
            extent=(0, jac_slice.shape[1], 0, jac_slice.shape[0]),
            pad_value=1.0  # Neutral Jacobian value
        )

        # Row 0: FA
        axes[0, col].imshow(padded_fa, cmap='gray', origin='lower',
                           vmin=0, vmax=0.8, extent=padded_extent)
        axes[0, col].set_title(f'{name}\nFA background')
        axes[0, col].axis('off')
        axes[0, col].set_box_aspect(1)

        # Row 1: Jacobian overlay
        axes[1, col].imshow(padded_fa, cmap='gray', origin='lower',
                           vmin=0, vmax=0.8, extent=padded_extent)
        im = axes[1, col].imshow(padded_jac, cmap='RdBu_r', origin='lower',
                                  alpha=0.6, vmin=0.5, vmax=1.5, extent=padded_extent)
        axes[1, col].set_title(f'{name}\nJacobian overlay')
        axes[1, col].axis('off')
        axes[1, col].set_box_aspect(1)

        if affine is not None:
            view = name.lower()
            _geo.finalize_image_view(axes[0, col], affine, view)
            _geo.finalize_image_view(axes[1, col], affine, view)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes[1, :], orientation='horizontal',
                        fraction=0.05, pad=0.1)
    cbar.set_label('Jacobian Determinant (1.0 = no change)')

    fig_path = viz_dir / f"{prefix}jacobian_map.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    if verbose:
        print(f"✓ Jacobian map: {fig_path}")

    return fig_path


# =============================================================================
# CST EXTRACTION VISUALIZATION (SIMPLIFIED - NO FA BACKGROUND ON STREAMLINES)
# =============================================================================

def plot_cst_extraction(
    cst_result,
    fa,
    affine,
    output_dir,
    subject_id=None,
    max_streamlines=2000,
    verbose=True
):
    """
    Create CST extraction visualization.
    
    Row 0: FA anatomy in three orthogonal views (voxel space)
    Row 1: Streamlines only on neutral background (world coordinates)
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    cst_left = cst_result['cst_left']
    cst_right = cst_result['cst_right']
    stats = cst_result['stats']
    
    # Subsample if needed
    def subsample(streamlines, max_n):
        # Deterministic (seeded) subsample; no global RNG mutation.
        return deterministic_subsample(streamlines, max_n)
    
    left_vis = subsample(cst_left, max_streamlines)
    right_vis = subsample(cst_right, max_streamlines)
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f"CST Extraction - {subject_id or 'Subject'}\n"
                 f"Left: {stats['cst_left_count']:,} | Right: {stats['cst_right_count']:,} | "
                 f"Total: {stats['cst_total_count']:,} ({stats['extraction_rate']:.1f}%)",
                 fontsize=14, fontweight='bold')
    
    # Row 0: FA background only (voxel space, no extent needed)
    views_fa = [
        ('Sagittal', fa[mid_sag, :, :].T),
        ('Coronal', fa[:, mid_cor, :].T),
        ('Axial', fa[:, :, mid_ax].T),
    ]
    
    for col, (name, fa_slice) in enumerate(views_fa):
        ax = axes[0, col]
        padded_slice, padded_extent = _pad_slice_to_square(
            fa_slice,
            extent=(0, fa_slice.shape[1], 0, fa_slice.shape[0]),
            pad_value=0.0
        )
        ax.imshow(
            padded_slice,
            cmap='gray',
            origin='lower',
            vmin=0,
            vmax=0.8,
            extent=padded_extent
        )
        ax.set_title(f'{name}\nFA background')
        ax.axis('off')
        ax.set_box_aspect(1)
        _geo.finalize_image_view(ax, affine, name.lower())

    # Row 1: Streamlines ONLY (no FA background) on neutral background
    views_sl = [
        ('Sagittal (Y-Z)', 1, 2, 'Y (mm)', 'Z (mm)'),   # Y vs Z
        ('Coronal (X-Z)', 0, 2, 'X (mm)', 'Z (mm)'),    # X vs Z
        ('Axial (X-Y)', 0, 1, 'X (mm)', 'Y (mm)'),      # X vs Y
    ]
    
    volume_bounds = _volume_world_bounds(fa.shape, affine)
    plane_limits = {}
    for title, d1, d2, _, _ in views_sl:
        fallback = (volume_bounds[d1], volume_bounds[d2])
        plane_limits[title] = _streamline_plane_limits(
            left_vis + right_vis,
            d1,
            d2,
            fallback
        )

    for col, (name, d1, d2, xlabel, ylabel) in enumerate(views_sl):
        ax = axes[1, col]
        
        # Set neutral background
        ax.set_facecolor('#f0f0f0')
        
        # Plot left CST (blue)
        for sl in left_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='blue', alpha=0.3, linewidth=0.8)
        
        # Plot right CST (red)
        for sl in right_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='red', alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name}')
        (xlim, ylim) = plane_limits[name]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_box_aspect(1)
        # Radiological world orientation + L/R markers for X-bearing planes.
        _geo.finalize_world_plane(ax, d1)
    
    # Add legend INSIDE the bottom-right plot (cleanest solution)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label=f'Left ({stats["cst_left_count"]:,})'),
        Line2D([0], [0], color='red', linewidth=2, label=f'Right ({stats["cst_right_count"]:,})'),
    ]
    
    # Place legend in the bottom-right plot (axial view)
    axes[1, 2].legend(handles=legend_elements, 
                      loc='upper right',  # Or 'lower right' or 'center right'
                      fontsize=10,
                      frameon=True,
                      fancybox=True,
                      framealpha=0.9,
                      edgecolor='gray')
    
    # Optional: Adjust the title of that specific plot
    axes[1, 2].set_title(f'{views_sl[2][0]}\n(Left: {stats["cst_left_count"]:,}, Right: {stats["cst_right_count"]:,})')
    
    fig_path = viz_dir / f"{prefix}cst_extraction.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ CST extraction: {fig_path}")

    return fig_path


# =============================================================================
# HEMISPHERE SEPARATION QC VISUALIZATION
# =============================================================================

def plot_hemisphere_separation(
    cst_result,
    fa,
    affine,
    output_dir,
    subject_id=None,
    max_streamlines=1500,
    verbose=True
):
    """
    Create hemisphere separation QC visualization.

    Shows left and right CST bundles in separate panels with
    anatomical midline reference for verifying correct hemisphere assignment.

    This visualization is specifically designed to QC the hemisphere
    splitting step and catch any cross-hemisphere contamination.

    Parameters
    ----------
    cst_result : dict
        Output from extract_bilateral_cst() with 'cst_left', 'cst_right', 'stats'
    fa : ndarray
        3D FA map for anatomical reference
    affine : ndarray, shape (4, 4)
        Affine matrix for coordinate transforms
    output_dir : str or Path
        Output directory
    subject_id : str, optional
        Subject identifier for filename
    max_streamlines : int
        Maximum streamlines to display per hemisphere
    verbose : bool
        Print progress information

    Returns
    -------
    fig_path : Path
        Path to saved visualization
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{subject_id}_" if subject_id else ""

    cst_left = cst_result['cst_left']
    cst_right = cst_result['cst_right']
    stats = cst_result['stats']

    # Subsample for visualization
    def subsample(streamlines, max_n):
        # Deterministic (seeded) subsample; no global RNG mutation.
        return deterministic_subsample(streamlines, max_n)

    left_vis = subsample(cst_left, max_streamlines)
    right_vis = subsample(cst_right, max_streamlines)

    # Compute anatomical midline in world coordinates
    # X=0 in RAS space is the midsagittal plane
    midline_x = 0.0

    # Create figure: 2 rows x 3 columns
    # Row 0: Coronal view (Left | Combined | Right)
    # Row 1: Axial view (Left | Combined | Right)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    # Calculate left-right ratio
    left_count = stats['cst_left_count']
    right_count = stats['cst_right_count']
    lr_ratio = left_count / right_count if right_count > 0 else float('inf')

    fig.suptitle(
        f"Hemisphere Separation QC - {subject_id or 'Subject'}\n"
        f"Left: {left_count:,} | Right: {right_count:,} | "
        f"L/R Ratio: {lr_ratio:.2f}",
        fontsize=14, fontweight='bold'
    )

    # View configurations. Radiological display (anatomical Left on the viewer's
    # right) is enforced below with explicit R/L markers, so the x label stays plain.
    views = [
        ('Coronal (X-Z)', 0, 2, 'X (mm)', 'Z (mm)'),
        ('Axial (X-Y)', 0, 1, 'X (mm)', 'Y (mm)'),
    ]

    # Compute axis limits from all streamlines
    volume_bounds = _volume_world_bounds(fa.shape, affine)

    for row, (view_name, d1, d2, xlabel, ylabel) in enumerate(views):
        fallback = (volume_bounds[d1], volume_bounds[d2])
        (xlim, ylim) = _streamline_plane_limits(
            left_vis + right_vis, d1, d2, fallback
        )

        # Column 0: Left CST only
        ax_left = axes[row, 0]
        ax_left.set_facecolor('#f5f5f5')
        for sl in left_vis:
            ax_left.plot(sl[:, d1], sl[:, d2], color='blue', alpha=0.4, linewidth=0.8)

        # Add midline reference
        ax_left.axvline(midline_x, color='gray', linestyle='--', linewidth=1.5,
                        alpha=0.7)

        ax_left.set_xlim(xlim)
        ax_left.set_ylim(ylim)
        ax_left.set_xlabel(xlabel)
        ax_left.set_ylabel(ylabel)
        ax_left.set_title(f'LEFT CST\n{view_name}' if row == 0 else '')
        ax_left.set_aspect('equal', adjustable='box')
        ax_left.grid(True, alpha=0.3, color='white')

        # Column 1: Combined (both hemispheres)
        ax_both = axes[row, 1]
        ax_both.set_facecolor('#f5f5f5')
        for sl in left_vis:
            ax_both.plot(sl[:, d1], sl[:, d2], color='blue', alpha=0.3, linewidth=0.8)
        for sl in right_vis:
            ax_both.plot(sl[:, d1], sl[:, d2], color='red', alpha=0.3, linewidth=0.8)

        ax_both.axvline(midline_x, color='black', linestyle='-', linewidth=2,
                        alpha=0.8)

        ax_both.set_xlim(xlim)
        ax_both.set_ylim(ylim)
        ax_both.set_xlabel(xlabel)
        ax_both.set_ylabel(ylabel)
        ax_both.set_title(f'BILATERAL\n{view_name}' if row == 0 else '')
        ax_both.set_aspect('equal', adjustable='box')
        ax_both.grid(True, alpha=0.3, color='white')

        # Column 2: Right CST only
        ax_right = axes[row, 2]
        ax_right.set_facecolor('#f5f5f5')
        for sl in right_vis:
            ax_right.plot(sl[:, d1], sl[:, d2], color='red', alpha=0.4, linewidth=0.8)

        ax_right.axvline(midline_x, color='gray', linestyle='--', linewidth=1.5,
                         alpha=0.7)

        ax_right.set_xlim(xlim)
        ax_right.set_ylim(ylim)
        ax_right.set_xlabel(xlabel)
        ax_right.set_ylabel(ylabel)
        ax_right.set_title(f'RIGHT CST\n{view_name}' if row == 0 else '')
        ax_right.set_aspect('equal', adjustable='box')
        ax_right.grid(True, alpha=0.3, color='white')

        # Radiological world orientation + L/R markers (all three panels of the row).
        for ax in (ax_left, ax_both, ax_right):
            _geo.finalize_world_plane(ax, d1)

    # Add legend to middle panel
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label=f'Left ({left_count:,})'),
        Line2D([0], [0], color='red', linewidth=2, label=f'Right ({right_count:,})'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Midsagittal plane'),
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=9,
                      frameon=True, fancybox=True, framealpha=0.9)

    # Compute QC metrics: check for midline crossings
    left_x_coords = np.concatenate([np.asarray(sl)[:, 0] for sl in left_vis]) if left_vis else np.array([])
    right_x_coords = np.concatenate([np.asarray(sl)[:, 0] for sl in right_vis]) if right_vis else np.array([])

    # Left CST should be predominantly X < 0 (left hemisphere in RAS)
    # Right CST should be predominantly X > 0 (right hemisphere in RAS)
    tolerance_mm = 5.0  # Allow 5mm tolerance for midline
    left_wrong_side = np.sum(left_x_coords > tolerance_mm) / len(left_x_coords) * 100 if len(left_x_coords) > 0 else 0
    right_wrong_side = np.sum(right_x_coords < -tolerance_mm) / len(right_x_coords) * 100 if len(right_x_coords) > 0 else 0

    qc_text = (
        f"QC Metrics:\n"
        f"Left CST points in right hemisphere: {left_wrong_side:.1f}%\n"
        f"Right CST points in left hemisphere: {right_wrong_side:.1f}%"
    )

    # Color-code QC status
    if left_wrong_side > 10 or right_wrong_side > 10:
        qc_color = '#ffcccc'  # Light red - Warning
        qc_text += "\n[WARNING: High cross-hemisphere contamination]"
    else:
        qc_color = '#ccffcc'  # Light green - OK
        qc_text += "\n[OK: Good hemisphere separation]"

    fig.text(0.5, 0.02, qc_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor=qc_color, alpha=0.8))

    # Save
    fig_path = viz_dir / f"{prefix}hemisphere_separation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    if verbose:
        print(f"✓ Hemisphere separation QC: {fig_path}")

    return fig_path


def create_extraction_summary(
    cst_result,
    fa,
    masks,
    affine,
    output_dir,
    subject_id=None,
    max_streamlines=1000,
    verbose=True
):
    """
    Create multi-panel extraction summary figure WITHOUT length histogram.
    """
    from dipy.tracking.streamline import length
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{subject_id}_" if subject_id else ""
    
    cst_left = cst_result['cst_left']
    cst_right = cst_result['cst_right']
    stats = cst_result['stats']
    
    # Compute lengths (for statistics only, not visualization)
    left_lengths = np.array([length(s) for s in cst_left]) if len(cst_left) > 0 else np.array([])
    right_lengths = np.array([length(s) for s in cst_right]) if len(cst_right) > 0 else np.array([])
    
    # Subsample for visualization
    def subsample(streamlines, max_n):
        # Deterministic (seeded) subsample; no global RNG mutation.
        return deterministic_subsample(streamlines, max_n)
    
    left_vis = subsample(cst_left, max_streamlines)
    right_vis = subsample(cst_right, max_streamlines)
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Create figure with optimized grid
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    fig.suptitle(f"CST Extraction Summary - {subject_id or 'Subject'}",
                 fontsize=16, fontweight='bold')
    
    # New grid: 3 rows, 3 columns with adjusted ratios
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1.4, 0.6], 
                         hspace=0.2, wspace=0.2)
    
    # ===================================================================
    # ROW 0: ROI MASKS (3 views with legends)
    # ===================================================================
    
    # Get masks
    motor_left = masks.get('motor_left', np.zeros_like(fa))
    motor_right = masks.get('motor_right', np.zeros_like(fa))
    brainstem = masks.get('brainstem', np.zeros_like(fa))
    
    roi_views = [
        ('Axial', fa[:, :, mid_ax], motor_left[:, :, mid_ax], 
         motor_right[:, :, mid_ax], brainstem[:, :, mid_ax]),
        ('Coronal', fa[:, mid_cor, :], motor_left[:, mid_cor, :],
         motor_right[:, mid_cor, :], brainstem[:, mid_cor, :]),
        ('Sagittal', fa[mid_sag, :, :], motor_left[mid_sag, :, :],
         motor_right[mid_sag, :, :], brainstem[mid_sag, :, :]),
    ]
    
    for col, (name, fa_slice, ml_slice, mr_slice, bs_slice) in enumerate(roi_views):
        ax = fig.add_subplot(gs[0, col])
        
        # FA background
        padded_fa, padded_extent = _pad_slice_to_square(
            fa_slice.T,
            extent=(0, fa_slice.T.shape[1], 0, fa_slice.T.shape[0]),
            pad_value=0.0
        )
        ax.imshow(padded_fa, cmap='gray', origin='lower', 
                 vmin=0, vmax=0.8, extent=padded_extent)
        
        # Create combined overlay
        padded_ml, _ = _pad_slice_to_square(ml_slice.T)
        padded_mr, _ = _pad_slice_to_square(mr_slice.T)
        padded_bs, _ = _pad_slice_to_square(bs_slice.T)
        
        # Create a single RGBA overlay for all ROIs
        overlay = np.zeros((*padded_fa.shape, 4))
        overlay[padded_ml > 0] = [0, 0, 1, 0.5]    # Blue for left motor
        overlay[padded_mr > 0] = [1, 0, 0, 0.5]    # Red for right motor
        overlay[padded_bs > 0] = [0, 1, 0, 0.5]    # Green for brainstem
        
        ax.imshow(overlay, origin='lower', extent=padded_extent)
        ax.set_title(f'{name}: ROI Masks', fontsize=12)
        ax.axis('off')
        _geo.finalize_image_view(ax, affine, name.lower())

        # Add ROI legend only to the sagittal view (last column)
        if col == 2:  # Sagittal view
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.5, label='Motor L'),
                Patch(facecolor='red', alpha=0.5, label='Motor R'),
                Patch(facecolor='green', alpha=0.5, label='Brainstem'),
            ]
            # Place legend in upper right corner
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                      frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')
    
    # ===================================================================
    # ROW 1: CST STREAMLINES (3 views with legend)
    # ===================================================================
    
    streamline_views = [
        ('Axial (X-Y)', 0, 1, 'X (mm)', 'Y (mm)'),
        ('Coronal (X-Z)', 0, 2, 'X (mm)', 'Z (mm)'),
        ('Sagittal (Y-Z)', 1, 2, 'Y (mm)', 'Z (mm)'),
    ]
    
    volume_bounds = _volume_world_bounds(fa.shape, affine)
    plane_limits = {}
    for title, d1, d2, _, _ in streamline_views:
        fallback = (volume_bounds[d1], volume_bounds[d2])
        plane_limits[title] = _streamline_plane_limits(
            left_vis + right_vis,
            d1,
            d2,
            fallback
        )
    
    for col, (title, d1, d2, xlabel, ylabel) in enumerate(streamline_views):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#f0f0f0')
        
        # Plot left CST
        for sl in left_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='blue', alpha=0.25, linewidth=0.8)
        
        # Plot right CST
        for sl in right_vis:
            ax.plot(sl[:, d1], sl[:, d2], color='red', alpha=0.25, linewidth=0.8)
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{title}: CST', fontsize=12)
        
        (xlim, ylim) = plane_limits[title]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, color='white')
        ax.tick_params(labelsize=10)
        _geo.finalize_world_plane(ax, d1)

        # Add CST legend only to the sagittal view (last column)
        if col == 2:  # Sagittal view
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=2, label=f'Left ({stats["cst_left_count"]:,})'),
                Line2D([0], [0], color='red', linewidth=2, label=f'Right ({stats["cst_right_count"]:,})'),
            ]
            # Place legend in upper right corner
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                      frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')
    
    # ===================================================================
    # ROW 2: COMPACT STATISTICS PANEL
    # ===================================================================
    
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Build statistics with and without lengths
    if len(left_lengths) > 0:
        left_stats = f"{stats['cst_left_count']:,} streamlines (mean: {np.mean(left_lengths):.1f} mm)"
    else:
        left_stats = "No streamlines"
    
    if len(right_lengths) > 0:
        right_stats = f"{stats['cst_right_count']:,} streamlines (mean: {np.mean(right_lengths):.1f} mm)"
    else:
        right_stats = "No streamlines"
    
    stats_text = (
        f"{'─' * 100}\n"
        f"EXTRACTION STATISTICS\n"
        f"{'─' * 100}\n\n"
        f"Total Input Streamlines: {stats['total_input']:,}\n"
        f"Extraction Rate:         {stats['extraction_rate']:.2f}% ({stats['cst_total_count']:,} streamlines)\n\n"
        f"Left CST:                {left_stats}\n"
        f"Right CST:               {right_stats}\n\n"
        f"{'─' * 100}"
    )
    
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                 fontsize=11, fontfamily='monospace', linespacing=1.5,
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    fig_path = viz_dir / f"{prefix}extraction_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"✓ Extraction summary: {fig_path}")
    
    return fig_path


# =============================================================================
# CONVENIENCE FUNCTION: Save all visualizations
# =============================================================================

def save_all_extraction_visualizations(
    cst_result,
    fa,
    masks,
    affine,
    output_dir,
    subject_id=None,
    mni_warped=None,
    jacobian_det=None,
    verbose=True
):
    """
    Generate and save all extraction visualizations.

    Parameters
    ----------
    cst_result : dict
        Output from extract_bilateral_cst().
    fa : ndarray
        3D FA map.
    masks : dict
        ROI masks dictionary.
    affine : ndarray
        4x4 affine matrix.
    output_dir : str or Path
        Output directory.
    subject_id : str, optional
        Subject identifier.
    mni_warped : ndarray, optional
        Warped MNI template for registration QC.
    jacobian_det : ndarray, optional
        Jacobian determinant map from registration for deformation QC.
    verbose : bool
        Print progress information.

    Returns
    -------
    viz_paths : dict
        Dictionary of paths to saved visualizations.
    """
    if verbose:
        print("\nGenerating extraction visualizations...")

    viz_paths = {}

    # Registration QC (if MNI warped provided)
    if mni_warped is not None:
        viz_paths['registration_qc'] = plot_registration_comparison(
            fa, mni_warped, output_dir, subject_id, affine=affine, verbose=verbose
        )

    # Jacobian map (if Jacobian data provided)
    if jacobian_det is not None:
        viz_paths['jacobian_map'] = plot_jacobian_map(
            jacobian_det, fa, output_dir, subject_id, affine=affine, verbose=verbose
        )

    # ROI masks
    viz_paths['roi_masks'] = plot_roi_masks(
        fa, masks, output_dir, subject_id, affine=affine, verbose=verbose
    )

    # CST extraction
    viz_paths['cst_extraction'] = plot_cst_extraction(
        cst_result, fa, affine, output_dir, subject_id, verbose=verbose
    )

    # Summary figure
    viz_paths['extraction_summary'] = create_extraction_summary(
        cst_result, fa, masks, affine, output_dir, subject_id, verbose=verbose
    )

    # Hemisphere separation QC
    viz_paths['hemisphere_separation'] = plot_hemisphere_separation(
        cst_result, fa, affine, output_dir, subject_id, verbose=verbose
    )

    if verbose:
        print(f"✓ All extraction visualizations saved to: {Path(output_dir) / 'visualizations'}")

    return viz_paths
