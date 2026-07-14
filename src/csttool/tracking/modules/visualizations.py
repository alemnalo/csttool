"""
visualizations.py - Tracking Module

Visualization functions for tractography QC.

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

from csttool.viz import geometry as _geo
from csttool.viz import style as _style
from csttool.viz.utils import deterministic_subsample, viz_rng


# =============================================================================
# TENSOR MAPS VISUALIZATION
# =============================================================================

def plot_tensor_maps(
    fa,
    md,
    brain_mask,
    output_dir,
    stem,
    tenfit=None,
    affine=None,
    verbose=True
):
    """
    Create tensor-derived scalar maps visualization.
    
    Shows FA, MD, and optionally RGB-encoded principal diffusion
    direction in three orthogonal views.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Compute RGB direction map if tenfit available
    if tenfit is not None:
        try:
            evecs = tenfit.evecs
            v1 = evecs[..., 0]
            rgb = np.abs(v1) * fa[..., np.newaxis]
            rgb = np.clip(rgb, 0, 1)
        except Exception:
            rgb = None
    else:
        rgb = None
    
    # Create figure
    n_cols = 4 if rgb is not None else 3
    fig, axes = plt.subplots(3, n_cols, figsize=(4*n_cols, 12), constrained_layout=True)
    fig.suptitle(f"Tensor Maps - {stem}", fontsize=14, fontweight='bold')
    
    # Statistics for display
    fa_brain = fa[brain_mask > 0]
    md_brain = md[brain_mask > 0]
    
    views = [
        ('Axial', mid_ax, lambda x: x[:, :, mid_ax], 2),
        ('Coronal', mid_cor, lambda x: x[:, mid_cor, :], 1),
        ('Sagittal', mid_sag, lambda x: x[mid_sag, :, :], 0),
    ]
    
    for row, (view_name, slice_idx, slicer, dim) in enumerate(views):
        # FA
        axes[row, 0].imshow(slicer(fa).T, cmap='gray', origin='lower', vmin=0, vmax=1)
        if row == 0:
            axes[row, 0].set_title(f'FA\nmean={fa_brain.mean():.3f}')
        axes[row, 0].axis('off')
        
        # MD (displayed ×10⁻³ mm²/s to match the labelled scale + colorbar)
        md_mappable = axes[row, 1].imshow(
            slicer(md).T * 1000, cmap=_style.MD_CMAP, origin='lower',
            vmin=0, vmax=np.percentile(md_brain * 1000, 99))
        if row == 0:
            axes[row, 1].set_title(f'MD (×10⁻³)\nmean={md_brain.mean()*1000:.2f}')
        axes[row, 1].axis('off')
        
        # Brain mask overlay
        axes[row, 2].imshow(slicer(fa).T, cmap='gray', origin='lower', vmin=0, vmax=1)
        mask_overlay = np.ma.masked_where(slicer(brain_mask).T == 0, slicer(brain_mask).T)
        axes[row, 2].imshow(mask_overlay, cmap='Blues', alpha=0.3, origin='lower')
        if row == 0:
            axes[row, 2].set_title(f'Brain Mask\n{brain_mask.sum():,} voxels')
        axes[row, 2].axis('off')
        
        # RGB if available
        if rgb is not None:
            axes[row, 3].imshow(slicer(rgb).transpose(1, 0, 2), origin='lower')
            if row == 0:
                axes[row, 3].set_title('Direction (RGB)')
            axes[row, 3].axis('off')
        
        # Add view label
        axes[row, 0].text(-0.15, 0.5, view_name, transform=axes[row, 0].transAxes,
                         fontsize=12, fontweight='bold', va='center', ha='right',
                         rotation=90)

        # Radiological orientation + L/R markers for every column of this view.
        if affine is not None:
            for c in range(n_cols):
                _geo.finalize_image_view(axes[row, c], affine, view_name.lower())

    # One shared colorbar for the continuous MD column (constrained_layout makes room).
    _style.add_scalar_colorbar(fig, md_mappable, list(axes[:, 1]),
                               'MD (×10⁻³ mm²/s)')

    fig_path = viz_dir / f"{stem}_tensor_maps.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


# =============================================================================
# WHITE MATTER MASK VISUALIZATION
# =============================================================================

def plot_white_matter_mask(
    fa,
    white_matter,
    brain_mask,
    output_dir,
    stem,
    fa_thresh=0.2,
    affine=None,
    verbose=True
):
    """
    Create white matter mask QC visualization.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Statistics
    wm_voxels = white_matter.sum()
    brain_voxels = brain_mask.sum()
    wm_fraction = wm_voxels / brain_voxels * 100 if brain_voxels > 0 else 0
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"White Matter Mask - {stem}\n"
                 f"FA threshold: {fa_thresh} | WM voxels: {wm_voxels:,} ({wm_fraction:.1f}% of brain)",
                 fontsize=14, fontweight='bold')
    
    views = [
        ('Axial', fa[:, :, mid_ax], white_matter[:, :, mid_ax], brain_mask[:, :, mid_ax]),
        ('Coronal', fa[:, mid_cor, :], white_matter[:, mid_cor, :], brain_mask[:, mid_cor, :]),
        ('Sagittal', fa[mid_sag, :, :], white_matter[mid_sag, :, :], brain_mask[mid_sag, :, :]),
    ]
    
    for col, (view_name, fa_slice, wm_slice, mask_slice) in enumerate(views):
        # Row 0: FA map
        axes[0, col].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[0, col].set_title(f'{view_name}\nFA map')
        axes[0, col].axis('off')
        
        # Row 1: FA with WM overlay
        axes[1, col].imshow(fa_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        wm_overlay = np.ma.masked_where(wm_slice.T == 0, wm_slice.T)
        axes[1, col].imshow(wm_overlay, cmap='Blues', alpha=0.5, origin='lower')
        axes[1, col].contour(mask_slice.T, levels=[0.5], colors='red', 
                            linewidths=1, linestyles='--')
        axes[1, col].set_title(f'{view_name}\nWM mask (blue) + brain (red)')
        axes[1, col].axis('off')

        if affine is not None:
            _geo.finalize_image_view(axes[0, col], affine, view_name.lower())
            _geo.finalize_image_view(axes[1, col], affine, view_name.lower())
    
    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label=f'White Matter (FA > {fa_thresh})'),
        Line2D([0], [0], color='red', linestyle='--', label='Brain Mask'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    fig_path = viz_dir / f"{stem}_wm_mask_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


# =============================================================================
# 2D STREAMLINE PROJECTIONS (SIMPLIFIED - NO FA BACKGROUND)
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


def plot_streamlines_2d(
    streamlines,
    fa,
    affine,
    output_dir,
    stem,
    max_streamlines=5000,
    verbose=True
):
    """
    Create 2D streamline projection visualization.
    
    Shows streamlines projected onto three orthogonal planes
    on a neutral background with axis labels for orientation.
    FA anatomy is shown in a separate row for reference.
    
    Parameters
    ----------
    streamlines : Streamlines
        Tractography streamlines (in RASMM world coordinates).
    fa : ndarray
        3D FA map for reference panels.
    affine : ndarray
        4x4 affine transformation matrix.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    max_streamlines : int, optional
        Maximum number of streamlines to plot (for performance).
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    fig_path : Path
        Path to saved figure.
    """
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    n_streamlines = len(streamlines)
    
    if n_streamlines == 0:
        if verbose:
            print("  ⚠️ No streamlines to visualize")
        return None
    
    # Subsample if needed (deterministic; no global RNG mutation)
    if n_streamlines > max_streamlines:
        vis_streamlines = deterministic_subsample(streamlines, max_streamlines)
        n_vis = len(vis_streamlines)
    else:
        vis_streamlines = streamlines
        n_vis = n_streamlines
    
    # Get FA slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Create figure with 2 rows: FA anatomy + streamlines
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Whole-Brain Tractography - {stem}\n"
                 f"{n_streamlines:,} streamlines (showing {n_vis:,})",
                 fontsize=14, fontweight='bold')
    
    # Row 0: FA anatomy (voxel space)
    fa_views = [
        ('Sagittal', fa[mid_sag, :, :].T),
        ('Coronal', fa[:, mid_cor, :].T),
        ('Axial', fa[:, :, mid_ax].T),
    ]

    for col, (name, fa_slice) in enumerate(fa_views):
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

    # Row 1: Streamlines only (world coordinates, no FA background)
    # World dimensions: 0=X, 1=Y, 2=Z
    # Whole-brain streamlines carry no L/R meaning here -> one neutral colour.
    _n = _style.NEUTRAL_STREAMLINE
    streamline_views = [
        ('Sagittal (Y-Z)', 1, 2, 'Y (mm)', 'Z (mm)', _n),
        ('Coronal (X-Z)', 0, 2, 'X (mm)', 'Z (mm)', _n),
        ('Axial (X-Y)', 0, 1, 'X (mm)', 'Y (mm)', _n),
    ]

    volume_bounds = _volume_world_bounds(fa.shape, affine)
    plane_limits = {}
    for title, d1, d2, _, _, _ in streamline_views:
        fallback = (volume_bounds[d1], volume_bounds[d2])
        plane_limits[title] = _streamline_plane_limits(vis_streamlines, d1, d2, fallback)
    
    for col, (title, d1, d2, xlabel, ylabel, color) in enumerate(streamline_views):
        ax = axes[1, col]
        
        # Set neutral background
        ax.set_facecolor('#f0f0f0')
        
        # Plot streamlines
        for sl in vis_streamlines:
            ax.plot(sl[:, d1], sl[:, d2], alpha=0.15, linewidth=0.3, color=color)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        (xlim, ylim) = plane_limits[title]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_box_aspect(1)
        _geo.finalize_world_plane(ax, d1)

    plt.tight_layout()

    fig_path = viz_dir / f"{stem}_streamlines_2d.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


# =============================================================================
# STREAMLINE STATISTICS VISUALIZATION
# =============================================================================

def plot_streamline_statistics(
    streamlines,
    fa,
    seeds,
    affine,
    output_dir,
    stem,
    verbose=True
):
    """
    Create streamline statistics visualization.
    """
    from dipy.tracking.streamline import length
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    n_streamlines = len(streamlines)
    
    if n_streamlines == 0:
        if verbose:
            print("  ⚠️ No streamlines for statistics")
        return None
    
    # Compute lengths
    lengths = np.array([length(s) for s in streamlines])
    points_per_sl = np.array([len(s) for s in streamlines])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Tractography Statistics - {stem}", fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Length histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(lengths):.1f} mm')
    ax1.axvline(np.median(lengths), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(lengths):.1f} mm')
    ax1.set_xlabel('Streamline Length (mm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Length Distribution')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Points per streamline histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(points_per_sl, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(points_per_sl), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(points_per_sl):.0f}')
    ax2.set_xlabel('Points per Streamline')
    ax2.set_ylabel('Count')
    ax2.set_title('Streamline Resolution')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Length vs points scatter
    ax3 = fig.add_subplot(gs[0, 2])
    sample_idx = np.sort(viz_rng().choice(n_streamlines, min(1000, n_streamlines), replace=False))
    ax3.scatter(lengths[sample_idx], points_per_sl[sample_idx],
               alpha=0.3, s=10, color='purple')
    ax3.set_xlabel('Length (mm)')
    ax3.set_ylabel('Points')
    ax3.set_title('Length vs Points')
    ax3.grid(True, alpha=0.3)
    
    # Seed density visualization
    ax4 = fig.add_subplot(gs[1, 0])
    mid_ax = fa.shape[2] // 2
    ax4.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=0.8)
    
    # Convert seeds to voxel coordinates
    inv_affine = np.linalg.inv(affine)
    seeds_h = np.hstack([seeds, np.ones((len(seeds), 1))])
    seeds_vox = (seeds_h @ inv_affine.T)[:, :3]
    
    near_slice = np.abs(seeds_vox[:, 2] - mid_ax) < 3
    ax4.scatter(seeds_vox[near_slice, 0], seeds_vox[near_slice, 1],
               s=1, alpha=0.3, color='yellow')
    ax4.set_title(f'Seed Points (axial, z≈{mid_ax})')
    ax4.axis('off')
    _geo.finalize_image_view(ax4, affine, 'axial')
    
    # Cumulative length distribution
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_lengths = np.sort(lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax5.plot(sorted_lengths, cumulative, color='steelblue', linewidth=2)
    ax5.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(np.median(lengths), color='orange', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Length (mm)')
    ax5.set_ylabel('Cumulative %')
    ax5.set_title('Cumulative Length Distribution')
    ax5.grid(True, alpha=0.3)
    
    # Statistics text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = (
        f"{'─' * 40}\n"
        f"TRACTOGRAPHY STATISTICS\n"
        f"{'─' * 40}\n\n"
        f"Total Streamlines:    {n_streamlines:,}\n"
        f"Total Seeds:          {len(seeds):,}\n"
        f"Yield Rate:           {n_streamlines/len(seeds)*100:.1f}%\n\n"
        f"Length Statistics:\n"
        f"  Mean:               {np.mean(lengths):.1f} mm\n"
        f"  Median:             {np.median(lengths):.1f} mm\n"
        f"  Std:                {np.std(lengths):.1f} mm\n"
        f"  Min:                {np.min(lengths):.1f} mm\n"
        f"  Max:                {np.max(lengths):.1f} mm\n"
        f"  IQR:                [{np.percentile(lengths, 25):.1f}, "
        f"{np.percentile(lengths, 75):.1f}] mm\n\n"
        f"Points per Streamline:\n"
        f"  Mean:               {np.mean(points_per_sl):.0f}\n"
        f"  Range:              [{np.min(points_per_sl)}, {np.max(points_per_sl)}]\n"
        f"{'─' * 40}"
    )
    
    ax6.text(0.5, 0.5, stats_text, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_streamline_stats.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


# =============================================================================
# TRACKING SUMMARY (SIMPLIFIED)
# =============================================================================

def create_tracking_summary(
    streamlines,
    fa,
    md,
    white_matter,
    brain_mask,
    seeds,
    affine,
    output_dir,
    stem,
    tracking_params=None,
    verbose=True
):
    """
    Create multi-panel tracking summary figure.
    """
    from dipy.tracking.streamline import length
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    n_streamlines = len(streamlines)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Tractography Summary - {stem}", fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Get slice indices
    mid_ax = fa.shape[2] // 2
    mid_cor = fa.shape[1] // 2
    mid_sag = fa.shape[0] // 2
    
    # Row 0: FA, MD, WM mask
    ax_fa = fig.add_subplot(gs[0, 0])
    ax_fa.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax_fa.set_title('FA (Axial)')
    ax_fa.axis('off')
    
    ax_md = fig.add_subplot(gs[0, 1])
    md_slice = md[:, :, mid_ax].T * 1000  # ×10⁻³ mm²/s to match the label
    md_vmax = np.percentile(md[brain_mask > 0] * 1000, 99) if brain_mask.any() else None
    md_im = ax_md.imshow(md_slice, cmap=_style.MD_CMAP, origin='lower', vmin=0, vmax=md_vmax)
    ax_md.set_title('MD (Axial, ×10⁻³)')
    ax_md.axis('off')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    _md_cax = make_axes_locatable(ax_md).append_axes('right', size='5%', pad=0.05)
    _md_cbar = fig.colorbar(md_im, cax=_md_cax)
    _md_cbar.ax.tick_params(labelsize=8)
    
    ax_wm = fig.add_subplot(gs[0, 2])
    ax_wm.imshow(fa[:, :, mid_ax].T, cmap='gray', origin='lower', vmin=0, vmax=1)
    wm_overlay = np.ma.masked_where(white_matter[:, :, mid_ax].T == 0, white_matter[:, :, mid_ax].T)
    ax_wm.imshow(wm_overlay, cmap='Blues', alpha=0.5, origin='lower')
    ax_wm.set_title('WM Mask (Axial)')
    ax_wm.axis('off')

    for _ax in (ax_fa, ax_md, ax_wm):
        _geo.finalize_image_view(_ax, affine, 'axial')

    # Parameters panel
    ax_params = fig.add_subplot(gs[0, 3])
    ax_params.axis('off')
    
    if tracking_params:
        params_text = "TRACKING PARAMETERS\n" + "─" * 25 + "\n"
        for key, val in tracking_params.items():
            params_text += f"{key}: {val}\n"
    else:
        params_text = "TRACKING PARAMETERS\n" + "─" * 25 + "\nNot specified"
    
    ax_params.text(0.5, 0.5, params_text, transform=ax_params.transAxes,
                  fontsize=10, fontfamily='monospace',
                  verticalalignment='center', horizontalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Row 1: Streamlines only (no FA background)
    if n_streamlines > 0:
        vis_sl = deterministic_subsample(streamlines, min(3000, n_streamlines))
        n_vis = len(vis_sl)
        
        # Whole-brain streamlines carry no L/R meaning here -> one neutral colour.
        _n = _style.NEUTRAL_STREAMLINE
        views_2d = [
            ('Sagittal', 1, 2, _n),
            ('Coronal', 0, 2, _n),
            ('Axial', 0, 1, _n),
        ]

        volume_bounds = _volume_world_bounds(fa.shape, affine)
        plane_limits = {}
        for title, d1, d2, _ in views_2d:
            fallback = (volume_bounds[d1], volume_bounds[d2])
            plane_limits[title] = _streamline_plane_limits(vis_sl, d1, d2, fallback)
        
        for col, (title, d1, d2, color) in enumerate(views_2d):
            ax = fig.add_subplot(gs[1, col])
            ax.set_facecolor('#f0f0f0')
            
            for sl in vis_sl:
                ax.plot(sl[:, d1], sl[:, d2], alpha=0.1, linewidth=0.3, color=color)
            
            ax.set_title(f'{title}\n({n_vis:,} shown)')
            xlim, ylim = plane_limits[title]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3, color='white')
            ax.set_box_aspect(1)
            _geo.finalize_world_plane(ax, d1)

    # Length histogram
    if n_streamlines > 0:
        lengths = np.array([length(s) for s in streamlines])
        ax_hist = fig.add_subplot(gs[1, 3])
        ax_hist.hist(lengths, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax_hist.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2)
        ax_hist.set_xlabel('Length (mm)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Length Distribution')
        ax_hist.grid(True, alpha=0.3)
    
    # Row 2: Statistics summary
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    if n_streamlines > 0:
        lengths = np.array([length(s) for s in streamlines])
        stats_text = (
            f"{'═' * 100}\n"
            f"TRACTOGRAPHY SUMMARY\n"
            f"{'═' * 100}\n\n"
            f"Total Streamlines:     {n_streamlines:,}\n"
            f"Total Seeds:           {len(seeds):,}\n"
            f"Yield Rate:            {n_streamlines/len(seeds)*100:.1f}%\n\n"
            f"Length (mm):           mean={np.mean(lengths):.1f}, "
            f"median={np.median(lengths):.1f}, "
            f"range=[{np.min(lengths):.1f}, {np.max(lengths):.1f}]\n\n"
            f"White Matter:          {white_matter.sum():,} voxels "
            f"({white_matter.sum()/brain_mask.sum()*100:.1f}% of brain)\n"
            f"FA (brain mean):       {fa[brain_mask > 0].mean():.3f}\n"
            f"{'═' * 100}"
        )
    else:
        stats_text = "No streamlines generated"
    
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                 fontsize=11, fontfamily='monospace',
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    fig_path = viz_dir / f"{stem}_tracking_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"  ✓ Saved: {fig_path}")
    
    return fig_path


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def save_all_tracking_visualizations(
    streamlines,
    fa,
    md,
    white_matter,
    brain_mask,
    seeds,
    affine,
    output_dir,
    stem,
    tenfit=None,
    fa_thresh=0.2,
    tracking_params=None,
    verbose=True
):
    """
    Generate and save all tracking visualizations.
    """
    _style.apply_house_style()  # shared typography/spines/DPI for this stage
    if verbose:
        print("  → Generating tracking visualizations...")

    viz_paths = {}

    viz_paths['tensor_maps'] = plot_tensor_maps(
        fa, md, brain_mask, output_dir, stem, tenfit, affine=affine, verbose=verbose
    )

    viz_paths['wm_mask_qc'] = plot_white_matter_mask(
        fa, white_matter, brain_mask, output_dir, stem, fa_thresh,
        affine=affine, verbose=verbose
    )

    viz_paths['streamlines_2d'] = plot_streamlines_2d(
        streamlines, fa, affine, output_dir, stem, verbose=verbose
    )

    viz_paths['streamline_stats'] = plot_streamline_statistics(
        streamlines, fa, seeds, affine, output_dir, stem, verbose=verbose
    )

    viz_paths['summary'] = create_tracking_summary(
        streamlines, fa, md, white_matter, brain_mask, seeds, affine,
        output_dir, stem, tracking_params, verbose=verbose
    )

    if verbose:
        print(f"  ✓ All visualizations saved to: {Path(output_dir) / 'visualizations'}")
    
    return viz_paths
