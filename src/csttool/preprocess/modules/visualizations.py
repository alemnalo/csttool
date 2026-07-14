"""
visualizations.py

Visualization functions for preprocessing QC.

This module provides file-saving visualizations for:
- Denoising comparison (before/after)
- Gibbs unringing comparison (before/after)
- Brain mask overlay
- Motion correction summary
- Multi-panel preprocessing summary

All functions save figures to disk and return the path to the saved file.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

from csttool.viz import geometry as _geo
from csttool.viz import style as _style
from csttool.viz.utils import viz_rng


def plot_denoising_comparison(
    data_before,
    data_after,
    brain_mask,
    output_dir,
    stem,
    denoise_method,
    vol_idx=None,
    affine=None,
    verbose=True
):
    """
    Create before/after denoising comparison figure.
    
    Shows three orthogonal views comparing original and denoised data,
    plus per-voxel absolute-difference residuals highlighting removed noise.
    
    Parameters
    ----------
    data_before : ndarray
        4D DWI data before denoising.
    data_after : ndarray
        4D DWI data after denoising.
    brain_mask : ndarray
        3D binary brain mask.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    denoise_method : str
        Denoising method used. 
    vol_idx : int, optional
        Volume index to visualize. Default picks a DWI volume (middle of 4th dim).
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
    
    # Default to a DWI volume (middle of 4th dimension)
    if vol_idx is None:
        vol_idx = data_before.shape[3] // 2
    
    # Get middle slice indices for each orientation
    mid_ax = data_before.shape[2] // 2
    mid_cor = data_before.shape[1] // 2
    mid_sag = data_before.shape[0] // 2
    
    # Extract 3D volumes
    before = data_before[..., vol_idx]
    after = data_after[..., vol_idx]
    
    # Per-voxel absolute difference (removed signal). sqrt(x**2) == |x|; there is
    # no mean here, so this is |before - after|, not an RMS.
    abs_diff = np.abs(before.astype(np.float64) - after.astype(np.float64))
    if brain_mask is not None and brain_mask.shape == before.shape:
        abs_diff[~brain_mask] = 0

    # Define orthogonal views
    views = [
        ('Axial', before[:, :, mid_ax], after[:, :, mid_ax], abs_diff[:, :, mid_ax]),
        ('Coronal', before[:, mid_cor, :], after[:, mid_cor, :], abs_diff[:, mid_cor, :]),
        ('Sagittal', before[mid_sag, :, :], after[mid_sag, :, :], abs_diff[mid_sag, :, :]),
    ]

    # Shared |difference| scale across all three views so panels are comparable.
    res_pool = abs_diff[brain_mask] if (brain_mask is not None and brain_mask.shape == before.shape) else abs_diff
    res_vmax = np.percentile(res_pool, 99) if np.any(res_pool) else None

    # Create figure: 3 rows (views) × 3 columns (original, denoised, residuals)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                              subplot_kw={'xticks': [], 'yticks': []},
                              constrained_layout=True)
    fig.suptitle(f"Denoising using {denoise_method} - {stem} (Volume {vol_idx})", fontsize=14, fontweight='bold')

    # Column titles
    axes[0, 0].set_title('Original', fontsize=12)
    axes[0, 1].set_title('Denoised', fontsize=12)
    axes[0, 2].set_title('|Difference|', fontsize=12)

    res_im = None
    for row, (view_name, orig, den, res) in enumerate(views):
        # Original
        axes[row, 0].imshow(orig.T, cmap='gray', interpolation='none', origin='lower')
        axes[row, 0].set_ylabel(view_name, fontsize=12, fontweight='bold')

        # Denoised
        axes[row, 1].imshow(den.T, cmap='gray', interpolation='none', origin='lower')

        # Residuals (shared scale + perceptually-uniform residual colormap)
        res_im = axes[row, 2].imshow(res.T, cmap=_style.RESIDUAL_CMAP, interpolation='none',
                                     origin='lower', vmin=0, vmax=res_vmax)

        # Radiological orientation + L/R markers for all three panels of this view.
        if affine is not None:
            for c in range(3):
                _geo.finalize_image_view(axes[row, c], affine, view_name.lower())

    # Labelled colorbar for the shared |difference| column.
    if res_im is not None:
        _style.add_scalar_colorbar(fig, res_im, list(axes[:, 2]), 'Removed signal |Δ|')

    fig_path = viz_dir / f"{stem}_denoising_qc.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    if verbose:
        print(f"  ✓ Denoising QC: {fig_path}")
    
    return fig_path


def plot_gibbs_unringing_comparison(
    data_before,
    data_after,
    brain_mask,
    output_dir,
    stem,
    vol_idx=None,
    affine=None,
    verbose=True
):
    """
    Create before/after Gibbs unringing comparison figure.
    
    Shows three orthogonal views comparing data before and after
    unringing, plus per-voxel absolute-difference residuals highlighting removed ringing artifacts.
    
    Parameters
    ----------
    data_before : ndarray
        4D DWI data before Gibbs unringing.
    data_after : ndarray
        4D DWI data after Gibbs unringing.
    brain_mask : ndarray
        3D binary brain mask.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    vol_idx : int, optional
        Volume index to visualize. Default picks a DWI volume (middle of 4th dim).
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
    
    # Default to a DWI volume (middle of 4th dimension)
    if vol_idx is None:
        vol_idx = data_before.shape[3] // 2
    
    # Get middle slice indices for each orientation
    mid_ax = data_before.shape[2] // 2
    mid_cor = data_before.shape[1] // 2
    mid_sag = data_before.shape[0] // 2
    
    # Extract 3D volumes
    before = data_before[..., vol_idx]
    after = data_after[..., vol_idx]
    
    # Per-voxel absolute difference (removed ringing). sqrt(x**2) == |x|; there is
    # no mean here, so this is |before - after|, not an RMS.
    abs_diff = np.abs(before.astype(np.float64) - after.astype(np.float64))
    if brain_mask is not None and brain_mask.shape == before.shape:
        abs_diff[~brain_mask] = 0

    # Define orthogonal views
    views = [
        ('Axial', before[:, :, mid_ax], after[:, :, mid_ax], abs_diff[:, :, mid_ax]),
        ('Coronal', before[:, mid_cor, :], after[:, mid_cor, :], abs_diff[:, mid_cor, :]),
        ('Sagittal', before[mid_sag, :, :], after[mid_sag, :, :], abs_diff[mid_sag, :, :]),
    ]

    # Shared |difference| scale across all three views so panels are comparable.
    res_pool = abs_diff[brain_mask] if (brain_mask is not None and brain_mask.shape == before.shape) else abs_diff
    res_vmax = np.percentile(res_pool, 99) if np.any(res_pool) else None

    # Create figure: 3 rows (views) × 3 columns (before, after, residuals)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                              subplot_kw={'xticks': [], 'yticks': []},
                              constrained_layout=True)
    fig.suptitle(f"Gibbs Unringing - {stem} (Volume {vol_idx})", fontsize=14, fontweight='bold')

    # Column titles
    axes[0, 0].set_title('Before', fontsize=12)
    axes[0, 1].set_title('After', fontsize=12)
    axes[0, 2].set_title('|Difference|', fontsize=12)

    res_im = None
    for row, (view_name, bef, aft, res) in enumerate(views):
        # Before
        axes[row, 0].imshow(bef.T, cmap='gray', interpolation='none', origin='lower')
        axes[row, 0].set_ylabel(view_name, fontsize=12, fontweight='bold')

        # After
        axes[row, 1].imshow(aft.T, cmap='gray', interpolation='none', origin='lower')

        # Residuals (shared scale + perceptually-uniform residual colormap)
        res_im = axes[row, 2].imshow(res.T, cmap=_style.RESIDUAL_CMAP, interpolation='none',
                                     origin='lower', vmin=0, vmax=res_vmax)

        # Radiological orientation + L/R markers for all three panels of this view.
        if affine is not None:
            for c in range(3):
                _geo.finalize_image_view(axes[row, c], affine, view_name.lower())

    # Labelled colorbar for the shared |difference| column.
    if res_im is not None:
        _style.add_scalar_colorbar(fig, res_im, list(axes[:, 2]), 'Removed signal |Δ|')

    fig_path = viz_dir / f"{stem}_gibbs_unringing_qc.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    if verbose:
        print(f"  ✓ Gibbs unringing QC: {fig_path}")
    
    return fig_path


def plot_brain_mask_overlay(
    data,
    brain_mask,
    gtab,
    output_dir,
    stem,
    affine=None,
    verbose=True
):
    """
    Create brain mask overlay visualization.
    
    Shows brain mask overlaid on b0 image in three orthogonal views,
    plus mask coverage statistics.
    
    Parameters
    ----------
    data : ndarray
        4D DWI data (masked or unmasked).
    brain_mask : ndarray
        3D binary brain mask.
    gtab : GradientTable
        Gradient table to identify b0 volumes.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
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
    
    # Get b0 volume
    b0_idx = np.where(gtab.bvals < 50)[0]
    if len(b0_idx) == 0:
        b0_idx = [0]
    b0 = data[..., b0_idx[0]]
    
    # Get slice indices
    mid_ax = data.shape[2] // 2
    mid_cor = data.shape[1] // 2
    mid_sag = data.shape[0] // 2
    
    # Compute statistics
    total_voxels = brain_mask.size
    brain_voxels = brain_mask.sum()
    coverage = brain_voxels / total_voxels * 100
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Brain Mask QC - {stem}\n"
                 f"Coverage: {brain_voxels:,} voxels ({coverage:.1f}%)",
                 fontsize=14, fontweight='bold')
    
    vmax = np.percentile(b0[brain_mask], 99) if brain_mask.any() else np.percentile(b0, 99)
    
    views = [
        ('Axial', b0[:, :, mid_ax], brain_mask[:, :, mid_ax], f'z={mid_ax}'),
        ('Coronal', b0[:, mid_cor, :], brain_mask[:, mid_cor, :], f'y={mid_cor}'),
        ('Sagittal', b0[mid_sag, :, :], brain_mask[mid_sag, :, :], f'x={mid_sag}'),
    ]
    
    for col, (view_name, b0_slice, mask_slice, coord) in enumerate(views):
        # Row 0: b0 only
        axes[0, col].imshow(b0_slice.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
        axes[0, col].set_title(f'{view_name} ({coord})\nb0 image')
        axes[0, col].axis('off')
        
        # Row 1: b0 with mask overlay
        axes[1, col].imshow(b0_slice.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
        
        # Create masked array for overlay
        mask_overlay = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
        axes[1, col].imshow(mask_overlay, cmap='Reds', alpha=0.4, origin='lower')
        
        # Add contour
        axes[1, col].contour(mask_slice.T, levels=[0.5], colors='red', linewidths=1)
        axes[1, col].set_title(f'{view_name}\nwith brain mask')
        axes[1, col].axis('off')

        # Radiological orientation + L/R markers for both rows of this view.
        if affine is not None:
            _geo.finalize_image_view(axes[0, col], affine, view_name.lower())
            _geo.finalize_image_view(axes[1, col], affine, view_name.lower())

    fig_path = viz_dir / f"{stem}_brain_mask_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"  ✓ Brain mask QC: {fig_path}")
    
    return fig_path


def plot_motion_correction_summary(
    reg_affines,
    output_dir,
    stem,
    verbose=True
):
    """
    Create motion correction summary visualization.
    
    Shows translation and rotation parameters across volumes,
    highlighting any large motion events.
    
    Parameters
    ----------
    reg_affines : list of ndarray
        List of 4x4 registration affine matrices (one per volume).
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
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
    
    n_vols = len(reg_affines)
    
    # Extract translation and rotation parameters
    translations = np.zeros((n_vols, 3))
    rotations = np.zeros((n_vols, 3))
    
    for i, affine in enumerate(reg_affines):
        # Translation is in the last column
        translations[i] = affine[:3, 3]
        
        # Approximate rotation angles from rotation matrix
        # Using small angle approximation for simplicity
        R = affine[:3, :3]
        rotations[i, 0] = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi  # Roll (x)
        rotations[i, 1] = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * 180 / np.pi  # Pitch (y)
        rotations[i, 2] = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi  # Yaw (z)
    
    # Compute relative motion (difference from first volume)
    translations_rel = translations - translations[0]
    rotations_rel = rotations - rotations[0]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Motion Correction QC - {stem}\n{n_vols} volumes",
                 fontsize=14, fontweight='bold')
    
    volumes = np.arange(n_vols)
    
    # Translation (absolute)
    ax = axes[0, 0]
    ax.plot(volumes, translations[:, 0], 'r-', label='X', linewidth=1.5)
    ax.plot(volumes, translations[:, 1], 'g-', label='Y', linewidth=1.5)
    ax.plot(volumes, translations[:, 2], 'b-', label='Z', linewidth=1.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Translation (mm)')
    ax.set_title('Absolute Translation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Translation (relative to first)
    ax = axes[0, 1]
    ax.plot(volumes, translations_rel[:, 0], 'r-', label='X', linewidth=1.5)
    ax.plot(volumes, translations_rel[:, 1], 'g-', label='Y', linewidth=1.5)
    ax.plot(volumes, translations_rel[:, 2], 'b-', label='Z', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Translation (mm)')
    ax.set_title('Relative Translation (vs. volume 0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Rotation (absolute)
    ax = axes[1, 0]
    ax.plot(volumes, rotations[:, 0], 'r-', label='Roll (X)', linewidth=1.5)
    ax.plot(volumes, rotations[:, 1], 'g-', label='Pitch (Y)', linewidth=1.5)
    ax.plot(volumes, rotations[:, 2], 'b-', label='Yaw (Z)', linewidth=1.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Rotation (degrees)')
    ax.set_title('Absolute Rotation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Rotation (relative to first)
    ax = axes[1, 1]
    ax.plot(volumes, rotations_rel[:, 0], 'r-', label='Roll (X)', linewidth=1.5)
    ax.plot(volumes, rotations_rel[:, 1], 'g-', label='Pitch (Y)', linewidth=1.5)
    ax.plot(volumes, rotations_rel[:, 2], 'b-', label='Yaw (Z)', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Rotation (degrees)')
    ax.set_title('Relative Rotation (vs. volume 0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    max_trans = np.max(np.abs(translations_rel))
    max_rot = np.max(np.abs(rotations_rel))
    
    fig.text(0.5, 0.02, 
             f"Max displacement: {max_trans:.2f} mm | Max rotation: {max_rot:.2f}°",
             ha='center', fontsize=11, style='italic')
    
    fig_path = viz_dir / f"{stem}_motion_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"  ✓ Motion correction QC: {fig_path}")
    
    return fig_path


def create_preprocessing_summary(
    data_original,
    data_preprocessed,
    brain_mask,
    gtab,
    output_dir,
    stem,
    motion_correction_applied=False,
    affine=None,
    verbose=True
):
    """
    Create multi-panel preprocessing summary figure.
    
    Combines key QC visualizations into a single summary figure
    for quick assessment of preprocessing quality.
    
    Parameters
    ----------
    data_original : ndarray
        4D DWI data before preprocessing.
    data_preprocessed : ndarray
        4D DWI data after preprocessing.
    brain_mask : ndarray
        3D binary brain mask.
    gtab : GradientTable
        Gradient table.
    output_dir : str or Path
        Output directory for saving figure.
    stem : str
        Subject/scan identifier for filename.
    motion_correction_applied : bool, optional
        Whether motion correction was applied.
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
    
    # Check if shapes are compatible
    orig_shape = data_original.shape[:3]
    proc_shape = data_preprocessed.shape[:3]
    mask_shape = brain_mask.shape
    
    # A true before/after in one slice grid needs matching shapes. Reslicing or
    # cropping changes the preprocessed grid, so the raw "original" is no longer
    # co-registered with it. Rather than silently substituting preprocessed data
    # while still labelling a panel "Original" (a misleading figure), record the
    # mismatch and annotate the affected panels honestly below.
    shapes_match = (orig_shape == proc_shape and orig_shape == mask_shape)
    if not shapes_match:
        data_original = data_preprocessed
    
    # Get b0 volume index
    b0_idx = np.where(gtab.bvals < 50)[0]
    if len(b0_idx) == 0:
        b0_idx = [0]
    vol_idx = b0_idx[0]
    
    # Get slice indices
    mid_ax = data_original.shape[2] // 2
    
    # Extract data
    orig_b0 = data_original[:, :, mid_ax, vol_idx]
    proc_b0 = data_preprocessed[:, :, mid_ax, vol_idx]
    mask_slice = brain_mask[:, :, mid_ax]
    
    # Compute difference
    diff = np.abs(proc_b0.astype(np.float64) - orig_b0.astype(np.float64))
    diff[~mask_slice] = 0
    
    # Compute statistics
    brain_voxels = brain_mask.sum()
    coverage = brain_voxels / brain_mask.size * 100
    
    # Intensity statistics
    orig_mean = np.mean(data_original[brain_mask])
    proc_mean = np.mean(data_preprocessed[brain_mask])
    orig_std = np.std(data_original[brain_mask])
    proc_std = np.std(data_preprocessed[brain_mask])
    
    # Create figure with constrained_layout
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    fig.suptitle(f"Preprocessing Summary - {stem}", fontsize=16, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    vmax = np.percentile(orig_b0[mask_slice], 99) if mask_slice.any() else np.percentile(orig_b0, 99)
    diff_vmax = np.percentile(diff[mask_slice], 99) if diff[mask_slice].any() else 1
    
    # Row 0: Original, Preprocessed, Difference, Mask
    orig_title = 'Original b0' if shapes_match else 'Preprocessed b0\n(original grid differs)'
    diff_title = 'Difference (denoising)' if shapes_match else 'Difference N/A\n(shape changed)'

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_b0.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
    ax1.set_title(orig_title)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(proc_b0.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
    ax2.set_title('Preprocessed b0')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    diff_im = ax3.imshow(diff.T, cmap=_style.RESIDUAL_CMAP, origin='lower', vmin=0, vmax=diff_vmax)
    ax3.set_title(diff_title)
    ax3.axis('off')
    if shapes_match:
        # constrained_layout-compatible colorbar (no make_axes_locatable here).
        fig.colorbar(diff_im, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(proc_b0.T, cmap='gray', origin='lower', vmin=0, vmax=vmax)
    ax4.contour(mask_slice.T, levels=[0.5], colors='red', linewidths=1.5)
    ax4.set_title(f'Brain Mask\n({brain_voxels:,} voxels)')
    ax4.axis('off')

    # Row 0 panels are all axial: radiological orientation + L/R markers.
    if affine is not None:
        for _ax in (ax1, ax2, ax3, ax4):
            _geo.finalize_image_view(_ax, affine, 'axial')
    
    # Row 1: Three orthogonal views with mask overlay
    views_data = [
        ('Sagittal', data_preprocessed[data_preprocessed.shape[0]//2, :, :, vol_idx],
         brain_mask[brain_mask.shape[0]//2, :, :]),
        ('Coronal', data_preprocessed[:, data_preprocessed.shape[1]//2, :, vol_idx],
         brain_mask[:, brain_mask.shape[1]//2, :]),
        ('Axial', data_preprocessed[:, :, data_preprocessed.shape[2]//2, vol_idx],
         brain_mask[:, :, brain_mask.shape[2]//2]),
    ]
    
    for i, (name, img, msk) in enumerate(views_data):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img.T, cmap='gray', origin='lower')
        ax.contour(msk.T, levels=[0.5], colors='cyan', linewidths=1)
        ax.set_title(name)
        ax.axis('off')
        if affine is not None:
            _geo.finalize_image_view(ax, affine, name.lower())
    
    # Row 1, col 3: Histogram comparison
    ax_hist = fig.add_subplot(gs[1, 3])
    
    orig_vals = data_original[brain_mask].flatten()
    proc_vals = data_preprocessed[brain_mask].flatten()
    
    # Subsample for efficiency (deterministic local RNG; no global-state mutation)
    if len(orig_vals) > 100000:
        idx = np.sort(viz_rng().choice(len(orig_vals), 100000, replace=False))
        orig_vals = orig_vals[idx]
        proc_vals = proc_vals[idx]
    
    ax_hist.hist(orig_vals, bins=100, alpha=0.5, label='Original', density=True)
    ax_hist.hist(proc_vals, bins=100, alpha=0.5, label='Preprocessed', density=True)
    ax_hist.set_xlabel('Intensity')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Intensity Distribution')
    ax_hist.legend(fontsize=8)
    
    # Row 2: Statistics panel
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    mc_status = "Applied" if motion_correction_applied else "Not applied"
    bvalues = sorted({int(b) for b in gtab.bvals})  # plain ints (no numpy repr leak)
    grid_note = (
        "" if shapes_match
        else "\nNOTE: original grid differs from preprocessed (reslice/crop);\n"
             "      'Original' and 'Difference' panels are not directly comparable.\n"
    )

    stats_text = (
        f"{'─' * 80}\n"
        f"PREPROCESSING STATISTICS\n"
        f"{'─' * 80}\n\n"
        f"Data Shape:           {data_original.shape}\n"
        f"Voxel Dimensions:     {data_original.shape[:3]}\n"
        f"Number of Volumes:    {data_original.shape[3]}\n"
        f"B-values:             {bvalues}\n\n"
        f"Brain Mask Coverage:  {brain_voxels:,} voxels ({coverage:.1f}%)\n\n"
        f"Intensity (in brain):\n"
        f"  Original:           mean = {orig_mean:.1f}, std = {orig_std:.1f}\n"
        f"  Preprocessed:       mean = {proc_mean:.1f}, std = {proc_std:.1f}\n\n"
        f"Motion Correction:    {mc_status}\n"
        f"{grid_note}"
        f"{'─' * 80}"
    )
    
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, fontfamily='monospace',
                  verticalalignment='center', horizontalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig_path = viz_dir / f"{stem}_preprocessing_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"  ✓ Preprocessing summary: {fig_path}")
    
    return fig_path


def save_all_preprocessing_visualizations(
    data_original,
    data_denoised,
    data_masked,
    data_unringed,
    data_preprocessed,
    brain_mask,
    gtab,
    output_dir,
    stem,
    denoise_method,
    reg_affines=None,
    motion_correction_applied=False,
    affine=None,
    verbose=True
):
    """
    Generate and save all preprocessing visualizations.
    
    Convenience function that calls all visualization functions
    and returns paths to all generated figures.
    
    Parameters
    ----------
    data_original : ndarray
        4D DWI data before any preprocessing.
    data_denoised : ndarray
        4D DWI data after denoising (before masking).
    data_masked : ndarray
        4D DWI data after brain masking (cropped).
    data_unringed : ndarray
        4D DWI data after Gibbs unringing (cropped).
    data_preprocessed : ndarray
        4D DWI data after full preprocessing.
    brain_mask : ndarray
        3D binary brain mask.
    gtab : GradientTable
        Gradient table.
    output_dir : str or Path
        Output directory for saving figures.
    stem : str
        Subject/scan identifier for filenames.
    denoise_method : str, optional
        Denoising method used.
    reg_affines : list of ndarray, optional
        Registration affines from motion correction.
    motion_correction_applied : bool, optional
        Whether motion correction was applied.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    viz_paths : dict
        Dictionary mapping visualization names to file paths.
    """
    _style.apply_house_style()  # shared typography/spines/DPI for this stage
    if verbose:
        print("\nGenerating preprocessing visualizations...")

    viz_paths = {}
    
    # Denoising comparison
    if data_denoised is not None:
        viz_paths['denoising_qc'] = plot_denoising_comparison(
            data_original, data_denoised, brain_mask,
            output_dir, stem, denoise_method, affine=affine, verbose=verbose
        )

    # Gibbs unringing comparison (both inputs are cropped/masked)
    if data_unringed is not None and data_masked is not None:
        viz_paths['gibbs_unringing_qc'] = plot_gibbs_unringing_comparison(
            data_masked, data_unringed, brain_mask,
            output_dir, stem, affine=affine, verbose=verbose
        )

    # Brain mask overlay
    viz_paths['brain_mask_qc'] = plot_brain_mask_overlay(
        data_preprocessed, brain_mask, gtab,
        output_dir, stem, affine=affine, verbose=verbose
    )

    # Motion correction (if applied)
    if motion_correction_applied and reg_affines is not None:
        viz_paths['motion_qc'] = plot_motion_correction_summary(
            reg_affines, output_dir, stem, verbose=verbose
        )

    # Summary figure
    viz_paths['summary'] = create_preprocessing_summary(
        data_original, data_preprocessed, brain_mask, gtab,
        output_dir, stem, motion_correction_applied, affine=affine, verbose=verbose
    )
    
    if verbose:
        print(f"  ✓ All preprocessing visualizations saved to: {Path(output_dir) / 'visualizations'}")
    
    return viz_paths
