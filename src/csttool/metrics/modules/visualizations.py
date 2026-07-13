"""
visualizations.py

Visualization functions for CST metrics analysis.

This module provides:
- Tract profile plots (FA/MD along the tract)
- Bilateral comparison bar charts
- 3D streamline visualizations with scalar coloring
- Multi-panel summary figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from csttool.viz import geometry as _geo


def plot_tract_profiles(
    left_metrics,
    right_metrics,
    output_dir,
    subject_id,
    scalar='fa',
    anatomical_labels=True
):
    """
    Plot along-tract profiles for bilateral comparison.
    
    Creates a figure showing FA or MD profiles along normalized tract length
    for both left and right CST.
    
    Parameters
    ----------
    left_metrics : dict
        Left hemisphere metrics with 'fa' or 'md' profile
    right_metrics : dict
        Right hemisphere metrics with 'fa' or 'md' profile
    output_dir : str or Path
        Output directory for saving figure
    subject_id : str
        Subject identifier for filename
    scalar : str
        'fa' or 'md' - which scalar to plot
    anatomical_labels : bool
        If True, add anatomical labels to x-axis (Pontine Level, PLIC, Precentral Gyrus)
        
    Returns
    -------
    fig_path : Path
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get profiles
    if scalar not in left_metrics or scalar not in right_metrics:
        print(f"Warning: {scalar.upper()} not available in metrics")
        return None
    
    left_profile = np.array(left_metrics[scalar]['profile'])
    right_profile = np.array(right_metrics[scalar]['profile'])
    
    n_points = len(left_profile)
    x = np.linspace(0, 100, n_points)  # Normalized position (0-100%)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot profiles
    ax.plot(x, left_profile, 'b-', linewidth=2, label='Left CST', marker='o', markersize=4)
    ax.plot(x, right_profile, 'r-', linewidth=2, label='Right CST', marker='s', markersize=4)
    
    # Add mean lines
    left_mean = left_metrics[scalar]['mean']
    right_mean = right_metrics[scalar]['mean']
    ax.axhline(left_mean, color='b', linestyle='--', alpha=0.5, label=f'Left mean: {left_mean:.3f}')
    ax.axhline(right_mean, color='r', linestyle='--', alpha=0.5, label=f'Right mean: {right_mean:.3f}')
    
    # Labels and formatting
    scalar_label = 'Fractional Anisotropy' if scalar == 'fa' else 'Mean Diffusivity (×10⁻³ mm²/s)'
    
    # Set x-axis with anatomical labels
    if anatomical_labels:
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(['0%', '50%', '100%'])
        # Add anatomical labels as secondary text
        ax.text(0, -0.12, 'Pontine Level', transform=ax.get_xaxis_transform(), 
                ha='center', fontsize=9, style='italic')
        ax.text(50, -0.12, 'PLIC', transform=ax.get_xaxis_transform(), 
                ha='center', fontsize=9, style='italic')
        ax.text(100, -0.12, 'Precentral Gyrus', transform=ax.get_xaxis_transform(), 
                ha='center', fontsize=9, style='italic')
        ax.set_xlabel('Normalized Tract Position', fontsize=12)
    else:
        ax.set_xlabel('Normalized Tract Position (%)', fontsize=12)
    
    ax.set_ylabel(scalar_label, fontsize=12)
    ax.set_title(f'{scalar_label.split(" (")[0]} Profile - {subject_id}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Extra space for anatomical labels
    
    # Save figure
    fig_path = output_dir / f"{subject_id}_tract_profile_{scalar}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Tract profile saved: {fig_path}")
    return fig_path


def plot_stacked_profiles(
    left_metrics,
    right_metrics,
    output_dir,
    subject_id
):
    """
    Create stacked FA, MD, RD, and AD profile plots for PDF report.
    
    Creates a vertically stacked figure with 4 subplots (if data available):
    - FA profile
    - MD profile
    - RD profile
    - AD profile
    
    All have shared x-axis (anatomical labels only on bottom).
    
    Parameters
    ----------
    left_metrics : dict
        Left hemisphere metrics
    right_metrics : dict
        Right hemisphere metrics
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    fig_path : Path
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define metrics to plot in order
    metrics_config = [
        {'key': 'fa', 'title': 'Fractional Anisotropy', 'ylabel': 'FA', 'ylim': (0, 0.8), 'scale': 1},
        {'key': 'md', 'title': 'Mean Diffusivity', 'ylabel': 'MD (×10⁻³)', 'ylim': (0.5, 1.2), 'scale': 1000},
        {'key': 'rd', 'title': 'Radial Diffusivity', 'ylabel': 'RD (×10⁻³)', 'ylim': (0.3, 1.0), 'scale': 1000},
        {'key': 'ad', 'title': 'Axial Diffusivity', 'ylabel': 'AD (×10⁻³)', 'ylim': (0.8, 1.8), 'scale': 1000}
    ]
    
    # Filter available metrics
    available_metrics = []
    for m in metrics_config:
        if m['key'] in left_metrics and m['key'] in right_metrics:
            available_metrics.append(m)
    
    if not available_metrics:
        print("Warning: No profiles available for stacking")
        return None
    
    n_plots = len(available_metrics)
    # Fixed height per plot
    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 2.2 * n_plots), sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (ax, m) in enumerate(zip(axes, available_metrics)):
        key = m['key']
        scale = m['scale']
        
        left_profile = np.array(left_metrics[key]['profile']) * scale
        right_profile = np.array(right_metrics[key]['profile']) * scale
        n_points = len(left_profile)
        x = np.linspace(0, 100, n_points)
        
        ax.plot(x, left_profile, 'b-', linewidth=2, label='Left CST', marker='o', markersize=3)
        ax.plot(x, right_profile, 'r-', linewidth=2, label='Right CST', marker='s', markersize=3)
        
        ax.set_ylabel(m['ylabel'], fontsize=10)
        # Auto stats for ylim might be better, but keeping fixed range as starting point logic
        # If 'ylim' is provided, use it, else auto
        if 'ylim' in m:
             # Basic check to see if data fits in default range, if not, auto-scale
             all_data = np.concatenate([left_profile, right_profile])
             if np.min(all_data) < m['ylim'][0] or np.max(all_data) > m['ylim'][1]:
                 # Auto scale with margin
                 margin = (np.max(all_data) - np.min(all_data)) * 0.1
                 ax.set_ylim(max(0, np.min(all_data) - margin), np.max(all_data) + margin)
             else:
                 ax.set_ylim(m['ylim'])
                 
        ax.text(0.5, 0.9, m['title'], transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center')
        
        ax.grid(True, alpha=0.3)
        
        # Only add legend to first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Shared X-axis formatting (only on bottom plot)
    axes[-1].set_xticks([0, 50, 100])
    axes[-1].set_xticklabels(['0%', '50%', '100%'])
    
    # Add anatomical annotations
    # Create transform for the bottom axis
    trans = axes[-1].get_xaxis_transform()
    
    axes[-1].text(0, -0.3, 'Pontine\nLevel', transform=trans, 
            ha='center', fontsize=9, style='italic')
    axes[-1].text(50, -0.3, 'PLIC', transform=trans, 
            ha='center', fontsize=9, style='italic')
    axes[-1].text(100, -0.3, 'Precentral\nGyrus', transform=trans, 
            ha='center', fontsize=9, style='italic')
            
    plt.tight_layout()
    # Increase bottom margin to prevent x-axis overlap
    plt.subplots_adjust(hspace=0.2, bottom=0.15)  # Minimize vertical space between plots
    
    fig_path = output_dir / f"{subject_id}_stacked_profiles.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Stacked profiles saved: {fig_path}")
    return fig_path


def plot_tractogram_qc_preview(
    streamlines_left,
    streamlines_right,
    background_image,
    affine,
    output_dir,
    subject_id,
    slice_type='axial',
    max_streamlines=500,
    set_title=True
):
    """
    Create compact 3D tractogram QC preview for PDF report.
    
    Renders left (blue) and right (red) CST streamlines overlaid
    on a brain slice at the internal capsule level.
    
    Parameters
    ----------
    streamlines_left : Streamlines
        Left CST streamlines
    streamlines_right : Streamlines
        Right CST streamlines
    background_image : ndarray
        3D T1 or FA image for background
    affine : ndarray
        4x4 affine transformation matrix
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
    slice_type : str
        'axial', 'sagittal', or 'coronal'
    max_streamlines : int
        Maximum streamlines to render per hemisphere (for performance)
        
    Returns
    -------
    fig_path : Path
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Get slice at center (internal capsule level)
    shape = background_image.shape
    
    if slice_type == 'axial':
        slice_idx = shape[2] // 2 + 5  # Slightly above center for IC
        bg_slice = background_image[:, :, slice_idx].T
    elif slice_type == 'sagittal':
        slice_idx = shape[0] // 2
        bg_slice = background_image[slice_idx, :, :].T
    else:  # coronal
        slice_idx = shape[1] // 2
        bg_slice = background_image[:, slice_idx, :].T
    
    # Display background
    ax.imshow(bg_slice, cmap='gray', origin='lower', aspect='equal')

    # Adjust figure size to match data aspect ratio
    # logical_height / logical_width
    data_ratio = bg_slice.shape[0] / bg_slice.shape[1]
    
    # Set fixed width of 4 inches and adjust height
    fig.set_size_inches(4, 4 * data_ratio)
    
    # Project streamlines onto slice
    def project_streamlines(streamlines, color, alpha=0.6):
        """Project streamlines onto 2D and plot."""
        count = 0
        for sl in streamlines:
            if count >= max_streamlines:
                break
            # Convert to voxel coordinates
            inv_affine = np.linalg.inv(affine)
            voxel_coords = np.dot(sl, inv_affine[:3, :3].T) + inv_affine[:3, 3]
            
            if slice_type == 'axial':
                # Filter points near the slice
                near_slice = np.abs(voxel_coords[:, 2] - slice_idx) < 5
                if np.any(near_slice):
                    ax.plot(voxel_coords[near_slice, 0], 
                           voxel_coords[near_slice, 1], 
                           color=color, linewidth=0.5, alpha=alpha)
                    count += 1
            elif slice_type == 'sagittal':
                near_slice = np.abs(voxel_coords[:, 0] - slice_idx) < 5
                if np.any(near_slice):
                    ax.plot(voxel_coords[near_slice, 1], 
                           voxel_coords[near_slice, 2], 
                           color=color, linewidth=0.5, alpha=alpha)
                    count += 1
            else:  # coronal
                near_slice = np.abs(voxel_coords[:, 1] - slice_idx) < 5
                if np.any(near_slice):
                    ax.plot(voxel_coords[near_slice, 0], 
                           voxel_coords[near_slice, 2], 
                           color=color, linewidth=0.5, alpha=alpha)
                    count += 1
    
    # Plot streamlines
    if len(streamlines_left) > 0:
        project_streamlines(streamlines_left, '#2196F3')  # Blue
    if len(streamlines_right) > 0:
        project_streamlines(streamlines_right, '#F44336')  # Red
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2196F3', linewidth=2, label='Left CST'),
        Line2D([0], [0], color='#F44336', linewidth=2, label='Right CST')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    if set_title:
        ax.set_title(f'CST Tractogram ({slice_type.title()})', fontsize=10, fontweight='bold')

    ax.axis('off')
    # Enforce radiological orientation + L/R markers (axial/coronal carry L/R).
    _geo.finalize_image_view(ax, affine, slice_type)
    
    plt.tight_layout()
    
    fig_path = output_dir / f"{subject_id}_tractogram_qc_{slice_type}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Tractogram QC preview saved: {fig_path}")
    return fig_path


def plot_bilateral_comparison(
    comparison,
    output_dir,
    subject_id
):
    """
    Create bar charts comparing left vs right CST metrics.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_bilateral_cst()
    output_dir : str or Path
        Output directory for saving figure
    subject_id : str
        Subject identifier
        
    Returns
    -------
    fig_path : Path
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    # Prepare data for plotting
    metrics_to_plot = []
    
    # Morphology
    metrics_to_plot.append({
        'name': 'Streamline\nCount',
        'left': left['morphology']['n_streamlines'],
        'right': right['morphology']['n_streamlines'],
        'unit': '',
        'li': asym['streamline_count']['laterality_index']
    })
    
    metrics_to_plot.append({
        'name': 'Tract Volume\n(mm³)',
        'left': left['morphology']['tract_volume'],
        'right': right['morphology']['tract_volume'],
        'unit': 'mm³',
        'li': asym['volume']['laterality_index']
    })
    
    metrics_to_plot.append({
        'name': 'Mean Length\n(mm)',
        'left': left['morphology']['mean_length'],
        'right': right['morphology']['mean_length'],
        'unit': 'mm',
        'li': asym['mean_length']['laterality_index']
    })
    
    # Microstructure
    if 'fa' in left:
        metrics_to_plot.append({
            'name': 'Mean FA',
            'left': left['fa']['mean'],
            'right': right['fa']['mean'],
            'unit': '',
            'li': asym['fa']['laterality_index']
        })
    
    if 'md' in left:
        metrics_to_plot.append({
            'name': 'Mean MD\n(×10⁻³)',
            'left': left['md']['mean'] * 1000,  # Convert to 10^-3
            'right': right['md']['mean'] * 1000,
            'unit': '×10⁻³',
            'li': asym['md']['laterality_index']
        })
    
    # Create figure with subplots
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for ax, metric in zip(axes, metrics_to_plot):
        x_pos = [0, 1]
        values = [metric['left'], metric['right']]
        colors = ['#2196F3', '#F44336']  # Blue for left, red for right
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if 'FA' in metric['name']:
                label_text = f'{val:.3f}'
            elif 'MD' in metric['name']:
                label_text = f'{val:.2f}'
            else:
                label_text = f'{val:.0f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_text,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add laterality index
        li_text = f'LI = {metric["li"]:+.3f}'
        ax.text(0.5, ax.get_ylim()[1]*0.9, li_text,
               ha='center', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Left', 'Right'], fontsize=11)
        ax.set_ylabel(metric['unit'], fontsize=10)
        ax.set_title(metric['name'], fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle(f'Bilateral CST Comparison - {subject_id}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / f"{subject_id}_bilateral_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Bilateral comparison saved: {fig_path}")
    return fig_path


def plot_3d_streamlines(
    streamlines_left,
    streamlines_right,
    fa_map,
    affine,
    output_dir,
    subject_id
):
    """
    Create 3D visualization of CST streamlines colored by FA.
    
    Parameters
    ----------
    streamlines_left : Streamlines
        Left CST streamlines
    streamlines_right : Streamlines
        Right CST streamlines
    fa_map : ndarray
        3D FA map for coloring
    affine : ndarray
        4x4 affine transformation matrix
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    fig_path : Path
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from dipy.viz import window, actor
        
        # Create renderer
        renderer = window.Renderer()
        renderer.SetBackground(1, 1, 1)  # White background
        
        # Add left CST (blue tones)
        if len(streamlines_left) > 0:
            renderer.add(actor.line(
                streamlines_left,
                colors=(0.2, 0.4, 0.8),
                linewidth=2,
                opacity=0.8
            ))
        
        # Add right CST (red tones)
        if len(streamlines_right) > 0:
            renderer.add(actor.line(
                streamlines_right,
                colors=(0.8, 0.2, 0.2),
                linewidth=2,
                opacity=0.8
            ))
        
        # Set camera
        renderer.set_camera(position=(200, 200, 200), focal_point=(0, 0, 0))
        
        # Save figure
        fig_path = output_dir / f"{subject_id}_3d_streamlines.png"
        window.record(renderer, out_path=str(fig_path), size=(800, 800))
        
        print(f"  ✓ 3D streamlines saved: {fig_path}")
        return fig_path
        
    except Exception as e:
        print(f"  ⚠️ 3D visualization failed: {e}")
        return None


def create_summary_figure(
    comparison,
    streamlines_left,
    streamlines_right,
    fa_map,
    affine,
    output_dir,
    subject_id
):
    """
    Create multi-panel summary figure with all key visualizations.
    
    Parameters
    ----------
    comparison : dict
        Bilateral comparison metrics
    streamlines_left : Streamlines
        Left CST streamlines
    streamlines_right : Streamlines
        Right CST streamlines
    fa_map : ndarray
        3D FA map
    affine : ndarray
        4x4 affine transformation matrix
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    fig_path : Path
        Path to saved summary figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    left = comparison['left']
    right = comparison['right']
    asym = comparison['asymmetry']
    
    # Panel 1: FA Tract Profile
    ax1 = fig.add_subplot(gs[0, :2])
    if 'fa' in left:
        left_profile = np.array(left['fa']['profile'])
        right_profile = np.array(right['fa']['profile'])
        x = np.linspace(0, 100, len(left_profile))
        
        ax1.plot(x, left_profile, 'b-', linewidth=2, label='Left', marker='o', markersize=3)
        ax1.plot(x, right_profile, 'r-', linewidth=2, label='Right', marker='s', markersize=3)
        ax1.set_xlabel('Normalized Position (%)')
        ax1.set_ylabel('Fractional Anisotropy')
        ax1.set_title('FA Tract Profile', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Panel 2: Key Metrics Table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    table_data = [
        ['Metric', 'Left', 'Right', 'LI'],
        ['Streamlines', f"{left['morphology']['n_streamlines']}", 
         f"{right['morphology']['n_streamlines']}", 
         f"{asym['streamline_count']['laterality_index']:+.2f}"],
        ['Volume (mm³)', f"{left['morphology']['tract_volume']:.0f}", 
         f"{right['morphology']['tract_volume']:.0f}", 
         f"{asym['volume']['laterality_index']:+.2f}"]
    ]
    
    if 'fa' in left:
        table_data.append(['Mean FA', f"{left['fa']['mean']:.3f}", 
                          f"{right['fa']['mean']:.3f}", 
                          f"{asym['fa']['laterality_index']:+.2f}"])
    
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('Summary Metrics', fontweight='bold', pad=20)
    
    # Panel 3: Streamline Count Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(['Left', 'Right'], 
           [left['morphology']['n_streamlines'], right['morphology']['n_streamlines']],
           color=['#2196F3', '#F44336'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count')
    ax3.set_title('Streamline Count', fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Panel 4: Volume Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(['Left', 'Right'],
           [left['morphology']['tract_volume'], right['morphology']['tract_volume']],
           color=['#2196F3', '#F44336'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Volume (mm³)')
    ax4.set_title('Tract Volume', fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Panel 5: FA Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    if 'fa' in left:
        ax5.bar(['Left', 'Right'],
               [left['fa']['mean'], right['fa']['mean']],
               color=['#2196F3', '#F44336'], alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Mean FA')
        ax5.set_title('Fractional Anisotropy', fontweight='bold')
        ax5.grid(True, axis='y', alpha=0.3)
    
    # Panel 6: Laterality Indices
    ax6 = fig.add_subplot(gs[2, :])
    
    metrics_names = ['Volume', 'Streamlines', 'Length']
    li_values = [
        asym['volume']['laterality_index'],
        asym['streamline_count']['laterality_index'],
        asym['mean_length']['laterality_index']
    ]
    
    if 'fa' in asym:
        metrics_names.append('FA')
        li_values.append(asym['fa']['laterality_index'])
    
    if 'md' in asym:
        metrics_names.append('MD')
        li_values.append(asym['md']['laterality_index'])
    
    colors = ['#2196F3' if li > 0 else '#F44336' for li in li_values]
    ax6.barh(metrics_names, li_values, color=colors, alpha=0.7, edgecolor='black')
    ax6.axvline(0, color='black', linewidth=1)
    ax6.axvline(-0.1, color='gray', linestyle='--', alpha=0.5)
    ax6.axvline(0.1, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Laterality Index (LI)')
    ax6.set_title('Asymmetry Analysis', fontweight='bold')
    ax6.grid(True, axis='x', alpha=0.3)
    ax6.text(0.12, 0.5, 'Left > Right', transform=ax6.transAxes, fontsize=9, color='blue')
    ax6.text(-0.12, 0.5, 'Right > Left', transform=ax6.transAxes, fontsize=9, color='red', ha='right')
    
    # Overall title
    plt.suptitle(f'CST Analysis Summary - {subject_id}', fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    fig_path = output_dir / f"{subject_id}_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary figure saved: {fig_path}")
    return fig_path


def plot_asymmetry_radar(asymmetry, output_dir, subject_id):
    """
    Create radar plot showing asymmetry across multiple metrics.
    
    Parameters
    ----------
    asymmetry : dict
        Asymmetry metrics from bilateral comparison
    output_dir : str or Path
        Output directory
    subject_id : str
        Subject identifier
        
    Returns
    -------
    fig_path : Path
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    metrics = []
    values = []
    
    if 'volume' in asymmetry:
        metrics.append('Volume')
        values.append(abs(asymmetry['volume']['laterality_index']))
    
    if 'streamline_count' in asymmetry:
        metrics.append('Streamline\nCount')
        values.append(abs(asymmetry['streamline_count']['laterality_index']))
    
    if 'fa' in asymmetry:
        metrics.append('FA')
        values.append(abs(asymmetry['fa']['laterality_index']))
    
    if 'md' in asymmetry:
        metrics.append('MD')
        values.append(abs(asymmetry['md']['laterality_index']))
    
    if len(metrics) < 3:
        print("Not enough metrics for radar plot")
        return None
    
    # Create radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # Close the plot
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#2196F3')
    ax.fill(angles, values, alpha=0.25, color='#2196F3')
    
    # Add threshold circle
    threshold = [0.1] * (len(metrics) + 1)
    ax.plot(angles, threshold, '--', linewidth=1, color='red', alpha=0.5, label='Asymmetry threshold (0.1)')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, max(0.3, max(values)))
    ax.set_title(f'Asymmetry Profile - {subject_id}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    # Save
    fig_path = output_dir / f"{subject_id}_asymmetry_radar.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Asymmetry radar plot saved: {fig_path}")
    return fig_path