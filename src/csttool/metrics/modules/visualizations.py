"""
visualizations.py

Visualization functions for CST metrics analysis.

This module provides:
- Tract profile plots (FA/MD along the tract)
- Bilateral comparison bar charts
- Tractogram QC slice previews
- Multi-panel summary figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from csttool.viz import geometry as _geo
from csttool.viz import style as _style
from csttool.viz.utils import deterministic_subsample, viz_rng
from csttool.metrics.modules.unilateral_analysis import TRACT_REGIONS


# Display names for the regions defined in TRACT_REGIONS. The extents themselves are not
# duplicated here: labels are positioned from TRACT_REGIONS so they always sit over the
# stretch of tract whose value the report tabulates.
_REGION_DISPLAY_NAMES = {
    'pontine': 'Pontine Level',
    'plic': 'PLIC',
    'precentral': 'Precentral Gyrus',
}


def _shade_tract_regions(ax, tick_suffix='%'):
    """Shade alternate anatomical region bands behind a profile, and tick the boundaries.

    ``tick_suffix`` is appended to each boundary tick label. The one-page report
    matrix passes ``''`` because the shared position axis is already labelled
    "(%)"; the standalone figures keep the per-tick ``%``.
    """
    for i, (_, start, end) in enumerate(TRACT_REGIONS):
        if i % 2:  # alternate, so the boundaries read without extra gridlines
            ax.axvspan(start * 100.0, end * 100.0, color=_style.REGION_BAND,
                       alpha=_style.REGION_BAND_ALPHA, linewidth=0, zorder=0)

    bounds = [0.0] + [end * 100.0 for _, _, end in TRACT_REGIONS]
    ax.set_xticks(bounds)
    ax.set_xticklabels([f'{b:g}{tick_suffix}' for b in bounds])


def _label_tract_regions(ax, y=-0.12, wrap=False):
    """
    Label each anatomical region at the centre of its extent.

    Regions are ranges, not points. The labels previously sat at 0/50/100%, but the regions
    they name span 0-35/35-70/70-100%, so only 'PLIC' was anywhere near the stretch of tract
    whose value the report tabulates. Positions come from TRACT_REGIONS, so the figure and
    `compute_localized_metrics` cannot disagree.
    """
    trans = ax.get_xaxis_transform()
    for name, start, end in TRACT_REGIONS:
        label = _REGION_DISPLAY_NAMES[name]
        ax.text((start + end) * 50.0, y, label.replace(' ', '\n') if wrap else label,
                transform=trans, ha='center', fontsize=9, style='italic')


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
    
    # MD is stored in mm²/s (~8e-4); the axis is labelled ×10⁻³ mm²/s, so scale
    # the profile and mean lines by 1000 to match (same convention as
    # plot_stacked_profiles). FA is dimensionless and unscaled.
    scale = 1000.0 if scalar == 'md' else 1.0

    left_profile = np.array(left_metrics[scalar]['profile']) * scale
    right_profile = np.array(right_metrics[scalar]['profile']) * scale

    n_points = len(left_profile)
    x = np.linspace(0, 100, n_points)  # Normalized position (0-100%)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot profiles
    ax.plot(x, left_profile, color=_style.LEFT, linewidth=2, label='Left CST', marker='o', markersize=4)
    ax.plot(x, right_profile, color=_style.RIGHT, linewidth=2, label='Right CST', marker='s', markersize=4)

    # Add mean lines
    left_mean = left_metrics[scalar]['mean'] * scale
    right_mean = right_metrics[scalar]['mean'] * scale
    ax.axhline(left_mean, color=_style.LEFT, linestyle='--', alpha=0.5, label=f'Left mean: {left_mean:.3f}')
    ax.axhline(right_mean, color=_style.RIGHT, linestyle='--', alpha=0.5, label=f'Right mean: {right_mean:.3f}')

    # Labels and formatting
    scalar_label = 'Fractional Anisotropy' if scalar == 'fa' else 'Mean Diffusivity (×10⁻³ mm²/s)'
    
    # Set x-axis with anatomical labels
    if anatomical_labels:
        _shade_tract_regions(ax)
        _label_tract_regions(ax, y=-0.12)
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
        
        ax.plot(x, left_profile, color=_style.LEFT, linewidth=2, label='Left CST', marker='o', markersize=3)
        ax.plot(x, right_profile, color=_style.RIGHT, linewidth=2, label='Right CST', marker='s', markersize=3)
        
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

        # Bands go behind every panel; only the bottom one carries the labels.
        _shade_tract_regions(ax)

        ax.grid(True, alpha=0.3)
        
        # Only add legend to first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Shared X-axis: labels go on the bottom plot only (shading is applied to every panel above)
    _label_tract_regions(axes[-1], y=-0.3, wrap=True)
            
    plt.tight_layout()
    # Increase bottom margin to prevent x-axis overlap
    plt.subplots_adjust(hspace=0.2, bottom=0.15)  # Minimize vertical space between plots
    
    fig_path = output_dir / f"{subject_id}_stacked_profiles.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Stacked profiles saved: {fig_path}")
    return fig_path


# Display titles + y-limits for the four along-tract scalars in the 2x2 matrix.
# y-limits are the existing fixed ranges from plot_stacked_profiles, retained
# with the existing auto-fallback (data that exceeds them auto-scales).
# Printed type sizes for the one-page report figures. The figures are generated
# at their final physical size (1 Matplotlib point == 1 printed point), so these
# are literal print sizes. The anatomical region labels are deliberately the
# smallest text in the matrix — they are positional context, never a heading, and
# must not out-shout the subplot titles.
_RPT_TITLE_PT = 9.0     # subplot titles
_RPT_LABEL_PT = 7.5     # axis labels
_RPT_TICK_PT = 7.0      # tick labels
_RPT_LEGEND_PT = 7.5    # shared legend
_RPT_REGION_PT = 6.5    # anatomical region labels in the shared strip

# Final printed sizes of the two composite report figures, in millimetres.
# The CSS places each image at exactly this width, and the figures are saved
# with their own figure bbox so the rendered height is exactly the second value.
# These two numbers are the whole figure half of the one-page A4 budget.
PROFILE_MATRIX_SIZE_MM = (194.0, 97.0)
QC_TRIPTYCH_SIZE_MM = (116.0, 38.0)


def _blend_on_white(color, alpha):
    """Composite ``color`` at ``alpha`` over white, as an opaque RGB tuple.

    The region bands are drawn as a translucent overlay; a legend swatch has to
    reproduce the *resulting* tint rather than the translucent source, otherwise
    the swatch reads as a different shade from the band it names.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def _draw_region_strip(ax):
    """Draw the shared anatomical-region strip beneath the 2x2 profile matrix.

    One strip for the whole matrix, spanning both columns: a position rule from
    0 to 100 % with ticks at the ``TRACT_REGIONS`` boundaries, the region names
    centred over their own intervals, and the shared position axis label. Region
    names are set at :data:`_RPT_REGION_PT` (smaller than the subplot titles) in
    the regular sans face — they are positional context, not headings.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    band_rgba = _blend_on_white(_style.REGION_BAND, _style.REGION_BAND_ALPHA)
    rule_y = 0.92
    ax.axhline(rule_y, color='#aeb9c8', linewidth=0.6)
    for _, start, end in TRACT_REGIONS:
        ax.plot([start * 100.0, start * 100.0], [rule_y - 0.22, rule_y],
                color='#aeb9c8', linewidth=0.6)
    ax.plot([100.0, 100.0], [rule_y - 0.22, rule_y], color='#aeb9c8', linewidth=0.6)

    for i, (name, start, end) in enumerate(TRACT_REGIONS):
        if i % 2:  # mirror the panels' alternating band so the strip keys them
            ax.axvspan(start * 100.0, end * 100.0, color=band_rgba,
                       linewidth=0, zorder=0)
        ax.text((start + end) * 50.0, rule_y - 0.52, _REGION_DISPLAY_NAMES[name],
                ha='center', va='center', fontsize=_RPT_REGION_PT, color='#333a45')

    ax.set_xlabel('Distance along tract (%)', fontsize=_RPT_LABEL_PT, labelpad=1)


_PROFILE_MATRIX_CONFIG = [
    {'key': 'fa', 'title': 'Fractional Anisotropy', 'ylabel': 'FA',
     'ylim': (0, 0.8), 'scale': 1},
    {'key': 'md', 'title': 'Mean Diffusivity', 'ylabel': 'MD (×10⁻³)',
     'ylim': (0.5, 1.2), 'scale': 1000},
    {'key': 'rd', 'title': 'Radial Diffusivity', 'ylabel': 'RD (×10⁻³)',
     'ylim': (0.3, 1.0), 'scale': 1000},
    {'key': 'ad', 'title': 'Axial Diffusivity', 'ylabel': 'AD (×10⁻³)',
     'ylim': (0.8, 1.8), 'scale': 1000},
]


def plot_profile_matrix(
    left_metrics,
    right_metrics,
    output_dir,
    subject_id,
):
    """2x2 along-tract profile matrix (FA / MD / RD / AD) for the PDF report.

    The scientific centerpiece of the report. Generated at its exact final print
    size (:data:`PROFILE_MATRIX_SIZE_MM`) and placed in the template at that same
    width, so 1 Matplotlib point = 1 printed point and the page budget is exact;
    DPI only affects raster sharpness. the exact figure bbox is mandatory
    here — a "tight" bounding box would crop the canvas by an unpredictable
    amount and silently change the printed height.

    Layout
    ------
    - One legend for the whole figure (Left CST / Right CST / PLIC region),
      owned by Matplotlib so it can be placed against the subplot geometry. The
      HTML template deliberately carries no profile legend of its own.
    - One shared anatomical-region strip spanning both columns underneath the
      grid, carrying the region names centred over their ``TRACT_REGIONS``
      intervals plus the shared position axis label. Per-panel landmark labels
      are not drawn: they used to collide with the x tick labels and were set in
      large italics that competed with the subplot titles.
    - Missing scalars degrade to an explicit ``N/A`` placeholder panel.
    """
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = [
        m for m in _PROFILE_MATRIX_CONFIG
        if m['key'] in left_metrics and m['key'] in right_metrics
    ]
    if not available:
        print("Warning: no profiles available for the profile matrix")
        return None

    fig_width = PROFILE_MATRIX_SIZE_MM[0] / 25.4
    fig_height = PROFILE_MATRIX_SIZE_MM[1] / 25.4
    fig = plt.figure(figsize=(fig_width, fig_height))
    # Two panel rows + a short shared region strip spanning both columns.
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 0.13],
        hspace=0.30, wspace=0.16,
        left=0.055, right=0.985, top=0.90, bottom=0.055,
    )
    axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    ]

    # The single profile legend for the report (see docstring).
    band_rgba = _blend_on_white(_style.REGION_BAND, _style.REGION_BAND_ALPHA)
    legend_handles = _style.hemisphere_legend_handles(
        left_label='Left CST', right_label='Right CST'
    ) + [Patch(facecolor=band_rgba, edgecolor='#b9c2cd', linewidth=0.5,
               label='PLIC region')]
    fig.legend(
        handles=legend_handles, loc='upper center', ncol=3,
        fontsize=_RPT_LEGEND_PT, frameon=False,
        bbox_to_anchor=(0.5, 1.005), handlelength=1.8, columnspacing=1.6,
        handletextpad=0.6, borderpad=0.0,
    )

    for i, m in enumerate(available):
        ax = axes[i]
        key = m['key']
        scale = m['scale']
        left_profile = np.array(left_metrics[key]['profile']) * scale
        right_profile = np.array(right_metrics[key]['profile']) * scale
        n_points = len(left_profile)
        x = np.linspace(0, 100, n_points)

        ax.plot(x, left_profile, color=_style.LEFT, linewidth=1.4, label='Left CST')
        ax.plot(x, right_profile, color=_style.RIGHT, linewidth=1.4, label='Right CST')

        ax.set_ylabel(m['ylabel'], fontsize=_RPT_LABEL_PT, labelpad=2)

        # y-limits: existing fixed ranges with the existing auto-fallback.
        all_data = np.concatenate([left_profile, right_profile])
        if np.min(all_data) < m['ylim'][0] or np.max(all_data) > m['ylim'][1]:
            margin = (np.max(all_data) - np.min(all_data)) * 0.1
            ax.set_ylim(max(0, np.min(all_data) - margin), np.max(all_data) + margin)
        else:
            ax.set_ylim(m['ylim'])

        ax.set_title(m['title'], fontsize=_RPT_TITLE_PT, pad=2.5)
        ax.tick_params(axis='both', labelsize=_RPT_TICK_PT, pad=1.5,
                       length=2, width=0.6)
        ax.set_xlim(0, 100)
        # Bands behind every panel; the shared strip below names them once.
        _shade_tract_regions(ax, tick_suffix='')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if i < 2:  # top row: the position ticks are read off the bottom row
            plt.setp(ax.get_xticklabels(), visible=False)

    # Fill any unused panels (fewer than 4 scalars) with an explicit N/A marker.
    for j in range(len(available), 4):
        axes[j].axis('off')
        axes[j].text(0.5, 0.5, 'N/A', transform=axes[j].transAxes,
                     ha='center', va='center', fontsize=9, color='#888888')

    _draw_region_strip(fig.add_subplot(gs[2, :]))

    fig_path = output_dir / f"{subject_id}_profile_matrix.png"
    # Save the exact figure canvas. The house style's savefig.bbox="tight" would
    # crop-and-pad by an unpredictable amount and silently change the printed
    # height, breaking the page budget; passing the figure's own bbox pins it.
    _style.save_figure(fig, fig_path, bbox_inches=fig.bbox_inches)
    plt.close(fig)
    print(f"✓ Profile matrix saved: {fig_path}")
    return fig_path


def _qc_slice_index(background_image, slice_type):
    """Center slice index for a QC view (axial slightly above center for IC)."""
    shape = background_image.shape
    if slice_type == 'axial':
        return shape[2] // 2 + 5
    if slice_type == 'sagittal':
        return shape[0] // 2
    return shape[1] // 2  # coronal


def _qc_slice_2d(background_image, slice_type, slice_idx):
    """The 2D display slice (already transposed for ``origin='lower'``)."""
    if slice_type == 'axial':
        return background_image[:, :, slice_idx].T
    if slice_type == 'sagittal':
        return background_image[slice_idx, :, :].T
    return background_image[:, slice_idx, :].T  # coronal


def _apply_common_canvas(ax, canvas_w, canvas_h):
    """Centre the axis's current view inside a common ``canvas_w x canvas_h`` box.

    Equivalent to padding the 2D slice to a common canvas before ``imshow``, but
    without touching the data or the voxel coordinates the streamline overlay is
    drawn in. Combined with ``aspect='equal'`` this gives every QC panel the same
    physical size and the same scale, letterboxed in black, with no anatomical
    distortion. Any x-axis inversion applied for the radiological convention is
    preserved.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    cx, cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
    x_sign = 1.0 if x_hi >= x_lo else -1.0
    y_sign = 1.0 if y_hi >= y_lo else -1.0
    ax.set_xlim(cx - x_sign * canvas_w / 2.0, cx + x_sign * canvas_w / 2.0)
    ax.set_ylim(cy - y_sign * canvas_h / 2.0, cy + y_sign * canvas_h / 2.0)
    # The axes background patch is not painted while the axis is off, so the
    # letterbox is drawn explicitly: an axes-spanning black rectangle behind the
    # image. Without it the padding reads as white and the three panels look
    # like different sizes even though their boxes are identical.
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes, facecolor='black',
        edgecolor='none', zorder=-10, clip_on=False,
    ))


def _render_qc_slice(
    ax,
    streamlines_left,
    streamlines_right,
    background_image,
    affine,
    slice_type,
    max_streamlines=500,
    norm=None,
    rng=None,
    slice_idx=None,
):
    """Render one QC slice (grayscale background + projected L/R streamlines).

    Shared by the standalone :func:`plot_tractogram_qc_preview` and the composite
    :func:`plot_tractogram_qc_triptych` so the two never carry independent
    rendering logic. When ``norm`` is provided it is passed to ``imshow`` so the
    three triptych views share one grayscale FA scale; when None each view uses
    Matplotlib's per-slice auto-normalization (the standalone behaviour).
    Streamlines are subsampled deterministically via :func:`viz_rng`/
    :func:`deterministic_subsample`. Returns the ``imshow`` mappable.

    ``slice_idx`` overrides the default centre-slice index when given.
    """
    if slice_idx is None:
        slice_idx = _qc_slice_index(background_image, slice_type)
    bg_slice = _qc_slice_2d(background_image, slice_type, slice_idx)

    im = ax.imshow(
        bg_slice, cmap='gray', origin='lower', aspect='equal', norm=norm,
    )

    inv_affine = np.linalg.inv(affine)

    axis_map = {
        'axial': (2, 0, 1),
        'sagittal': (0, 1, 2),
        'coronal': (1, 0, 2),
    }
    depth_axis, h_axis, v_axis = axis_map[slice_type]

    def _project(streamlines, color, alpha=0.6):
        if len(streamlines) == 0:
            return
        sub = deterministic_subsample(list(streamlines), max_streamlines, rng=rng)
        count = 0
        for sl in sub:
            if count >= max_streamlines:
                break
            voxel_coords = np.dot(sl, inv_affine[:3, :3].T) + inv_affine[:3, 3]
            near = np.abs(voxel_coords[:, depth_axis] - slice_idx) < 5
            if np.any(near):
                ax.plot(
                    voxel_coords[near, h_axis],
                    voxel_coords[near, v_axis],
                    color=color, linewidth=0.5, alpha=alpha,
                )
                count += 1

    _project(streamlines_left, _style.LEFT)
    _project(streamlines_right, _style.RIGHT)

    ax.axis('off')
    _geo.finalize_image_view(ax, affine, slice_type)
    return im


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
    Create compact 3D tractogram QC preview (single view, standalone output).

    Renders left (blue) and right (orange) CST streamlines overlaid on a brain
    slice at the internal-capsule level. Used for the standalone per-view QC
    PNGs written under ``--save-visualizations``; the PDF report uses the
    composite :func:`plot_tractogram_qc_triptych` instead.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = viz_rng()
    fig, ax = plt.subplots(figsize=(4, 4))
    _render_qc_slice(
        ax, streamlines_left, streamlines_right, background_image, affine,
        slice_type, max_streamlines=max_streamlines, rng=rng,
    )
    # Match the canvas aspect to the slice so the standalone PNG has minimal
    # whitespace (the slice is already drawn at correct pixel aspect via
    # aspect='equal').
    slice_idx = _qc_slice_index(background_image, slice_type)
    bg_slice = _qc_slice_2d(background_image, slice_type, slice_idx)
    data_ratio = bg_slice.shape[0] / bg_slice.shape[1]
    fig.set_size_inches(4, 4 * data_ratio)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=_style.LEFT, linewidth=2, label='Left CST'),
        Line2D([0], [0], color=_style.RIGHT, linewidth=2, label='Right CST'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    if set_title:
        ax.set_title(f'CST Tractogram ({slice_type.title()})', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path = output_dir / f"{subject_id}_tractogram_qc_{slice_type}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Tractogram QC preview saved: {fig_path}")
    return fig_path


def plot_tractogram_qc_triptych(
    streamlines_left,
    streamlines_right,
    background_image,
    affine,
    output_dir,
    subject_id,
    background_kind=None,
    max_streamlines=500,
    slice_indices=None,
):
    """
    Composite 1x3 tractogram QC figure for the one-page PDF report.

    Geometry
    --------
    The three views come from different voxel planes and therefore have
    different source shapes (e.g. 72x96 sagittal/coronal vs 96x96 axial). Each
    image axis is given the *same* physical box and the *same* common data
    canvas — the union of the three slice extents, centred on each slice — so
    the panels are equally sized and equally scaled, with black letterboxing
    where a slice is smaller than the canvas. Nothing is stretched: the
    anatomical aspect ratio is preserved by ``aspect='equal'``.

    The colorbar lives on its own GridSpec column and is then snapped to the
    vertical extent of the image row, so it can never overlap an image (it used
    to be stolen out of the shared axes area and landed on top of the axial
    panel).

    Scale
    -----
    Sagittal / coronal / axial slices share a single grayscale FA scale (one
    ``matplotlib.colors.Normalize(vmin=0, vmax=1)`` instance) which is also the
    colorbar's, so the three views and the colorbar cannot drift apart by
    construction. The FA colorbar is rendered **only** when
    ``background_kind == "fa"``; the caller must state what the background is
    (no boolean default) so an FA colorbar is never shown over a non-FA
    background.

    Size
    ----
    Generated at its exact final report size (:data:`QC_TRIPTYCH_SIZE_MM`) and
    saved with the exact figure bbox, so the printed height is the designed
    height and the one-page budget is exact.

    Parameters
    ----------
    background_kind : str or None
        ``"fa"`` to render the FA colorbar; anything else (e.g. ``"other"``)
        suppresses it. No implicit default — the caller must declare it.
    slice_indices : dict, optional
        ``{'sagittal': i, 'coronal': j, 'axial': k}`` override; falls back to
        :func:`_qc_slice_index` (center slices).
    """
    from matplotlib.colors import Normalize

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    show_colorbar = background_kind == "fa"
    # One Normalize instance shared by all three images and the colorbar.
    norm = Normalize(vmin=0.0, vmax=1.0) if show_colorbar else None
    rng = viz_rng()

    views = ('sagittal', 'coronal', 'axial')
    fig_width = QC_TRIPTYCH_SIZE_MM[0] / 25.4
    fig_height = QC_TRIPTYCH_SIZE_MM[1] / 25.4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 0.05], wspace=0.04,
        left=0.005, right=0.945, top=0.90, bottom=0.02,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    # Common display canvas: the largest slice extent across the three views, so
    # every panel covers the same number of voxels and is scaled identically.
    canvas_h, canvas_w = 0, 0
    for view in views:
        idx = (slice_indices or {}).get(view, _qc_slice_index(background_image, view))
        h, w = _qc_slice_2d(background_image, view, idx).shape
        canvas_h, canvas_w = max(canvas_h, h), max(canvas_w, w)

    last_im = None
    for ax, view in zip(axes, views):
        last_im = _render_qc_slice(
            ax, streamlines_left, streamlines_right, background_image, affine,
            view, max_streamlines=max_streamlines, norm=norm, rng=rng,
            slice_idx=(slice_indices or {}).get(view),
        )
        _apply_common_canvas(ax, canvas_w, canvas_h)
        ax.set_title(view.capitalize(), fontsize=_RPT_TICK_PT, pad=1.5)

    if show_colorbar and last_im is not None:
        cbar = fig.colorbar(
            last_im, cax=cax, orientation='vertical',
            ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        # Label above the bar, not rotated beside it: a rotated y-label needs a
        # wide right margin that would come straight out of the panel widths.
        cax.set_title('FA', fontsize=6.5, pad=2)
        cbar.ax.tick_params(labelsize=6, length=1.5, width=0.5, pad=1)
        cbar.outline.set_linewidth(0.5)
        # Snap the colorbar to the image row's real vertical extent. The image
        # axes shrink to their data aspect only at draw time, so this has to
        # happen after a draw pass, and must read the *active* position.
        fig.canvas.draw()
        img_box = axes[0].get_position(original=False)
        cbar_box = cax.get_position()
        cax.set_position([cbar_box.x0, img_box.y0, cbar_box.width, img_box.height])
    else:
        cax.set_visible(False)

    fig_path = output_dir / f"{subject_id}_tractogram_qc_triptych.png"
    # Exact canvas, not a tight bbox — see plot_profile_matrix.
    _style.save_figure(fig, fig_path, bbox_inches=fig.bbox_inches)
    plt.close(fig)
    print(f"✓ Tractogram QC triptych saved: {fig_path}")
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
        colors = [_style.LEFT, _style.RIGHT]  # Blue for left, red for right
        
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
        
        ax1.plot(x, left_profile, color=_style.LEFT, linewidth=2, label='Left', marker='o', markersize=3)
        ax1.plot(x, right_profile, color=_style.RIGHT, linewidth=2, label='Right', marker='s', markersize=3)
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
           color=[_style.LEFT, _style.RIGHT], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count')
    ax3.set_title('Streamline Count', fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Panel 4: Volume Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(['Left', 'Right'],
           [left['morphology']['tract_volume'], right['morphology']['tract_volume']],
           color=[_style.LEFT, _style.RIGHT], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Volume (mm³)')
    ax4.set_title('Tract Volume', fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Panel 5: FA Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    if 'fa' in left:
        ax5.bar(['Left', 'Right'],
               [left['fa']['mean'], right['fa']['mean']],
               color=[_style.LEFT, _style.RIGHT], alpha=0.7, edgecolor='black')
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
    
    colors = [_style.LEFT if li > 0 else _style.RIGHT for li in li_values]
    ax6.barh(metrics_names, li_values, color=colors, alpha=0.7, edgecolor='black')
    ax6.axvline(0, color='black', linewidth=1)
    ax6.axvline(-0.1, color='gray', linestyle='--', alpha=0.5)
    ax6.axvline(0.1, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Laterality Index (LI)')
    ax6.set_title('Asymmetry Analysis', fontweight='bold')
    ax6.grid(True, axis='x', alpha=0.3)
    # Positive LI bars extend right (Left > Right); negative extend left
    # (Right > Left). Label each side where its bars actually point.
    ax6.text(0.98, 0.95, 'Left > Right', transform=ax6.transAxes, fontsize=9,
             color=_style.LEFT, ha='right', va='top')
    ax6.text(0.02, 0.95, 'Right > Left', transform=ax6.transAxes, fontsize=9,
             color=_style.RIGHT, ha='left', va='top')
    
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
    ax.plot(angles, values, 'o-', linewidth=2, color=_style.LEFT)
    ax.fill(angles, values, alpha=0.25, color=_style.LEFT)
    
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