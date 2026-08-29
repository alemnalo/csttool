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
_RPT_BAND_NOTE_PT = 5.5  # the IQR clause in the shared strip

# The per-node interquartile range drawn behind each profile line. The band is a
# scientific claim (the bundle's spread at that node), so it is named on the
# figure rather than left to the caller's caption.
_BAND_CAPTION = 'shaded band = interquartile range across streamlines'

# Fill opacity of one hemisphere's IQR band. Blue over orange composites to
# ~0.36 combined alpha, which reads as a distinct third tint; the per-band edge
# lines below are what keep each band's own extent traceable through the
# overlap. Raising this is the documented remedy if a diffusivity band proves
# invisible at print size — hence the per-scalar override rather than a constant.
_BAND_ALPHA = 0.20
_BAND_EDGE_ALPHA = 0.45
_BAND_EDGE_LW = 0.5

# Layering. The profile lines are the data and must never be occluded, so their
# zorder is now explicit rather than relying on draw order.
_Z_REGION_BAND = 0      # _shade_tract_regions (unchanged)
_Z_IQR_BAND = 1
_Z_GRID = 2             # Matplotlib's default gridline zorder (unchanged)
_Z_PROFILE_LINE = 3

# Final printed sizes of the two composite report figures, in millimetres.
# The profile matrix is 94 mm rather than its original 97: the IQR bands make
# each panel more legible, so a 3 mm reduction is affordable, and it is the
# prescribed second step of the page-budget ladder.
# The CSS places each image at exactly this width, and the figures are saved
# with their own figure bbox so the rendered height is exactly the second value.
# These two numbers are the whole figure half of the one-page A4 budget.
PROFILE_MATRIX_SIZE_MM = (194.0, 94.0)
QC_TRIPTYCH_SIZE_MM = (116.0, 38.0)

# The 1x4 report QC strip that replaces the triptych. Full content width, so the
# coronal panels are ~47 mm rather than the triptych's 38.7 mm.
#
# Height is DERIVED, not fixed (see :func:`qc_strip_geometry`). The strip used to
# allocate each panel a fixed rectangle and let ``imshow``'s ``aspect='equal'``
# shrink it at draw time, which put the title wherever the subject's acquisition
# matrix happened to put it: measured title tops of 44.13 mm on a 24x20x18 grid
# (clipped by the exact-bbox save) against 40.58 mm on a 128x128x76 one, with
# ~5 mm of the canvas left as dead white in between. The image box is now
# computed from the data aspect so ``apply_aspect`` is a no-op and every
# annotation anchors to a box that will not move.
QC_STRIP_WIDTH_MM = 194.0

# Nominal height, for the page budget below. The real height comes from
# :func:`qc_strip_geometry`; this is what a typical 1.6-aspect coronal grid
# produces. Measured page use with it: 273.8 mm of 281.
QC_STRIP_NOMINAL_HEIGHT_MM = 46.0
QC_STRIP_SIZE_MM = (QC_STRIP_WIDTH_MM, QC_STRIP_NOMINAL_HEIGHT_MM)

# Bounds on the derived height. The report measured 271.4 mm of 281 before this
# change and ``test_layout_keeps_headroom`` demands 6 mm spare, so 3.6 mm was
# spendable against the old 44 mm — hence the 47.5 ceiling. The floor stops a
# very wide grid from producing a degenerate strip.
QC_STRIP_MIN_HEIGHT_MM = 42.0
QC_STRIP_MAX_HEIGHT_MM = 47.5

# Internal vertical budget, in millimetres. Each band is the height the type it
# holds actually renders at, plus its pad — the previous constants were a page
# budget ladder that had never been measured against rendered type, so the 2.5 mm
# title band held a 7 pt title needing ~3.0 mm and the 2.2 mm colourbar-label
# band held a label needing 2.77 mm, which is why that label overprinted the
# shared caption by a measured 0.46 mm.
# The vertical stack, top to bottom. Every panel is the same three zones in the
# same order — title, image, fixed-height key — so the four columns align band
# for band whatever their keys contain.
_STRIP_MARGIN_TOP_MM = 2.6   # below the HTML section heading
_STRIP_TITLE_MM = 4.0        # row of 9 pt bold panel titles (3.17 mm + 2 pt pad)
_STRIP_IMAGE_KEY_GAP_MM = 1.4   # image -> its own key
_STRIP_KEY_MM = 6.2          # per-panel key band: swatches, or panel 2's colourbar
_STRIP_KEY_CAPTION_GAP_MM = 2.6  # the four keys -> the shared caption
_STRIP_CAPTION_MM = 3.2      # the one shared caption line
_STRIP_MARGIN_BOTTOM_MM = 1.4
_STRIP_GAP_MM = 2.6          # between panels, so the four key bands read as columns

# The two gaps above are what separate the three zones. The key -> caption gap is
# deliberately the larger of the two, and larger than the image -> key gap: the
# caption describes the whole composite, so it must not read as a fifth legend
# sitting under panel 4. Proximity is the only thing distinguishing them, since
# both are centred type at the same size.

# Largest image height we will spend. Binding it caps the strip at
# 21.4 + 25.5 = 46.9 mm, inside QC_STRIP_MAX_HEIGHT_MM and leaving 6.7 mm of
# page headroom against the 6.0 the layout test demands. A grid tall enough to
# bind it letterboxes inside a full-width panel rather than narrowing it — see
# qc_strip_geometry for why the panel width is the thing held constant. The
# letterbox is black against a slice whose own margins are black, so it is
# invisible; what it costs is image scale, which is the only place the extra
# whitespace in this stack could have come from.
_STRIP_IMAGE_MAX_MM = 25.5

# Type sizes. The strip used to be set two points below the profile matrix at
# every level — 7 pt titles against its 9, and a 5.5 pt primary legend at the
# size of the matrix's *footnote*. Titles now match _RPT_TITLE_PT exactly so the
# two report figures cannot drift apart again.
_STRIP_TITLE_PT = _RPT_TITLE_PT   # 9.0
_STRIP_KEY_PT = 6.5
_STRIP_CAPTION_PT = 6.5
_STRIP_MARKER_PT = 6.0
_STRIP_NOTE_PT = 6.0       # the italic "not produced" overlay on a degraded panel
_STRIP_ROI_LINEWIDTH = 0.8

# Key-row metrics, in millimetres.
_STRIP_SWATCH_MM = 2.2      # length of one colour swatch line
_STRIP_SWATCH_GAP_MM = 0.9  # swatch -> its own label
_STRIP_ENTRY_GAP_MM = 1.9   # between one entry and the next
# Panel 2's colourbar. The bar is thick enough to read as a scale rather than a
# rule, and its width is *measured* rather than fixed (see the drawing code): the
# two end ticks flank it, and a subject whose vmax needs more digits must eat
# into the bar, never into the neighbouring panel.
_STRIP_CBAR_MM = 2.2        # the bar itself
_STRIP_CBAR_MAX_FRAC = 0.72  # widest the bar may be, as a fraction of the panel
_STRIP_CBAR_MIN_FRAC = 0.42  # narrowest, before the ticks are allowed to crowd
_STRIP_CBAR_GAP_MM = 0.7    # key label row -> the bar below it

# Padding from the top of the key band to the cap height of its label row. The
# label row is the alignment anchor shared by all four columns: panels 1, 3 and 4
# put their swatches on it and panel 2 puts its quantity name there, with the bar
# beneath. Every label is set on one shared baseline, derived from the cap-height
# fraction below, so glyph content cannot shift a column off the row.
_STRIP_KEY_TEXT_PAD_MM = 0.35
_STRIP_KEY_CAP_FRAC = 0.72  # cap height as a fraction of the em, for DejaVu Sans

# Floor for the auto-fit in _fit_key_fontsize. The ROI key is the widest row on
# the strip (three swatches and three words in one panel width), so it is what
# normally sets the common size; below this the key stops being readable at
# print size and the right answer would be shorter words, not smaller type.
_STRIP_KEY_MIN_PT = 5.5

_STRIP_INK = '#333a45'      # caption and key type
_STRIP_INK_MUTED = '#9aa3ae'  # an ROI that the display slab does not reach

# World-axis colours for the DEC direction key. Red/green/blue is the DEC
# convention itself (it is what the image encodes), not a csttool palette
# choice, so these are deliberately not style.LEFT/RIGHT.
_DEC_AXIS_KEY = (('#d62728', 'L–R'), ('#2ca02c', 'A–P'), ('#1f77b4', 'S–I'))


def qc_strip_geometry(canvas_w, canvas_h):
    """Figure size and panel rectangles for the QC strip, in millimetres.

    The strip's whole layout is a pure function of the displayed slice's shape,
    computed here so the caller can ``set_position`` each Axes to a rectangle
    whose aspect already equals the data's. That is what makes
    ``Axes.apply_aspect`` a no-op: Matplotlib's default ``adjustable='box'``
    otherwise shrinks and re-centres an ``aspect='equal'`` image box at draw
    time, and every annotation anchored to the Axes moves with it while every
    annotation anchored to a precomputed millimetre band does not.

    Two regimes, both handled:

    * **wide grids** (the normal case; a 96x96x60 volume gives aspect 1.6) are
      width-bound, so the image height falls straight out of the aspect and
      fills the panel exactly;
    * **tall grids** would want more height than :data:`_STRIP_IMAGE_MAX_MM`,
      so the height is capped and the image letterboxes horizontally inside a
      panel that keeps its full width — which is what
      :func:`csttool.viz.geometry.pad_axes_to_canvas` exists for.

    The panel width is deliberately **constant**. Narrowing it to keep the image
    flush would be the obvious alternative, but the key band underneath is
    width-critical: at aspect 1.33 the panel would fall to 40 mm while the ROI
    key needs ~45, so the keys and even the titles begin to collide. Trading
    image area for a key that fits is the right way round.

    The cap binds below aspect ~1.83, which includes the range real DWI grids
    occupy, so the typical panel *is* letterboxed — by ~2.9 mm a side at aspect
    1.6. That is deliberate and it is what pays for the whitespace in the
    vertical stack: the letterbox is black against a slice whose own margins are
    black, so it costs image scale and nothing else, while the page had only
    1.2 mm of headroom left to give.

    Parameters
    ----------
    canvas_w, canvas_h : int
        Width and height of the displayed slice in voxels, i.e. the reversed
        shape of :func:`csttool.viz.geometry.slice_2d`.

    Returns
    -------
    dict
        ``width_mm``/``height_mm`` (the figure), ``panel_w_mm``/``image_h_mm``
        (one panel's image box), ``panel_x0_mm`` (four left edges), the band
        origins ``image_y0_mm``/``key_y0_mm``/``caption_y0_mm``, ``aspect`` (the
        data's) and ``box_aspect`` (the panel box's; they differ only when the
        image-height cap binds, and the difference is the letterbox).
    """
    aspect = float(canvas_w) / float(canvas_h)

    furniture_mm = (_STRIP_MARGIN_TOP_MM + _STRIP_TITLE_MM
                    + _STRIP_IMAGE_KEY_GAP_MM + _STRIP_KEY_MM
                    + _STRIP_KEY_CAPTION_GAP_MM + _STRIP_CAPTION_MM
                    + _STRIP_MARGIN_BOTTOM_MM)

    panel_w = (QC_STRIP_WIDTH_MM - 3 * _STRIP_GAP_MM) / 4.0
    image_h = min(panel_w / aspect, _STRIP_IMAGE_MAX_MM)

    height_mm = furniture_mm + image_h
    clamped = min(max(height_mm, QC_STRIP_MIN_HEIGHT_MM), QC_STRIP_MAX_HEIGHT_MM)
    # Only the floor can bind (furniture + the image cap is 46.9 mm), so the
    # slack is non-negative; split it between the two margins so a very wide
    # grid centres its content rather than hanging from the top.
    slack = max(0.0, clamped - height_mm)
    height_mm = clamped
    margin_bottom = _STRIP_MARGIN_BOTTOM_MM + slack / 2.0

    # Stacked from the bottom, with the two gaps as real bands rather than
    # implied by the type's own leading — which is what let the key row and the
    # caption sit a bare 0.9 mm apart and read as one block of five legends.
    caption_y0 = margin_bottom
    key_y0 = caption_y0 + _STRIP_CAPTION_MM + _STRIP_KEY_CAPTION_GAP_MM
    image_y0 = key_y0 + _STRIP_KEY_MM + _STRIP_IMAGE_KEY_GAP_MM

    row_w = 4 * panel_w + 3 * _STRIP_GAP_MM
    x_start = (QC_STRIP_WIDTH_MM - row_w) / 2.0
    panel_x0 = [x_start + i * (panel_w + _STRIP_GAP_MM) for i in range(4)]

    return {
        "width_mm": QC_STRIP_WIDTH_MM,
        "height_mm": height_mm,
        "panel_w_mm": panel_w,
        "image_h_mm": image_h,
        "panel_x0_mm": panel_x0,
        "image_y0_mm": image_y0,
        "key_y0_mm": key_y0,
        "caption_y0_mm": caption_y0,
        "aspect": aspect,
        "box_aspect": panel_w / image_h,
    }


def _blend_on_white(color, alpha):
    """Composite ``color`` at ``alpha`` over white, as an opaque RGB tuple.

    The region bands are drawn as a translucent overlay; a legend swatch has to
    reproduce the *resulting* tint rather than the translucent source, otherwise
    the swatch reads as a different shade from the band it names.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def _draw_region_strip(ax, band_note=False):
    """Draw the shared anatomical-region strip beneath the 2x2 profile matrix.

    One strip for the whole matrix, spanning both columns: a position rule from
    0 to 100 % with ticks at the ``TRACT_REGIONS`` boundaries, the region names
    centred over their own intervals, and the shared position axis label. Region
    names are set at :data:`_RPT_REGION_PT` (smaller than the subplot titles) in
    the regular sans face — they are positional context, not headings.

    ``band_note`` adds the right-aligned clause naming the IQR band, drawn once
    for the whole matrix rather than per panel.
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

    if band_note:
        # The IQR band is named here rather than in the figure legend: a fourth
        # legend entry would push the three-entry row onto two lines, and the
        # band qualifies both hemisphere lines rather than being a series of its
        # own. It sits on the axis-label line, right-aligned — the region names
        # occupy the row above and the rightmost of them ("Precentral Gyrus") is
        # centred at 85 %, so a right-aligned note on that row would overlap it.
        ax.text(1.0, -0.62, _BAND_CAPTION, transform=ax.transAxes,
                ha='right', va='center', fontsize=_RPT_BAND_NOTE_PT,
                color='#5f6b7a')


# ``band_alpha`` is per-scalar so a diffusivity panel whose IQR proves too faint
# at print size can be darkened without touching the FA panel — and, critically,
# without changing any panel's fixed y-range, which would break cross-subject
# comparability. Omitted keys fall back to _BAND_ALPHA.
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


def _draw_iqr_band(ax, x, block, color, scale, alpha):
    """Fill one hemisphere's per-node IQR behind its profile line.

    Returns the (p25, p75) arrays in display units, or None when the block
    carries no dispersion — a metrics JSON written by an older version, or a
    hemisphere with no contributing streamlines. In both cases the panel renders
    the line alone: a band is never inferred, and its absence is never warned
    about, because a legacy report is a legitimate input.

    Each band's own p25 and p75 edges are stroked in the same hue so that where
    the blue and orange bands overlap, a reader can still trace which extent
    belongs to which hemisphere. Without the edges the composite tint is
    ambiguous at 90 mm panel width.
    """
    if 'profile_p25' not in block or not block.get('profile_n'):
        return None
    p25 = np.asarray(block['profile_p25'], dtype=float) * scale
    p75 = np.asarray(block['profile_p75'], dtype=float) * scale
    ax.fill_between(x, p25, p75, color=color, alpha=alpha, linewidth=0,
                    zorder=_Z_IQR_BAND)
    for edge in (p25, p75):
        ax.plot(x, edge, color=color, linewidth=_BAND_EDGE_LW,
                alpha=_BAND_EDGE_ALPHA, zorder=_Z_IQR_BAND)
    return p25, p75


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

    Dispersion
    ----------
    Each hemisphere's per-node interquartile range is filled behind its line, so
    a reader can tell a consensus from an average over dissent — two profiles
    whose means separate while their IQRs overlap everywhere do not support the
    difference the bare lines imply. The centre line stays the **mean**: the
    twelve regional values and their laterality indices are derived from this
    exact array, so switching it to the median would change every published
    regional metric. Where mean and IQR diverge visibly, that divergence is the
    finding.

    A block without ``profile_p25`` (legacy metrics JSON) or with
    ``profile_n == 0`` (empty hemisphere) simply gets no band; the other
    hemisphere still gets one, and the figure is otherwise unchanged.
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

    any_band = False
    for i, m in enumerate(available):
        ax = axes[i]
        key = m['key']
        scale = m['scale']
        left_profile = np.array(left_metrics[key]['profile']) * scale
        right_profile = np.array(right_metrics[key]['profile']) * scale
        n_points = len(left_profile)
        x = np.linspace(0, 100, n_points)

        # Bands first, and behind: the lines are the data and must stay legible
        # wherever the two hemispheres' bands overlap.
        alpha = m.get('band_alpha', _BAND_ALPHA)
        band_extents = [
            _draw_iqr_band(ax, x, left_metrics[key], _style.LEFT, scale, alpha),
            _draw_iqr_band(ax, x, right_metrics[key], _style.RIGHT, scale, alpha),
        ]
        any_band = any_band or any(e is not None for e in band_extents)

        ax.plot(x, left_profile, color=_style.LEFT, linewidth=1.4,
                label='Left CST', zorder=_Z_PROFILE_LINE)
        ax.plot(x, right_profile, color=_style.RIGHT, linewidth=1.4,
                label='Right CST', zorder=_Z_PROFILE_LINE)

        ax.set_ylabel(m['ylabel'], fontsize=_RPT_LABEL_PT, labelpad=2)

        # y-limits: existing fixed ranges with the existing auto-fallback, now
        # measured over the band extents as well as the lines. A band that
        # clipped at the axis edge would show a narrower spread than the bundle
        # has, which is the one thing this figure must not do.
        all_data = np.concatenate(
            [left_profile, right_profile]
            + [edge for extent in band_extents if extent for edge in extent]
        )
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

    _draw_region_strip(fig.add_subplot(gs[2, :]), band_note=any_band)

    fig_path = output_dir / f"{subject_id}_profile_matrix.png"
    # Save the exact figure canvas. The house style's savefig.bbox="tight" would
    # crop-and-pad by an unpredictable amount and silently change the printed
    # height, breaking the page budget; passing the figure's own bbox pins it.
    _style.save_figure(fig, fig_path, bbox_inches=fig.bbox_inches)
    plt.close(fig)
    _write_profile_matrix_sidecar(
        fig_path.with_suffix('.json'), subject_id, left_metrics, right_metrics,
        available, any_band,
    )
    print(f"✓ Profile matrix saved: {fig_path}")
    return fig_path


def _write_profile_matrix_sidecar(path, subject_id, left_metrics, right_metrics,
                                  available, band_drawn):
    """Record what the profile matrix's band claims, beside the PNG.

    The QC panels each carry a sidecar; the profile matrix did not, because until
    now it drew only values already tabulated in the report. The band is a new
    scientific claim — the spread of a population whose size is nowhere else on
    the page — so ``profile_n`` is recorded per scalar per hemisphere.

    Deliberately not `qc_figures._write_plot_sidecar`: that helper lives on the
    QC side of the report/QC boundary and writes a ``Question`` field this figure
    does not have.
    """
    import json

    payload = {
        'Subject': subject_id,
        'Panel': 'profile-matrix',
        'Band': 'interquartile range across contributing streamlines',
        'BandDrawn': bool(band_drawn),
        'ContributingStreamlines': {
            m['key']: {
                'left': left_metrics[m['key']].get('profile_n'),
                'right': right_metrics[m['key']].get('profile_n'),
            }
            for m in available
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# The 1x4 report QC strip
# ---------------------------------------------------------------------------

def _require_fa_grid(img, fa_shape, fa_affine, name):
    """Reject a volume that is not on the FA grid, rather than resampling it.

    FA, the world-frame V1 field, the CST density volume and the ROI
    segmentation are all produced on the FA grid, so the strip slices them
    against one another with no resampling at all. A volume on a different grid
    — a legacy original-orientation ROI file, say — would silently draw the
    right shape in the wrong place, which is worse than not drawing it.

    Grid transfer is a real operation with a right tool
    (``nibabel.processing.resample_from_to`` with ``order=0`` for a label map),
    but it belongs to the stage that owns the volume, not to a report figure.
    """
    if tuple(img.shape[:3]) != tuple(fa_shape[:3]):
        raise ValueError(
            f"{name} is not on the FA grid: shape {tuple(img.shape[:3])} vs "
            f"FA {tuple(fa_shape[:3])}. The report does not resample; produce "
            "the volume on the FA grid in the stage that owns it."
        )
    if not np.allclose(np.asarray(img.affine, dtype=float),
                       np.asarray(fa_affine, dtype=float), atol=1e-4):
        raise ValueError(
            f"{name} is not on the FA grid: its affine differs from FA's. The "
            "report does not resample; produce the volume on the FA grid in "
            "the stage that owns it."
        )


def _slab_projected_volume(mask, affine, view, index, slab_mm):
    """Project a 3-D mask through the display slab onto the displayed plane.

    A coronal plane chosen for maximum *bundle* occupancy is chosen for the
    bundle, not for the ROIs, and can miss a cortical ROI entirely. Projecting
    each mask through the same 10 mm slab panel 4 draws its streamlines in makes
    the ROI panel slab-consistent with the trajectory panel, and materially
    raises the chance all three ROIs appear at all.

    The projection is returned as a volume whose plane ``index`` holds the 2-D
    result, so :func:`csttool.viz.render.render_mask_contour` can stay a dumb
    primitive that slices a volume — no second contour implementation, and no
    2-D-vs-3-D branch inside the renderer.

    Returns ``(volume, n_voxels_in_slab)``.
    """
    mask = np.asarray(mask).astype(bool)
    h_axis, v_axis = _geo.VIEW_AXES[view]
    depth_axis = 3 - h_axis - v_axis

    n_planes = mask.shape[depth_axis]
    centres = np.zeros((n_planes, 3))
    centres[:, depth_axis] = np.arange(n_planes)
    world = centres @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
    in_slab = _geo.slab_membership(world, affine, view, index, slab_mm)

    slab = np.zeros_like(mask)
    slab_index = [slice(None)] * 3
    slab_index[depth_axis] = np.where(in_slab)[0]
    slab[tuple(slab_index)] = mask[tuple(slab_index)]
    projected = slab.any(axis=depth_axis)

    volume = np.zeros(mask.shape, dtype=np.uint8)
    plane_index = [slice(None)] * 3
    plane_index[depth_axis] = index
    volume[tuple(plane_index)] = projected.astype(np.uint8)
    return volume, int(np.count_nonzero(slab))


def _load_optional(path, fa_shape, fa_affine, name):
    """Load an optional strip input, or return None if it is not there.

    A missing product is a normal state (older derivatives, a skipped stage) and
    degrades its panel. A *present* product on the wrong grid is a defect and
    raises — the two must not be conflated.
    """
    import nibabel as nib

    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    img = nib.load(str(path))
    _require_fa_grid(img, fa_shape, fa_affine, name)
    return img


def _strip_caption_text(slice_index, provenance, slab_mm):
    """The one shared caption line — only what is true of all four panels.

    The hemisphere counts and the ROI colour key used to live here too, which
    put every key away from the data it described and merged four unrelated
    facts into one 5.5 pt line. They now sit in their own panel's key band, so
    what is left is genuinely shared: the plane every panel shows, the rule that
    chose it, the display slab, and the orientation convention (previously
    implied only by the per-panel R/L glyphs).

    The slab governs panels 3 and 4 only, but it is a property of how the
    composite is displayed rather than of either panel's science, so it stays
    shared rather than being printed twice.
    """
    rule = str(provenance.get('rule', 'unknown')).replace('_', ' ')
    return (f"Coronal slice {slice_index} · selected by {rule} · "
            f"{slab_mm:g} mm display slab · "
            f"radiological convention (R at viewer left)")


def _fit_key_fontsize(ax, renderer, rows, start_pt, floor_pt=_STRIP_KEY_MIN_PT):
    """The largest type at or below ``start_pt`` at which every key row fits.

    One size for all four columns, not one per column: a key band whose columns
    disagree about type size reads as four unrelated captions rather than one
    row. The ROI key is the widest row and therefore normally the one that
    decides, so the whole band tracks it.

    Swatch and gap widths are fixed millimetres and text width scales with point
    size, so the fit is solved directly from a single measurement per row rather
    than by iterating.
    """
    box_w = ax.get_window_extent(renderer).width
    px_per_mm = ax.figure.dpi / 25.4
    scale = 1.0
    for entries in rows:
        if not entries:
            continue
        fixed = sum((_STRIP_SWATCH_MM + _STRIP_SWATCH_GAP_MM) * px_per_mm
                    for color, *_ in entries if color)
        fixed += _STRIP_ENTRY_GAP_MM * px_per_mm * (len(entries) - 1)
        text = 0.0
        for entry in entries:
            probe = ax.text(0, 0, entry[1], fontsize=start_pt)
            text += probe.get_window_extent(renderer).width
            probe.remove()
        if text <= 0:
            continue
        scale = min(scale, max(0.0, box_w - fixed) / text)
    return max(floor_pt, min(start_pt, start_pt * scale))


def _draw_key_row(ax, renderer, entries, fontsize, *, y):
    """Lay out one centred row of ``swatch + label`` pairs for a single panel.

    This is the strip's one legend primitive: panels 1, 3 and 4 each get a row
    of it in their own column of the key band, so a reader decodes a contour or
    a trajectory without leaving the panel. Panel 2 cannot use it — a
    continuous scale is not a set of swatches — and draws a colourbar into the
    same band instead.

    Each token is measured and placed in sequence because Matplotlib has no
    rich text and the colour *is* the key. Measurement is a pure function of the
    text, the font and the figure DPI, so the result is reproducible.

    ``entries`` is a sequence of ``(colour, label)`` or
    ``(colour, label, label_colour)``; a ``colour`` of None draws the label with
    no swatch. Overflow is deliberately left visible rather than scaled away — a
    key that does not fit its column is a layout defect and the tests assert
    against it.

    ``y`` is the row's **baseline** in axes fraction, and the type is set on it
    with ``va='baseline'``. Centring instead (``va='center'``) centres each
    string's bounding box, so a row of all-caps labels sits 0.127 mm off a row
    with ascenders — visible as four legends that do not quite line up.
    """
    from matplotlib.lines import Line2D

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if not entries:
        return

    entries = [e if len(e) == 3 else (e[0], e[1], _STRIP_INK) for e in entries]
    fig = ax.figure
    px_per_mm = fig.dpi / 25.4
    swatch = _STRIP_SWATCH_MM * px_per_mm
    swatch_gap = _STRIP_SWATCH_GAP_MM * px_per_mm
    entry_gap = _STRIP_ENTRY_GAP_MM * px_per_mm

    widths = []
    for color, label, _ in entries:
        probe = ax.text(0, 0, label, fontsize=fontsize)
        text_w = probe.get_window_extent(renderer).width
        probe.remove()
        widths.append(text_w + (swatch + swatch_gap if color else 0.0))

    box = ax.get_window_extent(renderer)
    total = sum(widths) + entry_gap * (len(entries) - 1)
    x = max(0.0, (box.width - total) / 2.0)
    # The swatch is a mark, not type, so it rides at the optical middle of the
    # cap height rather than on the baseline the labels sit on.
    band_mm = box.height / fig.dpi * 25.4
    cap_mm = _STRIP_KEY_CAP_FRAC * fontsize * 25.4 / 72.0
    mark_y = y + 0.35 * cap_mm / band_mm

    for (color, label, label_color), width in zip(entries, widths):
        if color:
            ax.add_line(Line2D(
                [x / box.width, (x + swatch) / box.width], [mark_y, mark_y],
                transform=ax.transAxes, color=color, linewidth=1.6,
                solid_capstyle='butt', clip_on=False,
            ))
            label_x = (x + swatch + swatch_gap) / box.width
        else:
            label_x = x / box.width
        ax.text(label_x, y, label, transform=ax.transAxes, ha='left',
                va='baseline', fontsize=fontsize, color=label_color)
        x += width + entry_gap


def plot_report_qc_strip(
    fa_path,
    v1_path,
    density_path,
    roi_dseg_path,
    cst_left_path,
    cst_right_path,
    output_dir,
    subject_id,
    *,
    slab_mm=_geo.DEFAULT_SLAB_MM,
    max_streamlines=500,
    seed=None,
):
    """1x4 report QC strip: DEC-FA, CST density, extraction ROIs, CST over FA.

    Replaces the 1x3 triptych, which showed the same information (streamlines
    over FA) from three angles. These four panels answer four different
    questions on **one** shared coronal slice:

    1. does the local diffusion field support the expected CST orientation?
    2. is the reconstructed bundle spatially coherent?
    3. were the intended anatomical constraints applied?
    4. is the final tract anatomically plausible?

    One figure, one owner
    ---------------------
    The strip is a single PNG placed by CSS at :data:`QC_STRIP_WIDTH_MM`, so
    this function creates the Figure and the four Axes and draws into them with
    the ``csttool.viz.render`` primitives. It does **not** call the standalone
    ``qc_figures.plot_*_panel`` functions: each of those owns a Figure and saves
    it, which cannot be composed onto one canvas. The cost is four render calls
    per panel; the alternative — refactoring five reviewed standalone figures to
    accept an axes — is a larger change that would touch tested figures for no
    benefit here.

    That ownership extends to **every panel-level annotation**: titles, the DEC
    direction key, the density colourbar, the ROI key and the hemisphere counts
    are all drawn here, and the HTML template contributes only the section
    heading. Only the Figure knows where the panels actually landed, so only the
    Figure can align a key to a panel column; and a key drawn from
    ``style.LEFT`` / ``style.BRAINSTEM`` / ``style.DENSITY_CMAP`` cannot drift
    from the marks it describes, which a CSS copy can. This is the same rule
    ``plot_profile_matrix`` already follows.

    Layout
    ------
    Four columns, each carrying — top to bottom — a 9 pt bold title, its image,
    and its own key. One shared caption spans the strip beneath them, carrying
    only what is true of all four panels: the plane, the rule that chose it, the
    display slab and the orientation convention. Panel-specific facts live in
    panel-specific keys, so no reader has to cross the figure to decode a mark.

    Geometry is computed by :func:`qc_strip_geometry` from the displayed slice's
    aspect, which is what keeps the furniture bands meaningful — see that
    function for the defect this replaces.

    One slice
    ---------
    The slice is chosen **once**, by :func:`csttool.viz.geometry.select_qc_slice`,
    and every panel gets it. The standalone panels each select their own, and
    two of them are called with ``density=None`` and so fall through to the
    level-4 anatomical fallback while a third gets level 1 — meaning the
    "shared slice" has never actually held outside a prototype driver. It holds
    here.

    ``density_left`` / ``density_right`` are deliberately not plumbed. Level 2
    (``surviving_hemisphere``) exists for the case where exactly one hemisphere
    is non-empty — but then the *bilateral* density volume is, by construction,
    identical to the surviving hemisphere's, so level 1 already selects the
    correct plane. Passing per-hemisphere volumes would add two persisted
    products to reach an unreachable branch. Please do not "fix" this.

    Degradation
    -----------
    Every missing input degrades its own panel to a labelled grayscale-FA slot.
    The strip always renders four panels: it never reflows to three, because a
    reader must be able to see *which* question went unanswered.

    Parameters
    ----------
    fa_path : path
        Required. Supplies the grid, the backgrounds and the level-4 slice
        fallback.
    v1_path, density_path, roi_dseg_path : path or None
        Optional persisted products, each on the FA grid. A volume that is
        present but on a different grid raises ``ValueError`` rather than being
        resampled silently.
    cst_left_path, cst_right_path : path or None
        The two persisted tractograms.
    slab_mm : float
        Physical slab thickness for panels 3 and 4.
    max_streamlines : int
        Per-hemisphere subsample cap for panel 4.
    seed : int, optional
        Seed for that subsample. Defaults to
        :data:`csttool.reproducibility.context.DEFAULT_SEED`, which keeps this
        panel independent of figure subsampling. (``viz.utils.VIZ_SEED`` was
        itself randomised per process until it moved to
        :func:`~csttool.reproducibility.context.derive_seed`, which made the
        legacy panel irreproducible across runs; both are stable now.)

    Returns
    -------
    pathlib.Path
        The saved PNG. A JSON sidecar beside it records the slice, the rule that
        chose it, the slab, the counts, the density ``vmax``, the per-ROI slab
        voxel counts and which panels degraded.
    """
    import json

    import nibabel as nib
    from dipy.io.streamline import load_tractogram

    from csttool.reproducibility.context import DEFAULT_SEED
    from csttool.viz import render as _render

    if seed is None:
        seed = DEFAULT_SEED

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fa_img = nib.load(str(fa_path))
    affine = fa_img.affine
    fa = fa_img.get_fdata().astype(np.float32)

    v1_img = _load_optional(v1_path, fa.shape, affine, "V1 field")
    density_img = _load_optional(density_path, fa.shape, affine, "CST density")
    dseg_img = _load_optional(roi_dseg_path, fa.shape, affine, "ROI segmentation")

    density = density_img.get_fdata().astype(np.float32) if density_img is not None else None
    dseg = np.asarray(dseg_img.dataobj) if dseg_img is not None else None

    def _load_streamlines(path):
        if path is None or not Path(path).exists():
            return []
        try:
            return list(load_tractogram(str(path), 'same').streamlines)
        except Exception:
            return []

    left = _load_streamlines(cst_left_path)
    right = _load_streamlines(cst_right_path)

    # One slice for all four panels. The ROI dseg feeds the level-3 fallback, so
    # a run whose extraction produced no streamlines at all is still shown at a
    # plane its own targets occupy.
    roi_masks = None
    if dseg is not None:
        roi_masks = [dseg == value for value in (1, 2, 3)]
    slice_index, provenance = _geo.select_qc_slice(
        density=density, roi_masks=roi_masks,
        brain_mask=(fa > 0).astype(np.uint8), affine=affine, view="coronal",
    )

    # ---- canvas ----------------------------------------------------------
    # Geometry first, from the displayed slice's shape: the image boxes are
    # computed rather than allocated, so apply_aspect never moves them and the
    # furniture bands below hold what they were measured for.
    canvas_h, canvas_w = _geo.slice_2d(fa, "coronal", slice_index).shape
    geom = qc_strip_geometry(canvas_w, canvas_h)
    width_mm, height_mm = geom["width_mm"], geom["height_mm"]
    panel_mm, image_mm = geom["panel_w_mm"], geom["image_h_mm"]
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))

    axes = []
    for column in range(4):
        axes.append(fig.add_axes([
            geom["panel_x0_mm"][column] / width_mm,
            geom["image_y0_mm"] / height_mm,
            panel_mm / width_mm,
            image_mm / height_mm,
        ]))

    degraded = []

    def _background(ax):
        return _render.render_scalar_slice(
            ax, fa, affine, "coronal", slice_index,
            cmap=_style.ANATOMY_BG, norm=plt.Normalize(0, 1), markers=False,
        )

    def _title(ax, text):
        # 9 pt, matching the profile matrix's subplot titles. The pad is inside
        # _STRIP_TITLE_MM, which is the rendered height of this type plus it.
        ax.set_title(text, fontsize=_STRIP_TITLE_PT, fontweight='bold', pad=2.0)

    def _unavailable(ax, title, note):
        # The title names the panel; it never reports status. A reader must be
        # able to see which of the four questions went unanswered without the
        # heading row changing length from subject to subject.
        _background(ax)
        _title(ax, title)
        ax.text(0.5, 0.06, note, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=_STRIP_NOTE_PT, style='italic', color='#dddddd')

    # ---- panel 1: DEC-FA -------------------------------------------------
    if v1_img is not None:
        v1 = v1_img.get_fdata(dtype=np.float32)
        if v1.ndim == 5:  # the stored product is (X, Y, Z, 1, 3)
            v1 = v1.squeeze(axis=3)
        # |V1_world| * clip(FA, 0, 1) — dipy.reconst.dti.color_fa's formula
        # applied to the already-rotated field. Calling color_fa directly would
        # re-apply its voxel-frame assumption. No gamma and no percentile
        # stretch: brightness *is* FA, so a dark panel is a real finding.
        dec = np.abs(v1) * np.clip(fa, 0, 1)[..., None]
        dec = np.clip(dec, 0.0, 1.0).astype(np.float32)
        _render.render_rgb_slice(axes[0], dec, affine, "coronal", slice_index,
                                 markers=False)
        # The direction key is drawn in this panel's column of the key band
        # below, not as an inset over the image. The arrow glyph it replaces was
        # a 6.1 x 6.1 mm opaque box covering ~3.5% of the panel, in the
        # inferior-lateral corner the CST descends through; its own docstring
        # concedes it cannot be shrunk below 0.22 in and stay legible, so it had
        # to leave the image rather than get smaller. render.add_direction_legend
        # stays in place for the standalone 90 mm panel, where 6 mm is
        # proportionate.
        _title(axes[0], "DEC-FA")
    else:
        _unavailable(axes[0], "DEC-FA", "world-frame V1 not produced")
        degraded.append("dec_fa")

    # ---- panel 2: CST density -------------------------------------------
    vmax = None
    density_image = None
    if density is not None:
        nonzero = density[density > 0]
        # Subject-adaptive: measured maxima are ~0.10 and ~0.29 on the two
        # validation subjects, so a fixed [0, 1] scale would render both panels
        # nearly uniformly dark. The value is printed in the colourbar label, so
        # the scale is never anonymous. No log or power stretch: that would be a
        # second normalization on top of the persisted definition and would make
        # two subjects incomparable.
        vmax = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
        _background(axes[1])
        density_image = _render.render_density_overlay(
            axes[1], density, affine, "coronal", slice_index,
            cmap=_style.DENSITY_CMAP, vmax=vmax,
        )
        _title(axes[1], "CST density")
    else:
        _unavailable(axes[1], "CST density", "density map not produced")
        degraded.append("density")

    # ---- panel 3: extraction ROIs ---------------------------------------
    roi_slab_voxels = {}
    if dseg is not None:
        _background(axes[2])
        roi_colors = (
            (1, "brainstem", _style.BRAINSTEM),
            (2, "motor_left", _style.MOTOR_LEFT),
            (3, "motor_right", _style.MOTOR_RIGHT),
        )
        for value, name, color in roi_colors:
            projected, n_voxels = _slab_projected_volume(
                dseg == value, affine, "coronal", slice_index, slab_mm
            )
            roi_slab_voxels[name] = n_voxels
            if n_voxels:
                # Contours, not fills: three opaque blobs on a 48 mm panel would
                # hide the anatomy they exist to be checked against.
                _render.render_mask_contour(
                    axes[2], projected, affine, "coronal", slice_index,
                    color=color, linewidth=_STRIP_ROI_LINEWIDTH,
                )
        _title(axes[2], "Extraction ROIs")
    else:
        _unavailable(axes[2], "Extraction ROIs", "ROI segmentation not produced")
        degraded.append("roi_dseg")

    # ---- panel 4: final CST over FA -------------------------------------
    _background(axes[3])
    rng = np.random.default_rng(seed)
    for streamlines, color in ((left, _style.LEFT), (right, _style.RIGHT)):
        _render.render_streamline_overlay(
            axes[3], streamlines, affine, "coronal", slice_index, color=color,
            thickness_mm=slab_mm, max_streamlines=max_streamlines, rng=rng,
        )
    _title(axes[3], "CST over FA")
    if not left and not right:
        degraded.append("streamlines")

    # ---- shared geometry -------------------------------------------------
    # Every panel is the same view of the same grid, so one canvas gives all
    # four identical world limits: 1 mm of brain is 1 mm of paper everywhere,
    # and the four are directly comparable by eye.
    #
    # When the image-height cap binds, the panel box is wider than the data, so
    # the common canvas is widened to the box's aspect and the difference prints
    # as the black letterbox this helper draws. Without this the box and the
    # data would disagree and apply_aspect would start moving boxes again.
    canvas_w_eff, canvas_h_eff = canvas_w, canvas_h
    if geom["box_aspect"] > geom["aspect"]:
        canvas_w_eff = canvas_h * geom["box_aspect"]
    elif geom["box_aspect"] < geom["aspect"]:
        canvas_h_eff = canvas_w / geom["box_aspect"]
    for ax in axes:
        _geo.pad_axes_to_canvas(ax, canvas_w_eff, canvas_h_eff)
        # Markers, not legends: a reader must not have to look at a neighbouring
        # panel to orient the one they are reading.
        _geo.add_lr_markers(ax, fontsize=_STRIP_MARKER_PT)

    # ---- per-panel key band ----------------------------------------------
    # One key column per panel, aligned to that panel's own computed x-extent,
    # so every legend sits under the data it describes. Draw once first: the
    # renderer is what _draw_key_row measures against, and by this point the
    # image boxes are final (they were computed, so apply_aspect changed
    # nothing).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _key_axes(column):
        return fig.add_axes([
            geom["panel_x0_mm"][column] / width_mm,
            geom["key_y0_mm"] / height_mm,
            panel_mm / width_mm,
            _STRIP_KEY_MM / height_mm,
        ])

    key_axes = [_key_axes(column) for column in range(4)]
    for ax in key_axes:
        ax.set_axis_off()

    # Panel 1: which world axis each colour channel is. Not a legend — it is the
    # whole content of a DEC image.
    rows = [list(_DEC_AXIS_KEY) if v1_img is not None else []]

    # Panel 2's label row is measured as a single centred token so it takes part
    # in the common fit; its two numeric ticks flank the bar and are short.
    rows.append([(None, "streamline fraction")] if density_image is not None else [])

    # Panel 3: the ROI colour key, beside the contours it names. An ROI the
    # display slab never reaches is greyed rather than dropped, so a zero count
    # is visible instead of silently absent.
    if dseg is not None:
        rows.append([
            (_style.BRAINSTEM if roi_slab_voxels.get("brainstem") else _STRIP_INK_MUTED,
             "brainstem",
             _STRIP_INK if roi_slab_voxels.get("brainstem") else _STRIP_INK_MUTED),
            (_style.MOTOR_LEFT if roi_slab_voxels.get("motor_left") else _STRIP_INK_MUTED,
             "motor L",
             _STRIP_INK if roi_slab_voxels.get("motor_left") else _STRIP_INK_MUTED),
            (_style.MOTOR_RIGHT if roi_slab_voxels.get("motor_right") else _STRIP_INK_MUTED,
             "motor R",
             _STRIP_INK if roi_slab_voxels.get("motor_right") else _STRIP_INK_MUTED),
        ])
    else:
        rows.append([])

    # Panel 4: the hemisphere counts, beside the trajectories they count.
    rows.append([
        (_style.LEFT, f"Left n={len(left)}"),
        (_style.RIGHT, f"Right n={len(right)}"),
    ])

    # One size for the whole band, set by whichever row is tightest.
    key_pt = _fit_key_fontsize(key_axes[0], renderer, rows, _STRIP_KEY_PT)

    # The label row is the alignment anchor for all four columns: swatches for
    # panels 1, 3 and 4, the quantity name for panel 2. One shared baseline,
    # placed so the cap heights meet the band's top pad, and derived from the
    # fitted type size rather than fixed so the row holds whatever the auto-fit
    # settles on.
    text_h_mm = key_pt * 25.4 / 72.0
    cap_mm = _STRIP_KEY_CAP_FRAC * text_h_mm
    label_baseline_mm = _STRIP_KEY_MM - _STRIP_KEY_TEXT_PAD_MM - cap_mm
    label_y = label_baseline_mm / _STRIP_KEY_MM

    _draw_key_row(key_axes[0], renderer, rows[0], key_pt, y=label_y)
    _draw_key_row(key_axes[2], renderer, rows[2], key_pt, y=label_y)
    _draw_key_row(key_axes[3], renderer, rows[3], key_pt, y=label_y)

    # Panel 2: a continuous scale is not a set of swatches, so this column gets
    # the colourbar. Its quantity name sits on the shared label row with the
    # other three keys and the bar hangs beneath it, so the four columns still
    # read as one row of legends. The name is static and the number is dynamic,
    # so they are set separately: as one string the label measured 47.45 mm
    # inside a 48.05 mm column and overflowed into the neighbouring panels as
    # soon as vmax reached two integer digits.
    if density_image is not None:
        key_ax = key_axes[1]
        key_ax.text(0.5, label_y, "streamline fraction",
                    transform=key_ax.transAxes, ha='center', va='baseline',
                    fontsize=key_pt, color=_STRIP_INK)

        # The bar's width is what is left after the two end ticks have their
        # room, so a vmax needing more digits narrows the bar instead of
        # pushing a number into panel 1 or panel 3.
        tick_texts = ("0", f"{vmax:.3g}")
        tick_w_mm = []
        for text in tick_texts:
            probe = key_ax.text(0, 0, text, fontsize=key_pt)
            tick_w_mm.append(probe.get_window_extent(renderer).width
                             / fig.dpi * 25.4)
            probe.remove()
        gap_mm = _STRIP_SWATCH_GAP_MM
        # The bar is centred in the column, so each tick has (panel - bar) / 2
        # to live in and the *wider* of the two is what binds — using their sum
        # lets the longer one hang past the column edge.
        allowed = panel_mm - 2 * gap_mm - 2 * max(tick_w_mm)
        bar_w = min(allowed, _STRIP_CBAR_MAX_FRAC * panel_mm)
        bar_w = max(bar_w, _STRIP_CBAR_MIN_FRAC * panel_mm)
        # A vmax wide enough to fight the minimum takes the bar down with it
        # rather than clipping: a short bar is legible, a truncated number is not.
        bar_w = max(min(bar_w, allowed), 0.15 * panel_mm)

        bar_top_mm = label_baseline_mm - _STRIP_CBAR_GAP_MM
        bar_x0_mm = geom["panel_x0_mm"][1] + (panel_mm - bar_w) / 2.0
        cbar_ax = fig.add_axes([
            bar_x0_mm / width_mm,
            (geom["key_y0_mm"] + bar_top_mm - _STRIP_CBAR_MM) / height_mm,
            bar_w / width_mm,
            _STRIP_CBAR_MM / height_mm,
        ])
        cbar = fig.colorbar(density_image, cax=cbar_ax, orientation='horizontal',
                            ticks=[])
        cbar.outline.set_linewidth(0.4)

        # Ticks flank the bar, on a baseline set so their cap height straddles
        # the bar's middle, so bar and endpoints read as one object rather than
        # three stacked lines.
        bar_lo = (panel_mm - bar_w) / 2.0
        tick_baseline_mm = bar_top_mm - _STRIP_CBAR_MM / 2.0 - cap_mm / 2.0
        tick_y = tick_baseline_mm / _STRIP_KEY_MM
        for x_mm, text, align in (
            ((bar_lo - gap_mm) / panel_mm, tick_texts[0], 'right'),
            ((bar_lo + bar_w + gap_mm) / panel_mm, tick_texts[1], 'left'),
        ):
            key_ax.text(x_mm, tick_y, text, transform=key_ax.transAxes,
                        ha=align, va='baseline', fontsize=key_pt,
                        color=_STRIP_INK)

    # ---- shared caption --------------------------------------------------
    caption_ax = fig.add_axes([0.0, geom["caption_y0_mm"] / height_mm, 1.0,
                               _STRIP_CAPTION_MM / height_mm])
    caption_ax.set_axis_off()
    caption_ax.text(
        0.5, 0.5, _strip_caption_text(slice_index, provenance, slab_mm),
        transform=caption_ax.transAxes, ha='center', va='center',
        fontsize=_STRIP_CAPTION_PT, color=_STRIP_INK,
    )

    fig_path = output_dir / f"{subject_id}_report_qc_strip.png"
    # Exact canvas, not a tight bbox — see plot_profile_matrix.
    _style.save_figure(fig, fig_path, bbox_inches=fig.bbox_inches)
    plt.close(fig)

    sidecar = {
        "Subject": str(subject_id),
        "Panel": "report-qc-strip",
        "Panels": ["DEC-FA", "CST density", "Extraction ROIs", "CST over FA"],
        "SliceIndex": int(slice_index),
        "SliceSelectionRule": provenance.get("rule"),
        "SliceSelectionProvenance": provenance,
        "SlabThicknessMm": float(slab_mm),
        "StreamlineCounts": {"left": len(left), "right": len(right)},
        "StreamlinesDrawn": {"left": min(len(left), max_streamlines),
                             "right": min(len(right), max_streamlines)},
        "DensityVmax": vmax,
        "RoiSlabVoxelCounts": roi_slab_voxels,
        "DegradedPanels": degraded,
        "Seed": int(seed),
        # The figure size is derived from the displayed slice's aspect, so it is
        # disclosed rather than assumed to be the module constant.
        "FigureSizeMm": [round(width_mm, 3), round(height_mm, 3)],
        "PanelAspect": round(geom["aspect"], 6),
    }
    fig_path.with_suffix('.json').write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding='utf-8'
    )
    print(f"✓ Report QC strip saved: {fig_path}")
    return fig_path


def _qc_slice_index(background_image, slice_type):
    """Center slice index for a QC view (axial slightly above center for IC).

    .. deprecated::
        Magic-constant slice selection retained **only** for the legacy
        :func:`plot_tractogram_qc_triptych` / :func:`plot_tractogram_qc_preview`
        path, which must keep producing byte-identical figures until the
        replacement panels pass scientific review (visualization-refactoring-plan
        §2.9). New figures use :func:`viz.geometry.select_qc_slice`, which makes a
        data-driven choice with disclosed provenance. Do **not** call this from
        new code.
    """
    shape = background_image.shape
    if slice_type == 'axial':
        return shape[2] // 2 + 5
    if slice_type == 'sagittal':
        return shape[0] // 2
    return shape[1] // 2  # coronal


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
    bg_slice = _geo.slice_2d(background_image, slice_type, slice_idx)

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
    bg_slice = _geo.slice_2d(background_image, slice_type, slice_idx)
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
    Composite 1x3 tractogram QC figure.

    .. deprecated::
        Superseded in the PDF report by :func:`plot_report_qc_strip`, which
        answers four different questions on one shared slice instead of the same
        question (streamlines over FA) from three angles. This function is
        **not** removed: it stays exported and tested, and `generate_complete_report`
        still falls back to it for a caller that supplies no product paths. New
        code should use the strip.

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
        h, w = _geo.slice_2d(background_image, view, idx).shape
        canvas_h, canvas_w = max(canvas_h, h), max(canvas_w, w)

    last_im = None
    for ax, view in zip(axes, views):
        last_im = _render_qc_slice(
            ax, streamlines_left, streamlines_right, background_image, affine,
            view, max_streamlines=max_streamlines, norm=norm, rng=rng,
            slice_idx=(slice_indices or {}).get(view),
        )
        _geo.pad_axes_to_canvas(ax, canvas_w, canvas_h)
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