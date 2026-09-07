"""
render.py - composable panel renderers for the report QC figures.

This module owns *the act of drawing a panel*: each function takes an existing
``matplotlib.axes.Axes``, draws one layer into it, and returns a mappable (or
``None``). **None creates a Figure. None saves.** That contract lets the figure
composition layer (``csttool.metrics.modules.qc_figures``) lay panels out on an
exact A4 GridSpec and the test layer assert artist properties without ever
constructing a Figure (plan §6.7, R-5).

Coordinate conventions
-----------------------
Every renderer takes world-space data plus an affine and a voxel slice index;
none accepts pre-sliced 2D arrays, so orientation cannot be lost at a call
boundary. Anatomical images use the radiological convention via
:func:`csttool.viz.geometry.finalize_image_view`.

Determinism
-----------
The only stochastic element anywhere in this layer is streamline subsampling in
:func:`render_streamline_overlay`, seeded through :func:`csttool.viz.utils.viz_rng`
/ ``VIZ_SEED`` exactly as the legacy triptych is. No global NumPy state is
touched.
"""

import numpy as np

from csttool.viz import geometry as _geo
from csttool.viz import style as _style
from csttool.viz.utils import deterministic_subsample, viz_rng


def render_scalar_slice(ax, volume, affine, view, index, *, cmap=None, norm=None,
                        vmin=None, vmax=None, markers=True):
    """Draw one grayscale (or sequential) scalar slice and finalize radiologically.

    ``markers=False`` still enforces the radiological convention but leaves the
    "R"/"L" glyphs to the caller — needed by a composite figure that draws its
    own at a smaller size, or that layers several renderers into one axes and
    would otherwise stamp the markers once per layer.

    Returns the ``AxesImage`` mappable.
    """
    if view not in _geo.VIEW_AXES:
        raise ValueError(f"unknown view {view!r}")
    sl = _geo.slice_2d(volume, view, index)
    im = ax.imshow(sl, cmap=cmap, origin='lower', aspect='equal',
                   norm=norm, vmin=vmin, vmax=vmax)
    ax.axis('off')
    _geo.finalize_image_view(ax, affine, view, markers=markers)
    return im


def render_rgb_slice(ax, rgb_volume, affine, view, index, *, markers=True):
    """Draw one RGB (DEC) slice. ``rgb_volume`` has shape (X, Y, Z, 3) in [0, 1].

    No colormap, no norm: the colour is the data. Returns the ``AxesImage``.
    ``markers`` behaves as in :func:`render_scalar_slice`.
    """
    if view not in _geo.VIEW_AXES:
        raise ValueError(f"unknown view {view!r}")
    rgb = np.asarray(rgb_volume)
    if rgb.shape[-1] != 3:
        raise ValueError(f"rgb_volume last axis must be 3, got {rgb.shape}")
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        raise TypeError(f"rgb_volume must be float, got dtype {rgb.dtype}")
    if rgb.min() < -1e-6 or rgb.max() > 1.0 + 1e-6:
        raise ValueError(
            f"rgb_volume values must lie in [0, 1], got [{float(rgb.min())}, {float(rgb.max())}]"
        )
    # Slice the spatial axes while preserving the trailing channel axis, then
    # transpose to the same (vertical, horizontal) display order the legacy
    # ``_qc_slice_2d`` used for scalar volumes (e.g. coronal -> (Z, X) spatial).
    if view == 'axial':
        sl = rgb[:, :, index, :].transpose(1, 0, 2)
    elif view == 'coronal':
        sl = rgb[:, index, :, :].transpose(1, 0, 2)
    else:  # sagittal
        sl = rgb[index, :, :, :].transpose(1, 0, 2)
    im = ax.imshow(sl, origin='lower', aspect='equal')
    ax.axis('off')
    _geo.finalize_image_view(ax, affine, view, markers=markers)
    return im


def render_density_overlay(ax, density, affine, view, index, *, cmap=None,
                            vmax=None, alpha_floor=0.0):
    """Overlay the density volume on the current axes (over the FA background).

    Voxels at or below ``alpha_floor`` are masked fully transparent so zero-
    density voxels never tint the background (plan §8.2). Returns the
    ``AxesImage``.

    This is an overlay, so it deliberately does not finalize the view — but it
    must not *undo* the caller's finalize either. ``imshow`` resets the axes
    limits, which silently discards the x-inversion
    :func:`csttool.viz.geometry.enforce_radiological_image` applied when the
    background was drawn: the panel would then be displayed in neurological
    convention, mirrored relative to every neighbouring panel, with the "R"/"L"
    markers pointing at the wrong hemispheres. The inversion is therefore
    captured and restored around the ``imshow``.
    """
    if view not in _geo.VIEW_AXES:
        raise ValueError(f"unknown view {view!r}")
    if cmap is None:
        cmap = _style.DENSITY_CMAP
    if vmax is not None and vmax <= 0:
        raise ValueError(f"vmax must be positive, got {vmax}")
    sl = _geo.slice_2d(density, view, index).astype(np.float32)
    masked = np.ma.masked_where(sl <= alpha_floor, sl)
    was_x_inverted = ax.xaxis_inverted()
    was_y_inverted = ax.yaxis_inverted()
    im = ax.imshow(masked, cmap=cmap, origin='lower', aspect='equal',
                   vmin=0.0, vmax=vmax, alpha=1.0)
    if ax.xaxis_inverted() != was_x_inverted:
        ax.invert_xaxis()
    if ax.yaxis_inverted() != was_y_inverted:
        ax.invert_yaxis()
    ax.axis('off')
    return im


def render_streamline_overlay(ax, streamlines, affine, view, index, *, color,
                              thickness_mm=None, max_streamlines=500, rng=None):
    """Overlay streamlines whose points fall in the physical slab, as separate
    per-contiguous-run polylines.

    Fixes the legacy rendering defect (plan §3.3-3) where a streamline leaving
    and re-entering the slab was drawn as one polyline with a spurious straight
    chord across the gap: each contiguous in-slab run is now its own ``ax.plot``.
    Subsampling is deterministic via :func:`csttool.viz.utils.deterministic_subsample`.

    Coordinate frames
    -----------------
    Slab membership is decided in **world millimetres** (that is the whole point
    of a physical slab), but the points are plotted in **voxel** coordinates,
    because the background image drawn by :func:`render_scalar_slice` /
    :func:`render_rgb_slice` is an ``imshow`` whose data coordinates are voxel
    indices. Plotting the world points directly puts the bundle metres away from
    the anatomy it is supposed to overlay, and lets the axes autoscale out to
    contain both.

    Parameters
    ----------
    streamlines : sequence of (N, 3) ndarray, RASMM.
    thickness_mm : float, optional
        Physical slab thickness (plan §6.5). Defaults to
        :data:`csttool.viz.geometry.DEFAULT_SLAB_MM`.
    """
    if view not in _geo.VIEW_AXES:
        raise ValueError(f"unknown view {view!r}")
    if thickness_mm is None:
        thickness_mm = _geo.DEFAULT_SLAB_MM
    if rng is None:
        rng = viz_rng()
    if len(streamlines) == 0:
        return None

    h_axis, v_axis = _geo.VIEW_AXES[view]
    sub = deterministic_subsample(list(streamlines), max_streamlines, rng=rng)
    drawn = 0
    for sl in sub:
        sl = np.asarray(sl, dtype=float)
        if sl.size == 0:
            continue
        in_slab = _geo.slab_membership(sl, affine, view, index, thickness_mm)
        if not in_slab.any():
            continue
        # Membership decided in world mm above; drawn in voxel coordinates, which
        # is the frame the background image lives in.
        sl = _geo.world_to_voxel(sl, affine)
        # Split into contiguous in-slab runs (so a leave-and-re-enter yields two
        # separate polylines, not one chord across the gap).
        runs = []
        run_idx = []
        for i, keep in enumerate(in_slab):
            if keep:
                run_idx.append(i)
            elif run_idx:
                runs.append(run_idx); run_idx = []
        if run_idx:
            runs.append(run_idx)
        for r in runs:
            if len(r) < 2:
                continue
            ax.plot(sl[r, h_axis], sl[r, v_axis], color=color,
                    linewidth=0.5, alpha=0.6)
        drawn += 1
        if drawn >= max_streamlines:
            break
    return None


def render_mask_contour(ax, mask, affine, view, index, *, color='yellow',
                        linewidth=0.8):
    """Draw the 0.5-level contour of a binary mask on the current axes.

    Reserved for the deferred pipeline-focused panels (plan §6.7, §16 D-12).
    """
    if view not in _geo.VIEW_AXES:
        raise ValueError(f"unknown view {view!r}")
    sl = _geo.slice_2d(np.asarray(mask), view, index)
    ax.contour(sl, levels=[0.5], colors=[color], linewidths=linewidth)
    return None


def render_mask_overlay(ax, mask, affine, view, index, *, color,
                        fill_alpha=0.35, outline=True, outline_linewidth=1.2,
                        outline_alpha=1.0):
    """Overlay a binary mask as a translucent fill, optionally outlined.

    **Never routes a binary mask through a colormap.** A mask that reaches
    ``imshow`` as ``cmap='Blues'`` is normalized before it is coloured, and a
    constant-valued array normalizes to the colormap's *low* end - so a mask
    drawn this way renders near-white while the legend swatch beside it shows
    saturated blue, and the legend does not describe the mark. The fill is
    therefore built as an explicit RGBA array at the requested colour, which is
    the colour that appears.

    Fill plus outline is the default because the two do different jobs. The fill
    answers "which region is this?" at a glance, which is what a reader
    identifying an ROI needs; the outline answers "exactly where does it stop?",
    which is what a reader verifying a boundary needs, and it survives a fill
    alpha low enough to leave the underlying anatomy legible. Either can be
    turned off: ``fill_alpha=0`` gives contour-only, ``outline=False`` gives
    fill-only.

    Like :func:`render_density_overlay` this is an overlay, so it does not
    finalize the view - but it must not *undo* the caller's finalize either.
    ``imshow`` resets the axes limits, discarding the x-inversion
    :func:`csttool.viz.geometry.enforce_radiological_image` applied when the
    background was drawn, which would mirror the panel relative to its
    neighbours. The inversion is captured and restored around the draw.

    Returns the ``AxesImage`` for the fill, or ``None`` when ``fill_alpha`` is 0.
    """
    from matplotlib.colors import to_rgba

    if view not in _geo.VIEW_AXES:
        raise ValueError(f"unknown view {view!r}")
    if not 0.0 <= fill_alpha <= 1.0:
        raise ValueError(f"fill_alpha must lie in [0, 1], got {fill_alpha}")

    sl = _geo.slice_2d(np.asarray(mask), view, index)
    binary = sl > 0

    im = None
    if fill_alpha > 0:
        rgba = np.zeros((*binary.shape, 4), dtype=float)
        rgba[binary] = to_rgba(color, fill_alpha)
        was_x_inverted = ax.xaxis_inverted()
        was_y_inverted = ax.yaxis_inverted()
        im = ax.imshow(rgba, origin='lower', aspect='equal',
                       interpolation='nearest')
        if ax.xaxis_inverted() != was_x_inverted:
            ax.invert_xaxis()
        if ax.yaxis_inverted() != was_y_inverted:
            ax.invert_yaxis()

    if outline and binary.any():
        ax.contour(binary.astype(float), levels=[0.5], colors=[color],
                   linewidths=outline_linewidth, alpha=outline_alpha)

    return im


def add_direction_legend(ax, *, loc='lower right', size=0.12):
    """Draw the three-axis RGB direction glyph in the axes corner.

    Red = left-right, green = anterior-posterior, blue = superior-inferior, in
    anatomical world axes (plan §2.3). The glyph is a small axes-inset made of
    three colour-keyed arrows; it replaces a separate legend row in the DEC-FA
    panel.

    ``size`` is the inset's edge length **in inches**, so the glyph is a fixed
    physical size independent of the figure it sits on. Everything — arrows,
    arrowheads and labels — is laid out inside that box and scaled from it, so
    the glyph stays self-contained: the original layout put the labels beyond
    the arrow tips, where at report size they overflowed the box and printed on
    top of the anatomy. Below roughly 0.22 in (5.5 mm) three 3-character labels
    cannot be set legibly, so callers that need a smaller key should not use
    this glyph at all.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset = inset_axes(ax, width=size, height=size, loc=loc,
                       bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0.4)
    # Type and arrowheads scale with the box, so a caller changing `size` gets a
    # proportionally scaled glyph rather than a differently-proportioned one.
    label_pt = max(3.5, 22.0 * size)
    head = max(3.0, 26.0 * size)
    # Origin low-left; the three arrows sweep up, right and down into the box.
    cx, cy = 0.30, 0.40
    inset.annotate('', xy=(cx + 0.36, cy), xytext=(cx, cy),
                   arrowprops=dict(arrowstyle='-|>', color='red', linewidth=0.9,
                                   mutation_scale=head, shrinkA=0, shrinkB=0))
    inset.annotate('', xy=(cx, cy + 0.34), xytext=(cx, cy),
                   arrowprops=dict(arrowstyle='-|>', color='green', linewidth=0.9,
                                   mutation_scale=head, shrinkA=0, shrinkB=0))
    inset.annotate('', xy=(cx, cy - 0.30), xytext=(cx, cy),
                   arrowprops=dict(arrowstyle='-|>', color='blue', linewidth=0.9,
                                   mutation_scale=head, shrinkA=0, shrinkB=0))
    # Labels in the free corners, clear of every arrow.
    inset.text(0.98, 0.62, 'L-R', va='center', ha='right',
               fontsize=label_pt, color='red')
    inset.text(cx + 0.05, 0.98, 'A-P', va='top', ha='left',
               fontsize=label_pt, color='green')
    inset.text(0.98, 0.02, 'S-I', va='bottom', ha='right',
               fontsize=label_pt, color='blue')
    inset.set_xlim(0, 1); inset.set_ylim(0, 1)
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_facecolor('white')
    for s in inset.spines.values():
        s.set_visible(True); s.set_linewidth(0.4)
    return inset
