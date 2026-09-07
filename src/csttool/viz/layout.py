"""
layout.py - csttool physical (millimetre) figure layout primitives.

This module owns *where things go on a page*, in real units. It is the third
member of the viz triad:

* ``viz.geometry``  - coordinate/orientation logic (voxel <-> world, radiological
  convention, slicing). Knows about anatomy, knows nothing about paper.
* ``viz.render``    - draws one layer into one existing Axes. Creates no Figure,
  saves nothing.
* ``viz.layout``    - sizes Figures and Axes in millimetres, lays out panel rows,
  and draws the shared legend/key furniture. Creates Figures, saves nothing.
* ``viz.style``     - colours, colormaps, type policy, and the single savefig
  policy.

Why millimetres
---------------
A figure authored in inches at an arbitrary size and then scaled by
``\\includegraphics[width=...]`` prints its type at whatever the scale factor
happens to be. A figure authored at its final physical width prints 1 point of
Matplotlib type as 1 printed point, so the declared point sizes in the source
*are* the sizes on paper and can be reasoned about directly. The one-page report
has worked this way since the profile matrix was pinned to
``PROFILE_MATRIX_SIZE_MM``; this module is that approach extracted so every
figure - operational QC or publication - can use it.

**No document-specific dimensions live here.** Callers pass the width they need.
The report passes its content width; a thesis driver passes its text width. This
module supplies the machinery, never the page.

Composition vs. primitives
--------------------------
Nothing in this module decides which panels a figure has or what they contain.
It answers "given N panels of this data aspect on a canvas this wide, where does
each one go, and how large may its key be?" Figure composition - panel choice,
order, titles, final dimensions - belongs to the caller.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from csttool.viz import geometry as _geo

MM_PER_IN = 25.4


# ---------------------------------------------------------------------------
# Unit conversion and physically-sized Figures/Axes
# ---------------------------------------------------------------------------
def mm_to_in(mm):
    """Millimetres to inches (Matplotlib's native unit)."""
    return float(mm) / MM_PER_IN


def in_to_mm(inches):
    """Inches to millimetres."""
    return float(inches) * MM_PER_IN


def figure_mm(width_mm, height_mm, **kwargs):
    """Create a Figure whose canvas is exactly ``width_mm`` x ``height_mm``.

    Use with :func:`csttool.viz.style.save_figure_exact` so the saved raster is
    the size that was laid out. Saving with ``bbox_inches='tight'`` instead
    crops and pads by an unpredictable amount and silently changes the printed
    dimensions, which is precisely what physical sizing exists to prevent.
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError(
            f"figure size must be positive, got {width_mm} x {height_mm} mm")
    return plt.figure(figsize=(mm_to_in(width_mm), mm_to_in(height_mm)), **kwargs)


def axes_mm(fig, x0_mm, y0_mm, w_mm, h_mm, **kwargs):
    """Add an Axes positioned in millimetres from the figure's bottom-left.

    ``Figure.add_axes`` takes a rectangle in figure fractions, which means every
    caller that thinks in millimetres divides by the figure size at the call
    site. Doing it once here keeps the arithmetic in one place and lets a layout
    dict from :func:`panel_row_geometry` be consumed directly.
    """
    fig_w_mm = in_to_mm(fig.get_size_inches()[0])
    fig_h_mm = in_to_mm(fig.get_size_inches()[1])
    return fig.add_axes([
        x0_mm / fig_w_mm,
        y0_mm / fig_h_mm,
        w_mm / fig_w_mm,
        h_mm / fig_h_mm,
    ], **kwargs)


# ---------------------------------------------------------------------------
# Type scale
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TypeScale:
    """Point sizes for one figure family, at final print size.

    Frozen because a figure family's type scale is a design decision, not
    per-call state: a caller wanting different sizes declares a new scale rather
    than mutating a shared one, so two figures can never drift apart by accident.
    """

    title: float
    label: float
    tick: float
    legend: float
    key: float
    caption: float
    note: float
    marker: float

    def scaled(self, factor):
        """The same scale multiplied by ``factor``.

        For a figure family that shares proportions with an existing one but is
        set on a wider or narrower measure.
        """
        return TypeScale(*[getattr(self, f.name) * factor
                           for f in self.__dataclass_fields__.values()])


#: The one-page report's type scale. These are the values the report figures
#: have been reviewed at; they are the default for anything sharing that idiom.
#: Region/positional labels are deliberately the smallest type - they are
#: context, never headings.
REPORT_TYPE = TypeScale(
    title=9.0, label=7.5, tick=7.0, legend=7.5,
    key=6.5, caption=6.5, note=6.0, marker=6.0,
)


# ---------------------------------------------------------------------------
# Key-row metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KeyMetrics:
    """Millimetre metrics for a row of ``swatch + label`` legend entries."""

    swatch_mm: float = 2.2       # length of one colour swatch line
    swatch_gap_mm: float = 0.9   # swatch -> its own label
    entry_gap_mm: float = 1.9    # between one entry and the next
    cap_frac: float = 0.72       # cap height / em, for DejaVu Sans
    min_pt: float = 5.5          # floor for the auto-fit
    ink: str = '#333a45'         # default label colour


REPORT_KEY = KeyMetrics()


# ---------------------------------------------------------------------------
# Vertical band stack for a row of panels
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelRowBands:
    """The vertical budget of a panel row, in millimetres, top to bottom.

    Every panel in the row is the same zones in the same order - title, image,
    fixed-height key - so N columns align band for band whatever their keys
    contain. A band of 0.0 removes that zone.

    ``key_caption_gap_mm`` is deliberately the larger of the two gaps and larger
    than ``image_key_gap_mm``: a shared caption describes the whole composite and
    must not read as one more legend under the last panel. Proximity is the only
    thing distinguishing them, since both are centred type at the same size.
    """

    margin_top_mm: float = 2.6
    title_mm: float = 4.0
    image_key_gap_mm: float = 1.4
    key_mm: float = 6.2
    key_caption_gap_mm: float = 2.6
    caption_mm: float = 3.2
    margin_bottom_mm: float = 1.4
    gap_mm: float = 2.6          # horizontal, between panels

    @property
    def furniture_mm(self):
        """Total vertical space the bands consume, excluding the image."""
        return (self.margin_top_mm + self.title_mm + self.image_key_gap_mm
                + self.key_mm + self.key_caption_gap_mm + self.caption_mm
                + self.margin_bottom_mm)


#: The report QC strip's bands, tuned against its A4 page budget.
REPORT_STRIP_BANDS = PanelRowBands()


def panel_row_geometry(canvas_w, canvas_h, *, width_mm, n_panels,
                       bands=REPORT_STRIP_BANDS, image_max_mm,
                       min_height_mm=None, max_height_mm=None):
    """Figure size and panel rectangles for a row of ``n_panels`` image panels.

    The whole layout is a pure function of the displayed slice's shape, computed
    here so the caller can ``set_position`` each Axes to a rectangle whose aspect
    already equals the data's. That is what makes ``Axes.apply_aspect`` a no-op:
    Matplotlib's default ``adjustable='box'`` otherwise shrinks and re-centres an
    ``aspect='equal'`` image box at draw time, and every annotation anchored to
    the Axes moves with it while every annotation anchored to a precomputed
    millimetre band does not.

    Two regimes, both handled:

    * **wide grids** (a 96x96x60 volume gives aspect 1.6) are width-bound, so the
      image height falls out of the aspect and fills the panel exactly;
    * **tall grids** would want more height than ``image_max_mm``, so the height
      is capped and the image letterboxes horizontally inside a panel that keeps
      its full width - which is what
      :func:`csttool.viz.geometry.pad_axes_to_canvas` exists for.

    The panel width is deliberately **constant**. Narrowing it to keep the image
    flush would be the obvious alternative, but the key band underneath is
    width-critical: a narrowed panel makes the keys and eventually the titles
    collide. Trading image area for a key that fits is the right way round.

    Parameters
    ----------
    canvas_w, canvas_h : int
        Width and height of the displayed slice in voxels, i.e. the reversed
        shape of :func:`csttool.viz.geometry.slice_2d`.
    width_mm : float
        Final printed width of the whole row. Supplied by the caller; this
        module holds no document dimensions.
    n_panels : int
        Number of equal-width columns.
    bands : PanelRowBands
        The vertical budget.
    image_max_mm : float
        Largest image height to spend before letterboxing.
    min_height_mm, max_height_mm : float, optional
        Clamp on the derived figure height. Slack from the floor is split
        between the two margins so a very wide grid centres its content rather
        than hanging from the top.

    Returns
    -------
    dict
        ``width_mm``/``height_mm`` (the figure), ``panel_w_mm``/``image_h_mm``
        (one panel's image box), ``panel_x0_mm`` (the left edges), the band
        origins ``image_y0_mm``/``key_y0_mm``/``caption_y0_mm``, ``aspect`` (the
        data's) and ``box_aspect`` (the panel box's; they differ only when the
        image-height cap binds, and the difference is the letterbox).
    """
    if n_panels < 1:
        raise ValueError(f"n_panels must be >= 1, got {n_panels}")
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError(f"canvas must be positive, got {canvas_w}x{canvas_h}")

    aspect = float(canvas_w) / float(canvas_h)
    furniture_mm = bands.furniture_mm

    panel_w = (width_mm - (n_panels - 1) * bands.gap_mm) / float(n_panels)
    image_h = min(panel_w / aspect, image_max_mm)

    height_mm = furniture_mm + image_h
    # Unbounded on a side that was not given. Defaulting the ceiling to the
    # natural height instead would let it silently cancel a floor supplied on
    # its own, which is the one combination a caller is most likely to use.
    lo = -float("inf") if min_height_mm is None else min_height_mm
    hi = float("inf") if max_height_mm is None else max_height_mm
    if lo > hi:
        raise ValueError(
            f"min_height_mm ({lo}) exceeds max_height_mm ({hi})")
    clamped = min(max(height_mm, lo), hi)
    slack = max(0.0, clamped - height_mm)
    height_mm = clamped
    margin_bottom = bands.margin_bottom_mm + slack / 2.0

    # Stacked from the bottom, with the two gaps as real bands rather than
    # implied by the type's own leading.
    caption_y0 = margin_bottom
    key_y0 = caption_y0 + bands.caption_mm + bands.key_caption_gap_mm
    image_y0 = key_y0 + bands.key_mm + bands.image_key_gap_mm

    row_w = n_panels * panel_w + (n_panels - 1) * bands.gap_mm
    x_start = (width_mm - row_w) / 2.0
    panel_x0 = [x_start + i * (panel_w + bands.gap_mm) for i in range(n_panels)]

    return {
        "width_mm": width_mm,
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


# ---------------------------------------------------------------------------
# Key rows and captions
# ---------------------------------------------------------------------------
def fit_key_fontsize(ax, renderer, rows, start_pt, *, floor_pt=None,
                     metrics=REPORT_KEY):
    """The largest type at or below ``start_pt`` at which every key row fits.

    One size for all columns, not one per column: a key band whose columns
    disagree about type size reads as several unrelated captions rather than one
    row. The widest row therefore decides, and the whole band tracks it.

    Swatch and gap widths are fixed millimetres and text width scales with point
    size, so the fit is solved directly from a single measurement per row rather
    than by iterating.
    """
    floor_pt = metrics.min_pt if floor_pt is None else floor_pt
    box_w = ax.get_window_extent(renderer).width
    px_per_mm = ax.figure.dpi / MM_PER_IN
    scale = 1.0
    for entries in rows:
        if not entries:
            continue
        fixed = sum((metrics.swatch_mm + metrics.swatch_gap_mm) * px_per_mm
                    for color, *_ in entries if color)
        fixed += metrics.entry_gap_mm * px_per_mm * (len(entries) - 1)
        text = 0.0
        for entry in entries:
            probe = ax.text(0, 0, entry[1], fontsize=start_pt)
            text += probe.get_window_extent(renderer).width
            probe.remove()
        if text <= 0:
            continue
        scale = min(scale, max(0.0, box_w - fixed) / text)
    return max(floor_pt, min(start_pt, start_pt * scale))


def fit_caption_fontsize(ax, renderer, text, start_pt, *, floor_pt=5.5):
    """The largest type at or below ``start_pt`` at which the caption fits.

    Text width scales with point size, so the fit is solved from one measurement
    rather than by iterating. Normally a no-op, but a caption line grows with
    whatever provenance it carries and there is no combination the figure may
    clip.
    """
    box_w = ax.get_window_extent(renderer).width
    probe = ax.text(0, 0, text, fontsize=start_pt)
    text_w = probe.get_window_extent(renderer).width
    probe.remove()
    if text_w <= 0 or text_w <= box_w:
        return start_pt
    return max(floor_pt, start_pt * box_w / text_w)


def draw_key_row(ax, renderer, entries, fontsize, *, y, metrics=REPORT_KEY):
    """Lay out one centred row of ``swatch + label`` pairs for a single panel.

    This is the shared legend primitive: a panel gets a row of it in its own
    column of the key band, so a reader decodes a contour or a trajectory
    without leaving the panel. A continuous scale cannot use it - that needs a
    colourbar in the same band.

    Each token is measured and placed in sequence because Matplotlib has no rich
    text and the colour *is* the key. Measurement is a pure function of the text,
    the font and the figure DPI, so the result is reproducible.

    ``entries`` is a sequence of ``(colour, label)`` or
    ``(colour, label, label_colour)``; a ``colour`` of None draws the label with
    no swatch. Overflow is deliberately left visible rather than scaled away - a
    key that does not fit its column is a layout defect and should fail a test,
    not be hidden.

    ``y`` is the row's **baseline** in axes fraction, and the type is set on it
    with ``va='baseline'``. Centring instead centres each string's bounding box,
    so a row of all-caps labels sits ~0.13 mm off a row with ascenders - visible
    as legends that do not quite line up.
    """
    from matplotlib.lines import Line2D

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if not entries:
        return

    entries = [e if len(e) == 3 else (e[0], e[1], metrics.ink) for e in entries]
    fig = ax.figure
    px_per_mm = fig.dpi / MM_PER_IN
    swatch = metrics.swatch_mm * px_per_mm
    swatch_gap = metrics.swatch_gap_mm * px_per_mm
    entry_gap = metrics.entry_gap_mm * px_per_mm

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
    band_mm = box.height / fig.dpi * MM_PER_IN
    cap_mm = metrics.cap_frac * fontsize * MM_PER_IN / 72.0
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


# ---------------------------------------------------------------------------
# Cropping to content
# ---------------------------------------------------------------------------
def content_bbox_2d(slice_2d_array, *, threshold=0.0, margin_frac=0.04,
                    margin_vox=0.0, clamp=True):
    """Display-space bounding box of the non-background part of a 2D slice.

    Returns ``(col_lo, col_hi, row_lo, row_hi)`` in the pixel-index coordinates
    ``imshow(..., origin='lower')`` uses when no ``extent`` is given, i.e.
    directly usable as ``set_xlim`` / ``set_ylim`` arguments. Returns ``None``
    when nothing exceeds ``threshold``, so a caller can fall back to the full
    slice rather than being handed a degenerate box.

    This is a **presentation** operation: it changes which part of an already
    rendered array is visible, and never touches the array, the affine, or any
    voxel value. Anatomically it is the difference between a brain filling its
    panel and a brain sitting in a field of background that carries no
    information but consumes most of the page.

    ``margin_frac`` is a fraction of the box's own extent, so the breathing room
    scales with the structure rather than with the acquisition matrix.
    ``clamp`` keeps the box inside the slice; pass ``False`` only if the caller
    paints its own background out to the box.
    """
    arr = np.asarray(slice_2d_array)
    if arr.ndim == 3:            # RGB(A): any channel above threshold counts
        content = np.any(arr[..., :3] > threshold, axis=-1)
    else:
        content = arr > threshold
    if not content.any():
        return None

    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    row_lo, row_hi = float(rows[0]), float(rows[-1])
    col_lo, col_hi = float(cols[0]), float(cols[-1])

    pad_r = (row_hi - row_lo) * margin_frac + margin_vox
    pad_c = (col_hi - col_lo) * margin_frac + margin_vox

    # Half a voxel out on each side so the outermost voxel is drawn whole.
    x0, x1 = col_lo - 0.5 - pad_c, col_hi + 0.5 + pad_c
    y0, y1 = row_lo - 0.5 - pad_r, row_hi + 0.5 + pad_r

    if clamp:
        # Never show beyond the slice. Where content reaches the array edge the
        # margin has nowhere to go, and an unclamped box exposes bare canvas
        # outside the image - which is not black background, because an image
        # axis has its frame off and paints no patch. Anything placed in axes
        # fractions then lands on it: the R/L markers, drawn white with a dark
        # outline for legibility *against anatomy*, ended up on white paper.
        n_rows, n_cols = content.shape
        x0, x1 = max(x0, -0.5), min(x1, n_cols - 0.5)
        y0, y1 = max(y0, -0.5), min(y1, n_rows - 0.5)

    return (x0, x1, y0, y1)


def union_bbox(boxes):
    """The smallest box containing every non-None box in ``boxes``.

    Panels that are meant to be compared must share one box, or the reader is
    comparing two different magnifications. Callers computing a per-panel box
    across a row should pass them through this first.
    """
    real = [b for b in boxes if b is not None]
    if not real:
        return None
    return (min(b[0] for b in real), max(b[1] for b in real),
            min(b[2] for b in real), max(b[3] for b in real))


def square_bbox(box):
    """Expand ``box`` about its centre to a square, so ``aspect='equal'`` panels
    of differing content still tile evenly. ``None`` passes through."""
    if box is None:
        return None
    x0, x1, y0, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) / 2.0
    return (cx - half, cx + half, cy - half, cy + half)


def apply_bbox(ax, box):
    """Set an Axes' limits to ``box``, preserving any radiological x-inversion.

    :func:`csttool.viz.geometry.enforce_radiological_image` may have inverted the
    x axis; a naive ``set_xlim(lo, hi)`` silently undoes that and mirrors the
    panel relative to its neighbours, with the R/L markers then pointing at the
    wrong hemispheres. The current direction is read back and reapplied.
    ``None`` is a no-op so a caller can pass a failed crop straight through.
    """
    if box is None:
        return
    x0, x1, y0, y1 = box
    if ax.xaxis_inverted():
        ax.set_xlim(x1, x0)
    else:
        ax.set_xlim(x0, x1)
    if ax.yaxis_inverted():
        ax.set_ylim(y1, y0)
    else:
        ax.set_ylim(y0, y1)


def crop_axes_to_content(ax, volume, view, index, *, threshold=0.0,
                         margin_frac=0.04, square=False):
    """Crop ``ax`` to the non-background content of ``volume``'s display slice.

    Convenience wrapper over :func:`content_bbox_2d` + :func:`apply_bbox` for the
    common single-panel case. Returns the box applied, or ``None`` if the slice
    was empty and the axes was left alone.
    """
    box = content_bbox_2d(_geo.slice_2d(volume, view, index),
                          threshold=threshold, margin_frac=margin_frac)
    if square:
        box = square_bbox(box)
    apply_bbox(ax, box)
    return box
