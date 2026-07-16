"""
geometry.py - spatial/orientation logic for visualizations.

Everything about *where* things are drawn: affine handling, voxel<->world
coordinate transforms, image-display orientation (radiological convention),
slice padding, spatial bounds, and L/R marker placement.

No style policy here (that is ``viz.style``); no pipeline logic.

Display convention (project-wide)
---------------------------------
csttool figures use the **radiological** convention for anatomical images and for
world-coordinate streamline projections: the patient's **anatomical LEFT is drawn
on the viewer's RIGHT**. Helpers here enforce that convention regardless of the
sign of the subject affine's X axis, and place "R"/"L" markers accordingly.
"""

import numpy as np

# View -> (horizontal voxel axis, vertical voxel axis) for a slice built as
# ``volume[..].T`` displayed with ``origin='lower'`` (the csttool convention).
VIEW_AXES = {
    "axial": (0, 1),      # vol[:, :, z].T  -> horizontal = X(0), vertical = Y(1)
    "coronal": (0, 2),    # vol[:, y, :].T  -> horizontal = X(0), vertical = Z(2)
    "sagittal": (1, 2),   # vol[x, :, :].T  -> horizontal = Y(1), vertical = Z(2)
}


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------
def to_ras(data, affine):
    """Reorient a 3D array + affine to RAS+ voxel orientation.

    Returns ``(data_ras, affine_ras)``. Used so a figure's background image can
    be placed in the same voxel grid as data (e.g. ROI masks) that the pipeline
    produced in RAS-reoriented space. This is a pure relabelling of voxel axes
    (no resampling); anatomical content is unchanged.
    """
    import nibabel as nib
    from nibabel.orientations import (
        io_orientation, axcodes2ornt, ornt_transform, apply_orientation,
        inv_ornt_aff,
    )
    src = io_orientation(affine)
    dst = axcodes2ornt(("R", "A", "S"))
    transform = ornt_transform(src, dst)
    data_ras = apply_orientation(np.asarray(data), transform)
    affine_ras = affine @ inv_ornt_aff(transform, np.asarray(data).shape)
    return data_ras, affine_ras


def voxel_to_world(points_vox, affine):
    """Map voxel coordinates (N,3) to world/RASMM coordinates (N,3)."""
    pts = np.asarray(points_vox, dtype=float)
    h = np.c_[pts, np.ones(len(pts))]
    return (h @ affine.T)[:, :3]


def world_to_voxel(points_world, affine):
    """Map world/RASMM coordinates (N,3) to voxel coordinates (N,3)."""
    pts = np.asarray(points_world, dtype=float)
    inv = np.linalg.inv(affine)
    h = np.c_[pts, np.ones(len(pts))]
    return (h @ inv.T)[:, :3]


# ---------------------------------------------------------------------------
# Orientation / radiological convention
# ---------------------------------------------------------------------------
def voxel_axis_world_x_sign(affine, voxel_axis):
    """Sign of world-X change as the given voxel axis increases.

    Returns +1 if increasing this voxel index moves toward anatomical Right
    (world +X in RAS), -1 toward anatomical Left, and 0 if this voxel axis is
    not the dominant left/right axis (e.g. the anterior/superior axes, whose
    X-component is only a small rotation term).
    """
    col = affine[:3, voxel_axis]
    # The X (left/right) component must dominate this voxel axis for it to be a
    # left/right axis; otherwise a tiny rotation term must not be treated as L/R.
    if abs(col[0]) < max(abs(col[1]), abs(col[2])):
        return 0
    if col[0] > 0:
        return 1
    if col[0] < 0:
        return -1
    return 0


def image_is_radiological(affine, horizontal_voxel_axis):
    """True if a voxel image (horizontal = ``horizontal_voxel_axis``, no x-flip)
    already shows anatomical LEFT on the viewer's RIGHT.

    That happens when increasing the horizontal voxel index moves toward
    anatomical Left, i.e. world-X sign is negative.
    """
    return voxel_axis_world_x_sign(affine, horizontal_voxel_axis) < 0


def enforce_radiological_image(ax, affine, view):
    """Ensure an anatomical-image axis is displayed radiologically.

    For axial/coronal views (horizontal axis carries L/R) the x-axis is inverted
    if needed so anatomical Left ends up on the viewer's right. Sagittal views
    carry no L/R and are left untouched.

    Returns True if the view has a left/right axis (i.e. L/R markers are meaningful).
    """
    h_axis, _ = VIEW_AXES[view]
    if voxel_axis_world_x_sign(affine, h_axis) == 0:
        return False  # sagittal: no L/R
    if not image_is_radiological(affine, h_axis):
        ax.invert_xaxis()
    return True


def set_world_x_radiological(ax):
    """Orient a world-coordinate plot whose horizontal axis is world-X so that
    anatomical Left (negative X) is on the viewer's right (radiological).

    World-X increases toward anatomical Right, so radiological = X descending
    left->right, i.e. the x-axis is inverted relative to the natural ordering.
    """
    lo, hi = ax.get_xlim()
    if lo < hi:  # currently ascending (Right on the right) -> invert to radiological
        ax.set_xlim(hi, lo)


def add_lr_markers(ax, fontsize=11, color="white", pad=0.02):
    """Place "R" and "L" markers on an axis that is already radiological.

    Radiological => viewer's LEFT edge is anatomical Right, viewer's RIGHT edge
    is anatomical Left.
    """
    ax.text(pad, 0.5, "R", transform=ax.transAxes, ha="left", va="center",
            fontsize=fontsize, fontweight="bold", color=color,
            path_effects=_outline())
    ax.text(1 - pad, 0.5, "L", transform=ax.transAxes, ha="right", va="center",
            fontsize=fontsize, fontweight="bold", color=color,
            path_effects=_outline())


def finalize_image_view(ax, affine, view, markers=True):
    """One-call radiological finalize for an anatomical-image axis.

    Ensures the view is radiological for its affine and, if the view carries a
    left/right axis (axial/coronal), adds "R"/"L" markers. Returns True if L/R
    markers were added.
    """
    has_lr = enforce_radiological_image(ax, affine, view)
    if has_lr and markers:
        add_lr_markers(ax)
    return has_lr


def finalize_world_plane(ax, horizontal_world_axis, markers=True):
    """One-call radiological finalize for a world-coordinate plot axis.

    If the horizontal axis is world-X (index 0) the axis is oriented radiologically
    (anatomical Left on the viewer's right) and "R"/"L" markers are added.
    Non-X horizontal axes (e.g. sagittal Y-Z) are left untouched. Returns True
    if L/R markers were added.
    """
    if horizontal_world_axis != 0:
        return False
    set_world_x_radiological(ax)
    if markers:
        add_lr_markers(ax, color="black")
    return True


def _outline():
    """A thin dark outline so white L/R markers read on light or dark panels."""
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2, foreground="black")]


# ---------------------------------------------------------------------------
# Slice padding and spatial bounds (moved verbatim from the viz modules; these
# were duplicated in extract/ and tracking/ visualization files).
# ---------------------------------------------------------------------------
def pad_slice_to_square(image_slice, extent=None, pad_value=0.0):
    """Pad a 2D slice to a square shape, returning the padded slice and an updated
    display extent that preserves the original pixel spacing."""
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
        mode="constant",
        constant_values=pad_value,
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
        y_max + pad_y_after * dy,
    )
    return padded, padded_extent


def volume_world_bounds(volume_shape, affine):
    """World-coordinate bounding box (per axis) of a volume, from its 8 corners."""
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
    return [(world[:, dim].min(), world[:, dim].max()) for dim in range(3)]


def streamline_plane_limits(streamlines, d1, d2, fallback_bounds):
    """Padded (5%) axis limits enclosing all streamline points in the (d1, d2) plane."""
    min_d1 = max_d1 = min_d2 = max_d2 = None
    for sl in streamlines:
        if sl.size == 0:
            continue
        sl_d1, sl_d2 = sl[:, d1], sl[:, d2]
        a, b, c, d = sl_d1.min(), sl_d1.max(), sl_d2.min(), sl_d2.max()
        min_d1 = a if min_d1 is None else min(min_d1, a)
        max_d1 = b if max_d1 is None else max(max_d1, b)
        min_d2 = c if min_d2 is None else min(min_d2, c)
        max_d2 = d if max_d2 is None else max(max_d2, d)

    if min_d1 is None or min_d2 is None:
        (min_d1, max_d1), (min_d2, max_d2) = fallback_bounds

    range_d1 = max_d1 - min_d1
    range_d2 = max_d2 - min_d2
    pad_d1 = range_d1 * 0.05 if range_d1 else 1.0
    pad_d2 = range_d2 * 0.05 if range_d2 else 1.0
    return (min_d1 - pad_d1, max_d1 + pad_d1), (min_d2 - pad_d2, max_d2 + pad_d2)
