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
def orientation_code(affine) -> str:
    """Return the 3-letter voxel-orientation code of ``affine`` (e.g. ``"RAS"``).

    This is the compact, scientifically exact description of the image's voxel
    axis orientation (anatomical axis + direction of increasing index). It is
    computed from the affine, never hardcoded, so a subject reoriented to RAS
    during preprocessing is reported as RAS regardless of what a mockup may say.
    """
    import nibabel as nib
    return "".join(nib.orientations.aff2axcodes(affine))


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


# ---------------------------------------------------------------------------
# Slice extraction and common-canvas letterboxing.
#
# These were the private ``_qc_slice_2d`` / ``_apply_common_canvas`` helpers in
# ``metrics/modules/visualizations.py``. They are promoted to public API so the
# legacy triptych and the three new QC panels (DEC-FA, CST density, CST over FA)
# share the exact code that already produces the accepted A4 geometry. Behaviour
# is byte-identical to the originals; ``slice_2d`` only replaces its if/else
# chain with a validated lookup against :data:`VIEW_AXES`.
# ---------------------------------------------------------------------------
def slice_2d(volume, view, index):
    """Return the 2D display slice of ``volume`` for ``view`` at voxel ``index``.

    The returned array is already transposed for display with
    ``matplotlib.pyplot.imshow(..., origin='lower')``: its rows run along the
    view's vertical voxel axis and its columns along the horizontal voxel axis,
    matching :data:`VIEW_AXES`. This is the same convention the legacy triptych
    used, preserved so the existing report figures stay byte-identical.

    Parameters
    ----------
    volume : ndarray, shape (X, Y, Z)
    view : {"axial", "coronal", "sagittal"}
        Validated against :data:`VIEW_AXES`; any other value raises
        ``ValueError`` rather than silently falling through an ``else`` branch.
    index : int
        Voxel index along the view's depth (slice) axis.

    Returns
    -------
    ndarray
        2D slice with the depth axis removed and the two in-plane axes
        transposed for ``origin='lower'`` display.
    """
    if view not in VIEW_AXES:
        raise ValueError(
            f"unknown view {view!r}; expected one of {sorted(VIEW_AXES)}"
        )
    h_axis, v_axis = VIEW_AXES[view]
    depth_axis = 3 - h_axis - v_axis  # the axis not present in the plane
    return np.take(volume, index, axis=depth_axis).T


def pad_axes_to_canvas(ax, canvas_w, canvas_h):
    """Centre the axis's current view inside a common ``canvas_w x canvas_h`` box.

    Equivalent to padding a 2D slice to a common canvas before ``imshow``, but
    without touching the data or the voxel coordinates a streamline overlay is
    drawn in. Combined with ``aspect='equal'`` this gives every QC panel the
    same physical size and the same scale, letterboxed in black, with no
    anatomical distortion. Any x-axis inversion applied for the radiological
    convention is preserved.

    The axes background patch is not painted while the axis is off, so the
    letterbox is drawn explicitly as an axes-spanning black rectangle behind the
    image — without it the padding reads as white and panels of differing slice
    shapes look like different sizes even though their boxes are identical.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    cx, cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
    x_sign = 1.0 if x_hi >= x_lo else -1.0
    y_sign = 1.0 if y_hi >= y_lo else -1.0
    ax.set_xlim(cx - x_sign * canvas_w / 2.0, cx + x_sign * canvas_w / 2.0)
    ax.set_ylim(cy - y_sign * canvas_h / 2.0, cy + y_sign * canvas_h / 2.0)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes, facecolor='black',
        edgecolor='none', zorder=-10, clip_on=False,
    ))


# ---------------------------------------------------------------------------
# Data-driven QC slice selection and physical slab membership
# (visualization-refactoring-plan §6.4, §6.5).
#
# These replace the legacy magic-constant slice index and the voxel-measured
# ±5-voxel slab with a data-driven, deterministic choice whose provenance is
# disclosed on the figure, and a physical (millimetre) inclusion criterion
# whose rendered thickness is invariant under anisotropic voxels. Both are
# pure geometry (no Figure, no save) so they are unit-testable in isolation.
# ---------------------------------------------------------------------------

# Physical slab thickness in millimetres for the CST-over-FA streamline overlay.
# ±5 mm matches the visual density of the legacy ±5-voxel rule at 2 mm
# isotropic (the appearance the report was designed around) while being
# physically defined, so the same figure shows the same physical slab on
# 2 mm isotropic, 1.5 mm isotropic and 2×2×6 mm data (plan §6.5, R-2).
DEFAULT_SLAB_MM = 10.0


def select_qc_slice(density=None, density_left=None, density_right=None,
                   roi_masks=None, brain_mask=None, affine=None,
                   view="coronal"):
    """Deterministically choose the shared QC slice index for the report panels.

    All three report QC panels (DEC-FA, CST density, final CST over FA) share
    the single slice this returns, so the reader sees the same plane across the
    three complementary questions. The rule degrades in four documented levels,
    each one step away from the thing being QC'd and using only inputs the
    figure already requires (plan §6.4):

    1. ``bilateral_occupancy`` — the coronal plane with the most voxels the
       extracted bundle visits (count, not summed density — robust to one
       dense core voxel).
    2. ``surviving_hemisphere`` — when exactly one hemisphere's density is
       non-empty, that hemisphere's occupancy. A real extracted bundle shown
       when one side failed.
    3. ``roi_occupancy`` — the union of the warped extraction ROI masks' best
       plane, so a total extraction failure is diagnosable from its targets.
    4. ``anatomical_centroid`` — the plane whose world coordinate is nearest the
       brain-mask centre of mass. Always available; can never fail.

    Parameters
    ----------
    density, density_left, density_right : ndarray, optional
        Combined / per-hemisphere CST density volumes on the FA grid.
    roi_masks : sequence of ndarray, optional
        Warped extraction ROI masks to combine for the level-3 fallback.
    brain_mask : ndarray
        Brain mask for the always-available level-4 fallback.
    affine : (4,4) ndarray
        Voxel->RASMM affine (needed for the level-4 world-coordinate centroid).
    view : {"coronal", "axial", "sagittal"}
        View whose slice axis is selected.

    Returns
    -------
    index : int
        The chosen voxel slice index along the view's depth axis.
    provenance : dict
        ``rule`` (one of the four names above), ``occupancy`` (the count that
        won), ``n_candidates`` (planes with non-zero occupancy), ``tie_break``
        (a short description of how a tie, if any, was resolved).

    Notes
    -----
    Tie-breaking is deterministic, in order: highest occupancy → nearest to the
    density-weighted centre of mass along the view axis → lowest index. So the
    result is a pure function of the inputs (no RNG).
    """
    if view not in VIEW_AXES:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(VIEW_AXES)}")
    h_axis, v_axis = VIEW_AXES[view]
    depth_axis = 3 - h_axis - v_axis
    n_slices = None
    weights = None  # density-weighted centre of mass reference (for ties)

    def _occupancy_per_plane(vol):
        if vol is None:
            return None
        vol = np.asarray(vol)
        if vol.shape == () or vol.max() == 0:
            return None
        # Count of voxels with density > 0 in each plane (robust to a single
        # dense core voxel — a sum would not be).
        nonzero = vol > 0
        counts = nonzero.sum(axis=tuple(i for i in range(3) if i != depth_axis))
        return counts.astype(np.int64)

    def _depth_centroid(vol):
        if vol is None or np.asarray(vol).max() == 0:
            return None
        vol = np.asarray(vol)
        # density-weighted index centroid along the depth axis
        idx = np.arange(vol.shape[depth_axis])
        shape = [1, 1, 1]; shape[depth_axis] = vol.shape[depth_axis]
        idx = idx.reshape(shape)
        weights_ = (vol * idx).sum(axis=tuple(i for i in range(3) if i != depth_axis))
        total = vol.sum()
        if total == 0:
            return None
        return float(weights_.sum() / total)

    def _best(counts, centroid):
        # Deterministic tie-break: highest count -> nearest centroid -> lowest idx.
        if counts is None or counts.max() == 0:
            return None
        max_c = counts.max()
        candidates = np.where(counts == max_c)[0]
        if len(candidates) == 1:
            return int(candidates[0]), int(max_c), "single max-occupancy plane"
        if centroid is None:
            idx = int(candidates.min())
            return idx, int(max_c), f"tie of {len(candidates)} planes resolved to lowest index"
        nearest = int(min(candidates, key=lambda j: abs(j - centroid)))
        return nearest, int(max_c), f"tie of {len(candidates)} planes resolved to nearest density-weighted centroid"

    # Level 1: bilateral occupancy.
    counts = _occupancy_per_plane(density)
    if counts is not None and counts.max() > 0:
        centroid = _depth_centroid(density)
        idx, occ, tb = _best(counts, centroid)
        return idx, {"rule": "bilateral_occupancy", "occupancy": occ,
                     "n_candidates": int((counts > 0).sum()), "tie_break": tb}

    # Level 2: surviving hemisphere.
    cL = _occupancy_per_plane(density_left)
    cR = _occupancy_per_plane(density_right)
    if cL is not None and cR is None:
        centroid = _depth_centroid(density_left)
        idx, occ, tb = _best(cL, centroid)
        return idx, {"rule": "surviving_hemisphere", "occupancy": occ,
                     "n_candidates": int((cL > 0).sum()), "tie_break": tb + " (left)"}
    if cR is not None and cL is None:
        centroid = _depth_centroid(density_right)
        idx, occ, tb = _best(cR, centroid)
        return idx, {"rule": "surviving_hemisphere", "occupancy": occ,
                     "n_candidates": int((cR > 0).sum()), "tie_break": tb + " (right)"}

    # Level 3: combined ROI occupancy.
    if roi_masks:
        combined = None
        for m in roi_masks:
            m = np.asarray(m)
            if m.shape == ():
                continue
            combined = m.astype(bool) if combined is None else (combined | m.astype(bool))
        if combined is not None and combined.max() > 0:
            counts = _occupancy_per_plane(combined.astype(np.float32))
            if counts is not None and counts.max() > 0:
                idx, occ, tb = _best(counts, None)
                return idx, {"rule": "roi_occupancy", "occupancy": occ,
                             "n_candidates": int((counts > 0).sum()), "tie_break": tb}

    # Level 4: anatomical fallback (brain-mask centroid in world mm).
    if brain_mask is None:
        raise ValueError(
            "select_qc_slice exhausted all fallback levels: density, "
            "density_left/right, and roi_masks all empty and brain_mask is None. "
            "brain_mask is required so the anatomical fallback can always succeed."
        )
    bm = np.asarray(brain_mask)
    if bm.max() == 0:
        # Nothing at all in the brain mask: choose the middle slice so the figure
        # still renders (cannot fail). Disclose this plainly.
        idx = bm.shape[depth_axis] // 2
        return idx, {"rule": "anatomical_centroid", "occupancy": 0,
                     "n_candidates": 0,
                     "tie_break": "empty brain mask; middle slice chosen"}
    if affine is None:
        raise ValueError("affine is required for the anatomical_centroid fallback")
    affine = np.asarray(affine, dtype=float)
    # Centre of mass in voxel coords, then world.
    coords = np.argwhere(bm > 0)
    com_vox = coords.mean(axis=0)
    com_world = (affine[:3, :3] @ com_vox) + affine[:3, 3]
    # Project every plane centre (world coord of voxel-plane centre) to the
    # view's depth world axis and pick the nearest.
    depth_world_axis = depth_axis
    plane_centres = []
    for j in range(bm.shape[depth_axis]):
        vox = [0.0, 0.0, 0.0]; vox[depth_axis] = float(j)
        world = (affine[:3, :3] @ np.asarray(vox)) + affine[:3, 3]
        plane_centres.append(world[depth_world_axis])
    plane_centres = np.asarray(plane_centres)
    target = com_world[depth_world_axis]
    idx = int(np.argmin(np.abs(plane_centres - target)))
    return idx, {"rule": "anatomical_centroid", "occupancy": 0,
                 "n_candidates": int((bm > 0).sum()),
                 "tie_break": f"plane nearest brain-mask centroid (world {depth_world_axis}-axis)"}


def slab_membership(points_world, affine, view, slice_index, thickness_mm):
    """Boolean membership of world-space points in a physical slab.

    The slab is centred on voxel plane ``slice_index`` and has half-thickness
    ``thickness_mm / 2`` along the view's depth axis **in world millimetres**.
    Using the unit world normal ``n̂`` of the voxel axis (not the array axis with
    its voxel scaling) makes the rendered physical thickness identical for
    isotropic and anisotropic voxels (plan §6.5, R-2): the legacy ±5-voxel rule
    would show a 10/7.5/30 mm slab on 2 mm iso / 1.5 mm iso / 2×2×6 mm data; this
    shows 10 mm on all three.

    Parameters
    ----------
    points_world : ndarray, shape (N, 3)
        Streamline points in RASMM world coordinates.
    affine : (4,4) ndarray
        Voxel->RASMM affine.
    view : {"coronal", "axial", "sagittal"}
    slice_index : int
        Centre voxel plane of the slab.
    thickness_mm : float
        Full slab thickness in millimetres.

    Returns
    -------
    ndarray, shape (N,), bool
        ``True`` where the point lies within the slab.
    """
    if view not in VIEW_AXES:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(VIEW_AXES)}")
    affine = np.asarray(affine, dtype=float)
    h_axis, v_axis = VIEW_AXES[view]
    depth_axis = 3 - h_axis - v_axis
    # Unit world normal of the voxel depth axis (scale discarded).
    col = affine[:3, depth_axis]
    norm = np.linalg.norm(col)
    if norm == 0:
        raise ValueError(f"affine depth-axis column is zero for view {view!r}")
    n_hat = col / norm
    # World coordinate of the centre of voxel plane slice_index.
    vox = np.zeros(3); vox[depth_axis] = float(slice_index)
    centre = (affine[:3, :3] @ vox) + affine[:3, 3]
    pts = np.asarray(points_world, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    proj = (pts - centre) @ n_hat
    return np.abs(proj) <= (thickness_mm / 2.0)
