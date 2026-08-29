"""
density.py - CST streamline density data product.

Owns the single numerical data product of the extraction stage that the
visualization refactor introduces: the fraction of distinct retained bilateral
CST streamlines that visit each voxel at least once
(visualization-refactoring-plan §2.4).

Numerical primitive
-------------------
The per-voxel visit count comes directly from DIPY's public
:func:`dipy.tracking.utils.density_map`, whose unique-visit semantics match the
agreed definition (one count per distinct streamline per voxel, repeated
points / re-entries counted once). csttool implements **no** new counting
algorithm. This module supplies only:

* the denominator (the total number of retained bilateral streamlines),
* the per-hemisphere sum (left and right are disjoint bundles, so summing the
  two ``density_map`` outputs is exact),
* a step-size guard that densifies the streamlines with
  :func:`dipy.tracking.utils.subsegment` only when a tracking step could skip a
  voxel (so the "sampled point vs traversal" caveat in §5.2 is asserted rather
  than assumed),
* the grid guarantee (the output sits on the supplied FA-grid shape), and
* a csttool error message when a tractogram's coordinates fall outside the
  target grid (the bare DIPY ``IndexError`` does not identify which grid is
  wrong).

This module imports no Matplotlib and no pipeline command layer; it produces a
NIfTI volume plus a metadata dict, nothing more. Rendering is a separate layer
(§6.7, ``csttool.viz.render``).
"""

import numpy as np
import nibabel as nib
from dipy.tracking.utils import density_map, subsegment


DENSITY_DEFINITION = (
    "The fraction of distinct retained bilateral CST streamlines that visit "
    "each voxel at least once."
)


def compute_cst_density(streamlines_left, streamlines_right, affine, shape,
                        *, step_size_mm=None):
    """Fraction of retained bilateral CST streamlines visiting each voxel.

    Parameters
    ----------
    streamlines_left, streamlines_right : sequence of (N, 3) ndarray
        Left and right CST streamlines in **RASMM** world coordinates. The two
        bundles are disjoint by construction (left/right extraction), so summing
        their per-voxel counts is exact; a midline voxel visited from both sides
        correctly accumulates both.
    affine : (4, 4) ndarray
        Voxel->RASMM affine of the target grid (the native FA grid the streamlines
        were tracked in).
    shape : tuple of int
        Target grid shape (X, Y, Z).
    step_size_mm : float, optional
        Tracking step size in millimetres. When given and larger than half the
        smallest voxel extent, the streamlines are first densified with
        :func:`dipy.tracking.utils.subsegment` so a long step cannot skip a voxel
        (the documented DIPY caveat: ``density_map`` counts *sampled points*,
        not *traversal*). ``meta["densified"]`` reports whether this fired.

    Returns
    -------
    density : ndarray, shape ``shape``, float32
        Values in ``[0, 1]``: the per-voxel fraction defined in §2.4.
    meta : dict
        ``n_left``, ``n_right``, ``n_total``, ``max_fraction``, ``occupied_voxels``,
        ``densified`` (bool), ``definition`` (the literal §2.4 sentence).

    Notes
    -----
    Never divides by zero: an empty bundle returns an all-zero volume with
    ``n_total == 0``. ``IndexError`` from ``density_map`` (out-of-grid
    streamlines) is re-raised as a csttool ``ValueError`` naming both grids.
    """
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got {affine.shape}")
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3:
        raise ValueError(f"shape must be length-3, got {shape}")

    n_left = len(streamlines_left)
    n_right = len(streamlines_right)
    n_total = n_left + n_right

    densified = False
    if step_size_mm is not None and n_total > 0:
        voxel_sizes = nib.affines.voxel_sizes(affine)
        min_voxel = float(voxel_sizes.min())
        # A step can skip a voxel only if it exceeds the smallest voxel extent.
        # Densify at half that extent so every voxel along a segment receives a
        # sampled point (the conservative threshold from §6.3).
        if step_size_mm > min_voxel / 2.0:
            streamlines_left = list(subsegment(list(streamlines_left),
                                               max_segment_length=min_voxel / 2.0))
            streamlines_right = list(subsegment(list(streamlines_right),
                                                max_segment_length=min_voxel / 2.0))
            densified = True

    def _counts(streamlines):
        if len(streamlines) == 0:
            return np.zeros(shape, dtype=np.int64)
        try:
            return density_map(streamlines, affine, shape).astype(np.int64)
        except IndexError as exc:
            raise ValueError(
                f"streamlines fall outside the target CST density grid "
                f"(shape={shape}, affine axes={tuple(affine[:3, :3].diagonal())}): "
                f"{exc}. Check the tractogram and FA-grid are in the same space."
            ) from exc

    counts_left = _counts(streamlines_left)
    counts_right = _counts(streamlines_right)
    counts = counts_left + counts_right

    if n_total == 0:
        density = np.zeros(shape, dtype=np.float32)
    else:
        density = (counts / float(n_total)).astype(np.float32)

    occupied = int((counts > 0).sum())
    meta = {
        "n_left": int(n_left),
        "n_right": int(n_right),
        "n_total": int(n_total),
        "max_fraction": float(density.max()) if density.size else 0.0,
        "occupied_voxels": occupied,
        "densified": bool(densified),
        "definition": DENSITY_DEFINITION,
    }
    return density, meta


def save_cst_density(density, meta, affine, out_dir, stem, *,
                    sources=None, command_line=None, software_versions=None,
                    extra=None):
    """Persist the bilateral CST density volume + its self-describing sidecar.

    Writes ``{stem}_cst_density.nii.gz`` (float32, on the FA grid) and a JSON
    sidecar whose ``Description`` is the exact §2.4 sentence and whose
    ``Denominator`` equals the sum of the two hemisphere counts, so a reader can
    recover raw counts and compare two subjects' maps knowing what each was
    divided by. Unconditional whenever the extraction stage runs (§2.5).
    """
    import json
    from pathlib import Path
    from csttool.bids.output import write_derivative_sidecar

    out_dir = Path(out_dir)
    scalar_dir = out_dir / "scalar_maps"
    scalar_dir.mkdir(parents=True, exist_ok=True)

    nii_path = scalar_dir / f"{stem}_cst_density.nii.gz"
    img = nib.Nifti1Image(density.astype(np.float32), affine)
    nib.save(img, nii_path)

    sidecar_extra = {
        "Units": "dimensionless fraction",
        "Numerator": "dipy.tracking.utils.density_map (unique streamlines per voxel)",
        "Denominator": int(meta["n_total"]),
        "StreamlineCountLeft": int(meta["n_left"]),
        "StreamlineCountRight": int(meta["n_right"]),
        "VisitSemantics": ("voxel contains >= 1 sampled streamline point; "
                           "repeated visits by one streamline counted once"),
        "Densified": bool(meta["densified"]),
        # The volume's true maximum. It was computed and printed but never
        # persisted, so the one number that says how dense the densest voxel
        # actually got could not be recovered from the derivatives.
        #
        # This is NOT the report's colour-scale cap. The QC strip saturates its
        # scale at the 99th percentile of non-zero voxels and records that
        # separately as DensityDisplayVmax; on a real subject the true maximum
        # is roughly twice it. Do not use one where the other is meant.
        "MaxFraction": float(meta["max_fraction"]),
    }
    if extra:
        sidecar_extra.update(extra)
    write_derivative_sidecar(
        nii_path, sources=sources or [], description=DENSITY_DEFINITION,
        command_line=command_line, software_versions=software_versions,
        extra=sidecar_extra,
    )
    return nii_path
