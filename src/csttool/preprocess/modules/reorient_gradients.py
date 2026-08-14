"""
reorient_gradients.py

Rotate b-vectors to follow the per-volume transforms estimated by motion
correction (Leemans & Jones 2009).

Why this exists
---------------
``dipy.align.motion_correction`` resamples every DWI volume onto the reference
volume's grid but does not touch the gradient table. The image and its
b-vectors then describe different anatomies: the diffusion-encoding direction
recorded for volume *k* is the one that applied *before* the head moved, while
the voxels have been rotated back into the reference pose. Fitting a tensor to
that pair biases FA/MD and tilts the principal eigenvector — silently, because
nothing about the files looks wrong. The correction is to apply the inverse of
each volume's estimated rotation to its b-vector.

The convention chain (each link verified against installed source, DIPY 1.12.1)
------------------------------------------------------------------------------
1. **What the transforms are.** ``motion_correction`` is ``register_dwi_series``
   with the pipeline ``[center_of_mass, translation, rigid, affine]``
   (``dipy/align/_public.py:889``). It returns a ``(4, 4, n_volumes)`` array
   covering *all* volumes; the DWI entries sit at ``[..., ~b0s_mask]`` in the
   same order as ``gtab.bvecs[~gtab.b0s_mask]`` (``_public.py:869-887``).
2. **Which direction they point.** Each is an ``AffineMap`` affine operating in
   **world** coordinates and mapping static-world → moving-world
   (``dipy/align/imaffine.py:87-102``) — the pull transform used for
   resampling, which numerically equals the *forward head motion* of that
   volume. Confirmed empirically: planting a +10° world rotation about z into a
   synthetic volume and running ``motion_correction`` returns
   ``polar(reg_affine)[0] ≈ Rz(+10°)``, not its inverse.
3. **How a b-vector responds.** ``dipy.core.gradients.reorient_bvecs`` applies
   ``polar(aff)[0]⁻¹`` to the corresponding non-b0 b-vector
   (``dipy/core/gradients.py:797-805``), which is exactly the Leemans & Jones
   rule for a forward-motion affine. That direction convention is DIPY's, pinned
   by its own test (``dipy/core/tests/test_gradients.py:417-478``); this module
   does not re-derive it. Polar decomposition also discards the scale and shear
   picked up by the affine stage of the fit — the same approximation class as
   FSL eddy's rotation propagation.
4. **Which frame the b-vectors live in.** csttool follows the DIPY/FSL
   convention: b-vector components are given in the frame of the **voxel axes**
   (see ``gradient_validation.reorient_dwi_to_ras``). DIPY's affines are in
   world coordinates. A world rotation must therefore be *conjugated* into the
   b-vector frame before it can be applied::

       g' = normalise( V⁻¹ · R_w⁻¹ · V · g )

   This matters: for LAS-stored data (``V = diag(-1, 1, 1)``, the common
   dcm2niix output) ``V⁻¹ Rz(-θ) V = Rz(+θ)``, so applying the world rotation
   naively rotates the b-vector the wrong way and *doubles* the error instead
   of removing it. For RAS+ data ``V = I`` and the conjugation is a no-op.

Why ``V`` is the orthonormal orientation, not the full 3×3
----------------------------------------------------------
``V = polar(image_affine[:3, :3])[0]`` — the rotation part only, with voxel
scaling discarded.

A b-vector is a *physical unit direction* whose components happen to be
expressed in the frame of the voxel axes. Voxel size is a sampling property,
not a property of the direction: ``(1, 0, 0)`` means "along the +i voxel axis in
physical space" whether the voxel is 1 mm or 6 mm. Using the full 3×3 would
treat the b-vector as a displacement in index space, which is a different
object.

The decisive consequence: with an axis-aligned affine the ground truth is
unambiguous — the corrected b-vector is ``R_w⁻¹ g`` — and it cannot depend on
the zooms. ``A⁻¹ R_w⁻¹ A`` violates that for anisotropic voxels; the orthonormal
conversion satisfies it exactly. ``test_reorient_gradients.py`` asserts this
invariance directly. It is also what DIPY does one level down: ``reorient_bvecs``
strips scale from the motion affines by the same polar decomposition.

What this is not
----------------
Affine, between-volume motion correction only. No eddy-current model, no
outlier replacement, no slice-to-volume estimation, no susceptibility
distortion correction.

References
----------
.. [1] Leemans A., Jones D.K. "The B-matrix must be rotated when correcting for
   subject motion in DTI data." Magn Reson Med 61(6):1336-1349 (2009).
"""

import numpy as np
from scipy.linalg import polar


def _as_affine_stack(reg_affines) -> np.ndarray:
    """Normalise registration affines to a ``(4, 4, n)`` float array."""
    affines = np.asarray(reg_affines, dtype=float)
    if affines.ndim != 3:
        raise ValueError(
            f"reg_affines must be a stack of 4x4 matrices, got shape {affines.shape}"
        )
    # dipy.align returns (4, 4, n); a list of matrices stacks as (n, 4, 4).
    if affines.shape[0] == affines.shape[1] == 4:
        return affines
    if affines.shape[1] == affines.shape[2] == 4:
        return np.moveaxis(affines, 0, -1)
    raise ValueError(
        f"reg_affines must be (4, 4, n) or (n, 4, 4), got shape {affines.shape}"
    )


def rotation_part(matrix: np.ndarray) -> np.ndarray:
    """Orthonormal rotation component of a 3×3 or 4×4 linear/affine matrix.

    Uses polar decomposition, so scale and shear are discarded and the result is
    orthonormal for oblique and sign-flipped (negative-determinant) matrices
    alike. For the standard NIfTI form ``A = V · diag(zooms)`` this returns
    ``V`` exactly.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape == (4, 4):
        m = m[:3, :3]
    if m.shape != (3, 3):
        raise ValueError(f"expected a 3x3 or 4x4 matrix, got shape {m.shape}")
    rot, _scale = polar(m)
    return rot


def max_rotation_angle_deg(reg_affines) -> float:
    """Largest per-volume rotation magnitude in the stack, in degrees.

    QC summary for the provenance ledger and the console: how much head rotation
    the registration thinks it removed.
    """
    affines = _as_affine_stack(reg_affines)
    angles = []
    for i in range(affines.shape[-1]):
        rot = rotation_part(affines[..., i])
        # A rotation's trace fixes its angle: tr(R) = 1 + 2cos(theta).
        cos_theta = (np.trace(rot) - 1.0) / 2.0
        angles.append(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return float(np.max(angles)) if angles else 0.0


def rotate_bvecs_for_motion(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    reg_affines,
    b0s_mask: np.ndarray,
    image_affine: np.ndarray,
    *,
    b0_threshold: float = 50,
) -> np.ndarray:
    """Apply motion-correction rotations to b-vectors.

    Parameters
    ----------
    bvals : ndarray, shape (N,)
        b-values as carried by the gradient table used for motion correction.
    bvecs : ndarray, shape (N, 3)
        b-vectors in the image's **voxel** frame (the DIPY/FSL convention).
        b0 rows are all-zero.
    reg_affines : ndarray, shape (4, 4, N) or (N, 4, 4)
        Per-volume world-space transforms as returned by
        ``dipy.align.motion_correction`` — one for every volume, b0s included.
    b0s_mask : ndarray of bool, shape (N,)
        Which volumes are b0. Used both to select the DWI affines and to keep
        b0 rows zero.
    image_affine : ndarray, shape (4, 4)
        The affine of the data that was registered. Only its orientation is
        used (see module docstring).
    b0_threshold : float, optional
        The execution's b0 threshold, needed to rebuild an equivalent gradient
        table in the world frame. Must be the one ``b0s_mask`` was derived
        with; a mismatch raises.

    Returns
    -------
    rotated_bvecs : ndarray, shape (N, 3)
        b-vectors in the same voxel frame and volume order as the input. DWI
        rows are unit-norm; b0 rows are exactly zero.
    """
    from dipy.core.gradients import gradient_table_from_bvals_bvecs, reorient_bvecs

    bvals = np.asarray(bvals, dtype=float)
    bvecs = np.asarray(bvecs, dtype=float)
    b0s_mask = np.asarray(b0s_mask, dtype=bool)

    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        raise ValueError(f"bvecs must be (N, 3), got shape {bvecs.shape}")
    if bvals.shape[0] != bvecs.shape[0] or b0s_mask.shape[0] != bvecs.shape[0]:
        raise ValueError(
            "bvals, bvecs and b0s_mask must describe the same number of volumes"
        )

    affines = _as_affine_stack(reg_affines)
    if affines.shape[-1] != bvecs.shape[0]:
        raise ValueError(
            f"reg_affines covers {affines.shape[-1]} volumes but the gradient "
            f"table has {bvecs.shape[0]}"
        )

    dwi = ~b0s_mask
    if not dwi.any():
        # Nothing to rotate; hand back a copy so callers can always use the result.
        return bvecs.copy()

    # --- voxel frame -> world frame ---------------------------------------
    # Orthonormal, so norms are preserved exactly and the inverse is the
    # transpose (true for negative-determinant / sign-flipped orientations too).
    vox2world = rotation_part(image_affine)
    bvecs_world = bvecs @ vox2world.T

    gtab_world = gradient_table_from_bvals_bvecs(
        bvals, bvecs_world, b0_threshold=b0_threshold
    )
    if not np.array_equal(gtab_world.b0s_mask, b0s_mask):
        raise ValueError(
            "b0_threshold does not reproduce the supplied b0s_mask; the "
            "gradient table and the rotation would disagree about which "
            "volumes are b0"
        )

    # --- apply the inverse of each volume's rotation (DIPY's primitive) ----
    # Only the DWI affines, in the order reorient_bvecs expects.
    dwi_affines = affines[..., dwi]
    rotated_world = reorient_bvecs(gtab_world, dwi_affines).bvecs

    # --- world frame -> voxel frame ---------------------------------------
    rotated = rotated_world @ vox2world

    # Renormalise DWI rows to absorb floating-point drift; b0 rows stay zero.
    rotated[b0s_mask] = 0.0
    norms = np.linalg.norm(rotated[dwi], axis=1, keepdims=True)
    rotated[dwi] = rotated[dwi] / norms

    return rotated
