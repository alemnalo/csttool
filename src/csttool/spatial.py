"""
spatial.py - affine / frame mathematics for csttool.

A dependency-free leaf module (``numpy``, ``scipy.linalg``, ``nibabel.affines``
only) that both the pipeline layers and the visualization layer may import
without introducing a cycle. It follows the precedent of
:mod:`csttool.defaults`: a leaf with no csttool-package imports, readable from
anywhere regardless of import order (see ``preprocess/modules/gradient_validation.py``
for the same circular-import rationale).

Scope
-----
This module owns exactly the affine-frame mathematics the visualization refactor
needs and that DIPY does not provide: the rotation of a voxel-frame vector field
(specifically the DTI principal eigenvector V1) into the anatomical world
(RAS+) frame. No new numerical algorithm is invented here. The polar
decomposition is :func:`scipy.linalg.polar`; the obliquity diagnostic is
:func:`nibabel.affines.obliquity`. This module implements only the thin,
tested wrappers that supply csttool's error messages, its diagnostics, and the
``det(R)`` assertion.

It deliberately does **not** import matplotlib and does **not** import any
pipeline package, so the ``track`` stage can rotate an eigenvector field into
the world frame to produce a stored scientific product without importing the
presentation layer (visualization-refactoring-plan §4.1).

Coordinate conventions
----------------------
``affine`` is the 4x4 voxel->RASMM mapping of a NIfTI. ``M = affine[:3, :3]``
is its linear part. DIPY's ``TensorModel.fit`` solves for the diffusion tensor
in the frame of the b-vectors, which csttool pins to the data's voxel axes (see
``preprocess/modules/gradient_validation.py``). The principal eigenvector
``tenfit.evecs[..., :, 0]`` is therefore a set of *direction cosines with
respect to the voxel axes* — unit-length physical directions, **not** index-space
displacements.

To express such a direction in anatomical world (RAS+) axes the affine must not
be applied directly (it carries voxel scaling, which would rescale components by
voxel size and fake anisotropy of direction on anisotropic data), and its columns
must not be merely normalised (that removes scale but leaves shear, so the result
is not orthonormal). Instead we use the orthonormal factor ``R`` of the polar
decomposition ``M = R @ S`` (``R`` orthogonal, ``S`` symmetric positive
semidefinite), which is the closest orthogonal matrix to ``M`` in the Frobenius
norm: the pure rotation/reflection content of the affine with all scale and
shear removed.

Reflections (``det(R) = -1`` for LAS/LPS-stored data) are **not** corrected:
DEC takes the componentwise absolute value, so an axis reflection cannot change
any colour, and flipping a sign to force ``det(R) = +1`` would silently mutate
the stored V1 field for no visual gain. ``det(R)`` is recorded in diagnostics so
a consumer can tell whether the source was reflected.
"""

import numpy as np
import scipy.linalg
import nibabel.affines


__all__ = [
    "affine_shear_magnitude",
    "affine_obliquity_rad",
    "affine_rotation",
    "rotate_vector_field_to_world",
]


def affine_obliquity_rad(affine):
    """Per-axis obliquity (radians) of ``affine`` between each affine axis and
    the nearest cardinal axis.

    Thin re-export of :func:`nibabel.affines.obliquity` so callers do not have
    to import nibabel directly for a quantity this module's diagnostics already
    need. Returns ``[0, 0, 0]`` for any axis-aligned affine regardless of axis
    sign or voxel size.
    """
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got shape {affine.shape}")
    return [float(x) for x in nibabel.affines.obliquity(affine)]


def affine_shear_magnitude(affine, R=None, S=None):
    """Shear magnitude of the affine's symmetric polar factor ``S``.

    Defined as ``max |S_ij| / max |S_ii|`` for ``i != j`` (§5.1.6). This is the
    fraction by which the voxel axes fail to be mutually orthogonal in world
    space, which violates the assumption underlying the bvec convention itself.
    Returns 0.0 for an exactly diagonal ``S`` (orthogonal voxel axes).

    ``R`` and ``S`` may be supplied to avoid recomputing the polar
    decomposition; otherwise it is computed here.
    """
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got shape {affine.shape}")
    if R is None or S is None:
        R, S = scipy.linalg.polar(affine[:3, :3])
    diag = np.abs(np.diag(S))
    max_diag = float(diag.max()) if diag.size else 0.0
    if max_diag == 0.0:
        return 0.0
    off = S - np.diag(np.diag(S))
    max_off = float(np.abs(off).max()) if off.size else 0.0
    return max_off / max_diag


def affine_rotation(affine, *, shear_tol=1e-2):
    """Orthonormal factor ``R`` of the polar decomposition of ``affine[:3,:3]``.

    ``R`` maps voxel-frame unit directions to world RAS+ unit directions with
    all scale and shear removed; it is the closest orthogonal matrix to ``M`` in
    the Frobenius norm. Reflections are returned as-is (``det(R)`` may be -1);
    they are never "corrected" to ``det = +1`` (§5.1.4).

    Parameters
    ----------
    affine : (4, 4) ndarray
        Voxel->RASMM affine.
    shear_tol : float, default 1e-2
        Threshold above which a ``ShearWarning`` flag is set in the returned
        diagnostics (§5.1.6). A ``UserWarning`` is emitted for any non-negligible
        shear (``> 1e-6``); the function never raises on shear.

    Returns
    -------
    R : (3, 3) ndarray (float64)
        Orthonormal rotation/reflection factor.
    diag : dict
        Diagnostics with keys:
        - ``det``: ``float(np.linalg.det(R))`` (may be negative).
        - ``obliquity_rad``: list of 3 per-axis obliquity values (radians).
        - ``shear_magnitude``: float, the value defined in §5.1.6.
        - ``shear_warning``: bool, ``True`` iff ``shear_magnitude > shear_tol``.

    Raises
    ------
    ValueError
        If ``affine`` is not 4x4, if ``M`` is singular, or if
        ``abs(det(R)) - 1 > 1e-6`` (a degenerate or non-affine linear part).
    """
    import warnings

    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got shape {affine.shape}")
    M = affine[:3, :3]
    # polar raises a LinAlgError (a ValueError subclass) on singular input; the
    # csttool message names the failure rather than letting the bare scipy text
    # through.
    try:
        R, S = scipy.linalg.polar(M)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError(
            f"affine linear part is singular; polar decomposition failed: {exc}"
        ) from exc

    det = float(np.linalg.det(R))
    if abs(abs(det) - 1.0) > 1e-6:
        raise ValueError(
            f"affine rotation factor is degenerate: |det(R)| = {abs(det)!r} "
            f"(expected 1.0); the affine linear part may be singular or rank-deficient"
        )

    obliquity = affine_obliquity_rad(affine)
    shear = affine_shear_magnitude(affine, R=R, S=S)
    shear_warning = shear > shear_tol
    if shear > 1e-6:
        warnings.warn(
            f"affine has non-negligible shear (shear_magnitude={shear:.3g}); "
            f"world-frame vectors are an approximation (rotation-only encoding)",
            UserWarning,
            stacklevel=2,
        )

    diag = {
        "det": det,
        "obliquity_rad": obliquity,
        "shear_magnitude": float(shear),
        "shear_warning": bool(shear_warning),
    }
    return R, diag


def rotate_vector_field_to_world(vectors, affine, *, normalize=True):
    """Rotate a voxel-frame vector field into the anatomical world (RAS+) frame.

    Applies ``R`` (the orthonormal polar factor of ``affine[:3,:3]``, §5.1.3) to
    each vector. Because ``R`` is orthonormal it preserves norms exactly, so a
    field of unit direction cosines remains a field of unit direction cosines
    (up to floating-point). Zero vectors (e.g. unfitted voxels where DIPY leaves
    ``evecs`` at exactly zero) map to zero and are left as zero rather than
    normalised to NaN.

    Parameters
    ----------
    vectors : ndarray, shape (..., 3)
        Voxel-frame vectors. The last axis must be length 3.
    affine : (4, 4) ndarray
        Voxel->RASMM affine.
    normalize : bool, default True
        If True, renormalise non-zero output vectors to unit length. ``R`` is
        orthonormal so this only cleans up floating-point drift; it is on by
        default so the stored product is exactly unit. Set False to skip the
        normalisation (useful for testing norm preservation directly).

    Returns
    -------
    world : ndarray, same shape as ``vectors``, float32
        Vectors in the world RAS+ frame.
    diag : dict
        The diagnostics from :func:`affine_rotation`.
    """
    vectors = np.asarray(vectors)
    if vectors.shape[-1] != 3:
        raise ValueError(
            f"vectors last axis must be 3, got shape {vectors.shape}"
        )
    R, diag = affine_rotation(affine)
    flat = vectors.reshape(-1, 3).astype(np.float64)
    out = flat @ R.T  # (R @ v) for each row v -> v @ R.T
    if normalize:
        norms = np.linalg.norm(out, axis=1)
        # Leave zero vectors as zero; guard against division by zero.
        safe = norms > 0
        out[safe] = out[safe] / norms[safe, None]
    return out.reshape(vectors.shape).astype(np.float32), diag
