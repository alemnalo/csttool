"""
gradient_validation.py

Hard-fail validation of DWI gradient tables at load.

AU21 (three independent audits, GLM §3.9 / Qwen §3.10): ``gradient_table`` was
built at every load site with no checks that bvecs are unit-normalised, bvals
are non-negative, the b0 count is >= 1, or that bvecs live in the data's voxel
space. A malformed table would silently corrupt the tensor fit. This module
performs those four checks with clear, csttool-specific error messages and then
delegates to DIPY's :func:`gradient_table`, threading our single-source-of-truth
``DEFAULT_B0_THRESHOLD`` (which DIPY's own default of 50 only matched by
coincidence).

It is deliberately a leaf module (numpy + :mod:`csttool.defaults` only): it is
imported by the preprocess modules, which are in turn imported by
``csttool.preprocess.__init__``, so a constant living there could not be read
without a circular import — the same reason ``DEFAULT_B0_THRESHOLD`` lives in
:mod:`csttool.defaults`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from dipy.core.gradients import gradient_table

from csttool.defaults import DEFAULT_B0_THRESHOLD


class GradientTableValidationError(ValueError):
    """Raised when a DWI gradient table fails csttool's load-time checks.

    A subclass of :class:`ValueError` so callers that already catch
    ``ValueError`` for bad input keep working, while tests can assert on the
    specific type and message.
    """


def validate_bvals_bvecs(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    *,
    b0_threshold: float = DEFAULT_B0_THRESHOLD,
    atol: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate raw bvals/bvecs and return them in canonical (N,) / (N, 3) shape.

    Performs the four AU21 checks, raising :class:`GradientTableValidationError`
    with an actionable message (naming offending indices) on the first failure:

    1. bvals/bvecs shapes are compatible and counts match (bvecs may arrive as
       (3, N); we transpose to (N, 3), matching DIPY's convention).
    2. bvals are finite and non-negative (a negative bval would be silently
       treated as a b0 by DIPY, hiding a corrupted file).
    3. at least one b0 volume exists at the given ``b0_threshold`` (no b0 ⇒
       no reference signal ⇒ tensor fit is undefined).
    4. every DWI b-vector is unit-norm within ``atol``. b0 b-vectors may be zero
       (DIPY zeroes them), so the norm check applies only to DWI volumes.

    Parameters
    ----------
    bvals, bvecs : array_like
        Raw b-values and b-vectors as read from disk. ``bvals`` shape (N,) or
        (1, N); ``bvecs`` shape (N, 3) or (3, N).
    b0_threshold : float
        B-value at or below which a volume is considered b=0. Defaults to the
        single source of truth in :mod:`csttool.defaults`.
    atol : float
        Tolerance on b-vector unit norm, passed through to DIPY.

    Returns
    -------
    (bvals, bvecs) : tuple of ndarray
        Canonicalised arrays: ``bvals`` shape (N,), ``bvecs`` shape (N, 3).

    Raises
    ------
    GradientTableValidationError
        If any check fails.
    """
    bvals = np.asarray(bvals, dtype=float)
    bvecs = np.asarray(bvecs, dtype=float)

    # --- (0) Shape canonicalisation ----------------------------------------
    if bvals.ndim == 2 and bvals.shape[0] == 1:
        bvals = bvals[0]
    if bvals.ndim != 1:
        raise GradientTableValidationError(
            f"bvals must be 1-D (N,), got shape {bvals.shape}"
        )

    if bvecs.ndim != 2:
        raise GradientTableValidationError(
            f"bvecs must be 2-D (N, 3) or (3, N), got shape {bvecs.shape}"
        )
    # Transpose (3, N) -> (N, 3), but only when unambiguous: a (3, 3) array is
    # kept as-is (DIPY treats 3 directions x 3 components), matching DIPY's own
    # gradient_table logic.
    if bvecs.shape[1] != 3 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    if bvecs.shape[1] != 3:
        raise GradientTableValidationError(
            f"bvecs must have 3 columns, got shape {bvecs.shape}"
        )

    # --- (1) Count match ----------------------------------------------------
    if bvals.shape[0] != bvecs.shape[0]:
        raise GradientTableValidationError(
            f"bvals ({bvals.shape[0]}) and bvecs ({bvecs.shape[0]}) count mismatch"
        )

    # --- (2) bvals finite & non-negative -----------------------------------
    if not np.all(np.isfinite(bvals)):
        bad = np.where(~np.isfinite(bvals))[0]
        raise GradientTableValidationError(
            f"bvals contain non-finite values at indices {bad.tolist()}"
        )
    if np.any(bvals < 0):
        bad = np.where(bvals < 0)[0]
        raise GradientTableValidationError(
            f"bvals contain negative values at indices {bad.tolist()}: "
            f"{bvals[bad].tolist()}. Negative b-values are not physical."
        )

    # --- (3) At least one b0 -----------------------------------------------
    b0_mask = bvals <= b0_threshold
    if not b0_mask.any():
        raise GradientTableValidationError(
            f"No b=0 volumes found (b0_threshold={b0_threshold}). "
            "At least one b0 is required for a tensor fit."
        )

    # --- (4) DWI b-vectors unit-norm ---------------------------------------
    dwi_mask = bvals > b0_threshold
    if dwi_mask.any():
        dwi_bvecs = bvecs[dwi_mask]
        # NaNs in bvecs (e.g. dcm2niix writes 0 for b0 but some tools write NaN)
        # are a genuine corruption for DWI volumes, not a quiet zero.
        if not np.all(np.isfinite(dwi_bvecs)):
            raise GradientTableValidationError(
                "DWI b-vectors contain non-finite values."
            )
        norms = np.linalg.norm(dwi_bvecs, axis=1)
        non_unit = np.where(np.abs(norms - 1.0) > atol)[0]
        if non_unit.size:
            # Map back to original indices for an actionable message.
            dwi_indices = np.where(dwi_mask)[0]
            offending = dwi_indices[non_unit]
            raise GradientTableValidationError(
                f"DWI b-vectors must be unit-norm (atol={atol}); "
                f"{non_unit.size} of {dwi_mask.sum()} deviate. "
                f"First offending indices (global): {offending[:5].tolist()}, "
                f"norms {norms[non_unit][:5].tolist()}"
            )

    return bvals, bvecs


def validate_gradient_table(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    *,
    b0_threshold: float = DEFAULT_B0_THRESHOLD,
    atol: float = 1e-2,
):
    """Validate bvals/bvecs and build a DIPY :class:`GradientTable`.

    Thin wrapper over :func:`validate_bvals_bvecs` followed by
    :func:`dipy.core.gradients.gradient_table`, so the two load sites
    (``load_dataset`` and ``get_gtab_for_preproc``) call a single function and
    the gtab's ``b0s_mask`` is built with the same ``b0_threshold`` the rest of
    the pipeline uses, rather than DIPY's coincidental default of 50.
    """
    bvals, bvecs = validate_bvals_bvecs(
        bvals, bvecs, b0_threshold=b0_threshold, atol=atol
    )
    return gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=atol)


def reorient_dwi_to_ras(img, bvecs):
    """Reorient a DWI image and its b-vectors together to RAS+ voxel space.

    dicom2nifti's ``reorient_nifti=True`` reorients the *image* array to LAS but
    leaves the *bvecs* in the original scanner voxel space, so the two become
    mutually inconsistent — a classic DWI pitfall (AU21) that silently flips/
    permutes gradient directions and corrupts the tensor fit. The validator
    cannot detect this, because a scanner-space bvec that happens to be unit-norm
    passes the checks; the mismatch is purely about which voxel frame the bvecs
    live in, which is not observable from the files alone.

    The fix is to convert with ``reorient_nifti=False`` (so the image and bvecs
    start in the *same* scanner voxel space) and then reorient *both* together to
    RAS here. RAS+ is the convention the rest of csttool uses: dcm2niix (the
    primary, validated converter) emits RAS, and ``register_mni_to_subject``
    reorients the subject to RAS itself — so emitting RAS makes the dicom2nifti
    fallback match the primary path, and makes the registration's own
    reorientation a no-op (removing, not adding, an orientation conversion).

    Parameters
    ----------
    img : nibabel.Nifti1Image
        DWI image in its native (scanner) voxel space, as produced by
        dicom2nifti with ``reorient_nifti=False``.
    bvecs : ndarray of shape (N, 3) or (3, N)
        Gradient directions in the image's current voxel space (the DIPY
        convention). b0 rows may be all-zero.

    Returns
    -------
    reoriented_img : nibabel.Nifti1Image
        The image reoriented to RAS+ voxel axes (data permuted/flipped, affine
        updated).
    reoriented_bvecs : ndarray of shape (N, 3)
        bvecs re-expressed in the RAS voxel frame. DWI rows are renormalised to
        absorb floating-point noise from the permutation; b0 rows stay zero.

    Notes
    -----
    A gradient direction expressed in old-voxel-frame components ``g_old``
    becomes ``g_new = R @ g_old`` in new-voxel-frame components, where
    ``R = inv(A_new[:3,:3]) @ A_old[:3,:3]`` (the rotation part of the
    voxel-frame change). For a pure axis permutation/flip this R is orthogonal
    with entries in {0, ±1}, so norms are preserved exactly up to FP.
    """
    import nibabel as nib
    from nibabel.orientations import (
        axcodes2ornt,
        ornt_transform,
        inv_ornt_aff,
        apply_orientation,
    )

    bvecs = np.asarray(bvecs, dtype=float)
    if bvecs.ndim != 2:
        raise GradientTableValidationError(
            f"bvecs must be 2-D (N, 3) or (3, N), got shape {bvecs.shape}"
        )
    if bvecs.shape[1] != 3 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    if bvecs.shape[1] != 3:
        raise GradientTableValidationError(
            f"bvecs must have 3 columns, got shape {bvecs.shape}"
        )

    old_affine = img.affine
    current_ornt = nib.io_orientation(old_affine)
    target_ornt = axcodes2ornt(("R", "A", "S"))
    # If already RAS, reorientation is a no-op; still return fresh arrays so
    # callers can always use the return value.
    transform = ornt_transform(current_ornt, target_ornt)

    reoriented_data = apply_orientation(img.get_fdata(), transform)
    reoriented_affine = old_affine @ inv_ornt_aff(transform, img.shape)
    reoriented_img = nib.Nifti1Image(
        reoriented_data, reoriented_affine, img.header
    )

    # Voxel-frame rotation carrying old-voxel directions into new-voxel space.
    R = np.linalg.inv(reoriented_affine[:3, :3]) @ old_affine[:3, :3]
    reoriented_bvecs = bvecs @ R.T

    # Renormalise DWI rows; leave b0 (zero) rows as zero.
    norms = np.linalg.norm(reoriented_bvecs, axis=1, keepdims=True)
    nonzero = norms.ravel() != 0
    reoriented_bvecs[nonzero] = reoriented_bvecs[nonzero] / norms[nonzero]

    return reoriented_img, reoriented_bvecs
