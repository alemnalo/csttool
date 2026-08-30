"""
brain_mask.py

Resolve the brain mask used by the tensor/tracking pipeline.

By default csttool estimates its own mask with median Otsu
(``background_segmentation``). Some datasets ship a DWI-space brain mask that
is more faithful to the tissue actually present — the automatic mask can
truncate inferior brainstem, and every voxel it drops gets no tensor fit and
therefore an FA of exactly zero. ``--brain-mask`` lets such a mask replace the
automatic one; nothing else about the pipeline changes.

The external mask is used verbatim: it is validated against the DWI grid and
binarised, never resampled. A mismatch is an error, not something to paper
over.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Absolute tolerance in millimetres for the affine comparison. Tight enough
# that any real grid difference (orientation, offset, voxel size) is caught,
# loose enough to absorb float32 header round-tripping between tools.
AFFINE_ATOL = 1e-3
AFFINE_RTOL = 1e-5


def load_external_brain_mask(mask_path, dwi_shape, dwi_affine, verbose=False):
    """Load, validate, and binarise a user-supplied DWI-space brain mask.

    Args:
        mask_path (str | Path): Path to the NIfTI brain mask.
        dwi_shape (tuple): Shape of the DWI array the mask must match. Only the
            first three axes are compared.
        dwi_affine (ndarray): 4x4 affine of the DWI the mask must match.
        verbose (bool): Print mask statistics.

    Returns:
        ndarray: 3D boolean mask, True inside the brain.

    Raises:
        FileNotFoundError: The mask file does not exist.
        ValueError: The mask is not 3D, does not match the DWI grid, contains
            non-finite values, or is empty.
    """
    import nibabel as nib

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"External brain mask not found: {mask_path}")

    try:
        mask_img = nib.load(str(mask_path))
    except Exception as exc:
        raise ValueError(
            f"External brain mask could not be read as a NIfTI image "
            f"({mask_path}): {type(exc).__name__}: {exc}"
        ) from exc

    mask_data = np.asanyarray(mask_img.dataobj)

    if mask_data.ndim != 3:
        raise ValueError(
            f"External brain mask must be 3D, got {mask_data.ndim}D with shape "
            f"{mask_data.shape} ({mask_path}). Extract a single 3D volume first."
        )

    dwi_spatial = tuple(int(x) for x in dwi_shape[:3])
    mask_spatial = tuple(int(x) for x in mask_data.shape)
    if mask_spatial != dwi_spatial:
        raise ValueError(
            f"External brain mask shape {mask_spatial} does not match the DWI "
            f"spatial shape {dwi_spatial} ({mask_path}). csttool does not "
            f"resample brain masks — supply a mask already on the DWI grid."
        )

    dwi_affine = np.asarray(dwi_affine, dtype=float)
    mask_affine = np.asarray(mask_img.affine, dtype=float)
    if not np.allclose(mask_affine, dwi_affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL):
        max_diff = float(np.abs(mask_affine - dwi_affine).max())
        raise ValueError(
            f"External brain mask affine is not compatible with the DWI affine "
            f"(max element difference {max_diff:.6g}, tolerance atol="
            f"{AFFINE_ATOL:g}/rtol={AFFINE_RTOL:g}) ({mask_path}).\n"
            f"  mask affine:\n{mask_affine}\n"
            f"  DWI affine:\n{dwi_affine}\n"
            f"csttool does not resample brain masks — supply a mask already on "
            f"the DWI grid."
        )

    # Non-finite values make "nonzero" meaningless, so reject them outright
    # rather than letting NaN decide membership.
    numeric = np.asarray(mask_data, dtype=np.float64)
    if not np.isfinite(numeric).all():
        n_bad = int((~np.isfinite(numeric)).sum())
        raise ValueError(
            f"External brain mask contains {n_bad:,} non-finite (NaN/Inf) "
            f"values ({mask_path})."
        )

    # Any nonzero finite value counts as inside the mask: this accepts 0/1
    # masks, 0/255 masks, and probability maps thresholded upstream.
    brain_mask = numeric != 0

    n_inside = int(brain_mask.sum())
    if n_inside == 0:
        raise ValueError(
            f"External brain mask contains no nonzero voxels ({mask_path})."
        )

    if verbose:
        coverage = n_inside / brain_mask.size * 100
        print(f"    • External mask: {n_inside:,} voxels ({coverage:.1f}%)")

    return brain_mask


def apply_brain_mask(data, brain_mask):
    """Zero everything outside the brain mask, broadcasting over volumes.

    Args:
        data (ndarray): 3D or 4D DWI array.
        brain_mask (ndarray): 3D boolean mask.

    Returns:
        ndarray: Masked copy of ``data``.
    """
    data = np.asarray(data)
    if data.ndim == 4:
        return data * brain_mask[..., np.newaxis]
    return data * brain_mask


def resolve_brain_mask(data, gtab, affine, brain_mask_path=None, verbose=False):
    """Return the brain mask for this run, external if given, automatic if not.

    Args:
        data (ndarray): 4D DWI array (X, Y, Z, N).
        gtab (GradientTable): Gradient table (used only by the automatic path).
        affine (ndarray): 4x4 affine of ``data``, for validating an external mask.
        brain_mask_path (str | Path | None): Optional external brain mask. When
            None, ``background_segmentation`` is used exactly as before.
        verbose (bool): Print the mask source and statistics.

    Returns:
        tuple: (masked_data, brain_mask, mask_info) where ``mask_info`` is a
            dict with ``brain_mask_source`` ('automatic_background_segmentation'
            or 'external') and ``brain_mask_path`` (str or None).
    """
    if brain_mask_path is None:
        # Import inside the call so the automatic path stays patchable at its
        # source module, as the existing tracking tests expect.
        from csttool.preprocess.modules.background_segmentation import (
            background_segmentation,
        )

        if verbose:
            print("  → Brain mask source: automatic background segmentation")

        masked_data, brain_mask = background_segmentation(data, gtab)
        mask_info = {
            'brain_mask_source': 'automatic_background_segmentation',
            'brain_mask_path': None,
        }
    else:
        if verbose:
            print(f"  → Brain mask source: external brain mask: {brain_mask_path}")

        brain_mask = load_external_brain_mask(
            brain_mask_path,
            dwi_shape=np.asarray(data).shape,
            dwi_affine=affine,
            verbose=verbose,
        )
        masked_data = apply_brain_mask(data, brain_mask)
        mask_info = {
            'brain_mask_source': 'external',
            'brain_mask_path': str(brain_mask_path),
        }

    if verbose:
        coverage = brain_mask.sum() / brain_mask.size * 100
        print(f"    • Brain mask: {brain_mask.sum():,} voxels ({coverage:.1f}%)")

    return masked_data, brain_mask, mask_info
