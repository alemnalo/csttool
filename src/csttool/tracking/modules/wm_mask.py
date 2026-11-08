"""
wm_mask.py

Resolve the white-matter (WM) mask used for seeding and CSA peak estimation.

Policy chain, in precedence order (plan agreed 2026-09-22):

1. ``external``  — a user-supplied DWI-space WM mask (``--wm-mask``), used
   verbatim: validated against the DWI grid and binarised, never resampled,
   never morphologically edited. The user's mask is anatomically intended.
2. ``dam``        — DIPY's ``dipy.segment.tissue.dam_classifier`` (Cheng et
   al. 2020), which classifies WM/GM from the directional-average behaviour
   of the diffusion signal across shells. Needs no T1w — the whole point of
   choosing it. Requires >= 2 distinct non-b0 shells.
3. ``legacy``     — the historical ``dilate((FA > fa_thresh) & brain_mask)``
   from ``fit_tensors``, kept as the last line of defence and as the pinned
   behaviour behind ``--wm-method legacy`` so pre-existing results stay
   bit-reproducible.

``--wm-method auto`` (the default) picks external if a path was given, DAM
if the acquisition supports it, and falls back to legacy otherwise — always
recording *which* source was used and why, so the QC strip panel can name
what the reader is looking at. The fallback is a warning, never a failure:
an empty or inapplicable segmentation yields the legacy mask rather than an
aborted run (same philosophy as fit_tensors' AU31 empty-WM guard).

Only the legacy mask is dilated (one iteration, to reach the cortical GM/WM
interface — see fit_tensors). External and DAM masks are not edited.

The DAM classifier is deterministic (closed-form linear regression per voxel);
there is no stochastic step to plumb through ``reproducibility/``. It is,
however, a per-voxel Python loop inside DIPY, so on a full clinical volume it
takes minutes; that cost is disclosed here rather than optimised away, because
the point of using the DIPY implementation is that it is the reviewed one.
"""

from __future__ import annotations

import time

import numpy as np

from csttool.defaults import (
    DEFAULT_B0_THRESHOLD,
    DEFAULT_DAM_LOW_SIGNAL,
    DEFAULT_DAM_WM_THRESHOLD,
    DEFAULT_WM_METHOD,
)

from .brain_mask import load_external_mask

# Selectors accepted by --wm-method. ``auto`` follows the policy chain; the
# other two pin a source so results can be reproduced exactly.
WM_METHODS = ("auto", "dam", "legacy")


def load_external_wm_mask(mask_path, dwi_shape, dwi_affine, verbose=False):
    """Load a user-supplied DWI-space white-matter mask.

    Same contract as the external brain mask: 3D, on the DWI grid, never
    resampled, any nonzero finite voxel counts as WM.
    """
    return load_external_mask(mask_path, dwi_shape, dwi_affine,
                              mask_name="white matter mask", verbose=verbose)


def dam_shell_count(bvals, b0_threshold=DEFAULT_B0_THRESHOLD):
    """Number of distinct diffusion shells above the b0 threshold.

    ``dam_classifier`` fits a slope per voxel across shells, so it needs at
    least two of them; a single-shell DTI acquisition has nothing to fit.
    """
    bvals = np.asarray(bvals)
    shells = np.unique(bvals[bvals > b0_threshold])
    return int(shells.size)


def segment_wm_dam(data, bvals, wm_threshold=DEFAULT_DAM_WM_THRESHOLD,
                   b0_threshold=DEFAULT_B0_THRESHOLD,
                   low_signal_threshold=DEFAULT_DAM_LOW_SIGNAL,
                   verbose=False):
    """Segment white matter with DIPY's DAM (P-map) classifier.

    Runs on the *brain-masked* 4D data: voxels outside the brain mask are zero,
    their b0 mean is below ``low_signal_threshold`` and the classifier excludes
    them, so the returned mask lives inside the brain without an explicit
    intersection step.

    The input is sanitized first: eddy-corrected DWI commonly carries small
    negative signal values from interpolation at the brain edge, and DIPY's
    per-voxel fit takes ``log`` of the shell means — a negative or NaN value
    there raises ``Input y contains NaN`` and aborts the whole classification.
    NaNs are replaced by 0 and negatives clipped to 0; this only ever *adds*
    voxels to the low-signal exclusion (they classify as background), it never
    manufactures WM.

    Returns:
        tuple: (wm_mask, dam_params) where ``dam_params`` records the exact
        thresholds used, for the provenance block and the product sidecar.
    """
    from dipy.segment.tissue import dam_classifier

    data = np.asarray(data)
    bvals = np.asarray(bvals)

    if dam_shell_count(bvals, b0_threshold) < 2:
        raise ValueError(
            f"DAM tissue segmentation needs >= 2 distinct shells above "
            f"b0_threshold={b0_threshold:g}; this acquisition has "
            f"{dam_shell_count(bvals, b0_threshold)}. Use --wm-method legacy "
            f"or supply --wm-mask."
        )

    n_nan = int(np.isnan(data).sum())
    n_neg = int((data < 0).sum())
    if n_nan or n_neg:
        # int16-scale negatives: replace NaN, clip at zero. Sampled, not
        # loaded, in the sanity pass above — sum() over the full array is
        # one pass and the classifier loads it fully anyway.
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(data, 0.0, None, out=data)

    t0 = time.time()
    wm_mask, _gm_mask = dam_classifier(
        data, bvals, wm_threshold,
        b0_threshold=b0_threshold,
        low_signal_threshold=low_signal_threshold,
    )
    wm_mask = np.asarray(wm_mask, dtype=bool)
    elapsed = time.time() - t0

    if verbose:
        print(f"  → DAM tissue segmentation (P-map <= {wm_threshold:g})")
        if n_nan or n_neg:
            print(f"    • Input sanitized: {n_nan:,} NaN, {n_neg:,} negative "
                  f"values replaced by 0")
        print(f"    • White matter: {int(wm_mask.sum()):,} voxels "
              f"({elapsed:.1f}s)")

    dam_params = {
        "algorithm": "dipy.segment.tissue.dam_classifier (Cheng et al. 2020)",
        "wm_threshold": float(wm_threshold),
        "b0_threshold": float(b0_threshold),
        "low_signal_threshold": float(low_signal_threshold),
        "input_sanitized": bool(n_nan or n_neg),
    }
    return wm_mask, dam_params

def resolve_wm_mask(data, gtab, affine, brain_mask, legacy_wm,
                    wm_mask_path=None, method=DEFAULT_WM_METHOD,
                    dam_wm_threshold=DEFAULT_DAM_WM_THRESHOLD,
                    dam_low_signal=DEFAULT_DAM_LOW_SIGNAL,
                    b0_threshold=DEFAULT_B0_THRESHOLD,
                    verbose=False):
    """Return the white-matter mask for this run and a provenance dict.

    Precedence: external ``wm_mask_path`` > DAM (``method`` 'auto' when the
    acquisition supports it, or 'dam' when forced) > legacy FA-threshold mask.
    An external path wins regardless of ``method`` — mirroring
    ``resolve_brain_mask``, where supplying a path short-circuits the automatic
    path. An empty or inapplicable DAM result falls back to legacy with a
    warning rather than raising (AU31 philosophy: surface, do not abort).

    Args:
        data (ndarray): 4D *brain-masked* DWI array, for the DAM path.
        gtab (GradientTable): Gradient table (bvals drive shell counting).
        affine (ndarray): 4x4 affine of ``data``, for validating an external
            mask.
        brain_mask (ndarray): 3D boolean brain mask (for the coverage stat).
        legacy_wm (ndarray): The FA-threshold mask from ``fit_tensors``; used
            verbatim as the legacy source and as the fallback.
        wm_mask_path (str | Path | None): Optional external WM mask.
        method (str): 'auto', 'dam' or 'legacy'.
        dam_wm_threshold, dam_low_signal: DAM parameters, recorded in
            ``wm_info`` either way.
        b0_threshold (float): Shell partition threshold, shared with the rest
            of the pipeline.
        verbose (bool): Print the resolved source and statistics.

    Returns:
        tuple: (wm_mask, wm_info). ``wm_info`` carries ``wm_mask_source``
        ('external' | 'dam' | 'legacy'), ``wm_mask_path``, the requested
        ``wm_method``, the DAM parameters when DAM was used or attempted, the
        voxel count and the fraction of the brain it covers, and
        ``fallback_reason`` when the requested source could not be honoured.
    """
    if dam_wm_threshold is None:
        dam_wm_threshold = DEFAULT_DAM_WM_THRESHOLD
    if dam_low_signal is None:
        dam_low_signal = DEFAULT_DAM_LOW_SIGNAL
    if b0_threshold is None:
        b0_threshold = DEFAULT_B0_THRESHOLD
    if method not in WM_METHODS:
        raise ValueError(
            f"Unknown --wm-method {method!r}; expected one of {WM_METHODS}."
        )

    bvals = np.asarray(gtab.bvals)

    def _info(source, **extra):
        n = int(np.asarray(wm_mask).sum())
        brain = int(np.asarray(brain_mask).sum())
        info = {
            "wm_mask_source": source,
            "wm_mask_path": (str(wm_mask_path)
                             if source == "external" else None),
            "wm_method": method,
            "wm_voxels": n,
            "wm_fraction_of_brain": (n / brain if brain else 0.0),
            "dam_params": None,
            "fallback_reason": None,
        }
        info.update(extra)
        return info

    # 1. External wins, exactly as --brain-mask short-circuits Otsu.
    if wm_mask_path is not None:
        wm_mask = load_external_wm_mask(
            wm_mask_path,
            dwi_shape=np.asarray(data).shape,
            dwi_affine=affine,
            verbose=verbose,
        )
        return wm_mask, _info("external")

    # 2. DAM.
    if method in ("auto", "dam"):
        n_shells = dam_shell_count(bvals, b0_threshold)
        if n_shells < 2:
            if method == "dam":
                print(
                    f"  ⚠️ DAM segmentation forced but inapplicable "
                    f"({n_shells} shell(s) above b0_threshold={b0_threshold:g}); "
                    f"falling back to the legacy FA-threshold mask."
                )
            wm_mask = legacy_wm
            return wm_mask, _info(
                "legacy",
                dam_params={
                    "algorithm": "dipy.segment.tissue.dam_classifier",
                    "wm_threshold": float(dam_wm_threshold),
                    "low_signal_threshold": float(dam_low_signal),
                },
                fallback_reason=(
                    f"dam_inapplicable: {n_shells} shell(s) above "
                    f"b0_threshold={b0_threshold:g}"
                ),
            )
        try:
            wm_mask, dam_params = segment_wm_dam(
                data, bvals,
                wm_threshold=dam_wm_threshold,
                b0_threshold=b0_threshold,
                low_signal_threshold=dam_low_signal,
                verbose=verbose,
            )
        except Exception as exc:
            print(f"  ⚠️ DAM tissue segmentation failed: {exc}")
            wm_mask = legacy_wm
            return wm_mask, _info(
                "legacy",
                fallback_reason=f"dam_failed: {type(exc).__name__}: {exc}",
            )
        if int(np.asarray(wm_mask).sum()) == 0:
            print(
                "  ⚠️ DAM tissue segmentation produced an empty white-matter "
                "mask; falling back to the legacy FA-threshold mask. Check "
                "--dam-low-signal against this dataset's signal scale."
            )
            wm_mask = legacy_wm
            return wm_mask, _info(
                "legacy",
                dam_params={
                    "algorithm": "dipy.segment.tissue.dam_classifier",
                    "wm_threshold": float(dam_wm_threshold),
                    "low_signal_threshold": float(dam_low_signal),
                },
                fallback_reason="dam_empty_mask",
            )
        wm_info = _info("dam")
        wm_info["dam_params"] = dam_params
        if verbose:
            print(f"    • WM mask source: DAM ({dam_params['algorithm']})")
        return wm_mask, wm_info

    # 3. Legacy, pinned.
    wm_mask = legacy_wm
    return wm_mask, _info("legacy")