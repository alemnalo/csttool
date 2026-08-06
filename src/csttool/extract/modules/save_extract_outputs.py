"""
save_extract_outputs.py - persistence of CST extraction scientific data products.

Owns the persistence of the numerical (NIfTI) data products that the extraction
stage produces, separate from rendering. Currently this is the CST density
volume (visualization-refactoring-plan §7.2), written **unconditionally**
whenever the extraction stage runs (§2.5).

This module imports no Matplotlib: it computes science and writes NIfTI +
sidecar JSON. The PNG that visualizes the density is a separate layer gated by
``--save-visualizations`` (§6.7, ``csttool.viz.render`` + §8.2).
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from csttool.extract.modules.density import compute_cst_density, save_cst_density


def write_cst_density_product(cst_result, reference_img, out_dir, subject_id,
                              *, step_size_mm=None, sources=None,
                              command_line=None, software_versions=None,
                              verbose=False):
    """Compute and persist the bilateral CST density volume + sidecar.

    Unconditional: the density NIfTI is written whenever the extraction stage
    runs, exactly like FA/MD/RD/AD in the track stage (§2.5). The PNG is a
    separate, gated concern handled by the visualization layer.

    Parameters
    ----------
    cst_result : dict
        Extraction output with ``cst_left`` and ``cst_right`` streamlines (RASMM).
    reference_img : nibabel image
        The FA reference image supplying the density grid (affine + shape). The
        CST tractograms are saved in this image's RASMM space, so its grid is the
        correct target.
    out_dir : path
        Extraction output root; the volume is written to ``<out>/extraction/
        scalar_maps/`` ... actually ``<out>/scalar_maps/`` under the extraction
        stage dir.
    subject_id : str
        Stem prefix for the filename.
    step_size_mm : float, optional
        Tracking step size for the gap guard (§6.3). ``None`` skips the guard.
    sources, command_line, software_versions : optional
        Forwarded to the BIDS sidecar writer.
    verbose : bool

    Returns
    -------
    pathlib.Path or None
        Path to the written density NIfTI, or ``None`` if the stage had no
        bilateral bundle at all (in which case an all-zero volume is still
        written so downstream slice selection has a grid, and ``None`` is
        returned only on a hard failure).
    """
    try:
        affine = np.asarray(reference_img.affine, dtype=float)
        shape = tuple(int(s) for s in reference_img.shape[:3])
        left = list(cst_result.get("cst_left", []))
        right = list(cst_result.get("cst_right", []))
        density, meta = compute_cst_density(
            left, right, affine, shape, step_size_mm=step_size_mm,
        )
        path = save_cst_density(
            density, meta, affine, Path(out_dir), subject_id,
            sources=sources or [], command_line=command_line,
            software_versions=software_versions,
        )
        if verbose:
            print(f"  ✓ CST density: {path} (n_total={meta['n_total']}, "
                  f"max={meta['max_fraction']:.4f})")
        return path
    except Exception as exc:
        # Density is a new additive product; a failure must not break the
        # extracted tractograms or the metrics stage that follows.
        print(f"  ⚠️ Could not save CST density product: {exc}")
        import warnings
        warnings.warn(f"CST density product not written: {exc}")
        return None
